"""
LLM narration layer: renders the Intermediate Representation (IR) JSONs into
natural-language explanations with a LOCAL open-weights model, and scores every
narrative with the atom-matching faithfulness verifier.

The client speaks the OpenAI-compatible chat API (default: Ollama at
http://localhost:11434/v1, default model qwen2.5:7b-instruct) at temperature 0
with a fixed seed. Any local server exposing the same API (LM Studio,
llama.cpp server, vLLM) works via `base_url`. The pipeline never depends on
this layer: narratives are generated on demand from the IR files an
`--explain` run produced (see Explainability/narrate.py).

The anti-hallucination contract lives in SYSTEM_PROMPT: the model may only
restate the numbered fact sentences, must copy numbers and names verbatim,
must convey every [REQUIRED] fact, and may use [CAVEAT] lines only as
limitations. The verifier then measures how well the output honoured that
contract (hallucination + omission rates).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Callable, Dict, List, Optional

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

_SETUP_HINT = (
    "No LLM server reachable at {url}. Start one first, e.g.:\n"
    "    ollama serve                     (installs: https://ollama.com)\n"
    "    ollama pull {model}\n"
    "or point --base-url at any OpenAI-compatible local server "
    "(LM Studio, llama.cpp, vLLM)."
)


def _verifier_module():
    """Import Explainability.verifier with the same by-path fallback."""
    try:
        from Explainability import verifier as _v
        return _v
    except (ModuleNotFoundError, ImportError):
        import importlib.util
        _here = os.path.dirname(os.path.abspath(__file__))
        _spec = importlib.util.spec_from_file_location(
            "explainability_verifier", os.path.join(_here, "verifier.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod


# ── Client ───────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Minimal OpenAI-compatible chat client for local open-weights servers.

    Deterministic by construction: temperature 0.0 and a fixed seed (note that
    bitwise determinism across hardware/backends is not guaranteed by every
    runtime, which is why regeneration + verification stay cheap).

    `transport` is an injectable callable(payload: dict) -> str used by tests;
    when set, no network is touched.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 temperature: float = 0.0, seed: int = 0, timeout: int = 120,
                 transport: Optional[Callable[[Dict[str, Any]], str]] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.transport = transport

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "seed": self.seed,
            "stream": False,
        }
        if self.transport is not None:
            return self.transport(payload)
        import requests
        try:
            resp = requests.post(f"{self.base_url}/chat/completions",
                                 json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                _SETUP_HINT.format(url=self.base_url, model=self.model)) from e
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You turn verified facts about an anomaly-detection model-selection run "
    "into clear, plain-language prose for a reader who understands anomaly "
    "detection but not this framework's internals — explain the framework's "
    "own terms in plain words, keeping standard metric names (F1, PR-AUC) as "
    "they are.\n"
    "Rules — follow every one strictly:\n"
    "1. Use ONLY the numbered fact sentences given to you. Do not add facts, "
    "numbers, names, comparisons, or causes of your own.\n"
    "2. Copy every number and every model/detector name EXACTLY as written in "
    "the facts. Never re-round, convert, or estimate. A qualifier such as "
    "'(rank 2)' or '(negative influence)' belongs ONLY to the value it "
    "accompanies in the facts — never re-attach it to a different value.\n"
    "3. Every fact marked [REQUIRED] must be conveyed. Unmarked facts may be "
    "omitted if space demands.\n"
    "4. Lines marked [CAVEAT] are limitations, not findings: weave the "
    "relevant ones in briefly (e.g. 'note that ...').\n"
    "5. If a value reads 'not_available', either omit it or say the data is "
    "not available — never fill it in.\n"
    "6. Write ONE coherent paragraph of plain prose. No headings, lists, "
    "tables, or markdown.\n"
    "7. Write every detector name in full each time it appears. Never "
    "compress names into ranges, plurals, or shared prefixes: 'CBLOF_1 to "
    "-4', 'CBLOF_1-4', and 'LOFs 2 and 3' are all forbidden — write CBLOF_1, "
    "CBLOF_2, CBLOF_3, CBLOF_4. Likewise never merge numbers belonging to "
    "different detectors into a range such as 'from -0.0124 to -0.0243'; "
    "state each value next to the detector it belongs to."
)


def _render_value(v: Any) -> str:
    if isinstance(v, list):
        return ", ".join(_render_value(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)


def _output_lines(output: Dict[str, Any]) -> List[str]:
    return [f"- {k}: {_render_value(v)}" for k, v in sorted(output.items())]


def _word_budget(n_atoms: int, lo: int = 120, hi: int = 220) -> tuple:
    """Word budget for the WHOLE narrative, scaled with atom count so the floor
    matches how much there is to say: dense stages (many detectors/rules) get
    more room instead of triaging required facts out of a fixed paragraph, and
    sparse stages (e.g. the 2-source final aggregation — a couple of facts) get
    a low floor instead of being padded to 120 words of filler."""
    if n_atoms <= 3:
        return 40, 90
    return lo, min(400, hi + 8 * max(0, n_atoms - 12))


def _fact_lines(ir_doc: Dict[str, Any]) -> List[str]:
    required = set(ir_doc.get("required_atom_ids", []))
    evidence = ir_doc.get("evidence", [])
    # Atoms may carry a presentation `order` (e.g. rank-aggregation sources
    # best-Borda-rank first); ordered atoms come first, the rest keep their
    # file (id-sorted) position. The narrator tends to follow fact order.
    def _key(pair):
        idx, atom = pair
        o = atom.get("order")
        return (float(o) if o is not None else float("inf"), idx)
    lines = []
    for n, (_, atom) in enumerate(sorted(enumerate(evidence), key=_key), 1):
        marker = "[REQUIRED] " if atom.get("id") in required else ""
        lines.append(f"{n}. {marker}{atom.get('text', '')}")
    return lines


def _caveat_lines(ir_doc: Dict[str, Any]) -> List[str]:
    caveats = ir_doc.get("caveats", [])
    if not caveats:
        return []
    lines = ["", "CAVEATS (limitations to weave in where relevant, not findings):"]
    lines.extend(f"- [CAVEAT] {c.get('text', '')}" for c in caveats)
    return lines


def build_stage_prompt(ir_doc: Dict[str, Any]) -> str:
    lo, hi = _word_budget(len(ir_doc.get("evidence", [])))
    question = ir_doc.get("question")
    lines: List[str] = []
    lines.append(f"STAGE: {ir_doc.get('stage')}")
    lines.append(f"DATASET: {ir_doc.get('dataset')}  |  ENTITY: {ir_doc.get('entity')}")
    if question:
        lines.append(f"QUESTION THIS STAGE ANSWERS: {question}")
    lines.append("")
    lines.append("STAGE OUTPUT (context facts):")
    lines.extend(_output_lines(ir_doc.get("output", {})))
    lines.append("")
    lines.append("FACTS (use only these; copy numbers and names exactly):")
    lines.extend(_fact_lines(ir_doc))
    lines.extend(_caveat_lines(ir_doc))
    lines.append("")
    task = (f"TASK: Write ONE paragraph of {lo}-{hi} words")
    if question:
        task += (" that answers the question above, leading with the answer and "
                 "then presenting the facts in the order given as supporting "
                 "evidence")
    else:
        task += " explaining this stage's result"
    task += (". Convey every fact marked as required; copy all numbers and names "
             "verbatim, and keep each number attached to the exact metric name it "
             "accompanies in the facts.")
    lines.append(task)
    return "\n".join(lines)


def build_global_prompt(global_ir: Dict[str, Any]) -> str:
    """
    The global prompt is fact-based like the stage prompts: the assembler
    pre-renders the decision, one summary sentence-set per available stage,
    and the agreement facts as atoms — no raw key:value dumps, which small
    models misread into invented relations.
    """
    lo, hi = _word_budget(len(global_ir.get("evidence", [])), 150, 300)
    lines: List[str] = []
    lines.append("GLOBAL MODEL-SELECTION DECISION")
    lines.append(f"DATASET: {global_ir.get('dataset')}  |  ENTITY: {global_ir.get('entity')}")
    lines.append("")
    lines.append("FACTS (use only these; copy numbers and names exactly):")
    lines.extend(_fact_lines(global_ir))
    unavailable = [stage for stage, info in sorted(global_ir.get("stages", {}).items())
                   if info.get("status") != "ok"]
    if unavailable:
        lines.append("")
        lines.append("STAGES WITHOUT DATA (state as unavailable if mentioned; "
                     "never invent their results): " + ", ".join(unavailable))
    lines.extend(_caveat_lines(global_ir))
    lines.append("")
    lines.append(f"TASK: Write ONE paragraph of {lo}-{hi} words. Lead with the "
                 "final framework decision, then summarize what each available "
                 "stage found and where the stages agreed or disagreed with the "
                 "final pick. Copy all numbers and names verbatim, and keep each "
                 "number attached to the exact metric name it accompanies in the "
                 "facts.")
    return "\n".join(lines)


def narrate_stage(ir_doc: Dict[str, Any], client: LLMClient) -> str:
    return client.chat(SYSTEM_PROMPT, build_stage_prompt(ir_doc)).strip()


def narrate_global(global_ir: Dict[str, Any], client: LLMClient) -> str:
    return client.chat(SYSTEM_PROMPT, build_global_prompt(global_ir)).strip()


# ── Verifier-guided repair ───────────────────────────────────────────────────

def _violation_count(metrics: Dict[str, Any]) -> int:
    return (len(metrics.get("unsupported_numbers", []))
            + len(metrics.get("unsupported_entities", []))
            + len(metrics.get("misattributed_numbers", []))
            + len(metrics.get("missing_required_ids", [])))


def _violation_lines(metrics: Dict[str, Any], ir_doc: Dict[str, Any]) -> List[str]:
    """Human-readable repair feedback for every hard violation the verifier
    found (attribution warnings are diagnostic-only and not repaired)."""
    lines: List[str] = []
    for tok in metrics.get("unsupported_numbers", []):
        lines.append(f"The number '{tok}' does not appear in the facts. Remove "
                     f"it or use the exact value written in the facts. If it "
                     f"came from splitting a detector name (e.g. 'LOFs 2 and "
                     f"3'), write each full name instead.")
    for tok in metrics.get("unsupported_entities", []):
        lines.append(f"The name '{tok}' does not appear in the facts — remove it.")
    for m in metrics.get("misattributed_numbers", []):
        subjects = ", ".join(m.get("subjects", [])) or "the detectors it names"
        lines.append(f"The number '{m.get('number')}' is used in a sentence "
                     f"about {subjects}, but it does not belong to any of "
                     f"them. Re-check the facts and attach it to the right "
                     f"detector.")
    atoms_by_id = {a.get("id"): a for a in ir_doc.get("evidence", [])}
    for rid in metrics.get("missing_required_ids", []):
        atom = atoms_by_id.get(rid)
        if atom is not None:
            lines.append(f"This required fact was not conveyed: "
                         f"\"{atom.get('text', '')}\"")
    return lines


def _repair_prompt(base_prompt: str, draft: str, problems: List[str]) -> str:
    return (base_prompt
            + "\n\nYOUR PREVIOUS DRAFT:\n" + draft
            + "\n\nPROBLEMS DETECTED IN THE DRAFT — fix ALL of them:\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\n\nRewrite the paragraph, fixing every problem above while "
              "still following all the rules and the original task.")


# ── Entity-level orchestration ───────────────────────────────────────────────

def _stage_file_map(iteration: int) -> Dict[str, str]:
    return {
        "thompson_sampling": "ir_thompson",
        "ga_selection": "ir_ga_selection",
        "ga_combination": "ir_ga_combination",
        "rank_aggregation_robust": f"ir_rank_aggregation_robust_{iteration}",
        "rank_aggregation_final": f"ir_rank_aggregation_final_{iteration}",
        "monte_carlo": "ir_monte_carlo",
        "off_by_threshold": "ir_off_by",
    }


def narrate_entity(dataset: str, entity: str, iteration: int, client: LLMClient,
                   base_dir: str = "myresults/explanations_ir",
                   out_dir: str = "myresults/explanations_nl",
                   stages: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Narrate every available IR file for (dataset, entity, iteration) and score
    each narrative with the atom-matching verifier. Writes:
        {out_dir}/{ds}/{ent}/nl_{stage}.txt          (per stage)
        {out_dir}/{ds}/{ent}/nl_global_iter{n}.txt
        {out_dir}/{ds}/{ent}/faithfulness_iter{n}.json / .txt
    Per-stage failures are recorded, not fatal; missing IR files are 'skipped'.
    """
    verifier = _verifier_module()
    ir_dir = os.path.join(base_dir, str(dataset), str(entity))
    nl_dir = os.path.join(out_dir, str(dataset), str(entity))
    os.makedirs(nl_dir, exist_ok=True)

    def _load(fname: str, pattern: Optional[str] = None) -> Optional[Dict[str, Any]]:
        path = os.path.join(ir_dir, f"{fname}.json")
        if not os.path.exists(path) and pattern:
            # Tolerate iteration-number mismatches between pipeline phases:
            # fall back to the newest file matching the stage pattern.
            candidates = glob.glob(os.path.join(ir_dir, pattern))
            if candidates:
                path = max(candidates, key=os.path.getmtime)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    file_map = _stage_file_map(iteration)
    wanted = set(stages) if stages else set(file_map) | {"global"}
    report: Dict[str, Any] = {"dataset": str(dataset), "entity": str(entity),
                              "iteration": int(iteration), "model": client.model,
                              "stages": {}}

    def _run_one(stage_key: str, ir_doc: Dict[str, Any], nl_name: str,
                 is_global: bool) -> None:
        try:
            base_prompt = (build_global_prompt(ir_doc) if is_global
                           else build_stage_prompt(ir_doc))
            verify_fn = (verifier.verify_global if is_global
                         else verifier.verify_narrative)
            narrative = client.chat(SYSTEM_PROMPT, base_prompt).strip()
            metrics = verify_fn(narrative, ir_doc)
            entry: Dict[str, Any] = {"status": "ok"}

            # Verifier-guided repair: one bounded retry when the draft has
            # hard violations. The repaired draft is kept only if it is no
            # worse; both metric sets are recorded (pre-repair as
            # `verify_initial`).
            problems = _violation_lines(metrics, ir_doc)
            if problems:
                entry["verify_initial"] = metrics
                entry["repaired"] = True
                repaired = client.chat(
                    SYSTEM_PROMPT,
                    _repair_prompt(base_prompt, narrative, problems)).strip()
                repaired_metrics = verify_fn(repaired, ir_doc)
                if _violation_count(repaired_metrics) <= _violation_count(metrics):
                    narrative, metrics = repaired, repaired_metrics
                else:
                    entry["repair_discarded"] = True

            # A fixed glossary is appended verbatim AFTER verification, so its
            # definitions are never reworded by the model and never counted as
            # claims by the verifier.
            footer = ir_doc.get("info_footer")
            path = os.path.join(nl_dir, f"{nl_name}.txt")
            with open(path, "w") as f:
                f.write(narrative + "\n")
                if footer:
                    f.write(f"\nINFO: {footer}\n")
            entry.update({"narrative_path": path,
                          "words": len(narrative.split()), "verify": metrics})
            report["stages"][stage_key] = entry
        except ConnectionError:
            raise
        except Exception as e:  # non-fatal per stage
            report["stages"][stage_key] = {"status": "error", "error": str(e)}

    patterns = {
        "rank_aggregation_robust": "ir_rank_aggregation_robust_*.json",
        "rank_aggregation_final": "ir_rank_aggregation_final_*.json",
    }
    for stage_key, fname in file_map.items():
        if stage_key not in wanted:
            continue
        ir_doc = _load(fname, patterns.get(stage_key))
        if ir_doc is None:
            report["stages"][stage_key] = {"status": "skipped",
                                           "reason": f"{fname}.json not found"}
            continue
        nl_name = fname.replace("ir_", "nl_", 1)
        _run_one(stage_key, ir_doc, nl_name, is_global=False)

    if "global" in wanted:
        global_doc = _load(f"ir_global_iter{iteration}", "ir_global_iter*.json")
        if global_doc is None:
            report["stages"]["global"] = {
                "status": "skipped",
                "reason": f"ir_global_iter{iteration}.json not found"}
        else:
            _run_one("global", global_doc, f"nl_global_iter{iteration}",
                     is_global=True)

    # Micro-averaged overall rates across the verified narratives.
    tot_claims = tot_unsupported = tot_required = tot_missing = 0
    for info in report["stages"].values():
        v = info.get("verify")
        if not v:
            continue
        tot_claims += v["n_claims"]
        tot_unsupported += (len(v["unsupported_numbers"])
                            + len(v["unsupported_entities"])
                            + len(v.get("misattributed_numbers", [])))
        tot_required += v["n_required"]
        tot_missing += len(v["missing_required_ids"])
    report["overall"] = {
        "hallucination_rate": (tot_unsupported / tot_claims) if tot_claims else 0.0,
        "omission_rate": (tot_missing / tot_required) if tot_required else 0.0,
        "n_claims": tot_claims, "n_required": tot_required,
    }

    json_path = os.path.join(nl_dir, f"faithfulness_iter{iteration}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, sort_keys=True, indent=2)
    txt_path = os.path.join(nl_dir, f"faithfulness_iter{iteration}.txt")
    with open(txt_path, "w") as f:
        f.write("=== Narrative Faithfulness Report ===\n")
        f.write(f"Dataset: {dataset}  |  Entity: {entity}  |  Iteration: {iteration}\n")
        f.write(f"Model: {client.model}\n\n")
        f.write(f"{'stage':<26} {'status':>8} {'words':>6} {'halluc.':>8} "
                f"{'omiss.':>7} {'warn':>5} {'rep':>4}\n")
        f.write("-" * 71 + "\n")
        for stage_key in sorted(report["stages"]):
            info = report["stages"][stage_key]
            v = info.get("verify") or {}
            halluc = f"{v['hallucination_rate']:.3f}" if v else "-"
            omiss = f"{v['omission_rate']:.3f}" if v else "-"
            warn = str(len(v.get("attribution_warnings", []))) if v else "-"
            rep = "yes" if info.get("repaired") else "-"
            f.write(f"{stage_key:<26} {info['status']:>8} "
                    f"{str(info.get('words', '-')):>6} {halluc:>8} {omiss:>7} "
                    f"{warn:>5} {rep:>4}\n")
        ov = report["overall"]
        f.write("-" * 71 + "\n")
        f.write(f"overall hallucination rate: {ov['hallucination_rate']:.3f} "
                f"({ov['n_claims']} claims)\n")
        f.write(f"overall omission rate     : {ov['omission_rate']:.3f} "
                f"({ov['n_required']} required atoms)\n")
    report["faithfulness_json"] = json_path
    report["faithfulness_txt"] = txt_path
    return report
