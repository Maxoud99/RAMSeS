"""
LLM narration layer: renders the Intermediate Representation (IR) JSONs into
natural-language explanations with a LOCAL open-weights model, and scores every
narrative with the atom-matching faithfulness verifier.

The client speaks the OpenAI-compatible chat API (default: Ollama at
http://localhost:11434/v1, default model qwen2.5:14b-instruct) at temperature 0
with a fixed seed. Any local server exposing the same API (LM Studio,
llama.cpp server, vLLM) works via `base_url`. The pipeline never depends on
this layer: narratives are generated on demand from the IR files an
`--explain` run produced (see Explainability/narrate.py).

The contract lives in SYSTEM_PROMPT. The model writes the facts as prose in
its own words — merging, connecting and ordering them freely — but every
number, name and relationship must come from a fact, nothing may be derived
(no totals, no positions a fact does not give), a fact naming two detectors
keeps each in its role, and [CAVEAT] lines are respected without being stated,
since the card renders those verbatim from the IR in a section of their own.
Facts marked [AS GIVEN] are lists of measured values and go in as written.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from Utils.pipeline_spec import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL

# Re-exported under the names this module has always used, so callers and tests
# keep working; the values themselves live in the shared spec.
DEFAULT_BASE_URL = DEFAULT_LLM_BASE_URL
DEFAULT_MODEL = DEFAULT_LLM_MODEL

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
                 # 600s, not 120: the Thompson stage alone exceeds two
                 # minutes on a 14B model, and it fails quietly — narrate_entity
                 # records the error and the run still reports success.
                 temperature: float = 0.0, seed: int = 0, timeout: int = 600,
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
            # Qwen3-generation models think by default: without this the
            # reasoning block alone exhausts the timeout.
            "reasoning_effort": "none",
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
    "You write the findings of an anomaly-detection model-selection run as "
    "clear prose, for a researcher who knows anomaly detection but not this "
    "framework. You are writing an explanation someone will actually read, not "
    "a list.\n"
    "\n"
    "GROUNDING. Every number, name and relationship must come from the facts "
    "given. Add no cause, no comparison and no characterisation the facts do "
    "not state. Never print an internal field name, and never guess what one "
    "means. Write a stage's name as words.\n"
    "\n"
    "DIGITS. Numbers appear exactly as given — never re-rounded, never spelled "
    "as words.\n"
    "\n"
    "AS GIVEN. A fact marked [AS GIVEN] is a list of measured values. Put it in "
    "your prose as written — do not reorder it, do not shorten it, do not give "
    "anything in it a position, and do not move a named entry into a group the "
    "fact sets aside. Everything else you may write freely.\n"
    "\n"
    "STATE, DO NOT DERIVE. Every figure you write must appear in a fact. Where a "
    "winner is compared against several rivals one at a time, each count is its "
    "own head-to-head comparison over the same injected points — the sets "
    "overlap, so there is no total and adding them means nothing. Never say how "
    "many points the winner handled altogether. More generally: do not "
    "add counts together, do not work out a total or a share, and do not award a "
    "position — 'second', 'third', 'the runner-up' — from a list unless a fact "
    "says so. Where a fact sets a group aside ('the remaining 4.8% went to 3 "
    "further detectors'), that group is separate from the ones already named; do "
    "not fold a named one into it.\n"
    "\n"
    "ROLES. Where a fact names more than one detector or source, each keeps the "
    "role that fact gives it — which one leads, which one was beaten, which one "
    "was preferred instead. Swapping them reverses the finding:\n"
    "  fact: 'A_1 shaped the consensus most, yet its own ranking put B_1 first "
    "rather than the consensus's C_1.'\n"
    "  WRONG: 'A_1 shaped it most, ranking C_1 first in its own ranking.'\n"
    "  RIGHT: 'A_1 shaped it most, though it would itself have put B_1 on top "
    "rather than C_1.'\n"
    "\n"
    "PROSE. Answer the stage's question first, then explain what supports it. "
    "Connect your sentences — vary how they open, use 'Similarly', 'By "
    "contrast', 'Notably' where they fit, and group related findings into one "
    "sentence where that reads better. Continuous plain prose, no markdown and "
    "no lists.\n"
    "\n"
    "SCOPE. Cover every fact marked [REQUIRED]. Facts not so marked are "
    "optional. Lines marked [CAVEAT] are limits for you to respect while "
    "writing; the reader is shown them separately, so no sentence of yours may "
    "convey one:\n"
    "  caveat: 'For A_1, B_1 most folds had near-constant F1, so the held-out R2 "
    "is not a meaningful fidelity estimate'\n"
    "  WRONG: 'A_1 was excluded from meaningful fidelity estimates.'\n"
    "  RIGHT: (write nothing about it)"
)


def _render_value(v: Any) -> str:
    if isinstance(v, list):
        return ", ".join(_render_value(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)


# Unused since build_stage_prompt stopped listing the stage output.
def _output_lines(output: Dict[str, Any]) -> List[str]:
    return [f"- {k}: {_render_value(v)}" for k, v in sorted(output.items())]


# Unused since the v5 prompt: build_stage_prompt no longer sizes a word
# budget or appends a per-stage hint. Kept commented rather than deleted —
# these are model-size dependent and a smaller narrator would want them back.
# def _content_words(ir_doc: Dict[str, Any]) -> int:
#     """How many words of material the narrative actually has to convey.
#
#     REQUIRED evidence only: sizing to every atom left room for all of them, so
#     the optional ones were optional in name only. Caveats are excluded — they
#     are shown from the IR, and a floor above what there is to say forces
#     padding.
#     """
#     required = set(ir_doc.get("required_atom_ids") or ())
#     evidence = ir_doc.get("evidence", [])
#     marked = [a for a in evidence if a.get("id") in required]
#     # A required list naming no present atom is malformed, not empty: a 0 here
#     # would floor a 500-word stage at 40.
#     return sum(len(str(a.get("text", "")).split()) for a in (marked or evidence))
#
#
# # The floor is deliberately BELOW the content length: the narrative restates the
# # facts in connected prose, which compresses (shared subjects, pronouns) at least
# # as much as connectives add.
# _BUDGET_FLOOR_RATIO = 0.9
# _BUDGET_CEILING_RATIO = 2.2
# _BUDGET_MIN_FLOOR = 40
# The default ceiling, and what the floor is clamped against so the two can
# never cross.
# _BUDGET_HARD_CAP = 400


# Unused since neither prompt sizes a word budget.
# def _word_budget(n_atoms: int, lo: int = 120, hi: int = 220,
#                  content_words: Optional[int] = None) -> tuple:
#     """Word budget for the WHOLE narrative, scaled to how much there is to say.
#
#     Driven by the atoms' CONTENT LENGTH, not their count: consolidating
#     near-identical atoms cuts the count without cutting the material, and a
#     count-based floor then demands more words than the facts contain — which
#     the narrator supplies by inventing them.
#
#     Falls back to the count-based curve when the caller has no document.
#     """
#     if content_words:
#         floor = max(_BUDGET_MIN_FLOOR, int(content_words * _BUDGET_FLOOR_RATIO))
#         ceiling = max(floor + 40, int(content_words * _BUDGET_CEILING_RATIO))
#         # Clamped at BOTH ends. Capping the ceiling alone inverted the range
#         # past ~445 words of facts ("between 639 and 400 words"), and a model
#         # handed a contradictory range compresses by dropping the numbers.
#         if floor > _BUDGET_HARD_CAP - 40:
#             floor = _BUDGET_HARD_CAP - 40
#         return floor, min(_BUDGET_HARD_CAP, max(ceiling, floor + 40))
#     if n_atoms <= 3:
#         return 65, 120
#     return lo, min(_BUDGET_HARD_CAP, hi + 8 * max(0, n_atoms - 12))


# Unused since the v5 prompt: build_stage_prompt no longer sizes a word
# budget or appends a per-stage hint. Kept commented rather than deleted —
# these are model-size dependent and a smaller narrator would want them back.
# # Stages where the narrative must carry one statement per atom rather than a
# # summary, so the budget has to scale past the default 400-word ceiling.
# # (words_per_atom, base, ceiling), keyed by exact stage name.
# _STAGE_WORD_BUDGETS: Dict[str, tuple] = {
#     # Thompson narrates every regime individually; ~20 words each plus the
#     # lead, regime summary, winner context feature and state line.
#     "thompson_sampling": (20, 40, 700),
#     # The ranking sibling narrates regimes too, but plain run-length encoding of
#     # the ||mu||^2 leader yields a handful of them rather than a dozen, so it
#     # needs a lower ceiling than the stage above.
#     "thompson_ranking": (20, 40, 500),
# }
# # NEITHER point-injection stage gets an entry, deliberately. GAN had one —
# # (30, 40, 900) — and asked for 640-900 words it wrote 325 and dropped 36% of
# # its required atoms including the winner. A floor far above what the model
# # wants to write is not a nudge to write more, it makes it reorganise. The
# # ceiling does not bind either way: both stages routinely write past 400 with
# # zero omissions.
#
#
# def _stage_word_budget(stage: Any, n_atoms: int,
#                        content_words: Optional[int] = None) -> Optional[tuple]:
#     """Per-atom allowance for the stages that narrate one statement per atom.
#
#     `content_words` widens the range when the per-atom allowance lands below
#     what the required facts run to. The curve counts atoms, not their length,
#     and Thompson's regime sentences are long enough that the two nearly cross:
#     seven required atoms buy 180-270 words against 252 of fact. Below that it is
#     the contradictory range `_word_budget` documents.
#     """
#     cfg = _STAGE_WORD_BUDGETS.get(str(stage))
#     if not cfg:
#         return None
#     per, base, cap = cfg
#     lo = min(cap - 60, base + per * max(0, n_atoms))
#     hi = min(cap, int(lo * 1.5))
#     if content_words:
#         floor = int(content_words * _BUDGET_FLOOR_RATIO)
#         if hi < floor:
#             lo = min(lo, floor)
#             hi = min(cap, max(floor + 60, int(floor * 1.4)))
#     return lo, hi
#
#
_LIST_FACT_RE = re.compile(r"\d+\.\d+%")


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
    # Bulleted, never numbered: numbering handed the narrator a citation
    # handle it used ("... for their high utility (fact 2)").
    lines = []
    for _, atom in sorted(enumerate(evidence), key=_key):
        marker = "[REQUIRED] " if atom.get("id") in required else ""
        lines.append(f"- {marker}{atom.get('text', '')}")
    return lines


def _caveat_lines(ir_doc: Dict[str, Any]) -> List[str]:
    caveats = ir_doc.get("caveats", [])
    if not caveats:
        return []
    lines = ["", "CAVEATS (limits to respect; the reader is shown these "
                 "separately, so do not restate them):"]
    lines.extend(f"- [CAVEAT] {c.get('text', '')}" for c in caveats)
    return lines


# Shared by off_by_threshold and gan — ONE string, because two near-copies
# drifted: the threshold ban reached gan only, and off_by then reproduced 0 of
# its 11 rule thresholds against gan's 11 of 11. The ban has to be by phrasing,
# since the verifier scores an atom as covered from its subject and win count
# alone and cannot see a missing threshold. The importance figures are
# subordinated because they are optional yet crowded out the required rules.
# Unused since the v5 prompt: build_stage_prompt no longer sizes a word
# budget or appends a per-stage hint. Kept commented rather than deleted —
# these are model-size dependent and a smaller narrator would want them back.
# _POINT_INJECTION_HINT = (
#     " Open with one short sentence naming the highest-ranked model, then "
#     "give each fact about the models it beat as its OWN separate sentence. "
#     "The rival models named in a sentence must be EXACTLY the models that "
#     "fact lists. State every condition WITH ITS NUMBERS, copied exactly as "
#     "the fact writes them: never replace a threshold with 'specific ranges', "
#     "'certain conditions', 'specific values', 'certain constraints', "
#     "'specific criteria', 'conditions related to' or a bare list of property "
#     "names. A sentence that names a property without its number is wrong. "
#     "The importance figures come last, and only once every condition has "
#     "been stated with its numbers. If a fact says the highest-ranked model "
#     "never exclusively beat some models, state that too."
# )
#
# # Stage-specific rendering guidance appended to the prompt's TASK. A NARRATION
# # concern, so it lives here rather than in the grounded IR. Keyed by exact
# # stage name.
# _STAGE_TASK_HINTS: Dict[str, str] = {
#     # The opening sentence is load-bearing: without it the narrator went
#     # straight into the per-source walk and dropped both the consensus winner
#     # and the source list (omission 0.000 -> 0.250). One positive instruction
#     # replaced three defensive ones and scored better.
#     "rank_aggregation_robust": (
#         " Open by naming the source ranking that shaped the consensus most — "
#         "all of them, if several are tied at that rank — and the source "
#         "rankings being aggregated. Then, in a sentence of its own, name the "
#         "consensus's own top-ranked detector. That detector is the consensus's "
#         "pick, never a source's: do not attach it to any source's own ranking. "
#         "Then describe each source in the order "
#         "given; for each one, state its overall standing rank, its influence "
#         "rank, and its agreement rank. A rank is a position — rank "
#         "1 is best — so give the rank number itself. NEVER call a rank "
#         "'highest', 'lowest', 'best', 'worst', 'least', 'strongest' or "
#         "'weakest' — write the ordinal or the number instead. Keep the fact's "
#         "own wording for overall standing, such as 'shaped the consensus third "
#         "most', and never restate a standing as influence or agreement. Do not "
#         "compare one source's rank with another's unless a fact states that "
#         "comparison."
#     ),
#     # Two facts about different KINDS of thing. The narrator joined them into
#     # "LOF_3, which aligns more closely with Thompson_Sampling's ranking",
#     # inverting the finding on SKAB/7 at 0.000 hallucination.
#     "rank_aggregation_final": (
#         " Say which of the two sources the consensus followed more closely, "
#         "with both agreement scores and the gap. Then, in a sentence of its "
#         "own, name the consensus's top-ranked detector. Agreement is a property "
#         "of a whole source ranking, never of one detector, so do not write that "
#         "a detector agrees with, aligns with or is closer to either source."
#     ),
#     "ga_combination": (
#         " Describe each detector in the order given; for each one, state its "
#         "overall weight rank and its rank on absolute SHAP, PFI and total ALE "
#         "(rank 1 is strongest). Where a fact says a rank is a tie, say it is "
#         "tied. Finish with the sign summary, saying which detectors push "
#         "the meta-learner toward flagging an anomaly and which push the other "
#         "way. Report each sign exactly as the facts give it; how well a sign "
#         "is supported is a caveat, so leave it out of the paragraph. A detector "
#         "the facts give no sign at all keeps none — never assign it one of "
#         "your own."
#     ),
#     "ga_selection": (
#         " Open by naming the chosen ensemble. Then explain why the chosen "
#         "detectors were kept, following the facts in order and keeping the "
#         "detectors grouped exactly as the facts group them. Then explain why "
#         "the rest were left out, using the high/low utility and stability "
#         "wording the facts use."
#     ),
#     # Direction is the whole risk. Shares are sums of squares, so none can
#     # "drag the score down" — but that is what a narrator writes when a share
#     # is small, and the verifier cannot see it. Only the rival comparison has
#     # a sign.
#     "thompson_ranking": (
#         " Open with the winner and its score, then the context features its score is "
#         "built from. Describe those context features only as larger or smaller shares "
#         "of that detector's own score — a small share means a context feature "
#         "contributed little, never that it lowered the score or worked against "
#         "the detector. Only when comparing the winner with the named runner-up "
#         "may you say a context feature favoured one over the other, and there keep the "
#         "direction exactly as the fact states it. Then give the selection "
#         "counts, then how leadership divided into regimes and any fact comparing "
#         "that with the winner, then EVERY regime "
#         "its own sentence in the order listed, each naming its window range, "
#         "its leader and its context features. "
#         # The number or the window range is what pairs each sentence with its
#         # own figure in the disclosure. A word-ordinal cannot, and fails
#         # silently: the disclosure falls back to the IR's own wording.
#         "Begin each of those sentences with the literal words 'Regime N "
#         "(windows ...)', using the number the fact gives — never 'the first "
#         "regime', 'the second regime', or any other ordinal in place of it. "
#         "Name that leader outright — never "
#         "describe it by reference to the previous regime. Do not draw a conclusion "
#         "the facts do not state."
#     ),
#     # Describing a regime by reference to the previous one writes false
#     # continuity ("NN_3 continued as leader") that no metric can see, since the
#     # names and numbers are all correct.
#     "thompson_sampling": (
#         " Open with the winner and its margin, then how the run divided into "
#         "regimes. Then give EVERY regime its own sentence, in the order listed, "
#         "keeping each regime's window range, its leader and its context features "
#         "together. Begin each of those sentences with the literal words "
#         "'Regime N (windows ...)', then the detector that led it, then its "
#         "context features. Name that detector outright — never describe it by reference "
#         "to the previous regime. Three different things are said about context features "
#         "and they must not be merged or traded for one another: one context feature "
#         "SUPPLIES a share of a detector's expected reward, one GIVES IT AN EDGE "
#         "over the named rival, and one DEPARTS FURTHEST FROM ITS USUAL "
#         "contribution. The last is a separate sentence in the facts and must "
#         "stay separate clauses. Keep whichever wording the fact uses. "
#         # The rigidity IS the fidelity. Asking for varied openings cost half the
#         # content: SKAB/7 went 8/8 -> 1/8 regimes keeping all three claims.
#         "Keep each regime to a SINGLE sentence — never split a trailing clause "
#         "off into a sentence of its own, and never refer back with 'in this "
#         "regime' or 'here'. "
#         "Finish with the winner's overall context feature and the selection-state "
#         "percentages."
#     ),
#     "monte_carlo": (
#         " Open with one sentence restating the production-test result exactly "
#         "as the fact gives it — use the word 'first' — naming the top detector "
#         "for each metric. The F1 and PR-AUC leaders are not always the same "
#         "detector: if the fact names two different ones, keep them separate. "
#         "Then give each detector's winning noise ranges in the order listed, "
#         "one detector per statement. Finish with the win percentages. Copy each "
#         "noise range as it is written ('from 0.000 to 0.042') — never turn a "
#         "range into a hyphenated pair."
#     ),
#     "off_by_threshold": _POINT_INJECTION_HINT,
#     "gan": _POINT_INJECTION_HINT,
# }
# # Two clauses above must survive any future trim. monte_carlo's hyphenated-range
# # ban: the verifier's number extraction is sign-aware, so "0.000-0.042" reads as
# # -0.042 and is flagged unsupported. The degenerate clause: dropping it lost
# # ob.degenerate (omission 0.000 -> 0.200).
#
#
# def _stage_task_hint(stage: Any) -> str:
#     return _STAGE_TASK_HINTS.get(str(stage), "")
#
#
def build_stage_prompt(ir_doc: Dict[str, Any]) -> str:
    question = ir_doc.get("question")
    lines: List[str] = []
    # Words, not the identifier: handed "rank_aggregation_robust" the narrator
    # opens "The rank_aggregation_robust stage found that...".
    lines.append(f"STAGE: {str(ir_doc.get('stage', '')).replace('_', ' ')}")
    # Spelled out: "ENTITY: 7" was read as "seven entities".
    lines.append(f"DATASET: {ir_doc.get('dataset')}   ENTITY: "
                 f"{ir_doc.get('entity')} (one time series from that dataset)")
    if question:
        lines.append(f"QUESTION THIS STAGE ANSWERS: {question}")
    lines.append("")
    # The stage output is not listed: labelled background it was still narrated,
    # and everything that matters is an atom.
    lines.append("FACTS:")
    # A fact enumerating several measured values is where the narrator reranks
    # and regroups, so those are marked to be reproduced as given.
    for line in _fact_lines(ir_doc):
        if len(_LIST_FACT_RE.findall(line)) >= 3:
            line = line.replace("- ", "- [AS GIVEN] ", 1)
        lines.append(line)
    lines.extend(_caveat_lines(ir_doc))
    lines.append("")
    lines.append("TASK: Explain what this stage found, as prose.")
    return "\n".join(lines)



def build_global_prompt(global_ir: Dict[str, Any]) -> str:
    """
    The global prompt is fact-based like the stage prompts: the assembler
    pre-renders the decision, one summary sentence-set per available stage,
    and the agreement facts as atoms — no raw key:value dumps, which small
    models misread into invented relations.
    """
    # No word budget, as for the stage prompts: sized from the atoms it asked
    # for more words than the facts contained, and the model supplied them.
    # lo, hi = _word_budget(len(global_ir.get("evidence", [])), 150, 300)
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
    lines.append("TASK: Write ONE paragraph. Lead with the "
                 "final framework decision, then summarize what each available "
                 "stage found and where the stages agreed or disagreed with the "
                 "final pick. Copy all numbers and names verbatim, and keep each "
                 "number attached to the exact metric name it accompanies in the "
                 "facts.")
    return "\n".join(lines)


# ── Global narrative: deterministic merge ────────────────────────────────────
# Selected by `global_mode` on narrate_entity. "concat" stitches the narrated
# per-stage prose together, adding no new claims, so it inherits the per-stage
# faithfulness and is not scored again. "llm" narrates the global IR's own
# atoms — the original path, kept working.
GLOBAL_MODES = ("concat", "llm")

# The merged document follows the pipeline's order so it reads as the run ran,
# rather than the alphabetical order the IR files happen to load in.
_GLOBAL_STAGE_ORDER = (
    "ga_selection", "ga_combination",
    # The ranking criterion first: it explains the ordering the pipeline goes
    # on to consume. WebUI.artifacts.STAGES must stay in this order.
    "thompson_ranking", "thompson_sampling",
    # Robustness reads broadest perturbation to narrowest. A reading order,
    # not the execution order — GAN still runs first, at sub-stage 6.3.
    # NOTE: no parentheses in this comment — WebUI.test_webui parses this tuple
    # with a non-greedy regex that would stop at the first closing bracket.
    "monte_carlo", "off_by_threshold", "gan",
    "rank_aggregation_robust", "rank_aggregation_final",
)

_GLOBAL_STAGE_TITLES = {
    # Two stages explain one algorithm, so neither claims the plain name.
    # Duplicated verbatim in WebUI.artifacts.STAGES.
    "ga_selection": "Genetic Algorithm: Selection",
    "ga_combination": "Genetic Algorithm: Combination",
    "thompson_ranking": "Thompson Sampling: Ranking",
    "thompson_sampling": "Thompson Sampling: Selection",
    "monte_carlo": "Robustness: Monte Carlo",
    "off_by_threshold": "Robustness: Off-by-threshold",
    "gan": "Robustness: GAN",
    "rank_aggregation_robust": "Robustness Aggregation",
    "rank_aggregation_final": "Final Aggregation",
}


def compose_global_narrative(stage_texts: Dict[str, str],
                             global_ir: Optional[Dict[str, Any]] = None,
                             dataset: str = "", entity: str = "",
                             iteration: int = 0) -> str:
    """
    Merge the per-stage narratives into one document, deterministically.

    Pure and LLM-free: the decision block is taken verbatim from the global IR's
    own atom sentences and each stage contributes the prose already written and
    verified for it. Nothing is paraphrased, so the result cannot introduce a
    claim that was not already checked.
    """
    head = f"RAMSeS model selection — {dataset} / entity {entity} (iteration {iteration})"
    lines: List[str] = [head, "=" * len(head)]

    evidence = {a.get("id"): a for a in (global_ir or {}).get("evidence", [])}
    decision = evidence.get("global.decision")
    if decision and decision.get("text"):
        lines += ["", "DECISION", "-" * len("DECISION"), decision["text"]]

    agreement = [a["text"] for _, a in sorted(evidence.items())
                 if a.get("type") == "stage_agreement" and a.get("text")]
    if agreement:
        lines += ["", "Stage agreement"] + [f"  - {t}" for t in agreement]

    ordered = [s for s in _GLOBAL_STAGE_ORDER if stage_texts.get(s)]
    ordered += [s for s in sorted(stage_texts)
                if s not in _GLOBAL_STAGE_ORDER and stage_texts.get(s)]
    for stage in ordered:
        title = _GLOBAL_STAGE_TITLES.get(stage, stage.replace("_", " ").capitalize())
        lines += ["", title, "-" * len(title)]
        lines.append(stage_texts[stage].strip())

    # Name the stages the run could not narrate, so a short document is never
    # mistaken for a complete one.
    statuses = (global_ir or {}).get("stages", {}) or {}
    absent = sorted(s for s in statuses if s not in ordered)
    if absent:
        lines += ["", "Stages without a narrative: " + ", ".join(absent) + "."]

    return "\n".join(lines).rstrip() + "\n"


# ── Verifier-guided repair ───────────────────────────────────────────────────

# # Unused with repair off.
# def _violation_count(metrics: Dict[str, Any]) -> int:
#     return (len(metrics.get("unsupported_numbers", []))
#             + len(metrics.get("unsupported_entities", []))
#             + len(metrics.get("misattributed_numbers", []))
#             # False statements, not style notes: left out of this count they
#             # were measured and then ignored, and repair never ran.
#             + len(metrics.get("swapped_rivals", []))
#             + len(metrics.get("attribution_warnings", []))
#             + len(metrics.get("missing_required_ids", [])))


_PROFILE_WORD = {"H": "high", "L": "low"}


# def _violation_lines(metrics: Dict[str, Any], ir_doc: Dict[str, Any]) -> List[str]:
#     """Human-readable repair feedback for every hard violation the verifier
#     found, each naming the exact fact to go back to."""
#     lines: List[str] = []
#     for tok in metrics.get("unsupported_numbers", []):
#         lines.append(f"The number '{tok}' does not appear in the facts. Remove "
#                      f"it or use the exact value written in the facts. If it "
#                      f"came from splitting a detector name (e.g. 'LOFs 2 and "
#                      f"3'), write each full name instead.")
#     for tok in metrics.get("unsupported_entities", []):
#         lines.append(f"The name '{tok}' does not appear in the facts — remove it.")
#     for m in metrics.get("misattributed_numbers", []):
#         subjects = ", ".join(m.get("subjects", [])) or "the detectors it names"
#         lines.append(f"The number '{m.get('number')}' is used in a sentence "
#                      f"about {subjects}, but it does not belong to any of "
#                      f"them. Re-check the facts and attach it to the right "
#                      f"detector.")
#     atoms_by_id = {a.get("id"): a for a in ir_doc.get("evidence", [])}
#     for swap in metrics.get("swapped_rivals", []):
#         atom = atoms_by_id.get(swap.get("atom_id"))
#         expected = ", ".join(n.upper() for n in swap.get("expected", []))
#         wrong = ", ".join(n.upper() for n in swap.get("intruded", []))
#         detail = (f" You named {wrong}, which this fact does not mention."
#                   if wrong else "")
#         lines.append(f"This sentence names the wrong models: "
#                      f"\"{swap.get('sentence', '')}\"{detail} The fact it comes "
#                      f"from is about exactly {expected} — "
#                      f"\"{(atom or {}).get('text', '')}\". Use those names and "
#                      f"no others, and do not take model names from any other fact.")
#     for warn in metrics.get("attribution_warnings", []):
#         aspect = warn.get("aspect", "")
#         actual = _PROFILE_WORD.get(warn.get("actual", ""), warn.get("actual", ""))
#         claimed = ", ".join(_PROFILE_WORD.get(c, c) for c in warn.get("claimed", []))
#         if warn.get("contradictory"):
#             lines.append(f"This sentence calls {str(warn.get('subject', '')).upper()}'s "
#                          f"{aspect} both {claimed}: "
#                          f"\"{warn.get('sentence', '')}\". Its {aspect} is "
#                          f"{actual} — say that once and drop the other claim. "
#                          f"Do not add a reason for the outcome.")
#             continue
#         lines.append(f"{str(warn.get('subject', '')).upper()} is described with "
#                      f"{claimed} {aspect} in this sentence: "
#                      f"\"{warn.get('sentence', '')}\" — but the facts say its "
#                      f"{aspect} is {actual}. Restate it with the wording the "
#                      f"facts use, and do not group it with detectors that have a "
#                      f"different profile.")
#     for rid in metrics.get("missing_required_ids", []):
#         atom = atoms_by_id.get(rid)
#         if atom is not None:
#             lines.append(f"This required fact was not conveyed — every model "
#                          f"name in it must appear in your paragraph: "
#                          f"\"{atom.get('text', '')}\"")
#     return lines


# def _repair_prompt(base_prompt: str, draft: str, problems: List[str]) -> str:
#     # Repair is where invention spikes: told a statement is wrong, the model
#     # writes a justifying cause the facts never gave, carrying no number and no
#     # new name for a check to catch. The constraint is restated here.
#     return (base_prompt
#             + "\n\nYOUR PREVIOUS DRAFT:\n" + draft
#             + "\n\nPROBLEMS DETECTED IN THE DRAFT — fix ALL of them:\n"
#             + "\n".join(f"- {p}" for p in problems)
#             + "\n\nRewrite the paragraph, fixing every problem above while "
#               "still following all the rules and the original task. Correct "
#               "the wording only: do NOT add a reason, cause or justification "
#               "for anything, and do not explain why a result came out the way "
#               "it did — the facts say what happened, not why. Keep it to ONE "
#               "paragraph.")


# ── Entity-level orchestration ───────────────────────────────────────────────

def _stage_file_map(iteration: int) -> Dict[str, str]:
    return {
        "thompson_sampling": "ir_thompson",
        "thompson_ranking": "ir_thompson_ranking",
        "ga_selection": "ir_ga_selection",
        "ga_combination": "ir_ga_combination",
        "rank_aggregation_robust": f"ir_rank_aggregation_robust_{iteration}",
        "rank_aggregation_final": f"ir_rank_aggregation_final_{iteration}",
        "gan": "ir_gan",
        "monte_carlo": "ir_monte_carlo",
        "off_by_threshold": "ir_off_by",
    }


def narrate_entity(dataset: str, entity: str, iteration: int, client: LLMClient,
                   base_dir: str = "myresults/explanations_ir",
                   out_dir: str = "myresults/explanations_nl",
                   stages: Optional[List[str]] = None,
                   global_mode: str = "concat") -> Dict[str, Any]:
    """
    Narrate every available IR file for (dataset, entity, iteration) and score
    each narrative with the atom-matching verifier. Writes:
        {out_dir}/{ds}/{ent}/nl_{stage}.txt          (per stage)
        {out_dir}/{ds}/{ent}/nl_global_iter{n}.txt
        {out_dir}/{ds}/{ent}/faithfulness_iter{n}.json / .txt
    Per-stage failures are recorded, not fatal; missing IR files are 'skipped'.

    global_mode: "concat" (default) merges the per-stage narratives into the
        global document deterministically — no model call, no new claims, so it
        is not verified again and does not enter the micro-average (the stage
        prose it contains is already counted once). "llm" narrates the global
        IR's own atoms instead, the atom-based path, and is verified normally.
    """
    if global_mode not in GLOBAL_MODES:
        raise ValueError(f"global_mode must be one of {GLOBAL_MODES}, got {global_mode!r}")
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
    stage_texts: Dict[str, str] = {}
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

            # Repair is off. It fired on the verifier's coverage metrics, which
            # assume the narrative reproduces the atom sentences: once the model
            # merges and rewords them, omission reads high on correct prose
            # (0.50 on a verified-correct stage) and the repair prompt's
            # "correct the wording only" pushes the output back toward the atoms.
            # problems = _violation_lines(metrics, ir_doc)
            # if problems:
            #     entry["verify_initial"] = metrics
            #     entry["repaired"] = True
            #     repaired = client.chat(
            #         SYSTEM_PROMPT,
            #         _repair_prompt(base_prompt, narrative, problems)).strip()
            #     repaired_metrics = verify_fn(repaired, ir_doc)
            #     if _violation_count(repaired_metrics) <= _violation_count(metrics):
            #         narrative, metrics = repaired, repaired_metrics
            #     else:
            #         entry["repair_discarded"] = True

            path = os.path.join(nl_dir, f"{nl_name}.txt")
            with open(path, "w") as f:
                f.write(narrative + "\n")
            entry.update({"narrative_path": path,
                          "words": len(narrative.split()), "verify": metrics})
            report["stages"][stage_key] = entry
            # Kept for the deterministic global merge, which reuses the prose
            # exactly as written here rather than re-narrating it.
            if not is_global:
                stage_texts[stage_key] = narrative
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
        if global_mode == "llm":
            if global_doc is None:
                report["stages"]["global"] = {
                    "status": "skipped",
                    "reason": f"ir_global_iter{iteration}.json not found"}
            else:
                _run_one("global", global_doc, f"nl_global_iter{iteration}",
                         is_global=True)
        elif not stage_texts:
            report["stages"]["global"] = {
                "status": "skipped", "mode": "concat",
                "reason": "no stage narratives to merge"}
        else:
            try:
                merged = compose_global_narrative(
                    stage_texts, global_doc, dataset=str(dataset),
                    entity=str(entity), iteration=int(iteration))
                path = os.path.join(nl_dir, f"nl_global_iter{iteration}.txt")
                with open(path, "w") as f:
                    f.write(merged)
                report["stages"]["global"] = {
                    "status": "ok", "mode": "concat",
                    "narrative_path": path, "words": len(merged.split()),
                    # No `verify`: the merge is deterministic and reuses prose
                    # already scored per stage, so re-scoring it here would
                    # double-count those claims in the micro-average.
                    "merged_stages": sorted(stage_texts),
                }
            except Exception as e:  # non-fatal, same as a stage failure
                report["stages"]["global"] = {"status": "error", "mode": "concat",
                                              "error": str(e)}

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
