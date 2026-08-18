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

The anti-hallucination contract lives in SYSTEM_PROMPT: the model may only
restate the numbered fact sentences, must copy numbers and names verbatim,
must convey every [REQUIRED] fact, and must respect the [CAVEAT] lines
without restating them — the card renders those verbatim from the IR in a
section of their own. The verifier then measures how well the output honoured
that contract (hallucination + omission rates).
"""

from __future__ import annotations

import glob
import json
import os
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
                 # 600s, not 120: a 14B model narrating the Thompson stage
                 # exceeds two minutes. ir_thompson.json is 35.9 KB and 19 atoms
                 # on skab/1 — nearly double the next-largest IR and eight times
                 # the smallest — so it is the one stage that reliably times out
                 # while the other ten finish in 14-46s. The failure is quiet:
                 # narrate_entity records a per-stage error and carries on, the
                 # run still reports "LLM narratives written" with a clean
                 # faithfulness score (a stage with no prose cannot hallucinate),
                 # and the staleness warning on the result page is the only sign.
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
    "detection but not this framework's internals.\n"
    "Rules — follow every one strictly:\n"
    "1. Use ONLY the fact sentences given to you. Do not add facts, "
    "numbers, names, comparisons, or causes of your own.\n"
    "2. Copy every number and every model/detector name EXACTLY as written in "
    "the facts. Never re-round, convert, or estimate. A qualifier such as "
    "'(rank 2)' or '(negative influence)' belongs ONLY to the value it "
    "accompanies in the facts — never re-attach it to a different value.\n"
    "3. Every fact marked [REQUIRED] must be conveyed. Unmarked facts may be "
    "omitted if space demands.\n"
    "4. Lines marked [CAVEAT] are limits on what the facts mean. Respect them "
    "— never write a claim one of them rules out — but do NOT restate them: "
    "they are shown to the reader separately, and a second, looser copy in "
    "your paragraph is the same limitation said twice.\n"
    "5. If a value reads 'not_available', either omit it or say the data is "
    "not available — never fill it in.\n"
    "6. Write ONE coherent paragraph of plain prose. No headings, lists, "
    "tables, or markdown."
)
# Four further rules were dropped when the narrator moved from qwen2.5:7b to
# 14b. Measured on SKAB/7 across five stages, removing them changed no metric:
#   - name compression ('CBLOF_1 to -4') was a 7b artifact; 14b writes the
#     names out unprompted.
#   - 'never cite fact 2' patched a hole that no longer exists: the facts are
#     bulleted rather than numbered, so there is nothing to cite.
#   - 'no invented conclusions' addressed padding forced by a word-budget floor
#     that demanded more words than the facts contained; the budget now scales
#     with content length.
#   - 'name every detector in a list' is enforced by the verifier's conjunctive
#     coverage check and repaired by the repair loop.
# Rules 1-5 are the contract the verifier measures and are not a model-size
# question; rule 6 is format. If the narrator is ever downgraded, restore the
# four from git history before trusting the output.


def _render_value(v: Any) -> str:
    if isinstance(v, list):
        return ", ".join(_render_value(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)


def _output_lines(output: Dict[str, Any]) -> List[str]:
    return [f"- {k}: {_render_value(v)}" for k, v in sorted(output.items())]


def _content_words(ir_doc: Dict[str, Any]) -> int:
    """How many words of material the narrative actually has to convey.

    Evidence only. Caveats are shown to the reader from the IR rather than
    narrated, so counting them would budget words for prose that must not be
    written — and a floor set above what there is to say is what forces
    padding.
    """
    return sum(len(str(a.get("text", "")).split())
               for a in ir_doc.get("evidence", []))


# The floor is deliberately BELOW the content length: the narrative restates the
# facts in connected prose, which compresses (shared subjects, pronouns) at least
# as much as connectives add.
_BUDGET_FLOOR_RATIO = 0.9
_BUDGET_CEILING_RATIO = 2.2
_BUDGET_MIN_FLOOR = 40
# The default ceiling, and now also what the floor is clamped against so the two
# can never cross. Was an inline literal in two places.
_BUDGET_HARD_CAP = 400


def _word_budget(n_atoms: int, lo: int = 120, hi: int = 220,
                 content_words: Optional[int] = None) -> tuple:
    """Word budget for the WHOLE narrative, scaled to how much there is to say.

    Driven by the atoms' CONTENT LENGTH rather than their count. Counting atoms
    is a poor proxy: consolidating several near-identical atoms into one (which
    is what stops a narrator shuffling names between them) cuts the count
    without cutting the material, and the old count-based floor then demanded
    more words than the facts contained. A 4-atom ga_selection carrying 74 words
    of facts was asked for at least 120 — so ~46 words had to be invented, and
    they arrived as an unsupported concluding sentence and "(fact 2)" citations
    of the prompt's own numbering.

    Falls back to the count-based curve when the caller has no document.
    """
    if content_words:
        floor = max(_BUDGET_MIN_FLOOR, int(content_words * _BUDGET_FLOOR_RATIO))
        ceiling = max(floor + 40, int(content_words * _BUDGET_CEILING_RATIO))
        # The cap is applied to BOTH ends, not to the ceiling alone. Capping only
        # the ceiling inverted the range once a stage carried more than ~445
        # words of facts: the GAN stage, at 711, was told to write "between 639
        # and 400 words". A model handed a contradictory range compresses, and
        # what it drops is the numbers — 4 of 29 rule thresholds survived, the
        # rest becoming "under specific conditions related to ...".
        #
        # Clamping the floor keeps the global ceiling meaningful. A stage that
        # genuinely needs to exceed it says so in _STAGE_WORD_BUDGETS below,
        # which is the mechanism for exactly that and guards its own range.
        if floor > _BUDGET_HARD_CAP - 40:
            floor = _BUDGET_HARD_CAP - 40
        return floor, min(_BUDGET_HARD_CAP, max(ceiling, floor + 40))
    if n_atoms <= 3:
        return 65, 120
    return lo, min(_BUDGET_HARD_CAP, hi + 8 * max(0, n_atoms - 12))


# Stages where the narrative must carry one statement per atom rather than a
# summary, so the budget has to scale past the default 400-word ceiling.
# (words_per_atom, base, ceiling), keyed by exact stage name.
_STAGE_WORD_BUDGETS: Dict[str, tuple] = {
    # Thompson narrates every regime individually; ~20 words each plus the
    # lead, regime summary, winner channel and state line.
    "thompson_sampling": (20, 40, 700),
    # The ranking sibling narrates regimes too, but plain run-length encoding of
    # the ||mu||^2 leader yields a handful of them rather than a dozen, so it
    # needs a lower ceiling than the stage above.
    "thompson_ranking": (20, 40, 500),
}
# NEITHER point-injection stage gets an entry, and that is deliberate.
#
# GAN had one — (30, 40, 900), sized so its rules would fit — and it backfired:
# asked for 640-900 words it wrote 325, dropped 36% of its required atoms
# INCLUDING the winner, opened on the feature importances and broke into five
# paragraphs. Off-by, carrying MORE material (827 content words against GAN's
# 771) but left on the generic budget, wrote 818 words in one paragraph with
# zero omissions. A floor far above what the model wants to write is not a
# nudge to write more; it makes it reorganise.
#
# The inverted range that first motivated the entry is fixed at source in
# `_word_budget` above, and that fix — not the entry — is what recovered the
# thresholds: on SKAB/3, the entity where they were first lost, the same IR goes
# from 4 of 29 reproduced under the inverted range to 29 of 29 on the generic
# budget with no entry at all.
#
# Both stages now run the generic path, and their prompts were diffed to confirm
# nothing else drifted: identical but for the stage name, the question's wording,
# and one sentence of task hint (gan's spells out that thresholds must be copied
# with their numbers, which off-by does not need). The IRs match on envelope
# keys, atom-type set and order bands, and their edge atoms are the same size —
# 53 words and 3.7 thresholds each on SKAB/4 against off-by's 52 and 3.4.
#
# Note what does NOT bind: the 400 ceiling. Both stages routinely write past it
# — 641 and 818 words on SKAB/4 — with zero omissions, across repeat runs. The
# ceiling is advisory; it is a FLOOR above what the model wants to write that
# breaks it.


def _stage_word_budget(stage: Any, n_atoms: int) -> Optional[tuple]:
    cfg = _STAGE_WORD_BUDGETS.get(str(stage))
    if not cfg:
        return None
    per, base, cap = cfg
    lo = min(cap - 60, base + per * max(0, n_atoms))
    return lo, min(cap, int(lo * 1.5))


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
    # Bulleted, never numbered. Nothing references the numbers — they were pure
    # decoration — but they handed the narrator a citation handle, and it used
    # it: "These detectors were selected for their high utility (fact 2). Fact 3
    # reveals that…". With no numbers there is nothing to cite.
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


# Stage-specific rendering guidance appended to the prompt's TASK. This is a
# NARRATION concern (how to render), so it lives here in the narrator, not in
# the grounded IR — and it is the same for every run of a stage, so it is not a
# per-IR field. Keyed by exact stage name; add other stages here as their
# narratives need shaping.
_STAGE_TASK_HINTS: Dict[str, str] = {
    # The opening sentence is load-bearing: without it the narrator went
    # straight into the per-source walk and dropped both the consensus winner
    # and the source list (omission 0.000 -> 0.250). One positive instruction
    # replaced three defensive ones and scored better.
    "rank_aggregation_robust": (
        " Open by naming the consensus's own top-ranked detector and the source "
        "rankings being aggregated. Then describe each source in the order "
        "given; for each one, state its overall standing rank, its influence "
        "rank, its agreement rank, and its pattern. A rank is a position — rank "
        "1 is best — so give the rank number itself rather than calling it high "
        "or low."
    ),
    "ga_combination": (
        " Describe each detector in the order given; for each one, state its "
        "overall weight rank and its rank on absolute SHAP, PFI and total ALE "
        "(rank 1 is strongest). Where a fact says a rank is a tie, say it is "
        "tied. Finish with the sign summary, saying which detectors push "
        "the meta-learner toward flagging an anomaly and which push the other "
        "way. Report each sign exactly as the facts give it; how well a sign "
        "is supported is a caveat, so leave it out of the paragraph. A detector "
        "the facts give no sign at all keeps none — never assign it one of "
        "your own."
    ),
    "ga_selection": (
        " Open by naming the chosen ensemble. Then explain why the chosen "
        "detectors were kept, following the facts in order and keeping the "
        "detectors grouped exactly as the facts group them. Then explain why "
        "the rest were left out, using the high/low utility and stability "
        "wording the facts use."
    ),
    # Direction is the whole risk here. A detector's channel shares are sums of
    # squares, so every one is positive and none can "drag the score down" —
    # but that is exactly the sentence a narrator reaches for when a share is
    # small, and the verifier cannot see it: the number and the channel name are
    # both correct. Only the comparison against the named rival has a sign, so
    # the hint puts direction language where it belongs and nowhere else.
    "thompson_ranking": (
        " Open with the winner and its score, then the channels its score is "
        "built from. Describe those channels only as larger or smaller shares "
        "of that detector's own score — a small share means a channel "
        "contributed little, never that it lowered the score or worked against "
        "the detector. Only when comparing the winner with the named runner-up "
        "may you say a channel favoured one over the other, and there keep the "
        "direction exactly as the fact states it. Then give the selection "
        "counts, then how leadership divided into regimes, then EVERY regime "
        "its own sentence in the order listed, each naming its window range, "
        "its leader and its channels. "
        # The regime NUMBER is what pairs each sentence with its own figure in
        # the page's regime disclosure — an ordinal cannot do it, and there is
        # no error when one is used: the disclosure quietly falls back to the
        # IR's own wording, so the narrative simply stops being what the reader
        # sees. Left to itself the narrator wrote "In the first regime (windows
        # 10 to 12)" for all seven.
        "Begin each of those sentences with the literal words 'Regime N "
        "(windows ...)', using the number the fact gives — never 'the first "
        "regime', 'the second regime', or any other ordinal in place of it. "
        "Name that leader outright — never "
        "describe it by reference to the previous regime. Do not add a sentence "
        "interpreting what any of this implies."
    ),
    # The regime-shape instruction is the one clause here that earns its length:
    # a narrator that describes a regime by reference to the previous one writes
    # false continuity ("NN_3 continued as leader" when the previous regime was
    # led by NN_2), and no metric can see it — the names and numbers are all
    # correct. A positive template held where a shorter ban leaked.
    "thompson_sampling": (
        " Open with the winner and its margin, then how the run divided into "
        "regimes. Then give EVERY regime its own sentence, in the order listed, "
        "keeping each regime's window range, its leader and its channels "
        "together. Begin each of those sentences with the literal words "
        "'Regime N (windows ...)', then the detector that led it, then its "
        "channels. Name that detector outright — never describe it by reference "
        "to the previous regime. Three different things are said about channels "
        "and they must not be merged or traded for one another: one channel "
        "SUPPLIES a share of a detector's expected reward, one GIVES IT AN EDGE "
        "over the named rival, and one DEPARTS FURTHEST FROM ITS USUAL "
        "contribution. The last is a separate sentence in the facts and must "
        "stay separate clauses. Keep whichever wording the fact uses. "
        # Every claim about a regime now arrives in ONE fact sentence carrying
        # that regime's number, so keeping it in one sentence is what keeps the
        # page able to file it. Splitting off a trailing clause strands it: it
        # carries no regime number, matches no atom, and lands in a heap of
        # context-free sentences at the end of the summary.
        "Keep each regime to a SINGLE sentence — never split a trailing clause "
        "off into a sentence of its own, and never refer back with 'in this "
        "regime' or 'here'. "
        "Finish with the winner's overall channel and the selection-state "
        "percentages."
    ),
    "monte_carlo": (
        " Open with one sentence restating the production-test result exactly "
        "as the fact gives it — use the word 'first' — naming the top detector "
        "for each metric. The F1 and PR-AUC leaders are not always the same "
        "detector: if the fact names two different ones, keep them separate. "
        "Then give each detector's winning noise ranges in the order listed, "
        "one detector per statement. Finish with the win percentages. Copy each "
        "noise range as it is written ('from 0.000 to 0.042') — never turn a "
        "range into a hyphenated pair."
    ),
    "off_by_threshold": (
        " Open with one short sentence naming the highest-ranked model, then "
        "give each fact about the models it beat as its OWN separate sentence, "
        "then the importance figures. The rival models named in a sentence must "
        "be EXACTLY the models that fact lists. State each condition exactly as "
        "it is worded in the facts. If a fact says the highest-ranked model "
        "never exclusively beat some models, state that too."
    ),
    # The same shape as off_by_threshold: both stages narrate a winner's
    # exclusive wins over injected points, and the degenerate clause is
    # load-bearing here for the same reason (see the note below).
    #
    # The threshold sentence is NOT boilerplate. Squeezed by an inverted word
    # budget, this stage wrote "under specific conditions related to generated
    # point magnitude, gap from the surrounding series, and spread across
    # channels" — a list of property names with every number dropped, which is
    # unfalsifiable and tells a reader nothing about where the edge was. The
    # budget is fixed above; this forbids the phrasing directly, because the
    # verifier scores an atom as covered from its subject and win count alone
    # and cannot see a missing threshold.
    "gan": (
        " Open with one short sentence naming the highest-ranked model, then "
        "give each fact about the models it beat as its OWN separate sentence, "
        "then the importance figures. The rival models named in a sentence must "
        "be EXACTLY the models that fact lists. State every condition WITH ITS "
        "NUMBERS, copied exactly as the fact writes them: never replace a "
        "threshold with 'specific ranges', 'certain conditions', 'specific "
        "values' or a bare list of property names. A sentence that names a "
        "property without its number is wrong. If a fact says the "
        "highest-ranked model never exclusively beat some models, state that too."
    ),
}
# The hyphenated-range ban in monte_carlo and the degenerate clause in
# off_by_threshold are NOT model-capability patches and must survive any future
# trim. The first exists because the verifier's number extraction is sign-aware,
# so "0.000-0.042" reads as the negative number -0.042 and is flagged
# unsupported. The second was measured load-bearing: dropping it lost
# ob.degenerate (omission 0.000 -> 0.200), and that atom is the negation that
# makes a swapped rival set self-contradictory.


def _stage_task_hint(stage: Any) -> str:
    return _STAGE_TASK_HINTS.get(str(stage), "")


def build_stage_prompt(ir_doc: Dict[str, Any]) -> str:
    n_atoms = len(ir_doc.get("evidence", []))
    lo, hi = (_stage_word_budget(ir_doc.get("stage", ""), n_atoms)
              or _word_budget(n_atoms, content_words=_content_words(ir_doc)))
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
    task += _stage_task_hint(ir_doc.get("stage", ""))
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


# ── Global narrative: deterministic merge ────────────────────────────────────
#
# Two interchangeable ways to produce the global document, selected by
# `global_mode` on narrate_entity:
#   "concat" (default) — stitch the already-narrated per-stage prose together.
#                        Adds no new claims, so it inherits the per-stage
#                        faithfulness and is not scored again.
#   "llm"              — narrate the global IR's own atoms (build_global_prompt
#                        / narrate_global / verify_global), the original path.
# Both are kept working; switching back is a one-argument change.
GLOBAL_MODES = ("concat", "llm")

# The merged document follows the pipeline's order so it reads as the run ran,
# rather than the alphabetical order the IR files happen to load in.
_GLOBAL_STAGE_ORDER = (
    "ga_selection", "ga_combination",
    # The ranking criterion first: it explains the ordering the pipeline goes on
    # to consume, and the selection dynamics then account for how the run got
    # there. WebUI.artifacts.STAGES must stay in this order.
    "thompson_ranking", "thompson_sampling",
    # GAN leads the robustness block: it is sub-stage 6.3, ahead of off-by at
    # 6.4 and Monte Carlo at 6.5. The existing monte_carlo/off_by_threshold
    # order is left as it stands rather than reshuffled here.
    # NOTE: no parentheses in this comment — WebUI.test_webui parses this tuple
    # with a non-greedy regex that would stop at the first closing bracket.
    "gan", "monte_carlo", "off_by_threshold",
    "rank_aggregation_robust", "rank_aggregation_final",
)

_GLOBAL_STAGE_TITLES = {
    "ga_selection": "Ensemble selection (genetic algorithm)",
    "ga_combination": "Ensemble weighting (meta-learner)",
    # Two stages explain one algorithm, so neither may claim the plain name:
    # these titles say which question each answers. Duplicated verbatim in
    # WebUI.artifacts.STAGES.
    "thompson_ranking": "Thompson Sampling: ranking criterion",
    "thompson_sampling": "Thompson Sampling: selection dynamics",
    "gan": "Robustness: GAN perturbations",
    "monte_carlo": "Robustness: Monte Carlo noise sweep",
    "off_by_threshold": "Sensitivity: off-by-threshold test",
    "rank_aggregation_robust": "Robustness consensus",
    "rank_aggregation_final": "Final consensus",
}


def compose_global_narrative(stage_texts: Dict[str, str],
                             global_ir: Optional[Dict[str, Any]] = None,
                             dataset: str = "", entity: str = "",
                             iteration: int = 0,
                             stage_footers: Optional[Dict[str, str]] = None) -> str:
    """
    Merge the per-stage narratives into one document, deterministically.

    Pure and LLM-free: the decision block is taken verbatim from the global IR's
    own atom sentences and each stage contributes the prose already written and
    verified for it. Nothing is paraphrased, so the result cannot introduce a
    claim that was not already checked.

    `stage_footers` is optional; when given, each stage's glossary is appended
    under its section (the per-stage .txt files always carry their own).
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
        # Glossary first, matching the per-stage files: the reader meets the
        # terms before the prose that uses them.
        footer = (stage_footers or {}).get(stage)
        if footer:
            lines += [f"INFO: {footer}", ""]
        lines.append(stage_texts[stage].strip())

    # Name the stages the run could not narrate, so a short document is never
    # mistaken for a complete one.
    statuses = (global_ir or {}).get("stages", {}) or {}
    absent = sorted(s for s in statuses if s not in ordered)
    if absent:
        lines += ["", "Stages without a narrative: " + ", ".join(absent) + "."]

    return "\n".join(lines).rstrip() + "\n"


# ── Verifier-guided repair ───────────────────────────────────────────────────

def _violation_count(metrics: Dict[str, Any]) -> int:
    return (len(metrics.get("unsupported_numbers", []))
            + len(metrics.get("unsupported_entities", []))
            + len(metrics.get("misattributed_numbers", []))
            # A swapped rival set and a wrong utility/stability profile are both
            # false statements, not style notes. Left out of this count they were
            # measured and then ignored: repair never ran, so the one place the
            # model is told what it specifically got wrong stayed silent.
            + len(metrics.get("swapped_rivals", []))
            + len(metrics.get("attribution_warnings", []))
            + len(metrics.get("missing_required_ids", [])))


_PROFILE_WORD = {"H": "high", "L": "low"}


def _violation_lines(metrics: Dict[str, Any], ir_doc: Dict[str, Any]) -> List[str]:
    """Human-readable repair feedback for every hard violation the verifier
    found, each naming the exact fact to go back to."""
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
    for swap in metrics.get("swapped_rivals", []):
        atom = atoms_by_id.get(swap.get("atom_id"))
        expected = ", ".join(n.upper() for n in swap.get("expected", []))
        wrong = ", ".join(n.upper() for n in swap.get("intruded", []))
        detail = (f" You named {wrong}, which this fact does not mention."
                  if wrong else "")
        lines.append(f"This sentence names the wrong models: "
                     f"\"{swap.get('sentence', '')}\"{detail} The fact it comes "
                     f"from is about exactly {expected} — "
                     f"\"{(atom or {}).get('text', '')}\". Use those names and "
                     f"no others, and do not take model names from any other fact.")
    for warn in metrics.get("attribution_warnings", []):
        aspect = warn.get("aspect", "")
        actual = _PROFILE_WORD.get(warn.get("actual", ""), warn.get("actual", ""))
        claimed = ", ".join(_PROFILE_WORD.get(c, c) for c in warn.get("claimed", []))
        if warn.get("contradictory"):
            lines.append(f"This sentence calls {str(warn.get('subject', '')).upper()}'s "
                         f"{aspect} both {claimed}: "
                         f"\"{warn.get('sentence', '')}\". Its {aspect} is "
                         f"{actual} — say that once and drop the other claim. "
                         f"Do not add a reason for the outcome.")
            continue
        lines.append(f"{str(warn.get('subject', '')).upper()} is described with "
                     f"{claimed} {aspect} in this sentence: "
                     f"\"{warn.get('sentence', '')}\" — but the facts say its "
                     f"{aspect} is {actual}. Restate it with the wording the "
                     f"facts use, and do not group it with detectors that have a "
                     f"different profile.")
    for rid in metrics.get("missing_required_ids", []):
        atom = atoms_by_id.get(rid)
        if atom is not None:
            lines.append(f"This required fact was not conveyed — every model "
                         f"name in it must appear in your paragraph: "
                         f"\"{atom.get('text', '')}\"")
    return lines


def _repair_prompt(base_prompt: str, draft: str, problems: List[str]) -> str:
    # Repair is where invention spikes: told a statement is wrong, the model
    # reaches for justifying language and writes a cause the facts never gave
    # ("left out due to its lower utility compared to other factors"). Such a
    # sentence carries no number and no new name, so no mechanical check can
    # see it — the constraint has to be restated at the point of failure.
    return (base_prompt
            + "\n\nYOUR PREVIOUS DRAFT:\n" + draft
            + "\n\nPROBLEMS DETECTED IN THE DRAFT — fix ALL of them:\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\n\nRewrite the paragraph, fixing every problem above while "
              "still following all the rules and the original task. Correct "
              "the wording only: do NOT add a reason, cause or justification "
              "for anything, and do not explain why a result came out the way "
              "it did — the facts say what happened, not why. Keep it to ONE "
              "paragraph.")


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
    stage_footers: Dict[str, str] = {}
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

            # The fixed glossary is written verbatim and OUTSIDE the model's
            # output, so its definitions are never reworded and never counted
            # as claims by the verifier. It leads the file: the terms it
            # defines are the ones the narrative is about, so a reader meets
            # them before the prose that uses them.
            footer = ir_doc.get("info_footer")
            path = os.path.join(nl_dir, f"{nl_name}.txt")
            with open(path, "w") as f:
                if footer:
                    f.write(f"INFO: {footer}\n\n")
                f.write(narrative + "\n")
            entry.update({"narrative_path": path,
                          "words": len(narrative.split()), "verify": metrics})
            report["stages"][stage_key] = entry
            # Kept for the deterministic global merge, which reuses the prose
            # exactly as written here rather than re-narrating it.
            if not is_global:
                stage_texts[stage_key] = narrative
                if footer:
                    stage_footers[stage_key] = footer
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
                    entity=str(entity), iteration=int(iteration),
                    stage_footers=stage_footers)
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
