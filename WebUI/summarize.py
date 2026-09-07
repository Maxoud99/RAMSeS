"""
The summarisation seam.

Each stage card shows a summary by default with a disclosure to the full
narrative. Two kinds of summary exist, chosen per stage by `_STAGE_SUMMARY`:

  * **drop** — the narrative minus the sentences carrying a given class of
    fact, so the default view answers the stage's question and the detail moves
    behind the click. Nothing is paraphrased: every sentence shown is a
    sentence the model already wrote and the verifier already scored.
  * **table** — a deterministic table built from the IR, for stages whose
    answer is a ranking rather than a story. Built from the IR's own fields,
    never by parsing the rendered `*_explainability_*.txt` reports: those are a
    display format, and re-parsing one would be a lossy round-trip that no test
    would catch when the layout changed.

Stages absent from `_STAGE_SUMMARY` keep the whole narrative.

The contract that keeps this swappable:

* `artifacts.build_payload` is the **only** caller.
* The API always returns both `summary` and `full`, plus `summary_is_full`. The
  frontend keys its disclosure off that flag and renders the same DOM either
  way.
* `summarize` receives the narrative and the IR, never the INFO glossary — the
  glossary is fixed boilerplate, identical across runs.
* A summariser that fails must never break the page: `summarize` catches its
  own errors and falls back to the full text.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Detector- and source-shaped tokens: LOF_1, CBLOF_4, GAN_PR_AUC, MonteCarlo_F1.
# Broader than the verifier's pattern, which requires a numeric suffix and so
# would miss every rank-aggregation source name.
_NAME_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_NUM_RE = re.compile(r"(?<![\w.\-])[-+]?\d+(?:\.\d+)?%?(?![\w\-])(?!\.\d)")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# "Regime 7 (windows 110 to 121) …" — regimes attribute on an exact anchor
# rather than by the generic name/number overlap that every regime sentence
# would satisfy equally. The index is one anchor; a regime's window range is the
# other, and accepting both is what frees the narrator from opening all of them
# the same way.
_REGIME_RE = re.compile(r"\bregime\s+(\d+)\b", re.IGNORECASE)
# How much of a non-regime atom a sentence must account for to break out of the
# regime it follows. Below this it is read as that regime's second sentence.
_CARRY_MIN_COVERAGE = 0.5

# Per stage: which atom types leave the default view, which table to build, or
# both. Caveats are NOT listed here — `strip_caveats` removes them from every
# stage and every view of the prose, because the card renders them verbatim
# from the IR in a section of their own.
_STAGE_SUMMARY: Dict[str, Dict[str, Any]] = {
    "ga_selection": {"mode": "drop", "drop": ("excluded_detector", "excluded_group")},
    # Cutting at a count of atoms put the per-noise walk in the summary: the
    # narrator merges, so five facts arrived as three sentences and the cut
    # landed two sentences into the walk. Dropping by type does not care.
    # "monte_carlo": {"mode": "lead", "lead_trailing": ("win_region",)},
    "monte_carlo": {"mode": "drop", "drop": ("win_region",)},
    # One sentence per rival, all of the same atom type, so there is nothing to
    # drop BY TYPE that would not take every rival with it — dropping the two
    # importance families (which is what these did) left eleven near-identical
    # rival sentences in the default view. The cut is by position instead: the
    # winner and its three closest competitors open the card, and the rest —
    # the weaker rivals, the per-rival importances and the roll-up — are what
    # the click buys. The IR orders the rivals hardest-first, so the four kept
    # here are the four that matter.
    "off_by_threshold": {"mode": "lead", "sentences": 4},
    "gan": {"mode": "lead", "sentences": 4},
    # The per-regime walk is the bulk of this narrative, and it already has a
    # disclosure of its own where each regime sits beside its SHAP plot.
    # `extended_in` says the dropped sentences are rendered there, so the card
    # must not also offer a generic "read the full explanation" — that would be
    # a second copy of the same fourteen sentences, without the plots.
    "thompson_sampling": {"mode": "drop", "drop": ("regime",),
                          "extended_in": "regimes"},
    # The narrative opens with the answer — the winner, the context features its score
    # is built from, and which of them decided the margin — so the first three
    # sentences are the summary and everything after is behind the click.
    # `extended_drop` keeps the regime walk out of the full-text view too: those
    # sentences belong beside their own plots in the regime disclosure.
    "thompson_ranking": {"mode": "lead", "sentences": 3,
                         "extended_drop": ("regime",),
                         "table": "ts_ranking"},
    # The table IS the per-detector ranking, so the sentences restating it row
    # by row are the one thing the summary holds back. Everything else — which
    # ensemble was selected, and which way each detector signed — is the answer
    # and stays in view.
    "ga_combination": {"mode": "drop", "drop": ("detector_role",),
                       "table": "ga_combination"},
    # The summary is the sentence answering "which source shaped this most", so
    # it is picked BY TYPE. Taking the narrative's first sentence literally put
    # whatever the narrator opened with on the card, which on nine-source
    # entities is reliably the consensus winner.
    "rank_aggregation_robust": {"mode": "table", "table": "rank_aggregation",
                                "lead_types": ("source_role",)},
    # rank_aggregation_final is deliberately absent: two sources, a couple of
    # sentences, nothing to hold back.
}


def _tokens(text: str) -> Tuple[frozenset, frozenset]:
    names = frozenset(t.lower() for t in _NAME_RE.findall(text or ""))
    numbers = frozenset(m.group(0).rstrip("%") for m in _NUM_RE.finditer(text or ""))
    return names, numbers


def split_sentences(text: str) -> List[str]:
    return [s for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]


def _regime_spans(regimes: Dict[str, dict],
                  atoms: List[dict]) -> Dict[Tuple[str, str], dict]:
    """Window range -> regime, for the ranges that identify one regime alone.

    Two kinds of range are refused. A degenerate one (a one-window regime) is a
    lone integer, which is the collision the index check exists to avoid. And
    one whose endpoints both turn up in some other atom is refused for the same
    reason it looks like an anchor — SKAB/9's ranking summary counts 12 regimes
    from window 10, and a regime there runs 10 to 12. Those regimes keep their
    index anchor; only the extra one is withdrawn.
    """
    elsewhere = [n for a in atoms if a.get("type") != "regime"
                 for n in [_tokens(str(a.get("text", "")))[1]]]
    spans: Dict[Tuple[str, str], Optional[dict]] = {}
    for atom in regimes.values():
        value = atom.get("value") or {}
        start, end = value.get("start"), value.get("end")
        if start is None or end is None or str(start) == str(end):
            continue
        key = (str(start), str(end))
        if any(set(key) <= numbers for numbers in elsewhere):
            spans[key] = None
            continue
        spans[key] = None if key in spans else atom
    return {k: a for k, a in spans.items() if a is not None}


def attribute_sentences(narrative: str,
                        ir_doc: Dict[str, Any]) -> List[Tuple[str, Optional[dict]]]:
    """Pair each narrative sentence with the atom it most likely conveys.

    Scored on shared names and numbers, names weighted higher — a paraphrase
    keeps the proper nouns even when it rewrites everything else. A sentence
    matching nothing gets None and is always kept: the summary drops only on
    positive evidence, so an unrecognised sentence degrades to being shown
    rather than silently lost.

    Regimes are the exception: they are anchored, never scored. A sentence
    anchors on the regime's index or on its window range, and an unanchored
    sentence continues the regime it follows unless it accounts for a
    non-regime atom on its own. Only the FIRST sentence about a regime needs an
    anchor, which is what lets a regime run to two sentences and refer back.
    """
    atoms = list(ir_doc.get("evidence", []) or [])
    scored = [(a, *_tokens(str(a.get("text", "")))) for a in atoms]
    # `ts.regime.3` and its `ts.regime.3.supply` sibling both carry index 3. The
    # canonical one wins, because the disclosure files sentences by that id.
    regimes: Dict[str, dict] = {}
    for atom in atoms:
        if atom.get("type") != "regime":
            continue
        m = _REGIME_RE.search(str(atom.get("id", "")).replace(".", " "))
        if not m:
            continue
        if str(atom.get("id", "")).endswith("." + m.group(1)) or m.group(1) not in regimes:
            regimes[m.group(1)] = atom
    spans = _regime_spans(regimes, atoms)

    out: List[Tuple[str, Optional[dict]]] = []
    current: Optional[dict] = None
    for sentence in split_sentences(narrative):
        s_names, s_numbers = _tokens(sentence)

        hit = _REGIME_RE.search(sentence)
        anchor = regimes.get(hit.group(1)) if hit else None
        if anchor is None:
            matched = [a for key, a in spans.items() if set(key) <= s_numbers]
            anchor = matched[0] if len(matched) == 1 else None
        if anchor is not None:
            current = anchor
            out.append((sentence, anchor))
            continue

        best, best_key = None, (0, 0.0)
        for atom, a_names, a_numbers in scored:
            # A regime is reached by anchor or by carry-forward, never by
            # overlap. Overlap alone cannot tell "Across the regimes NN_1 led,
            # context feature 7 contributed most" from a regime led by NN_1
            # whose context feature 7 mattered — both share one name and one
            # number — and the roll-up sentences lost those ties, disappearing
            # from the summary along with the regimes. A sentence that names a
            # regime the IR has no atom for still falls back to overlap.
            if not hit and atom.get("type") == "regime":
                continue
            shared = 2 * len(s_names & a_names) + len(s_numbers & a_numbers)
            if not shared:
                continue
            # Tie-break on how much of the ATOM the sentence accounts for, so a
            # short atom fully covered beats a long one grazed. Always < 1, so
            # it can only order equal scores.
            size = 2 * len(a_names) + len(a_numbers)
            key = (shared, shared / size if size else 0.0)
            if key > best_key:
                best, best_key = atom, key

        if current is not None and best_key[1] < _CARRY_MIN_COVERAGE:
            out.append((sentence, current))
            continue
        current = None
        out.append((sentence, best))
    return out


# Words too common to identify anything, plus a length floor. Caveats are
# matched lexically rather than by the name/number scorer above: most carry
# neither a detector name nor a number ("Correctness is judged on thresholded
# predictions; PR-AUC has no per-point notion of correct or incorrect"), so
# that scorer cannot see them at all.
_STOPWORDS = frozenset("""
that this these those with from have been were which when what over into
each their there than then they them some such only also both more most
""".split())
_WORD_RE = re.compile(r"[a-z][a-z0-9\-]{3,}")
# A narrator restating a caveat stays close to its wording, so a high bar is
# safe and keeps evidence sentences that merely share vocabulary.
_CAVEAT_OVERLAP = 0.5


def _content_tokens(text: str) -> frozenset:
    return frozenset(w for w in _WORD_RE.findall((text or "").lower())
                     if w not in _STOPWORDS)


def caveat_sentences(narrative: str, ir_doc: Dict[str, Any]) -> List[str]:
    """Narrative sentences that restate one of the IR's caveats."""
    caveats = [_content_tokens(str(c.get("text", "")))
               for c in (ir_doc.get("caveats") or [])]
    caveats = [c for c in caveats if c]
    if not caveats:
        return []
    out = []
    for sentence in split_sentences(narrative):
        tokens = _content_tokens(sentence)
        if not tokens:
            continue
        if any(len(tokens & c) / len(c) >= _CAVEAT_OVERLAP for c in caveats):
            out.append(sentence)
    return out


def strip_caveats(narrative: str, ir_doc: Dict[str, Any]) -> str:
    """The narrative with any sentence that restates an IR caveat removed.

    Applied to EVERY stage and to every view of the prose, because the card
    renders the caveats verbatim from the IR in a section of their own. A
    narrated copy is the same limitation said twice, in looser words, and it
    lands wherever the narrator chose to put it — which on `ga_combination`
    was in front of the findings the caveats qualify.

    A sentence that also conveys an EVIDENCE atom is never stripped, however
    well it matches. The lexical bar is a similarity score, and a short caveat
    made of common vocabulary can reach it on a sentence that is really a
    finding: the robustness consensus's opening line ("The robustness consensus
    ranking, which LOF_1 tops, is derived from aggregating six source
    rankings…") scored exactly 0.50 against "The consensus ranking is produced
    by Markov-chain rank aggregation over the source rankings" on the four
    tokens they share, and vanished from the page. Losing a finding is the
    worse error, so carrying one is a veto.
    """
    skip = set(caveat_sentences(narrative, ir_doc))
    if not skip:
        return (narrative or "").strip()
    carries_fact = {s for s, atom in attribute_sentences(narrative, ir_doc) if atom}
    skip -= carries_fact
    if not skip:
        return (narrative or "").strip()
    kept = [s for s in split_sentences(narrative) if s not in skip]
    # Never return nothing: a narrative that matched end to end is a matcher
    # failure, and an empty card is worse than a redundant one.
    return " ".join(s.strip() for s in kept).strip() or (narrative or "").strip()


def _drop_summary(narrative: str, ir_doc: Dict[str, Any],
                  drop_types: Sequence[str]) -> str:
    drop = set(drop_types)
    pairs = attribute_sentences(narrative, ir_doc)
    # A sentence may convey more than one fact and attribution names only the
    # likeliest, so a merged sentence would take a kept fact down with the
    # dropped one. For each kept atom, the sentence that best accounts for it is
    # protected.
    def _cov(atom_names, atom_numbers, size, toks):
        s_names, s_numbers = toks
        return (2 * len(s_names & atom_names) + len(s_numbers & atom_numbers)) / size

    dropped = [(s, _tokens(s)) for s, atom in pairs
               if atom and atom.get("type") in drop]
    surviving = [_tokens(s) for s, atom in pairs
                 if not (atom and atom.get("type") in drop)]
    protected = set()
    for atom in (ir_doc.get("evidence") or []):
        if atom.get("type") in drop:
            continue
        a_names, a_numbers = _tokens(str(atom.get("text", "")))
        size = 2 * len(a_names) + len(a_numbers)
        if not size:
            continue
        # Only rescue a fact no surviving sentence carries. Rescuing on best
        # coverage alone kept a regime sentence that happened to cover a short
        # atom in full ("Regime 3 ... LOF_2" against "Across the regimes LOF_2
        # led, context feature 3 ...") while the real carrier survived anyway.
        if any(_cov(a_names, a_numbers, size, t) >= _CARRY_MIN_COVERAGE
               for t in surviving):
            continue
        best, best_cov = None, 0.0
        for sentence, toks in dropped:
            cov = _cov(a_names, a_numbers, size, toks)
            if cov > best_cov:
                best, best_cov = sentence, cov
        if best is not None and best_cov >= _CARRY_MIN_COVERAGE:
            protected.add(best)
    kept = [s for s, atom in pairs
            if not (atom and atom.get("type") in drop) or s in protected]
    return " ".join(s.strip() for s in kept).strip()


# ── Tables ───────────────────────────────────────────────────────────────────

def _atoms_of(ir_doc: Dict[str, Any], atom_type: str) -> List[dict]:
    return [a for a in (ir_doc.get("evidence") or [])
            if a.get("type") == atom_type and isinstance(a.get("value"), dict)]


def _rank_key(value: Any) -> Any:
    return float("inf") if value is None else value


def _ga_combination_table(ir_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Overall weight rank, the three method ranks, and the sign.

    The raw Markov score stays out: it is the quantity the rank is derived
    from, its ties are decided at the 16th decimal, and quoting it invites the
    reader to compare values that are not meaningfully different.

    The sign is the sign of ALE's net accumulated effect, not of a signed SHAP
    average, and it is reported for every detector that has one. How well a
    sign is supported has no column: it is a caveat, the card renders the
    caveats in a section of their own, and a column repeating them would be the
    same qualification in two places.
    """
    rows = []
    for atom in (_atoms_of(ir_doc, "detector_role_lead")
                 + _atoms_of(ir_doc, "detector_role")):
        v = atom["value"]
        rank = v.get("final_rank")
        rows.append({
            "_sort": (_rank_key(rank), str(atom.get("subject", ""))),
            "cells": [
                f"{rank} (tie)" if v.get("final_rank_tied") else rank,
                atom.get("subject"),
                v.get("mean_abs_shap_rank"),
                v.get("pfi_rank"),
                v.get("ale_rank"),
                str(v.get("sign") or "").replace("_", " "),
            ],
        })
    if not rows:
        return None
    rows.sort(key=lambda r: r["_sort"])
    return {
        "columns": ["Weight rank", "Detector", "|SHAP| rank", "PFI rank",
                    "ALE rank", "Sign"],
        "align": ["num", "name", "num", "num", "num", "text"],
        "rows": [r["cells"] for r in rows],
    }


def _rank_aggregation_table(ir_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rows = []
    for atom in _atoms_of(ir_doc, "source_role"):
        v = atom["value"]
        rows.append({
            "_sort": (_rank_key(v.get("borda_rank")), str(atom.get("subject", ""))),
            "cells": [
                v.get("borda_rank"),
                atom.get("subject"),
                v.get("influence_rank"),
                v.get("agreement_rank"),
            ],
        })
    if not rows:
        return None
    rows.sort(key=lambda r: r["_sort"])
    return {
        "columns": ["Overall Rank", "Source", "Influence Rank", "Agreement Rank"],
        "align": ["num", "name", "num", "num"],
        "rows": [r["cells"] for r in rows],
    }


def _ts_ranking_table(ir_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The winner's score split by context feature, beside the margin over the runner-up.

    Two columns from two atoms: the share is non-negative and sums to the
    winner's score, the delta is signed and sums to the margin. They are shown
    together because the reader's question — which context features produced this
    ranking — needs both: a context feature can carry most of the winner's score and
    still be one the runner-up won.
    """
    winner = next(iter(_atoms_of(ir_doc, "winner_channels")), None)
    if not winner:
        return None
    per_context_feature = winner["value"].get("per_channel") or []
    if not per_context_feature:
        return None
    total = sum(float(v) for _c, v in per_context_feature if isinstance(v, (int, float)))

    gap_atom = next(iter(_atoms_of(ir_doc, "rank_gap")), None)
    runner = (gap_atom or {}).get("value", {}).get("runner_up")
    gaps = dict((str(c), v) for c, v in
                ((gap_atom or {}).get("value", {}).get("per_channel") or []))

    rows = []
    # Every context feature, however many: the decomposition only means anything if the
    # shares add up, so truncating it would misrepresent the total. `collapse_after`
    # keeps the default view to the five that matter and puts the rest one click
    # away, which is what an SMD entity's 38 context features need.
    for index, value in per_context_feature:
        share = (100.0 * float(value) / total) if total else None
        delta = gaps.get(str(index))
        rows.append([
            f"context feature {index}",
            None if share is None else f"{share:.1f}%",
            round(float(value), 6),
            None if delta is None else f"{float(delta):+.6f}",
        ])
    return {
        "columns": ["Context feature", "Share", "Contribution",
                    f"vs {runner}" if runner else "vs runner-up"],
        "align": ["name", "num", "num", "num"],
        "rows": rows,
        "collapse_after": 5,
    }


_TABLE_BUILDERS = {
    "ga_combination": _ga_combination_table,
    "rank_aggregation": _rank_aggregation_table,
    "ts_ranking": _ts_ranking_table,
}


def _lead_sentence(narrative: str, ir_doc: Dict[str, Any],
                   lead_types: Sequence[str]) -> str:
    """The narrative's own sentence for the stage's headline fact, so the table
    is introduced in the stage's voice rather than by invented copy.

    Caveats are already gone by the time this runs (`strip_caveats` is applied
    to the whole body first), so neither branch can pick one up.
    """
    for sentence, atom in attribute_sentences(narrative, ir_doc):
        if atom and atom.get("type") in lead_types:
            return sentence.strip()
    for sentence in split_sentences(narrative):
        return sentence.strip()
    return ""


# ── Entry point ──────────────────────────────────────────────────────────────

def summarize(text: str, *, stage: Optional[str] = None,
              ir_doc: Optional[Dict[str, Any]] = None) -> dict:
    """Narrative text -> {"summary", "body", "is_full", "mode", "table"}.

    `body` is the narrative with caveat restatements removed — what any view of
    the prose should render, including the full-text disclosure. It is not the
    file on disk, which stays the verbatim record of what the model wrote.

    `is_full` is True when the summary is the whole narrative, which tells the
    frontend to render the disclosure pre-expanded and labelled "Full text"
    instead of offering a redundant expand. `extended_in`, when set, names a
    section of the card that already shows what the summary dropped, so the
    card suppresses its own full-text disclosure. `extended`, when set, is what
    that disclosure should show instead of the raw narrative — for sentences
    that belong to a different section of the same card rather than to either
    view of the prose.
    """
    body = (text or "").strip()
    if not body:
        return {"summary": "", "body": "", "is_full": True, "mode": "full",
                "table": None}

    # Before anything else, and for every stage — including ones with no spec
    # below. The card lists the caveats verbatim from the IR in their own
    # section, so a narrated restatement is the same limitation twice, and it
    # sits wherever the narrator put it rather than after the findings.
    if ir_doc:
        body = strip_caveats(body, ir_doc)

    spec = _STAGE_SUMMARY.get(str(stage or ""))
    if not spec or not ir_doc:
        return {"summary": body, "body": body, "is_full": True, "mode": "full",
                "table": None}

    try:
        if spec["mode"] == "drop":
            short = _drop_summary(body, ir_doc, spec["drop"])
            # A stage may hold sentences back AND carry a table; the table is
            # the answer at a glance, the kept prose says what it means.
            table = (_TABLE_BUILDERS[spec["table"]](ir_doc)
                     if spec.get("table") else None)
            # Sentences that belong to another section of the card leave the
            # full-text view too, so the card never shows them twice.
            extended = None
            if spec.get("extended_drop"):
                trimmed = _drop_summary(body, ir_doc, spec["extended_drop"])
                if trimmed and trimmed != body:
                    extended = trimmed
            # An empty or unchanged result means attribution found nothing to
            # act on; showing the whole narrative is the safe outcome.
            if short and short != body:
                return {"summary": short, "body": body, "is_full": False,
                        "mode": "drop", "table": table,
                        "extended_in": spec.get("extended_in"),
                        "extended": extended}
            return {"summary": body, "body": body, "is_full": True,
                    "mode": "full", "table": table}

        if spec["mode"] == "lead":
            # The first N sentences, and the rest behind the disclosure.
            #
            # Attribution-based dropping cannot serve the point-injection
            # stages: their narrative is one sentence per rival, all of the same
            # atom type, so there is no type to drop that would not take every
            # rival with it. What makes them long is the NUMBER of rivals —
            # eleven on a twelve-detector pool — and the card only needs to open
            # with the winner and its closest competition. Position, not type.
            table = (_TABLE_BUILDERS[spec["table"]](ir_doc)
                     if spec.get("table") else None)
            extended = None
            if spec.get("extended_drop"):
                trimmed = _drop_summary(body, ir_doc, spec["extended_drop"])
                if trimmed and trimmed != body:
                    extended = trimmed
            sentences = split_sentences(body)
            # `lead_trailing` derived the count from the IR's atoms. Unused:
            # sentences do not map one-to-one onto atoms any more.
            # trailing = set(spec.get("lead_trailing") or ())
            n = int(spec.get("sentences", 4))
            if len(sentences) > n:
                short = " ".join(s.strip() for s in sentences[:n]).strip()
                if short:
                    return {"summary": short, "body": body, "is_full": False,
                            "mode": "lead", "table": table,
                            "extended_in": spec.get("extended_in"),
                            "extended": extended}
            # `extended` travels on both paths: it says those sentences are
            # rendered elsewhere on the card, which holds whether or not the
            # summary happened to be a trim.
            return {"summary": body, "body": body, "is_full": True,
                    "mode": "full", "table": table, "extended": extended}

        if spec["mode"] == "table":
            table = _TABLE_BUILDERS[spec["table"]](ir_doc)
            if table:
                # `lead_first` took the narrative's opening sentence literally.
                # No stage uses it now that the lead is picked by atom type.
                # lead = (split_sentences(body)[0].strip() if spec.get("lead_first")
                #         and split_sentences(body) else ...)
                lead = _lead_sentence(body, ir_doc,
                                      spec.get("lead_types", ("stage_output",)))
                return {"summary": lead, "body": body, "is_full": False,
                        "mode": "table", "table": table}
    except Exception:
        pass  # any summariser failure degrades to the full text

    return {"summary": body, "body": body, "is_full": True, "mode": "full",
            "table": None}
