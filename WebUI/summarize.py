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
# "Regime 7 (windows 110 to 121) …" — the narrator always writes the index, so
# regimes attribute exactly rather than by the generic name/number overlap that
# every regime sentence would satisfy equally.
_REGIME_RE = re.compile(r"\bregime\s+(\d+)\b", re.IGNORECASE)

# Per stage: which atom types leave the default view, which table to build, or
# both. Caveats are NOT listed here — `strip_caveats` removes them from every
# stage and every view of the prose, because the card renders them verbatim
# from the IR in a section of their own.
_STAGE_SUMMARY: Dict[str, Dict[str, Any]] = {
    "ga_selection": {"mode": "drop", "drop": ("excluded_detector", "excluded_group")},
    "monte_carlo": {"mode": "drop", "drop": ("win_region",)},
    # Both importance families: the per-competitor "Against X …" atoms and the
    # "Across all competitors …" roll-up.
    "off_by_threshold": {"mode": "drop", "drop": ("feature_importance", "summary")},
    # The per-regime walk is the bulk of this narrative, and it already has a
    # disclosure of its own where each regime sits beside its SHAP plot.
    # `extended_in` says the dropped sentences are rendered there, so the card
    # must not also offer a generic "read the full explanation" — that would be
    # a second copy of the same fourteen sentences, without the plots.
    "thompson_sampling": {"mode": "drop", "drop": ("regime",),
                          "extended_in": "regimes"},
    # The default view answers the stage's question and stops: the winner, the
    # channels its score is built from, and which channels decided the margin —
    # plus the table, which is the answer at a glance. Everything after that in
    # the narrative (the selection counts, the regime walk) is supporting detail
    # behind the click.
    # `extended_drop` keeps the regime walk out of the full-text view as well:
    # those sentences belong beside their own plots, in the regime disclosure,
    # and printing them twice on one card is just a second copy without the
    # figures. Everything else the summary held back — the selection counts and
    # the regime roll-up — is what the click buys.
    "thompson_ranking": {"mode": "drop",
                         "drop": ("support", "regime_summary", "regime"),
                         "extended_drop": ("regime",),
                         "table": "ts_ranking"},
    "ga_combination": {"mode": "table", "table": "ga_combination"},
    "rank_aggregation_robust": {"mode": "table", "table": "rank_aggregation"},
    # rank_aggregation_final is deliberately absent: two sources, a couple of
    # sentences, nothing to hold back.
}


def _tokens(text: str) -> Tuple[frozenset, frozenset]:
    names = frozenset(t.lower() for t in _NAME_RE.findall(text or ""))
    numbers = frozenset(m.group(0).rstrip("%") for m in _NUM_RE.finditer(text or ""))
    return names, numbers


def split_sentences(text: str) -> List[str]:
    return [s for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]


def attribute_sentences(narrative: str,
                        ir_doc: Dict[str, Any]) -> List[Tuple[str, Optional[dict]]]:
    """Pair each narrative sentence with the atom it most likely conveys.

    Scored on shared names and numbers, names weighted higher — a paraphrase
    keeps the proper nouns even when it rewrites everything else. A sentence
    matching nothing gets None and is always kept: the summary drops only on
    positive evidence, so an unrecognised sentence degrades to being shown
    rather than silently lost.
    """
    atoms = list(ir_doc.get("evidence", []) or [])
    scored = [(a, *_tokens(str(a.get("text", "")))) for a in atoms]
    regimes = {m.group(1): a for a in atoms
               for m in [_REGIME_RE.search(str(a.get("id", "")).replace(".", " "))]
               if m and a.get("type") == "regime"}

    out: List[Tuple[str, Optional[dict]]] = []
    for sentence in split_sentences(narrative):
        s_names, s_numbers = _tokens(sentence)

        hit = _REGIME_RE.search(sentence)
        if hit and hit.group(1) in regimes:
            out.append((sentence, regimes[hit.group(1)]))
            continue

        best, best_key = None, (0, 0.0)
        for atom, a_names, a_numbers in scored:
            # A regime is identified by its index or not at all. Overlap alone
            # cannot tell "Across the regimes NN_1 led, channel 7 contributed
            # most" from a regime led by NN_1 whose channel 7 mattered — both
            # share one name and one number — and the roll-up sentences lost
            # those ties, disappearing from the summary along with the regimes.
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
    """
    skip = set(caveat_sentences(narrative, ir_doc))
    if not skip:
        return (narrative or "").strip()
    kept = [s for s in split_sentences(narrative) if s not in skip]
    # Never return nothing: a narrative that matched end to end is a matcher
    # failure, and an empty card is worse than a redundant one.
    return " ".join(s.strip() for s in kept).strip() or (narrative or "").strip()


def _drop_summary(narrative: str, ir_doc: Dict[str, Any],
                  drop_types: Sequence[str]) -> str:
    drop = set(drop_types)
    kept = [s for s, atom in attribute_sentences(narrative, ir_doc)
            if not (atom and atom.get("type") in drop)]
    return " ".join(s.strip() for s in kept).strip()


# ── Tables ───────────────────────────────────────────────────────────────────

def _atoms_of(ir_doc: Dict[str, Any], atom_type: str) -> List[dict]:
    return [a for a in (ir_doc.get("evidence") or [])
            if a.get("type") == atom_type and isinstance(a.get("value"), dict)]


def _rank_key(value: Any) -> Any:
    return float("inf") if value is None else value


# How a sign's support reads in the table. Its own column rather than a
# parenthetical on the sign: the sign is the finding, and a reader scanning the
# column should see the findings line up, not a ragged mix of bare words and
# qualified ones.
_SIGN_SUPPORT_LABEL = {"low_consistency": "low consistency",
                       "weak_influence": "weak influence"}


def _ga_combination_table(ir_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Overall weight rank, the three method ranks, the sign and its support.

    The raw Markov score stays out: it is the quantity the rank is derived
    from, its ties are decided at the 16th decimal, and quoting it invites the
    reader to compare values that are not meaningfully different.

    The sign is the sign of ALE's net accumulated effect, not of a signed SHAP
    average. It is reported for every detector that has one; the support column
    is where a thinly-evidenced sign is marked as such, so a measured negative
    is never shown as a blank.
    """
    rows = []
    for atom in _atoms_of(ir_doc, "detector_role"):
        v = atom["value"]
        rank = v.get("final_rank")
        sign = str(v.get("sign") or "").replace("_", " ")
        reasons = list(v.get("sign_support") or [])
        if sign not in ("positive", "negative"):
            support = "—"
        elif reasons:
            support = ", ".join(_SIGN_SUPPORT_LABEL.get(r, str(r)) for r in reasons)
        else:
            support = "strong"
        rows.append({
            "_sort": (_rank_key(rank), str(atom.get("subject", ""))),
            "cells": [
                f"{rank} (tie)" if v.get("final_rank_tied") else rank,
                atom.get("subject"),
                v.get("mean_abs_shap_rank"),
                v.get("pfi_rank"),
                v.get("ale_rank"),
                sign,
                support,
            ],
        })
    if not rows:
        return None
    rows.sort(key=lambda r: r["_sort"])
    return {
        "columns": ["Weight rank", "Detector", "|SHAP| rank", "PFI rank",
                    "ALE rank", "Sign", "Sign support"],
        "align": ["num", "name", "num", "num", "num", "text", "text"],
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
                str(v.get("pattern") or "").replace("_", " "),
            ],
        })
    if not rows:
        return None
    rows.sort(key=lambda r: r["_sort"])
    return {
        "columns": ["Overall standing", "Source", "Influence", "Agreement", "Pattern"],
        "align": ["num", "name", "num", "num", "text"],
        "rows": [r["cells"] for r in rows],
    }


def _ts_ranking_table(ir_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The winner's score split by channel, beside the margin over the runner-up.

    Two columns from two atoms: the share is non-negative and sums to the
    winner's score, the delta is signed and sums to the margin. They are shown
    together because the reader's question — which channels produced this
    ranking — needs both: a channel can carry most of the winner's score and
    still be one the runner-up won.
    """
    winner = next(iter(_atoms_of(ir_doc, "winner_channels")), None)
    if not winner:
        return None
    per_channel = winner["value"].get("per_channel") or []
    if not per_channel:
        return None
    total = sum(float(v) for _c, v in per_channel if isinstance(v, (int, float)))

    gap_atom = next(iter(_atoms_of(ir_doc, "rank_gap")), None)
    runner = (gap_atom or {}).get("value", {}).get("runner_up")
    gaps = dict((str(c), v) for c, v in
                ((gap_atom or {}).get("value", {}).get("per_channel") or []))

    rows = []
    # Every channel, however many: the decomposition only means anything if the
    # shares add up, so truncating it would misrepresent the total. `collapse_after`
    # keeps the default view to the five that matter and puts the rest one click
    # away, which is what an SMD entity's 38 channels need.
    for channel, value in per_channel:
        share = (100.0 * float(value) / total) if total else None
        delta = gaps.get(str(channel))
        rows.append([
            f"channel {channel}",
            None if share is None else f"{share:.1f}%",
            round(float(value), 6),
            None if delta is None else f"{float(delta):+.6f}",
        ])
    return {
        "columns": ["Channel", "Share", "Contribution",
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

        if spec["mode"] == "table":
            table = _TABLE_BUILDERS[spec["table"]](ir_doc)
            if table:
                lead = _lead_sentence(body, ir_doc, ("stage_output",))
                return {"summary": lead, "body": body, "is_full": False,
                        "mode": "table", "table": table}
    except Exception:
        pass  # any summariser failure degrades to the full text

    return {"summary": body, "body": body, "is_full": True, "mode": "full",
            "table": None}
