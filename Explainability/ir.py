"""
Intermediate Representation (IR) for the LLM interface layer.

Structures every explainability layer's output into standardized, grounded JSON
so a language model can render natural-language explanations WITHOUT computing,
ranking, or inferring anything itself. Each fact is an *atom*:

    {"id", "type", "subject", "value", "text", "confidence"?}

where `text` is a canonical sentence rendered here, in code, with numbers
already rounded — the LLM's job is compression and fluency over given
sentences. Atom ids make the thesis's faithfulness verifier mechanical
(hallucination = generated claims matching no atom; omission = required atoms
not stated → `required_atom_ids`).

Anti-hallucination principles implemented here:
  * judgments are computed in code and shipped as closed enums, never derived
    by the LLM;
  * no arrays, matrices, trajectories, or ASCII rule dumps — only scalars,
    top-k lists (k recorded), pre-computed comparatives, and structured rules
    extracted from fitted trees;
  * confidence is data: held-out fidelity, support counts, and degenerate
    flags become fields + caveat atoms (a caveat atom is a pre-written
    limitation sentence the LLM may restate);
  * deterministic bytes: sorted keys, fixed rounding, `ir_version`, no
    timestamps → identical inputs give identical JSON;
  * nothing implicit: missing/undefined → the explicit string "not_available".

This module is numpy + stdlib only. Tree rules are extracted by introspecting
a fitted DecisionTreeClassifier's `tree_` arrays, so sklearn is never imported
here.
"""

from __future__ import annotations

import glob as _glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

IR_VERSION = "1.0"
NOT_AVAILABLE = "not_available"
TOP_K = 5

# Closing line of every stage glossary. The footer describes the methods a
# stage used; this marks where those definitions stop and the narrative they
# support begins. Appended centrally in _envelope, never per builder.
FOOTER_CLOSING_SENTENCE = (
    "Following claims are based on the aforementioned explanation methods.")
# Support gate for per-rule confidence: the fidelity estimate is stratified
# 5-fold CV, and with fewer positives than folds the CV cannot place one
# positive per fold, so the held-out accuracy for that rule is undefined or
# unstable. Anchored to the fold count on purpose — not a magic number.
N_CV_FOLDS = 5


# ── Formatting / sanitising ──────────────────────────────────────────────────

def _is_nan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and np.isnan(x)) or bool(np.isnan(float(x)))
    except (TypeError, ValueError):
        return False


def _fmt(x: Any, nd: int = 3) -> str:
    """Canonical string for a number as it must appear in `text` fields."""
    if _is_nan(x):
        return NOT_AVAILABLE
    return f"{float(x):.{nd}f}"


def _val(x: Any, nd: int = 3) -> Any:
    """Rounded plain-python value for `value` fields; NaN/None → not_available."""
    if _is_nan(x):
        return NOT_AVAILABLE
    return round(float(x), nd)


def _py(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays into plain python (JSON-safe)."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_py(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return NOT_AVAILABLE if np.isnan(f) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return NOT_AVAILABLE
    return obj


def make_atom(atom_id: str, atom_type: str, subject: str, value: Any, text: str,
              confidence: Optional[str] = None,
              order: Optional[int] = None) -> Dict[str, Any]:
    atom = {
        "id": atom_id,
        "type": atom_type,
        "subject": subject,
        "value": _py(value),
        "text": text,
    }
    if confidence is not None:
        atom["confidence"] = confidence
    if order is not None:
        # Presentation order for the narration prompt (file bytes stay
        # id-sorted); lower comes first, unordered atoms follow.
        atom["order"] = int(order)
    return atom


def fidelity_grade(cv_acc: Any) -> str:
    """Closed enum for held-out surrogate fidelity."""
    if _is_nan(cv_acc):
        return NOT_AVAILABLE
    a = float(cv_acc)
    if a >= 0.8:
        return "high"
    if a >= 0.6:
        return "medium"
    return "low"


def support_grade(n_positive: Any, min_support: int = N_CV_FOLDS) -> str:
    """'low' when the positive class is smaller than the CV fold count."""
    if _is_nan(n_positive):
        return NOT_AVAILABLE
    return "adequate" if int(n_positive) >= min_support else "low"


# ── Structured rules from a fitted decision tree ─────────────────────────────

def tree_to_rules(clf: Any, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Walk a fitted DecisionTreeClassifier's tree_ arrays (no sklearn import) and
    return one dict per leaf:
        {"conditions": [{"feature", "op", "threshold"}...],
         "outcome": class_label, "n_samples": int}
    Conditions are in root→leaf order; thresholds rounded via _val (4 decimals,
    matching the report's export_text precision).
    """
    tree = clf.tree_
    classes = [str(c) for c in getattr(clf, "classes_", [])]
    rules: List[Dict[str, Any]] = []

    def _walk(node: int, conditions: List[Dict[str, Any]]) -> None:
        left, right = int(tree.children_left[node]), int(tree.children_right[node])
        if left == -1 and right == -1:  # leaf
            counts = tree.value[node].flatten()
            outcome = classes[int(np.argmax(counts))] if classes else _val(float(counts[0]))
            rules.append({
                "conditions": list(conditions),
                "outcome": outcome,
                "n_samples": int(tree.n_node_samples[node]),
            })
            return
        feat = feature_names[int(tree.feature[node])]
        thr = round(float(tree.threshold[node]), 4)
        _walk(left, conditions + [{"feature": feat, "op": "<=", "threshold": thr}])
        _walk(right, conditions + [{"feature": feat, "op": ">", "threshold": thr}])

    _walk(0, [])
    return rules


def simplify_conditions(conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse a root→leaf condition chain to the tightest bound per feature:
    repeated '<=' keep the minimum threshold, repeated '>' the maximum
    ("x <= 0.0368 and x > 0.0053 and x > 0.0158" → "x > 0.0158 and
    x <= 0.0368"). Redundant same-feature bounds read as contradictions in
    prose and get garbled by the narrator.
    """
    lower: Dict[str, float] = {}
    upper: Dict[str, float] = {}
    order: List[str] = []
    for c in conditions:
        f = str(c["feature"])
        if f not in order:
            order.append(f)
        thr = float(c["threshold"])
        if c["op"] == "<=":
            upper[f] = min(upper.get(f, thr), thr)
        else:
            lower[f] = max(lower.get(f, thr), thr)
    out: List[Dict[str, Any]] = []
    for f in order:
        if f in lower:
            out.append({"feature": f, "op": ">", "threshold": lower[f]})
        if f in upper:
            out.append({"feature": f, "op": "<=", "threshold": upper[f]})
    return out


def merge_single_feature_rules(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For rules over exactly ONE feature (already simplified to intervals),
    merge adjacent intervals with the same outcome, summing their sample
    counts — a depth-3 tree on one feature is just a partition of the axis,
    and three consecutive LOF_1 leaves are one fact, not three. Rules over
    multiple features are returned unchanged.
    """
    feats = {c["feature"] for r in rules for c in r["conditions"]}
    if len(feats) != 1 or any(len(r["conditions"]) > 2 for r in rules):
        return rules

    def _interval(r: Dict[str, Any]) -> Tuple[float, float]:
        lo, hi = float("-inf"), float("inf")
        for c in r["conditions"]:
            if c["op"] == ">":
                lo = float(c["threshold"])
            else:
                hi = float(c["threshold"])
        return lo, hi

    feat = next(iter(feats))
    ordered = sorted(rules, key=lambda r: _interval(r))
    merged: List[Dict[str, Any]] = []
    for r in ordered:
        lo, hi = _interval(r)
        if (merged and merged[-1]["outcome"] == r["outcome"]
                and _interval(merged[-1])[1] == lo):
            prev_lo = _interval(merged[-1])[0]
            merged[-1] = {
                "conditions": (
                    ([{"feature": feat, "op": ">", "threshold": prev_lo}]
                     if prev_lo != float("-inf") else [])
                    + ([{"feature": feat, "op": "<=", "threshold": hi}]
                       if hi != float("inf") else [])),
                "outcome": r["outcome"],
                "n_samples": merged[-1]["n_samples"] + r["n_samples"],
            }
        else:
            merged.append(dict(r))
    return merged


def rule_to_text(rule: Dict[str, Any], outcome_label: str = "the outcome is") -> str:
    """Canonical one-sentence rendering of a structured rule. `outcome_label`
    names what the outcome MEANS (e.g. "the noise-sweep F1 winner is") so the
    narrator never has to guess — and misbind — the rule's semantics."""
    if not rule["conditions"]:
        return f"In every observed case {outcome_label} {rule['outcome']}."
    cond = " and ".join(f"{c['feature']} {c['op']} {c['threshold']}" for c in rule["conditions"])
    return f"If {cond}, {outcome_label} {rule['outcome']} ({rule['n_samples']} samples)."


# ── Envelope / writer ────────────────────────────────────────────────────────

def _envelope(stage: str, dataset: str, entity: str, output: Dict[str, Any],
              evidence: List[Dict[str, Any]], caveats: List[Dict[str, Any]],
              required_atom_ids: List[str],
              confidence: Optional[Dict[str, Any]] = None,
              question: Optional[str] = None,
              info_footer: Optional[str] = None) -> Dict[str, Any]:
    env = {
        "ir_version": IR_VERSION,
        "stage": stage,
        "dataset": str(dataset),
        "entity": str(entity),
        "output": _py(output),
        "evidence": [_py(a) for a in sorted(evidence, key=lambda a: a["id"])],
        "caveats": [_py(a) for a in sorted(caveats, key=lambda a: a["id"])],
        "required_atom_ids": sorted(required_atom_ids),
        "confidence": _py(confidence or {}),
    }
    # `question` frames the narration prompt (the stage's headline question);
    # `info_footer` is a fixed glossary written verbatim into the .txt outside
    # the model's output, so definitions are never reworded and never count
    # toward faithfulness metrics. The closing sentence is appended here rather
    # than in each builder, so every stage — including ones added later — ends
    # the glossary the same way.
    if question is not None:
        env["question"] = str(question)
    if info_footer is not None:
        footer = str(info_footer).rstrip()
        if not footer.endswith(FOOTER_CLOSING_SENTENCE):
            footer = (footer + " " if footer else "") + FOOTER_CLOSING_SENTENCE
        env["info_footer"] = footer
    return env


def write_stage_ir(ir: Dict[str, Any], dataset: str, entity: str, filename: str,
                   base_dir: str = "myresults/explanations_ir") -> str:
    directory = os.path.join(base_dir, str(dataset), str(entity))
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{filename}.json")
    with open(path, "w") as f:
        json.dump(ir, f, sort_keys=True, indent=2)
    return path


def _top_k(seq: Sequence[Any], k: int = TOP_K) -> List[Any]:
    return list(seq[:k])


def _top_of_ranking(value: Any) -> Any:
    """
    First detector of an aggregation result, whatever shape it arrives in.

    Callers pass three different things: the ranking list itself
    (["LOF_1", "CBLOF_4", ...]), the aggregator's (score, ranking) pair, or —
    after a re-optimisation overwrite — a bare winner name. Indexing [1][0]
    blindly turns ["LOF_1", "CBLOF_4"] into "C", the first letter of the
    SECOND name, so the shape is inspected rather than assumed.
    """
    if isinstance(value, str):
        return value or NOT_AVAILABLE
    if isinstance(value, (list, tuple)):
        if not value:
            return NOT_AVAILABLE
        # A (score, ranking) pair: non-string head, sequence in second slot.
        if (len(value) > 1 and not isinstance(value[0], str)
                and isinstance(value[1], (list, tuple))):
            return _top_of_ranking(value[1])
        head = value[0]
        if isinstance(head, (str, list, tuple)):
            return _top_of_ranking(head)
    return NOT_AVAILABLE


# ── Stage builders ───────────────────────────────────────────────────────────

def _ts_channel_label(idx: Any, channel_names: Optional[Sequence[str]]) -> str:
    """Name a channel when the dataset supplies names, else fall back to its
    index. Names come from the loader, so datasets without column headers
    (SMD, SMAP/MSL) keep the numeric form."""
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return str(idx)
    if channel_names and 0 <= i < len(channel_names) and str(channel_names[i]).strip():
        return str(channel_names[i]).strip()
    return f"channel {i}"


def build_thompson_ir(dataset: str, entity: str, *, n_windows: int,
                      final_ranking: List[Tuple[str, float]],
                      regimes: List[Dict[str, Any]],
                      shifts: List[Dict[str, Any]],
                      blip_count: int,
                      state_fractions: Dict[str, float],
                      final_state: str,
                      channel_names: Optional[Sequence[str]] = None,
                      n_channels: Optional[int] = None) -> Dict[str, Any]:
    """
    `regimes` entries are precomputed by explain_thompson_sampling (it owns the
    SHAP helpers): {"index","start","end","duration","leader",
    "rewards_top": [(model, mean_reward)], "reward_gap": float|None,
    "runner_up": str|None,
    "reward_raising": [(ch, val)]|None, "reward_lowering": [(ch, val)]|None,
    "shap_raising": [(ch, val)]|None, "shap_lowering": [(ch, val)]|None,
    "edge_favor_leader": [(ch, delta)]|None,
    "edge_favor_runner": [(ch, delta)]|None, "edge_gap": float|None,
    "pref_favor_leader": [(ch, delta)]|None,
    "pref_favor_runner": [(ch, delta)]|None,
    "pref_gap": float|None}.
    Every channel list arrives pre-split by sign so no consumer ever has to
    infer direction from a signed list sorted by magnitude.

    Two comparisons against the runner-up are supplied and they are not
    interchangeable: `edge_*` differences the raw expected-reward split (its
    parts sum to the leader's margin, and it is what the prose says), while
    `pref_*` differences the SHAP split against the run's average window.

    `shifts` and `final_state` are accepted for call-site compatibility but are
    not narrated: a shift is the boundary between two regime spans (the spans
    already carry it), and the final window's state is one sample of a
    distribution the state summary reports in full.
    """
    evidence: List[Dict[str, Any]] = []
    required: List[str] = []

    def _ch(idx: Any) -> str:
        return _ts_channel_label(idx, channel_names)

    top_pairs = _top_k(final_ranking)
    top_model = top_pairs[0][0] if top_pairs else NOT_AVAILABLE
    output = {
        "top_pick": top_model,
        "final_ranking_top_k": [{"model": m, "score": _val(s, 6)} for m, s in top_pairs],
        "n_windows": int(n_windows),
        "n_regimes": len(regimes),
    }

    # ── Lead: the forwarded ranking, with the margin over the runner-up ──
    if top_pairs:
        score = top_pairs[0][1]
        if len(top_pairs) > 1 and not _is_nan(score) and not _is_nan(top_pairs[1][1]):
            runner, r_score = top_pairs[1][0], top_pairs[1][1]
            margin = float(score) - float(r_score)
            lead_val = {"top": top_model, "score": _val(score, 6),
                        "runner_up": runner, "margin": _val(margin, 6)}
            lead_txt = (f"Thompson Sampling ranked {top_model} first with a final "
                        f"score of {_fmt(score, 6)}, ahead of {runner} by "
                        f"{_fmt(margin, 6)}.")
        else:
            lead_val = {"top": top_model, "score": _val(score, 6)}
            lead_txt = (f"Thompson Sampling ranked {top_model} first with a final "
                        f"score of {_fmt(score, 6)}.")
        evidence.append(make_atom("ts.output.top", "stage_output", str(top_model),
                                  lead_val, lead_txt, order=1))
        required.append("ts.output.top")

    # ── Family sweep: only when the top three share a name prefix ──
    if len(top_pairs) >= 3:
        fams = [str(m).split("_")[0] for m, _ in top_pairs[:3]]
        if len(set(fams)) == 1 and fams[0]:
            names = [m for m, _ in top_pairs[:3]]
            evidence.append(make_atom(
                "ts.output.family", "family_sweep", fams[0],
                {"family": fams[0], "detectors": names},
                f"The {fams[0]} detectors took the top three places: "
                f"{_oxford(names)}.", order=2))

    # ── How the run divided into regimes ──
    leaders = [str(r.get("leader")) for r in regimes if r.get("leader")]
    if regimes:
        counts: Dict[str, int] = {}
        for lname in leaders:
            counts[lname] = counts.get(lname, 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        led = _oxford([f"{m} led {c}" for m, c in ordered])
        blips = int(blip_count or 0)
        blip_txt = ("" if not blips else
                    f" {blips} brief blip window{'' if blips == 1 else 's'} "
                    f"did not last long enough to count as a regime.")
        evidence.append(make_atom(
            "ts.regimes.summary", "regime_summary", "regimes",
            {"n_regimes": len(regimes), "n_windows": int(n_windows),
             "n_leaders": len(counts), "regimes_led": counts,
             "blip_count": blips},
            f"The {int(n_windows)} windows split into {len(regimes)} regimes led "
            f"by {len(counts)} different detectors: {led}.{blip_txt}", order=3))
        required.append("ts.regimes.summary")

    # ── One sentence per regime, chronological; every one required ──
    for i, r in enumerate(sorted(regimes, key=lambda x: x.get("index", 0))):
        idx = r.get("index", i)
        leader = str(r.get("leader", NOT_AVAILABLE))
        # Both narrated channel facts are now slices of the same total — the raw
        # split of mu.x, whose parts sum to the prediction. `edge_favor_leader`
        # is that same split differenced against the runner-up, so its parts sum
        # to the leader's margin in expected reward. Putting the edge in SHAP's
        # units made one sentence carry two incomparable quantities.
        supplying = [c for c, _ in (r.get("reward_raising") or [])][:2]
        favor = [c for c, _ in (r.get("edge_favor_leader") or [])][:1]
        runner = r.get("runner_up")

        # ONE sentence, carrying the regime index and every claim about it.
        #
        # This is structural, not stylistic. summarize.attribute_sentences files
        # a sentence under a regime only when it says "regime N" — the atoms
        # otherwise share all their names and numbers and cannot be told apart.
        # A trailing sentence carrying just a channel index therefore attributed
        # to nothing, or was captured by whichever unrelated atom held that
        # integer ("channel 4" landing on the regime summary's "4 different
        # detectors"); either way it escaped the walk, survived the summary drop
        # and piled up as a block of context-free sentences. Repeating "In
        # regime N" to anchor it worked only while the narrator cooperated, and
        # it stopped cooperating whenever the repetition read as redundant.
        # Keeping everything in one sentence removes the anchor problem instead
        # of policing it — and there is no second sentence to lose.
        #
        # The three claims stay three distinct clauses, in parallel participles.
        # They are different quantities: a SHARE of the expected reward, an EDGE
        # over the runner-up in those same units, and a DEPARTURE from what the
        # channel usually contributes, which is not a share of anything.
        deviating = ((r.get("shap_raising") or []) + (r.get("shap_lowering") or []))
        clauses: List[str] = []
        if supplying:
            clauses.append(f"{_oxford([_ch(c) for c in supplying])} raising its "
                           f"expected reward the most")
        if favor and runner:
            # One channel often does both jobs; "also" says so rather than
            # presenting the same channel twice as two separate findings.
            also = "also " if favor[0] in supplying else ""
            clauses.append(f"{_ch(favor[0])} {also}giving it its biggest edge "
                           f"over {runner}")
        if deviating:
            c, v = max(deviating, key=lambda cv: abs(cv[1]) if not _is_nan(cv[1]) else -1)
            direction = "above" if (not _is_nan(v) and float(v) >= 0) else "below"
            clauses.append(f"{_ch(c)} departing furthest from its usual "
                           f"contribution, running {direction} it")

        text = (f"Regime {idx} (windows {r.get('start')} to {r.get('end')}, "
                f"{r.get('duration')} windows) was led by {leader}")
        if len(clauses) == 2 and any(" and " in c for c in clauses):
            # _oxford drops the serial comma for two items, which collides with
            # the "channel 8 and channel 3" inside the first clause and yields
            # two bare "and"s. The comma is what marks where one clause ends.
            text += f", with {clauses[0]}, and {clauses[1]}"
        elif clauses:
            text += f", with {_oxford(clauses)}"
        parts = [text + "."]

        rid = f"ts.regime.{idx}"
        evidence.append(make_atom(
            rid, "regime", leader,
            {"index": idx, "start": r.get("start"), "end": r.get("end"),
             "duration": r.get("duration"), "leader": leader,
             "runner_up": runner,
             "supplying_channels": [(c, _val(v, 4)) for c, v in (r.get("reward_raising") or [])],
             "reducing_channels": [(c, _val(v, 4)) for c, v in (r.get("reward_lowering") or [])],
             # The edge, in contribution units: these sum to the leader's margin
             # in expected reward over the runner-up.
             "edge_channels": [(c, _val(v, 4)) for c, v in (r.get("edge_favor_leader") or [])],
             "edge_gap": _val(r.get("edge_gap"), 4),
             # SHAP's deviation split, kept machine-readable: it is what the
             # deviation sentence and the alternate per-regime plot are built
             # from. `deviation_edge_channels` is the same comparison as
             # `edge_channels` measured against the average window instead, and
             # is retained for provenance only.
             "deviation_raising": [(c, _val(v, 4)) for c, v in (r.get("shap_raising") or [])],
             "deviation_lowering": [(c, _val(v, 4)) for c, v in (r.get("shap_lowering") or [])],
             "deviation_edge_channels": [(c, _val(v, 4)) for c, v in (r.get("pref_favor_leader") or [])],
             "mean_rewards": [(m, _val(v, 4)) for m, v in (r.get("rewards_top") or [])],
             "mean_reward_gap": _val(r.get("reward_gap"), 4),
             "preference_score_gap": _val(r.get("pref_gap"), 4)},
            " ".join(parts), order=10 + i))
        required.append(rid)

    # ── Which channel carried the winner, across the regimes it led ──
    if top_model != NOT_AVAILABLE:
        totals: Dict[Any, float] = {}
        for r in regimes:
            if str(r.get("leader")) != str(top_model):
                continue
            for c, v in (r.get("reward_raising") or []):
                if not _is_nan(v):
                    totals[c] = totals.get(c, 0.0) + float(v)
        if totals:
            best = max(totals.items(), key=lambda kv: kv[1])
            evidence.append(make_atom(
                "ts.winner.channels", "winner_channels", str(top_model),
                {"channel": best[0], "total": _val(best[1], 4),
                 "per_channel": [(c, _val(v, 4)) for c, v in
                                 sorted(totals.items(), key=lambda kv: -kv[1])]},
                f"Across the regimes {top_model} led, {_ch(best[0])} contributed "
                f"most to its expected reward.", order=150))
            required.append("ts.winner.channels")

    # ── How the run was spent ──
    if state_fractions:
        ordered_states = sorted(state_fractions.items(), key=lambda kv: -kv[1])
        frac_txt = _oxford([f"{s.replace('_', ' ')} {_fmt(100.0 * f, 1)}% of the time"
                            for s, f in ordered_states])
        evidence.append(make_atom(
            "ts.states.summary", "behavior_summary", "selection_states",
            state_fractions,
            f"Over the {int(n_windows)} windows the sampler was in {frac_txt}.",
            order=200))
        required.append("ts.states.summary")

    caveats: List[Dict[str, Any]] = []
    if n_channels is not None and int(n_channels) == 1:
        caveats.append(make_atom(
            "ts.caveat.single_channel", "caveat", "channels", int(n_channels),
            "This dataset has a single channel, so splitting a detector's "
            "expected reward across channels carries no information — that one "
            "channel necessarily accounts for all of it."))

    # Not "why did it rank the winner first" any more: the ranking criterion has
    # its own stage (build_thompson_ranking_ir), and this one never explained it
    # — its regimes, its SHAP and its channels are all about the per-window
    # expected reward that drove selection. The question now names what the
    # evidence below actually answers.
    question = ("When did each detector have the highest chance of being chosen "
                "— which channels raised its expected reward above its rivals, "
                "and how much of the run was spent exploring rather than "
                "exploiting?")
    info_footer = (
        "Thompson Sampling learns a mean vector μ for each detector and, at every "
        "window, predicts that detector's reward as the weights applied to the "
        "window's data μᵀx; that prediction is its expected reward. The score in "
        "the resulting ranking is a different quantity and the ranking-criterion "
        "stage is where it is explained; here it is only the headline, so a "
        "detector can lead much of the run without finishing first. A regime is a "
        "stretch of at least three consecutive windows in which one detector "
        "holds the highest expected reward μᵀx; shorter changes are counted as "
        "blips. These regimes need not line up with the ranking-criterion "
        "stage's.  A channel's contribution is its own share of that expected "
        "reward: the weights on that channel applied to that channel's readings, "
        "and those shares add up to the whole prediction μᵀx. A second, narrower "
        "measure also appears: how far a channel's contribution departs from what "
        "that channel usually contributes, measured against the average of the "
        "run. The three selection states describe each choice: exploitation means "
        "the sampler picked the detector it already believed was best, informed "
        "exploration means the random draw over its uncertainty steered it "
        "elsewhere (a high share means the detectors stayed closely matched and "
        "the beliefs uncertain), and random means a forced exploration step fired "
        "(a high share means much of the run was spent sampling blindly). "
        "Thompson Sampling is a seeded stochastic sampler, so a different seed "
        "can produce a different trajectory, and the behavioural states are "
        "observed labels rather than causes.")

    return _envelope("thompson_sampling", dataset, entity, output, evidence,
                     caveats, required, question=question,
                     info_footer=info_footer)


def build_thompson_ranking_ir(dataset: str, entity: str, *, n_windows: int,
                              final_ranking: List[Tuple[str, float]],
                              winner_channels: List[Tuple[Any, float]],
                              gap_channels: List[Tuple[Any, float]],
                              selection_counts: Dict[str, int],
                              regimes: List[Dict[str, Any]],
                              warmup_windows: int,
                              channel_names: Optional[Sequence[str]] = None,
                              n_channels: Optional[int] = None) -> Dict[str, Any]:
    """
    The sibling of build_thompson_ir, for the ranking criterion rather than the
    selection dynamics.

    Thompson Sampling ranks detectors by ||mu_k||^2, but build_thompson_ir
    explains mu^T x — the expected reward that drove per-window selection. This
    builder explains the ranking itself: ||mu||^2 splits exactly into one
    non-negative contribution per channel, and the winner's margin over the
    runner-up splits exactly into one signed term per channel.

    `winner_channels` is [(channel, contribution)] for the winner, descending;
    `gap_channels` is [(channel, signed delta)] for winner-minus-runner-up,
    sorted by descending magnitude. Both are precomputed by
    explain_thompson_ranking, which owns the decomposition helpers.

    `regimes` entries: {"index","start","end","duration","leader","runner_up",
    "top_channels": [(ch, contribution)], "gap_channels": [(ch, delta)],
    "score", "runner_score"} — regimes here are stretches of windows in which
    one detector held the highest ||mu||^2, which is a different quantity from
    the expected-reward regimes of the sibling stage.
    """
    evidence: List[Dict[str, Any]] = []
    required: List[str] = []

    def _ch(idx: Any) -> str:
        return _ts_channel_label(idx, channel_names)

    top_pairs = _top_k(final_ranking)
    top_model = top_pairs[0][0] if top_pairs else NOT_AVAILABLE
    runner_up = top_pairs[1][0] if len(top_pairs) > 1 else None
    output = {
        "top_pick": top_model,
        "final_ranking_top_k": [{"model": m, "score": _val(s, 6)} for m, s in top_pairs],
        "n_windows": int(n_windows),
        "n_regimes": len(regimes),
        "n_channels": None if n_channels is None else int(n_channels),
        "warmup_windows": int(warmup_windows or 0),
    }

    # ── Lead: the ranking this stage exists to explain ──
    if top_pairs:
        score = top_pairs[0][1]
        if runner_up is not None and not _is_nan(score) and not _is_nan(top_pairs[1][1]):
            margin = float(score) - float(top_pairs[1][1])
            lead_val = {"top": top_model, "score": _val(score, 6),
                        "runner_up": runner_up, "margin": _val(margin, 6)}
            lead_txt = (f"Ranked by the size of its learned weights, {top_model} "
                        f"scored {_fmt(score, 6)}, ahead of {runner_up} by "
                        f"{_fmt(margin, 6)}.")
        else:
            lead_val = {"top": top_model, "score": _val(score, 6)}
            lead_txt = (f"Ranked by the size of its learned weights, {top_model} "
                        f"scored {_fmt(score, 6)}.")
        evidence.append(make_atom("tsr.output.top", "stage_output", str(top_model),
                                  lead_val, lead_txt, order=1))
        required.append("tsr.output.top")

    # ── Where the winner's score came from ──
    total = sum(float(v) for _, v in winner_channels if not _is_nan(v))
    if winner_channels and total > 0:
        lead_ch = winner_channels[:3]
        shares = [(c, 100.0 * float(v) / total) for c, v in lead_ch]
        listed = _oxford([f"{_ch(c)} ({_fmt(p, 1)}%)" for c, p in shares])
        evidence.append(make_atom(
            "tsr.winner.channels", "winner_channels", str(top_model),
            {"channel": lead_ch[0][0], "total": _val(total, 6),
             "per_channel": [(c, _val(v, 6)) for c, v in winner_channels],
             "top_shares": [(c, _val(p, 1)) for c, p in shares]},
            f"{listed} contributed the majority of {top_model}'s score.",
            order=10))
        required.append("tsr.winner.channels")

    # ── What actually decided the top spot ──
    if gap_channels and runner_up:
        gains = [(c, v) for c, v in gap_channels if not _is_nan(v) and float(v) > 0][:2]
        losses = [(c, v) for c, v in gap_channels if not _is_nan(v) and float(v) < 0][:1]
        parts = []
        if gains:
            # No raw values in the prose. The table and the gap plot carry the
            # per-channel numbers, and quoting six-decimal figures mid-sentence
            # only invites the narrator to gloss what they mean.
            labels = _oxford([_ch(c) for c, _v in gains])
            parts.append(f"{labels[:1].upper() + labels[1:]} contributed "
                         f"significantly to the lead that {top_model} had over "
                         f"{runner_up}")
        if losses:
            c, _v = losses[0]
            # Direction in words, no number and no sign to interpret. An earlier
            # phrasing paired the signed value with "in NN_3's favour" and got
            # read the other way round ("channel 8 had a negative impact on
            # NN_3's score"), inverting the one claim this stage exists to make.
            # Naming BOTH sides ("favoured X more than Y") is what makes the
            # comparison unambiguous without a number.
            parts.append(f"while {_ch(c)} favoured {runner_up} more than "
                         f"{top_model}")
        if parts:
            evidence.append(make_atom(
                "tsr.gap.runner_up", "rank_gap", str(top_model),
                # `rivals` is one of verifier._RIVAL_KEYS, so naming the key this
                # way gets the rival set checked against the narrated sentence.
                {"rivals": [runner_up], "runner_up": runner_up,
                 "per_channel": [(c, _val(v, 6)) for c, v in gap_channels],
                 "favouring_winner": [(c, _val(v, 6)) for c, v in gains],
                 "favouring_runner_up": [(c, _val(v, 6)) for c, v in losses]},
                ", ".join(parts) + ".", order=20))
            required.append("tsr.gap.runner_up")

    # ── How much evidence each ranking score rests on ──
    if selection_counts and top_model != NOT_AVAILABLE:
        ordered_counts = sorted(selection_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        winner_n = int(selection_counts.get(top_model, 0))
        support_val = {"counts": {str(m): int(c) for m, c in ordered_counts},
                       "winner_selections": winner_n,
                       "n_windows": int(n_windows)}
        support_txt = (f"{top_model} was selected in {winner_n} of the "
                       f"{int(n_windows)} windows")
        # Against the runner-up, not the least-tried detector. The comparison
        # that matters is between the two the ranking actually separates, and
        # it is often the interesting one — the winner can come first on FEWER
        # selections. It is also the safer name: the runner-up is already named
        # in the lead and gap sentences, whereas a least-tried detector appears
        # nowhere else and got narrated as the wrong name outright.
        if runner_up is not None and str(runner_up) != str(top_model):
            runner_n = int(selection_counts.get(runner_up, 0))
            support_val.update({"runner_up": runner_up,
                                "runner_up_selections": runner_n})
            support_txt += f", against {runner_n} for {runner_up}"
        evidence.append(make_atom("tsr.support", "support", str(top_model),
                                  support_val, support_txt + ".", order=30))
        required.append("tsr.support")

    # ── How leadership on the ranking score changed over the run ──
    if regimes:
        counts: Dict[str, int] = {}
        spans: Dict[str, int] = {}
        for r in regimes:
            lname = str(r.get("leader"))
            counts[lname] = counts.get(lname, 0) + 1
            try:
                spans[lname] = spans.get(lname, 0) + int(r.get("duration") or 0)
            except (TypeError, ValueError):
                spans.setdefault(lname, 0)
        # Sorted by windows held, not by regime count: a detector that led four
        # short spells did not lead more of the run than one that held a single
        # long one, and the count alone reads as though it did.
        ordered = sorted(counts.items(), key=lambda kv: (-spans.get(kv[0], 0), kv[0]))
        led = _oxford([
            f"{m} led {c} regime{'' if c == 1 else 's'}, spanning "
            f"{spans.get(m, 0)} window{'' if spans.get(m, 0) == 1 else 's'}"
            for m, c in ordered])
        warm = int(warmup_windows or 0)
        warm_txt = ("" if not warm else
                    f" The first {warm} windows are left out, because until every "
                    f"detector has been tried its score is still zero.")
        if len(regimes) == 1:
            # The count phrasing ("X led 1 regime, spanning N windows") reads as
            # nonsense when there is only one spell to count.
            solo = ordered[0][0]
            head = (f"One detector held the highest score for the whole run: "
                    f"{solo}, across {spans.get(solo, 0)} windows.")
        else:
            head = (f"Leadership on this score changed hands over the run: it "
                    f"splits into {len(regimes)} regimes led by {len(counts)} "
                    f"different detectors: {led}.")
        evidence.append(make_atom(
            "tsr.regimes.summary", "regime_summary", "regimes",
            {"n_regimes": len(regimes), "n_windows": int(n_windows),
             "n_leaders": len(counts), "regimes_led": counts,
             "windows_led": spans, "warmup_windows": warm},
            head + warm_txt, order=40))
        required.append("tsr.regimes.summary")

    # ── One sentence per regime, chronological; every one required ──
    for i, r in enumerate(sorted(regimes, key=lambda x: x.get("index", 0))):
        idx = r.get("index", i)
        leader = str(r.get("leader", NOT_AVAILABLE))
        top_channels = r.get("top_channels") or []
        runner = r.get("runner_up")

        dur = r.get("duration")
        parts = [f"Regime {idx} (windows {r.get('start')} to {r.get('end')}, "
                 f"{dur} window{'' if dur == 1 else 's'}) was led by {leader}."]
        if top_channels:
            labels = _oxford([_ch(c) for c, _ in top_channels[:2]])
            # Uppercase only the first letter — .capitalize() would lowercase
            # the rest, mangling real channel names like Accelerometer1RMS.
            labels = labels[:1].upper() + labels[1:]
            # No "ahead of {runner_up}" tail. The narrator drops that clause
            # from every regime sentence, and because coverage is conjunctive
            # the atom then only passes when the runner-up happens to be named
            # elsewhere in the narrative — true for the NN_* that lead most
            # regimes, false for the odd early one, so a run's faithfulness
            # turned on which detector placed second in regime 0. The runner-up
            # stays in `value`, and the per-regime figure plots it beside the
            # leader, which is where a comparison belongs anyway.
            parts.append(f"{labels} supplied most of the score that kept it "
                         f"there.")

        rid = f"tsr.regime.{idx}"
        evidence.append(make_atom(
            rid, "regime", leader,
            {"index": idx, "start": r.get("start"), "end": r.get("end"),
             "duration": r.get("duration"), "leader": leader,
             "runner_up": runner,
             "top_channels": [(c, _val(v, 6)) for c, v in top_channels],
             "gap_channels": [(c, _val(v, 6)) for c, v in (r.get("gap_channels") or [])],
             "score": _val(r.get("score"), 6),
             "runner_score": _val(r.get("runner_score"), 6)},
            " ".join(parts), order=50 + i))
        required.append(rid)

    # ── Caveats ──
    # Both are properties of the method rather than of this run, but neither can
    # move into the footer the way Thompson's did: the first is the one mistake
    # the prose can make that the verifier cannot catch, and the second is a
    # real limit on what the ranking means.
    caveats: List[Dict[str, Any]] = [
        make_atom(
            "tsr.caveat.nonnegative", "caveat", "channels", None,
            "Each channel's share is a sum of squared weights, so it can never "
            "be negative: the split shows how a detector's score is divided "
            "among channels, not which channels pushed it down. Only the "
            "comparison against a rival has a direction."),
        make_atom(
            "tsr.caveat.exposure", "caveat", "selections", None,
            "A detector's weights only move in windows where it was selected, "
            "so this score reflects how often a detector was tried as well as "
            "how well it did; a rarely selected detector ranks low on thin "
            "evidence rather than on a poor showing."),
    ]
    if n_channels is not None and int(n_channels) == 1:
        caveats.append(make_atom(
            "tsr.caveat.single_channel", "caveat", "channels", int(n_channels),
            "This dataset has a single channel, so splitting the score across "
            "channels carries no information — that one channel necessarily "
            "accounts for all of it."))

    question = ("Why did Thompson Sampling rank the detectors as it did — which "
                "channels drove each detector's ranking score up, and how much "
                "of the winner's margin came from each channel?")
    info_footer = (
        "Thompson Sampling learns a mean vector μ for each detector, one weight "
        "per channel per timestep of a window. It ranks the detectors by the "
        "overall size of those weights μᵀμ (‖μ_k‖²). That total splits exactly "
        "into one number per channel: each channel's share is the sum of its own "
        "squared weights, and the shares add up to the whole score. Because they "
        "are squares, shares are never negative, so a small share means a channel "
        "contributed little, not that it worked against the detector. Comparing "
        "two detectors channel by channel does have a direction: the difference "
        "between their shares adds up exactly to the gap between their scores, "
        "and a negative difference marks a channel on which the rival was "
        "stronger. A regime here is a stretch of windows in which one detector "
        "held the highest score; these are not the same regimes as the "
        "selection-dynamics stage reports. Thompson Sampling is a seeded "
        "stochastic sampler, so a different seed can produce a different "
        "trajectory.")

    return _envelope("thompson_ranking", dataset, entity, output, evidence,
                     caveats, required, question=question,
                     info_footer=info_footer)


# Included-member reason buckets, in narration order. `needed` (a low-profile
# member kept because removing it costs fitness) is grouped by LOFO cost below.
_GA_SEL_BUCKETS = ("both", "utility", "stability", "marginal")

# The utility/stability profile each bucket asserts, as the two-letter code the
# verifier's attribution channel checks claimed "high/low utility" wording
# against. The code never reaches the prompt (only atom `text` does) — the
# terminology stays out of the prose while the claim stays machine-checkable.
_GA_SEL_BUCKET_CODES = {"both": "HH", "utility": "HL", "stability": "LH",
                        "marginal": "LL"}


def build_ga_selection_ir(dataset: str, entity: str, result: Dict[str, Any]) -> Dict[str, Any]:
    best = list(result.get("best_ensemble", []))
    lofo: Dict[str, float] = result.get("lofo", {})
    mm: Dict[str, Dict[str, float]] = result.get("mean_marginal", {})
    archetypes: Dict[str, Dict[str, Any]] = result.get("archetypes", {})
    detectors = list(archetypes.keys())
    util = {d: mm.get(d, {}).get("contribution", float("nan")) for d in detectors}

    def _flags(d: str) -> Tuple[Any, Any]:
        """Relative (median-split) high/low utility & stability flags. Prefers
        the explicit booleans; falls back to the 2-letter archetype code
        (e.g. 'HL' -> high utility, low stability) when only the code is given."""
        rel = archetypes.get(d, {}).get("relative", {})
        u, s = rel.get("u_high"), rel.get("s_high")
        if u is None and s is None:
            code = rel.get("archetype", "")
            if isinstance(code, str) and len(code) == 2 and set(code) <= {"H", "L"}:
                return code[0] == "H", code[1] == "H"
        return u, s

    def _num(d: str) -> Dict[str, Any]:
        """Per-detector utility/stability, kept in `value` for grounding but out
        of the prose (the narrative reasons in high/low terms)."""
        sm = archetypes.get(d, {}).get("stability_mean", float("nan"))
        return {"utility": _val(util.get(d), 4), "stability": _val(sm, 3)}

    def _were(names: Sequence[str]) -> str:
        return "was" if len(names) == 1 else "were"

    def _have(names: Sequence[str]) -> str:
        return "had" if len(names) == 1 else "each had"

    def _them(names: Sequence[str]) -> str:
        return "it" if len(names) == 1 else "them"

    evidence: List[Dict[str, Any]] = []
    required: List[str] = []
    n = len(best)
    output = {"best_ensemble": best, "ensemble_size": n}
    evidence.append(make_atom(
        "ga_sel.output.ensemble", "stage_output", "best_ensemble", best,
        (f"The genetic algorithm selected the {n}-detector ensemble "
         f"{{{', '.join(best)}}}." if best
         else "The genetic algorithm selected no ensemble."), order=1))
    required.append("ga_sel.output.ensemble")

    # ── Included members: one reason per member, then grouped by reason ──
    buckets: Dict[str, List[str]] = {b: [] for b in _GA_SEL_BUCKETS}
    needed: List[str] = []
    for d in best:
        u_high, s_high = _flags(d)
        lv = lofo.get(d, float("nan"))
        if u_high and s_high:
            buckets["both"].append(d)
        elif u_high:
            buckets["utility"].append(d)
        elif s_high:
            buckets["stability"].append(d)
        elif not _is_nan(lv) and lv > 0:
            needed.append(d)          # low profile, but removing it costs fitness
        else:
            buckets["marginal"].append(d)

    def _bucket_text(b: str, names: Sequence[str]) -> str:
        w, th = _were(names), _them(names)
        if b == "both":
            return (f"{_oxford(names)} {w} chosen for both high utility and high "
                    f"stability.")
        if b == "utility":
            return (f"{_oxford(names)} {w} chosen for high utility, despite lower "
                    f"stability.")
        if b == "stability":
            return (f"{_oxford(names)} {w} chosen for high stability — the genetic "
                    f"algorithm kept {th} in most generations — despite low utility.")
        return (f"{_oxford(names)} {w} low on both utility and stability, and "
                f"removing {th} barely changes fitness; the genetic algorithm "
                f"retained {th} in its best-scoring subset.")

    order = 10
    for b in _GA_SEL_BUCKETS:
        names = buckets[b]
        if not names:
            continue
        bid = f"ga_sel.included.{b}"
        evidence.append(make_atom(
            bid, "member_reason", b,
            {"detectors": names, "reason": b,
             "archetype": _GA_SEL_BUCKET_CODES[b],
             "per_detector": {d: _num(d) for d in names}},
            _bucket_text(b, names), order=order))
        required.append(bid)
        order += 10
    # Detectors kept only because removing them costs fitness are grouped by
    # that cost. Emitting one near-identical sentence each — same verb, same
    # structure, often the same LOFO number — is what made a narrator open the
    # second with "Similarly," and then reach for the dominant "high utility and
    # high stability" phrasing from the group above, inverting the very fact the
    # sentence was carrying. One sentence per distinct cost has nothing to drift
    # into. `despite` is avoided for the same reason: a concessive clause
    # backgrounds the low/low finding and invites restatement as a positive.
    needed_by_cost: Dict[str, List[str]] = {}
    for d in needed:
        needed_by_cost.setdefault(_fmt(lofo.get(d, float("nan")), 4), []).append(d)
    for cost, names in sorted(needed_by_cost.items(),
                              key=lambda kv: (-len(kv[1]), kv[0])):
        rid = ("ga_sel.needed." + names[0] if len(names) == 1
               else "ga_sel.needed.group" + str(len(needed_by_cost)))
        if len(names) == 1:
            text = (f"{names[0]} has low utility and low stability, yet removing "
                    f"it lowers the ensemble's fitness by {cost}, which is why it "
                    f"was kept.")
        else:
            text = (f"{_oxford(names)} each have low utility and low stability, "
                    f"yet removing any one of them lowers the ensemble's fitness "
                    f"by {cost} apiece, which is why all of them were kept.")
        evidence.append(make_atom(
            rid, "member_reason", names[0] if len(names) == 1 else "needed",
            {"detectors": names, "reason": "needed",
             "lofo": _val(lofo.get(names[0], float("nan")), 4),
             "per_detector": {d: _num(d) for d in names},
             # The verifier's archetype channel checks claimed high/low wording
             # against this code; without it the inversion above was invisible.
             "archetype": "LL"},
            text, order=order))
        required.append(rid)
        order += 10

    # ── Excluded detectors: grouped by profile, notable ones called out ──
    excluded = sorted((d for d in detectors if d not in best),
                      key=lambda d: (float("-inf") if _is_nan(util[d]) else util[d]),
                      reverse=True)
    exc_stable: List[str] = []
    exc_plain: List[str] = []
    exc_nodata: List[str] = []
    for d in excluded:
        if _is_nan(util[d]):
            exc_nodata.append(d)
            continue
        u_high, s_high = _flags(d)
        if u_high:                    # high-utility yet not selected — the anomaly
            eid = f"ga_sel.excluded.{d}"
            # The profile LEADS the sentence. Three excluded atoms in a row all
            # opened "X was left out …" and differed only in the high/low tail,
            # so a narrator merged them and gave one detector another's profile.
            # Fronting the distinguishing clause is the same reshape that fixed
            # the `needed` atoms; it also stops the reader meeting the verdict
            # before the property that makes it surprising.
            stab = "high stability" if s_high else "low stability"
            evidence.append(make_atom(
                eid, "excluded_detector", d,
                dict(_num(d), u_high=True, s_high=bool(s_high),
                     archetype="HH" if s_high else "HL"),
                f"{d} had high utility and {stab}, but was still left out.",
                order=order))
            required.append(eid)
            order += 10
        elif s_high:
            exc_stable.append(d)
        else:
            exc_plain.append(d)
    for gid, names, code, txt in (
        ("ga_sel.excluded.stable", exc_stable, "LH",
         lambda ns: f"{_oxford(ns)} {_have(ns)} low utility and high stability, "
                    f"and {_were(ns)} left out."),
        ("ga_sel.excluded.plain", exc_plain, "LL",
         lambda ns: f"{_oxford(ns)} {_have(ns)} low utility and low stability, "
                    f"and {_were(ns)} left out."),
        # No utility data means no profile to assert, so no code to check.
        ("ga_sel.excluded.nodata", exc_nodata, None,
         lambda ns: f"{_oxford(ns)} {_were(ns)} left out with no marginal-"
                    f"contribution data to judge utility."),
    ):
        if names:
            value: Dict[str, Any] = {"detectors": names,
                                     "per_detector": {d: _num(d) for d in names}}
            if code:
                value["archetype"] = code
            evidence.append(make_atom(
                gid, "excluded_group", gid.rsplit(".", 1)[1], value,
                txt(names), order=order))
            required.append(gid)
            order += 10

    caveats: List[Dict[str, Any]] = []
    if n < 2:
        caveats.append(make_atom(
            "ga_sel.caveat.lofo_na", "caveat", "lofo", None,
            "With fewer than two detectors, LOFO (the leave-one-out fitness "
            "change) is undefined."))

    question = ("Why were the detectors in the ensemble chosen, and why were the "
                "rest left out?")
    info_footer = (
        "The genetic algorithm chooses the subset by searching for the ensemble "
        "with the highest fitness; it never scores detectors individually. The "
        "two properties below are measured afterwards, from the subsets that "
        "search evaluated, to explain the ensemble it arrived at. Utility is a "
        "detector's mean marginal contribution — the average lift in the "
        "ensemble's F1 score when "
        "it is added to a subset, across the subsets the genetic algorithm tried. "
        "Stability is the fraction of generations in which the algorithm kept the "
        "detector, so a high-stability detector is one it selected in most of its "
        "ensembles. High and low are relative to this cohort — a median split "
        "across the detectors evaluated together — so they mean above or below the "
        "others, not absolute quality. Fitness is the ensemble's best-threshold F1 "
        "score, the objective the algorithm maximises. LOFO is how much that "
        "fitness drops when a single detector is removed from the final ensemble; "
        "a positive value means the detector was pulling weight.")

    return _envelope("ga_selection", dataset, entity, output, evidence, caveats,
                     required, question=question, info_footer=info_footer)


# Markov scores that differ by less than this are one tie. The chain behind
# them is fed by three measures, so C[j,i] - C[i,j] takes only five values and
# exact ties are common; np.linalg.eig then returns those ties a few ulp
# apart. Comparing raw floats turned a 1.1e-16 wobble into "carries the most
# weight", with the eigen-solver — not the data — deciding which detector led.
_RANK_TIE_ATOL = 1e-9


def _tied(a: Any, b: Any) -> bool:
    """Are two scores the same rank? Tolerant, and NaN-safe."""
    if a is None or b is None:
        return a is b
    if _is_nan(a) or _is_nan(b):
        return _is_nan(a) and _is_nan(b)
    try:
        return abs(float(a) - float(b)) <= _RANK_TIE_ATOL
    except (TypeError, ValueError):
        return a == b


def _competition_rank(scores: Dict[str, Any], order: List[str]) -> Dict[str, int]:
    """Competition ranking ('1224') over `order` (already score-descending).

    Ties are compared against the running block's score, not the immediately
    preceding item: with a tolerance, pairwise chaining would let a long drift
    of near-equal values collapse into one rank even when the ends differ.
    """
    ranks: Dict[str, int] = {}
    block = None
    rank = 0
    for i, d in enumerate(order):
        v = scores.get(d)
        if block is None or not _tied(v, block):
            rank = i + 1
            block = v
        ranks[d] = rank
    return ranks


def _leaders(scores: Dict[str, Any], order: List[str]) -> List[str]:
    """Every item sharing the top rank — usually one, but a tie is a real tie."""
    if not order:
        return []
    return [d for d in order if _tied(scores.get(d), scores.get(order[0]))]


# The measures whose RANKS explain a detector's weight — the three magnitudes
# that feed the Markov aggregation. Signed SHAP is absent on purpose: it does
# not feed the aggregation, and it no longer supplies the sign either
# (that is now ALE's net effect), so quoting its rank here would imply a
# contribution it does not make.
_GA_METHODS = ("absolute SHAP", "PFI", "total ALE")

_WEIGHT_ORD = {1: "the most", 2: "the second-most", 3: "the third-most",
               4: "the fourth-most", 5: "the fifth-most", 6: "the sixth-most",
               7: "the seventh-most", 8: "the eighth-most", 9: "the ninth-most",
               10: "the tenth-most", 11: "the eleventh-most", 12: "the twelfth-most"}


def _weight_phrase(final_rank: Any, n_total: int, tied: bool = False) -> str:
    """'carries the Nth-most weight in the ensemble (overall weight rank N of M)'.

    The parenthetical is not redundant: the ordinal is the only quantity in the
    sentence with no digit, and it sits beside two digit method ranks on the
    same small-integer scale. Narrators re-derived it from whichever digit was
    nearest ("coming in fourth" from an absolute-SHAP rank of 4), so it now
    carries its own label and its own number and cannot be inferred wrongly.
    """
    if final_rank is None or _is_nan(final_rank):
        return "weight in the ensemble (overall weight rank not available)"
    fr = int(final_rank)
    ordinal = _WEIGHT_ORD.get(fr, f"the {fr}th-most")
    verb_obj = f"{ordinal} weight in the ensemble"
    if tied:
        return f"{verb_obj} (overall weight rank {fr} of {n_total}, a tie)"
    return f"{verb_obj} (overall weight rank {fr} of {n_total})"


def _oxford(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _rank_phrase(ranks: Sequence[Any]) -> str:
    """Render the weight-bearing method ranks, grouping measures that share one:
    (1, 1, 1) -> 'ranking 1 on absolute SHAP, PFI, and total ALE';
    (2, 3, 2) -> 'ranking 2 on absolute SHAP and total ALE, and 3 on PFI'."""
    order: List[int] = []
    groups: Dict[int, List[str]] = {}
    na: List[str] = []
    for label, rk in zip(_GA_METHODS, ranks):
        if rk is None or _is_nan(rk):
            na.append(label)
            continue
        rk = int(rk)
        if rk not in groups:
            groups[rk] = []
            order.append(rk)
        groups[rk].append(label)
    parts = [f"{rk} on {_oxford(groups[rk])}" for rk in order]
    if not parts:
        phrase = "with no method ranking available"
    elif len(parts) == 2 and any(" and " in p for p in parts):
        # _oxford drops the serial comma for two items, which with three
        # measures can put an "and" inside a group AND between the groups
        # ("2 on absolute SHAP and total ALE and 3 on PFI"). Forcing the comma
        # back in is what marks where one rank group ends and the next begins.
        phrase = f"ranking {parts[0]}, and {parts[1]}"
    else:
        phrase = "ranking " + _oxford(parts)
    if na:
        phrase += f" (not ranked on {_oxford(na)})"
    return phrase


def _sign_summary_text(members: Sequence[str], signs: Dict[str, str]) -> str:
    """One sentence naming each member's sign.

    Names the sign and stops. What a sign MEANS — which way a rising score
    moves the meta-learner — is defined once in the info footer, so spelling it
    out per detector here restates the glossary in every run.

    The third bucket is narrow: every detector with a net effect gets its sign,
    however thinly supported (the caveat carries that), so landing here means
    the effect nets to nothing at all rather than that the sign was withheld.
    """
    pos = [d for d in members if signs.get(d) == "positive"]
    neg = [d for d in members if signs.get(d) == "negative"]
    na = [d for d in members if signs.get(d) not in ("positive", "negative")]
    # Ensembles of one are ordinary here, so the verb has to agree or every
    # such run reads as broken English.
    def _v(group: Sequence[str], singular: str, plural: str) -> str:
        return singular if len(group) == 1 else plural

    if pos and neg:
        # The second clause elides "signs" — the first has just supplied it.
        text = (f"{_oxford(pos)} had {_v(pos, 'a positive sign', 'positive signs')}, "
                f"while {_oxford(neg)} had negative")
    elif pos:
        text = f"{_oxford(pos)} had {_v(pos, 'a positive sign', 'positive signs')}"
    elif neg:
        text = f"{_oxford(neg)} had {_v(neg, 'a negative sign', 'negative signs')}"
    elif na:
        return (f"No detector has a sign: {_oxford(na)} "
                f"{_v(na, 'has', 'have')} no net effect.")
    else:
        return ""
    if na:
        text += f"; {_oxford(na)} had no sign"
    return text + "."


def build_ga_combination_ir(dataset: str, entity: str, result: Dict[str, Any]) -> Dict[str, Any]:
    ranking = list(result.get("final_ranking", []))
    members = list(result.get("best_ensemble", []))
    pi: Dict[str, float] = result.get("markov_scores", {})
    s_abs: Dict[str, float] = result.get("shap_importance", {})
    pfi: Dict[str, float] = result.get("pfi_importance", {})
    ale_tv: Dict[str, float] = result.get("ale_total_variation", {})
    ale_net: Dict[str, float] = result.get("ale_net", {})
    ale_cons: Dict[str, float] = result.get("ale_consistency", {})
    # The sign is decided by the producer, which owns the gate thresholds and
    # the ALE curves they are read from. Reproducing that logic here would give
    # the report and the narrative two chances to disagree.
    ale_sign: Dict[str, str] = result.get("ale_sign", {})
    # Per detector, the reasons its sign is weakly supported ([] when it is not).
    ale_support: Dict[str, Any] = result.get("ale_sign_support", {})

    def _rank_of(imp: Dict[str, float]) -> Dict[str, int]:
        order = sorted((d for d in ranking if not _is_nan(imp.get(d))),
                       key=lambda d: imp[d], reverse=True)
        return {d: i + 1 for i, d in enumerate(order)}

    r_abs, r_pfi, r_ale = _rank_of(s_abs), _rank_of(pfi), _rank_of(ale_tv)
    final_rank = _competition_rank(pi, ranking)  # competition rank, display only
    # How many detectors share each rank, so a tied member says so instead of
    # claiming a lead that rests on eigen-solver noise.
    rank_counts: Dict[int, int] = {}
    for d in ranking:
        r = final_rank.get(d)
        if r is not None:
            rank_counts[r] = rank_counts.get(r, 0) + 1
    leaders = _leaders(pi, ranking)

    # Every ensemble member is narrated (no top-k cap): the sign summary must
    # classify all of them, and GA ensembles are small. `detectors` ARE the
    # ensemble members — never rank-aggregation sources.
    detectors = ranking if ranking else members
    signs: Dict[str, str] = {d: str(ale_sign.get(d, NOT_AVAILABLE)) for d in detectors}
    support: Dict[str, List[str]] = {
        d: [str(r) for r in (ale_support.get(d) or [])] for d in detectors}
    # Detectors with no sign at all: ALE was undefined for the column, or the
    # net effect is exactly zero. A sign that merely rests on thin evidence is
    # NOT in here — it is reported, and qualified by the caveat below.
    unsigned = [d for d in detectors if signs[d] not in ("positive", "negative")]
    weak = [d for d in detectors if support[d] and d not in unsigned]

    evidence: List[Dict[str, Any]] = []
    required: List[str] = []
    top = ranking[0] if ranking else NOT_AVAILABLE
    output = {
        "top_pick": top,
        # When several detectors tie for the top, `top_pick` is the first of an
        # arbitrary order — record the whole tied set so a reader is not misled
        # into treating it as a sole winner.
        "top_pick_tied_with": [d for d in leaders if d != top],
        "ensemble_members": members,
        "ensemble_size": len(members),
        "meta_model_type": result.get("meta_model_type", NOT_AVAILABLE),
        "baseline_f1": _val(result.get("baseline_f1"), 4),
    }

    if detectors:
        n = len(detectors)
        evidence.append(make_atom(
            "ga_comb.output.subset", "stage_output", "best_ensemble", list(detectors),
            # "this stage measures…" left the subject implicit and a narrator
            # reattached it to the nearest noun ("This ensemble measures how
            # each detector influences…"), which is nonsense the verifier
            # cannot see — every name and number in it is correct. Naming the
            # ranking gives the clause a referent it cannot slide off.
            f"The genetic algorithm selected the {n}-detector ensemble "
            f"{{{', '.join(detectors)}}}; the ranking below measures how much "
            f"each of those detectors moves the trained meta-learner's output.",
            order=1))
        required.append("ga_comb.output.subset")

    for i, d in enumerate(detectors):
        rid = f"ga_comb.detector.{d}.role"
        evidence.append(make_atom(
            rid, "detector_role", d,
            {"final_rank": final_rank.get(d),
             "final_rank_tied": rank_counts.get(final_rank.get(d), 1) > 1,
             "markov_score": _val(pi.get(d), 4),
             "mean_abs_shap": _val(s_abs.get(d), 6), "mean_abs_shap_rank": r_abs.get(d),
             "pfi_f1_drop": _val(pfi.get(d), 6), "pfi_rank": r_pfi.get(d),
             "ale_total": _val(ale_tv.get(d), 6), "ale_rank": r_ale.get(d),
             # Net and consistency travel with the sign they produced, so a
             # reader can see WHY a detector was left without one.
             "ale_net": _val(ale_net.get(d), 6),
             "sign_consistency": _val(ale_cons.get(d), 2),
             "sign": signs[d],
             "sign_support": support[d]},
            f"{d} carries "
            f"{_weight_phrase(final_rank.get(d), len(detectors), tied=rank_counts.get(final_rank.get(d), 1) > 1)}, "
            f"{_rank_phrase((r_abs.get(d), r_pfi.get(d), r_ale.get(d)))}.",
            order=10 * (i + 1)))
        required.append(rid)

    if detectors:
        sign_text = _sign_summary_text(detectors, signs)
        if sign_text:
            evidence.append(make_atom(
                "ga_comb.sign_summary", "sign_summary", "sign",
                {"positive": [d for d in detectors if signs[d] == "positive"],
                 "negative": [d for d in detectors if signs[d] == "negative"],
                 "no_sign": unsigned},
                sign_text, order=10 * (len(detectors) + 1)))
            required.append("ga_comb.sign_summary")

    caveats = [
        make_atom("ga_comb.caveat.methods", "caveat", "attribution", None,
                  "Absolute SHAP and ALE are label-free — they explain the "
                  "meta-learner's own output — while PFI is label-based, measuring "
                  "the F1 drop when a detector's scores are shuffled."),
        make_atom("ga_comb.caveat.aggregation", "caveat", "markov", None,
                  "The overall weighting is the stationary distribution of a Markov "
                  "chain over the pairwise preferences of the three magnitude "
                  "measures."),
    ]
    # Run-dependent, unlike the two standing caveats above. A weakly supported
    # sign is still REPORTED — withholding it turned measured negatives into
    # blanks that read as missing data — so this is where the qualification
    # lives, naming the detectors and the reason for each.
    if weak:
        # Short on purpose. This is read as a limitation beside the findings,
        # not as prose, and the long form ("pushes the meta-learner one way
        # over part of its score range and the other way over the rest, so the
        # sign is only where the effect happens to end up") buried the two
        # detector names it exists to name.
        reason_text = {
            "low_consistency": "pushed the meta-learner both ways across {its} "
                               "score range",
            "weak_influence": "moved the meta-learner too little",
        }
        by_reason: Dict[str, List[str]] = {}
        for d in weak:
            by_reason.setdefault(" and ".join(support[d]), []).append(d)
        clauses = []
        for key, names in by_reason.items():
            its = "its" if len(names) == 1 else "their"
            why = _oxford([reason_text.get(r, r).format(its=its)
                           for r in key.split(" and ")])
            clauses.append(f"{_oxford(names)} {why}")
        caveats.append(make_atom(
            "ga_comb.caveat.sign_consistency", "caveat", "sign",
            {"weakly_supported": weak,
             "reasons": {d: support[d] for d in weak}},
            f"Weakly supported sign: {'; '.join(clauses)}. Keep that in mind "
            f"when reading it."))
    # Separate case, and a different statement: these detectors have no sign at
    # all, because ALE could not be computed for them or their net effect is
    # exactly zero.
    if unsigned:
        plural = "detector" if len(unsigned) == 1 else "detectors"
        caveats.append(make_atom(
            "ga_comb.caveat.sign_missing", "caveat", "sign",
            {"no_sign": unsigned},
            f"No sign for {_oxford(unsigned)}: the meta-learner's output ends "
            f"where it started as that {plural} sweeps its range, so there is "
            f"no net effect to take a sign from."))

    question = ("Which detectors does the GA-selected ensemble rely on most, and "
                "which way does each push the meta-learner's decision?")
    info_footer = (
        "Three measures rank how much weight a detector carries, each a position "
        "where rank 1 is strongest. Absolute SHAP ranks detectors by the size of "
        "their influence on the meta-learner's output; PFI by the F1 drop when "
        "their scores are shuffled; total ALE by how far the meta-learner's "
        "predicted anomaly probability moves in total as that detector sweeps "
        "its own observed range. Absolute SHAP and ALE are label-free, PFI is "
        "label-based. The overall weight rank is a fourth, separate position "
        "that merges the three. "
        "The sign is the sign of ALE's net accumulated effect, read as a "
        "detector's score rises from its lowest to its highest observed value: "
        "a detector whose rising score pushes the meta-learner toward flagging "
        "the point as an anomaly has a positive sign, and one whose rising "
        "score pushes it toward 'not an anomaly' has a negative sign. A sign is "
        "called weakly supported when the effect changes direction across the "
        "score range or when the detector barely moves the meta-learner at all.")

    return _envelope("ga_combination", dataset, entity, output, evidence, caveats,
                     required, question=question, info_footer=info_footer)


def build_rank_aggregation_ir(dataset: str, entity: str, stage_name: str, iteration: int,
                              result: Dict[str, Any], source_names: List[str],
                              source_top_picks: Dict[str, str],
                              full_ranking: List[str]) -> Dict[str, Any]:
    verdicts = result.get("verdicts", [])
    kendall_only = result.get("kendall_only")
    prefix = f"ra_{stage_name}"
    # Human-facing name for the consensus ("robust" → "robustness").
    stage_word = {"robust": "robustness", "final": "final"}.get(stage_name, stage_name)

    evidence: List[Dict[str, Any]] = []
    required: List[str] = []
    top = full_ranking[0] if full_ranking else NOT_AVAILABLE
    output = {
        "top_pick": top,
        "consensus_ranking_top_k": _top_k(full_ranking),
        "k": min(TOP_K, len(full_ranking)),
        "n_sources": len(source_names),
        "sources": sorted(source_names),
    }
    if full_ranking:
        # "ranking of detectors, first-ranked detector is X" — the winner reads
        # unmistakably as a DETECTOR (not one of the source rankings analysed
        # below, which the narrator had conflated), and grounding "first-ranked"
        # here means the narrator's natural "X ranked first" has the value 1 to
        # match instead of reading as an ungrounded number.
        evidence.append(make_atom(
            f"{prefix}.output.top", "stage_output", top, top,
            f"The {stage_word} consensus is a ranking of detectors; its "
            f"first-ranked detector is {top}.", order=0))
        required.append(f"{prefix}.output.top")

    caveats = [
        make_atom(f"{prefix}.caveat.consensus", "caveat", "aggregation", None,
                  "The consensus ranking is produced by Markov-chain rank aggregation "
                  "over the source rankings."),
    ]

    if kendall_only:
        # Two-source case (e.g. the final aggregation: robust consensus vs
        # Thompson). Influence (leave-one-out) and Borda are degenerate here —
        # dropping one source leaves a single source — so the per-source role
        # atoms are omitted entirely and a single AGREEMENT-driven sentence
        # carries the explanation: which source the consensus followed more.
        winner = kendall_only.get("winner")
        runner = kendall_only.get("runner_up")
        kid = f"{prefix}.kendall_only.winner"
        evidence.append(make_atom(
            kid, "kendall_only", str(winner),
            {"winner": winner,
             "winner_agreement": _val(kendall_only.get("winner_tau"), 4),
             "runner_up": runner,
             "runner_up_agreement": _val(kendall_only.get("runner_up_tau"), 4),
             "gap": _val(kendall_only.get("alignment_gap"), 4)},
            f"{winner} drove the {stage_word} consensus most: it agreed with the "
            f"consensus more closely than {runner} (agreement "
            f"{_fmt(kendall_only.get('winner_tau'), 4)} vs "
            f"{_fmt(kendall_only.get('runner_up_tau'), 4)}, gap "
            f"{_fmt(kendall_only.get('alignment_gap'), 4)})."))
        required.append(kid)
        caveats.append(make_atom(
            f"{prefix}.caveat.two_sources", "caveat", "loo", None,
            "With exactly two sources, influence (leave-one-out) and the combined "
            "(Borda) rank are undefined — dropping one leaves a single source — so "
            "agreement is the only meaningful diagnostic here."))
        question = (f"Which of the two sources did the {stage_word} consensus "
                    f"follow more closely?")
        # Footer is a pure glossary DEFINITION only; the two-source rationale
        # (why influence is undefined here) is owned by caveat.two_sources, so
        # keeping it out of the footer avoids stating it twice in the output.
        info_footer = (
            "Agreement compares the consensus ranking with a source's own ranking.")
    else:
        # Multi-source case: one human-readable role sentence per source, ordered
        # by Borda rank (the dominant combined rank), built from its two component
        # ranks — INFLUENCE (leave-one-out: how much the consensus moves when the
        # source is dropped) and AGREEMENT (Kendall tau of the source vs the
        # consensus) — plus its pattern. Raw LOO/tau scores stay in `value` for
        # provenance; the prose carries only the ranks.

        # Required relational atom: names the source set explicitly and states
        # that the ranked detectors (incl. the winner) are NOT sources — the
        # narrator had folded the winning detector into the list of sources.
        src_list = sorted(source_names)
        cid = f"{prefix}.context.sources"
        evidence.append(make_atom(
            cid, "stage_context", "sources",
            {"sources": src_list, "n_sources": len(src_list), "winner": top},
            f"The {len(src_list)} sources aggregated into this consensus are the "
            f"rankings {', '.join(src_list)}. Every fact below describes one of "
            f"these source rankings; the detectors they rank — including the "
            f"winner {top} — are the items being ranked, not sources.",
            order=5))
        required.append(cid)

        def _borda_key(v: Dict[str, Any]) -> Tuple[float, str]:
            br = v.get("borda_rank")
            return (float(br) if br is not None else float("inf"), str(v.get("source")))

        def _pattern_phrase(p: Any) -> str:
            """The pattern as prose. The underscore form is the enum, which
            stays in `value` for machine consumers; a snake_case token reads as
            code in a sentence written for a human."""
            if not p or p == NOT_AVAILABLE:
                return ""
            words = str(p).replace("_", " ")
            article = "an" if words[0].lower() in "aeiou" else "a"
            return f", {article} {words} pattern"

        # "shaped the consensus [Nth] most" — the ordinal is the source's
        # combined Borda standing: rank 1 → "most", 2 → "second most", … Ties
        # share an ordinal (two sources at Borda rank 3 are both "third most").
        _ORD_MOST = {1: "", 2: "second ", 3: "third ", 4: "fourth ", 5: "fifth ",
                     6: "sixth ", 7: "seventh ", 8: "eighth ", 9: "ninth ",
                     10: "tenth ", 11: "eleventh ", 12: "twelfth "}

        def _shaped_prefix(br: Any) -> str:
            if br is None or _is_nan(br):
                return ""
            return _ORD_MOST.get(int(br), f"{int(br)}th ")

        n_src = len(src_list)
        for i, v in enumerate(sorted(verdicts, key=_borda_key)):
            name = v["source"]
            loo_rank, align_rank = v.get("loo_rank"), v.get("align_rank")
            br = v.get("borda_rank")
            pp = _pattern_phrase(v.get("pattern"))
            # Every source is described exactly like the lead: how much it
            # "shaped the consensus" (its combined Borda standing) plus BOTH
            # explicit component ranks and its pattern. The combined standing
            # gets its own NAME and its own NUMBER ("overall standing rank N of
            # M"): expressed only as a bare verb phrase it was the one ordinal
            # in the sentence without a label, so narrators borrowed the nearest
            # rank-noun and reported it as influence — the same value described
            # twice, contradictorily ("ranked sixth for influence (rank 4)").
            standing = ("" if br is None or _is_nan(br)
                        else f" (overall standing rank {int(br)} of {n_src})")
            text = (f"{name} shaped the {stage_word} consensus "
                    f"{_shaped_prefix(br)}most{standing}, ranking "
                    f"{loo_rank} for influence and {align_rank} for agreement{pp}.")
            rid = f"{prefix}.source.{name}.role"
            evidence.append(make_atom(
                rid, "source_role", name,
                {"influence_rank": loo_rank, "agreement_rank": align_rank,
                 "borda_rank": v.get("borda_rank"), "pattern": v.get("pattern"),
                 "influence_score": _val(v.get("loo_score"), 4),
                 "agreement_score": _val(v.get("align_score"), 4),
                 "top_pick": source_top_picks.get(name, NOT_AVAILABLE)},
                text, order=10 * (i + 1)))
            required.append(rid)

        question = (f"Which source rankings most shaped the {stage_word} consensus, "
                    f"and how much did each agree with it?")
        info_footer = (
            "Influence measures how much a source moved the consensus: it compares "
            "the consensus ranking with the ranking that emerges when that source is "
            "left out. Agreement compares the consensus ranking with the source's own "
            "ranking. The overall standing rank is a third, separate position: it is "
            "influence and agreement merged into one Borda order, and it is what 'shaped "
            "the consensus most, second most, and so on' reports. A source can stand "
            "high overall on the strength of one component while ranking low on the "
            "other. An influential disagreer has high influence but low "
            "agreement; a redundant agreer has high agreement but low influence; a "
            "consistent source ranks similarly on both.")

    ir = _envelope(f"rank_aggregation_{stage_name}", dataset, entity, output,
                   evidence, caveats, required, question=question,
                   info_footer=info_footer)
    ir["iteration"] = int(iteration)
    return ir


def _mc_region_phrase(regions: Sequence[Any]) -> str:
    """Render win regions as prose. Ranges are written 'from A to B', never
    'A-B': the verifier's number extraction is sign-aware, so a hyphenated
    range would be read as the negative number -B and flagged unsupported."""
    spans = [(a, b) for a, b in regions if a != b]
    points = [a for a, b in regions if a == b]
    parts: List[str] = []
    if spans:
        parts.append(_oxford([f"from {_fmt(a)} to {_fmt(b)}" for a, b in spans]))
    if points:
        pts = _oxford([_fmt(p) for p in points])
        # Isolated grid points read as bare values; only prefix them with "at"
        # when they follow spans, so the two kinds stay distinguishable.
        parts.append(f"at {pts}" if spans else pts)
    return "at noise levels " + ", and ".join(parts)


def build_monte_carlo_ir(dataset: str, entity: str, result: Dict[str, Any],
                         ranked_f1: Optional[List[str]] = None,
                         ranked_pr: Optional[List[str]] = None) -> Dict[str, Any]:
    curves_f1 = result.get("curves_f1", {})
    curves_pr = result.get("curves_pr", {})
    winner_f1 = result.get("winner_f1", {})
    permodel_f1 = result.get("permodel_f1", {})

    evidence: List[Dict[str, Any]] = []
    required: List[str] = []
    output = {
        "production_ranking_f1_top_k": _top_k(ranked_f1 or []),
        "production_ranking_pr_top_k": _top_k(ranked_pr or []),
        "top_pick_f1": (ranked_f1[0] if ranked_f1 else NOT_AVAILABLE),
        "top_pick_pr": (ranked_pr[0] if ranked_pr else NOT_AVAILABLE),
    }
    top_f1 = ranked_f1[0] if ranked_f1 else None
    top_pr = ranked_pr[0] if ranked_pr else None
    if top_f1 or top_pr:
        if top_f1 and top_pr and top_f1 == top_pr:
            lead = (f"In the production Monte Carlo test, {top_f1} ranked first "
                    f"both by F1 score and by PR-AUC.")
        elif top_f1 and top_pr:
            lead = (f"In the production Monte Carlo test, {top_f1} ranked first "
                    f"by F1 score and {top_pr} ranked first by PR-AUC.")
        else:
            only, metric = (top_f1, "F1 score") if top_f1 else (top_pr, "PR-AUC")
            lead = (f"In the production Monte Carlo test, {only} ranked first "
                    f"by {metric}.")
        evidence.append(make_atom(
            "mc.output.top", "stage_output", str(top_f1 or top_pr),
            {"top_f1": top_f1 or NOT_AVAILABLE, "top_pr": top_pr or NOT_AVAILABLE},
            lead, order=1))
        required.append("mc.output.top")

    # ONE atom per detector, covering BOTH metrics. Two atoms about the same
    # detector (one per metric) is the same-subject-collapse trap: the narrator
    # states one and silently drops the other. Crossover atoms are dropped
    # entirely — a crossover is the derivative of the win regions, so emitting
    # both floods the prose with "at 0.042 ... at 0.053 ..." without adding a
    # single fact the regions do not already carry.
    regions_by_model: Dict[str, Dict[str, Any]] = {}
    for metric, curves in (("F1", curves_f1), ("PR-AUC", curves_pr)):
        for m, regions in sorted((curves.get("win_regions") or {}).items()):
            if regions:
                regions_by_model.setdefault(m, {})[metric] = regions

    wr_all = (winner_f1.get("win_rates") or {}) if isinstance(winner_f1, dict) else {}

    def _region_order(m: str) -> Any:
        cov = sum(abs(b - a) for rs in regions_by_model[m].values() for a, b in rs)
        return (-float(wr_all.get(m, 0.0) or 0.0), -cov, m)

    for i, m in enumerate(sorted(regions_by_model, key=_region_order)):
        per = regions_by_model[m]
        # Metric first: "won by F1 at ...; by PR-AUC at ..." keeps the two
        # metric clauses visibly distinct (the narrator otherwise copies one
        # metric's ranges into the other) and avoids repeating the preamble.
        clauses = [f"by {metric} {_mc_region_phrase(per[metric])}"
                   for metric in ("F1", "PR-AUC") if metric in per]
        wid = f"mc.win_region.{m}"
        evidence.append(make_atom(
            wid, "win_region", m,
            {metric: [(_val(a), _val(b)) for a, b in rs]
             for metric, rs in per.items()},
            f"{m} won {'; '.join(clauses)}.", order=10 + i))
        required.append(wid)

    conf: Dict[str, Any] = {}
    if winner_f1.get("feasible"):
        cv_acc = winner_f1.get("cv_accuracy", float("nan"))
        grade = fidelity_grade(cv_acc)
        conf["winner_surrogate_f1"] = {
            "train_accuracy": _val(winner_f1.get("train_accuracy"), 3),
            "cv_accuracy": _val(cv_acc, 3),
            "grade": grade,
        }
        wr = winner_f1.get("win_rates", {})
        winners = sorted(((m, r) for m, r in wr.items() if r > 0),
                         key=lambda kv: kv[1], reverse=True)
        top_wr, rest = winners[:TOP_K], winners[TOP_K:]
        if top_wr:
            wr_txt = ", ".join(f"{m} {_fmt(100.0 * r, 1)}%" for m, r in top_wr)
            # These are shares of the same trials, so they sum to 100% across
            # ALL winners. Listing only the top few left a reader adding up 96%
            # and looking for the bug; the tail is now stated instead of simply
            # missing. Kept as one clause rather than naming the stragglers,
            # which is what the cut is for.
            tail = sum(r for _, r in rest)
            tail_txt = ""
            if rest:
                tail_txt = (f"; the remaining {_fmt(100.0 * tail, 1)}% went to "
                            f"{len(rest)} further detector"
                            f"{'' if len(rest) == 1 else 's'}")
            evidence.append(make_atom(
                "mc.surrogate.win_rates", "surrogate_win_rates", "winner_surrogate",
                {"listed": [(m, _val(r, 3)) for m, r in top_wr],
                 "n_other": len(rest), "other_share": _val(tail, 3)},
                f"Across the noise sweep the trials were won by: {wr_txt}"
                f"{tail_txt}.", order=100))
            required.append("mc.surrogate.win_rates")
        # The winner-surrogate RULES are deliberately not emitted as evidence:
        # the tree is fitted on (noise level -> winner), so "the winner is X
        # when noise <= Y" restates the win regions above in weaker, fitted
        # form. Its held-out fidelity stays in `confidence` above.

    # Per-model held-out R² as confidence data, each graded for trust. When a
    # majority of a model's CV folds had (near-)constant test targets the
    # held-out estimate is not assessable — grade it not_available but keep the
    # computed number visible for transparency.
    permodel_cv: Dict[str, Any] = {}
    degenerate_models: List[str] = []
    for m, pm in sorted(permodel_f1.items()):
        n_splits = int(pm.get("cv_n_splits", 0) or 0)
        n_deg = int(pm.get("cv_degenerate_folds", 0) or 0)
        majority_degenerate = n_splits > 0 and n_deg > n_splits / 2
        entry = {"cv_r2": _val(pm.get("cv_r2"), 3),
                 "n_splits": n_splits, "n_degenerate_folds": n_deg,
                 "grade": NOT_AVAILABLE if majority_degenerate
                          else fidelity_grade(pm.get("cv_r2"))}
        permodel_cv[m] = entry
        if majority_degenerate:
            degenerate_models.append(m)
    if permodel_cv:
        conf["permodel_cv_r2"] = permodel_cv

    # Both the run-invariant notes — the sweep is explain-only, and it scores
    # with a fast point-wise proxy not comparable to production — now live in
    # the info footer (appended verbatim, not scored). Only run-specific caveats
    # (e.g. degenerate CV folds for this entity) stay in the caveats list.
    caveats: List[Dict[str, Any]] = []
    if degenerate_models:
        caveats.append(make_atom(
            "mc.caveat.cv_degenerate", "caveat", "confidence", degenerate_models,
            f"For {', '.join(degenerate_models)} most cross-validation folds had "
            f"(near-)constant F1 across the sweep, so the held-out R² is not a "
            f"meaningful fidelity estimate (marked not_available); the number is "
            f"kept only for transparency."))
    question = ("Which detector is most robust to noise, and does the best "
                "detector change as the noise level rises?")
    info_footer = (
        "The production Monte Carlo test injects Gaussian noise whose standard "
        "deviation is the noise level, fixed at 0.1, and ranks the detectors "
        "over 100 such runs. This sweep repeats that same injection across 20 "
        "noise levels from 0.0 to 0.2, five times each, only to show how the "
        "ranking behaves as noise grows — it is explanatory and never feeds "
        "model selection. The sweep scores with a fast point-wise best-threshold "
        "F1 and PR-AUC, whereas the production ranking uses a range-based metric, "
        "so sweep values are not directly comparable to production values. A win "
        "percentage is the share of sweep trials in which that detector scored "
        "best.")

    return _envelope("monte_carlo", dataset, entity, output, evidence, caveats,
                     required, confidence=conf, question=question,
                     info_footer=info_footer)


# Clause-shaped labels used inside a surrogate condition ("… when <label> is
# at most 0.3"), and bare noun forms used when a feature is named on its own.
_OFFBY_LABELS = {
    "position": "its position in the series",
    "local_volatility": "the local volatility",
    "boundary_distance": "the distance from the boundary",
}

_OFFBY_NOUNS = {
    "position": "the point's position in the series",
    "local_volatility": "the local volatility",
    "boundary_distance": "the distance from the boundary",
    "is_anomaly": "whether the point is a real anomaly",
}


def _offby_condition_phrase(conditions: List[Dict[str, Any]]) -> str:
    """Render simplified surrogate conditions as plain prose. `is_anomaly` is a
    0/1 label, so its 0.5 split becomes a statement rather than a comparison;
    the other three features keep their raw thresholds (grounded numbers)."""
    lower: Dict[str, Any] = {}
    upper: Dict[str, Any] = {}
    order: List[str] = []
    for c in conditions:
        f = str(c["feature"])
        if f not in order:
            order.append(f)
        if c["op"] == "<=":
            upper[f] = c["threshold"]
        else:
            lower[f] = c["threshold"]
    parts: List[str] = []
    for f in order:
        if f == "is_anomaly":
            parts.append("the point is not a real anomaly" if f in upper
                         else "the point is a real anomaly")
            continue
        label = _OFFBY_LABELS.get(f, f)
        if f in lower and f in upper:
            parts.append(f"{label} is between {lower[f]} and {upper[f]}")
        elif f in upper:
            parts.append(f"{label} is at most {upper[f]}")
        else:
            parts.append(f"{label} is above {lower[f]}")
    return " and ".join(parts) if parts else "always"


def build_off_by_ir(dataset: str, entity: str, result: Dict[str, Any],
                    ranked_f1_names: Optional[List[str]] = None) -> Dict[str, Any]:
    winner = result.get("winner", NOT_AVAILABLE)
    n_points = result.get("n_points", 0)
    surrogates = result.get("surrogates", {}) or {}
    per_comp: Dict[str, Dict[str, Any]] = surrogates.get("per_competitor", {}) or {}
    feature_names = list(surrogates.get("feature_names", []) or [])

    evidence: List[Dict[str, Any]] = []
    required: List[str] = []
    output = {
        "winner": winner,
        "production_ranking_top_k": _top_k(ranked_f1_names or []),
        "n_injected_points": int(n_points),
    }
    evidence.append(make_atom(
        "ob.output.winner", "stage_output", str(winner), winner,
        f"{winner} was the highest-ranked model of the off-by-threshold stage.",
        order=1))
    required.append("ob.output.winner")
    evidence.append(make_atom(
        "ob.points", "injected_points", "injection", int(n_points),
        f"{int(n_points)} borderline points were injected around the decision "
        f"boundary.", order=2))

    conf: Dict[str, Any] = {}
    caveats = [
        make_atom("ob.caveat.f1_side", "caveat", "scope", None,
                  "Correctness is judged on thresholded predictions (the F1 side); "
                  "PR-AUC has no per-point notion of correct or incorrect."),
    ]

    agg_imp: Dict[str, List[float]] = {fn: [] for fn in feature_names}
    # Rules are deduplicated across competitors: the same winner-only condition
    # often separates the winner from several rivals (e.g. all LOF variants),
    # and repeating it per competitor wastes prompt budget.
    rule_groups: Dict[str, Dict[str, Any]] = {}
    degenerate: List[str] = []
    per_comp_top: Dict[str, Any] = {}
    low_support: List[Any] = []
    # Competitors sharing an identical (count, rate) collapse into one atom:
    # with ~10 rivals most share the same single-win figure, and repeating it
    # per rival both burns budget and gives the narrator near-identical
    # sentences to shuffle model names between.
    wins_groups: Dict[Any, List[str]] = {}
    for k in sorted(per_comp.keys()):
        info = per_comp[k]
        n_w = info.get("n_exclusive_wins", 0)
        rate = info.get("exclusive_win_rate", 0.0)
        if info.get("degenerate"):
            degenerate.append(k)
            continue
        sup = support_grade(n_w)
        wins_groups.setdefault((int(n_w), _val(rate, 4)), []).append(k)

        clf = info.get("clf")
        if clf is not None:
            try:
                rules = tree_to_rules(clf, feature_names)
                pos_rules = [r for r in rules if str(r["outcome"]) == "1"]
                for rule in pos_rules:
                    # Simplify BEFORE dedup: chains that differ only in
                    # redundant bounds collapse to the same signature.
                    conds = simplify_conditions(rule["conditions"])
                    sig = json.dumps({"conditions": conds}, sort_keys=True)
                    grp = rule_groups.setdefault(
                        sig, {"rule": {"conditions": conds},
                              "competitors": [], "support": "adequate"})
                    grp["competitors"].append(k)
                    if sup == "low":
                        grp["support"] = "low"
            except Exception:
                pass

        imps = info.get("feature_importances", {})
        if imps:
            per_comp_top[k] = max(imps.items(), key=lambda kv: kv[1])
            for fn, im in imps.items():
                if fn in agg_imp:
                    agg_imp[fn].append(float(im))

        conf[f"surrogate_vs_{k}"] = {
            "train_accuracy": _val(info.get("train_accuracy"), 3),
            "cv_accuracy": _val(info.get("cv_accuracy"), 3),
            "grade": fidelity_grade(info.get("cv_accuracy")),
            "support": sup,
        }
        if sup == "low":
            low_support.append((k, int(n_w)))

    # ONE consolidated low-support caveat: repeating a near-identical sentence
    # per competitor is what pushes the narrator into compressing names
    # ("CBLOF_1 to -4"), which Rule 7 forbids. Per-competitor counts are
    # already carried by the exclusive-wins atoms, so nothing is lost.
    if len(low_support) == 1:
        k0, n0 = low_support[0]
        caveats.append(make_atom(
            "ob.caveat.support", "caveat", "support", {k0: n0},
            f"The rule for {k0} rests on only {n0} exclusive-win "
            f"point{'' if n0 == 1 else 's'} — fewer than the {N_CV_FOLDS} "
            f"cross-validation folds — so its held-out fidelity is unstable; "
            f"treat it as indicative."))
    elif low_support:
        caveats.append(make_atom(
            "ob.caveat.support", "caveat", "support",
            {k0: n0 for k0, n0 in low_support},
            f"The rules for {_oxford([k0 for k0, _ in low_support])} each rest "
            f"on fewer than {N_CV_FOLDS} exclusive-win points — fewer than the "
            f"{N_CV_FOLDS} cross-validation folds — so their held-out fidelity "
            f"is unstable; treat them as indicative."))

    # ── One atom per rival group: its rule(s) AND its win count together ──
    #
    # These used to be two families — a rule atom per condition signature and a
    # wins atom per (count, rate) — which expressed the SAME rivals twice, in
    # two different groupings and two different orders. That redundancy is what
    # let a narrator merge across them and emit one sentence carrying a rule
    # from one group and names from another. Grouping by (rules, count, rate)
    # means every rival set is named exactly once, with everything said about
    # it in a single sentence: there is no second sentence to cross-contaminate.
    comp_sigs: Dict[str, List[str]] = {}
    for sig in sorted(rule_groups):
        for c in rule_groups[sig]["competitors"]:
            if sig not in comp_sigs.setdefault(c, []):
                comp_sigs[c].append(sig)

    edges: Dict[Any, List[str]] = {}
    for key, names in wins_groups.items():
        for c in names:
            edges.setdefault((tuple(comp_sigs.get(c, ())), key), []).append(c)

    def _edge_order(item: Any) -> Any:
        """Biggest edge first; the rival name breaks ties deterministically."""
        (_sigs, (n_w, _rate)), names = item
        return (-int(n_w), sorted(names)[0])

    for ei, ((sigs, (n_w, rate)), names) in enumerate(
            sorted(edges.items(), key=_edge_order)):
        names = sorted(names)
        pct = _fmt(100.0 * float(rate or 0.0), 2)
        pts = f"{n_w} injected point{'' if n_w == 1 else 's'}"
        head = (f"{winner} correctly handles {pts} ({pct}%) that {names[0]} misses"
                if len(names) == 1 else
                f"{winner} correctly handles {pts} ({pct}%) apiece that "
                f"{_oxford(names)} each miss")
        phrases = [_offby_condition_phrase(rule_groups[s]["rule"]["conditions"])
                   for s in sigs]
        conditions = [rule_groups[s]["rule"]["conditions"] for s in sigs]
        them = "it" if len(names) == 1 else "them"
        if not phrases:
            text = f"{head}."
        elif "always" in phrases:
            text = f"{head}, beating {them} across all injected points."
        else:
            # Separate leaves of the surrogate tree are ALTERNATIVES. Joining
            # them with "and" (as _oxford does) both inverts the logic and, since
            # each condition already contains its own "and", collapses two rules
            # into one unreadable conjunction. "; or when" keeps them disjunctive
            # and makes the nesting legible.
            text = f"{head}, uniquely beating {them} when {'; or when '.join(phrases)}."
        support = ("low" if any(rule_groups[s]["support"] == "low" for s in sigs)
                   else "adequate")
        eid = f"ob.edge.{ei}"
        evidence.append(make_atom(
            eid, "exclusive_wins", str(winner),
            {"count": int(n_w), "rate": rate, "competitors": names,
             "conditions": conditions},
            text, confidence=support if sigs else None, order=10 + ei))
        required.append(eid)

    if degenerate:
        # Stated LAST and REQUIRED. It is the only place these names appear as a
        # group, and sitting next to the win atoms it became a ready-made trio
        # for a narrator to lift into them — while the negation it carried, the
        # hardest thing for a small model to keep, was dropped. Placed after the
        # summary, the win sentences are already written before these names are
        # seen at all, and stating it explicitly makes any such swap
        # self-contradictory within the narrative.
        evidence.append(make_atom(
            "ob.degenerate", "degenerate_comparison", str(winner),
            {"competitors": degenerate},
            f"{winner} never exclusively beat {_oxford(degenerate)}, so "
            f"{'none of them appears' if len(degenerate) > 1 else 'it does not appear'} "
            f"above.", order=500))
        required.append("ob.degenerate")

    mean_imp = {fn: float(np.mean(v)) for fn, v in agg_imp.items() if v}
    top = (max(mean_imp.items(), key=lambda kv: kv[1])
           if mean_imp and any(mean_imp.values()) else None)
    # A per-competitor importance atom only earns its place when its driver
    # DIFFERS from the overall one; otherwise it restates the summary below.
    # Rivals sharing that driver AND its rounded value collapse into one atom:
    # four near-identical sentences differing only in a model name are exactly
    # what a narrator shuffles names between.
    imp_groups: Dict[Any, List[str]] = {}
    for k in sorted(per_comp_top):
        feat, imp = per_comp_top[k]
        if top is not None and feat == top[0]:
            continue
        imp_groups.setdefault((feat, _fmt(imp, 2)), []).append(k)
    for ii, (feat, shown) in enumerate(sorted(imp_groups,
                                              key=lambda kv: (kv[0], kv[1]))):
        ks = sorted(imp_groups[(feat, shown)])
        against = _oxford(ks)
        each = "" if len(ks) == 1 else " in each case"
        evidence.append(make_atom(
            f"ob.vs.{ks[0]}.importance", "feature_importance",
            ks[0] if len(ks) == 1 else "competitors",
            {"feature": feat, "importance": float(shown), "competitors": ks},
            f"Against {against}, the property that best separates those points "
            f"is {_OFFBY_NOUNS.get(feat, feat)} (importance {shown}{each}).",
            order=300 + ii))

    if top is not None:
        evidence.append(make_atom(
            "ob.summary.top_feature", "summary", top[0], _val(top[1], 3),
            f"Across all competitors, {winner}'s edge is best explained by "
            f"{_OFFBY_NOUNS.get(top[0], top[0])} (mean importance "
            f"{_fmt(top[1], 2)}).", order=400))
        required.append("ob.summary.top_feature")

    question = ("Which model handled the injected borderline points best, and "
                "what distinguishes the points it uniquely got right?")
    info_footer = (
        "The surrogate rules describe each injected point with four properties. "
        "is_anomaly is the point's true label - 1 for a real anomaly, 0 for a "
        "normal point. boundary_distance is how far the point was scaled away "
        "from the decision boundary, so 0 sits exactly on it. local_volatility "
        "is the standard deviation of the series around the injection site - how "
        "noisy that neighbourhood is. position is where the point falls in the "
        "series, from 0 at the start to 1 at the end. An exclusive win is a "
        "point the highest-ranked model classified correctly and the named rival "
        "did not; the rules come from a decision tree fitted to those wins, so "
        "they describe where the edge occurred, not why.")

    return _envelope("off_by_threshold", dataset, entity, output, evidence,
                     caveats, required, confidence=conf, question=question,
                     info_footer=info_footer)


# ── Global assembly ──────────────────────────────────────────────────────────

_STAGE_FILES = {
    "thompson_sampling": "ir_thompson",
    # The ranking-criterion sibling. Deliberately absent from stage_picks below:
    # both Thompson stages report the same winner (they share rank_models), so
    # counting it again would double-weight Thompson in the cross-stage consensus.
    "thompson_ranking": "ir_thompson_ranking",
    "ga_selection": "ir_ga_selection",
    "ga_combination": "ir_ga_combination",
    "monte_carlo": "ir_monte_carlo",
    "off_by_threshold": "ir_off_by",
}


def assemble_global_ir(results_dict: Dict[str, Any], dataset: str, entity: str,
                       iteration: int,
                       base_dir: str = "myresults/explanations_ir") -> str:
    """
    Combine the per-stage IR JSONs (written by each explainability orchestrator)
    with the pipeline's decision context into ir_global_iter{iteration}.json.
    Missing stage files → explicit not_available; GAN is always reserved
    not_available (no explainability layer implemented).

    Besides the machine-readable blocks (decision / stage_agreement / stages),
    the global IR carries its own `evidence` atoms — pre-rendered SENTENCES
    (the decision, one summary per available stage, the GA ensemble-membership
    relation, per-stage agreement) — plus `required_atom_ids`, so the global
    narrative is prompted from canonical sentences rather than key:value dumps
    and its omissions are measurable like any stage's.
    """
    directory = os.path.join(base_dir, str(dataset), str(entity))

    def _load(fname: str, pattern: Optional[str] = None) -> Optional[Dict[str, Any]]:
        path = os.path.join(directory, f"{fname}.json")
        if not os.path.exists(path) and pattern:
            # Iteration-number mismatches between pipeline phases should not
            # silently drop a stage — fall back to the newest matching file.
            candidates = _glob.glob(os.path.join(directory, pattern))
            if candidates:
                path = max(candidates, key=os.path.getmtime)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    stages: Dict[str, Any] = {}
    loaded_docs: Dict[str, Dict[str, Any]] = {}
    all_caveats: List[Dict[str, Any]] = []
    stage_files = {stage: (fname, None) for stage, fname in _STAGE_FILES.items()}
    stage_files["rank_aggregation_robust"] = (
        f"ir_rank_aggregation_robust_{iteration}", "ir_rank_aggregation_robust_*.json")
    stage_files["rank_aggregation_final"] = (
        f"ir_rank_aggregation_final_{iteration}", "ir_rank_aggregation_final_*.json")
    for stage, (fname, pattern) in sorted(stage_files.items()):
        loaded = _load(fname, pattern)
        if loaded is None:
            stages[stage] = {"status": NOT_AVAILABLE}
        else:
            loaded_docs[stage] = loaded
            stages[stage] = {"status": "ok", "output": loaded.get("output", {})}
            all_caveats.extend(loaded.get("caveats", []))
    stages["gan"] = {"status": NOT_AVAILABLE,
                     "note": "no explainability layer implemented for the GAN test"}

    fd = results_dict.get("final_decision", {}) or {}
    choice = fd.get("framework_choice", NOT_AVAILABLE)
    ens_f1 = fd.get("ensemble_f1", float("nan"))
    sng_f1 = fd.get("single_model_f1", float("nan"))
    margin = (ens_f1 - sng_f1) if not (_is_nan(ens_f1) or _is_nan(sng_f1)) else float("nan")
    if choice == "ensemble":
        reason = (f"The ensemble was chosen because its F1 ({_fmt(ens_f1, 4)}) is greater "
                  f"than or equal to the best single model's F1 ({_fmt(sng_f1, 4)}).")
    elif choice == "single_model":
        reason = (f"The single model was chosen because its F1 ({_fmt(sng_f1, 4)}) exceeds "
                  f"the ensemble's F1 ({_fmt(ens_f1, 4)}).")
    else:
        reason = NOT_AVAILABLE
    decision = {
        "framework_choice": choice,
        "chosen": _py(fd.get("chosen_model", NOT_AVAILABLE)),
        "ensemble": _py(fd.get("ensemble", [])),
        "ensemble_f1": _val(ens_f1, 4),
        "ensemble_pr_auc": _val(fd.get("ensemble_pr_auc"), 4),
        "single_model": fd.get("single_model", NOT_AVAILABLE),
        "single_model_f1": _val(sng_f1, 4),
        "single_model_pr_auc": _val(fd.get("single_model_pr_auc"), 4),
        "f1_margin_ensemble_minus_single": _val(margin, 4),
        "reason": reason,
    }

    # Cross-stage top-pick agreement (single-branch stages vs the final single pick).
    single_pick = fd.get("single_model", NOT_AVAILABLE)
    stage_picks = {
        "thompson": (results_dict.get("thompson", {}) or {}).get("best_model", NOT_AVAILABLE),
        "gan": (results_dict.get("gan_robustness", {}) or {}).get("best_model", NOT_AVAILABLE),
        "borderline": (results_dict.get("borderline", {}) or {}).get("best_model", NOT_AVAILABLE),
        "monte_carlo": (results_dict.get("monte_carlo", {}) or {}).get("best_model_f1", NOT_AVAILABLE),
    }
    agg = results_dict.get("aggregation", {}) or {}
    # Only the robust consensus is compared. The final consensus IS the source
    # of the single-model pick, so asking whether they agree is tautological —
    # it would always report agreement and tell the reader nothing.
    stage_picks["robust_consensus"] = _top_of_ranking(agg.get("robust_agg"))
    agreement = {
        name: {"top_pick": _py(pick),
               "agrees_with_final_single": (pick == single_pick)
               if pick not in (NOT_AVAILABLE, "N/A") else NOT_AVAILABLE}
        for name, pick in sorted(stage_picks.items())
    }

    seen = set()
    caveats = []
    for c in all_caveats:
        if c.get("id") not in seen:
            seen.add(c.get("id"))
            caveats.append(c)

    # ── Global evidence atoms (canonical sentences for the narrative) ────────
    evidence: List[Dict[str, Any]] = []
    required: List[str] = []
    ens = list(fd.get("ensemble", []) or [])
    single = fd.get("single_model", NOT_AVAILABLE)
    if choice == "ensemble":
        dec_text = (f"The final decision is the ensemble {{{', '.join(ens)}}} "
                    f"(F1 {_fmt(ens_f1, 4)}), chosen over the best single model "
                    f"{single} (F1 {_fmt(sng_f1, 4)}).")
    elif choice == "single_model":
        dec_text = (f"The final decision is the single model {single} "
                    f"(F1 {_fmt(sng_f1, 4)}), chosen over the GA ensemble "
                    f"(F1 {_fmt(ens_f1, 4)}).")
    else:
        dec_text = "The final framework decision is not available."
    evidence.append(make_atom("global.decision", "decision", str(choice),
                              {"framework_choice": choice,
                               "ensemble_f1": _val(ens_f1, 4),
                               "single_model_f1": _val(sng_f1, 4)},
                              dec_text))
    required.append("global.decision")

    # One summary atom per available stage: its own canonical output sentences
    # (stage_output + stage_context atoms), so the global narrative is composed
    # from the same verified sentences the stage narratives use.
    for stage, doc in sorted(loaded_docs.items()):
        req_ids = set(doc.get("required_atom_ids", []))
        texts = [a["text"] for a in doc.get("evidence", [])
                 if a.get("type") in ("stage_output", "stage_context")
                 and a.get("id") in req_ids]
        if not texts:
            texts = [a["text"] for a in doc.get("evidence", [])[:1]]
        if not texts:
            continue
        sid = f"global.stage.{stage}"
        evidence.append(make_atom(sid, "stage_summary", stage,
                                  doc.get("output", {}), " ".join(texts)))
        required.append(sid)

    for name, info in sorted(agreement.items()):
        pick = info.get("top_pick")
        agrees = info.get("agrees_with_final_single")
        if agrees is NOT_AVAILABLE or agrees == NOT_AVAILABLE:
            continue
        verb = "matches" if agrees else "differs from"
        evidence.append(make_atom(
            f"global.agreement.{name}", "stage_agreement", name,
            {"top_pick": pick, "agrees": agrees},
            f"{name}'s top pick ({pick}) {verb} the final single-model pick "
            f"({single})."))

    global_ir = {
        "ir_version": IR_VERSION,
        "stage": "global",
        "dataset": str(dataset),
        "entity": str(entity),
        "iteration": int(iteration),
        "decision": decision,
        "stage_agreement": agreement,
        "stages": stages,
        "evidence": [_py(a) for a in sorted(evidence, key=lambda a: a["id"])],
        "required_atom_ids": sorted(required),
        "caveats": sorted(caveats, key=lambda c: c.get("id", "")),
    }
    return write_stage_ir(global_ir, dataset, entity, f"ir_global_iter{iteration}",
                          base_dir=base_dir)
