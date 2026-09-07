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
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Utils.pipeline_spec import (decision_metric_formula, decision_metric_label,
                                 metric_weights, ranking_metrics_for)

import numpy as np

IR_VERSION = "1.0"
NOT_AVAILABLE = "not_available"
TOP_K = 5

# How many items of a ranked series are part of the answer rather than a
# refinement of it. The head stays required, the tail is offered. Nothing is
# hidden either way: WebUI.summarize tables come from the IR, not the prose.
HEAD_REQUIRED = 3

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

_ID_DIGITS = re.compile(r"(\d+)")


def _id_key(atom_id: Any) -> Tuple[Any, ...]:
    """Sort ids by number where they carry one: plain string order files
    `regime.10` between `regime.1` and `regime.2`."""
    return tuple(int(p) if p.isdigit() else p
                 for p in _ID_DIGITS.split(str(atom_id)))


def _envelope(stage: str, dataset: str, entity: str, output: Dict[str, Any],
              evidence: List[Dict[str, Any]], caveats: List[Dict[str, Any]],
              required_atom_ids: List[str],
              confidence: Optional[Dict[str, Any]] = None,
              question: Optional[str] = None) -> Dict[str, Any]:
    env = {
        "ir_version": IR_VERSION,
        "stage": stage,
        "dataset": str(dataset),
        "entity": str(entity),
        "output": _py(output),
        "evidence": [_py(a) for a in sorted(evidence, key=lambda a: _id_key(a["id"]))],
        "caveats": [_py(a) for a in sorted(caveats, key=lambda a: _id_key(a["id"]))],
        "required_atom_ids": sorted(required_atom_ids),
        "confidence": _py(confidence or {}),
    }
    # `question` frames the narration prompt (the stage's headline question).
    if question is not None:
        env["question"] = str(question)
    return env


def write_stage_ir(ir: Dict[str, Any], dataset: str, entity: str, filename: str,
                   base_dir: str = "myresults/explanations_ir") -> str:
    directory = os.path.join(base_dir, str(dataset), str(entity))
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{filename}.json")
    with open(path, "w") as f:
        json.dump(ir, f, sort_keys=True, indent=2)
    return path


def _join_and(items: Sequence[str]) -> str:
    """['a'] -> 'a'; ['a','b'] -> 'a and b'; ['a','b','c'] -> 'a, b and c'."""
    parts = [str(i) for i in items]
    if len(parts) < 3:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


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


def _full_ranking(value: Any) -> List[str]:
    """The whole detector ordering, from the same shapes `_top_of_ranking` takes.

    The agreement strip shows only each source's winner, which cannot say
    whether a disagreeing source ranked the consensus pick second or last. The
    ordering answers that, so it travels beside the pick.
    """
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if (len(value) > 1 and not isinstance(value[0], str)
                and isinstance(value[1], (list, tuple))):
            return _full_ranking(value[1])
        # A list of names, or a list of (name, score) rows.
        out: List[str] = []
        for item in value:
            name = _top_of_ranking(item)
            if name != NOT_AVAILABLE:
                out.append(str(name))
        return out
    return []


# ── Stage builders ───────────────────────────────────────────────────────────

def _ts_context_feature_label(idx: Any, context_feature_names: Optional[Sequence[str]]) -> str:
    """Name a context feature when the dataset supplies names, else fall back to its
    index. Names come from the loader, so datasets without column headers
    (SMD, SMAP/MSL) keep the numeric form."""
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return str(idx)
    if context_feature_names and 0 <= i < len(context_feature_names) and str(context_feature_names[i]).strip():
        return str(context_feature_names[i]).strip()
    return f"context feature {i}"


def build_thompson_ir(dataset: str, entity: str, *, n_windows: int,
                      final_ranking: List[Tuple[str, float]],
                      regimes: List[Dict[str, Any]],
                      shifts: List[Dict[str, Any]],
                      blip_count: int,
                      state_fractions: Dict[str, float],
                      final_state: str,
                      state_counts: Optional[Dict[str, int]] = None,
                      context_feature_names: Optional[Sequence[str]] = None,
                      n_context_features: Optional[int] = None) -> Dict[str, Any]:
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
    Every context feature list arrives pre-split by sign so no consumer ever has to
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
        return _ts_context_feature_label(idx, context_feature_names)

    top_pairs = _top_k(final_ranking)
    counts: Dict[str, int] = {}
    spans: Dict[str, int] = {}
    for r in regimes:
        if not r.get("leader"):
            continue
        lname = str(r.get("leader"))
        counts[lname] = counts.get(lname, 0) + 1
        try:
            spans[lname] = spans.get(lname, 0) + int(r.get("duration") or 0)
        except (TypeError, ValueError):
            spans.setdefault(lname, 0)

    # This stage explains the expected reward mu^T x, so its headline is the
    # detector that held the highest expected reward longest — not the ranking
    # by ||mu||^2, which is the sibling card's subject and was this atom's score.
    # Carrying it here left the one sentence answering "which detector had the
    # highest chance of being chosen" reporting a quantity that does not bear on
    # the chance, and on SKAB/7 naming LOF_3 while the regime summary three
    # sentences later gave CBLOF_4 nearly twice as many windows.
    ranked_spans = sorted(spans.items(), key=lambda kv: (-kv[1], kv[0]))
    top_model = (ranked_spans[0][0] if ranked_spans
                 else (top_pairs[0][0] if top_pairs else NOT_AVAILABLE))
    output = {
        "top_pick": top_model,
        "final_ranking_top_k": [{"model": m, "score": _val(s, 6)} for m, s in top_pairs],
        "n_windows": int(n_windows),
        "n_regimes": len(regimes),
    }

    # ── Lead: who was best by expected reward, and for how much of the run ──
    if ranked_spans:
        held = ranked_spans[0][1]
        tied = [m for m, w in ranked_spans if w == held]
        lead_val = {"top": top_model, "windows_led": held,
                    "n_windows": int(n_windows), "tied_with": tied[1:]}
        if len(tied) > 1:
            # "more than any other" is false when there is no other.
            more = "" if len(tied) == len(ranked_spans) else \
                ", more than any other detector"
            lead_txt = (f"{_oxford(tied)} each held the highest expected reward "
                        f"in {held} of the {int(n_windows)} windows{more}.")
        else:
            lead_txt = (f"{top_model} held the highest expected reward in {held} "
                        f"of the {int(n_windows)} windows, more than any other "
                        f"detector")
            if len(ranked_spans) > 1:
                runner, r_held = ranked_spans[1]
                lead_val.update({"runner_up": runner, "runner_up_windows": r_held})
                lead_txt += f" — {runner} held it in {r_held}"
            lead_txt += "."
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
        # Windows held, not regime count, and sorted by it — the same reasoning
        # as the ranking stage's summary: four short spells are not more of the
        # run than one long one, and a bare count reads as though they were.
        ordered = sorted(counts.items(), key=lambda kv: (-spans.get(kv[0], 0), kv[0]))
        led = _oxford([
            f"{m} led {c} regime{'' if c == 1 else 's'}, spanning "
            f"{spans.get(m, 0)} window{'' if spans.get(m, 0) == 1 else 's'}"
            for m, c in ordered])
        blips = int(blip_count or 0)
        blip_txt = ("" if not blips else
                    f" {blips} brief blip window{'' if blips == 1 else 's'} "
                    f"did not last long enough to count as a regime.")
        evidence.append(make_atom(
            "ts.regimes.summary", "regime_summary", "regimes",
            {"n_regimes": len(regimes), "n_windows": int(n_windows),
             "n_leaders": len(counts), "regimes_led": counts,
             "windows_led": spans, "blip_count": blips},
            f"The {int(n_windows)} windows split into {len(regimes)} regimes led "
            f"by {len(counts)} different detectors: {led}.{blip_txt}", order=3))
        required.append("ts.regimes.summary")

    # ── One sentence per regime, chronological; every one required ──
    for i, r in enumerate(sorted(regimes, key=lambda x: x.get("index", 0))):
        idx = r.get("index", i)
        leader = str(r.get("leader", NOT_AVAILABLE))
        # Both narrated context feature facts are now slices of the same total — the raw
        # split of mu.x, whose parts sum to the prediction. `edge_favor_leader`
        # is that same split differenced against the runner-up, so its parts sum
        # to the leader's margin in expected reward. Putting the edge in SHAP's
        # units made one sentence carry two incomparable quantities.
        supplying = [c for c, _ in (r.get("reward_raising") or [])][:2]
        favor = [c for c, _ in (r.get("edge_favor_leader") or [])][:1]
        runner = r.get("runner_up")

        # The claims stay distinct clauses, in parallel participles. They are
        # different quantities: a SHARE of the expected reward, an EDGE over the
        # runner-up in those same units, and a DEPARTURE from what the context
        # feature usually contributes, which is not a share of anything.
        #
        # The departure leaves this sentence for an atom of its own, below: three
        # participles were one too many, and the narrator elided the second
        # supplying feature in 7 of 8 regimes on SKAB/7.
        deviating = ((r.get("shap_raising") or []) + (r.get("shap_lowering") or []))
        worst = (max(deviating, key=lambda cv: abs(cv[1]) if not _is_nan(cv[1]) else -1)
                 if deviating else None)
        clauses: List[str] = []
        if supplying:
            clauses.append(f"{_oxford([_ch(c) for c in supplying])} raising its "
                           f"expected reward the most")
        if favor and runner:
            # One context feature often does both jobs; "also" says so rather than
            # presenting the same context feature twice as two separate findings.
            also = "also " if favor[0] in supplying else ""
            clauses.append(f"{_ch(favor[0])} {also}giving it its biggest edge "
                           f"over {runner}")

        text = (f"Regime {idx} (windows {r.get('start')} to {r.get('end')}, "
                f"{r.get('duration')} windows) was led by {leader}")
        if len(clauses) == 2 and any(" and " in c for c in clauses):
            # _oxford drops the serial comma for two items, which collides with
            # the "context feature 8 and context feature 3" inside the first clause and yields
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

        # Carries "regime N" so the disclosure can file it; the id keeps the
        # suffix so artifacts._REGIME_RE, which anchors on the index, does not
        # read it as an extra regime.
        if worst is not None:
            c, v = worst
            direction = "above" if (not _is_nan(v) and float(v) >= 0) else "below"
            did = f"{rid}.deviation"
            evidence.append(make_atom(
                did, "regime", leader,
                {"index": idx, "leader": leader,
                 "deviation_raising": [(c, _val(v, 4)) for c, v in (r.get("shap_raising") or [])],
                 "deviation_lowering": [(c, _val(v, 4)) for c, v in (r.get("shap_lowering") or [])]},
                f"In regime {idx}, {_ch(c)} departed furthest from its usual "
                f"contribution, running {direction} it.", order=10 + i))
            required.append(did)

    # ── Which context feature carried the winner, across the regimes it led ──
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

        # Window counts only when the sampler's own tallies were passed. They
        # could be derived from the shares, but a share times a window count is
        # a rounded number, and a rounded number stated as a fact is exactly
        # what this layer exists to avoid. Without them the sentence falls back
        # to shares alone.
        have_counts = bool(state_counts) and all(
            s in state_counts for s, _f in ordered_states)
        parts = []
        for i, (s, f) in enumerate(ordered_states):
            name, share = s.replace("_", " "), _fmt(100.0 * f, 1)
            if not have_counts:
                parts.append(f"{name} {share}% of the time")
                continue
            n = int(state_counts[s])
            # "windows" on the first item only; after that the unit is carried,
            # and repeating it turns a three-item list into a stutter.
            unit = (f" window{'' if n == 1 else 's'}" if i == 0 else "")
            parts.append(f"{name} for {n}{unit} ({share}%)")
        frac_txt = _oxford(parts)
        value: Dict[str, Any] = dict(state_fractions)
        if have_counts:
            value = {"fractions": dict(state_fractions),
                     "windows": {s: int(state_counts[s]) for s, _f in ordered_states}}
        evidence.append(make_atom(
            "ts.states.summary", "behavior_summary", "selection_states", value,
            f"Over the {int(n_windows)} windows the sampler was in {frac_txt}.",
            order=200))
        required.append("ts.states.summary")

    caveats: List[Dict[str, Any]] = []
    if n_context_features is not None and int(n_context_features) == 1:
        caveats.append(make_atom(
            "ts.caveat.single_channel", "caveat", "context features", int(n_context_features),
            "This dataset has a single context feature, so splitting a detector's "
            "expected reward across context features carries no information — that one "
            "context feature necessarily accounts for all of it."))

    # Not "why did it rank the winner first" any more: the ranking criterion has
    # its own stage (build_thompson_ranking_ir), and this one never explained it
    # — its regimes, its SHAP and its context features are all about the per-window
    # expected reward that drove selection. The question now names what the
    # evidence below actually answers.
    question = ("When did each detector have the highest chance of being chosen "
                "— which context features raised its expected reward above its rivals, "
                "and how much of the run was spent exploring rather than "
                "exploiting?")

    return _envelope("thompson_sampling", dataset, entity, output, evidence,
                     caveats, required, question=question)


def build_thompson_ranking_ir(dataset: str, entity: str, *, n_windows: int,
                              final_ranking: List[Tuple[str, float]],
                              winner_context_features: List[Tuple[Any, float]],
                              gap_context_features: List[Tuple[Any, float]],
                              selection_counts: Dict[str, int],
                              regimes: List[Dict[str, Any]],
                              warmup_windows: int,
                              context_feature_names: Optional[Sequence[str]] = None,
                              n_context_features: Optional[int] = None,
                              context_feature_shares: Optional[Dict[str, List[float]]] = None
                              ) -> Dict[str, Any]:
    """
    The sibling of build_thompson_ir, for the ranking criterion rather than the
    selection dynamics.

    Thompson Sampling ranks detectors by ||mu_k||^2, but build_thompson_ir
    explains mu^T x — the expected reward that drove per-window selection. This
    builder explains the ranking itself: ||mu||^2 splits exactly into one
    non-negative contribution per context feature, and the winner's margin over the
    runner-up splits exactly into one signed term per context feature.

    `winner_context_features` is [(context feature, contribution)] for the winner, descending;
    `gap_context_features` is [(context feature, signed delta)] for winner-minus-runner-up,
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
        return _ts_context_feature_label(idx, context_feature_names)

    top_pairs = _top_k(final_ranking)
    top_model = top_pairs[0][0] if top_pairs else NOT_AVAILABLE
    runner_up = top_pairs[1][0] if len(top_pairs) > 1 else None
    output = {
        "top_pick": top_model,
        "final_ranking_top_k": [{"model": m, "score": _val(s, 6)} for m, s in top_pairs],
        "n_windows": int(n_windows),
        "n_regimes": len(regimes),
        "n_channels": None if n_context_features is None else int(n_context_features),
        "warmup_windows": int(warmup_windows or 0),
    }

    # ── Lead: the ranking this stage exists to explain ──
    if top_pairs:
        score = top_pairs[0][1]
        if runner_up is not None and not _is_nan(score) and not _is_nan(top_pairs[1][1]):
            margin = float(score) - float(top_pairs[1][1])
            lead_val = {"top": top_model, "score": _val(score, 6),
                        "runner_up": runner_up, "margin": _val(margin, 6)}
            lead_txt = (f"Ranked first by the size of its mean vector, {top_model} "
                        f"scored {_fmt(score, 6)}, ahead of {runner_up} by "
                        f"{_fmt(margin, 6)}.")
        else:
            lead_val = {"top": top_model, "score": _val(score, 6)}
            lead_txt = (f"Ranked first by the size of its mean vector, {top_model} "
                        f"scored {_fmt(score, 6)}.")
        evidence.append(make_atom("tsr.output.top", "stage_output", str(top_model),
                                  lead_val, lead_txt, order=1))
        required.append("tsr.output.top")

    # ── Where the winner's score came from ──
    total = sum(float(v) for _, v in winner_context_features if not _is_nan(v))
    if winner_context_features and total > 0:
        lead_ch = winner_context_features[:3]
        shares = [(c, 100.0 * float(v) / total) for c, v in lead_ch]
        listed = _oxford([f"{_ch(c)} ({_fmt(p, 1)}%)" for c, p in shares])
        evidence.append(make_atom(
            "tsr.winner.channels", "winner_channels", str(top_model),
            {"channel": lead_ch[0][0], "total": _val(total, 6),
             "per_channel": [(c, _val(v, 6)) for c, v in winner_context_features],
             "top_shares": [(c, _val(p, 1)) for c, p in shares]},
            f"{listed} contributed the majority of {top_model}'s score.",
            order=10))
        required.append("tsr.winner.channels")

    # ── What actually decided the top spot ──
    if gap_context_features and runner_up:
        gains = [(c, v) for c, v in gap_context_features if not _is_nan(v) and float(v) > 0][:2]
        losses = [(c, v) for c, v in gap_context_features if not _is_nan(v) and float(v) < 0][:1]
        parts = []
        if gains:
            # No raw values in the prose. The table and the gap plot carry the
            # per-context-feature numbers, and quoting six-decimal figures mid-sentence
            # only invites the narrator to gloss what they mean.
            labels = _oxford([_ch(c) for c, _v in gains])
            parts.append(f"{labels[:1].upper() + labels[1:]} contributed "
                         f"significantly to the lead that {top_model} had over "
                         f"{runner_up}")
        if losses:
            c, _v = losses[0]
            # Direction in words, with BOTH sides named ("favoured X more than
            # Y") and no signed number. A signed value plus "in X's favour" is
            # read the other way round often enough to invert the one claim
            # this stage exists to make.
            parts.append(f"while {_ch(c)} favoured {runner_up} more than "
                         f"{top_model}")
        if parts:
            evidence.append(make_atom(
                "tsr.gap.runner_up", "rank_gap", str(top_model),
                # `rivals` is one of verifier._RIVAL_KEYS, so naming the key this
                # way gets the rival set checked against the narrated sentence.
                {"rivals": [runner_up], "runner_up": runner_up,
                 "per_channel": [(c, _val(v, 6)) for c, v in gap_context_features],
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
                    f" The first {warm} windows are left out, because all "
                    f"detectors start with score zero.")
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
        longest = max(spans, key=lambda m: (spans[m], str(m))) if spans else None
        evidence.append(make_atom(
            "tsr.regimes.summary", "regime_summary", "regimes",
            {"n_regimes": len(regimes), "n_windows": int(n_windows),
             "n_leaders": len(counts), "regimes_led": counts,
             "windows_led": spans, "warmup_windows": warm,
             "longest_leader": longest},
            head + warm_txt, order=40))
        required.append("tsr.regimes.summary")

        # The score accumulates only in the windows a detector is picked, so
        # whoever led longest need not finish first.
        #
        # Strictly longer, not merely a different name: on a tie the winner is
        # as much "in front for longest" as anyone, and the claim would be false.
        if longest is not None and top_model != NOT_AVAILABLE and \
                str(longest) != str(top_model) and \
                spans[longest] > spans.get(top_model, 0):
            evidence.append(make_atom(
                "tsr.tension.led_vs_won", "stage_tension", str(top_model),
                {"winner": top_model, "longest_leader": longest,
                 "winner_windows": spans.get(top_model, 0),
                 "longest_windows": spans[longest]},
                f"{top_model} finished first on this score without being the "
                f"detector in front for longest — {longest} led for more "
                f"windows.", order=41))
            required.append("tsr.tension.led_vs_won")

    # ── One sentence per regime, chronological; every one required ──
    for i, r in enumerate(sorted(regimes, key=lambda x: x.get("index", 0))):
        idx = r.get("index", i)
        leader = str(r.get("leader", NOT_AVAILABLE))
        top_context_features = r.get("top_channels") or []
        runner = r.get("runner_up")

        dur = r.get("duration")
        # ONE sentence, opening with the literal "Regime N (windows ...)". The
        # number is what pairs this sentence with its own figure in the page's
        # regime disclosure, and a trailing clause split into a sentence of its
        # own would carry neither the number nor a detector name.
        #
        # No "ahead of {runner_up}" tail. The narrator drops that clause from
        # every regime sentence, and because coverage is conjunctive the atom
        # then only passes when the runner-up happens to be named elsewhere in
        # the narrative — true for the NN_* that lead most regimes, false for
        # the odd early one, so a run's faithfulness turned on which detector
        # placed second in regime 0. The runner-up stays in `value`, and the
        # per-regime figure plots it beside the leader, which is where a
        # comparison belongs anyway.
        text = (f"Regime {idx} (windows {r.get('start')} to {r.get('end')}, "
                f"{dur} window{'' if dur == 1 else 's'}) was led by {leader}")
        if top_context_features:
            labels = _oxford([_ch(c) for c, _ in top_context_features[:2]])
            text += f", with {labels} raising its score the most"
        parts = [text + "."]

        rid = f"tsr.regime.{idx}"
        evidence.append(make_atom(
            rid, "regime", leader,
            {"index": idx, "start": r.get("start"), "end": r.get("end"),
             "duration": r.get("duration"), "leader": leader,
             "runner_up": runner,
             "top_channels": [(c, _val(v, 6)) for c, v in top_context_features],
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
            "tsr.caveat.nonnegative", "caveat", "context features", None,
            "Each context feature's share is a sum of squared weights, so it can never "
            "be negative: the split shows how a detector's score is divided "
            "among context features, not which context features pushed it down. Only the "
            "comparison against a rival has a direction."),
        make_atom(
            "tsr.caveat.exposure", "caveat", "selections", None,
            "A detector's weights only move in windows where it was selected, "
            "so this score reflects how often a detector was tried as well as "
            "how well it did; a rarely selected detector ranks low on thin "
            "evidence rather than on a poor showing."),
    ]
    if n_context_features is not None and int(n_context_features) == 1:
        caveats.append(make_atom(
            "tsr.caveat.single_channel", "caveat", "context features", int(n_context_features),
            "This dataset has a single context feature, so splitting the score across "
            "context features carries no information — that one context feature necessarily "
            "accounts for all of it."))

    question = ("Why did Thompson Sampling rank the detectors as it did — which "
                "context features drove each detector's ranking score up, and how much "
                "of the winner's margin came from each context feature?")

    env = _envelope("thompson_ranking", dataset, entity, output, evidence,
                    caveats, required, question=question)
    # Kept OUT of `output` and out of `evidence` on purpose. It is display data,
    # not a claim: the narrator renders every `output` key into its prompt, and
    # a detectors x context features matrix there would be a wall of floats the model
    # must ignore. The page reads it to decompose any pair's gap on demand,
    # since that split is exactly shares(a) - shares(b).
    if context_feature_shares:
        env["channel_shares"] = {str(m): [_val(v, 6) for v in vals]
                                 for m, vals in context_feature_shares.items()}
    return env


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

    return _envelope("ga_selection", dataset, entity, output, evidence, caveats,
                     required, question=question)


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


def _mark_heaviest(names: Sequence[str], heaviest: Optional[str]) -> List[str]:
    return [f"{n} — the detector carrying the most weight —" if n == heaviest else n
            for n in names]


def _sign_summary_text(members: Sequence[str], signs: Dict[str, str],
                       heaviest: Optional[str] = None) -> str:
    """One sentence naming each member's sign.

    `heaviest` is named inline when it signs negative: weight says how much a
    detector moves the meta-learner and sign says which way, and the two
    pointing against each other is the one thing here a reader should not have
    to assemble from two sentences.

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
                f"while {_oxford(_mark_heaviest(neg, heaviest))} had negative")
    elif pos:
        text = f"{_oxford(pos)} had {_v(pos, 'a positive sign', 'positive signs')}"
    elif neg:
        text = (f"{_oxford(_mark_heaviest(neg, heaviest))} had "
                f"{_v(neg, 'a negative sign', 'negative signs')}")
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
            # No trailing "the ranking below measures…" clause: it said what
            # this stage's own question says, and the narrator merged the two
            # into the ensemble sentence ("selected the 5-detector ensemble {…}
            # to determine which detectors the ensemble relies on most"). Each
            # detector atom states its own weight rank, so nothing is lost.
            f"The genetic algorithm selected the {n}-detector ensemble "
            f"{{{', '.join(detectors)}}}.",
            order=1))
        required.append("ga_comb.output.subset")

    for i, d in enumerate(detectors):
        rid = f"ga_comb.detector.{d}.role"
        # The most-weighted detector answers half of what this stage is asked, so
        # it carries its own type: the card drops the per-detector walk by type,
        # and dropping this one with it left the summary never naming the winner.
        role_type = ("detector_role_lead" if final_rank.get(d) == 1
                     else "detector_role")
        evidence.append(make_atom(
            rid, role_type, d,
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
        if i < HEAD_REQUIRED:
            required.append(rid)

    if detectors:
        heaviest = next((d for d in detectors if final_rank.get(d) == 1), None)
        sign_text = _sign_summary_text(detectors, signs, heaviest)
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
                  "the fitness drop when a detector's scores are shuffled."),
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

    return _envelope("ga_combination", dataset, entity, output, evidence, caveats,
                     required, question=question)


def _pick_verdict(prefix: str, stage_word: str, source: Any, pick: Any,
                  top: Any) -> Optional[Dict[str, Any]]:
    """Atom of its own: agreement is a rank correlation over the whole order and
    says nothing about whether a source wanted the same winner."""
    if pick == NOT_AVAILABLE or top == NOT_AVAILABLE or source is None:
        return None
    agrees = str(pick) == str(top)
    text = (f"{source}'s own ranking put {top} first, matching the "
            f"{stage_word} consensus." if agrees else
            f"{source}'s own ranking put {pick} first rather than the "
            f"{stage_word} consensus's {top}.")
    return make_atom(f"{prefix}.verdict.top_pick", "source_verdict", str(source),
                     {"source": source, "source_top_pick": pick,
                      "consensus_top": top, "agrees": agrees},
                     text, order=7)


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
            f"{winner} agreed with the {stage_word} consensus more closely "
            f"than {runner} did (agreement "
            f"{_fmt(kendall_only.get('winner_tau'), 4)} vs "
            f"{_fmt(kendall_only.get('runner_up_tau'), 4)}, gap "
            f"{_fmt(kendall_only.get('alignment_gap'), 4)})."))
        required.append(kid)
        verdict = _pick_verdict(prefix, stage_word, winner,
                                source_top_picks.get(str(winner), NOT_AVAILABLE), top)
        if verdict is not None:
            evidence.append(verdict)
            required.append(verdict["id"])
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
    else:
        # Multi-source case: one human-readable role sentence per source, ordered
        # by Borda rank (the dominant combined rank), built from its two component
        # ranks — INFLUENCE (leave-one-out: how much the consensus moves when the
        # source is dropped) and AGREEMENT (Kendall tau of the source vs the
        # consensus). Raw LOO/tau scores stay in `value` for provenance; the
        # prose carries only the ranks.

        # Required relational atom: names the source set explicitly. It used to
        # add that the ranked detectors are NOT sources, which was aimed at the
        # narrator and got printed as a sentence of its own.
        src_list = sorted(source_names)
        cid = f"{prefix}.context.sources"
        evidence.append(make_atom(
            cid, "stage_context", "sources",
            {"sources": src_list, "n_sources": len(src_list), "winner": top},
            f"The {len(src_list)} sources aggregated into this consensus are the "
            f"rankings {', '.join(src_list)}.",
            order=5))
        required.append(cid)

        def _borda_key(v: Dict[str, Any]) -> Tuple[float, str]:
            br = v.get("borda_rank")
            return (float(br) if br is not None else float("inf"), str(v.get("source")))

        # "shaped the consensus [Nth] most" — the ordinal is the source's
        # combined Borda standing: rank 1 → "most", 2 → "second most", … Ties
        # share an ordinal (two sources at Borda rank 3 are both "third most").
        _ORD_MOST = {1: "", 2: "second ", 3: "third ", 4: "fourth ", 5: "fifth ",
                     6: "sixth ", 7: "seventh ", 8: "eighth ", 9: "ninth ",
                     10: "tenth ", 11: "eleventh ", 12: "twelfth "}

        def _ordinal(k: Any) -> str:
            if k is None or _is_nan(k):
                return str(k)
            k = int(k)
            return f"{k}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(k if k < 20 else k % 10, 'th') }"

        def _shaped_prefix(br: Any) -> str:
            if br is None or _is_nan(br):
                return ""
            return _ORD_MOST.get(int(br), f"{int(br)}th ")

        # The source that moved the consensus most need not have wanted its
        # winner: influence measures how far the result shifts without a source,
        # not whether it agreed. This rides ON the leading source's own sentence
        # rather than in an atom of its own — as two sentences about one source
        # the narrator kept welding them together and inverting which detector
        # went where ("ranks X first in its own ranking despite putting Y at the
        # top instead").
        ranked = sorted(verdicts, key=_borda_key)
        # Only when one source is strictly ahead: sources tie at Borda rank 1
        # often enough, and "more than any other" is false when two share it.
        alone = (len(ranked) > 1
                 and _borda_key(ranked[0])[0] < _borda_key(ranked[1])[0])
        lead_src = str(ranked[0].get("source")) if ranked else None
        lead_pick = source_top_picks.get(lead_src, NOT_AVAILABLE)
        lead_tension = (alone and lead_src and top != NOT_AVAILABLE
                        and lead_pick != NOT_AVAILABLE and str(lead_pick) != str(top))

        n_src = len(src_list)
        for i, v in enumerate(ranked):
            name = v["source"]
            loo_rank, align_rank = v.get("loo_rank"), v.get("align_rank")
            br = v.get("borda_rank")
            # The combined Borda standing is stated once, as the verb phrase.
            # Repeating it as "(standing N of M)" put three "N of M" phrases in
            # one sentence, two of them the same number, and the narrator glued
            # the standing onto the component frame ("standing 2 of 6 for
            # influence and 2 of 6 for agreement, placing 3rd of 6 for
            # influence..."). The components carry their own ordinals and
            # labels, which is what the parenthetical was there to supply.
            text = (f"{name} shaped the {stage_word} consensus "
                    f"{_shaped_prefix(br)}most, placing "
                    f"{_ordinal(loo_rank)} of {n_src} for influence and "
                    f"{_ordinal(align_rank)} of {n_src} for agreement.")
            rid = f"{prefix}.source.{name}.role"
            evidence.append(make_atom(
                rid, "source_role", name,
                {"influence_rank": loo_rank, "agreement_rank": align_rank,
                 "borda_rank": v.get("borda_rank"),
                 "influence_score": _val(v.get("loo_score"), 4),
                 "agreement_score": _val(v.get("align_score"), 4),
                 "top_pick": source_top_picks.get(name, NOT_AVAILABLE)},
                # The leading source goes first: it is the answer this stage is
                # asked for, and with it further down the narrator invents an
                # opening sentence and then restates the fact verbatim.
                text, order=0 if i == 0 else 10 * (i + 1)))
            if i < HEAD_REQUIRED:
                required.append(rid)

        # Only when one source is strictly ahead: under a Borda tie at rank 1
        # "shaped the consensus most" does not identify whose pick this is.
        if alone:
            verdict = _pick_verdict(prefix, stage_word, lead_src,
                                    source_top_picks.get(str(lead_src), NOT_AVAILABLE), top)
            if verdict is not None:
                evidence.append(verdict)
                required.append(verdict["id"])

        question = (f"Which source rankings most shaped the {stage_word} consensus, "
                    f"and how much did each agree with it?")

    ir = _envelope(f"rank_aggregation_{stage_name}", dataset, entity, output,
                   evidence, caveats, required, question=question)
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
                         ranked_pr: Optional[List[str]] = None,
                         ranked_vus: Optional[List[str]] = None) -> Dict[str, Any]:
    curves_f1 = result.get("curves_f1", {})
    curves_pr = result.get("curves_pr", {})
    curves_vus = result.get("curves_vus", {})
    winner_f1 = result.get("winner_f1", {})
    winner_pr = result.get("winner_pr", {})
    winner_vus = result.get("winner_vus", {})
    permodel_f1 = result.get("permodel_f1", {})

    evidence: List[Dict[str, Any]] = []
    required: List[str] = []
    output = {
        "production_ranking_f1_top_k": _top_k(ranked_f1 or []),
        "production_ranking_pr_top_k": _top_k(ranked_pr or []),
        "production_ranking_vus_top_k": _top_k(ranked_vus or []),
        "top_pick_f1": (ranked_f1[0] if ranked_f1 else NOT_AVAILABLE),
        "top_pick_pr": (ranked_pr[0] if ranked_pr else NOT_AVAILABLE),
        "top_pick_vus": (ranked_vus[0] if ranked_vus else NOT_AVAILABLE),
    }
    top_f1 = ranked_f1[0] if ranked_f1 else None
    top_pr = ranked_pr[0] if ranked_pr else None
    top_vus = ranked_vus[0] if ranked_vus else None
    # One clause per metric the run actually ranked by; metrics that agree on a
    # winner share a clause rather than repeating the name.
    tops = [(label, ranking[0]) for label, ranking in
            (("F1 score", ranked_f1), ("PR-AUC", ranked_pr), ("VUS", ranked_vus))
            if ranking]
    if tops:
        by_winner: Dict[str, List[str]] = {}
        for label, winner in tops:
            by_winner.setdefault(winner, []).append(label)
        clauses = [f"{winner} ranked first by {_join_and(labels)}"
                   for winner, labels in by_winner.items()]
        lead = f"In the production Monte Carlo test, {_join_and(clauses)}."
        evidence.append(make_atom(
            "mc.output.top", "stage_output", str(tops[0][1]),
            {"top_f1": top_f1 or NOT_AVAILABLE, "top_pr": top_pr or NOT_AVAILABLE,
             "top_vus": top_vus or NOT_AVAILABLE},
            lead, order=0))
        required.append("mc.output.top")

    # ONE atom per detector, covering BOTH metrics. Two atoms about the same
    # detector (one per metric) is the same-subject-collapse trap: the narrator
    # states one and silently drops the other. Crossover atoms are dropped
    # entirely — a crossover is the derivative of the win regions, so emitting
    # both floods the prose with "at 0.042 ... at 0.053 ..." without adding a
    # single fact the regions do not already carry.
    regions_by_model: Dict[str, Dict[str, Any]] = {}
    for metric, curves in (("F1", curves_f1), ("PR-AUC", curves_pr),
                           ("VUS", curves_vus)):
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
                   for metric in ("F1", "PR-AUC", "VUS") if metric in per]
        wid = f"mc.win_region.{m}"
        evidence.append(make_atom(
            wid, "win_region", m,
            {metric: [(_val(a), _val(b)) for a, b in rs]
             for metric, rs in per.items()},
            f"{m} won {'; '.join(clauses)}.", order=30 + i))
        if i < HEAD_REQUIRED:
            required.append(wid)

    conf: Dict[str, Any] = {}
    # One surrogate per swept metric, but ONE atom for the sweep result and one
    # per DISAGREEMENT, not per metric. A sentence each restated the same
    # finding two or three times over — and metrics that disagree the same way,
    # same sweep winner against the same production winner, are one fact.
    sweeps = []
    for suffix, label, winner, prod_top in (
            ("f1", "F1", winner_f1, top_f1),
            ("pr", "PR-AUC", winner_pr, top_pr),
            ("vus", "VUS", winner_vus, top_vus)):
        if not winner.get("feasible"):
            continue
        cv_acc = winner.get("cv_accuracy", float("nan"))
        conf[f"winner_surrogate_{suffix}"] = {
            "train_accuracy": _val(winner.get("train_accuracy"), 3),
            "cv_accuracy": _val(cv_acc, 3),
            "grade": fidelity_grade(cv_acc),
        }
        wr = winner.get("win_rates", {})
        winners = sorted(((m, r) for m, r in wr.items() if r > 0),
                         key=lambda kv: kv[1], reverse=True)
        # One over the cut is named rather than summarised: "the remaining 1.0%
        # went to 1 further detector" spends a clause withholding a name it has
        # room for. The tail therefore always stands for two or more.
        cut = len(winners) if len(winners) <= TOP_K + 1 else TOP_K
        top_wr, rest = winners[:cut], winners[cut:]
        if top_wr:
            sweeps.append({"suffix": suffix, "label": label, "top": top_wr,
                           "rest": rest, "prod_top": prod_top, "rates": wr})

    if sweeps:
        clauses, by_metric = [], []
        for i, s in enumerate(sweeps):
            label, top_wr, rest = s["label"], s["top"], s["rest"]
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
                            f"{len(rest)} further detectors")
            lead = f"Measured by {label}" if i == 0 else f"while by {label}"
            clauses.append(f"{lead}, the noise-sweep trials were won by: "
                           f"{wr_txt}{tail_txt}")
            by_metric.append({"metric": label,
                              "listed": [(m, _val(r, 3)) for m, r in top_wr],
                              "n_other": len(rest), "other_share": _val(tail, 3)})
        evidence.append(make_atom(
            "mc.surrogate.win_rates", "surrogate_win_rates", "winner_surrogate",
            {"by_metric": by_metric}, "; ".join(clauses) + ".", order=20))
        required.append("mc.surrogate.win_rates")

        # A comparison between two facts belongs to neither of them, so
        # without this the narrative listed both and left the disagreement
        # for the reader to spot.
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for s in sweeps:
            sweep_top, prod_top = s["top"][0][0], s["prod_top"]
            if prod_top:
                grouped.setdefault((str(sweep_top), str(prod_top)), []).append(s)
        for i, ((sweep_top, prod_top), hits) in enumerate(grouped.items()):
            labels = [h["label"] for h in hits]
            agree = sweep_top == prod_top
            vid = "mc.sweep_verdict." + "_".join(h["suffix"] for h in hits)
            evidence.append(make_atom(
                vid, "sweep_verdict", sweep_top,
                {"metrics": labels, "agree": agree, "production_top": prod_top,
                 "sweep_top": sweep_top,
                 "sweep_top_win_rate": {h["label"]: _val(h["top"][0][1], 3)
                                        for h in hits},
                 "production_top_win_rate": {
                     h["label"]: _val(h["rates"].get(prod_top, 0.0), 3)
                     for h in hits}},
                f"Measured by {_join_and(labels)}, the sweep and the production "
                f"run {'agree' if agree else 'do not agree'}: {sweep_top} won "
                f"most of the noise trials"
                + (" and the production run ranked it first too." if agree else
                   f", while it was {prod_top} that the production run "
                   f"ranked first."),
                order=10 + i))
            required.append(vid)
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
    question = ("Which detector handles the injected noise best across "
                "different noise levels?")

    return _envelope("monte_carlo", dataset, entity, output, evidence, caveats,
                     required, confidence=conf, question=question)


# Clause-shaped labels used inside a surrogate condition ("… when <label> is
# at most 0.3"), and bare noun forms used when a feature is named on its own.
# One pair per point-injection stage. `is_anomaly` is a 0/1 label rather than a
# quantity, so it carries two ready-made statements instead of a clause label.
_OFFBY_LABELS = {
    "position": "its position in the series",
    "local_volatility": "the local volatility",
    "boundary_distance": "the distance from the boundary",
    "is_anomaly:false": "the point is not a real anomaly",
    "is_anomaly:true": "the point is a real anomaly",
}

_OFFBY_NOUNS = {
    "position": "the point's position in the series",
    "local_volatility": "the local volatility",
    "boundary_distance": "the distance from the boundary",
    "is_anomaly": "whether the point is a real anomaly",
}

# The GAN stage describes a generated point rather than a scaled one: its
# distance from the discriminator's threshold stands where off-by's distance
# from the decision boundary does, and three more features describe the signal
# itself and the site it was dropped into.
_GAN_LABELS = {
    "position": "its position in the series",
    "local_volatility": "the local volatility",
    "ambiguity": "its distance from the discriminator's threshold",
    "signal_magnitude": "the generated point's magnitude",
    "signal_spread": "the spread across the injected point's features",
    "context_gap": "its gap from the surrounding series",
    "is_anomaly:false": "the point was labelled normal",
    "is_anomaly:true": "the point was labelled anomalous",
}

_GAN_NOUNS = {
    "position": "the point's position in the series",
    "local_volatility": "the local volatility",
    "ambiguity": "the point's distance from the discriminator's threshold",
    "signal_magnitude": "the generated point's magnitude",
    "signal_spread": "the spread across the injected point's features",
    "context_gap": "its gap from the surrounding series",
    "is_anomaly": "whether the point was labelled anomalous",
}


def _condition_phrase(conditions: List[Dict[str, Any]],
                      labels: Dict[str, str]) -> str:
    """Render simplified surrogate conditions as plain prose. `is_anomaly` is a
    0/1 label, so its 0.5 split becomes a statement rather than a comparison;
    every other feature keeps its raw threshold (a grounded number)."""
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
            parts.append(labels["is_anomaly:false"] if f in upper
                         else labels["is_anomaly:true"])
            continue
        label = labels.get(f, f)
        if f in lower and f in upper:
            parts.append(f"{label} is between {lower[f]} and {upper[f]}")
        elif f in upper:
            parts.append(f"{label} is at most {upper[f]}")
        else:
            parts.append(f"{label} is above {lower[f]}")
    return " and ".join(parts) if parts else "always"


def build_off_by_ir(dataset: str, entity: str, result: Dict[str, Any],
                    ranked_f1_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Off-by-threshold: the winner's exclusive wins over the injected borderline
    points, described by the properties of those points."""
    return _build_exclusive_win_ir(
        "off_by_threshold", "ob", dataset, entity, result, ranked_f1_names,
        labels=_OFFBY_LABELS, nouns=_OFFBY_NOUNS,
        winner_text=lambda w: f"{w} was the highest-ranked model of the "
                              f"off-by-threshold stage.",
        points_text=lambda n: f"{n} borderline points were injected around the "
                              f"decision boundary.",
        question="Which model handled the injected borderline points best, and "
                 "what distinguishes the points it uniquely got right?")


def build_gan_ir(dataset: str, entity: str, result: Dict[str, Any],
                 ranked_f1_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """GAN perturbations: the same question as off-by, asked of generated points.

    The features differ — a GAN point is described by how ambiguous the
    discriminator found it and by the shape of the signal itself — but the
    target, the surrogate and the atom structure are the stage's sibling's.
    """
    return _build_exclusive_win_ir(
        "gan", "gn", dataset, entity, result, ranked_f1_names,
        labels=_GAN_LABELS, nouns=_GAN_NOUNS,
        winner_text=lambda w: f"{w} was the highest-ranked model of the GAN "
                              f"perturbation stage.",
        points_text=lambda n: f"{n} generated points were injected near the "
                              f"discriminator's decision threshold.",
        question="Which model handled the injected GAN points best, and what "
                 "distinguishes the points it uniquely got right?")


def _build_exclusive_win_ir(stage: str, prefix: str, dataset: str, entity: str,
                            result: Dict[str, Any],
                            ranked_f1_names: Optional[List[str]],
                            *, labels: Dict[str, str], nouns: Dict[str, str],
                            winner_text, points_text,
                            question: str) -> Dict[str, Any]:
    """Shared IR builder for the two point-injection robustness stages.

    Both ask the same question of their own injected points — which points did
    the winner uniquely get right, and what were those points like — so the atom
    families, the deduplication and the ordering are one implementation. Only the
    feature vocabulary and the stage's own sentences are passed in.
    """
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
        f"{prefix}.output.winner", "stage_output", str(winner), winner,
        winner_text(winner), order=1))
    required.append(f"{prefix}.output.winner")
    evidence.append(make_atom(
        f"{prefix}.points", "injected_points", "injection", int(n_points),
        points_text(int(n_points)), order=2))

    conf: Dict[str, Any] = {}
    caveats = [
        make_atom(f"{prefix}.caveat.f1_side", "caveat", "scope", None,
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
            f"{prefix}.caveat.support", "caveat", "support", {k0: n0},
            f"The rule for {k0} rests on only {n0} exclusive-win "
            f"point{'' if n0 == 1 else 's'} — fewer than the {N_CV_FOLDS} "
            f"cross-validation folds — so its held-out fidelity is unstable; "
            f"treat it as indicative."))
    elif low_support:
        caveats.append(make_atom(
            f"{prefix}.caveat.support", "caveat", "support",
            {k0: n0 for k0, n0 in low_support},
            f"The rules for {_oxford([k0 for k0, _ in low_support])} each rest "
            f"on fewer than {N_CV_FOLDS} exclusive-win points — fewer than the "
            f"{N_CV_FOLDS} cross-validation folds — so their held-out fidelity "
            f"is unstable; treat them as indicative."))

    # ── One atom per rival group: its rule(s) AND its win count together ──
    #
    # Grouped by (rules, count, rate) so every rival set is named exactly once,
    # with everything about it in a single sentence. Splitting rules and wins
    # into two atom families names the same rivals twice in two different
    # orders, and a narrator will merge across them — emitting one sentence
    # that carries a rule from one group and names from another.
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
        """Biggest edge first; the rival name breaks ties deterministically.

        This is the order the narrator follows, and the card shows only the
        opening sentences by default (`WebUI.summarize._STAGE_SUMMARY`), so it
        also decides which rivals a reader sees without clicking.

        The opposite order was tried — smallest count first, on the reading that
        a rival the winner rarely beat outright is the one that ran it closest —
        and rejected for both stages. Biggest margin first is what stands.
        """
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
        phrases = [_condition_phrase(rule_groups[s]["rule"]["conditions"], labels)
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
        eid = f"{prefix}.edge.{ei}"
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
            f"{prefix}.degenerate", "degenerate_comparison", str(winner),
            {"competitors": degenerate},
            f"{winner} never exclusively beat {_oxford(degenerate)}, so "
            f"{'none of them appears' if len(degenerate) > 1 else 'it does not appear'} "
            f"above.", order=500))
        required.append(f"{prefix}.degenerate")

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
            f"{prefix}.vs.{ks[0]}.importance", "feature_importance",
            ks[0] if len(ks) == 1 else "competitors",
            {"feature": feat, "importance": float(shown), "competitors": ks},
            f"Against {against}, the property that best separates those points "
            f"is {nouns.get(feat, feat)} (importance {shown}{each}).",
            order=300 + ii))

    if top is not None:
        evidence.append(make_atom(
            f"{prefix}.summary.top_feature", "summary", top[0], _val(top[1], 3),
            f"Across all competitors, {winner}'s edge is best explained by "
            f"{nouns.get(top[0], top[0])} (mean importance "
            f"{_fmt(top[1], 2)}).", order=400))
        required.append(f"{prefix}.summary.top_feature")


    return _envelope(stage, dataset, entity, output, evidence,
                     caveats, required, confidence=conf, question=question)


# ── Global assembly ──────────────────────────────────────────────────────────

_STAGE_FILES = {
    "thompson_sampling": "ir_thompson",
    # The ranking-criterion sibling. Deliberately absent from stage_picks below:
    # both Thompson stages report the same winner (they share rank_models), so
    # counting it again would double-weight Thompson in the cross-stage consensus.
    "thompson_ranking": "ir_thompson_ranking",
    "ga_selection": "ir_ga_selection",
    "ga_combination": "ir_ga_combination",
    # The robustness block in the order it is read on the page and in the global
    # narrative; see Explainability.llm._GLOBAL_STAGE_ORDER, which this feeds.
    "monte_carlo": "ir_monte_carlo",
    "off_by_threshold": "ir_off_by",
    "gan": "ir_gan",
}


def assemble_global_ir(results_dict: Dict[str, Any], dataset: str, entity: str,
                       iteration: int,
                       base_dir: str = "myresults/explanations_ir") -> str:
    """
    Combine the per-stage IR JSONs (written by each explainability orchestrator)
    with the pipeline's decision context into ir_global_iter{iteration}.json.
    Missing stage files → explicit not_available.

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
            # Carry a reason rather than a bare status: the page lists these
            # under "stages without an explanation", and "not available" on its
            # own reads as a defect where the usual cause is simply that the run
            # skipped this stage or was not given --explain.
            stages[stage] = {
                "status": NOT_AVAILABLE,
                "note": "no explanation was written for this stage — the run "
                        "either skipped it or did not use --explain"}
        else:
            loaded_docs[stage] = loaded
            stages[stage] = {"status": "ok", "output": loaded.get("output", {})}
            all_caveats.extend(loaded.get("caveats", []))

    fd = results_dict.get("final_decision", {}) or {}
    choice = fd.get("framework_choice", NOT_AVAILABLE)
    ens_f1 = fd.get("ensemble_f1", float("nan"))
    sng_f1 = fd.get("single_model_f1", float("nan"))
    margin = (ens_f1 - sng_f1) if not (_is_nan(ens_f1) or _is_nan(sng_f1)) else float("nan")
    # The f1_* keys stay whatever they always were; the score_* ones carry the
    # metric the run actually decided on. Result trees written before the metric
    # was selectable have no score_*, so these fall back to F1.
    metric = fd.get("decision_metric", ("f1",))
    label = decision_metric_label(metric)
    # The weighted formula, not just the metric names: a reader of the decision
    # needs to know what each metric contributed. The weights also travel in the
    # atom's value, which is what grounds those numbers for the verifier.
    formula = decision_metric_formula(metric)
    weights = {m: round(w, 4) for m, w in metric_weights(metric).items()}
    ens_score = fd.get("ensemble_score", ens_f1)
    sng_score = fd.get("single_model_score", sng_f1)
    score_margin = ((ens_score - sng_score)
                    if not (_is_nan(ens_score) or _is_nan(sng_score)) else float("nan"))
    if choice == "ensemble":
        reason = (f"The ensemble was chosen because its fitness ({formula}, {_fmt(ens_score, 4)}) "
                  f"is greater than or equal to the best single model's fitness "
                  f"({_fmt(sng_score, 4)}).")
    elif choice == "single_model":
        reason = (f"The single model was chosen because its fitness ({formula}, {_fmt(sng_score, 4)}) "
                  f"exceeds the ensemble's fitness ({_fmt(ens_score, 4)}).")
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
        "ensemble_vus": _val(fd.get("ensemble_vus"), 4),
        "single_model_vus": _val(fd.get("single_model_vus"), 4),
        "f1_margin_ensemble_minus_single": _val(margin, 4),
        "decision_metric": metric,
        "decision_metric_label": label,
        "decision_metric_formula": formula,
        "decision_metric_weights": weights,
        "ensemble_score": _val(ens_score, 4),
        "single_model_score": _val(sng_score, 4),
        "score_margin_ensemble_minus_single": _val(score_margin, 4),
        "reason": reason,
    }

    # Cross-stage top-pick agreement (single-branch stages vs the final single pick).
    # One entry per ranking that actually entered the robustness consensus, so
    # the strip shows the sources the run voted with and no others.
    single_pick = fd.get("single_model", NOT_AVAILABLE)
    agg = results_dict.get("aggregation", {}) or {}
    _robust_stages = (("gan", "gan_robustness"), ("borderline", "borderline"),
                      ("monte_carlo", "monte_carlo"))
    sources = []
    for i, m in enumerate(ranking_metrics_for(metric)):
        for key, slot in _robust_stages:
            sources.append((f"{key}_{m}", key, m, _full_ranking(
                (results_dict.get(slot, {}) or {}).get(f"{m}_names"))))
        if i == 0:
            # Only the robust consensus is compared. The final consensus IS the
            # source of the single-model pick, so asking whether they agree is
            # tautological — it would always report agreement and say nothing.
            sources.append(("robust_consensus", "robust_consensus", NOT_AVAILABLE,
                            _full_ranking(agg.get("robust_agg"))))
            sources.append(("thompson", "thompson", NOT_AVAILABLE, _full_ranking(
                (results_dict.get("thompson", {}) or {}).get("top_models"))))
    agreement = {}
    for position, (name, stage, m, ranking) in enumerate(sources):
        pick = ranking[0] if ranking else NOT_AVAILABLE
        agreement[name] = {
            "top_pick": _py(pick),
            "ranking": ranking or [],
            "stage": stage,
            "metric": m,
            # The document is written with sort_keys=True, so display order has
            # to travel as a value: one row per metric, each stage in the same
            # column on every row.
            "order": position,
            "agrees_with_final_single": (pick == single_pick)
            if pick not in (NOT_AVAILABLE, "N/A") else NOT_AVAILABLE,
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
        dec_text = (f"The final decision is the ensemble {{{', '.join(ens)}}}, whose fitness "
                    f"({formula}) is {_fmt(ens_score, 4)}, chosen over the best single model "
                    f"{single} at {_fmt(sng_score, 4)}.")
    elif choice == "single_model":
        dec_text = (f"The final decision is the single model {single}, whose fitness "
                    f"({formula}) is {_fmt(sng_score, 4)}, chosen over the GA ensemble "
                    f"at {_fmt(ens_score, 4)}.")
    else:
        dec_text = "The final framework decision is not available."
    evidence.append(make_atom("global.decision", "decision", str(choice),
                              {"framework_choice": choice,
                               "decision_metric": metric,
                               "decision_metric_weights": weights,
                               "ensemble_score": _val(ens_score, 4),
                               "single_model_score": _val(sng_score, 4)},
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
