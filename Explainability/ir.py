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
    # `info_footer` is a fixed glossary appended verbatim to the .txt AFTER
    # generation and verification, so definitions are never reworded and never
    # count toward faithfulness metrics.
    if question is not None:
        env["question"] = str(question)
    if info_footer is not None:
        env["info_footer"] = str(info_footer)
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


# ── Stage builders ───────────────────────────────────────────────────────────

def build_thompson_ir(dataset: str, entity: str, *, n_windows: int,
                      final_ranking: List[Tuple[str, float]],
                      regimes: List[Dict[str, Any]],
                      shifts: List[Dict[str, Any]],
                      blip_count: int,
                      state_fractions: Dict[str, float],
                      final_state: str) -> Dict[str, Any]:
    """
    `regimes` entries are precomputed by explain_thompson_sampling (it owns the
    SHAP helpers): {"index","start","end","duration","leader",
    "rewards_top": [(model, mean_reward)], "reward_gap": float|None,
    "runner_up": str|None,
    "shap_raising": [(ch, val)]|None, "shap_lowering": [(ch, val)]|None,
    "pref_favor_leader": [(ch, delta)]|None,
    "pref_favor_runner": [(ch, delta)]|None,
    "pref_gap": float|None}.
    SHAP/preference channels arrive pre-split by sign so no consumer ever has
    to infer direction from a signed list sorted by magnitude.
    """
    evidence: List[Dict[str, Any]] = []
    required: List[str] = []

    top_pairs = _top_k(final_ranking)
    top_model = top_pairs[0][0] if top_pairs else NOT_AVAILABLE
    output = {
        "top_pick": top_model,
        "final_ranking_top_k": [{"model": m, "score": _val(s, 6)} for m, s in top_pairs],
        "k": len(top_pairs),
        "n_windows": int(n_windows),
    }
    if top_pairs:
        evidence.append(make_atom(
            "ts.output.top", "stage_output", top_model, _val(top_pairs[0][1], 6),
            f"Thompson Sampling ranks {top_model} first "
            f"(final posterior score {_fmt(top_pairs[0][1], 6)})."))
        required.append("ts.output.top")

    frac_txt = ", ".join(f"{s}: {_fmt(100.0 * f, 1)}%" for s, f in sorted(state_fractions.items()))
    evidence.append(make_atom(
        "ts.states.summary", "behavior_summary", "selection_states", state_fractions,
        f"Selection behavior over {n_windows} windows — {frac_txt}."))
    required.append("ts.states.summary")
    evidence.append(make_atom(
        "ts.states.final", "behavior_final", "selection_states", final_state,
        f"The final window's selection state was {final_state}."))

    evidence.append(make_atom(
        "ts.shifts.count", "regime_shift_count", "regimes", len(shifts),
        f"{len(shifts)} regime shift(s) were detected."
        if shifts else "No regime shifts were detected."))
    for i, s in enumerate(shifts):
        evidence.append(make_atom(
            f"ts.shift.{i}", "regime_shift", str(s.get("to_model", NOT_AVAILABLE)),
            {"window": s.get("window"), "from": s.get("from_model"),
             "to": s.get("to_model"), "reward_delta": _val(s.get("reward_delta"), 4)},
            f"At window {s.get('window')} the expected-reward leader changed from "
            f"{s.get('from_model')} to {s.get('to_model')} "
            f"(reward delta {_fmt(s.get('reward_delta'), 4)})."))
    evidence.append(make_atom(
        "ts.blips.count", "blip_count", "regimes", int(blip_count),
        f"{int(blip_count)} brief blip window(s) were observed."
        if blip_count else "No brief blips were observed."))

    for r in regimes:
        i = r["index"]
        span_id = f"ts.regime.{i}.span"
        evidence.append(make_atom(
            span_id, "regime_span", str(r["leader"]),
            {"start": r["start"], "end": r["end"], "duration": r["duration"]},
            f"Regime {i}: windows {r['start']}-{r['end']} ({r['duration']} windows), "
            f"led by {r['leader']}."))
        required.append(span_id)

        if r.get("rewards_top"):
            rw_txt = ", ".join(f"{m} {_fmt(v, 4)}" for m, v in r["rewards_top"])
            gap_txt = (f"; leader-vs-runner-up mean-reward gap {_fmt(r['reward_gap'], 4)}"
                       if not _is_nan(r.get("reward_gap")) else "")
            rid = f"ts.regime.{i}.rewards"
            evidence.append(make_atom(
                rid, "regime_rewards", str(r["leader"]),
                {"top": [(m, _val(v, 4)) for m, v in r["rewards_top"]],
                 "mean_reward_gap": _val(r.get("reward_gap"), 4)},
                f"Mean expected rewards in regime {i}: {rw_txt}{gap_txt}."))
            required.append(rid)

        if r.get("shap_raising") or r.get("shap_lowering"):
            raising = r.get("shap_raising") or []
            lowering = r.get("shap_lowering") or []
            raise_txt = (", ".join(f"channel {c} ({_fmt(v, 4)})" for c, v in raising)
                         or "none")
            lower_txt = (", ".join(f"channel {c} ({_fmt(v, 4)})" for c, v in lowering)
                         or "none")
            evidence.append(make_atom(
                f"ts.regime.{i}.shap", "regime_shap", str(r["leader"]),
                {"raising": [(c, _val(v, 4)) for c, v in raising],
                 "lowering": [(c, _val(v, 4)) for c, v in lowering]},
                f"In regime {i}, channels raising {r['leader']}'s expected reward: "
                f"{raise_txt}; channels lowering it: {lower_txt}."))

        has_pref = r.get("pref_favor_leader") or r.get("pref_favor_runner")
        if has_pref and r.get("runner_up") and not _is_nan(r.get("pref_gap")):
            leader, runner = str(r["leader"]), str(r["runner_up"])
            gap = float(r["pref_gap"])
            favor_l = r.get("pref_favor_leader") or []
            favor_r = r.get("pref_favor_runner") or []
            fl_txt = (", ".join(f"channel {c} ({_fmt(d, 4)})" for c, d in favor_l)
                      or "none")
            fr_txt = (", ".join(f"channel {c} ({_fmt(d, 4)})" for c, d in favor_r)
                      or "none")
            if gap >= 0:
                head = (f"In regime {i}, the linear preference score at the "
                        f"regime-average context favors {leader} over {runner} "
                        f"by {_fmt(gap, 4)}.")
            else:
                # Direction-honest: the selection leader and the posterior
                # score can disagree; say so instead of leaving a signed
                # number open to misreading.
                head = (f"In regime {i}, although {leader} led selections, the "
                        f"linear preference score at the regime-average context "
                        f"favors {runner} by {_fmt(abs(gap), 4)}.")
            evidence.append(make_atom(
                f"ts.regime.{i}.pref", "regime_preference", leader,
                {"runner_up": runner, "preference_score_gap": _val(gap, 4),
                 "favor_leader": [(c, _val(d, 4)) for c, d in favor_l],
                 "favor_runner": [(c, _val(d, 4)) for c, d in favor_r]},
                f"{head} Channels favoring {leader}: {fl_txt}. "
                f"Channels favoring {runner}: {fr_txt}."))

    caveats = [
        make_atom("ts.caveat.stochastic", "caveat", "thompson_sampling", None,
                  "Thompson Sampling is a stochastic (seeded) sampler; a different "
                  "seed can produce a different selection trajectory."),
        make_atom("ts.caveat.states", "caveat", "selection_states", None,
                  "Behavioral states are observational labels derived from the "
                  "sampler's decisions, not causal mechanisms."),
        make_atom("ts.caveat.shap_framing", "caveat", "shap", None,
                  "Per-regime SHAP attributes the FINAL posterior means on each "
                  "regime's aggregated context; it is not a replay of the beliefs "
                  "held during that regime."),
    ]
    return _envelope("thompson_sampling", dataset, entity, output, evidence, caveats, required)


def build_ga_selection_ir(dataset: str, entity: str, result: Dict[str, Any]) -> Dict[str, Any]:
    best = list(result.get("best_ensemble", []))
    lofo: Dict[str, float] = result.get("lofo", {})
    mm: Dict[str, Dict[str, float]] = result.get("mean_marginal", {})
    archetypes: Dict[str, Dict[str, Any]] = result.get("archetypes", {})
    detectors = list(archetypes.keys())

    util = {d: mm.get(d, {}).get("contribution", float("nan")) for d in detectors}
    finite_sorted = sorted((d for d in detectors if not _is_nan(util[d])),
                           key=lambda d: util[d], reverse=True)
    # Standard competition ranking ("1224") so tied contributions share a rank
    # and the next skips — consistent with every other ranking in the layer.
    util_rank = _competition_rank(util, finite_sorted)

    def _code_words(code: str) -> str:
        if code == "Unclassified" or len(code) != 2:
            return "unclassified"
        u = "high" if code[0] == "H" else "low"
        s = "high" if code[1] == "H" else "low"
        return f"{u} utility, {s} stability"

    evidence: List[Dict[str, Any]] = []
    required: List[str] = []
    output = {"best_ensemble": best, "ensemble_size": len(best)}
    evidence.append(make_atom(
        "ga_sel.output.ensemble", "stage_output", "best_ensemble", best,
        f"The genetic algorithm selected the ensemble {{{', '.join(best)}}}."))
    required.append("ga_sel.output.ensemble")

    # One self-contained "member card" atom per ensemble member. Keeping the
    # archetype, utility, stability, and LOFO of a detector in a SINGLE
    # sentence prevents the narrator from re-associating values across
    # neighboring detectors (observed with the previous four-atoms-per-member
    # layout: swapped utilities, wrongly generalized archetypes).
    for d in best:
        arch = archetypes.get(d, {})
        code = arch.get("relative", {}).get("archetype", NOT_AVAILABLE)
        u = util.get(d, float("nan"))
        sm = arch.get("stability_mean", float("nan"))
        lv = lofo.get(d, float("nan"))

        parts = [f"{d}: archetype {code} ({_code_words(code)}, relative "
                 f"median-split scheme)"]
        if not _is_nan(u):
            parts.append(f"mean marginal contribution {_fmt(u, 4)} "
                         f"(rank {util_rank.get(d, NOT_AVAILABLE)} of "
                         f"{len(finite_sorted)})")
        if not _is_nan(sm):
            parts.append(f"survived {_fmt(100.0 * sm, 1)}% of GA generations")
        if not _is_nan(lv):
            direction = "hurts" if lv > 0 else ("does not hurt" if lv < 0 else "does not change")
            parts.append(f"removing it changes ensemble fitness by {_fmt(-lv, 4)} "
                         f"(LOFO {_fmt(lv, 4)}; removal {direction} the ensemble)")
        cid = f"ga_sel.member.{d}.card"
        evidence.append(make_atom(
            cid, "member_card", d,
            {"archetype": code, "utility": _val(u, 4),
             "utility_rank": util_rank.get(d), "stability": _val(sm, 3),
             "lofo": _val(lv, 4)},
            "; ".join(parts) + "."))
        required.append(cid)
        if not _is_nan(lv) and not _is_nan(u) and lv != 0 and u != 0 and (lv > 0) != (u > 0):
            evidence.append(make_atom(
                f"ga_sel.member.{d}.disagreement", "signal_disagreement", d,
                {"lofo": _val(lv, 4), "mean_marginal": _val(u, 4)},
                f"For {d}, LOFO ({_fmt(lv, 4)}) and the mean marginal contribution "
                f"({_fmt(u, 4)}) disagree in sign — the two utility views conflict."))

    excluded = [d for d in finite_sorted if d not in best]
    if excluded:
        d0 = excluded[0]
        evidence.append(make_atom(
            "ga_sel.excluded.top_utility", "excluded_detector", d0, _val(util[d0], 4),
            f"Among detectors not selected, {d0} had the highest mean marginal "
            f"contribution ({_fmt(util[d0], 4)})."))

    caveats = [
        make_atom("ga_sel.caveat.relative", "caveat", "archetypes", None,
                  "Archetype labels use relative (median-split) thresholds, so they "
                  "describe standing within this detector cohort, not absolute quality."),
    ]
    if len(best) < 2:
        caveats.append(make_atom(
            "ga_sel.caveat.lofo_na", "caveat", "lofo", None,
            "LOFO is undefined for ensembles with fewer than two members."))
    return _envelope("ga_selection", dataset, entity, output, evidence, caveats, required)


def _competition_rank(scores: Dict[str, Any], order: List[str]) -> Dict[str, int]:
    """Competition ranking ('1224') over `order` (already score-descending)."""
    ranks: Dict[str, int] = {}
    prev = None
    rank = 0
    for i, d in enumerate(order):
        v = scores.get(d)
        if prev is None or v != prev:
            rank = i + 1
            prev = v
        ranks[d] = rank
    return ranks


def build_ga_combination_ir(dataset: str, entity: str, result: Dict[str, Any]) -> Dict[str, Any]:
    ranking = list(result.get("final_ranking", []))
    members = list(result.get("best_ensemble", []))
    pi: Dict[str, float] = result.get("markov_scores", {})
    s_abs: Dict[str, float] = result.get("shap_importance", {})
    s_sgn: Dict[str, float] = result.get("shap_signed_importance", {})
    pfi: Dict[str, float] = result.get("pfi_importance", {})

    def _rank_of(imp: Dict[str, float]) -> Dict[str, int]:
        order = sorted((d for d in ranking if not _is_nan(imp.get(d))),
                       key=lambda d: imp[d], reverse=True)
        return {d: i + 1 for i, d in enumerate(order)}

    r_abs, r_sgn, r_pfi = _rank_of(s_abs), _rank_of(s_sgn), _rank_of(pfi)
    final_rank = _competition_rank(pi, ranking)

    evidence: List[Dict[str, Any]] = []
    required: List[str] = []
    top = ranking[0] if ranking else NOT_AVAILABLE
    output = {
        "top_pick": top,
        "final_ranking_top_k": [{"detector": d, "final_rank": final_rank.get(d),
                                 "markov_score": _val(pi.get(d), 4)}
                                for d in _top_k(ranking)],
        "k": min(TOP_K, len(ranking)),
        "ensemble_members": members,
        "meta_model_type": result.get("meta_model_type", NOT_AVAILABLE),
        "model_source": result.get("model_source", NOT_AVAILABLE),
        "baseline_f1": _val(result.get("baseline_f1"), 4),
    }
    if ranking:
        evidence.append(make_atom(
            "ga_comb.output.top", "stage_output", top, _val(pi.get(top), 4),
            f"The meta-learner weighs {top} most heavily "
            f"(final rank 1, Markov consensus score {_fmt(pi.get(top), 4)})."))
        required.append("ga_comb.output.top")
    if members:
        # Relational fact: this ranking is INSIDE the ensemble branch — every
        # ranked detector is a member of the GA-selected ensemble.
        evidence.append(make_atom(
            "ga_comb.context.members", "stage_context", "best_ensemble", members,
            f"This ranking weighs the {len(members)} members of the GA-selected "
            f"ensemble ({', '.join(members)}); every ranked detector is part of "
            f"that ensemble."))
        required.append("ga_comb.context.members")

    for d in _top_k(ranking):
        sgn = s_sgn.get(d, float("nan"))
        direction = (NOT_AVAILABLE if _is_nan(sgn)
                     else ("positive" if sgn >= 0 else "negative"))
        evidence.append(make_atom(
            f"ga_comb.detector.{d}.methods", "method_evidence", d,
            {"final_rank": final_rank.get(d),
             "markov_score": _val(pi.get(d), 4),
             "mean_abs_shap": _val(s_abs.get(d), 6), "mean_abs_shap_rank": r_abs.get(d),
             "signed_shap": _val(sgn, 6), "signed_shap_rank": r_sgn.get(d),
             "signed_direction": direction,
             "pfi_f1_drop": _val(pfi.get(d), 6), "pfi_rank": r_pfi.get(d)},
            f"{d}: final rank {final_rank.get(d, NOT_AVAILABLE)} "
            f"(Markov score {_fmt(pi.get(d), 4)}); "
            f"mean |SHAP| {_fmt(s_abs.get(d), 4)} (rank {r_abs.get(d, NOT_AVAILABLE)}), "
            f"signed SHAP {_fmt(sgn, 4)} (rank {r_sgn.get(d, NOT_AVAILABLE)}, "
            f"{direction} influence), "
            f"PFI F1-drop {_fmt(pfi.get(d), 4)} (rank {r_pfi.get(d, NOT_AVAILABLE)})."))
        if r_abs.get(d) == 1 and r_sgn.get(d) == 1 and r_pfi.get(d) == 1:
            evidence.append(make_atom(
                f"ga_comb.detector.{d}.agreement", "method_agreement", d, 3,
                f"All three attribution methods rank {d} first."))

    caveats = [
        make_atom("ga_comb.caveat.methods", "caveat", "attribution", None,
                  "Mean |SHAP| and signed SHAP are label-free (they explain the "
                  "meta-learner's output); PFI is label-based (it measures the F1 "
                  "drop when a detector's scores are shuffled)."),
        make_atom("ga_comb.caveat.aggregation", "caveat", "markov", None,
                  "The final ranking is the stationary distribution of a Markov "
                  "chain over the three methods' pairwise preferences."),
    ]
    return _envelope("ga_combination", dataset, entity, output, evidence, caveats, required)


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
            if not p or p == NOT_AVAILABLE:
                return ""
            article = "an" if str(p)[0].lower() in "aeiou" else "a"
            return f", {article} {p} pattern"

        for i, v in enumerate(sorted(verdicts, key=_borda_key)):
            name = v["source"]
            loo_rank, align_rank = v.get("loo_rank"), v.get("align_rank")
            pp = _pattern_phrase(v.get("pattern"))
            if i == 0:
                if loo_rank == 1 and align_rank == 1:
                    # State the explicit ranks (not just "leading both") so the
                    # narrator never has to infer them — it previously filled in
                    # the lead's agreement rank itself and got it wrong.
                    text = (f"{name} shaped the {stage_word} consensus most, "
                            f"topping both the influence ranking (rank 1) and the "
                            f"agreement ranking (rank 1){pp}.")
                else:
                    text = (f"{name} shaped the {stage_word} consensus most overall, "
                            f"with influence rank {loo_rank} and agreement rank "
                            f"{align_rank}{pp}.")
            else:
                text = (f"{name} followed with influence rank {loo_rank} and "
                        f"agreement rank {align_rank}{pp}.")
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
            "Influence measures how much a source shaped the consensus: it compares "
            "the consensus ranking with the ranking that emerges when that source is "
            "left out. Agreement compares the consensus ranking with the source's own "
            "ranking. The overall order combines both (Borda). An influential_disagreer "
            "has high influence but low agreement; a redundant_agreer has high "
            "agreement but low influence; a consistent source ranks similarly on both.")

    ir = _envelope(f"rank_aggregation_{stage_name}", dataset, entity, output,
                   evidence, caveats, required, question=question,
                   info_footer=info_footer)
    ir["iteration"] = int(iteration)
    return ir


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
        "k": min(TOP_K, len(ranked_f1 or [])),
    }
    if ranked_f1:
        evidence.append(make_atom(
            "mc.output.top_f1", "stage_output", ranked_f1[0], ranked_f1[0],
            f"The production Monte Carlo test (fixed noise) ranks {ranked_f1[0]} "
            f"first by F1."))
        required.append("mc.output.top_f1")

    for metric, curves in (("F1", curves_f1), ("PR-AUC", curves_pr)):
        tag = metric.replace("-", "").lower()
        for m, regions in sorted((curves.get("win_regions") or {}).items()):
            if not regions:
                continue
            # Degenerate one-grid-point regions read as "from 0.063 to 0.063"
            # and flood the narrative; compress them to a list of isolated
            # levels while real spans keep their interval form.
            spans = [(a, b) for a, b in regions if a != b]
            points = [a for a, b in regions if a == b]
            parts = []
            if spans:
                parts.append("for noise levels in "
                             + ", ".join(f"[{_fmt(a)}, {_fmt(b)}]" for a, b in spans))
            if points:
                parts.append("at the isolated noise level(s) "
                             + ", ".join(_fmt(p) for p in points))
            wid = f"mc.win_region.{tag}.{m}"
            evidence.append(make_atom(
                wid, "win_region", m, [(_val(a), _val(b)) for a, b in regions],
                f"Under the {metric} sweep, {m} leads " + " and ".join(parts) + "."))
            required.append(wid)
        for i, cx in enumerate(curves.get("crossovers") or []):
            evidence.append(make_atom(
                f"mc.crossover.{tag}.{i}", "crossover", str(cx.get("to_model")),
                {"noise": _val(cx.get("noise")), "from": cx.get("from_model"),
                 "to": cx.get("to_model")},
                f"On the {metric} curves the leader changes from {cx.get('from_model')} "
                f"to {cx.get('to_model')} at noise level {_fmt(cx.get('noise'))}."))

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
        top_wr = sorted(((m, r) for m, r in wr.items() if r > 0),
                        key=lambda kv: kv[1], reverse=True)[:TOP_K]
        if top_wr:
            wr_txt = ", ".join(f"{m} {_fmt(100.0 * r, 1)}%" for m, r in top_wr)
            evidence.append(make_atom(
                "mc.surrogate.win_rates", "surrogate_win_rates", "winner_surrogate",
                [(m, _val(r, 3)) for m, r in top_wr],
                f"Across the noise sweep the trials were won by: {wr_txt}."))
        rules = winner_f1.get("rules")
        if rules:
            # IR-level cleanup: collapse each leaf's redundant same-feature
            # bounds to one interval, then merge adjacent same-winner
            # intervals (single-feature tree = partition of the noise axis).
            rules = [dict(r, conditions=simplify_conditions(r.get("conditions", [])))
                     for r in rules]
            rules = merge_single_feature_rules(rules)
            # A depth-3 tree can have up to 8 leaves; requiring them all in a
            # fixed word budget inflates omission artifactually. Only the three
            # best-supported rules are required; the rest stay available.
            by_support = sorted(range(len(rules)),
                                key=lambda i: rules[i].get("n_samples", 0),
                                reverse=True)
            required_rule_idx = set(by_support[:3])
            for i, rule in enumerate(rules):
                rid = f"mc.surrogate.rule.{i}"
                evidence.append(make_atom(
                    rid, "surrogate_rule", str(rule.get("outcome")), rule,
                    rule_to_text(rule, "the noise-sweep F1 winner is"),
                    confidence=grade))
                if i in required_rule_idx:
                    required.append(rid)
        elif winner_f1.get("rules_text"):
            evidence.append(make_atom(
                "mc.surrogate.rule.0", "surrogate_rule",
                str((winner_f1.get("classes") or [NOT_AVAILABLE])[0]),
                winner_f1.get("rules_text"),
                str(winner_f1.get("rules_text")), confidence=grade))

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

    caveats = [
        make_atom("mc.caveat.proxy", "caveat", "metric", None,
                  "The noise sweep scores with a fast point-wise best-threshold F1 "
                  "and PR-AUC; the production ranking uses a range-based metric, so "
                  "sweep values are not directly comparable to production values."),
        make_atom("mc.caveat.scope", "caveat", "sweep", None,
                  "The sweep (noise 0.0-0.2, 20 levels) exists only to explain "
                  "robustness; the pipeline's forwarded ranking comes from the "
                  "production run at fixed noise."),
    ]
    if degenerate_models:
        caveats.append(make_atom(
            "mc.caveat.cv_degenerate", "caveat", "confidence", degenerate_models,
            f"For {', '.join(degenerate_models)} most cross-validation folds had "
            f"(near-)constant F1 across the sweep, so the held-out R² is not a "
            f"meaningful fidelity estimate (marked not_available); the number is "
            f"kept only for transparency."))
    return _envelope("monte_carlo", dataset, entity, output, evidence, caveats,
                     required, confidence=conf)


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
        "k": min(TOP_K, len(ranked_f1_names or [])),
        "n_injected_points": int(n_points),
    }
    evidence.append(make_atom(
        "ob.output.winner", "stage_output", str(winner), winner,
        f"The off-by-threshold test's F1 winner is {winner}."))
    required.append("ob.output.winner")
    evidence.append(make_atom(
        "ob.points", "injected_points", "injection", int(n_points),
        f"{int(n_points)} borderline points were injected around the decision boundary."))

    conf: Dict[str, Any] = {}
    caveats = [
        make_atom("ob.caveat.f1_side", "caveat", "scope", None,
                  "Correctness is judged on thresholded predictions (the F1 side); "
                  "PR-AUC has no per-point notion of correct or incorrect."),
    ]

    agg_imp: Dict[str, List[float]] = {fn: [] for fn in feature_names}
    # Only the top-3 competitors by exclusive wins are REQUIRED content — with
    # ~10 competitors a fixed word budget cannot convey them all, and marking
    # every wins atom required would inflate the omission metric artifactually.
    non_degenerate = [k for k in per_comp if not per_comp[k].get("degenerate")]
    top_required = set(sorted(non_degenerate,
                              key=lambda k: per_comp[k].get("n_exclusive_wins", 0),
                              reverse=True)[:3])
    # Rules are deduplicated across competitors: the same winner-only condition
    # often separates the winner from several rivals (e.g. all LOF variants),
    # and repeating it per competitor wastes prompt budget.
    rule_groups: Dict[str, Dict[str, Any]] = {}
    for k in sorted(per_comp.keys()):
        info = per_comp[k]
        n_w = info.get("n_exclusive_wins", 0)
        rate = info.get("exclusive_win_rate", 0.0)
        if info.get("degenerate"):
            evidence.append(make_atom(
                f"ob.vs.{k}.degenerate", "degenerate_comparison", k,
                {"n_exclusive_wins": int(n_w)}, str(info.get("rules_text", ""))))
            continue
        sup = support_grade(n_w)
        wid = f"ob.vs.{k}.wins"
        evidence.append(make_atom(
            wid, "exclusive_wins", k, {"count": int(n_w), "rate": _val(rate, 4)},
            f"{winner} correctly handles {int(n_w)} injected point(s) "
            f"({_fmt(100.0 * rate, 2)}%) that {k} misses.", confidence=sup))
        if k in top_required:
            required.append(wid)

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
            top_f = max(imps.items(), key=lambda kv: kv[1])
            evidence.append(make_atom(
                f"ob.vs.{k}.importance", "feature_importance", k,
                {"feature": top_f[0], "importance": _val(top_f[1], 3)},
                f"The property that best separates those points is {top_f[0]} "
                f"(importance {_fmt(top_f[1], 2)})."))
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
            caveats.append(make_atom(
                f"ob.caveat.support.{k}", "caveat", k, int(n_w),
                f"The rule for {k} rests on only {int(n_w)} exclusive-win point(s) — "
                f"fewer than the {N_CV_FOLDS} cross-validation folds — so its held-out "
                f"fidelity is unstable; treat it as indicative."))

    # Emit one atom per distinct rule, naming every competitor it separates.
    for gi, sig in enumerate(sorted(rule_groups)):
        grp = rule_groups[sig]
        comps = sorted(set(grp["competitors"]))
        cond = " and ".join(f"{c['feature']} {c['op']} {c['threshold']}"
                            for c in grp["rule"]["conditions"]) or "always"
        evidence.append(make_atom(
            f"ob.rule.{gi}", "surrogate_rule", winner,
            {"conditions": grp["rule"]["conditions"], "competitors": comps},
            f"{winner} uniquely beats {', '.join(comps)} when: {cond}.",
            confidence=grp["support"]))

    mean_imp = {fn: float(np.mean(v)) for fn, v in agg_imp.items() if v}
    if mean_imp and any(mean_imp.values()):
        top = max(mean_imp.items(), key=lambda kv: kv[1])
        evidence.append(make_atom(
            "ob.summary.top_feature", "summary", top[0], _val(top[1], 3),
            f"Across all competitors, the winner's edge is best explained by "
            f"{top[0]} (mean importance {_fmt(top[1], 2)})."))
        required.append("ob.summary.top_feature")

    return _envelope("off_by_threshold", dataset, entity, output, evidence,
                     caveats, required, confidence=conf)


# ── Global assembly ──────────────────────────────────────────────────────────

_STAGE_FILES = {
    "thompson_sampling": "ir_thompson",
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
    for key, label in (("robust_agg", "robust_consensus"), ("final_agg", "final_consensus")):
        val = agg.get(key)
        try:
            stage_picks[label] = val[1][0] if val and len(val) > 1 and val[1] else NOT_AVAILABLE
        except (TypeError, IndexError):
            stage_picks[label] = NOT_AVAILABLE
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
