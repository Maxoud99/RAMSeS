"""
Standalone unit tests for the Intermediate Representation (IR) layer.
Loads Explainability/ir.py by file path (numpy + stdlib only); the tree-rule
test importorskips sklearn.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ir", os.path.join(_THIS, "ir.py"))
ir = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ir)


# ── Fixtures shaped like the verified explain_* returns ─────────────────────

def _ga_selection_result():
    return {
        "best_ensemble": ["A", "B"],
        "lofo": {"A": 0.05, "B": -0.02},
        "mean_marginal": {"A": {"contribution": 0.12}, "B": {"contribution": 0.08},
                          "C": {"contribution": 0.15}},
        "survival": {"A": [0.5, 0.75], "B": [0.25, 0.5], "C": [0.5, 0.25]},
        "archetypes": {
            "A": {"utility": 0.12, "stability_mean": 0.625,
                  "relative": {"archetype": "HH"}, "absolute": {"archetype": "HH"}},
            "B": {"utility": 0.08, "stability_mean": 0.375,
                  "relative": {"archetype": "LL"}, "absolute": {"archetype": "LL"}},
            "C": {"utility": 0.15, "stability_mean": 0.375,
                  "relative": {"archetype": "HL"}, "absolute": {"archetype": "HL"}},
        },
        "n_subsets_evaluated": 9, "n_generations": 3,
    }


def _ga_combination_result():
    return {
        "best_ensemble": ["A", "B"], "feature_names": ["A", "B"],
        "meta_model_type": "rf", "model_source": "captured", "baseline_f1": 0.87,
        "shap_importance": {"A": 0.4, "B": 0.1},
        "shap_signed_importance": {"A": 0.35, "B": -0.05},
        "pfi_importance": {"A": 0.2, "B": 0.01},
        "markov_scores": {"A": 0.7, "B": 0.3},
        "final_ranking": ["A", "B"],
    }


def _rank_agg_result(two_sources=False):
    verdicts = [
        {"source": "S1", "loo_score": 0.3, "loo_rank": 1, "align_score": 0.6,
         "align_rank": 2, "borda_count": 3.0, "borda_rank": 1,
         "pattern": "influential_disagreer", "lo_align_rank_delta": 1.0},
        {"source": "S2", "loo_score": 0.1, "loo_rank": 2, "align_score": 0.8,
         "align_rank": 1, "borda_count": 3.0, "borda_rank": 1,
         "pattern": "redundant_agreer", "lo_align_rank_delta": 1.0},
    ]
    kendall = ({"align_scores": {"S1": 0.6, "S2": 0.8}, "winner": "S2",
                "winner_tau": 0.8, "runner_up": "S1", "runner_up_tau": 0.6,
                "alignment_gap": 0.2} if two_sources else None)
    return {"loo_scores": {"S1": 0.3, "S2": 0.1},
            "align_scores": {"S1": 0.6, "S2": 0.8},
            "borda_counts": {"S1": 3.0, "S2": 3.0},
            "verdicts": verdicts, "prominent_contradictions": [verdicts[0]],
            "kendall_only": kendall}


def _mc_result():
    curves = {
        "grid_levels": np.array([0.0, 0.1, 0.2]),
        "win_regions": {"A": [(0.0, 0.1)], "B": [(0.2, 0.2)]},
        "crossovers": [{"noise": 0.2, "from_model": "A", "to_model": "B"}],
        "breakdown_points": {"A": None, "B": 0.0},
    }
    return {
        "curves_f1": curves, "curves_pr": curves, "curves_f1_fixed": curves,
        "winner_f1": {"feasible": True, "train_accuracy": 0.95, "cv_accuracy": 0.85,
                      "win_rates": {"A": 0.6, "B": 0.4},
                      "rules": [{"conditions": [{"feature": "noise_level", "op": "<=",
                                                 "threshold": 0.15}],
                                 "outcome": "A", "n_samples": 10}],
                      "rules_text": "", "classes": ["A", "B"], "root_threshold": 0.15},
        "winner_pr": {"feasible": False},
        "permodel_f1": {"A": {"cv_r2": 0.7, "trend": "robust"},
                        "B": {"cv_r2": float("nan"), "trend": "fragile"}},
        "permodel_pr": {}, "n_trials": 30,
    }


class _FakeInfoClf:  # placeholder object; builder must not touch it when rules exist
    pass


def _off_by_result(n_wins=8):
    return {
        "table": {"n_points": 40},
        "winner": "A", "runnerup": "B", "n_points": 40,
        "surrogates": {
            "feasible": True, "winner": "A",
            "feature_names": ["boundary_distance", "local_std"],
            "per_competitor": {
                "B": {"degenerate": False, "clf": None,
                      "feature_importances": {"boundary_distance": 0.9, "local_std": 0.1},
                      "train_accuracy": 1.0, "cv_accuracy": 0.75,
                      "n_exclusive_wins": n_wins, "exclusive_win_rate": n_wins / 40.0,
                      "rules_text": "..."},
                "C": {"degenerate": True, "clf": None, "feature_importances": {},
                      "train_accuracy": float("nan"), "n_exclusive_wins": 0,
                      "exclusive_win_rate": 0.0,
                      "rules_text": "A has no exclusive wins over C."},
            },
        },
    }


def _thompson_kwargs():
    return dict(
        n_windows=6,
        final_ranking=[("A", 1.5), ("B", 0.7)],
        regimes=[{"index": 0, "start": 0, "end": 2, "duration": 3, "leader": "A",
                  "rewards_top": [("A", 0.5), ("B", 0.2)], "reward_gap": 0.3,
                  "runner_up": "B",
                  "shap_raising": [(0, 0.4)], "shap_lowering": [(2, -0.1)],
                  "pref_favor_leader": [(0, 0.3)],
                  "pref_favor_runner": [(3, -0.05)], "pref_gap": 0.3},
                 {"index": 1, "start": 3, "end": 5, "duration": 3, "leader": "B",
                  "rewards_top": [("B", 0.6), ("A", 0.4)], "reward_gap": 0.2,
                  "runner_up": "A", "shap_raising": None, "shap_lowering": None,
                  "pref_favor_leader": None, "pref_favor_runner": None,
                  "pref_gap": float("nan")}],
        shifts=[{"window": 3, "from_model": "A", "to_model": "B",
                 "reward_delta": 0.2, "regime_length": 3}],
        blip_count=1,
        state_fractions={"random": 0.2, "exploitation": 0.6, "informed_exploration": 0.2},
        final_state="exploitation",
    )


def _results_dict():
    return {
        "thompson": {"best_model": "A"},
        "gan_robustness": {"best_model": "A"},
        "borderline": {"best_model": "B"},
        "monte_carlo": {"best_model_f1": "A"},
        "aggregation": {"robust_agg": (0.5, ["A", "B"]), "final_agg": (0.4, ["A", "B"])},
        "final_decision": {"framework_choice": "ensemble", "chosen_model": ["A", "B"],
                           "ensemble": ["A", "B"], "ensemble_f1": 0.9,
                           "ensemble_pr_auc": 0.8, "single_model": "A",
                           "single_model_f1": 0.85, "single_model_pr_auc": 0.75},
    }


# ── Schema helpers ───────────────────────────────────────────────────────────

def _check_envelope(tc, doc, stage):
    tc.assertEqual(doc["ir_version"], ir.IR_VERSION)
    tc.assertEqual(doc["stage"], stage)
    ids = [a["id"] for a in doc["evidence"]]
    tc.assertEqual(len(ids), len(set(ids)), "atom ids must be unique")
    for a in doc["evidence"] + doc["caveats"]:
        for key in ("id", "type", "subject", "value", "text"):
            tc.assertIn(key, a)
    all_ids = set(ids)
    for rid in doc["required_atom_ids"]:
        tc.assertIn(rid, all_ids, f"required id {rid} missing from evidence")
    json.dumps(doc)  # must be JSON-serialisable


# ════════════════════════════════════════════════════════════════════════════

class TestCore(unittest.TestCase):

    def test_fmt_and_val_nan(self):
        self.assertEqual(ir._fmt(float("nan")), ir.NOT_AVAILABLE)
        self.assertEqual(ir._val(None), ir.NOT_AVAILABLE)
        self.assertEqual(ir._fmt(0.28713), "0.287")
        self.assertEqual(ir._val(0.28713), 0.287)

    def test_fidelity_grade(self):
        self.assertEqual(ir.fidelity_grade(0.9), "high")
        self.assertEqual(ir.fidelity_grade(0.7), "medium")
        self.assertEqual(ir.fidelity_grade(0.3), "low")
        self.assertEqual(ir.fidelity_grade(float("nan")), ir.NOT_AVAILABLE)

    def test_support_grade_anchored_to_folds(self):
        self.assertEqual(ir.support_grade(ir.N_CV_FOLDS), "adequate")
        self.assertEqual(ir.support_grade(ir.N_CV_FOLDS - 1), "low")


class TestTreeToRules(unittest.TestCase):

    def test_1d_intervals(self):
        import importlib
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("scikit-learn not installed")
        from sklearn.tree import DecisionTreeClassifier
        X = np.array([[0.0], [0.05], [0.1], [0.3], [0.35], [0.4]])
        y = np.array(["A", "A", "A", "B", "B", "B"])
        clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
        rules = ir.tree_to_rules(clf, ["noise_level"])
        self.assertEqual(len(rules), 2)
        outcomes = {r["outcome"] for r in rules}
        self.assertEqual(outcomes, {"A", "B"})
        thr = rules[0]["conditions"][0]["threshold"]
        self.assertTrue(0.1 < thr < 0.3)
        self.assertTrue(all(r["n_samples"] == 3 for r in rules))
        self.assertIn("noise_level", ir.rule_to_text(rules[0]))


class TestBuilders(unittest.TestCase):

    def test_thompson(self):
        doc = ir.build_thompson_ir("DS", "e1", **_thompson_kwargs())
        _check_envelope(self, doc, "thompson_sampling")
        self.assertEqual(doc["output"]["top_pick"], "A")
        ids = {a["id"] for a in doc["evidence"]}
        # one span + rewards block per regime; SHAP only where provided
        self.assertIn("ts.regime.0.span", ids)
        self.assertIn("ts.regime.1.span", ids)
        self.assertIn("ts.regime.0.shap", ids)
        self.assertNotIn("ts.regime.1.shap", ids)
        self.assertIn("ts.regime.0.pref", ids)
        self.assertIn("ts.shift.0", ids)
        # Channels arrive sign-grouped; both directions are spelled out.
        shap = next(a for a in doc["evidence"] if a["id"] == "ts.regime.0.shap")
        self.assertIn("channels raising A's expected reward", shap["text"])
        self.assertIn("channels lowering it", shap["text"])
        # Preference atom names the favored side and groups channels by sign.
        pref = next(a for a in doc["evidence"] if a["id"] == "ts.regime.0.pref")
        self.assertIn("favors A over B by 0.3", pref["text"])
        self.assertIn("Channels favoring A:", pref["text"])
        self.assertIn("Channels favoring B:", pref["text"])
        # The mean-reward gap is labelled distinctly from the preference score.
        rewards = next(a for a in doc["evidence"] if a["id"] == "ts.regime.0.rewards")
        self.assertIn("mean-reward gap", rewards["text"])

    def test_thompson_pref_direction_honest_when_gap_negative(self):
        """A negative preference score must be stated as favoring the runner,
        never left as a signed number next to 'preferred'."""
        kwargs = _thompson_kwargs()
        kwargs["regimes"][0]["pref_gap"] = -0.12
        kwargs["regimes"][0]["pref_favor_leader"] = [(1, 0.02)]
        kwargs["regimes"][0]["pref_favor_runner"] = [(0, -0.14)]
        doc = ir.build_thompson_ir("DS", "e1", **kwargs)
        pref = next(a for a in doc["evidence"] if a["id"] == "ts.regime.0.pref")
        self.assertIn("although A led selections", pref["text"])
        self.assertIn("favors B by 0.12", pref["text"])
        self.assertNotIn("-0.12", pref["text"])

    def test_ga_selection_no_complementarity(self):
        doc = ir.build_ga_selection_ir("DS", "e1", _ga_selection_result())
        _check_envelope(self, doc, "ga_selection")
        self.assertNotIn("complementarity", json.dumps(doc).lower())
        ids = {a["id"] for a in doc["evidence"]}
        self.assertIn("ga_sel.member.A.card", ids)
        self.assertIn("ga_sel.member.B.card", ids)
        # B: lofo -0.02 vs mean_marginal +0.08 → sign disagreement flagged.
        self.assertIn("ga_sel.member.B.disagreement", ids)
        self.assertNotIn("ga_sel.member.A.disagreement", ids)
        # C is the top excluded detector.
        exc = next(a for a in doc["evidence"] if a["id"] == "ga_sel.excluded.top_utility")
        self.assertEqual(exc["subject"], "C")

    def test_ga_selection_member_card_is_self_contained(self):
        """One atom per member carries archetype, utility, stability, and LOFO
        together, so the narrator never re-associates values across
        detectors; every card is required."""
        doc = ir.build_ga_selection_ir("DS", "e1", _ga_selection_result())
        card = next(a for a in doc["evidence"] if a["id"] == "ga_sel.member.A.card")
        self.assertEqual(card["type"], "member_card")
        self.assertEqual(card["value"]["archetype"], "HH")
        self.assertEqual(card["value"]["utility"], 0.12)
        self.assertEqual(card["value"]["lofo"], 0.05)
        for fragment in ("archetype HH", "high utility, high stability",
                         "mean marginal contribution 0.12",
                         "survived 62.5% of GA generations",
                         "LOFO 0.05"):
            self.assertIn(fragment, card["text"])
        self.assertIn("ga_sel.member.A.card", doc["required_atom_ids"])
        self.assertIn("ga_sel.member.B.card", doc["required_atom_ids"])
        # The old four-atoms-per-member layout is gone.
        ids = {a["id"] for a in doc["evidence"]}
        for stale in ("ga_sel.member.A.archetype", "ga_sel.member.A.utility",
                      "ga_sel.member.A.stability", "ga_sel.member.A.lofo"):
            self.assertNotIn(stale, ids)

    def test_ga_combination_no_matrix(self):
        doc = ir.build_ga_combination_ir("DS", "e1", _ga_combination_result())
        _check_envelope(self, doc, "ga_combination")
        self.assertEqual(doc["output"]["top_pick"], "A")
        # A is rank 1 by all three methods → agreement atom.
        ids = {a["id"] for a in doc["evidence"]}
        self.assertIn("ga_comb.detector.A.agreement", ids)
        m = next(a for a in doc["evidence"] if a["id"] == "ga_comb.detector.B.methods")
        self.assertEqual(m["value"]["signed_direction"], "negative")
        # Every method rank AND the final Markov rank are explicit — the LLM
        # must never have to infer a rank from list order.
        self.assertEqual(m["value"]["final_rank"], 2)
        self.assertEqual(m["value"]["signed_shap_rank"], 2)
        self.assertIn("final rank 2", m["text"])
        # Ensemble-membership relation is a required atom.
        self.assertIn("ga_comb.context.members", doc["required_atom_ids"])
        members = next(a for a in doc["evidence"] if a["id"] == "ga_comb.context.members")
        self.assertIn("every ranked detector is part of that ensemble", members["text"])
        self.assertEqual(doc["output"]["ensemble_members"], ["A", "B"])

    def test_rank_aggregation_robust_and_final(self):
        robust = ir.build_rank_aggregation_ir(
            "DS", "e1", "robust", 0, _rank_agg_result(False),
            ["S1", "S2"], {"S1": "A", "S2": "B"}, ["A", "B"])
        _check_envelope(self, robust, "rank_aggregation_robust")
        self.assertEqual(robust["output"]["top_pick"], "A")
        ids = {a["id"] for a in robust["evidence"]}
        self.assertIn("ra_robust.source.S1.verdict", ids)
        self.assertNotIn("ra_robust.kendall_only.winner", ids)

        final = ir.build_rank_aggregation_ir(
            "DS", "e1", "final", 0, _rank_agg_result(True),
            ["S1", "S2"], {"S1": "A", "S2": "A"}, ["A", "B"])
        _check_envelope(self, final, "rank_aggregation_final")
        ids = {a["id"] for a in final["evidence"]}
        self.assertIn("ra_final.kendall_only.winner", ids)
        cav = {c["id"] for c in final["caveats"]}
        self.assertIn("ra_final.caveat.two_sources", cav)

    def test_rank_aggregation_presentation_order_by_borda(self):
        """Sources are presented best Borda rank first, the consensus pick
        leads, and each source's top-pick atom follows its verdict."""
        result = _rank_agg_result(False)
        result["verdicts"][0]["borda_rank"] = 2  # S1 second
        result["verdicts"][1]["borda_rank"] = 1  # S2 first
        doc = ir.build_rank_aggregation_ir(
            "DS", "e1", "robust", 0, result,
            ["S1", "S2"], {"S1": "A", "S2": "B"}, ["A", "B"])
        atoms = {a["id"]: a for a in doc["evidence"]}
        self.assertEqual(atoms["ra_robust.output.top"]["order"], 0)
        self.assertLess(atoms["ra_robust.source.S2.verdict"]["order"],
                        atoms["ra_robust.source.S1.verdict"]["order"])
        self.assertEqual(atoms["ra_robust.source.S2.top_pick"]["order"],
                         atoms["ra_robust.source.S2.verdict"]["order"] + 1)

    def test_monte_carlo_lean(self):
        doc = ir.build_monte_carlo_ir("DS", "e1", _mc_result(), ["A", "B"], ["B", "A"])
        _check_envelope(self, doc, "monte_carlo")
        blob = json.dumps(doc)
        # Lean IR: no breakdown / trend / tau content.
        self.assertNotIn("breakdown", blob)
        self.assertNotIn("robust\"", blob)
        self.assertNotIn("fragile", blob)
        ids = {a["id"] for a in doc["evidence"]}
        self.assertIn("mc.win_region.f1.A", ids)
        self.assertIn("mc.crossover.f1.0", ids)
        self.assertIn("mc.surrogate.rule.0", ids)
        conf = doc["confidence"]
        self.assertEqual(conf["winner_surrogate_f1"]["grade"], "high")
        # Per-model cv R² is graded confidence data, number kept visible.
        self.assertEqual(conf["permodel_cv_r2"]["B"]["cv_r2"], ir.NOT_AVAILABLE)
        self.assertEqual(conf["permodel_cv_r2"]["A"]["cv_r2"], 0.7)
        self.assertIn("grade", conf["permodel_cv_r2"]["A"])

    def test_mc_majority_degenerate_cv_r2_graded_not_available(self):
        result = _mc_result()
        # A: 4 of 5 folds degenerate → number kept, graded not_available.
        # B: 1 of 5 → graded normally.
        result["permodel_f1"] = {
            "A": {"cv_r2": 0.6, "cv_n_splits": 5, "cv_degenerate_folds": 4},
            "B": {"cv_r2": 0.9, "cv_n_splits": 5, "cv_degenerate_folds": 1},
        }
        doc = ir.build_monte_carlo_ir("DS", "e1", result, ["A", "B"], ["B", "A"])
        conf = doc["confidence"]["permodel_cv_r2"]
        self.assertEqual(conf["A"]["cv_r2"], 0.6)          # number stays visible
        self.assertEqual(conf["A"]["grade"], ir.NOT_AVAILABLE)
        self.assertEqual(conf["A"]["n_degenerate_folds"], 4)
        self.assertEqual(conf["B"]["cv_r2"], 0.9)
        self.assertEqual(conf["B"]["grade"], "high")
        # A caveat names the majority-degenerate model.
        cav = next(c for c in doc["caveats"] if c["id"] == "mc.caveat.cv_degenerate")
        self.assertIn("A", cav["value"])
        self.assertNotIn("B", cav["value"])
        self.assertIn("not a meaningful fidelity estimate", cav["text"])

    def test_off_by_support_gate(self):
        low = ir.build_off_by_ir("DS", "e1", _off_by_result(n_wins=2), ["A", "B"])
        _check_envelope(self, low, "off_by_threshold")
        self.assertEqual(low["confidence"]["surrogate_vs_B"]["support"], "low")
        self.assertIn("ob.caveat.support.B", {c["id"] for c in low["caveats"]})

        ok = ir.build_off_by_ir("DS", "e1", _off_by_result(n_wins=8), ["A", "B"])
        self.assertEqual(ok["confidence"]["surrogate_vs_B"]["support"], "adequate")
        self.assertNotIn("ob.caveat.support.B", {c["id"] for c in ok["caveats"]})
        # Degenerate competitor appears as a text atom.
        ids = {a["id"] for a in ok["evidence"]}
        self.assertIn("ob.vs.C.degenerate", ids)

    def test_off_by_required_curated_to_top3(self):
        # 4 non-degenerate competitors with wins 10/8/6/1 → only the top 3
        # wins atoms are required (all 4 remain as evidence).
        result = _off_by_result(n_wins=10)
        pc = result["surrogates"]["per_competitor"]
        for name, wins in (("D", 8), ("E", 6), ("F", 1)):
            pc[name] = dict(pc["B"], n_exclusive_wins=wins,
                            exclusive_win_rate=wins / 40.0)
        doc = ir.build_off_by_ir("DS", "e1", result, ["A", "B"])
        _check_envelope(self, doc, "off_by_threshold")
        req = set(doc["required_atom_ids"])
        self.assertIn("ob.vs.B.wins", req)   # 10 wins
        self.assertIn("ob.vs.D.wins", req)   # 8 wins
        self.assertIn("ob.vs.E.wins", req)   # 6 wins
        self.assertNotIn("ob.vs.F.wins", req)  # 1 win → optional
        self.assertIn("ob.vs.F.wins", {a["id"] for a in doc["evidence"]})

    def test_off_by_rules_deduplicated_across_competitors(self):
        import importlib
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("scikit-learn not installed")
        from sklearn.tree import DecisionTreeClassifier
        X = np.array([[0.01, 0.2], [0.02, 0.3], [0.04, 0.2], [0.05, 0.3]])
        y = np.array([1, 1, 0, 0])
        clf1 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
        clf2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
        result = _off_by_result(n_wins=8)
        pc = result["surrogates"]["per_competitor"]
        pc["B"] = dict(pc["B"], clf=clf1)
        pc["D"] = dict(pc["B"], clf=clf2)
        doc = ir.build_off_by_ir("DS", "e1", result, ["A", "B"])
        rules = [a for a in doc["evidence"] if a["type"] == "surrogate_rule"]
        # Identical rule fitted for B and D → ONE merged atom naming both.
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["value"]["competitors"], ["B", "D"])
        self.assertIn("uniquely beats B, D when:", rules[0]["text"])

    def test_mc_required_rules_curated_to_top3(self):
        result = _mc_result()
        result["winner_f1"]["rules"] = [
            {"conditions": [{"feature": "noise_level", "op": "<=", "threshold": 0.05}],
             "outcome": "A", "n_samples": 50},
            {"conditions": [{"feature": "noise_level", "op": ">", "threshold": 0.15}],
             "outcome": "B", "n_samples": 30},
            {"conditions": [{"feature": "noise_level", "op": "<=", "threshold": 0.1}],
             "outcome": "A", "n_samples": 10},
            {"conditions": [{"feature": "noise_level", "op": ">", "threshold": 0.05}],
             "outcome": "B", "n_samples": 2},
        ]
        doc = ir.build_monte_carlo_ir("DS", "e1", result, ["A", "B"], ["B", "A"])
        req = set(doc["required_atom_ids"])
        rule_atoms = [a for a in doc["evidence"]
                      if a["id"].startswith("mc.surrogate.rule.")]
        self.assertEqual(len(rule_atoms), 4)
        # The three best-supported rules are required; the 2-sample rule is
        # evidence-only. (Rules are re-ordered along the noise axis, so
        # identify them by support, not index.)
        for a in rule_atoms:
            if a["value"]["n_samples"] == 2:
                self.assertNotIn(a["id"], req)
            else:
                self.assertIn(a["id"], req)

    def test_simplify_conditions_tightest_bounds(self):
        conds = [
            {"feature": "noise_level", "op": "<=", "threshold": 0.0368},
            {"feature": "noise_level", "op": ">", "threshold": 0.0053},
            {"feature": "noise_level", "op": ">", "threshold": 0.0158},
        ]
        self.assertEqual(ir.simplify_conditions(conds), [
            {"feature": "noise_level", "op": ">", "threshold": 0.0158},
            {"feature": "noise_level", "op": "<=", "threshold": 0.0368},
        ])

    def test_merge_single_feature_rules(self):
        rules = [
            {"conditions": [{"feature": "n", "op": "<=", "threshold": 0.0053}],
             "outcome": "LOF_1", "n_samples": 5},
            {"conditions": [{"feature": "n", "op": ">", "threshold": 0.0053},
                            {"feature": "n", "op": "<=", "threshold": 0.0368}],
             "outcome": "LOF_1", "n_samples": 15},
            {"conditions": [{"feature": "n", "op": ">", "threshold": 0.0368},
                            {"feature": "n", "op": "<=", "threshold": 0.0474}],
             "outcome": "NN_3", "n_samples": 5},
            {"conditions": [{"feature": "n", "op": ">", "threshold": 0.0474}],
             "outcome": "LOF_1", "n_samples": 65},
        ]
        merged = ir.merge_single_feature_rules(rules)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["outcome"], "LOF_1")
        self.assertEqual(merged[0]["n_samples"], 20)
        self.assertEqual(merged[0]["conditions"],
                         [{"feature": "n", "op": "<=", "threshold": 0.0368}])
        # Rules over multiple features pass through unchanged.
        multi = [{"conditions": [{"feature": "a", "op": "<=", "threshold": 1.0}],
                  "outcome": "x", "n_samples": 1},
                 {"conditions": [{"feature": "b", "op": ">", "threshold": 2.0}],
                  "outcome": "x", "n_samples": 1}]
        self.assertEqual(ir.merge_single_feature_rules(multi), multi)

    def test_mc_rules_simplified_merged_and_labeled(self):
        result = _mc_result()
        result["winner_f1"]["rules"] = [
            {"conditions": [{"feature": "noise_level", "op": "<=", "threshold": 0.0368},
                            {"feature": "noise_level", "op": "<=", "threshold": 0.0053}],
             "outcome": "A", "n_samples": 5},
            {"conditions": [{"feature": "noise_level", "op": "<=", "threshold": 0.0368},
                            {"feature": "noise_level", "op": ">", "threshold": 0.0053}],
             "outcome": "A", "n_samples": 15},
            {"conditions": [{"feature": "noise_level", "op": ">", "threshold": 0.0368}],
             "outcome": "B", "n_samples": 80},
        ]
        doc = ir.build_monte_carlo_ir("DS", "e1", result, ["A", "B"], ["B", "A"])
        rule_atoms = [a for a in doc["evidence"]
                      if a["id"].startswith("mc.surrogate.rule.")]
        # The two adjacent A-leaves collapse into one interval fact.
        self.assertEqual(len(rule_atoms), 2)
        merged = next(a for a in rule_atoms if a["value"]["outcome"] == "A")
        self.assertEqual(merged["value"]["n_samples"], 20)
        self.assertEqual(merged["value"]["conditions"],
                         [{"feature": "noise_level", "op": "<=", "threshold": 0.0368}])
        # The rule text names what the outcome means.
        for a in rule_atoms:
            self.assertIn("the noise-sweep F1 winner is", a["text"])
            self.assertNotIn("the outcome is", a["text"])

    def test_mc_win_regions_compress_isolated_points(self):
        result = _mc_result()
        result["curves_f1"]["win_regions"] = {"A": [(0.0, 0.1), (0.15, 0.15)],
                                              "B": [(0.2, 0.2)]}
        doc = ir.build_monte_carlo_ir("DS", "e1", result, ["A", "B"], [])
        a_atom = next(x for x in doc["evidence"] if x["id"] == "mc.win_region.f1.A")
        self.assertIn("for noise levels in [", a_atom["text"])
        self.assertIn("isolated noise level", a_atom["text"])
        b_atom = next(x for x in doc["evidence"] if x["id"] == "mc.win_region.f1.B")
        self.assertIn("isolated noise level", b_atom["text"])
        self.assertNotIn("[", b_atom["text"])

    def test_determinism(self):
        a = json.dumps(ir.build_ga_combination_ir("DS", "e1", _ga_combination_result()),
                       sort_keys=True)
        b = json.dumps(ir.build_ga_combination_ir("DS", "e1", _ga_combination_result()),
                       sort_keys=True)
        self.assertEqual(a, b)


class TestWriterAndAssembler(unittest.TestCase):

    def test_write_and_assemble(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "explanations_ir")
            ir.write_stage_ir(ir.build_ga_combination_ir("DS", "e1", _ga_combination_result()),
                              "DS", "e1", "ir_ga_combination", base_dir=base)
            ir.write_stage_ir(ir.build_monte_carlo_ir("DS", "e1", _mc_result(), ["A"], ["A"]),
                              "DS", "e1", "ir_monte_carlo", base_dir=base)
            path = ir.assemble_global_ir(_results_dict(), "DS", "e1", 5, base_dir=base)
            with open(path) as f:
                g = json.load(f)
            self.assertEqual(g["stage"], "global")
            self.assertEqual(g["decision"]["framework_choice"], "ensemble")
            self.assertEqual(g["stages"]["ga_combination"]["status"], "ok")
            self.assertEqual(g["stages"]["monte_carlo"]["status"], "ok")
            # Missing stages and GAN are explicit, never silent.
            self.assertEqual(g["stages"]["thompson_sampling"]["status"], ir.NOT_AVAILABLE)
            self.assertEqual(g["stages"]["gan"]["status"], ir.NOT_AVAILABLE)
            # Agreement facts computed in code.
            self.assertTrue(g["stage_agreement"]["thompson"]["agrees_with_final_single"])
            self.assertFalse(g["stage_agreement"]["borderline"]["agrees_with_final_single"])
            # The global IR carries its own sentence atoms + required ids.
            ids = {a["id"] for a in g["evidence"]}
            self.assertIn("global.decision", ids)
            self.assertIn("global.stage.ga_combination", ids)
            self.assertIn("global.agreement.thompson", ids)
            self.assertIn("global.decision", g["required_atom_ids"])
            self.assertIn("global.stage.monte_carlo", g["required_atom_ids"])
            dec = next(a for a in g["evidence"] if a["id"] == "global.decision")
            self.assertIn("The final decision is the ensemble", dec["text"])

    def test_assembler_glob_fallback_for_rank_agg(self):
        ra_result = {"loo_scores": {"S1": 0.3, "S2": 0.1},
                     "align_scores": {"S1": 0.6, "S2": 0.8},
                     "borda_counts": {"S1": 3.0, "S2": 3.0},
                     "verdicts": [], "prominent_contradictions": [],
                     "kendall_only": None}
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "explanations_ir")
            # Written under iteration 0, assembled under iteration 5.
            ir.write_stage_ir(
                ir.build_rank_aggregation_ir("DS", "e1", "robust", 0, ra_result,
                                             ["S1", "S2"], {"S1": "A", "S2": "B"},
                                             ["A", "B"]),
                "DS", "e1", "ir_rank_aggregation_robust_0", base_dir=base)
            path = ir.assemble_global_ir(_results_dict(), "DS", "e1", 5, base_dir=base)
            with open(path) as f:
                g = json.load(f)
            self.assertEqual(g["stages"]["rank_aggregation_robust"]["status"], "ok")

    def test_assembler_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "explanations_ir")
            p1 = ir.assemble_global_ir(_results_dict(), "DS", "e1", 5, base_dir=base)
            with open(p1) as f:
                b1 = f.read()
            p2 = ir.assemble_global_ir(_results_dict(), "DS", "e1", 5, base_dir=base)
            with open(p2) as f:
                b2 = f.read()
            self.assertEqual(b1, b2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
