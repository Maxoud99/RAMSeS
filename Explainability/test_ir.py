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
    # A: rank 1 on all three (positive); B: rank 2 on |SHAP|+signed but 3 on PFI
    # (positive); C: rank 3 on |SHAP|+signed but 2 on PFI (negative). Exercises
    # the shared-rank collapse and the positive/negative sign grouping.
    return {
        "best_ensemble": ["A", "B", "C"], "feature_names": ["A", "B", "C"],
        "meta_model_type": "rf", "model_source": "captured", "baseline_f1": 0.87,
        "shap_importance": {"A": 0.4, "B": 0.2, "C": 0.1},
        "shap_signed_importance": {"A": 0.35, "B": 0.15, "C": -0.05},
        "pfi_importance": {"A": 0.2, "B": 0.05, "C": 0.08},
        "markov_scores": {"A": 0.5, "B": 0.3, "C": 0.2},
        "final_ranking": ["A", "B", "C"],
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
        by_id = {a["id"]: a for a in doc["evidence"]}

        # Lead carries the forwarded score AND the margin over the runner-up.
        self.assertEqual(
            by_id["ts.output.top"]["text"],
            "Thompson Sampling ranked A first with a final score of 1.500000, "
            "ahead of B by 0.800000.")

        # ONE atom per regime: span + leader + channels in a single sentence.
        self.assertEqual(
            by_id["ts.regime.0"]["text"],
            "Regime 0 (windows 0 to 2, 3 windows) was led by A. Channel 0 raised "
            "its expected reward the most, with channel 0 also giving it its "
            "biggest edge over B.")
        # Regime 1 has no SHAP/preference data -> just the span sentence.
        self.assertEqual(
            by_id["ts.regime.1"]["text"],
            "Regime 1 (windows 3 to 5, 3 windows) was led by B.")
        for rid in ("ts.regime.0", "ts.regime.1", "ts.regimes.summary",
                    "ts.output.top", "ts.states.summary"):
            self.assertIn(rid, doc["required_atom_ids"])

        # Regime summary counts regimes and distinct leaders.
        self.assertIn("split into 2 regimes led by 2 different detectors",
                      by_id["ts.regimes.summary"]["text"])
        self.assertIn("blip window", by_id["ts.regimes.summary"]["text"])

        # States are narrated as shares, best-first; no final-state atom.
        self.assertIn("exploitation 60.0% of the time",
                      by_id["ts.states.summary"]["text"])
        self.assertNotIn("ts.states.final", by_id)

        # The per-regime split, the shift atoms and the blip atom are gone.
        for stale in ("ts.regime.0.span", "ts.regime.0.shap", "ts.regime.0.pref",
                      "ts.regime.0.rewards", "ts.shift.0", "ts.shifts.count",
                      "ts.blips.count"):
            self.assertNotIn(stale, by_id)

        # Raw reward numbers stay in `value`, out of the prose.
        self.assertEqual(by_id["ts.regime.0"]["value"]["mean_reward_gap"], 0.3)
        prose = " ".join(a["text"] for a in doc["evidence"])
        self.assertNotIn("0.3000", prose)

        # Envelope: headline question + glossary; the three run-invariant
        # caveats moved into the footer, leaving no per-run caveat here.
        self.assertIn("how much of the run was spent exploring", doc["question"])
        self.assertIn("expected reward", doc["info_footer"])
        self.assertEqual(doc["caveats"], [])

    def test_thompson_regime_channels_kept_with_their_own_regime(self):
        """Each regime's channels are named inside that regime's sentence, and a
        differing edge channel is reported separately from the raising ones."""
        kwargs = _thompson_kwargs()
        kwargs["regimes"][0]["shap_raising"] = [(2, 0.5), (5, 0.2)]
        kwargs["regimes"][0]["pref_favor_leader"] = [(7, 0.3)]
        doc = ir.build_thompson_ir("DS", "e1", **kwargs)
        txt = next(a for a in doc["evidence"] if a["id"] == "ts.regime.0")["text"]
        self.assertIn("Channel 2 and channel 5 raised its expected reward the most",
                      txt)
        self.assertIn("channel 7 gave it its biggest edge over B", txt)

    def test_thompson_negative_pref_gap_is_not_dramatised(self):
        """A negative preference score is a level-vs-deviation difference, not a
        contradiction: the prose names the leader's best channel either way and
        never editorialises with 'although'."""
        kwargs = _thompson_kwargs()
        kwargs["regimes"][0]["pref_gap"] = -0.12
        kwargs["regimes"][0]["pref_favor_leader"] = [(1, 0.02)]
        kwargs["regimes"][0]["pref_favor_runner"] = [(0, -0.14)]
        doc = ir.build_thompson_ir("DS", "e1", **kwargs)
        atom = next(a for a in doc["evidence"] if a["id"] == "ts.regime.0")
        self.assertIn("channel 1 gave it its biggest edge over B", atom["text"])
        for word in ("although", "however", "despite", "-0.12"):
            self.assertNotIn(word, atom["text"])
        # The signed gap is still grounded in `value` for the verifier.
        self.assertEqual(atom["value"]["preference_score_gap"], -0.12)

    def test_thompson_channel_names_used_when_available(self):
        kwargs = _thompson_kwargs()
        kwargs["regimes"][0]["shap_raising"] = [(1, 0.5)]
        kwargs["regimes"][0]["pref_favor_leader"] = [(1, 0.3)]
        named = ir.build_thompson_ir(
            "DS", "e1", channel_names=["Pressure", "Accelerometer1RMS"], **kwargs)
        txt = next(a for a in named["evidence"] if a["id"] == "ts.regime.0")["text"]
        # Name is used verbatim — never lower-cased by a blanket .capitalize().
        self.assertIn("Accelerometer1RMS raised its expected reward", txt)
        self.assertNotIn("channel 1", txt)
        # Out-of-range indices fall back to the numeric form.
        short = ir.build_thompson_ir("DS", "e1", channel_names=["Pressure"], **kwargs)
        self.assertIn("channel 1", next(
            a for a in short["evidence"] if a["id"] == "ts.regime.0")["text"])

    def test_thompson_family_sweep_and_single_channel_caveat(self):
        kwargs = _thompson_kwargs()
        kwargs["final_ranking"] = [("NN_1", 1.5), ("NN_2", 1.2), ("NN_3", 0.9),
                                   ("LOF_1", 0.2)]
        doc = ir.build_thompson_ir("DS", "e1", n_channels=1, **kwargs)
        fam = next(a for a in doc["evidence"] if a["id"] == "ts.output.family")
        self.assertEqual(
            fam["text"],
            "The NN detectors took the top three places: NN_1, NN_2, and NN_3.")
        self.assertIn("single channel", doc["caveats"][0]["text"])
        # A mixed top three gets no family atom.
        kwargs["final_ranking"] = [("NN_1", 1.5), ("LOF_2", 1.2), ("NN_3", 0.9)]
        mixed = ir.build_thompson_ir("DS", "e1", **kwargs)
        self.assertNotIn("ts.output.family", {a["id"] for a in mixed["evidence"]})

    def test_ga_selection_no_archetype_codes_or_complementarity(self):
        doc = ir.build_ga_selection_ir("DS", "e1", _ga_selection_result())
        _check_envelope(self, doc, "ga_selection")
        self.assertNotIn("complementarity", json.dumps(doc).lower())
        # The prose reasons in plain high/low terms — no archetype codes or the
        # old member-card jargon. (The footer still DEFINES the terms; that's
        # its job, so the jargon ban applies to the atom texts only.)
        prose = " ".join(a["text"] for a in doc["evidence"])
        for jargon in ("archetype", "HH", "HL", "LH", "LL", "median",
                       "mean marginal contribution", "survived"):
            self.assertNotIn(jargon, prose)
        self.assertNotIn("member_card", json.dumps(doc))
        # Question + footer replace the standalone relative-threshold caveat.
        self.assertIn("why were the rest left out", doc["question"].lower())
        # The footer must not present utility/stability as the GA's selection
        # criteria: the search optimises ensemble fitness, and the two
        # properties are measured post hoc to explain the subset it reached.
        self.assertIn("never scores detectors individually", doc["info_footer"])
        self.assertIn("measured afterwards", doc["info_footer"])
        self.assertIn("Utility is a detector's mean marginal contribution",
                      doc["info_footer"])
        self.assertNotIn("ga_sel.caveat.relative", {c["id"] for c in doc["caveats"]})

    def test_ga_selection_reason_grouping(self):
        # Fixture: A = HH (both), B = LL with lofo<=0 (marginal), C = HL excluded
        # with high utility (individual "why not this one?" callout).
        doc = ir.build_ga_selection_ir("DS", "e1", _ga_selection_result())
        by_id = {a["id"]: a for a in doc["evidence"]}
        self.assertEqual(
            by_id["ga_sel.included.both"]["text"],
            "A was chosen for both high utility and high stability.")
        self.assertIn("B was low on both utility and stability",
                      by_id["ga_sel.included.marginal"]["text"])
        self.assertEqual(
            by_id["ga_sel.excluded.C"]["text"],
            "C was left out even though it had high utility but low stability.")
        # Utility/stability numbers stay in `value`, never in the prose.
        self.assertEqual(
            by_id["ga_sel.included.both"]["value"]["per_detector"]["A"],
            {"utility": 0.12, "stability": 0.625})
        for a in doc["evidence"]:
            self.assertNotIn("0.12", a["text"])
        for rid in ("ga_sel.output.ensemble", "ga_sel.included.both",
                    "ga_sel.included.marginal", "ga_sel.excluded.C"):
            self.assertIn(rid, doc["required_atom_ids"])

    def test_ga_selection_full_reason_cascade(self):
        def arch(u, s):
            return {"stability_mean": 0.5, "relative": {"u_high": u, "s_high": s}}
        result = {
            "best_ensemble": ["Mb", "Mu", "Ms", "Mn", "Mm"],
            "lofo": {"Mb": 0.1, "Mu": 0.1, "Ms": 0.1, "Mn": 0.03, "Mm": -0.01},
            "mean_marginal": {d: {"contribution": 0.1}
                              for d in ("Mb", "Mu", "Ms", "Mn", "Mm", "Xh", "Xs", "Xp")},
            "archetypes": {
                "Mb": arch(True, True), "Mu": arch(True, False),
                "Ms": arch(False, True), "Mn": arch(False, False),
                "Mm": arch(False, False), "Xh": arch(True, False),
                "Xs": arch(False, True), "Xp": arch(False, False),
                "Xn": arch(False, False),          # not in mean_marginal → no data
            },
        }
        doc = ir.build_ga_selection_ir("DS", "e1", result)
        _check_envelope(self, doc, "ga_selection")
        by_id = {a["id"]: a for a in doc["evidence"]}
        self.assertEqual(by_id["ga_sel.included.both"]["value"]["detectors"], ["Mb"])
        self.assertEqual(by_id["ga_sel.included.utility"]["value"]["detectors"], ["Mu"])
        self.assertEqual(by_id["ga_sel.included.stability"]["value"]["detectors"], ["Ms"])
        self.assertEqual(by_id["ga_sel.included.marginal"]["value"]["detectors"], ["Mm"])
        # Mn: low profile but lofo>0 → individual "needed" callout, with number.
        self.assertIn("Removing Mn lowers the ensemble's fitness by 0.0300",
                      by_id["ga_sel.needed.Mn"]["text"])
        # Excluded: high-utility anomaly individual; the rest grouped by profile.
        self.assertIn("high utility but low stability", by_id["ga_sel.excluded.Xh"]["text"])
        self.assertEqual(by_id["ga_sel.excluded.stable"]["value"]["detectors"], ["Xs"])
        self.assertEqual(by_id["ga_sel.excluded.plain"]["value"]["detectors"], ["Xp"])
        self.assertEqual(by_id["ga_sel.excluded.nodata"]["value"]["detectors"], ["Xn"])

    def test_ga_combination_no_matrix(self):
        doc = ir.build_ga_combination_ir("DS", "e1", _ga_combination_result())
        _check_envelope(self, doc, "ga_combination")
        self.assertEqual(doc["output"]["top_pick"], "A")
        self.assertEqual(doc["output"]["ensemble_size"], 3)
        by_id = {a["id"] for a in doc["evidence"]}

        # Lead atom names the subset and frames the members AS the detectors.
        self.assertIn("ga_comb.output.subset", doc["required_atom_ids"])
        lead = next(a for a in doc["evidence"] if a["id"] == "ga_comb.output.subset")
        self.assertIn("3-detector ensemble {A, B, C}", lead["text"])

        # Every member gets a role atom (no top-k cap); ordinal from final rank,
        # method ranks collapsed when shared, raw magnitudes NOT in the prose.
        for d in ("A", "B", "C"):
            self.assertIn(f"ga_comb.detector.{d}.role", by_id)
            self.assertIn(f"ga_comb.detector.{d}.role", doc["required_atom_ids"])
        # Only the two MAGNITUDE measures are quoted: signed SHAP no longer
        # feeds the aggregation, so citing its rank here would imply a
        # contribution to the weight that it does not make.
        a = next(x for x in doc["evidence"] if x["id"] == "ga_comb.detector.A.role")
        self.assertEqual(
            a["text"],
            "A carries the most weight in the ensemble, ranking 1 on absolute "
            "SHAP and PFI.")
        b = next(x for x in doc["evidence"] if x["id"] == "ga_comb.detector.B.role")
        self.assertEqual(
            b["text"],
            "B carries the second-most weight in the ensemble, ranking 2 on "
            "absolute SHAP and 3 on PFI.")
        c = next(x for x in doc["evidence"] if x["id"] == "ga_comb.detector.C.role")
        self.assertEqual(
            c["text"],
            "C carries the third-most weight in the ensemble, ranking 3 on "
            "absolute SHAP and 2 on PFI.")
        prose = " ".join(x["text"] for x in doc["evidence"])
        self.assertNotIn("signed SHAP", prose)
        # ...but the signed value and its rank stay machine-readable.
        self.assertEqual(b["value"]["signed_shap"], 0.15)
        self.assertEqual(b["value"]["signed_shap_rank"], 2)
        # No raw magnitude leaks into the prose (they stay in `value`).
        self.assertNotIn("Markov", a["text"])
        self.assertNotIn("0.35", b["text"])
        self.assertEqual(b["value"]["signed_direction"], "positive")
        self.assertEqual(c["value"]["signed_direction"], "negative")

        # One sign-summary atom classifies all members by full name.
        self.assertIn("ga_comb.sign_summary", doc["required_atom_ids"])
        sign = next(a for a in doc["evidence"] if a["id"] == "ga_comb.sign_summary")
        self.assertEqual(sign["text"], "A and B signed positive, while C signed negative.")

        # Retired atoms from the old dense layout are gone.
        for gone in ("ga_comb.output.top", "ga_comb.context.members",
                     "ga_comb.detector.A.agreement", "ga_comb.detector.A.methods"):
            self.assertNotIn(gone, by_id)

        # Envelope carries the headline question and the sign/rank glossary.
        self.assertIn("push the meta-learner's decision", doc["question"])
        self.assertIn("A positive sign means", doc["info_footer"])
        self.assertEqual(doc["output"]["ensemble_members"], ["A", "B", "C"])

    def test_rank_aggregation_robust_and_final(self):
        robust = ir.build_rank_aggregation_ir(
            "DS", "e1", "robust", 0, _rank_agg_result(False),
            ["S1", "S2"], {"S1": "A", "S2": "B"}, ["A", "B"])
        _check_envelope(self, robust, "rank_aggregation_robust")
        self.assertEqual(robust["output"]["top_pick"], "A")
        ids = {a["id"] for a in robust["evidence"]}
        # One human-readable role atom per source; the old verdict/top_pick
        # atoms and their jargon are gone.
        self.assertIn("ra_robust.source.S1.role", ids)
        self.assertNotIn("ra_robust.source.S1.verdict", ids)
        self.assertNotIn("ra_robust.source.S1.top_pick", ids)
        self.assertNotIn("ra_robust.kendall_only.winner", ids)
        blob = json.dumps(robust)
        for jargon in ("leave-one-out", "Kendall tau", "Borda-resolved", "pivotality"):
            self.assertNotIn(jargon, blob)
        self.assertIn("for influence", blob)
        self.assertIn("for agreement", blob)
        # Winner reads as a DETECTOR, not a source; a required context atom
        # names the source set and says the ranked detectors are not sources.
        self.assertIn("first-ranked detector is A", blob)
        self.assertIn("ra_robust.context.sources", robust["required_atom_ids"])
        ctx = next(a for a in robust["evidence"]
                   if a["id"] == "ra_robust.context.sources")
        self.assertIn("are the items being ranked, not sources", ctx["text"])
        self.assertIn("S1", ctx["text"])
        self.assertIn("S2", ctx["text"])
        # Friendly consensus naming + question + glossary footer.
        self.assertIn("robustness consensus", robust["question"])
        self.assertIn("influential_disagreer", robust["info_footer"])

        final = ir.build_rank_aggregation_ir(
            "DS", "e1", "final", 0, _rank_agg_result(True),
            ["S1", "S2"], {"S1": "A", "S2": "A"}, ["A", "B"])
        _check_envelope(self, final, "rank_aggregation_final")
        ids = {a["id"] for a in final["evidence"]}
        # Two-source case: NO per-source role atoms (influence/Borda degenerate);
        # a single agreement-driver sentence carries the explanation.
        self.assertIn("ra_final.kendall_only.winner", ids)
        self.assertFalse(any(i.endswith(".role") for i in ids),
                         "two-source final must not emit role atoms")
        driver = next(a for a in final["evidence"]
                      if a["id"] == "ra_final.kendall_only.winner")
        self.assertIn("drove the final consensus most", driver["text"])
        self.assertIn("agreed with the consensus more closely", driver["text"])
        cav = {c["id"] for c in final["caveats"]}
        self.assertIn("ra_final.caveat.two_sources", cav)
        # Footer is a pure agreement DEFINITION: no influence/Borda talk, and it
        # does NOT restate the two-source rationale (that lives in the caveat).
        self.assertNotIn("Influence measures", final["info_footer"])
        self.assertIn("Agreement compares", final["info_footer"])
        two_src_cav = next(c for c in final["caveats"]
                           if c["id"] == "ra_final.caveat.two_sources")
        self.assertIn("single source", two_src_cav["text"])       # rationale here
        self.assertNotIn("single source", final["info_footer"])    # not duplicated
        self.assertIn("follow more closely", final["question"])

    def test_rank_aggregation_presentation_order_by_borda(self):
        """Sources are presented best Borda rank first, the consensus pick
        leads (order 0), and the Borda-#1 source's role sentence says it
        shaped the consensus most."""
        result = _rank_agg_result(False)
        result["verdicts"][0]["borda_rank"] = 2  # S1 second
        result["verdicts"][1]["borda_rank"] = 1  # S2 first
        doc = ir.build_rank_aggregation_ir(
            "DS", "e1", "robust", 0, result,
            ["S1", "S2"], {"S1": "A", "S2": "B"}, ["A", "B"])
        atoms = {a["id"]: a for a in doc["evidence"]}
        self.assertEqual(atoms["ra_robust.output.top"]["order"], 0)
        self.assertLess(atoms["ra_robust.source.S2.role"]["order"],
                        atoms["ra_robust.source.S1.role"]["order"])
        # S2 is Borda #1 → "shaped ... most"; S1 is Borda #2 → "second most".
        self.assertIn("shaped the robustness consensus most,",
                      atoms["ra_robust.source.S2.role"]["text"])
        self.assertIn("shaped the robustness consensus second most,",
                      atoms["ra_robust.source.S1.role"]["text"])
        # Both component ranks are stated for each source (never inferred).
        self.assertIn("ranking 1 for influence and 2 for agreement",
                      atoms["ra_robust.source.S1.role"]["text"])
        # The combined (Borda) standing is carried by the ordinal, plus value.
        self.assertEqual(atoms["ra_robust.source.S2.role"]["value"]["borda_rank"], 1)
        self.assertEqual(atoms["ra_robust.source.S1.role"]["value"]["borda_rank"], 2)
        # Component ranks live in value for provenance.
        self.assertEqual(atoms["ra_robust.source.S1.role"]["value"]["influence_rank"],
                         result["verdicts"][0]["loo_rank"])

    def test_rank_aggregation_lead_states_explicit_ranks(self):
        """A source tied-top on both axes must state 'influence rank 1 and
        agreement rank 1' explicitly, not just 'leading both' — otherwise the
        narrator infers (and mis-states) the lead's ranks."""
        result = _rank_agg_result(False)
        # Make S1 the sole top on both influence and agreement, Borda #1.
        result["verdicts"][0].update(loo_rank=1, align_rank=1, borda_rank=1)
        result["verdicts"][1].update(loo_rank=2, align_rank=2, borda_rank=2)
        doc = ir.build_rank_aggregation_ir(
            "DS", "e1", "robust", 0, result,
            ["S1", "S2"], {"S1": "A", "S2": "B"}, ["A", "B"])
        lead = next(a for a in doc["evidence"]
                    if a["id"] == "ra_robust.source.S1.role")
        self.assertIn("shaped the robustness consensus most,", lead["text"])
        self.assertIn("ranking 1 for influence and 1 for agreement", lead["text"])

    def test_monte_carlo_lean(self):
        doc = ir.build_monte_carlo_ir("DS", "e1", _mc_result(), ["A", "B"], ["B", "A"])
        _check_envelope(self, doc, "monte_carlo")
        blob = json.dumps(doc)
        # Lean IR: no breakdown / trend / tau content.
        self.assertNotIn("breakdown", blob)
        self.assertNotIn("robust\"", blob)
        self.assertNotIn("fragile", blob)
        ids = {a["id"] for a in doc["evidence"]}
        # One win-region atom per DETECTOR (both metrics in one sentence); the
        # crossover and surrogate-rule atoms are gone — a crossover is the
        # derivative of the regions and the rules restate them in fitted form.
        self.assertIn("mc.win_region.A", ids)
        self.assertNotIn("mc.win_region.f1.A", ids)
        self.assertNotIn("mc.crossover.f1.0", ids)
        self.assertNotIn("mc.surrogate.rule.0", ids)
        # Lead names BOTH production winners (they differ in this fixture).
        lead = next(a for a in doc["evidence"] if a["id"] == "mc.output.top")
        self.assertEqual(
            lead["text"],
            "In the production Monte Carlo test, A ranked first by F1 score "
            "and B ranked first by PR-AUC.")
        self.assertIn("mc.surrogate.win_rates", doc["required_atom_ids"])
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
        # Low-support caveats are consolidated into ONE atom naming the rules.
        self.assertIn("ob.caveat.support", {c["id"] for c in low["caveats"]})
        sup = next(c for c in low["caveats"] if c["id"] == "ob.caveat.support")
        self.assertIn("The rule for B rests on only 2 exclusive-win points", sup["text"])

        ok = ir.build_off_by_ir("DS", "e1", _off_by_result(n_wins=8), ["A", "B"])
        self.assertEqual(ok["confidence"]["surrogate_vs_B"]["support"], "adequate")
        self.assertNotIn("ob.caveat.support", {c["id"] for c in ok["caveats"]})
        # Degenerate competitors are consolidated into ONE atom naming them all.
        ids = {a["id"] for a in ok["evidence"]}
        self.assertIn("ob.degenerate", ids)
        self.assertNotIn("ob.vs.C.degenerate", ids)
        deg = next(a for a in ok["evidence"] if a["id"] == "ob.degenerate")
        self.assertEqual(deg["text"], "A never exclusively beat C.")

    def test_off_by_wins_grouped_by_identical_counts(self):
        # Distinct win counts → one atom each, ordered best-first and all
        # required (grouping keeps the atom count low enough to require them).
        result = _off_by_result(n_wins=10)
        pc = result["surrogates"]["per_competitor"]
        for name, wins in (("D", 8), ("E", 6), ("F", 1)):
            pc[name] = dict(pc["B"], n_exclusive_wins=wins,
                            exclusive_win_rate=wins / 40.0)
        doc = ir.build_off_by_ir("DS", "e1", result, ["A", "B"])
        _check_envelope(self, doc, "off_by_threshold")
        req = set(doc["required_atom_ids"])
        for wid in ("ob.wins.0", "ob.wins.1", "ob.wins.2", "ob.wins.3"):
            self.assertIn(wid, req)
        wins = {a["id"]: a for a in doc["evidence"] if a["type"] == "exclusive_wins"}
        self.assertEqual(wins["ob.wins.0"]["value"]["competitors"], ["B"])  # 10
        self.assertEqual(wins["ob.wins.3"]["value"]["competitors"], ["F"])  # 1
        self.assertIn("10 injected points", wins["ob.wins.0"]["text"])
        self.assertIn("1 injected point ", wins["ob.wins.3"]["text"])  # singular

    def test_off_by_wins_merge_when_counts_identical(self):
        # Rivals sharing the same (count, rate) collapse into ONE atom naming
        # both — the repetition that pushes the narrator into compressing names.
        result = _off_by_result(n_wins=1)
        pc = result["surrogates"]["per_competitor"]
        for name in ("D", "E"):
            pc[name] = dict(pc["B"])
        doc = ir.build_off_by_ir("DS", "e1", result, ["A", "B"])
        wins = [a for a in doc["evidence"] if a["type"] == "exclusive_wins"]
        self.assertEqual(len(wins), 1)
        self.assertEqual(wins[0]["value"]["competitors"], ["B", "D", "E"])
        self.assertIn("apiece that B, D, and E each miss", wins[0]["text"])

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
        # Competitors read as a full oxford list and the condition is prose,
        # never a raw "feature op threshold" comparison.
        self.assertIn("uniquely beats B and D when ", rules[0]["text"])
        self.assertIn("the distance from the boundary is at most", rules[0]["text"])
        self.assertNotIn("boundary_distance <=", rules[0]["text"])

    def test_mc_winner_surrogate_rules_not_emitted_but_fidelity_kept(self):
        # The winner-surrogate tree restates the win regions in fitted form, so
        # its rules are no longer evidence — but its held-out fidelity stays.
        result = _mc_result()
        result["winner_f1"]["rules"] = [
            {"conditions": [{"feature": "noise_level", "op": "<=", "threshold": 0.05}],
             "outcome": "A", "n_samples": 50},
            {"conditions": [{"feature": "noise_level", "op": ">", "threshold": 0.15}],
             "outcome": "B", "n_samples": 30},
        ]
        doc = ir.build_monte_carlo_ir("DS", "e1", result, ["A", "B"], ["B", "A"])
        self.assertEqual(
            [a for a in doc["evidence"] if a["type"] == "surrogate_rule"], [])
        self.assertNotIn("noise-sweep F1 winner is", json.dumps(doc))
        self.assertEqual(doc["confidence"]["winner_surrogate_f1"]["grade"], "high")
        self.assertEqual(doc["confidence"]["winner_surrogate_f1"]["cv_accuracy"], 0.85)

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

    def test_mc_win_regions_compress_isolated_points(self):
        # NOTE: the fixture shares one curves dict across F1 and PR-AUC, so
        # each detector reports the same ranges under both metrics.
        result = _mc_result()
        result["curves_f1"]["win_regions"] = {"A": [(0.0, 0.1), (0.15, 0.15)],
                                              "B": [(0.2, 0.2)]}
        doc = ir.build_monte_carlo_ir("DS", "e1", result, ["A", "B"], [])
        a_atom = next(x for x in doc["evidence"] if x["id"] == "mc.win_region.A")
        # Spans read "from A to B" — never "A-B", which the sign-aware number
        # extractor would parse as the negative number -B.
        self.assertIn("A won by F1 at noise levels from 0.000 to 0.100, and at 0.150",
                      a_atom["text"])
        self.assertNotIn("0.000-0.100", a_atom["text"])
        # Both metrics live in ONE atom, metric-first, split by a semicolon.
        self.assertIn("; by PR-AUC at noise levels", a_atom["text"])
        # Points-only reads as bare levels, with no dangling "at ... at".
        b_atom = next(x for x in doc["evidence"] if x["id"] == "mc.win_region.B")
        self.assertIn("B won by F1 at noise levels 0.200", b_atom["text"])
        self.assertNotIn("from", b_atom["text"])

    def test_determinism(self):
        a = json.dumps(ir.build_ga_combination_ir("DS", "e1", _ga_combination_result()),
                       sort_keys=True)
        b = json.dumps(ir.build_ga_combination_ir("DS", "e1", _ga_combination_result()),
                       sort_keys=True)
        self.assertEqual(a, b)


class TestWriterAndAssembler(unittest.TestCase):

    def test_every_footer_ends_with_the_closing_sentence(self):
        """The closing line is appended centrally in _envelope, so every stage
        — including ones added later — ends its glossary the same way."""
        docs = [
            ir.build_ga_selection_ir("DS", "e1", _ga_selection_result()),
            ir.build_ga_combination_ir("DS", "e1", _ga_combination_result()),
            ir.build_monte_carlo_ir("DS", "e1", _mc_result(), ["A"], ["A"]),
            ir.build_off_by_ir("DS", "e1", _off_by_result(), ["A", "B"]),
            ir.build_thompson_ir("DS", "e1", **_thompson_kwargs()),
            ir.build_rank_aggregation_ir("DS", "e1", "robust", 0,
                                         _rank_agg_result(), ["S1", "S2"],
                                         {"S1": "A", "S2": "B"}, ["A", "B"]),
        ]
        for doc in docs:
            footer = doc.get("info_footer", "")
            self.assertTrue(footer, doc["stage"])
            self.assertTrue(footer.endswith(ir.FOOTER_CLOSING_SENTENCE),
                            f"{doc['stage']}: {footer[-80:]!r}")
            # Exactly once, and separated from the definitions before it.
            self.assertEqual(footer.count(ir.FOOTER_CLOSING_SENTENCE), 1)
            self.assertIn(" " + ir.FOOTER_CLOSING_SENTENCE, footer)

    def test_closing_sentence_is_not_doubled(self):
        env = ir._envelope("s", "DS", "e1", {}, [], [], [],
                           info_footer="Utility is a lift. "
                                       + ir.FOOTER_CLOSING_SENTENCE)
        self.assertEqual(env["info_footer"].count(ir.FOOTER_CLOSING_SENTENCE), 1)

    def test_top_of_ranking_handles_every_caller_shape(self):
        """The pipeline hands aggregation results in three shapes. Indexing
        [1][0] blindly turned ["LOF_1", "CBLOF_4"] into "C" — the first letter
        of the SECOND name — so each shape is checked explicitly."""
        self.assertEqual(ir._top_of_ranking(["LOF_1", "CBLOF_4"]), "LOF_1")
        self.assertEqual(ir._top_of_ranking((0.5, ["A", "B"])), "A")
        self.assertEqual(ir._top_of_ranking("CBLOF_1"), "CBLOF_1")
        for empty in (None, [], "", (0.0, []), [None, []]):
            self.assertEqual(ir._top_of_ranking(empty), ir.NOT_AVAILABLE)

    def test_global_consensus_picks_are_whole_detector_names(self):
        results = _results_dict()
        # The shape run_model_selection_algorithms_2 actually returns: the
        # ranking list itself, already unwrapped from the (score, ranking) pair.
        results["aggregation"] = {"robust_agg": ["LOF_1", "CBLOF_4", "NN_3"],
                                  "final_agg": ["CBLOF_1", "LOF_1"]}
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "explanations_ir")
            path = ir.assemble_global_ir(results, "DS", "e1", 0, base_dir=base)
            with open(path) as f:
                g = json.load(f)
        picks = g["stage_agreement"]
        self.assertEqual(picks["robust_consensus"]["top_pick"], "LOF_1")
        text = " ".join(a["text"] for a in g["evidence"])
        self.assertNotIn("top pick (C)", text)
        self.assertNotIn("top pick (L)", text)

    def test_final_consensus_is_not_an_agreement_row(self):
        """The final consensus produces the single-model pick, so comparing the
        two would always report agreement and carry no information."""
        results = _results_dict()
        results["aggregation"] = {"robust_agg": ["LOF_1", "CBLOF_4"],
                                  "final_agg": ["A", "B"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = ir.assemble_global_ir(results, "DS", "e1", 0,
                                         base_dir=os.path.join(tmp, "ir"))
            with open(path) as f:
                g = json.load(f)
        self.assertNotIn("final_consensus", g["stage_agreement"])
        self.assertIn("robust_consensus", g["stage_agreement"])
        self.assertNotIn("final_consensus", json.dumps(g["evidence"]))

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
