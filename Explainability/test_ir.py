"""
Standalone unit tests for the Intermediate Representation (IR) layer.
Loads Explainability/ir.py by file path (numpy + stdlib only); the tree-rule
test importorskips sklearn.
"""

import importlib.util
import json
import os
import re
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
    # A: rank 1 on all three, pushes toward anomaly. B: rank 2 on |SHAP| and
    # ALE but 3 on PFI, also toward anomaly. C: rank 3 on |SHAP| and ALE but 2
    # on PFI, and pushes the other way. Exercises the shared-rank collapse and
    # the two-way sign grouping. `shap_signed_importance` is still
    # produced (the report keeps it for comparison) but the IR must ignore it.
    return {
        "best_ensemble": ["A", "B", "C"], "feature_names": ["A", "B", "C"],
        "meta_model_type": "rf", "model_source": "captured", "baseline_f1": 0.87,
        "shap_importance": {"A": 0.4, "B": 0.2, "C": 0.1},
        "shap_signed_importance": {"A": 0.35, "B": 0.15, "C": -0.05},
        "pfi_importance": {"A": 0.2, "B": 0.05, "C": 0.08},
        "ale_total_variation": {"A": 0.5, "B": 0.3, "C": 0.2},
        "ale_net": {"A": 0.5, "B": 0.3, "C": -0.2},
        "ale_consistency": {"A": 1.0, "B": 1.0, "C": 1.0},
        "ale_sign": {"A": "positive", "B": "positive", "C": "negative"},
        "ale_sign_support": {"A": [], "B": [], "C": []},
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


def _gan_result(n_wins=8):
    """The GAN stage's result has off-by's shape with GAN's features — the two
    stages share one surrogate implementation and one IR builder."""
    result = _off_by_result(n_wins=n_wins)
    result["surrogates"]["feature_names"] = ["ambiguity", "local_volatility"]
    pc = result["surrogates"]["per_competitor"]
    pc["B"]["feature_importances"] = {"ambiguity": 0.9, "local_volatility": 0.1}
    return result


def _thompson_kwargs():
    return dict(
        n_windows=6,
        final_ranking=[("A", 1.5), ("B", 0.7)],
        regimes=[{"index": 0, "start": 0, "end": 2, "duration": 3, "leader": "A",
                  "rewards_top": [("A", 0.5), ("B", 0.2)], "reward_gap": 0.3,
                  "runner_up": "B",
                  # Three distinct quantities: what the reward is MADE of,
                  # where the leader's EDGE over the runner-up comes from (same
                  # units, differenced), and how far each channel departs from
                  # its usual contribution.
                  "reward_raising": [(0, 0.6), (1, 0.2)],
                  "reward_lowering": [(4, -0.05)],
                  "edge_favor_leader": [(0, 0.25)],
                  "edge_favor_runner": [(3, -0.04)], "edge_gap": 0.3,
                  "shap_raising": [(0, 0.4)], "shap_lowering": [(2, -0.1)],
                  "pref_favor_leader": [(0, 0.3)],
                  "pref_favor_runner": [(3, -0.05)], "pref_gap": 0.3},
                 {"index": 1, "start": 3, "end": 5, "duration": 3, "leader": "B",
                  "rewards_top": [("B", 0.6), ("A", 0.4)], "reward_gap": 0.2,
                  "runner_up": "A",
                  "reward_raising": None, "reward_lowering": None,
                  "edge_favor_leader": None, "edge_favor_runner": None,
                  "edge_gap": float("nan"),
                  "shap_raising": None, "shap_lowering": None,
                  "pref_favor_leader": None, "pref_favor_runner": None,
                  "pref_gap": float("nan")}],
        shifts=[{"window": 3, "from_model": "A", "to_model": "B",
                 "reward_delta": 0.2, "regime_length": 3}],
        blip_count=1,
        state_fractions={"random": 0.2, "exploitation": 0.6, "informed_exploration": 0.2},
        state_counts={"random": 1, "exploitation": 3, "informed_exploration": 1},
        final_state="exploitation",
    )


def _thompson_ranking_kwargs():
    return dict(
        n_windows=40,
        final_ranking=[("A", 1.5), ("B", 0.7), ("C", 0.2)],
        # Sums to A's score of 1.5, as the real decomposition does exactly.
        winner_channels=[(0, 0.9), (2, 0.4), (1, 0.2)],
        # Sums to the 0.8 margin over B; the third entry is a channel B won.
        gap_channels=[(0, 0.7), (2, 0.3), (1, -0.2)],
        selection_counts={"A": 22, "B": 12, "C": 6},
        regimes=[{"index": 0, "start": 10, "end": 24, "duration": 15,
                  "leader": "B", "runner_up": "A",
                  "top_channels": [(2, 0.3), (0, 0.1)],
                  "gap_channels": [(2, 0.2)], "score": 0.4, "runner_score": 0.3},
                 {"index": 1, "start": 25, "end": 39, "duration": 15,
                  "leader": "A", "runner_up": "B",
                  "top_channels": [(0, 0.9)], "gap_channels": [(0, 0.5)],
                  "score": 1.5, "runner_score": 0.7}],
        warmup_windows=10,
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
        # try/except, not find_spec: another test module in the same run may
        # already have partially imported sklearn, leaving sklearn.__spec__ None,
        # and find_spec then raises ValueError instead of answering the question.
        try:
            from sklearn.tree import DecisionTreeClassifier
        except ImportError:
            self.skipTest("scikit-learn not installed")
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

        # One atom per regime, carrying THREE distinct claims: what the reward
        # was made of, where the edge over the runner-up came from (the same
        # units, so the two share a sentence), and — separately, because it is
        # a different quantity — which channel departed furthest from its usual.
        self.assertEqual(
            by_id["ts.regime.0"]["text"],
            "Regime 0 (windows 0 to 2, 3 windows) was led by A, with channel 0 "
            "and channel 1 raising its expected reward the most, channel 0 also "
            "giving it its biggest edge over B, and channel 0 departing "
            "furthest from its usual contribution, running above it.")
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

        # States are narrated best-first as a window count and a share; no
        # final-state atom. The unit is written once and then carried.
        self.assertEqual(
            by_id["ts.states.summary"]["text"],
            "Over the 6 windows the sampler was in exploitation for 3 windows "
            "(60.0%), random for 1 (20.0%), and informed exploration for 1 "
            "(20.0%).")
        self.assertNotIn("ts.states.final", by_id)

    def test_thompson_states_fall_back_to_shares_without_counts(self):
        """A count derived from a share is a rounded number, and rounding is
        what this layer exists to keep out of the prose — 60% of 6 windows is
        3.6, not 4. Without the sampler's own tallies the sentence stays a
        share."""
        kwargs = _thompson_kwargs()
        kwargs.pop("state_counts")
        doc = ir.build_thompson_ir("DS", "e1", **kwargs)
        by_id = {a["id"]: a for a in doc["evidence"]}
        self.assertEqual(
            by_id["ts.states.summary"]["text"],
            "Over the 6 windows the sampler was in exploitation 60.0% of the "
            "time, random 20.0% of the time, and informed exploration 20.0% of "
            "the time.")

    def test_thompson_regime_summary_states_windows_held(self):
        """Regime count alone reads as though four short spells were more of the
        run than one long one, so each leader carries the windows it held and
        the list is ordered by them — the same rule the ranking stage uses."""
        doc = ir.build_thompson_ir("DS", "e1", **_thompson_kwargs())
        by_id = {a["id"]: a for a in doc["evidence"]}
        text = by_id["ts.regimes.summary"]["text"]
        self.assertIn("A led 1 regime, spanning 3 windows", text)
        self.assertIn("B led 1 regime, spanning 3 windows", text)
        self.assertEqual(by_id["ts.regimes.summary"]["value"]["windows_led"],
                         {"A": 3, "B": 3})

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
        differing edge channel is reported separately from the supplying ones.

        Both clauses read the CONTRIBUTION split — the edge is that split
        differenced against the runner-up — so they are slices of one total and
        can share a sentence. Sourcing the edge from SHAP is what this pins
        against: it would put a deviation and a share in one breath.
        """
        kwargs = _thompson_kwargs()
        kwargs["regimes"][0]["reward_raising"] = [(2, 0.5), (5, 0.2)]
        kwargs["regimes"][0]["edge_favor_leader"] = [(7, 0.3)]
        doc = ir.build_thompson_ir("DS", "e1", **kwargs)
        txt = next(a for a in doc["evidence"] if a["id"] == "ts.regime.0")["text"]
        self.assertIn("channel 2 and channel 5 raising its expected reward the most",
                      txt)
        self.assertIn("channel 7 giving it its biggest edge over B", txt)

    def test_thompson_edge_never_comes_from_the_deviation_split(self):
        """The edge clause is in expected-reward units. A run whose SHAP
        comparison points at a different channel must not move it: the two
        measure different things and only one sums to the reported gap."""
        kwargs = _thompson_kwargs()
        kwargs["regimes"][0]["edge_favor_leader"] = [(1, 0.25)]
        kwargs["regimes"][0]["pref_favor_leader"] = [(6, 0.9)]
        doc = ir.build_thompson_ir("DS", "e1", **kwargs)
        atom = next(a for a in doc["evidence"] if a["id"] == "ts.regime.0")
        self.assertIn("channel 1 also giving it its biggest edge over B",
                      atom["text"])
        self.assertNotIn("channel 6", atom["text"])
        # SHAP's version of the comparison survives for the alternate plot.
        self.assertEqual(atom["value"]["deviation_edge_channels"], [[6, 0.9]])

    def test_thompson_deviation_sentence_names_the_furthest_departure(self):
        """The deviation clause reports the largest departure in EITHER
        direction and says which way — taking only the positive list would
        silently drop a channel that collapsed below its usual."""
        kwargs = _thompson_kwargs()
        kwargs["regimes"][0]["shap_raising"] = [(1, 0.2)]
        kwargs["regimes"][0]["shap_lowering"] = [(3, -0.9)]
        doc = ir.build_thompson_ir("DS", "e1", **kwargs)
        atom = next(a for a in doc["evidence"] if a["id"] == "ts.regime.0")
        self.assertIn("channel 3 departing furthest from its usual "
                      "contribution, running below it.", atom["text"])
        self.assertEqual(atom["value"]["deviation_lowering"], [[3, -0.9]])

    def test_thompson_negative_edge_gap_is_not_dramatised(self):
        """A regime whose leader trails on the aggregate gap is still narrated
        by naming its best channel — the prose never editorialises with
        'although' or leaks the signed gap into the sentence."""
        kwargs = _thompson_kwargs()
        kwargs["regimes"][0]["edge_gap"] = -0.12
        kwargs["regimes"][0]["edge_favor_leader"] = [(1, 0.02)]
        kwargs["regimes"][0]["edge_favor_runner"] = [(0, -0.14)]
        doc = ir.build_thompson_ir("DS", "e1", **kwargs)
        atom = next(a for a in doc["evidence"] if a["id"] == "ts.regime.0")
        self.assertIn("channel 1 also giving it its biggest edge over B",
                      atom["text"])
        for word in ("although", "however", "despite", "-0.12"):
            self.assertNotIn(word, atom["text"])
        # The signed gaps are still grounded in `value` for the verifier.
        self.assertEqual(atom["value"]["edge_gap"], -0.12)
        self.assertEqual(atom["value"]["preference_score_gap"], 0.3)

    def test_thompson_channel_names_used_when_available(self):
        kwargs = _thompson_kwargs()
        kwargs["regimes"][0]["shap_raising"] = [(1, 0.5)]
        kwargs["regimes"][0]["pref_favor_leader"] = [(1, 0.3)]
        named = ir.build_thompson_ir(
            "DS", "e1", channel_names=["Pressure", "Accelerometer1RMS"], **kwargs)
        txt = next(a for a in named["evidence"] if a["id"] == "ts.regime.0")["text"]
        # Name is used verbatim — never lower-cased by a blanket .capitalize().
        self.assertIn("Accelerometer1RMS raising its expected reward the most", txt)
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

    def test_thompson_ranking(self):
        doc = ir.build_thompson_ranking_ir("DS", "e1", n_channels=4,
                                           **_thompson_ranking_kwargs())
        _check_envelope(self, doc, "thompson_ranking")
        self.assertEqual(doc["output"]["top_pick"], "A")
        self.assertEqual(doc["output"]["warmup_windows"], 10)
        by_id = {a["id"]: a for a in doc["evidence"]}

        # The lead states the criterion, not just the winner: this stage exists
        # because the sibling's headline reports a ranking it never explains.
        self.assertEqual(
            by_id["tsr.output.top"]["text"],
            "Ranked by the size of its learned weights, A scored 1.500000, "
            "ahead of B by 0.800000.")

        # Shares are percentages of the winner's own score, computed from the
        # forwarded contributions rather than restated by the caller.
        self.assertEqual(
            by_id["tsr.winner.channels"]["text"],
            "channel 0 (60.0%), channel 2 (26.7%), and channel 1 (13.3%) "
            "contributed the majority of A's score.")
        # No concentration atom: "the top 3 channels are N% of the score" is a
        # restatement of the shares just given, and the narrator turned it into
        # editorial ("indicating the remaining six contributed less
        # significantly").
        self.assertNotIn("tsr.winner.concentration", by_id)

        # The gap is the only directional quantity, and it must name the rival.
        # No raw values in the prose at all — the table and the gap plot carry
        # them — and the losing channel's direction is stated in words. A signed
        # "-0.200000" next to "in B's favour" had read as "bad for B" and
        # inverted the claim in a real narration.
        self.assertEqual(
            by_id["tsr.gap.runner_up"]["text"],
            "Channel 0 and channel 2 contributed significantly to the lead "
            "that A had over B, while channel 1 favoured B more than A.")
        for atom_id in ("tsr.gap.runner_up", "tsr.winner.channels"):
            self.assertNotIn("0.700000", by_id[atom_id]["text"])
        self.assertEqual(by_id["tsr.gap.runner_up"]["value"]["per_channel"],
                         [[0, 0.7], [2, 0.3], [1, -0.2]])
        # `rivals` is a verifier._RIVAL_KEYS name, so the rival set gets checked
        # against the narrated sentence for free.
        self.assertEqual(by_id["tsr.gap.runner_up"]["value"]["rivals"], ["B"])

        # Against the runner-up, not the least-tried detector: it is the
        # comparison the ranking actually turns on, and it can cut against the
        # winner. It is also the safer name — the runner-up is already in the
        # lead sentence, whereas the least-tried one appeared nowhere else and
        # got narrated as an outright wrong name (CBLOF_3 for LOF_3).
        self.assertEqual(
            by_id["tsr.support"]["text"],
            "A was selected in 22 of the 40 windows, against 12 for B.")
        self.assertEqual(by_id["tsr.support"]["value"]["runner_up"], "B")

        # Each leader's regime count AND the windows it held: four short spells
        # are not more of the run than one long one, so the count alone misleads.
        self.assertEqual(
            by_id["tsr.regimes.summary"]["text"],
            "Leadership on this score changed hands over the run: it splits "
            "into 2 regimes led by 2 different detectors: A led 1 regime, "
            "spanning 15 windows and B led 1 regime, spanning 15 windows. "
            "The first 10 windows are left out, because all detectors start "
            "with score zero.")
        self.assertEqual(by_id["tsr.regimes.summary"]["value"]["windows_led"],
                         {"A": 15, "B": 15})
        # No runner-up in the regime sentence. Coverage is conjunctive, the
        # narrator drops that clause, and the atom then passed only when the
        # runner-up was named elsewhere by luck — so a run's faithfulness hung
        # on which detector placed second. It stays in `value` and on the plot.
        self.assertEqual(
            by_id["tsr.regime.0"]["text"],
            "Regime 0 (windows 10 to 24, 15 windows) was led by B, with "
            "channel 2 and channel 0 raising its score the most.")
        self.assertEqual(by_id["tsr.regime.0"]["value"]["runner_up"], "A")
        for atom in doc["evidence"]:
            if atom["type"] != "regime":
                continue
            runner = atom["value"]["runner_up"]
            # The leader is named; the runner-up is not — that is the whole
            # point, since every name in an atom's text becomes a coverage
            # requirement the narrator has to satisfy.
            self.assertRegex(atom["text"], rf"\b{atom['value']['leader']}\b")
            self.assertNotRegex(atom["text"], rf"\b{runner}\b")

        for rid in ("tsr.output.top", "tsr.winner.channels",
                    "tsr.gap.runner_up",
                    "tsr.support", "tsr.regimes.summary",
                    "tsr.regime.0", "tsr.regime.1"):
            self.assertIn(rid, doc["required_atom_ids"])

        # Both method caveats always ride along: the non-negativity one is the
        # single mistake the verifier cannot catch, and the exposure one is a
        # real limit on what the ranking means.
        caveats = {c["id"]: c["text"] for c in doc["caveats"]}
        self.assertEqual(sorted(caveats),
                         ["tsr.caveat.exposure", "tsr.caveat.nonnegative"])
        self.assertIn("never be negative", caveats["tsr.caveat.nonnegative"])
        self.assertIn("only move in windows where it was selected",
                      caveats["tsr.caveat.exposure"])

        self.assertIn("which channels drove each detector's ranking score up",
                      doc["question"])
        # The footer must say the two stages' regimes are different things.
        self.assertIn("not the same regimes", doc["info_footer"])

    def test_thompson_ranking_prose_never_gives_a_share_a_direction(self):
        """A share is a sum of squares. "Channel 1 lowered the score" is the one
        false sentence the verifier cannot see — its number and its channel name
        are both correct — so the atoms must never model that phrasing."""
        doc = ir.build_thompson_ranking_ir("DS", "e1", n_channels=4,
                                           **_thompson_ranking_kwargs())
        share_atoms = [a["text"] for a in doc["evidence"]
                       if a["type"] in ("winner_channels", "regime")]
        for text in share_atoms:
            for direction in ("lowered", "reduced", "worked against",
                              "dragged", "pushed it down", "negative"):
                self.assertNotIn(direction, text.lower())
        # The gap atom, and only the gap atom, may take a side — and it names
        # the detector the channel went to, so direction cannot be inferred
        # from a sign the narrator has to interpret.
        gap = next(a for a in doc["evidence"] if a["type"] == "rank_gap")
        self.assertIn("favoured B more than A", gap["text"])

    def test_thompson_ranking_degenerate_runs(self):
        kwargs = _thompson_ranking_kwargs()
        # A single regime must not claim leadership "changed hands", and a
        # one-window regime must not read "1 windows".
        kwargs["regimes"] = [{"index": 0, "start": 10, "end": 10, "duration": 1,
                              "leader": "A", "runner_up": "B",
                              "top_channels": [(0, 0.9)], "gap_channels": [],
                              "score": 1.5, "runner_score": 0.7}]
        doc = ir.build_thompson_ranking_ir("DS", "e1", n_channels=4, **kwargs)
        by_id = {a["id"]: a for a in doc["evidence"]}
        self.assertEqual(
            by_id["tsr.regimes.summary"]["text"].split(".")[0],
            "One detector held the highest score for the whole run: A, "
            "across 1 windows")
        self.assertIn("1 window)", by_id["tsr.regime.0"]["text"])
        self.assertNotIn("1 windows", by_id["tsr.regime.0"]["text"])

        # With one detector there is no runner-up, so the support sentence
        # must not compare the winner against itself.
        solo = _thompson_ranking_kwargs()
        solo["final_ranking"] = [("A", 1.5)]
        solo["selection_counts"] = {"A": 40}
        solo["gap_channels"] = []
        doc = ir.build_thompson_ranking_ir("DS", "e1", n_channels=4, **solo)
        support = next(a for a in doc["evidence"] if a["id"] == "tsr.support")
        self.assertEqual(support["text"], "A was selected in 40 of the 40 windows.")
        self.assertNotIn("tsr.gap.runner_up", {a["id"] for a in doc["evidence"]})

    def test_thompson_ranking_is_absent_from_cross_stage_agreement(self):
        """Both Thompson stages report the same winner — they share rank_models
        — so counting the new one would double-weight Thompson in the
        consensus."""
        self.assertIn("thompson_ranking", ir._STAGE_FILES)
        with open(os.path.join(_THIS, "ir.py"), encoding="utf-8") as f:
            source = f.read()
        picks = re.search(r"stage_picks = \{(.*?)\n    \}", source, re.S)
        self.assertIsNotNone(picks)
        self.assertNotIn("thompson_ranking", picks.group(1))

    def test_ga_combination_reports_a_thinly_supported_sign_and_qualifies_it(self):
        """A detector whose effect changes sign across its range still ends
        somewhere, and where it ends is a measurement. Withholding it turned a
        measured negative into a blank that read as missing data; the sign is
        reported and the caveat is what says how far to trust it."""
        result = _ga_combination_result()
        result["ale_net"] = {"A": 0.5, "B": -0.01, "C": -0.2}
        result["ale_consistency"] = {"A": 1.0, "B": 0.03, "C": 1.0}
        result["ale_sign"] = {"A": "positive", "B": "negative", "C": "negative"}
        result["ale_sign_support"] = {"A": [], "B": ["low_consistency"], "C": []}
        doc = ir.build_ga_combination_ir("DS", "e1", result)
        by_id = {a["id"]: a for a in doc["evidence"]}

        role = by_id["ga_comb.detector.B.role"]["value"]
        self.assertEqual(role["sign"], "negative")
        self.assertEqual(role["sign_support"], ["low_consistency"])
        # B still carries full weight — it is rank 2 of 3 — so a qualified sign
        # must not read as "unimportant".
        self.assertIn("second-most weight",
                      by_id["ga_comb.detector.B.role"]["text"])
        # B is grouped with the other negative, not exiled to a third bucket.
        sign = by_id["ga_comb.sign_summary"]
        self.assertEqual(
            sign["text"],
            "A had a positive sign, while B and C had negative.")
        self.assertEqual(sign["value"]["no_sign"], [])

        # The run-dependent caveat fires, names exactly the affected detector,
        # and gives the reason its sign is thin.
        caveat = next(c for c in doc["caveats"]
                      if c["id"] == "ga_comb.caveat.sign_consistency")
        self.assertEqual(
            caveat["text"],
            "Weakly supported sign: B pushed the meta-learner both ways across "
            "its score range. Keep that in mind when reading it.")
        self.assertEqual(caveat["value"]["weakly_supported"], ["B"])
        # A detector with a well-supported sign is never dragged in with it.
        self.assertNotIn("A and", caveat["text"])

    def test_ga_combination_weak_and_missing_signs_are_separate_caveats(self):
        """Two different statements. 'Trust this less' and 'there is nothing
        here' collapsed into one atom would let a narrator report either as the
        other."""
        result = _ga_combination_result()
        result["ale_sign"] = {"A": "positive", "B": "negative",
                              "C": "not_available"}
        result["ale_sign_support"] = {"A": [], "B": ["weak_influence"], "C": []}
        doc = ir.build_ga_combination_ir("DS", "e1", result)
        by_cid = {c["id"]: c for c in doc["caveats"]}

        self.assertEqual(by_cid["ga_comb.caveat.sign_consistency"]
                         ["value"]["weakly_supported"], ["B"])
        self.assertIn("moved the meta-learner too little",
                      by_cid["ga_comb.caveat.sign_consistency"]["text"])
        self.assertEqual(by_cid["ga_comb.caveat.sign_missing"]["value"]["no_sign"],
                         ["C"])
        self.assertIn("no net effect to take a sign from",
                      by_cid["ga_comb.caveat.sign_missing"]["text"])

    def test_ga_combination_no_weak_signs_emits_no_caveat(self):
        """The qualification is run-dependent: a run where every sign is well
        supported must not carry a caveat implying otherwise."""
        result = _ga_combination_result()
        result["ale_sign_support"] = {"A": [], "B": [], "C": []}
        doc = ir.build_ga_combination_ir("DS", "e1", result)
        ids = {c["id"] for c in doc["caveats"]}
        self.assertNotIn("ga_comb.caveat.sign_consistency", ids)
        self.assertNotIn("ga_comb.caveat.sign_missing", ids)

    def test_ga_combination_sign_absent_for_every_detector(self):
        """When nothing has a sign the summary must still say something
        true, not fall through to an empty atom."""
        result = _ga_combination_result()
        result["ale_sign"] = {"A": "not_available", "B": "not_available",
                              "C": "not_available"}
        doc = ir.build_ga_combination_ir("DS", "e1", result)
        sign = next(a for a in doc["evidence"] if a["id"] == "ga_comb.sign_summary")
        self.assertEqual(
            sign["text"],
            "No detector has a sign: A, B, and C have no net effect.")
        self.assertIn("ga_comb.sign_summary", doc["required_atom_ids"])

    def test_ga_combination_output_does_not_credit_the_stage_with_selecting(self):
        """The GA picks the subset; this stage only measures. The atom used to
        say the 'combination step selected' the ensemble and that the
        meta-learner 'then weighted' it — two claims about steps that do not
        happen."""
        doc = ir.build_ga_combination_ir("DS", "e1", _ga_combination_result())
        atom = next(a for a in doc["evidence"] if a["id"] == "ga_comb.output.subset")
        self.assertEqual(
            atom["text"],
            "The genetic algorithm selected the 3-detector ensemble {A, B, C}; "
            "the ranking below measures how much each of those detectors moves "
            "the trained meta-learner's output.")

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
        # The profile LEADS the sentence. Three excluded atoms in a row all
        # opened "X was left out …" and differed only in the high/low tail, so a
        # narrator merged them and handed one detector another's profile.
        self.assertEqual(
            by_id["ga_sel.excluded.C"]["text"],
            "C had high utility and low stability, but was still left out.")
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
        # Mn: low profile but lofo>0 → "needed" callout, with number. The low/low
        # finding LEADS the sentence rather than sitting in a `despite` clause:
        # backgrounded that way, narrators restated it as high/high.
        needed = by_id["ga_sel.needed.Mn"]
        self.assertEqual(
            needed["text"],
            "Mn has low utility and low stability, yet removing it lowers the "
            "ensemble's fitness by 0.0300, which is why it was kept.")
        # The profile is machine-checkable, so the verifier's attribution channel
        # catches an inversion even though the code never reaches the prose.
        self.assertEqual(needed["value"]["archetype"], "LL")
        self.assertEqual(by_id["ga_sel.included.both"]["value"]["archetype"], "HH")
        self.assertEqual(by_id["ga_sel.included.utility"]["value"]["archetype"], "HL")
        self.assertEqual(by_id["ga_sel.included.stability"]["value"]["archetype"], "LH")
        # Excluded: high-utility anomaly individual; the rest grouped by profile.
        self.assertEqual(by_id["ga_sel.excluded.Xh"]["text"],
                         "Xh had high utility and low stability, but was still left out.")
        self.assertEqual(by_id["ga_sel.excluded.stable"]["value"]["detectors"], ["Xs"])
        self.assertEqual(by_id["ga_sel.excluded.plain"]["value"]["detectors"], ["Xp"])
        # Excluded groups lead with the profile too, for the same reason.
        self.assertEqual(by_id["ga_sel.excluded.plain"]["text"],
                         "Xp had low utility and low stability, and was left out.")
        self.assertEqual(by_id["ga_sel.excluded.nodata"]["value"]["detectors"], ["Xn"])

    def test_competition_rank_tolerates_float_noise(self):
        """Markov scores that are mathematically tied come back from
        np.linalg.eig a few ulp apart. An exact `!=` promoted that wobble into a
        real rank difference, so which detector 'carries the most weight' was
        decided by the eigen-solver rather than by the data."""
        scores = {"A": 0.18347554726124723,      # one ulp above the other two
                  "B": 0.18347554726124712,
                  "C": 0.18347554726124712,
                  "D": 0.17937750940504008}
        ranks = ir._competition_rank(scores, ["A", "B", "C", "D"])
        self.assertEqual(ranks, {"A": 1, "B": 1, "C": 1, "D": 4})
        # A genuine gap still separates, and competition ranking still skips.
        self.assertEqual(
            ir._competition_rank({"A": 0.5, "B": 0.3, "C": 0.3, "D": 0.1},
                                 ["A", "B", "C", "D"]),
            {"A": 1, "B": 2, "C": 2, "D": 4})
        # Ties are measured against the running block, not the previous item,
        # so a chain of small steps cannot collapse into one rank.
        drift = {"A": 1.0, "B": 1.0 - 6e-10, "C": 1.0 - 1.2e-9}
        self.assertEqual(ir._competition_rank(drift, ["A", "B", "C"]),
                         {"A": 1, "B": 1, "C": 3})

    def test_ga_combination_reports_a_tied_lead_as_tied(self):
        result = _ga_combination_result()
        # A and B mathematically tied, one ulp apart as the solver returns them.
        result["markov_scores"] = {"A": 0.5, "B": 0.5 - 1.1e-16, "C": 0.2}
        doc = ir.build_ga_combination_ir("DS", "e1", result)
        by_id = {a["id"]: a for a in doc["evidence"]}
        a = by_id["ga_comb.detector.A.role"]
        b = by_id["ga_comb.detector.B.role"]
        self.assertEqual(a["value"]["final_rank"], 1)
        self.assertEqual(b["value"]["final_rank"], 1)
        self.assertTrue(a["value"]["final_rank_tied"])
        self.assertIn("(overall weight rank 1 of 3, a tie)", a["text"])
        self.assertIn("(overall weight rank 1 of 3, a tie)", b["text"])
        # `top_pick` is the first of an arbitrary order, so the tied set is
        # recorded rather than presenting it as a sole winner.
        self.assertEqual(doc["output"]["top_pick_tied_with"], ["B"])
        self.assertFalse(by_id["ga_comb.detector.C.role"]["value"]["final_rank_tied"])

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
        # The three MAGNITUDE measures are quoted — the ones that actually feed
        # the aggregation. Signed SHAP is not among them and no longer supplies
        # the direction either, so citing it here would imply a contribution it
        # does not make.
        # The overall weight rank carries its own LABEL and its own NUMBER: as a
        # bare ordinal it was the only quantity in the sentence without a digit,
        # sitting beside the digit method ranks, and narrators re-derived it from
        # whichever digit was nearest.
        a = next(x for x in doc["evidence"] if x["id"] == "ga_comb.detector.A.role")
        self.assertEqual(
            a["text"],
            "A carries the most weight in the ensemble (overall weight rank 1 of "
            "3), ranking 1 on absolute SHAP, PFI, and total ALE.")
        # Two rank groups where one holds two measures: the comma before the
        # final "and" is what separates the groups. Without it this reads
        # "...absolute SHAP and total ALE and 3 on PFI".
        b = next(x for x in doc["evidence"] if x["id"] == "ga_comb.detector.B.role")
        self.assertEqual(
            b["text"],
            "B carries the second-most weight in the ensemble (overall weight "
            "rank 2 of 3), ranking 2 on absolute SHAP and total ALE, and 3 on PFI.")
        c = next(x for x in doc["evidence"] if x["id"] == "ga_comb.detector.C.role")
        self.assertEqual(
            c["text"],
            "C carries the third-most weight in the ensemble (overall weight "
            "rank 3 of 3), ranking 3 on absolute SHAP and total ALE, and 2 on PFI.")
        prose = " ".join(x["text"] for x in doc["evidence"])
        self.assertNotIn("signed SHAP", prose)
        # Signed SHAP is gone from the machine-readable block too: the verifier
        # admits every number in a `value` dict, so leaving the superseded one
        # there would let a narrator quote it unflagged.
        self.assertNotIn("signed_shap", b["value"])
        self.assertNotIn("signed_shap_rank", b["value"])
        # ALE supplies magnitude, its rank, the net effect and the consistency
        # that decided whether a sign could be claimed at all.
        self.assertEqual(b["value"]["ale_total"], 0.3)
        self.assertEqual(b["value"]["ale_rank"], 2)
        self.assertEqual(b["value"]["ale_net"], 0.3)
        self.assertEqual(b["value"]["sign_consistency"], 1.0)
        # No raw magnitude leaks into the prose (they stay in `value`).
        self.assertNotIn("Markov", a["text"])
        self.assertNotIn("0.35", b["text"])
        self.assertEqual(b["value"]["sign"], "positive")
        self.assertEqual(c["value"]["sign"], "negative")

        # One sign-summary atom classifies all members by full name, and says
        # what the sign MEANS rather than just naming it.
        self.assertIn("ga_comb.sign_summary", doc["required_atom_ids"])
        sign = next(a for a in doc["evidence"] if a["id"] == "ga_comb.sign_summary")
        self.assertEqual(
            sign["text"],
            "A and B had positive signs, while C had negative.")
        # Every detector has a sign here, so no consistency caveat fires.
        self.assertNotIn("ga_comb.caveat.sign_consistency",
                         {x["id"] for x in doc["caveats"]})

        # Retired atoms from the old dense layout are gone.
        for gone in ("ga_comb.output.top", "ga_comb.context.members",
                     "ga_comb.detector.A.agreement", "ga_comb.detector.A.methods"):
            self.assertNotIn(gone, by_id)

        # Envelope carries the headline question and the sign/rank glossary.
        self.assertIn("push the meta-learner's decision", doc["question"])
        self.assertIn("The sign is the sign of ALE's net accumulated effect",
                      doc["info_footer"])
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
        # Prose spells the pattern out; the underscored form is an enum and
        # stays in `value` for machine consumers, not in a human sentence.
        self.assertIn("influential disagreer", robust["info_footer"])
        self.assertNotIn("influential_disagreer", robust["info_footer"])
        role = next(a for a in robust["evidence"]
                    if a["id"] == "ra_robust.source.S1.role")
        self.assertIn("an influential disagreer pattern", role["text"])
        self.assertEqual(role["value"]["pattern"], "influential_disagreer")

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
        self.assertIn("shaped the robustness consensus most (overall standing rank 1 of 2),",
                      atoms["ra_robust.source.S2.role"]["text"])
        self.assertIn("shaped the robustness consensus second most "
                      "(overall standing rank 2 of 2),",
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
        self.assertIn("shaped the robustness consensus most (overall standing rank 1 of 2),", lead["text"])
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
        # All winners fit inside the top-K cut here, so there is no tail clause.
        wr = next(a for a in doc["evidence"] if a["id"] == "mc.surrogate.win_rates")
        self.assertEqual(
            wr["text"],
            "Across the noise sweep the trials were won by: A 60.0%, B 40.0%.")
        self.assertEqual(wr["value"]["n_other"], 0)
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
        self.assertEqual(
            deg["text"],
            "A never exclusively beat C, so it does not appear above.")
        # REQUIRED, and stated LAST: it is the only place these names appear as a
        # group, and next to the win atoms it became a ready-made set for a
        # narrator to lift into them while dropping the negation it carried.
        self.assertIn("ob.degenerate", ok["required_atom_ids"])
        self.assertGreater(deg["order"],
                           max(a["order"] for a in ok["evidence"]
                               if a["id"] != "ob.degenerate"))

    def test_edges_are_ordered_biggest_margin_first(self):
        """Distinct win counts -> one atom each, LARGEST count first.

        This is the order the narrator follows, and both stages show only their
        opening sentences by default (WebUI.summarize._STAGE_SUMMARY), so it
        decides which rivals a reader sees without clicking.

        The opposite order was tried and rejected: smallest-first, on the reading
        that a rival the winner rarely beat outright is the one that ran it
        closest. Asserted for BOTH stages here, since they share one builder and
        the ordering is a property of that shared code.
        """
        for name, build, prefix in (("off_by_threshold", ir.build_off_by_ir, "ob"),
                                    ("gan", ir.build_gan_ir, "gn")):
            with self.subTest(stage=name):
                result = (_off_by_result if prefix == "ob" else _gan_result)(n_wins=10)
                pc = result["surrogates"]["per_competitor"]
                for rival, wins in (("D", 8), ("E", 6), ("F", 1)):
                    pc[rival] = dict(pc["B"], n_exclusive_wins=wins,
                                     exclusive_win_rate=wins / 40.0)
                doc = build("DS", "e1", result, ["A", "B"])
                _check_envelope(self, doc, name)
                req = set(doc["required_atom_ids"])
                for i in range(4):
                    self.assertIn(f"{prefix}.edge.{i}", req)
                wins = {a["id"]: a for a in doc["evidence"]
                        if a["type"] == "exclusive_wins"}
                self.assertEqual(wins[f"{prefix}.edge.0"]["value"]["competitors"], ["B"])
                self.assertEqual(wins[f"{prefix}.edge.3"]["value"]["competitors"], ["F"])
                self.assertIn("10 injected points", wins[f"{prefix}.edge.0"]["text"])
                self.assertIn("1 injected point ", wins[f"{prefix}.edge.3"]["text"])
                # Counts descend across the family, and `order` follows them.
                edges = sorted((a for a in doc["evidence"]
                                if a["type"] == "exclusive_wins"),
                               key=lambda a: a["order"])
                counts = [a["value"]["count"] for a in edges]
                self.assertEqual(counts, sorted(counts, reverse=True))

    def test_off_by_edges_merge_when_counts_identical(self):
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
        # try/except, not find_spec: another test module in the same run may
        # already have partially imported sklearn, leaving sklearn.__spec__ None,
        # and find_spec then raises ValueError instead of answering the question.
        try:
            from sklearn.tree import DecisionTreeClassifier
        except ImportError:
            self.skipTest("scikit-learn not installed")
        X = np.array([[0.01, 0.2], [0.02, 0.3], [0.04, 0.2], [0.05, 0.3]])
        y = np.array([1, 1, 0, 0])
        clf1 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
        clf2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
        result = _off_by_result(n_wins=8)
        pc = result["surrogates"]["per_competitor"]
        pc["B"] = dict(pc["B"], clf=clf1)
        pc["D"] = dict(pc["B"], clf=clf2)
        doc = ir.build_off_by_ir("DS", "e1", result, ["A", "B"])
        edges = [a for a in doc["evidence"] if a["type"] == "exclusive_wins"]
        # B and D share both the rule and the win count → ONE atom naming both.
        # Rule and count live in the SAME atom: as two families they expressed
        # the same rivals twice, in two groupings and two orders, which is what
        # let a narrator carry one group's names into the other's sentence.
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["value"]["competitors"], ["B", "D"])
        # Competitors read as a full oxford list and the condition is prose,
        # never a raw "feature op threshold" comparison.
        self.assertIn("apiece that B and D each miss", edges[0]["text"])
        self.assertIn("uniquely beating them when ", edges[0]["text"])
        self.assertIn("the distance from the boundary is at most", edges[0]["text"])
        self.assertNotIn("boundary_distance <=", edges[0]["text"])
        # No separate rule atom family survives to be cross-contaminated.
        self.assertEqual(
            [a for a in doc["evidence"] if a["type"] == "surrogate_rule"], [])

    # ── GAN: the same builder, a different vocabulary ───────────────────────
    #
    # build_gan_ir and build_off_by_ir share one implementation, so these do not
    # re-test the grouping, dedup and ordering above. They pin the three things
    # that are genuinely the GAN stage's own: its atom prefix, its feature prose,
    # and that the sibling's structure survives the parameterisation.

    def test_gan_ir_mirrors_off_by_with_its_own_prefix(self):
        doc = ir.build_gan_ir("DS", "e1", _gan_result(n_wins=8), ["A", "B"])
        _check_envelope(self, doc, "gan")
        ids = {a["id"] for a in doc["evidence"]}
        self.assertIn("gn.output.winner", ids)
        self.assertIn("gn.points", ids)
        self.assertIn("gn.degenerate", ids)
        # The off-by prefix must not leak through the shared builder.
        self.assertFalse([i for i in ids if i.startswith("ob.")])
        self.assertIn("gn.output.winner", doc["required_atom_ids"])
        self.assertIn("gn.degenerate", doc["required_atom_ids"])
        # Stated LAST, for the same reason it is in off-by: it is the only place
        # these names appear as a group.
        deg = next(a for a in doc["evidence"] if a["id"] == "gn.degenerate")
        self.assertGreater(deg["order"],
                           max(a["order"] for a in doc["evidence"]
                               if a["id"] != "gn.degenerate"))
        self.assertIn("generated points were injected near the discriminator",
                      next(a for a in doc["evidence"] if a["id"] == "gn.points")["text"])

    def test_gan_support_gate_names_the_gan_stage(self):
        low = ir.build_gan_ir("DS", "e1", _gan_result(n_wins=2), ["A", "B"])
        self.assertEqual(low["confidence"]["surrogate_vs_B"]["support"], "low")
        sup = next(c for c in low["caveats"] if c["id"] == "gn.caveat.support")
        self.assertIn("The rule for B rests on only 2 exclusive-win points", sup["text"])
        ok = ir.build_gan_ir("DS", "e1", _gan_result(n_wins=8), ["A", "B"])
        self.assertNotIn("gn.caveat.support", {c["id"] for c in ok["caveats"]})

    def test_gan_conditions_read_as_gan_prose(self):
        """The features differ from off-by's, so the rendered clauses must too —
        a shared builder that reused off-by's labels would silently describe a
        generated point as if it had been scaled away from a threshold."""
        try:
            from sklearn.tree import DecisionTreeClassifier
        except ImportError:
            self.skipTest("scikit-learn not installed")
        def _doc_for(X, y):
            clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
            result = _gan_result(n_wins=8)
            result["surrogates"]["feature_names"] = ["ambiguity", "is_anomaly"]
            pc = result["surrogates"]["per_competitor"]
            pc["B"] = dict(pc["B"], clf=clf,
                           feature_importances={"ambiguity": 0.9, "is_anomaly": 0.1})
            return ir.build_gan_ir("DS", "e1", result, ["A", "B"])

        # is_anomaly alternates, so it separates nothing and ambiguity is the
        # only split available.
        doc = _doc_for(np.array([[0.01, 0.0], [0.02, 1.0], [0.04, 0.0], [0.05, 1.0]]),
                       np.array([1, 1, 0, 0]))
        edge = next(a for a in doc["evidence"] if a["type"] == "exclusive_wins")
        self.assertIn("its distance from the discriminator's threshold is at most",
                      edge["text"])
        # Never a raw comparison, and never off-by's wording for the same slot.
        self.assertNotIn("ambiguity <=", edge["text"])
        self.assertNotIn("the distance from the boundary", edge["text"])
        top = next(a for a in doc["evidence"] if a["id"] == "gn.summary.top_feature")
        self.assertIn("the point's distance from the discriminator's threshold",
                      top["text"])

        # is_anomaly is a 0/1 label, so its 0.5 split becomes a statement rather
        # than a comparison — and the statement is the GAN's, not off-by's: these
        # points are labelled by the discriminator, not known to be real anomalies.
        doc = _doc_for(np.array([[0.01, 0.0], [0.02, 0.0], [0.04, 1.0], [0.05, 1.0]]),
                       np.array([1, 1, 0, 0]))
        edge = next(a for a in doc["evidence"] if a["type"] == "exclusive_wins")
        self.assertIn("the point was labelled normal", edge["text"])
        self.assertNotIn("real anomaly", edge["text"])

    def test_mc_win_rates_account_for_the_detectors_the_cut_drops(self):
        """The shares are of the same trials, so they sum to 100% across ALL
        winners. Listing only the top few left a reader adding up 96% and
        hunting for the bug, so the tail is stated rather than simply absent."""
        result = _mc_result()
        result["winner_f1"]["win_rates"] = {
            "A": 0.39, "B": 0.20, "C": 0.13, "D": 0.12, "E": 0.12,   # top 5
            "F": 0.03, "G": 0.01,                                    # cut
        }
        doc = ir.build_monte_carlo_ir("DS", "e1", result, ["A", "B"], ["B", "A"])
        wr = next(a for a in doc["evidence"] if a["id"] == "mc.surrogate.win_rates")
        self.assertIn("the remaining 4.0% went to 2 further detectors", wr["text"])
        self.assertEqual(wr["value"]["n_other"], 2)
        self.assertEqual(wr["value"]["other_share"], 0.04)
        # The cut detectors are still not named — that is what the cut is for.
        self.assertNotIn("F", [m for m, _ in wr["value"]["listed"]])
        # Listed shares plus the tail reach 100%.
        self.assertAlmostEqual(
            sum(r for _, r in wr["value"]["listed"]) + wr["value"]["other_share"],
            1.0, places=6)

    def test_mc_win_rates_singular_tail(self):
        result = _mc_result()
        result["winner_f1"]["win_rates"] = {
            "A": 0.39, "B": 0.20, "C": 0.13, "D": 0.12, "E": 0.12, "F": 0.04}
        doc = ir.build_monte_carlo_ir("DS", "e1", result, ["A", "B"], ["B", "A"])
        wr = next(a for a in doc["evidence"] if a["id"] == "mc.surrogate.win_rates")
        self.assertIn("went to 1 further detector.", wr["text"])

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
            ir.build_thompson_ranking_ir("DS", "e1", **_thompson_ranking_kwargs()),
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
