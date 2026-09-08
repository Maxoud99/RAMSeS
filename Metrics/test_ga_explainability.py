"""
Standalone unit tests for the GA-ensemble selection-explainability layer.
Mocks the heavy module-level imports of Ensemble_GA.py (sklearn.*, loguru,
Metrics.metrics, Utils.model_selection_utils) so the pure analysis + plot
functions can be imported in any env that has numpy + matplotlib.
"""

import os
import re
import sys
import tempfile
import types
import unittest
import unittest.mock

import numpy as np


# ── Mock heavy module-level imports so Ensemble_GA.py loads ─────────────────
def _make_mock_module(*names):
    for name in names:
        parts = name.split(".")
        parent = None
        for i, part in enumerate(parts):
            full = ".".join(parts[: i + 1])
            if full not in sys.modules:
                mod = types.ModuleType(full)
                sys.modules[full] = mod
                if parent is not None:
                    setattr(parent, part, mod)
            parent = sys.modules[full]


_make_mock_module(
    "loguru",
    "sklearn",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.svm",
    "Metrics",
    "Metrics.metrics",
    "Utils",
    "Utils.model_selection_utils",
)
# Ensemble_GA does `from loguru import logger`
class _Logger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def error(self, *a, **k): pass
sys.modules["loguru"].logger = _Logger()
# Stub the specific classes / callables imported at module top.
sys.modules["sklearn.ensemble"].RandomForestClassifier = type("RandomForestClassifier", (), {})
sys.modules["sklearn.ensemble"].GradientBoostingClassifier = type("GradientBoostingClassifier", (), {})
sys.modules["sklearn.linear_model"].LogisticRegression = type("LogisticRegression", (), {})
sys.modules["sklearn.svm"].SVC = type("SVC", (), {})
sys.modules["Metrics.metrics"].prauc = lambda *a, **k: 0.5
sys.modules["Metrics.metrics"].f1_score = lambda *a, **k: (0.5,) * 7
sys.modules["Metrics.metrics"].vus_score = lambda *a, **k: 0.5
sys.modules["Utils.model_selection_utils"].evaluate_model = lambda *a, **k: None
sys.modules["Utils.model_selection_utils"].ScoringTimeout = type(
    "ScoringTimeout", (Exception,), {})


# Add project root to sys.path so `from Metrics.Ensemble_GA import ...` works.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Make the stubbed `Utils` a PACKAGE, so submodules that are safe to import for
# real resolve to the real files. `Utils.pipeline_spec` (the display-name map)
# and `Utils.plot_labels` (which only reads it) are deliberately stdlib-only —
# that is the whole reason pipeline_spec exists apart from `Utils.utils` — so
# there is nothing heavy to keep out, and stubbing them would mean this suite
# tested a fake rendering instead of the shipped one.
#
# Submodules already in sys.modules above keep their stubs: an entry there wins
# over the finder, so `Utils.model_selection_utils` stays mocked.
sys.modules["Utils"].__path__ = [os.path.join(_PROJECT_ROOT, "Utils")]

from Metrics.Ensemble_GA import (
    compute_lofo_utility,
    compute_mean_marginal_contribution,
    compute_survival_rates,
    classify_detector_archetypes,
    explain_ga_selection,
    compute_meta_shap,
    compute_meta_shap_values,
    compute_meta_pfi,
    compute_meta_ale,
    ale_signs,
    ale_sign_support,
    markov_aggregate_importances,
    explain_ga_combination,
)
from Metrics.Ensemble_GA import (_assign_archetype, ARCHETYPE_ORDER,
                                 _competition_ranks, score_fn_for,
                                 _best_threshold_f1)


class TestScoreFnFollowsFitness(unittest.TestCase):
    """PFI's scorer must be the same objective the GA maximises, or the
    combination ranking mixes one metric's importance with another's fitness."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = np.zeros(120)
        self.y[40:70] = 1
        self.s = rng.random(120)
        self.s[self.y == 1] += 0.5
        self.s = (self.s - self.s.min()) / np.ptp(self.s)

    def test_f1_scorer_is_the_best_threshold_f1(self):
        self.assertAlmostEqual(score_fn_for("f1")(self.y, self.s),
                               _best_threshold_f1(self.y, self.s), places=9)

    def test_pr_auc_scorer_reads_pr_auc_not_f1(self):
        """Both stubs return 0.5, so the dispatch is checked by patching the
        name Ensemble_GA bound at import time."""
        import Metrics.Ensemble_GA as ga
        with unittest.mock.patch.object(ga, "prauc", lambda *a, **k: 0.123):
            self.assertAlmostEqual(score_fn_for("pr_auc")(self.y, self.s), 0.123, places=9)
            self.assertNotAlmostEqual(score_fn_for("f1")(self.y, self.s), 0.123, places=3)

    def test_vus_scorer_reads_vus_when_a_window_is_given(self):
        import Metrics.Ensemble_GA as ga
        with unittest.mock.patch.object(ga, "vus_score", lambda *a, **k: 0.777):
            self.assertAlmostEqual(score_fn_for("vus", 50)(self.y, self.s), 0.777, places=9)

    def test_vus_without_a_window_falls_back_rather_than_returning_nan(self):
        """A nan importance would rank every detector equally and silently
        flatten the Markov consensus."""
        value = score_fn_for("vus", None)(self.y, self.s)
        self.assertFalse(np.isnan(value))
        self.assertAlmostEqual(value, _best_threshold_f1(self.y, self.s), places=9)

    def test_every_metric_yields_a_finite_score(self):
        for token in ("f1", "pr_auc"):
            self.assertTrue(np.isfinite(score_fn_for(token)(self.y, self.s)), token)


# ════════════════════════════════════════════════════════════════════════════
# 1.  compute_lofo_utility
# ════════════════════════════════════════════════════════════════════════════

class TestLofoUtility(unittest.TestCase):

    def test_marginal_equals_base_minus_reduced(self):
        # Stub evaluate_fitness: fitness = sum of indices of detectors in subset.
        # Detector 'A' contributes +1, 'B' +2, 'C' +3 → base = 6.
        # Removing 'B' → reduced = {A,C} = 4 → marginal_B = 6 − 4 = 2.
        values = {"A": 1.0, "B": 2.0, "C": 3.0}
        def evaluate_fitness(subset):
            return float(sum(values[d] for d in subset))
        lofo = compute_lofo_utility(["A", "B", "C"], evaluate_fitness)
        self.assertAlmostEqual(lofo["A"], 1.0)
        self.assertAlmostEqual(lofo["B"], 2.0)
        self.assertAlmostEqual(lofo["C"], 3.0)

    def test_singleton_ensemble_returns_nan(self):
        lofo = compute_lofo_utility(["A"], lambda s: 1.0)
        self.assertTrue(np.isnan(lofo["A"]))

    def test_empty_ensemble_returns_empty_dict(self):
        self.assertEqual(compute_lofo_utility([], lambda s: 1.0), {})


# ════════════════════════════════════════════════════════════════════════════
# 2.  compute_mean_marginal_contribution
# ════════════════════════════════════════════════════════════════════════════

class TestMeanMarginalContribution(unittest.TestCase):

    @staticmethod
    def _ee(subsets):
        """Build an evaluated_ensembles dict from {tuple: fitness} pairs.
        Values are (f1, pr_auc, fitness, y_scores, y_true) — only [2] matters."""
        return {tuple(sorted(k)): (0.0, 0.0, float(v), None, None)
                for k, v in subsets.items()}

    def test_difference_of_conditional_means(self):
        # E[fit|A present] over subsets containing A: 0.8, 0.6 → 0.7
        # E[fit|A absent]  over subsets without A : 0.2, 0.4 → 0.3
        # → contribution = 0.4
        ee = self._ee({
            ("A", "B"): 0.8,
            ("A", "C"): 0.6,
            ("B", "C"): 0.2,
            ("B",):     0.4,
        })
        mm = compute_mean_marginal_contribution(ee, ["A", "B", "C"])
        self.assertAlmostEqual(mm["A"]["contribution"], 0.4)
        self.assertAlmostEqual(mm["A"]["e_present"], 0.7)
        self.assertAlmostEqual(mm["A"]["e_absent"], 0.3)
        self.assertEqual(mm["A"]["n_present"], 2)
        self.assertEqual(mm["A"]["n_absent"], 2)

    def test_detector_in_every_subset_yields_nan(self):
        # A appears in all evaluated subsets → no "absent" sample → NaN.
        ee = self._ee({("A", "B"): 0.5, ("A", "C"): 0.7, ("A",): 0.3})
        mm = compute_mean_marginal_contribution(ee, ["A", "B", "C"])
        self.assertTrue(np.isnan(mm["A"]["contribution"]))
        self.assertEqual(mm["A"]["n_absent"], 0)
# ════════════════════════════════════════════════════════════════════════════
# 4.  compute_survival_rates
# ════════════════════════════════════════════════════════════════════════════

class TestSurvivalRates(unittest.TestCase):

    def test_counts_and_division(self):
        # Generation 1: 4 individuals; A in 3 of them → 0.75 (denom 4).
        # Generation 2: A in 2 of 4 → 0.5.
        gen_pops = [
            [["A", "B"], ["A", "C"], ["A", "B", "C"], ["B"]],
            [["A"],        ["B"],     ["A", "B"],      ["C"]],
        ]
        rates = compute_survival_rates(gen_pops, ["A", "B", "C"], 4)
        self.assertEqual(rates["A"], [0.75, 0.5])
        self.assertEqual(rates["B"], [0.75, 0.5])
        self.assertEqual(rates["C"], [0.5,  0.25])

    def test_handles_zero_population_size(self):
        # Defensive: division-by-zero must not crash.
        rates = compute_survival_rates([[["A"]]], ["A"], 0)
        # Denom falls back to 1 internally; rate equals the raw count.
        self.assertEqual(rates["A"], [1.0])


# ════════════════════════════════════════════════════════════════════════════
# 5.  classify_detector_archetypes
# ════════════════════════════════════════════════════════════════════════════

class TestArchetypes(unittest.TestCase):

    @staticmethod
    def _mm(contribs):
        return {d: {'contribution': v, 'e_present': float('nan'),
                    'e_absent': float('nan'), 'n_present': 0, 'n_absent': 0}
                for d, v in contribs.items()}

    def test_all_four_cells_unique(self):
        # Every (U, S) high/low cell maps to its own 2-letter H/L code.
        from itertools import product
        codes = {(u, s): _assign_archetype(u, s, util_nan=False)
                 for u, s in product([True, False], repeat=2)}
        self.assertEqual(len(set(codes.values())), 4)          # 4 cells, 4 codes
        # Codes are the (U,S) levels as H/L.
        self.assertEqual(codes[(True, True)], "HH")
        self.assertEqual(codes[(False, True)], "LH")
        self.assertEqual(codes[(True, False)], "HL")
        self.assertEqual(codes[(False, False)], "LL")
        # NaN utility short-circuits to Unclassified.
        self.assertEqual(_assign_archetype(True, True, util_nan=True), "Unclassified")
        # ARCHETYPE_ORDER = the 4 codes + Unclassified, no duplicates.
        self.assertEqual(len(ARCHETYPE_ORDER), 5)
        self.assertEqual(len(set(ARCHETYPE_ORDER)), 5)
        self.assertEqual(set(codes.values()), set(ARCHETYPE_ORDER) - {"Unclassified"})

    def test_core_support_marginal(self):
        # Absolute scheme has independent cutoffs (util>0, surv>0.5).
        algos = ["A", "B", "C"]
        mm = self._mm({"A": 0.5, "B": -0.1, "C": -0.2})
        surv = {"A": [0.6, 0.7, 0.8], "B": [0.6, 0.7, 0.8], "C": [0.1, 0.1, 0.1]}
        arch = classify_detector_archetypes(mm, surv, algos)
        # A = (H,H) → HH ; B = (L,H) → LH ; C = (L,L) → LL.
        self.assertEqual(arch["A"]["absolute"]["archetype"], "HH")
        self.assertEqual(arch["B"]["absolute"]["archetype"], "LH")
        self.assertEqual(arch["C"]["absolute"]["archetype"], "LL")

    def test_stability_uses_mean_only_not_trend(self):
        # High mean survival (0.6 > 0.5) but DECLINING trend — stability depends
        # only on the mean now, so s_high is True despite the downward trend.
        algos = ["D"]
        mm = self._mm({"D": 0.5})       # util > 0  → H
        surv = {"D": [0.9, 0.6, 0.3]}   # mean 0.6 > 0.5 → S high; trend −0.6
        arch = classify_detector_archetypes(mm, surv, algos)
        self.assertTrue(arch["D"]["absolute"]["s_high"])
        # (H, H) → HH; trend still reported for context.
        self.assertEqual(arch["D"]["absolute"]["archetype"], "HH")
        self.assertAlmostEqual(arch["D"]["stability_trend"], -0.6)

    def test_unclassified_on_nan_utility(self):
        algos = ["A", "B"]
        mm = self._mm({"A": float("nan"), "B": 0.3})
        surv = {"A": [0.8, 0.8], "B": [0.8, 0.8]}
        arch = classify_detector_archetypes(mm, surv, algos)
        self.assertEqual(arch["A"]["relative"]["archetype"], "Unclassified")
        self.assertEqual(arch["A"]["absolute"]["archetype"], "Unclassified")

    def test_relative_and_absolute_can_differ(self):
        # All utilities positive (absolute → high) but one below the median
        # (relative → low) → the two schemes disagree for at least one detector.
        algos = ["A", "B", "C"]
        mm = self._mm({"A": 0.2, "B": 0.3, "C": 0.4})
        surv = {"A": [0.1, 0.1, 0.1], "B": [0.1, 0.1, 0.1], "C": [0.1, 0.1, 0.1]}
        arch = classify_detector_archetypes(mm, surv, algos)
        differ = any(arch[d]["relative"]["archetype"] != arch[d]["absolute"]["archetype"]
                     for d in algos)
        self.assertTrue(differ)


# ════════════════════════════════════════════════════════════════════════════
# 6.  Combination layer — SHAP + PFI + Markov
# ════════════════════════════════════════════════════════════════════════════

class TestCombination(unittest.TestCase):

    def test_shap_linear_tracks_weights(self):
        # Meta-learner f(z) = 2*z0 + 3*z1 + 0*z2. With a mean baseline and feature
        # spreads equalised, mean|SHAP| ratio ≈ |weight|.  → SHAP[f1] > SHAP[f0] > SHAP[f2].
        feats = ["f0", "f1", "f2"]
        rng = np.random.RandomState(0)
        X = rng.rand(200, 3)
        baseline = X.mean(axis=0)
        def predict_fn(Z):
            return 2.0 * Z[:, 0] + 3.0 * Z[:, 1] + 0.0 * Z[:, 2]
        shap = compute_meta_shap(predict_fn, X, baseline, feats)
        self.assertGreater(shap["f1"], shap["f0"])
        self.assertGreater(shap["f0"], shap["f2"])
        self.assertAlmostEqual(shap["f2"], 0.0, places=6)

    def test_shap_single_feature_carries_all(self):
        feats = ["only"]
        X = np.array([[1.0], [0.0], [0.5]])
        baseline = X.mean(axis=0)
        shap = compute_meta_shap(lambda Z: Z[:, 0] * 2.0, X, baseline, feats)
        self.assertGreater(shap["only"], 0.0)

    def test_shap_signed_reflects_direction(self):
        # f(z) = z_pos − z_neg with a zero baseline and positive inputs ⇒ phi_pos > 0,
        # phi_neg < 0 every row. mode="signed" keeps the direction; mode="abs" doesn't.
        feats = ["pos", "neg"]
        X = np.random.RandomState(0).rand(200, 2)   # all entries in [0, 1]
        baseline = np.zeros(2)
        predict_fn = lambda Z: Z[:, 0] - Z[:, 1]
        signed = compute_meta_shap(predict_fn, X, baseline, feats, mode="signed")
        abs_ = compute_meta_shap(predict_fn, X, baseline, feats, mode="abs")
        self.assertGreater(signed["pos"], 0.0)
        self.assertLess(signed["neg"], 0.0)
        # abs magnitude is positive for the negative-direction feature too.
        self.assertGreater(abs_["neg"], 0.0)

    def test_shap_matrix_and_aggregates_agree(self):
        """compute_meta_shap is now a thin aggregation over the per-row matrix,
        so the enumeration runs once for both summaries instead of twice."""
        feats = ["a", "b", "c"]
        X = np.random.RandomState(3).rand(50, 3)
        baseline = X.mean(axis=0)
        f = lambda Z: 2 * Z[:, 0] - Z[:, 1]
        phi = compute_meta_shap_values(f, X, baseline, len(feats))
        self.assertEqual(phi.shape, (50, 3))
        for i, name in enumerate(feats):
            self.assertAlmostEqual(
                compute_meta_shap(f, X, baseline, feats, mode="abs")[name],
                float(np.abs(phi[:, i]).mean()))
            self.assertAlmostEqual(
                compute_meta_shap(f, X, baseline, feats, mode="signed")[name],
                float(phi[:, i].mean()))

    # ── ALE ─────────────────────────────────────────────────────────────────

    def test_ale_recovers_weight_times_range_for_a_linear_model(self):
        """For a linear model the accumulated effect IS the coefficient times
        how far the feature actually moves — the anchor that makes ALE the
        generalisation of 'the weight' for a model that has none."""
        feats = ["a", "b"]
        a = np.linspace(0.05, 0.95, 200)
        X = np.column_stack([a, np.random.RandomState(0).rand(200)])
        ale = compute_meta_ale(lambda Z: 0.5 * Z[:, 0], X, feats)
        self.assertAlmostEqual(ale["a"]["net"], 0.5 * (0.95 - 0.05), places=6)
        self.assertAlmostEqual(ale["a"]["consistency"], 1.0, places=6)
        # A feature the model ignores moves nothing.
        self.assertAlmostEqual(ale["b"]["total_variation"], 0.0, places=9)

    def test_ale_book_keeping(self):
        feats = ["a", "b"]
        X = np.random.RandomState(1).rand(120, 2)
        ale = compute_meta_ale(lambda Z: Z[:, 0] ** 2 - Z[:, 1], X, feats)
        for rec in ale.values():
            self.assertAlmostEqual(sum(rec["deltas"]), rec["net"], places=12)
            self.assertAlmostEqual(sum(abs(v) for v in rec["deltas"]),
                                   rec["total_variation"], places=12)
            # The curve starts at zero and ends at the net effect.
            self.assertEqual(len(rec["curve"]), len(rec["edges"]))
            self.assertAlmostEqual(rec["curve"][0], 0.0)
            self.assertAlmostEqual(rec["curve"][-1], rec["net"], places=12)

    def test_ale_separates_a_cancelling_effect_from_no_effect(self):
        """The failure that made signed SHAP unusable: an effect that rises then
        falls nets to nothing. Total variation must still see it, or the
        detector ranks below pure noise."""
        feats = ["u", "n"]
        u = np.linspace(0.0, 1.0, 300)
        X = np.column_stack([u, np.random.RandomState(2).rand(300)])
        ale = compute_meta_ale(lambda Z: np.abs(Z[:, 0] - 0.5) * 2, X, feats)
        self.assertAlmostEqual(ale["u"]["net"], 0.0, places=2)
        self.assertGreater(ale["u"]["total_variation"], 0.9)
        self.assertLess(ale["u"]["consistency"], 0.1)

    def test_ale_is_deterministic_and_handles_a_constant_column(self):
        feats = ["a", "const"]
        X = np.column_stack([np.linspace(0, 1, 60), np.full(60, 0.7)])
        f = lambda Z: Z[:, 0] + Z[:, 1]
        first = compute_meta_ale(f, X, feats)
        self.assertEqual(first["const"]["n_bins"], 0)
        self.assertNotEqual(first["const"]["net"], first["const"]["net"])  # NaN
        self.assertEqual(first["a"]["net"], compute_meta_ale(f, X, feats)["a"]["net"])

    def _sign_fixture(self):
        return {
            "up":     {"net": 0.8, "total_variation": 0.8, "consistency": 1.0, "n_bins": 10},
            "down":   {"net": -0.8, "total_variation": 0.8, "consistency": 1.0, "n_bins": 10},
            "swings": {"net": -0.01, "total_variation": 0.8, "consistency": 0.0125, "n_bins": 10},
            "tiny":   {"net": 0.01, "total_variation": 0.01, "consistency": 1.0, "n_bins": 10},
            "flat":   {"net": 0.0, "total_variation": 0.0,
                       "consistency": float("nan"), "n_bins": 10},
            "gone":   {"net": float("nan"), "total_variation": float("nan"),
                       "consistency": float("nan"), "n_bins": 0},
        }

    def test_sign_is_reported_whenever_there_is_one(self):
        """The sign is where the curve ends, and nothing else decides it.

        A weakly-supported sign is still a measurement: suppressing it turned a
        detector measured at a clear negative into a blank, which reads as
        missing data. Only two cases have no sign to take — a net of exactly
        zero (neither direction is true) and an ALE that could not be computed.
        """
        ale = self._sign_fixture()
        got = ale_signs(ale, list(ale))
        self.assertEqual(got["up"], "positive")
        self.assertEqual(got["down"], "negative")
        self.assertEqual(got["swings"], "negative")   # moves both ways, ends down
        self.assertEqual(got["tiny"], "positive")     # small, but a real direction
        self.assertEqual(got["flat"], "not_available")
        self.assertEqual(got["gone"], "not_available")

    def test_sign_support_names_why_a_sign_is_thin(self):
        """The gates that used to suppress a sign now qualify it, and each
        reports its own reason — the two are different findings."""
        ale = self._sign_fixture()
        got = ale_sign_support(ale, list(ale))
        self.assertEqual(got["up"], [])
        self.assertEqual(got["down"], [])
        self.assertEqual(got["swings"], ["low_consistency"])
        self.assertEqual(got["tiny"], ["weak_influence"])
        # Nothing to qualify where there is no sign in the first place.
        self.assertEqual(got["flat"], [])
        self.assertEqual(got["gone"], [])

    def test_sign_support_floor_is_relative_to_this_ensemble(self):
        """The magnitude gate is a fraction of the strongest detector here, so
        the same absolute movement is weak beside a big detector and strong on
        its own. Units are the meta-learner's own probability scale, which has
        no absolute threshold to appeal to."""
        rec = lambda tv: {"net": tv, "total_variation": tv,
                          "consistency": 1.0, "n_bins": 10}
        beside_big = ale_sign_support({"small": rec(0.01), "big": rec(1.0)},
                                      ["small", "big"])
        self.assertEqual(beside_big["small"], ["weak_influence"])
        alone = ale_sign_support({"small": rec(0.01)}, ["small"])
        self.assertEqual(alone["small"], [])

    def test_pfi_informative_vs_noise(self):
        # f uses only column 0; y derived from column 0. Permuting col 0 hurts the
        # score; permuting the noise col 1 does not.
        feats = ["info", "noise"]
        rng = np.random.RandomState(1)
        X = rng.rand(300, 2)
        y = (X[:, 0] >= 0.5).astype(int)
        def predict_fn(Z):
            return Z[:, 0]
        def acc(yy, ss):
            return float(np.mean((ss >= 0.5).astype(int) == yy))
        pfi = compute_meta_pfi(predict_fn, X, y, feats, score_fn=acc,
                               n_repeats=5, random_state=0)
        self.assertGreater(pfi["info"], 0.1)
        self.assertLess(abs(pfi["noise"]), 0.05)

    def test_markov_aggregates_rankings(self):
        feats = ["A", "B", "C"]
        shap = {"A": 0.9, "B": 0.5, "C": 0.1}   # A > B > C
        pfi = {"A": 0.8, "B": 0.1, "C": 0.4}    # A > C > B
        scores, final = markov_aggregate_importances({"SHAP": shap, "PFI": pfi}, feats)
        self.assertEqual(final[0], "A")                       # wins both → top
        self.assertEqual(scores["A"], max(scores.values()))  # highest stationary prob
        self.assertAlmostEqual(sum(scores.values()), 1.0)    # π is a distribution

    def test_markov_plot_title_names_every_source_that_feeds_the_chain(self):
        """The figure's label and the chain's actual inputs, checked against
        each other. ALE was added to the aggregation but the title kept saying
        "mean|SHAP| + PFI", so the plot claimed a two-source consensus beside a
        panel drawing three bars — a caption drifting off its own figure is
        invisible to every other test here.
        """
        import inspect
        from Metrics import Ensemble_GA
        agg = inspect.getsource(Ensemble_GA.explain_ga_combination)
        title = inspect.getsource(Ensemble_GA.plot_ga_combination)
        call = agg.split("markov_aggregate_importances(")[1].split(")")[0]
        # Every source key handed to the aggregator must be recognisable in the
        # title, and the count must match — a title naming four would be as
        # wrong as one naming two.
        keys = re.findall(r'"([A-Za-z_]+)":', call)
        self.assertEqual(sorted(keys), ["ALE", "PFI", "SHAP_abs"])
        shown = title.split('set_title("Final ranking')[1].split('")')[0]
        for token in ("SHAP", "PFI", "ALE"):
            self.assertIn(token, shown, shown)
        self.assertEqual(shown.count("+"), len(keys) - 1, shown)

    def test_markov_single_feature(self):
        scores, final = markov_aggregate_importances({"SHAP": {"only": 0.5}}, ["only"])
        self.assertEqual(final, ["only"])
        self.assertAlmostEqual(scores["only"], 1.0)

    def test_competition_ranks_share_rank_on_ties(self):
        # B and C tie (1 pt each), D and E tie (3 pts each).
        points = {"A": 6.0, "B": 4.0, "C": 4.0, "D": 3.0, "E": 3.0}
        order = ["A", "B", "C", "D", "E"]          # points-descending
        ranks = _competition_ranks(points, order)
        self.assertEqual(ranks["A"], 1)
        self.assertEqual(ranks["B"], 2)
        self.assertEqual(ranks["C"], 2)            # tied with B → same rank
        self.assertEqual(ranks["D"], 4)            # skips 3 (competition style)
        self.assertEqual(ranks["E"], 4)

    def test_competition_ranks_tolerate_eigenvector_noise(self):
        """Markov scores that are mathematically tied come back from
        np.linalg.eig a few ulp apart — only two measures feed the chain, so
        exact ties are the norm. An exact `!=` turned that wobble into a real
        rank difference in the report table."""
        points = {"A": 0.18347554726124723,       # one ulp above the other two
                  "B": 0.18347554726124712,
                  "C": 0.18347554726124712,
                  "D": 0.17937750940504008}
        ranks = _competition_ranks(points, ["A", "B", "C", "D"])
        self.assertEqual(ranks, {"A": 1, "B": 1, "C": 1, "D": 4})

    def test_markov_aggregation_ties_are_exact_not_ordered(self):
        """The three-way tie this came from: with two measures each pair splits
        1-1, so the stationary distribution is equal for all three and only
        float noise separates them."""
        feats = ["X", "Y", "Z"]
        a = {"X": -3.0, "Y": -2.0, "Z": -1.0}     # ranks X<Y<Z
        b = {"X": -1.0, "Y": -2.0, "Z": -3.0}     # exactly reversed
        scores, ranking = markov_aggregate_importances(
            {"SHAP_abs": a, "PFI": b}, feats)
        spread = max(scores.values()) - min(scores.values())
        self.assertLess(spread, 1e-9)
        # Whatever arbitrary order `ranking` came back in, all three share rank 1.
        self.assertEqual(set(_competition_ranks(scores, ranking).values()), {1})

    def test_fitness_function_returns_meta_model(self):
        # Step 0: fitness_function appends the trained meta-model as a 6th element,
        # while the first five indices stay unchanged (callers index [0..4]).
        import Metrics.Ensemble_GA as GA

        class _FakeModel:
            def predict_proba(self, Z):
                n = len(Z)
                return np.column_stack([np.zeros(n), np.full(n, 0.6)])

        orig = GA.train_meta_model_rf
        GA.train_meta_model_rf = lambda X, y: _FakeModel()
        try:
            algos = ["A", "B", "C"]
            rng = np.random.RandomState(0)
            Xtr, Xte = rng.rand(20, 3), rng.rand(15, 3)
            ytr = (rng.rand(20) >= 0.5).astype(int)
            yte = (rng.rand(15) >= 0.5).astype(int)
            res = GA.fitness_function(["A", "C"], None, None, None, None,
                                      Xtr, algos, Xte, ytr, yte, meta_model_type="rf")
            self.assertEqual(len(res), 6)
            self.assertIsInstance(res[5], _FakeModel)
        finally:
            GA.train_meta_model_rf = orig

    def test_explain_combination_writes_outputs(self):
        algorithm_list = ["A", "B", "C"]
        best_ensemble = ["A", "C"]               # features = [A, C] (algorithm_list order)
        rng = np.random.RandomState(0)
        Xtr = rng.rand(50, 3)
        Xte = rng.rand(40, 3)
        ytr = (Xtr[:, 0] >= 0.5).astype(int)
        yte = (Xte[:, 0] >= 0.5).astype(int)
        # Injected predict_fn over the FILTERED (2-column) space: relies on A only.
        def predict_fn(Z):
            return Z[:, 0]
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = explain_ga_combination(
                    best_ensemble, algorithm_list, Xtr, Xte, ytr, yte,
                    meta_model_type="rf", dataset="TEST", entity="e1",
                    predict_fn=predict_fn, explain=True,
                )
                self.assertIsInstance(result, dict)
                for key in ("feature_names", "shap_importance", "shap_signed_importance",
                            "pfi_importance", "markov_scores", "final_ranking",
                            "baseline_f1", "model_source",
                            "ale_total_variation", "ale_net", "ale_consistency",
                            "ale_sign", "ale_sign_support", "ale_curves",
                            "ale_n_bins"):
                    self.assertIn(key, result)
                self.assertEqual(result["feature_names"], ["A", "C"])
                self.assertEqual(sorted(result["final_ranking"]), ["A", "C"])
                # predict_fn returns column 0 (=A) unchanged, so A rises with its
                # own score and C does nothing at all.
                self.assertEqual(result["ale_sign"]["A"], "positive")
                self.assertAlmostEqual(result["ale_total_variation"]["C"], 0.0,
                                       places=9)
                out = os.path.join("myresults", "GA_Ens", "TEST", "e1")
                self.assertTrue(os.path.exists(os.path.join(
                    out, "ga_combination_explainability_TEST_e1.txt")))
                self.assertTrue(os.path.exists(os.path.join(
                    out, "ga_combination_importance_TEST_e1.png")))
                self.assertTrue(os.path.exists(os.path.join(
                    out, "ga_combination_ale_TEST_e1.png")))
                # The bin-marked view is a second figure, not a replacement:
                # the plain curve is the one that belongs in a thesis.
                self.assertTrue(os.path.exists(os.path.join(
                    out, "ga_combination_ale_bins_TEST_e1.png")))
                # The superseded signed-SHAP table stays in the report, labelled.
                with open(os.path.join(
                        out, "ga_combination_explainability_TEST_e1.txt")) as fh:
                    report = fh.read()
                self.assertIn("SUPERSEDED BY ALE", report)
                self.assertIn("Markov aggregation (SHAP |.| + PFI + ALE)", report)
                # Intermediate Representation JSON is emitted alongside.
                import json
                ir_path = os.path.join("myresults", "explanations_ir", "TEST", "e1",
                                       "ir_ga_combination.json")
                self.assertTrue(os.path.exists(ir_path), ir_path)
                with open(ir_path) as fh:
                    self.assertEqual(json.load(fh)["stage"], "ga_combination")
            finally:
                os.chdir(cwd)


# ════════════════════════════════════════════════════════════════════════════
# 7.  explain_ga_selection — integration smoke test
# ════════════════════════════════════════════════════════════════════════════

class TestExplainGASelection(unittest.TestCase):

    def test_writes_report_and_three_plots(self):
        algorithm_list = ["A", "B", "C"]
        best_ensemble = ["A", "B"]
        ee = {tuple(sorted(k)): (0.0, 0.0, float(v), None, None) for k, v in {
            ("A", "B"):     0.8,
            ("A", "C"):     0.5,
            ("B", "C"):     0.4,
            ("A", "B", "C"): 0.7,
            ("A",):          0.3,
            ("B",):          0.4,
            ("C",):          0.2,
        }.items()}
        gen_pops = [
            [["A", "B"], ["A", "C"], ["B", "C"], ["A", "B", "C"]],
            [["A", "B"], ["A", "B"], ["B", "C"], ["A", "B", "C"]],
        ]
        # Fitness closure for LOFO — uses the evaluated_ensembles when cached
        # and a simple stub otherwise (test triggers paths for both).
        def evaluate_fitness(subset):
            key = tuple(sorted(subset))
            if key in ee:
                return ee[key][2]
            return 0.6  # fresh subsets get a fixed stub value

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = explain_ga_selection(
                    best_ensemble, ee, gen_pops, algorithm_list,
                    population_size=4,
                    evaluate_fitness=evaluate_fitness,
                    dataset="TEST", entity="e1", explain=True,
                )
                self.assertIsInstance(result, dict)
                for key in ("best_ensemble", "lofo", "mean_marginal",
                            "survival",
                            "archetypes", "n_subsets_evaluated", "n_generations"):
                    self.assertIn(key, result)

                out = os.path.join("myresults", "GA_Ens", "TEST", "e1")
                self.assertTrue(os.path.exists(
                    os.path.join(out, "ga_selection_explainability_TEST_e1.txt")))
                self.assertTrue(os.path.exists(
                    os.path.join(out, "ga_selection_utility_TEST_e1.png")))
                # Intermediate Representation JSON is emitted alongside.
                import json
                ir_path = os.path.join("myresults", "explanations_ir", "TEST", "e1",
                                       "ir_ga_selection.json")
                self.assertTrue(os.path.exists(ir_path), ir_path)
                with open(ir_path) as fh:
                    ir_doc = json.load(fh)
                self.assertEqual(ir_doc["stage"], "ga_selection")
                self.assertTrue(os.path.exists(
                    os.path.join(out, "ga_selection_survival_TEST_e1.png")))
                self.assertTrue(os.path.exists(
                    os.path.join(out, "ga_selection_survival_all_TEST_e1.png")))
                self.assertTrue(os.path.exists(
                    os.path.join(out, "ga_selection_archetypes_TEST_e1.png")))
            finally:
                os.chdir(cwd)

    def test_explain_false_is_noop(self):
        result = explain_ga_selection(
            ["A", "B"], {}, [], ["A", "B"],
            population_size=2,
            evaluate_fitness=lambda s: 0.5,
            dataset="X", entity="Y",
            explain=False,
        )
        self.assertIsNone(result)


# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
