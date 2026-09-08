"""
Self-contained unit tests for the fixed Thompson_Sampling.py.
Mocks all RAMSeS-internal imports so this runs standalone.
"""

import sys
import types
import random
import unittest
import numpy as np

# ── Mock all RAMSeS-internal imports ────────────────────────────────────────
def _make_mock_module(*names):
    for name in names:
        parts = name.split(".")
        # build nested module chain
        parent = None
        for i, part in enumerate(parts):
            full = ".".join(parts[: i + 1])
            if full not in sys.modules:
                mod = types.ModuleType(full)
                sys.modules[full] = mod
                if parent is not None:
                    setattr(parent, part, mod)
            parent = sys.modules[full]

_make_mock_module("Metrics", "Metrics.Ensemble_GA", "Metrics.metrics")

# Stub the specific callables the module imports
sys.modules["Metrics.Ensemble_GA"].evaluate_individual_models   = lambda *a, **kw: None
sys.modules["Metrics.Ensemble_GA"].evaluate_model_consistently  = lambda *a, **kw: None
sys.modules["Metrics.metrics"].prauc    = lambda *a, **kw: 0.5
sys.modules["Metrics.metrics"].vus_score  = lambda *a, **kw: 0.5
sys.modules["Metrics.metrics"].vus_window = lambda *a, **kw: 10
sys.modules["Metrics.metrics"].f1_score = lambda *a, **kw: (0.5, 0.5, 0.5, 0, 0, 0, 0)
# The reward's F1 half comes from the range-based metric, as in every other
# stage; the module imports it at file scope, so the stub has to exist before
# Thompson_Sampling is imported. Signature: (precision, recall, f1, prauc, pred).
sys.modules["Metrics.metrics"].range_based_precision_recall_f1_auc = \
    lambda *a, **kw: (0.5, 0.5, 0.5, 0.5, None)

# ── Now import the module under test ────────────────────────────────────────
from Thompson_Sampling import (
    initialize_sliding_windows,
    sample_model,
    update_posteriors,
    calculate_reward,
    rank_models,
    calculate_score,
    compute_expected_rewards,
    detect_regime_shifts,
    classify_selection,
    compute_shap_values,
    aggregate_shap_per_context_feature,
    reconstruct_regime_segments,
    reward_contribution_per_context_feature,
    aggregate_squared_per_context_feature,
    rank_gap_decomposition,
    leadership_regimes,
    _top_k_models_by_expected_reward,
)


# ════════════════════════════════════════════════════════════════════════════
# Helper
# ════════════════════════════════════════════════════════════════════════════

def _make_prior(model_names, d):
    means       = {m: np.zeros(d) for m in model_names}
    covariances = {m: np.eye(d)   for m in model_names}
    return means, covariances


# ════════════════════════════════════════════════════════════════════════════
# 1.  initialize_sliding_windows
# ════════════════════════════════════════════════════════════════════════════

class TestSlidingWindows(unittest.TestCase):

    def test_basic_shape(self):
        data    = np.random.randn(1, 100)
        targets = np.random.randint(0, 2, 100)
        mask    = np.ones((1, 100))
        dw, tw, mw, n = initialize_sliding_windows(data, targets, mask,
                                                    window_size=20, step_size=10)
        self.assertEqual(n, len(dw))
        # with size=100, window=20, step=10 → 9 windows
        self.assertEqual(n, 9)
        for w in dw:
            self.assertEqual(w.shape, (1, 20))

    def test_empty_data_raises(self):
        with self.assertRaises(ValueError):
            initialize_sliding_windows(np.array([[]]), np.array([]),
                                       np.array([[]]), 10, 5)

    def test_bad_window_raises(self):
        data    = np.random.randn(1, 50)
        targets = np.random.randint(0, 2, 50)
        mask    = np.ones((1, 50))
        with self.assertRaises(ValueError):
            initialize_sliding_windows(data, targets, mask, 0, 5)


# ════════════════════════════════════════════════════════════════════════════
# 2.  sample_model  (the fixed function)
# ════════════════════════════════════════════════════════════════════════════

class TestSampleModel(unittest.TestCase):

    def setUp(self):
        self.models = {"LOF": object(), "RNN": object(), "DGHL": object()}
        self.d = 5
        self.means, self.covs = _make_prior(list(self.models), self.d)
        self.context = np.random.randn(self.d)

    # --- core correctness: returns a valid model name --------------------
    def test_returns_valid_model(self):
        for _ in range(20):
            chosen, _ = sample_model(self.models, self.means, self.covs,
                                     epsilon=0.0, context=self.context)
            self.assertIn(chosen, self.models)

    # --- epsilon=1 → always random pick ---------------------------------
    def test_epsilon_one_always_random(self):
        random.seed(0)
        chosen_set = set()
        random_flags = []
        for _ in range(60):
            chosen, was_random = sample_model(self.models, self.means, self.covs,
                                              epsilon=1.0, context=self.context)
            chosen_set.add(chosen)
            random_flags.append(was_random)
        # with 60 draws and 3 models, all should appear
        self.assertEqual(chosen_set, set(self.models.keys()))
        # epsilon=1.0 means every pick must be flagged as random
        self.assertTrue(all(random_flags))

    # --- epsilon=0 → uses context (θ̃ᵀx), not just θ̃[0] ----------------
    def test_uses_full_context_not_first_element(self):
        """
        Bias one model's mean strongly in one direction and context in
        another direction, then verify selection is driven by the dot product,
        not by the first element alone.

        Setup (d=3):
          context x = [0, 0, 1]   ← only the 3rd dimension matters
          LOF  μ = [10, 0, 0]     ← high on dim-0, zero on dim-2  → θ̃ᵀx ≈ 0
          RNN  μ = [ 0, 0, 5]     ← high on dim-2                 → θ̃ᵀx ≈ 5
          DGHL μ = [ 0, 0, 0]     ← zero everywhere               → θ̃ᵀx ≈ 0

        Correct LinTS: RNN wins (θ̃ᵀx is highest).
        Buggy code:    LOF wins (θ̃[0] is highest).
        """
        d = 3
        models = {"LOF": object(), "RNN": object(), "DGHL": object()}
        means = {
            "LOF":  np.array([10.0, 0.0, 0.0]),
            "RNN":  np.array([ 0.0, 0.0, 5.0]),
            "DGHL": np.array([ 0.0, 0.0, 0.0]),
        }
        # Very tight covariances so samples ≈ means (deterministic)
        covs = {m: np.eye(d) * 1e-6 for m in models}
        context = np.array([0.0, 0.0, 1.0])  # only dim-2 matters

        wins = {"LOF": 0, "RNN": 0, "DGHL": 0}
        N = 200
        for _ in range(N):
            c, was_random = sample_model(models, means, covs, epsilon=0.0, context=context)
            wins[c] += 1
            self.assertFalse(was_random, "epsilon=0.0 must never flag random pick")

        # RNN should win nearly every time
        self.assertGreater(wins["RNN"], 180,
            f"RNN should dominate (θ̃ᵀx ≈ 5) but wins={wins}")
        self.assertLess(wins["LOF"], 10,
            f"LOF should rarely win (θ̃ᵀx ≈ 0) but wins={wins}")

    # --- dimension mismatch raises --------------------------------------
    def test_mismatched_context_raises(self):
        bad_context = np.ones(self.d + 5)  # wrong size
        with self.assertRaises(Exception):
            sample_model(self.models, self.means, self.covs,
                         epsilon=0.0, context=bad_context)


# ════════════════════════════════════════════════════════════════════════════
# 3.  update_posteriors
# ════════════════════════════════════════════════════════════════════════════

class TestUpdatePosteriors(unittest.TestCase):

    def setUp(self):
        self.d = 4
        self.means, self.covs = _make_prior(["LOF", "RNN"], self.d)

    def test_only_chosen_model_updated(self):
        features = np.random.randn(self.d)
        mu_rnn_before  = self.means["RNN"].copy()
        cov_rnn_before = self.covs["RNN"].copy()

        update_posteriors(self.means, self.covs, "LOF", reward=0.8, features=features)

        # RNN must be UNCHANGED
        np.testing.assert_array_equal(self.means["RNN"],  mu_rnn_before)
        np.testing.assert_array_equal(self.covs["RNN"],   cov_rnn_before)

    def test_mean_shifts_after_update(self):
        features = np.ones(self.d)
        mu_before = self.means["LOF"].copy()
        update_posteriors(self.means, self.covs, "LOF", reward=1.0, features=features)
        # Mean must change
        self.assertFalse(np.allclose(self.means["LOF"], mu_before))

    def test_covariance_shrinks_after_update(self):
        """Diagonal of Σ should decrease (more certain after an observation)."""
        features = np.ones(self.d)
        diag_before = np.diag(self.covs["LOF"]).copy()
        update_posteriors(self.means, self.covs, "LOF", reward=0.5, features=features)
        diag_after = np.diag(self.covs["LOF"])
        self.assertTrue(np.all(diag_after < diag_before),
                        "Covariance diagonal should shrink after an update")

    def test_covariance_stays_symmetric(self):
        features = np.random.randn(self.d)
        update_posteriors(self.means, self.covs, "LOF", reward=0.6, features=features)
        cov = self.covs["LOF"]
        np.testing.assert_array_almost_equal(cov, cov.T,
                                             err_msg="Covariance should remain symmetric")

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            update_posteriors(self.means, self.covs, "UNKNOWN", 0.5, np.ones(self.d))

    def test_no_svd_failure_on_high_dimensional_data(self):
        """
        Simulate SMD-scale data (38 context features × 20 time steps = d=760) for 100
        iterations. The old double-inv code crashed with 'SVD did not converge'
        around iteration 40. Sherman-Morrison must complete all rounds without error.
        """
        d = 760
        means = {"SMD_model": np.zeros(d)}
        covs  = {"SMD_model": np.eye(d)}
        np.random.seed(0)
        for _ in range(100):
            x = np.random.randn(d) * 50       # large-magnitude SMD-like values
            x = x / (np.linalg.norm(x) + 1e-10)
            update_posteriors(means, covs, "SMD_model", reward=0.7, features=x)
        self.assertTrue(True, "Should complete 100 high-d iterations without SVD error")

    def test_old_precision_alias_bug_is_fixed(self):
        """
        In the old code `old_precision = precision` was a Python alias, not a copy.
        After `precision += outer(x, x)` both names pointed to the updated matrix,
        so the mean used the wrong (already updated) precision.

        With Sherman-Morrison and x=[1,0], reward=1, mu=0, Sigma=I:
            u = [1, 0], alpha = 1 + 1 = 2
            mu_new = [0,0] + [1,0] * (1 - 0) / 2 = [0.5, 0]
        The old buggy code would have produced [1.0, 0] or similar.
        """
        means = {"m": np.zeros(2)}
        covs  = {"m": np.eye(2)}
        update_posteriors(means, covs, "m", reward=1.0, features=np.array([1.0, 0.0]))
        np.testing.assert_array_almost_equal(
            means["m"], [0.5, 0.0],
            err_msg="Alias bug still present: mean update used wrong precision")

    def test_mean_stays_reasonable_after_many_updates(self):
        """
        Mean vector entries must not collapse to e-03 magnitude after many rounds
        (the symptom reported on SMD before the Sherman-Morrison fix).
        """
        d = 27
        means = {"m": np.zeros(d)}
        covs  = {"m": np.eye(d)}
        np.random.seed(1)
        for _ in range(80):
            x = np.random.randn(d)
            x = x / (np.linalg.norm(x) + 1e-10)
            update_posteriors(means, covs, "m", reward=0.8, features=x)
        max_abs = np.max(np.abs(means["m"]))
        self.assertGreater(max_abs, 1e-2,
            f"Mean collapsed to tiny values ({max_abs:.2e}) — numerical instability")


# ════════════════════════════════════════════════════════════════════════════
# 4.  calculate_reward
# ════════════════════════════════════════════════════════════════════════════

class TestCalculateReward(unittest.TestCase):

    def test_equal_weights(self):
        r = calculate_reward(f1=0.8, pr_auc=0.6, f1_weight=0.5, pr_auc_weight=0.5)
        self.assertAlmostEqual(r, 0.70)

    def test_f1_only(self):
        r = calculate_reward(f1=0.9, pr_auc=0.0, f1_weight=1.0, pr_auc_weight=0.0)
        self.assertAlmostEqual(r, 0.9)

    def test_zero_reward(self):
        self.assertAlmostEqual(calculate_reward(0, 0, 0.5, 0.5), 0.0)

    def test_weights_sum_to_one(self):
        # r must be in [0,1] when f1,pr_auc in [0,1] and weights sum to 1
        r = calculate_reward(1.0, 1.0, 0.7, 0.3)
        self.assertAlmostEqual(r, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# 5.  rank_models & calculate_score
# ════════════════════════════════════════════════════════════════════════════

class TestRankModels(unittest.TestCase):

    def test_ordering(self):
        means = {
            "LOF":  np.array([0.1, 0.1]),   # ‖μ‖² = 0.02
            "RNN":  np.array([0.5, 0.5]),   # ‖μ‖² = 0.50
            "DGHL": np.array([0.3, 0.3]),   # ‖μ‖² = 0.18
        }
        ranked = rank_models(means)
        names = [m for m, _ in ranked]
        self.assertEqual(names, ["RNN", "DGHL", "LOF"])

    def test_calculate_score(self):
        v = np.array([3.0, 4.0])
        self.assertAlmostEqual(calculate_score(v), 25.0)  # 3²+4² = 25


# ════════════════════════════════════════════════════════════════════════════
# 6.  Integration: full mini LinTS loop
# ════════════════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):

    def test_full_loop_runs_and_rankings_make_sense(self):
        """
        Simulate a small LinTS loop entirely in Python (no dataset objects).
        One model ('DGHL') always gets reward 0.9; others always get 0.3.
        After enough rounds, DGHL should be ranked first.
        """
        np.random.seed(42)
        random.seed(42)

        d       = 6
        models  = {"LOF": object(), "RNN": object(), "DGHL": object()}
        means, covs = _make_prior(list(models), d)

        def fake_reward(model_name):
            return 0.9 if model_name == "DGHL" else 0.3

        epsilon       = 0.0   # pure TS, no random
        epsilon_decay = 0.99
        n_rounds      = 30

        for _ in range(n_rounds):
            context = np.random.randn(d)
            chosen, _ = sample_model(models, means, covs, epsilon, context)
            r         = fake_reward(chosen)
            update_posteriors(means, covs, chosen, r, context)
            epsilon *= epsilon_decay

        ranked = rank_models(means)
        best   = ranked[0][0]
        self.assertEqual(best, "DGHL",
            f"Expected DGHL to be ranked first, got {ranked}")


# ════════════════════════════════════════════════════════════════════════════
# 7.  compute_expected_rewards
# ════════════════════════════════════════════════════════════════════════════

class TestComputeExpectedRewards(unittest.TestCase):

    def test_dot_product_correctness(self):
        means = {"LOF": np.array([1.0, 2.0, 3.0]), "RNN": np.array([0.0, 0.0, 1.0])}
        context = np.array([1.0, 1.0, 1.0])
        rewards = compute_expected_rewards(means, context)
        self.assertAlmostEqual(rewards["LOF"], 6.0)
        self.assertAlmostEqual(rewards["RNN"], 1.0)

    def test_negative_reward_possible(self):
        means = {"M": np.array([-1.0, -1.0])}
        context = np.array([1.0, 1.0])
        self.assertLess(compute_expected_rewards(means, context)["M"], 0.0)

    def test_all_models_returned(self):
        model_names = ["LOF_1", "LOF_2", "CBLOF_1", "NN_1"]
        means = {m: np.zeros(4) for m in model_names}
        context = np.ones(4)
        self.assertEqual(set(compute_expected_rewards(means, context).keys()), set(model_names))

    def test_column_vector_mean_handled(self):
        """2-D column mean (d, 1) must give the same result as 1-D (d,)."""
        ctx = np.array([3.0, 4.0])
        flat = compute_expected_rewards({"M": np.array([1.0, 2.0])}, ctx)["M"]
        col  = compute_expected_rewards({"M": np.array([[1.0], [2.0]])}, ctx)["M"]
        self.assertAlmostEqual(flat, col)


# ════════════════════════════════════════════════════════════════════════════
# 8.  detect_regime_shifts
# ════════════════════════════════════════════════════════════════════════════

def _make_reward_history(dominant_sequence, n_models=3):
    """Build expected_rewards_history where dominant_sequence[t] is the model
    index (0-based) that gets reward 1.0; all others get 0.0."""
    model_names = [f"M{i}" for i in range(n_models)]
    history = {m: [] for m in model_names}
    for dom in dominant_sequence:
        for i, m in enumerate(model_names):
            history[m].append(1.0 if i == dom else 0.0)
    return history, model_names


class TestDetectRegimeShifts(unittest.TestCase):

    def test_no_shift_returns_empty(self):
        history, _ = _make_reward_history([0] * 20)
        shifts, blips = detect_regime_shifts(history, smoothing_window=1, min_regime_length=3)
        self.assertEqual(shifts, [])
        self.assertEqual(blips, [])

    def test_single_sustained_shift_detected(self):
        history, _ = _make_reward_history([0] * 10 + [1] * 10)
        shifts, blips = detect_regime_shifts(history, smoothing_window=1, min_regime_length=3)
        self.assertEqual(len(shifts), 1)
        self.assertEqual(shifts[0]["from_model"], "M0")
        self.assertEqual(shifts[0]["to_model"], "M1")
        self.assertEqual(shifts[0]["window"], 10)
        self.assertEqual(blips, [])

    def test_blip_not_counted_as_shift(self):
        """M0 for 10, then M1 for 2 windows (< min_regime_length=3), then M0 again."""
        history, _ = _make_reward_history([0] * 10 + [1] * 2 + [0] * 10)
        shifts, blips = detect_regime_shifts(history, smoothing_window=1, min_regime_length=3)
        self.assertEqual(shifts, [], "Transient should not be a regime shift")
        self.assertGreater(len(blips), 0, "Transient should be recorded as blip")

    def test_reward_delta_is_positive(self):
        """The winning model should have higher smoothed reward at the shift window."""
        history, _ = _make_reward_history([0] * 8 + [1] * 8)
        shifts, _ = detect_regime_shifts(history, smoothing_window=1, min_regime_length=3)
        self.assertGreater(shifts[0]["reward_delta"], 0.0)

    def test_regime_length_in_shift_event(self):
        """regime_length should equal the old regime's window count."""
        history, _ = _make_reward_history([0] * 12 + [1] * 10)
        shifts, _ = detect_regime_shifts(history, smoothing_window=1, min_regime_length=3)
        self.assertEqual(shifts[0]["regime_length"], 12)

    def test_smoothing_suppresses_single_window_noise(self):
        """With smoothing_window=5, a single-window blip should be absorbed."""
        seq = [0] * 10 + [1] + [0] * 10
        history, _ = _make_reward_history(seq)
        shifts, _ = detect_regime_shifts(history, smoothing_window=5, min_regime_length=3)
        self.assertEqual(shifts, [], "Single-window blip must be smoothed away")

    def test_empty_history_returns_empty(self):
        shifts, blips = detect_regime_shifts({}, smoothing_window=1, min_regime_length=3)
        self.assertEqual(shifts, [])
        self.assertEqual(blips, [])

    def test_multiple_shifts(self):
        """Three distinct regimes should produce two shift events."""
        history, _ = _make_reward_history([0] * 8 + [1] * 8 + [2] * 8)
        shifts, _ = detect_regime_shifts(history, smoothing_window=1, min_regime_length=3)
        self.assertEqual(len(shifts), 2)
        self.assertEqual(shifts[0]["to_model"], "M1")
        self.assertEqual(shifts[1]["to_model"], "M2")


# ════════════════════════════════════════════════════════════════════════════
# 9.  classify_selection
# ════════════════════════════════════════════════════════════════════════════

class TestClassifySelection(unittest.TestCase):

    def test_random_overrides_argmax_match(self):
        """was_random=True must yield "random" even when chosen equals argmax."""
        expected = {"LOF": 0.9, "RNN": 0.5}
        # LOF is the argmax, but ε-greedy fired, so the state is still "random"
        self.assertEqual(classify_selection("LOF", True, expected), "random")

    def test_exploitation_when_chosen_matches_argmax(self):
        expected = {"LOF": 0.9, "RNN": 0.5, "DGHL": 0.1}
        self.assertEqual(classify_selection("LOF", False, expected), "exploitation")

    def test_informed_exploration_when_chosen_differs(self):
        expected = {"LOF": 0.9, "RNN": 0.5, "DGHL": 0.1}
        # RNN was sampled via TS even though LOF has the highest mean reward
        self.assertEqual(classify_selection("RNN", False, expected), "informed_exploration")

    def test_epsilon_one_yields_only_random_states(self):
        """Loop sample_model + classify_selection with ε=1.0; every state must be "random"."""
        random.seed(123)
        models = {"LOF": object(), "RNN": object(), "DGHL": object()}
        d = 4
        means, covs = _make_prior(list(models), d)
        states = []
        for _ in range(40):
            ctx = np.random.randn(d)
            chosen, was_random = sample_model(models, means, covs, epsilon=1.0, context=ctx)
            expected = compute_expected_rewards(means, ctx)
            states.append(classify_selection(chosen, was_random, expected))
        self.assertTrue(all(s == "random" for s in states))


# ════════════════════════════════════════════════════════════════════════════
# 10.  SHAP feature attribution
# ════════════════════════════════════════════════════════════════════════════

class TestSHAP(unittest.TestCase):

    def test_shap_additivity(self):
        """phi_0 + sum(phi_i) == mean^T context (SHAP additivity property)."""
        mean     = np.array([0.4, -0.7, 1.1, 0.2])
        context  = np.array([1.0, 2.0, -0.5, 3.0])
        baseline = np.array([0.5, 0.5, 0.5, 0.5])
        phi      = compute_shap_values(mean, context, baseline)
        phi_0    = float(np.dot(mean, baseline))
        self.assertAlmostEqual(phi_0 + float(phi.sum()), float(np.dot(mean, context)))

    def test_shap_zero_at_baseline(self):
        """When context == baseline, every per-feature SHAP must be exactly zero."""
        mean     = np.array([1.0, 2.0, 3.0])
        baseline = np.array([0.7, -0.3, 0.1])
        phi      = compute_shap_values(mean, baseline, baseline)
        np.testing.assert_array_almost_equal(phi, np.zeros(3))

    def test_shap_signs_track_feature_deltas(self):
        """With mean=ones, SHAP sign equals sign(context - baseline) element-wise."""
        mean     = np.array([1.0, 1.0, 1.0])
        context  = np.array([2.0, 0.0, 1.0])
        baseline = np.array([1.0, 1.0, 1.0])
        phi      = compute_shap_values(mean, context, baseline)
        np.testing.assert_array_almost_equal(phi, [+1.0, -1.0, 0.0])

    def test_per_context_feature_aggregation_correctness(self):
        """shap = [1, 2, 3, 4], n_context_features = 2 → per_context_feature = [1+2, 3+4] = [3, 7]."""
        shap = np.array([1.0, 2.0, 3.0, 4.0])
        per_context_feature = aggregate_shap_per_context_feature(shap, n_context_features=2)
        np.testing.assert_array_almost_equal(per_context_feature, [3.0, 7.0])

    def test_per_context_feature_aggregation_handles_uneven_division(self):
        """If shap.size is not divisible by n_context_features, trailing entries are dropped."""
        shap = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 elements, n_context_features=2
        per_context_feature = aggregate_shap_per_context_feature(shap, n_context_features=2)
        # window_size = 5 // 2 = 2; uses shap[:4] = [1, 2, 3, 4] → [3, 7]
        np.testing.assert_array_almost_equal(per_context_feature, [3.0, 7.0])


# ════════════════════════════════════════════════════════════════════════════
# 11.  reconstruct_regime_segments
# ════════════════════════════════════════════════════════════════════════════

class TestReconstructRegimeSegments(unittest.TestCase):

    def test_no_shifts_single_segment(self):
        segs = reconstruct_regime_segments([], T=20, fallback_model="LOF_1")
        self.assertEqual(segs, [(0, 19, "LOF_1", 20)])

    def test_empty_when_T_zero(self):
        self.assertEqual(reconstruct_regime_segments([], T=0), [])

    def test_segments_match_shift_events(self):
        """Shifts at windows 2, 17, 35 over T=50 → four contiguous regimes."""
        shifts = [
            {"window": 2,  "from_model": "LOF_4", "to_model": "NN_3"},
            {"window": 17, "from_model": "NN_3",  "to_model": "NN_1"},
            {"window": 35, "from_model": "NN_1",  "to_model": "NN_3"},
        ]
        segs = reconstruct_regime_segments(shifts, T=50)
        self.assertEqual(segs, [
            (0,  1,  "LOF_4", 2),
            (2,  16, "NN_3",  15),
            (17, 34, "NN_1",  18),
            (35, 49, "NN_3",  15),
        ])

    def test_segments_are_contiguous_and_cover_all_windows(self):
        shifts = [
            {"window": 5,  "from_model": "A", "to_model": "B"},
            {"window": 12, "from_model": "B", "to_model": "C"},
        ]
        T = 30
        segs = reconstruct_regime_segments(shifts, T=T)
        self.assertEqual(segs[0][0], 0)
        self.assertEqual(segs[-1][1], T - 1)
        for earlier, later in zip(segs, segs[1:]):
            self.assertEqual(earlier[1] + 1, later[0])
        self.assertEqual(sum(d for _, _, _, d in segs), T)


# ════════════════════════════════════════════════════════════════════════════
# 11b. Expected-reward decomposition (mu^T x per context feature)
# ════════════════════════════════════════════════════════════════════════════

class TestRewardContribution(unittest.TestCase):
    """The claim this decomposition makes, and SHAP cannot: the per-context-feature
    parts ARE the prediction. SHAP explains only the deviation from a typical
    window, discarding the mu.baseline term."""

    def test_parts_sum_to_the_expected_reward(self):
        rng = np.random.default_rng(4)
        mu, x = rng.normal(size=(24, 1)), rng.normal(size=24)
        parts = reward_contribution_per_context_feature(mu, x, n_context_features=6)
        self.assertEqual(parts.shape, (6,))
        self.assertAlmostEqual(float(parts.sum()),
                               float(np.dot(mu.flatten(), x)), places=12)

    def test_signed_both_ways(self):
        # mu and the normalised context can each be negative, so a context feature can
        # pull the expected reward down. An all-positive split would be a bug.
        mu = np.array([1.0, 1.0, -1.0, -1.0])
        x = np.array([1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(
            reward_contribution_per_context_feature(mu, x, n_context_features=2), [2.0, -2.0])

    def test_differs_from_shap_by_the_discarded_baseline_term(self):
        """SHAP per context feature = this, minus what the context feature contributes at the
        baseline. That difference is exactly the term SHAP throws away."""
        rng = np.random.default_rng(5)
        mu, x, b = rng.normal(size=12), rng.normal(size=12), rng.normal(size=12)
        raw = reward_contribution_per_context_feature(mu, x, 3)
        shap = aggregate_shap_per_context_feature(compute_shap_values(mu, x, b), 3)
        at_baseline = reward_contribution_per_context_feature(mu, b, 3)
        np.testing.assert_array_almost_equal(shap, raw - at_baseline)

    def test_degenerate_context_feature_counts(self):
        mu = x = np.array([1.0, 2.0])
        self.assertEqual(reward_contribution_per_context_feature(mu, x, 0).size, 0)
        np.testing.assert_array_almost_equal(
            reward_contribution_per_context_feature(mu, x, 4), np.zeros(4))

    def test_leader_minus_runner_sums_to_the_expected_reward_gap(self):
        """The regime prose's edge clause. Differencing two contribution splits
        gives context feature terms that sum EXACTLY to the gap in expected reward the
        same regime reports — so the sentence's context feature and its headline number
        describe one quantity. SHAP's version of the comparison sums to a
        baseline-relative gap instead, which is a different number."""
        rng = np.random.default_rng(11)
        mu_l, mu_r, x = (rng.normal(size=15), rng.normal(size=15),
                         rng.normal(size=15))
        delta = (reward_contribution_per_context_feature(mu_l, x, 5)
                 - reward_contribution_per_context_feature(mu_r, x, 5))
        self.assertAlmostEqual(float(delta.sum()),
                               float(np.dot(mu_l - mu_r, x)), places=12)


class TestTopKByExpectedReward(unittest.TestCase):
    """Which detectors a per-regime figure draws. The bars are computed from the
    beliefs held at each window, so the selection has to read the same ones — or
    a plot shows three detectors while the sentence beside it names a fourth as
    the runner-up."""

    def _means(self, a, b):
        return {"A": np.array([[a]]), "B": np.array([[b]])}

    def test_per_context_beliefs_beat_the_final_ones(self):
        contexts = [np.array([1.0]), np.array([1.0])]
        # A ends the run stronger, but B held the higher mu.x in both windows.
        final = self._means(5.0, 0.1)
        history = [{"A": np.array([0.1]), "B": np.array([9.0])},
                   {"A": np.array([0.1]), "B": np.array([9.0])}]
        self.assertEqual(
            _top_k_models_by_expected_reward(final, contexts, 1), ["A"])
        self.assertEqual(
            _top_k_models_by_expected_reward(final, contexts, 1,
                                             means_per_context=history), ["B"])

    def test_falls_back_per_window_when_history_is_short(self):
        """A truncated history must not drop the windows it does not cover."""
        contexts = [np.array([1.0]), np.array([1.0])]
        final = self._means(1.0, 1.0)
        history = [{"A": np.array([10.0]), "B": np.array([0.0])}]
        self.assertEqual(
            _top_k_models_by_expected_reward(final, contexts, 2,
                                             means_per_context=history),
            ["A", "B"])


# ════════════════════════════════════════════════════════════════════════════
# 12.  Ranking criterion ||mu_k||^2 — decomposition
# ════════════════════════════════════════════════════════════════════════════

class TestRankingDecomposition(unittest.TestCase):
    """The ranking stage's whole claim is that these two splits are EXACT: the
    shares account for all of a detector's score, the gap terms for all of its
    margin. If either stopped summing, the explanation would be quietly wrong
    while still looking plausible."""

    def test_per_context_feature_squares_sum_to_the_score(self):
        mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        per_context_feature = aggregate_squared_per_context_feature(mu, n_context_features=3)
        # context feature c owns a contiguous block: [1,2], [3,4], [5,6].
        np.testing.assert_array_almost_equal(per_context_feature, [5.0, 25.0, 61.0])
        self.assertAlmostEqual(float(per_context_feature.sum()), float(np.dot(mu, mu)))

    def test_contributions_are_never_negative(self):
        rng = np.random.default_rng(3)
        mu = rng.normal(size=(60, 1))          # column vector, as means are stored
        per_context_feature = aggregate_squared_per_context_feature(mu, n_context_features=6)
        self.assertTrue((per_context_feature >= 0).all())
        self.assertAlmostEqual(float(per_context_feature.sum()),
                               float(np.dot(mu.flatten(), mu.flatten())))

    def test_uneven_division_drops_the_tail_like_the_shap_helper(self):
        mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0])       # 5 // 2 = 2 per context feature
        np.testing.assert_array_almost_equal(
            aggregate_squared_per_context_feature(mu, n_context_features=2), [5.0, 25.0])

    def test_degenerate_context_feature_counts(self):
        mu = np.array([1.0, 2.0])
        self.assertEqual(aggregate_squared_per_context_feature(mu, 0).size, 0)
        # More context features than features: window_size floors to 0.
        np.testing.assert_array_almost_equal(
            aggregate_squared_per_context_feature(mu, 4), np.zeros(4))

    def test_gap_sums_to_the_score_difference(self):
        rng = np.random.default_rng(11)
        a, b = rng.normal(size=48), rng.normal(size=48)
        gap = rank_gap_decomposition(a, b, n_context_features=8)
        expected = float(np.dot(a, a)) - float(np.dot(b, b))
        self.assertAlmostEqual(float(gap.sum()), expected)

    def test_gap_is_antisymmetric_and_zero_against_itself(self):
        rng = np.random.default_rng(12)
        a, b = rng.normal(size=24), rng.normal(size=24)
        np.testing.assert_array_almost_equal(
            rank_gap_decomposition(a, b, 4), -rank_gap_decomposition(b, a, 4))
        np.testing.assert_array_almost_equal(
            rank_gap_decomposition(a, a, 4), np.zeros(4))


# ════════════════════════════════════════════════════════════════════════════
# 13.  Ranking criterion — leadership regimes
# ════════════════════════════════════════════════════════════════════════════

class TestLeadershipRegimes(unittest.TestCase):

    @staticmethod
    def _history(rows):
        """[{model: ||mu||^2}] -> the means_history shape the segmenter reads."""
        return [{m: np.array([np.sqrt(v)]) for m, v in row.items()} for row in rows]

    def test_run_length_encodes_the_argmax_without_merging(self):
        """Pure RLE, no minimum length: a one-window lead is a regime. This is
        the deliberate simplification over detect_regime_shifts, and it is what
        produces short regimes wherever two detectors are near-tied."""
        hist = self._history([{"A": 1.0, "B": 0.0}] * 10
                            + [{"A": 1.0, "B": 2.0}] * 3
                            + [{"A": 5.0, "B": 2.0}])
        segments, warmup = leadership_regimes(hist, warmup=10)
        self.assertEqual(warmup, 10)
        self.assertEqual(segments, [(10, 12, "B", 3), (13, 13, "A", 1)])

    def test_warmup_windows_are_excluded(self):
        """Until every arm has been sampled most scores are exactly zero, so
        leadership there reflects sampling order, not merit."""
        hist = self._history([{"A": 0.0, "B": 1.0}] * 10
                            + [{"A": 3.0, "B": 1.0}] * 10)
        segments, _ = leadership_regimes(hist, warmup=10)
        self.assertEqual(segments, [(10, 19, "A", 10)])
        # Window indices stay on the original timeline, so they still line up
        # with the plots and the per-window artefacts.
        self.assertEqual(segments[0][0], 10)

    def test_short_runs_collapse_the_warmup_rather_than_returning_nothing(self):
        hist = self._history([{"A": 1.0, "B": 0.0}] * 5)
        segments, warmup = leadership_regimes(hist, warmup=10)
        self.assertEqual(warmup, 0)
        self.assertEqual(segments, [(0, 4, "A", 5)])

    def test_exact_ties_never_manufacture_a_boundary(self):
        """All-zero scores are the normal early state; argmax over them must be
        stable or every window would look like a regime change."""
        hist = self._history([{"A": 0.0, "B": 0.0, "C": 0.0}] * 20)
        segments, _ = leadership_regimes(hist, warmup=10)
        self.assertEqual(len(segments), 1)

    def test_a_tie_at_the_top_keeps_the_incumbent(self):
        hist = self._history([{"A": 2.0, "B": 1.0}] * 12
                            + [{"A": 2.0, "B": 2.0}] * 4)
        segments, _ = leadership_regimes(hist, warmup=10)
        self.assertEqual([s[2] for s in segments], ["A"])

    def test_non_finite_scores_never_lead(self):
        """A failed window leaves NaN behind; it must not hand that detector
        the lead, and it must not split the surrounding regime in two."""
        hist = self._history([{"A": 1.0, "B": 0.5}] * 11
                            + [{"A": float("nan"), "B": 0.5}]
                            + [{"A": 1.0, "B": 0.5}] * 3)
        segments, _ = leadership_regimes(hist, warmup=10)
        self.assertEqual([(s[2]) for s in segments], ["A", "B", "A"])
        # The NaN window goes to B (the only finite score), one window wide.
        self.assertEqual(segments[1], (11, 11, "B", 1))

    def test_empty_history(self):
        self.assertEqual(leadership_regimes([], warmup=10), ([], 0))


# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
