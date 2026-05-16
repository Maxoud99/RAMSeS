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
sys.modules["Metrics.metrics"].f1_score = lambda *a, **kw: (0.5, 0.5, 0.5, 0, 0, 0, 0)

# ── Now import the module under test ────────────────────────────────────────
from Thompson_Sampling import (
    initialize_sliding_windows,
    sample_model,
    update_posteriors,
    calculate_reward,
    rank_models,
    calculate_score,
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
            chosen = sample_model(self.models, self.means, self.covs,
                                  epsilon=0.0, context=self.context)
            self.assertIn(chosen, self.models)

    # --- epsilon=1 → always random pick ---------------------------------
    def test_epsilon_one_always_random(self):
        random.seed(0)
        chosen_set = {
            sample_model(self.models, self.means, self.covs,
                         epsilon=1.0, context=self.context)
            for _ in range(60)
        }
        # with 60 draws and 3 models, all should appear
        self.assertEqual(chosen_set, set(self.models.keys()))

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
            c = sample_model(models, means, covs, epsilon=0.0, context=context)
            wins[c] += 1

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
        Simulate SMD-scale data (38 channels × 20 time steps = d=760) for 100
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
            chosen  = sample_model(models, means, covs, epsilon, context)
            r       = fake_reward(chosen)
            update_posteriors(means, covs, chosen, r, context)
            epsilon *= epsilon_decay

        ranked = rank_models(means)
        best   = ranked[0][0]
        self.assertEqual(best, "DGHL",
            f"Expected DGHL to be ranked first, got {ranked}")


# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
