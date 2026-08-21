"""The property Thompson's reward depends on: it has to stay in [0, 1].

Nothing pinned this before, which is how the stage shipped for months computing
an "F1" of 8.52. These tests import the REAL Metrics.metrics (sklearn, vus,
loguru), so they run under the venv:

    .venv/bin/python -m unittest Model_Selection.test_reward_domain

The sibling test_thompson_sampling.py mocks Metrics.metrics away to stay
importable anywhere, so it structurally cannot cover this.
"""

import unittest

import numpy as np

from Metrics.metrics import f1_score, range_based_precision_recall_f1_auc
from Model_Selection.Thompson_Sampling import calculate_reward


def _negative_score_window(n=40, seed=0):
    """A window scored like LSTMVAE does: a Gaussian NLL, mostly below zero.

    On skab/1 the VAE's predicted sigma stays in [0.127, 0.657], so the
    0.5*log(var) term is always negative and 71.9% of timesteps score negative.
    """
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < 0.2).astype(int)
    var = rng.uniform(0.016, 0.43, n)          # sigma in [0.127, 0.657]
    resid = rng.random(n) ** 2 * var * 3
    y_scores = 0.5 * (np.log(var) + resid / var)
    return y_true, y_scores


class TestRawScoresAreNotPredictions(unittest.TestCase):
    """Why the old call was wrong, pinned so nobody reinstates it."""

    def test_f1_score_on_raw_scores_escapes_the_unit_interval(self):
        y_true, y_scores = _negative_score_window()
        self.assertLess(y_scores.min(), 0.0, "fixture should produce negatives")
        f1 = f1_score(y_scores, y_true)[0]
        # Not a bound check — a demonstration. f1_score sums `predict` as if it
        # were 0/1, so negative scores drive the precision denominator toward
        # zero and the result off the scale it is named for. It escapes in
        # EITHER direction depending on where sum(s) lands relative to zero:
        # this fixture gives -3.27, skab/1's real windows gave +8.52.
        self.assertTrue(f1 < 0.0 or f1 > 1.0,
                        f"expected a value outside [0, 1], got {f1}")

    def test_thresholding_the_same_scores_stays_bounded(self):
        y_true, y_scores = _negative_score_window()
        for cut in np.quantile(y_scores, np.linspace(0.0, 1.0, 21)):
            f1 = f1_score((y_scores >= cut).astype(int), y_true)[0]
            self.assertGreaterEqual(f1, 0.0)
            self.assertLessEqual(f1, 1.0)


class TestRangeBasedMetricIsBounded(unittest.TestCase):
    """What the reward now reads."""

    def test_f1_in_unit_interval_for_negative_scores(self):
        y_true, y_scores = _negative_score_window()
        _, _, f1, pr_auc, _ = range_based_precision_recall_f1_auc(y_true, y_scores)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)
        self.assertGreaterEqual(pr_auc, 0.0)
        self.assertLessEqual(pr_auc, 1.0)

    def test_f1_in_unit_interval_across_wildly_different_score_scales(self):
        """A detector scoring in [0.0004, 0.85] and one in [-1, 140] alike.

        These are the measured ranges of DGHL and LSTMVAE on skab/1 and SMD.
        """
        rng = np.random.default_rng(1)
        y_true = (rng.random(40) < 0.2).astype(int)
        for lo, hi in ((0.0004, 0.85), (-1.0, 140.0), (0.042, 0.39), (-2.07, 138.5)):
            with self.subTest(scale=(lo, hi)):
                y_scores = rng.uniform(lo, hi, 40)
                _, _, f1, _, _ = range_based_precision_recall_f1_auc(y_true, y_scores)
                self.assertGreaterEqual(f1, 0.0)
                self.assertLessEqual(f1, 1.0)

    def test_constant_scores_do_not_raise(self):
        """A degenerate detector still has to return a usable reward."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        _, _, f1, _, _ = range_based_precision_recall_f1_auc(y_true, np.full(10, 0.5))
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)


class TestRewardStaysInRange(unittest.TestCase):
    """calculate_reward is a convex combination, so bounded halves bound it."""

    def test_reward_bounded_when_both_halves_are(self):
        y_true, y_scores = _negative_score_window()
        _, _, f1, pr_auc, _ = range_based_precision_recall_f1_auc(y_true, y_scores)
        reward = calculate_reward(f1, pr_auc, 0.5, 0.5)
        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 1.0)

    def test_the_old_path_would_have_broken_this(self):
        y_true, y_scores = _negative_score_window()
        old_f1 = f1_score(y_scores, y_true)[0]
        old_reward = calculate_reward(old_f1, 0.5, 0.5, 0.5)
        self.assertTrue(old_reward < 0.0 or old_reward > 1.0,
                        f"expected an out-of-range reward, got {old_reward}")


if __name__ == "__main__":
    unittest.main()
