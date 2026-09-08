"""The VUS helpers behind the GA's fitness and the final ensemble-vs-single
decision, plus the fallback rule that guards both."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Utils.vus_utils import find_length
    from Metrics.metrics import vus_score, vus_window
    from app import _fmt_metric
except Exception:                                         # pragma: no cover
    find_length = vus_score = vus_window = _fmt_metric = None


# The pipeline's own interpreter has torch/statsmodels; the lighter one the
# Explainability/WebUI suites run under does not.
needs_app = unittest.skipIf(vus_score is None,
                            "needs the pipeline interpreter (torch, statsmodels)")


def series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.zeros(n)
    labels[n // 3: n // 3 + 40] = 1
    scores = rng.random(n)
    scores[labels == 1] += 0.6
    return scores, labels


@needs_app
class TestVusScore(unittest.TestCase):

    def test_a_separating_score_beats_a_random_one(self):
        good, labels = series()
        rng = np.random.default_rng(1)
        noise = rng.random(len(labels))
        w = find_length(good)
        self.assertGreater(vus_score(good, labels, w), vus_score(noise, labels, w))

    def test_is_bounded_and_finite(self):
        scores, labels = series()
        v = vus_score(scores, labels, find_length(scores))
        self.assertTrue(0.0 <= v <= 1.0, v)

    def test_scale_invariant(self):
        """Min-max scaling happens inside, so a rescaled score is the same VUS."""
        scores, labels = series()
        w = find_length(scores)
        self.assertAlmostEqual(vus_score(scores, labels, w),
                               vus_score(scores * 37.0 + 5.0, labels, w), places=6)

    def test_single_class_labels_are_nan_not_zero(self):
        """A degenerate label vector must not read as a real score of 0, which
        would silently hand the decision to the other candidate."""
        scores, _ = series()
        for labels in (np.zeros(len(scores)), np.ones(len(scores))):
            self.assertTrue(np.isnan(vus_score(scores, labels, 50)))

    def test_empty_and_mismatched_lengths(self):
        scores, labels = series()
        self.assertTrue(np.isnan(vus_score([], [], 50)))
        # Truncated to the shorter of the two rather than raising.
        self.assertFalse(np.isnan(vus_score(scores[:300], labels, find_length(scores))))

    def test_constant_scores_do_not_raise(self):
        _, labels = series()
        self.assertFalse(np.isinf(vus_score(np.ones(len(labels)), labels, 50)))

    def test_window_is_one_value_for_both_candidates(self):
        """Two candidates scored on different windows would not be comparable."""
        Y = np.tile(np.sin(np.arange(600) / 7.0), (4, 1))
        w = vus_window(Y)
        self.assertEqual(w, vus_window(Y))
        self.assertGreaterEqual(w, 2)
        self.assertNotEqual(w, 100, "fell through to the fallback, not a real window")

    def test_window_reads_the_first_feature_of_a_multivariate_series(self):
        row = np.sin(np.arange(600) / 7.0)
        self.assertEqual(vus_window(np.tile(row, (4, 1))), vus_window(row))

    def test_window_survives_a_malformed_series(self):
        self.assertGreaterEqual(vus_window(object()), 2)


@needs_app
class TestMetricFormatting(unittest.TestCase):

    def test_unavailable_reads_as_words_not_zero(self):
        for bad in (None, float("nan"), "x"):
            self.assertEqual(_fmt_metric(bad), "not available")

    def test_a_real_value_is_formatted(self):
        self.assertEqual(_fmt_metric(0.5), "0.500000")


class TestDecisionFallback(unittest.TestCase):
    # Pure arithmetic, so this one runs under either interpreter.
    """The rule app.py applies when the chosen metric has no value."""

    @staticmethod
    def choose(metric, by_metric):
        ens, sng = by_metric[metric]
        if np.isnan(ens) or np.isnan(sng):
            metric = "f1"
            ens, sng = by_metric["f1"]
        return metric, ("ensemble" if ens >= sng else "single_model")

    def setUp(self):
        # Ensemble leads on F1, trails on PR-AUC, and has no VUS.
        self.by_metric = {"f1": (0.90, 0.85), "pr_auc": (0.70, 0.95),
                          "vus": (float("nan"), 0.88)}

    def test_each_metric_picks_its_own_winner(self):
        self.assertEqual(self.choose("f1", self.by_metric), ("f1", "ensemble"))
        self.assertEqual(self.choose("pr_auc", self.by_metric), ("pr_auc", "single_model"))

    def test_unavailable_vus_falls_back_to_f1(self):
        """Comparing against nan is always False, which would have handed the
        run to the single model on a technicality."""
        self.assertEqual(self.choose("vus", self.by_metric), ("f1", "ensemble"))

    def test_ties_go_to_the_ensemble(self):
        self.assertEqual(self.choose("f1", {"f1": (0.5, 0.5)})[1], "ensemble")


if __name__ == "__main__":
    unittest.main()
