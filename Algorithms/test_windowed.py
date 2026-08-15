"""
The window handling shared by every PyOD-backed detector.

These detectors used to fit on one sample per NUMBER with a single feature, so a
9-channel by 64-timestep window became 576 unrelated one-dimensional points and a
detector could only ask "is this value rare overall". The tests below pin the
shape that replaced it — one sample per window, the flattened window as its
feature vector — because the difference is invisible in any output the pipeline
prints and would be easy to undo by accident.

Run with the project venv (torch and pyod are needed):
    .venv/bin/python -m unittest Algorithms.test_windowed
"""

import unittest

import numpy as np
import torch as t

from Algorithms import windowed


class _Loader:
    """The one attribute `fit_windows` reads off a real Loader."""

    def __init__(self, Y_windows):
        self.Y_windows = Y_windows


class _Recorder:
    """Stands in for a PyOD estimator and remembers the shape it was handed."""

    def __init__(self, scores=None):
        self.seen = None
        self._scores = scores

    def fit(self, X):
        self.seen = np.asarray(X).shape

    def decision_function(self, X):
        X = np.asarray(X)
        self.seen = X.shape
        if self._scores is not None:
            return np.asarray(self._scores, dtype=float)
        return np.arange(len(X), dtype=float)


class TestWindowsAsRows(unittest.TestCase):

    def test_one_row_per_window_not_one_per_reading(self):
        Y = t.zeros((5, 9, 64))          # 5 windows, 9 channels, 64 timesteps
        rows = windowed.windows_as_rows(Y)
        self.assertEqual(rows.shape, (5, 576))
        # The shape the old code produced, for contrast: 2880 samples of 1.
        self.assertNotEqual(rows.shape, (5 * 9 * 64, 1))

    def test_the_window_survives_intact(self):
        """Row i has to be window i's own readings, in order — a reshape that
        mixed windows together would still give the right shape."""
        Y = t.arange(2 * 3 * 4, dtype=t.float32).reshape(2, 3, 4)
        rows = windowed.windows_as_rows(Y)
        np.testing.assert_array_equal(rows[0], np.arange(12))
        np.testing.assert_array_equal(rows[1], np.arange(12, 24))

    def test_accepts_numpy_as_well_as_torch(self):
        self.assertEqual(windowed.windows_as_rows(np.zeros((4, 2, 8))).shape, (4, 16))

    def test_fit_hands_the_estimator_windows(self):
        est = _Recorder()
        windowed.fit_windows(est, _Loader(t.zeros((7, 3, 16))))
        self.assertEqual(est.seen, (7, 48))


class TestScoreWindows(unittest.TestCase):

    def test_one_score_per_window(self):
        est = _Recorder()
        scores = windowed.score_windows(est, t.zeros((6, 2, 8)))
        self.assertEqual(scores.shape, (6,))
        self.assertEqual(est.seen, (6, 16))

    def test_non_finite_scores_become_zero(self):
        """A single NaN would otherwise poison the whole de-unfolded series, and
        downstream a zero reads as "no evidence" rather than "no answer"."""
        est = _Recorder(scores=[1.0, np.nan, np.inf, -np.inf])
        scores = windowed.score_windows(est, t.zeros((4, 2, 8)))
        np.testing.assert_array_equal(scores, [1.0, 0.0, 0.0, 0.0])

    def test_scores_are_magnitudes(self):
        est = _Recorder(scores=[-2.0, 3.0])
        np.testing.assert_array_equal(
            windowed.score_windows(est, t.zeros((2, 2, 8))), [2.0, 3.0])

    def test_clip_is_opt_in(self):
        est = _Recorder(scores=[10.0, 0.5])
        np.testing.assert_array_equal(
            windowed.score_windows(est, t.zeros((2, 1, 4))), [10.0, 0.5])
        np.testing.assert_array_equal(
            windowed.score_windows(est, t.zeros((2, 1, 4)), clip=1.5), [1.5, 0.5])


class TestBroadcast(unittest.TestCase):

    def test_shape_matches_what_evaluate_model_expects(self):
        out = windowed.broadcast_to_window(np.array([1.0, 2.0]), 2, 3, 4)
        self.assertEqual(tuple(out.shape), (2, 3, 4))

    def test_every_reading_in_a_window_carries_that_window_s_score(self):
        """The detector saw the window as one object, so it has nothing finer to
        say about which reading inside it was anomalous."""
        out = windowed.broadcast_to_window(np.array([1.0, 2.0]), 2, 3, 4).numpy()
        self.assertEqual(set(np.unique(out[0])), {1.0})
        self.assertEqual(set(np.unique(out[1])), {2.0})

    def test_result_is_writable(self):
        """np.broadcast_to returns a read-only view; evaluate_model assigns into
        the tensor it gets back."""
        out = windowed.broadcast_to_window(np.array([1.0]), 1, 2, 2)
        out[0, 0, 0] = 9.0        # would raise on a read-only buffer
        self.assertEqual(float(out[0, 0, 0]), 9.0)


class TestDetectorsUseIt(unittest.TestCase):
    """The seven bespoke classes and the generic wrapper all route through the
    helper; a copy that drifted back to the old reshape is the failure mode."""

    def test_no_detector_flattens_to_a_single_column(self):
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        for name in ("lof", "cblof", "abod", "cof", "kde", "sos", "alad",
                     "pyod_model"):
            with open(os.path.join(here, f"{name}.py")) as f:
                src = f.read()
            self.assertNotIn("n_features * n_time, -1", src, name)
            self.assertIn("windowed.", src, name)


if __name__ == "__main__":
    unittest.main()
