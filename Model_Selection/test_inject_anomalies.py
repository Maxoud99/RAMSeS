"""The anomaly rate: does asking for a fraction produce that fraction?

`spikes` scatters points and the eight other types cut one contiguous segment,
so the rate reaches them by two different routes and both need covering.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model_Selection.inject_anomalies import Inject, InjectAnomalies  # noqa: E402
from Model_Selection.anomaly_parameters import ANOMALY_PARAM_GRID  # noqa: E402
from Utils.pipeline_spec import ALL_ANOMALY_TYPES  # noqa: E402


def series(n_features=5, n_time=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)
    Y = np.stack([np.sin(2 * np.pi * t / 50 + i) + 0.1 * rng.normal(size=n_time)
                  for i in range(n_features)])
    return (Y - Y.min(axis=1, keepdims=True)) / np.ptp(Y, axis=1, keepdims=True)


class _Entity:
    def __init__(self, Y):
        self.Y = Y
        self.n_features, self.n_time = Y.shape
        self.labels = np.zeros((1, self.n_time))
        self.mask = np.ones(Y.shape)


class _Dataset:
    def __init__(self, Y):
        self.entities = [_Entity(Y)]
        self.total_time = Y.shape[1]


def labelled_fraction(anomaly_type, rate, Y=None):
    Y = series() if Y is None else Y
    np.random.seed(7)
    data, _ = Inject(_Dataset(Y.copy()), [anomaly_type], rate=rate)
    labels = np.asarray(data.entities[0].labels).flatten()
    return float(labels.mean())


class TestAnomalyRate(unittest.TestCase):

    def test_segment_types_hit_the_requested_rate(self):
        """The six segment types whose length the series preserves."""
        for anomaly_type in ("contextual", "flip", "noise", "cutoff",
                             "scale", "wander", "average"):
            for rate in (0.05, 0.1, 0.25, 0.5):
                got = labelled_fraction(anomaly_type, rate)
                self.assertAlmostEqual(
                    got, rate, delta=0.01,
                    msg=f"{anomaly_type} at rate {rate} labelled {got:.3f}")

    def test_spikes_approaches_the_requested_rate_from_below(self):
        """Spikes are labelled only where |spike| > 0.05, so a few selected
        timesteps draw too small a value and stay unlabelled."""
        for rate in (0.05, 0.25, 0.5):
            got = labelled_fraction("spikes", rate)
            self.assertLessEqual(got, rate + 1e-9)
            self.assertGreater(got, 0.8 * rate)

    def test_speedup_resamples_so_its_rate_only_tracks_the_target(self):
        """Its segment is stretched or compressed after labelling, which moves
        the denominator; the rate stays monotone but is not the target."""
        rates = [labelled_fraction("speedup", r) for r in (0.05, 0.1, 0.25, 0.5)]
        self.assertEqual(rates, sorted(rates))
        self.assertGreater(rates[0], 0)
        self.assertLess(rates[-1], 1.0)

    def test_every_declared_type_accepts_a_rate(self):
        for anomaly_type in ALL_ANOMALY_TYPES:
            self.assertGreater(labelled_fraction(anomaly_type, 0.1), 0)

    def test_rate_none_leaves_the_grid_defaults_untouched(self):
        """Callers that omit the rate must inject exactly what they did before
        it existed, so existing result trees stay reproducible."""
        Y = series()
        for anomaly_type in ALL_ANOMALY_TYPES:
            np.random.seed(11)
            before, _ = Inject(_Dataset(Y.copy()), [anomaly_type])
            np.random.seed(11)
            after, _ = Inject(_Dataset(Y.copy()), [anomaly_type], rate=None)
            np.testing.assert_array_equal(before.entities[0].Y,
                                          after.entities[0].Y)
            np.testing.assert_array_equal(before.entities[0].labels,
                                          after.entities[0].labels)

    def test_a_full_rate_does_not_overrun_the_series(self):
        for anomaly_type in ("contextual", "scale"):
            self.assertLessEqual(labelled_fraction(anomaly_type, 1.0), 1.0)

    def test_the_segment_is_placed_so_the_rate_is_not_truncated(self):
        """A late-peaking series would otherwise clip the segment at the end
        and silently under-deliver the rate."""
        obj = InjectAnomalies(random_state=3, verbose=False,
                              max_window_size=128, min_window_size=8)
        Y = series()
        obj.anomalous_feature = 0
        obj.max_anomaly_length = 4
        obj.compute_anomaly_properties(Y, anomaly_rate=0.9)
        self.assertLessEqual(obj.anomaly_end, Y.shape[1])
        self.assertGreaterEqual(obj.anomaly_end - obj.anomaly_start,
                                int(0.9 * Y.shape[1]) - 2 * obj.min_window_size)


class TestVocabularyAgreement(unittest.TestCase):

    def test_spec_matches_the_injector_and_the_parameter_grid(self):
        """Three lists name the anomaly types; a type missing from any of them
        is offerable in the UI but unrunnable, or runnable but unoffered."""
        injector = InjectAnomalies(random_state=0).get_valid_anomaly_types()
        self.assertEqual(set(ALL_ANOMALY_TYPES), set(injector))
        self.assertEqual(set(ALL_ANOMALY_TYPES), set(ANOMALY_PARAM_GRID))


if __name__ == "__main__":
    unittest.main()
