"""Online windows and how the deployed selection is scored on them.

Detector scoring is stubbed (`evaluate_model_consistently`); everything after
it — the window construction, the meta-model fit, the GA fitness, the
single-model metric and the decision-metric combination — is the real code.
"""

import json
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Metrics.Ensemble_GA import fitness_function, score_meta_model  # noqa: E402
from Model_Selection import online_scoring as scoring  # noqa: E402
from Model_Selection.online_phase import OnlineWindow, Selection  # noqa: E402
from Model_Selection.online_reopt import CompletedReopt  # noqa: E402


def series(n=600, channels=2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.stack([np.sin(2 * np.pi * t / 40 + c) + 0.05 * rng.normal(size=n)
                     for c in range(channels)])


class FakeModel:
    """Scores = the first channel plus a per-model offset; deterministic."""

    def __init__(self, offset=0.0, fail=False):
        self.offset, self.fail = offset, fail


def fake_evaluate(data, model, name, is_ensemble=False):
    if is_ensemble:
        cols = {m: data.entities[0].Y[0] + model[m].offset for m in name}
        y = np.asarray(data.entities[0].labels).flatten()
        return y, np.column_stack(list(cols.values())), {}, cols
    if model.fail:
        raise RuntimeError(f"{name} cannot score")
    y = np.asarray(data.entities[0].labels).flatten()
    # A detector that tracks the labels, with noise, so metrics are non-trivial.
    rng = np.random.default_rng(int(model.offset * 1000) % 97)
    return y, y * (1.0 + model.offset) + 0.3 * rng.random(len(y)), {}, {}


class TestStream(unittest.TestCase):
    """The online part is injected ONCE and every window is cut from it."""

    def stream(self, seed=13, types=("spikes",), n=300, rate=0.05):
        Y = series(n)
        return scoring.build_online_stream(Y, np.zeros(n), np.ones_like(Y), list(types),
                                           rate, seed)

    def build(self, seed=13, types=("spikes",), indices=(0, 1, 2), step=20):
        return scoring.build_online_windows(self.stream(seed, types), list(indices),
                                            offset=1000, step_size=step, window_size=100)

    def test_same_seed_same_windows_and_anomalies(self):
        a, b = self.build(13), self.build(13)
        for wa, wb in zip(a, b):
            np.testing.assert_array_equal(wa.data.entities[0].Y, wb.data.entities[0].Y)
            np.testing.assert_array_equal(wa.data.entities[0].labels,
                                          wb.data.entities[0].labels)
        self.assertTrue(any(np.asarray(w.data.entities[0].labels).sum() > 0 for w in a))

    def test_a_different_seed_draws_different_anomalies(self):
        a, b = self.build(13), self.build(14)
        self.assertFalse(all(np.array_equal(np.asarray(x.data.entities[0].labels),
                                            np.asarray(y.data.entities[0].labels))
                             for x, y in zip(a, b)))

    def test_the_global_generator_is_left_as_it_was(self):
        np.random.seed(5)
        expected = np.random.random()
        np.random.seed(5)
        self.build(13)
        self.assertEqual(np.random.random(), expected)

    def test_a_sample_has_one_value_and_label_in_every_window(self):
        """Windows overlap by 80 samples here: the shared samples must agree.
        Injecting per window gave each of them a different value and label."""
        w = self.build(13, step=20)
        for left, right in zip(w, w[1:]):
            Y_l, Y_r = left.data.entities[0].Y, right.data.entities[0].Y
            l_l = np.asarray(left.data.entities[0].labels).reshape(-1)
            l_r = np.asarray(right.data.entities[0].labels).reshape(-1)
            np.testing.assert_array_equal(Y_l[:, 20:], Y_r[:, :80])
            np.testing.assert_array_equal(l_l[20:], l_r[:80])

    def test_each_window_carries_a_fingerprint_of_what_is_scored(self):
        a, b, c = self.build(13), self.build(13), self.build(14)
        self.assertEqual([w.digest for w in a], [w.digest for w in b])
        self.assertNotEqual([w.digest for w in a], [w.digest for w in c])
        self.assertEqual(len({w.digest for w in a}), 3)
        for w in a:
            labels = np.asarray(w.data.entities[0].labels).reshape(-1)
            self.assertEqual((w.n_samples, w.n_anomalous), (100, int(labels.sum())))
            self.assertEqual(w.digest, scoring.window_digest(w.data.entities[0].Y, labels))

    def test_coordinates_and_window_count(self):
        w = self.build(13)
        self.assertEqual([(x.index, x.start, x.end, x.online_start, x.online_end)
                          for x in w],
                         [(0, 1000, 1100, 0, 100), (1, 1020, 1120, 20, 120),
                          (2, 1040, 1140, 40, 140)])
        from Utils.pipeline_spec import count_windows
        stream = self.stream()
        n = count_windows(stream.length, 100, 20)
        self.assertEqual(n, 11)
        scoring.build_online_windows(stream, range(n), 0, 20, 100)     # all fit
        with self.assertRaises(ValueError):
            scoring.build_online_windows(stream, [n], 0, 20, 100)       # one past

    def test_multi_copy_types_take_the_same_time_slice_of_every_copy(self):
        stream = self.stream(types=("scale",))
        self.assertEqual(len(stream.copies), 4)
        w = scoring.build_online_windows(stream, [2], 0, 20, 100)[0]
        Y = w.data.entities[0].Y
        labels = np.asarray(w.data.entities[0].labels).reshape(-1)
        self.assertEqual(Y.shape[1], 400)
        self.assertEqual([(s["data_start"], s["data_end"], s["copy_start"], s["copy_end"])
                          for s in w.segments],
                         [(0, 100, 40, 140), (100, 200, 40, 140),
                          (200, 300, 40, 140), (300, 400, 40, 140)])
        for c, copy_ in enumerate(stream.copies):
            np.testing.assert_array_equal(Y[:, 100 * c:100 * (c + 1)], copy_.Y[:, 40:140])
            np.testing.assert_array_equal(labels[100 * c:100 * (c + 1)],
                                          np.asarray(copy_.labels)[40:140])
        self.assertEqual([p["amplitude_scaling"] for p in stream.copy_params()],
                         [0.25, 0.5, 2, 4])

    def test_speedup_copies_are_cut_at_the_same_original_times(self):
        """speedup resamples its segment, so its copies differ in length; a
        window still covers the same original time slice in each."""
        stream = self.stream(types=("speedup",), rate=None)
        lengths = [c.length for c in stream.copies]
        self.assertEqual(len(set(lengths)), 4)                  # all different
        w = scoring.build_online_windows(stream, [0, 10], 0, 20, 100)
        for window in w:
            total = window.data.entities[0].Y.shape[1]
            self.assertEqual(window.segments[-1]["data_end"], total)
            for seg, copy_ in zip(window.segments, stream.copies):
                self.assertEqual((seg["copy_start"], seg["copy_end"]),
                                 (copy_.position(window.online_start),
                                  copy_.position(window.online_end)))
        # The last window ends at each copy's own end.
        self.assertEqual([s["copy_end"] for s in w[1].segments], lengths)


class TestSlidingSeries(unittest.TestCase):
    """What a re-optimisation selects on: offline data slid forward by the
    online samples seen so far, each sample once."""

    def setUp(self):
        from Model_Selection.inject_anomalies import inject_copies
        rng = np.random.default_rng(0)
        self.off_Y = rng.normal(size=(2, 400))
        self.off_labels = (rng.random(400) > 0.9).astype(int)
        np.random.seed(3)
        self.off_copies = inject_copies(self.off_Y.copy(), ["spikes"], rate=0.05)
        on_Y = rng.normal(size=(2, 150)) + 10.0        # distinguishable from offline
        self.stream = scoring.build_online_stream(on_Y, np.zeros(150), np.ones_like(on_Y),
                                                  ["spikes"], 0.05, 13)

    def slide(self, end):
        return scoring.sliding_series(
            self.off_copies, (self.off_Y, self.off_labels, np.ones_like(self.off_Y)),
            self.stream, end)

    def test_offline_length_kept_and_each_online_sample_once(self):
        injected, raw = self.slide(120)
        Y = injected.entities[0].Y
        self.assertEqual(Y.shape, (2, 400))
        # The last 120 samples are the injected stream's first 120, once each.
        np.testing.assert_array_equal(Y[:, -120:], self.stream.copies[0].Y[:, :120])
        np.testing.assert_array_equal(Y[:, :280], self.off_copies[0].Y[:, 120:])
        labels = np.asarray(injected.entities[0].labels).reshape(-1)
        np.testing.assert_array_equal(labels[-120:],
                                      np.asarray(self.stream.copies[0].labels)[:120])

    def test_the_raw_series_is_the_same_window_before_injection(self):
        _, raw = self.slide(120)
        Y = raw.entities[0].Y
        np.testing.assert_array_equal(Y[:, -120:], self.stream.raw_Y[:, :120])
        np.testing.assert_array_equal(Y[:, :280], self.off_Y[:, 120:])
        np.testing.assert_array_equal(np.asarray(raw.entities[0].labels).reshape(-1)[:280],
                                      self.off_labels[120:])

    def test_copies_must_match(self):
        with self.assertRaises(ValueError):
            scoring.sliding_series(self.off_copies * 2, (self.off_Y, self.off_labels, None),
                                   self.stream, 100)

    def test_multi_copy_series_slide_each_copy_on_its_own(self):
        from Model_Selection.inject_anomalies import inject_copies
        np.random.seed(4)
        off = inject_copies(self.off_Y.copy(), ["cutoff"])
        stream = scoring.build_online_stream(self.stream.raw_Y, np.zeros(150),
                                             None, ["cutoff"], None, 13)
        injected, _ = scoring.sliding_series(
            off, (self.off_Y, self.off_labels, np.ones_like(self.off_Y)), stream, 100)
        Y = injected.entities[0].Y
        self.assertEqual(Y.shape[1], 800)
        for c in range(2):
            np.testing.assert_array_equal(Y[:, 400 * c + 300:400 * (c + 1)],
                                          stream.copies[c].Y[:, :100])


class TestRealLabels(unittest.TestCase):
    """--online_labels real: nothing injected online, windows scored on the
    true labels, re-optimisations fed the true labels seen so far."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.off_Y = rng.normal(size=(1, 400))
        self.off_labels = np.zeros(400, dtype=int)
        self.off_labels[100:110] = 1
        self.on_Y = rng.normal(size=(1, 300)) + 5.0
        self.on_labels = np.zeros(300, dtype=int)
        self.on_labels[150:160] = 1                     # one real anomaly online
        self.stream = scoring.real_label_stream(self.on_Y, self.on_labels,
                                                np.ones_like(self.on_Y))
        self.copies = [scoring.identity_copy(self.off_Y, self.off_labels)]

    def windows(self, indices=range(21), step=10, size=100):
        return scoring.build_online_windows(self.stream, list(indices), offset=400,
                                            step_size=step, window_size=size)

    def test_a_window_is_the_raw_test_slice_and_its_true_labels(self):
        for w in self.windows():
            lo, hi = w.online_start, w.online_end
            np.testing.assert_array_equal(w.data.entities[0].Y, self.on_Y[:, lo:hi])
            np.testing.assert_array_equal(np.asarray(w.data.entities[0].labels).reshape(-1),
                                          self.on_labels[lo:hi])
            self.assertEqual((w.start, w.end), (400 + lo, 400 + hi))
            self.assertEqual(w.n_anomalous, int(self.on_labels[lo:hi].sum()))
            self.assertEqual(w.segments, [{"copy": 0, "data_start": 0, "data_end": 100,
                                           "copy_start": lo, "copy_end": hi}])

    def test_nothing_is_drawn_and_the_stream_is_not_modified(self):
        np.random.seed(11)
        expected = np.random.random()
        np.random.seed(11)
        before = self.on_Y.copy()
        self.windows()
        self.assertEqual(np.random.random(), expected)
        np.testing.assert_array_equal(self.on_Y, before)
        self.assertEqual(self.stream.copy_params(),
                         [{"copy": 0, "anomaly_type": None, "length": 300}])

    def test_only_windows_with_both_classes_are_scorable(self):
        flags = {w.index: w.scorable for w in self.windows()}
        # Window i covers [10i, 10i + 100); it holds 150..159 for i = 6..15.
        self.assertEqual({i for i, f in flags.items() if f}, set(range(6, 16)))
        full = scoring.real_label_stream(self.on_Y, np.ones(300), None)
        w = scoring.build_online_windows(full, [0], offset=0, step_size=10,
                                         window_size=100)[0]
        self.assertFalse(w.scorable)             # anomalous throughout: one class

    def test_unscorable_windows_have_no_score(self):
        models = {"A": FakeModel(0.0), "B": FakeModel(0.3)}
        rng = np.random.default_rng(1)
        y_train = (rng.random(300) > 0.85).astype(int)
        X_train = np.column_stack([y_train + 0.3 * rng.random(300),
                                   1.2 * y_train + 0.3 * rng.random(300)])
        selection = scoring.fit_selection(["A", "B"], "A", "lr", {"A": 0, "B": 0},
                                          X_train, y_train)
        with mock.patch.object(scoring, "evaluate_model_consistently", fake_evaluate):
            scorer = scoring.WindowScorer(models, ("f1", "pr_auc"))
            for w in self.windows(indices=(0, 8)):
                out = scorer(w, selection)
                if w.scorable:
                    self.assertIsNotNone(out.single)
                    self.assertIsNotNone(out.ensemble)
                else:
                    self.assertIsNone(out.single)
                    self.assertIsNone(out.ensemble)
                    self.assertIn("one class", out.errors["single"])

    def test_reopt_data_is_the_true_labels_seen_so_far_each_sample_once(self):
        seen_end = 170
        injected, raw = scoring.sliding_series(
            self.copies, (self.off_Y, self.off_labels, np.ones_like(self.off_Y)),
            self.stream, seen_end)
        Y = injected.entities[0].Y
        labels = np.asarray(injected.entities[0].labels).reshape(-1)
        # The offline length, the offline part slid forward by the 170 online
        # samples seen, each sample exactly once and in time order.
        expected_Y = np.concatenate([self.off_Y, self.on_Y[:, :seen_end]], axis=1)[:, -400:]
        expected_labels = np.concatenate([self.off_labels, self.on_labels[:seen_end]])[-400:]
        np.testing.assert_array_equal(Y, expected_Y)
        np.testing.assert_array_equal(labels, expected_labels)
        positions = np.arange(400 + 300)[seen_end:400 + seen_end]
        self.assertEqual(len(set(positions)), 400)
        # Nothing injected: the "injected" series IS the raw one.
        np.testing.assert_array_equal(raw.entities[0].Y, Y)
        np.testing.assert_array_equal(np.asarray(raw.entities[0].labels).reshape(-1), labels)
        # The offline anomaly at 100..109 has slid out; the online one is in.
        self.assertEqual(int(labels.sum()), 10)
        self.assertEqual(int(labels[-20:-10].sum()), 10)

    def test_a_reopt_without_a_labelled_anomaly_is_refused(self):
        quiet = [scoring.identity_copy(self.off_Y, np.zeros(400))]
        stream = scoring.real_label_stream(self.on_Y, np.zeros(300), None)
        injected, _ = scoring.sliding_series(
            quiet, (self.off_Y, np.zeros(400), np.ones_like(self.off_Y)), stream, 100)
        with self.assertRaisesRegex(ValueError, "no labelled anomaly in the seen data"):
            scoring.require_labelled_anomaly(injected, what="the seen data")
        ok, _ = scoring.sliding_series(
            self.copies, (self.off_Y, self.off_labels, np.ones_like(self.off_Y)),
            self.stream, 100)
        scoring.require_labelled_anomaly(ok)            # does not raise

    def test_identity_copy_checks_its_length(self):
        with self.assertRaises(ValueError):
            scoring.identity_copy(self.off_Y, np.zeros(10))


class TestSelections(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = (rng.random(300) > 0.8).astype(int)
        # One distinguishable column per pool member, in pool order.
        self.pool = ["A", "B", "C"]
        self.X = np.column_stack([self.y + rng.normal(0, s, 300) for s in (0.1, 0.5, 2.0)])
        self.individual = {m: None for m in self.pool}

    def test_members_are_taken_by_name_in_pool_order(self):
        sel = scoring.fit_selection(["C", "A"], "B", "lr", self.individual, self.X, self.y)
        self.assertEqual(sel.ensemble, ["A", "C"])
        self.assertEqual(sel.meta_model_source, "refit")
        self.assertEqual(sel.meta_model.coef_.shape[1], 2)
        # A fitted on columns A and C predicts like a fresh fit on exactly those.
        from sklearn.linear_model import LogisticRegression
        ref = LogisticRegression().fit(self.X[:, [0, 2]].astype(np.float32), self.y)
        np.testing.assert_allclose(sel.meta_model.coef_, ref.coef_, rtol=1e-5)

    def test_misaligned_training_predictions_are_rescored_not_guessed(self):
        models = {"A": FakeModel(0.0), "C": FakeModel(0.5)}

        class Train:
            entities = [type("E", (), {"Y": self.X.T[:1], "labels": self.y})()]

        with mock.patch.object(scoring, "evaluate_model_consistently", fake_evaluate):
            sel = scoring.fit_selection(["A", "C"], "A", "lr", self.individual,
                                        self.X[:, :2], self.y,      # one column short
                                        train_data=Train(), trained_models=models)
        self.assertEqual(sel.meta_model.coef_.shape[1], 2)
        with self.assertRaises(ValueError):
            scoring.fit_selection(["A", "C"], "A", "lr", self.individual, self.X[:, :2], self.y)

    def test_the_ga_meta_model_is_deployed_as_is(self):
        from sklearn.linear_model import LogisticRegression
        pool = ["C", "A", "B"]              # the GA's pool order, not sorted
        ga_model = LogisticRegression().fit(self.X[:, [2, 0]], self.y)
        sel = scoring.fit_selection(["A", "C"], "B", "lr", {m: None for m in pool},
                                    self.X, self.y, meta_model=ga_model)
        self.assertIs(sel.meta_model, ga_model)
        self.assertEqual(sel.meta_model_source, "ga")
        # Its columns are the members in pool order.
        self.assertEqual(sel.ensemble, ["C", "A"])

    def test_a_meta_model_of_the_wrong_width_is_refitted(self):
        from sklearn.linear_model import LogisticRegression
        three = LogisticRegression().fit(self.X, self.y)
        sel = scoring.fit_selection(["A", "C"], "B", "lr", self.individual, self.X, self.y,
                                    meta_model=three)
        self.assertIsNot(sel.meta_model, three)
        self.assertEqual(sel.meta_model_source, "refit")

    def test_a_refit_is_reproducible_from_its_seed(self):
        a = scoring.fit_selection(["A", "C"], "B", "rf", self.individual, self.X, self.y,
                                  seed=13)
        b = scoring.fit_selection(["A", "C"], "B", "rf", self.individual, self.X, self.y,
                                  seed=13)
        X = self.X[:, [0, 2]].astype(np.float32)
        np.testing.assert_array_equal(a.meta_model.predict_proba(X),
                                      b.meta_model.predict_proba(X))

    def test_no_ensemble_is_a_selection_without_a_meta_model(self):
        sel = scoring.fit_selection(None, "A", None, self.individual, self.X, self.y)
        self.assertEqual((sel.ensemble, sel.meta_model, sel.meta_model_type), ([], None, "rf"))

    def test_random_baseline_is_reproducible_and_sized_to_the_pool(self):
        pool = [f"M{i}" for i in range(8)]
        individual = {m: None for m in pool}
        X = np.column_stack([self.y + i for i in range(8)])
        a = scoring.random_selection(pool, 13, "lr", individual, X, self.y)
        b = scoring.random_selection(pool, 13, "lr", individual, X, self.y)
        self.assertEqual((a.ensemble, a.single), (b.ensemble, b.single))
        self.assertTrue(3 <= len(a.ensemble) <= 6)
        self.assertEqual(a.origin, "random")
        small = scoring.random_selection(self.pool[:2], 1, "lr", {"A": 0, "B": 0},
                                         self.X[:, :2], self.y)
        self.assertEqual(len(small.ensemble), 2)


class TestWindowScorer(unittest.TestCase):

    def setUp(self):
        # Short: the range-based F1 sweeps 1000 thresholds over every sample.
        Y = series(150, channels=1)
        labels = np.zeros(150)
        labels[50:58] = 1
        labels[110:114] = 1
        self.window = OnlineWindow(index=1, start=0, end=150,
                                   data=scoring.window_dataset(Y, labels, None))
        self.flat = OnlineWindow(index=2, start=0, end=150,
                                 data=scoring.window_dataset(Y, np.zeros(150), None))
        self.models = {"A": FakeModel(0.0), "B": FakeModel(0.4), "BAD": FakeModel(fail=True)}
        rng = np.random.default_rng(1)
        y_train = (rng.random(400) > 0.85).astype(int)
        X_train = np.column_stack([y_train + 0.3 * rng.random(400),
                                   1.4 * y_train + 0.3 * rng.random(400)])
        self.selection = scoring.fit_selection(["A", "B"], "A", "lr", {"A": 0, "B": 0},
                                               X_train, y_train)
        patcher = mock.patch.object(scoring, "evaluate_model_consistently", fake_evaluate)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_both_signals_on_the_decision_metric(self):
        scorer = scoring.WindowScorer(self.models, ("f1", "pr_auc"))
        out = scorer(self.window, self.selection)
        self.assertEqual(out.errors, {})
        m = out.metrics["single"]
        self.assertAlmostEqual(out.single, 0.5 * m["f1"] + 0.5 * m["pr_auc"])
        e = out.metrics["ensemble"]
        self.assertAlmostEqual(out.ensemble, 0.5 * e["f1"] + 0.5 * e["pr_auc"])
        self.assertTrue(0 <= out.single <= 1 and 0 <= out.ensemble <= 1)
        self.assertIsInstance(out.misclassified["single"], int)
        self.assertIsInstance(out.misclassified["ensemble"], int)

    def test_the_ensemble_signal_is_the_ga_fitness_of_the_deployed_meta_model(self):
        scorer = scoring.WindowScorer(self.models, ("f1",))
        out = scorer(self.window, self.selection)
        y = np.asarray(self.window.data.entities[0].labels).flatten().astype(int)
        cols = np.column_stack([fake_evaluate(self.window.data, self.models[m], m)[1]
                                for m in ("A", "B")])
        expected = score_meta_model(self.selection.meta_model, cols, y, metric=("f1",))
        self.assertAlmostEqual(out.ensemble, expected["fitness"])

    def test_vus_is_computed_only_when_the_metric_needs_it(self):
        import Metrics.Ensemble_GA as ga
        with mock.patch.object(scoring, "vus_score", return_value=0.5) as single_vus, \
                mock.patch.object(ga, "vus_score", return_value=0.25) as ensemble_vus:
            scoring.WindowScorer(self.models, ("f1", "pr_auc"), vus_win=37)(
                self.window, self.selection)
            self.assertEqual((single_vus.call_count, ensemble_vus.call_count), (0, 0))
            out = scoring.WindowScorer(self.models, ("f1", "vus"), vus_win=37)(
                self.window, self.selection)
            self.assertEqual((single_vus.call_count, ensemble_vus.call_count), (1, 1))
            self.assertEqual(out.metrics["single"]["vus"], 0.5)
            self.assertEqual(out.metrics["ensemble"]["vus"], 0.25)

    def test_one_vus_window_for_every_window(self):
        """The window length is the run's, not re-estimated from each window's
        (injected) data, which jumped between ~35 and 125 on overlapping windows."""
        import Metrics.Ensemble_GA as ga
        scorer = scoring.WindowScorer(self.models, ("f1", "vus"), vus_win=37)
        with mock.patch.object(scoring, "vus_score", return_value=0.5) as single_vus, \
                mock.patch.object(ga, "vus_score", return_value=0.25) as ensemble_vus:
            scorer(self.window, self.selection)
            scorer(self.flat, self.selection)
        self.assertEqual({c.args[2] for c in single_vus.call_args_list}, {37})
        self.assertEqual({c.args[2] for c in ensemble_vus.call_args_list}, {37})

    def test_a_one_class_window_has_no_score(self):
        out = scoring.WindowScorer(self.models, ("f1",))(self.flat, self.selection)
        self.assertIsNone(out.single)
        self.assertIsNone(out.ensemble)
        self.assertIn("one class", out.errors["single"])
        self.assertIn("one class", out.errors["ensemble"])

    def test_a_failing_member_voids_only_the_ensemble(self):
        sel = Selection(["A", "BAD"], "A", "lr", self.selection.meta_model)
        out = scoring.WindowScorer(self.models, ("f1",))(self.window, sel)
        self.assertIsNotNone(out.single)
        self.assertIsNone(out.ensemble)
        self.assertIn("cannot score", out.errors["ensemble"])

    def test_the_window_is_not_modified_by_scoring(self):
        before = self.window.data.entities[0].Y.copy()
        scoring.WindowScorer(self.models, ("f1",))(self.window, self.selection)
        np.testing.assert_array_equal(self.window.data.entities[0].Y, before)



def stochastic_evaluate(data, model, name, is_ensemble=False):
    """Like `fake_evaluate`, plus noise from numpy's AND torch's global
    generators, the way LSTMVAE and DGHL draw whenever they score."""
    import torch
    assert not is_ensemble
    if getattr(model, "fail_first", False) and not model.failed_once:
        model.failed_once = True
        raise RuntimeError(f"{name} failed once")
    y = np.asarray(data.entities[0].labels).flatten()
    # Noise as large as the signal, so it moves the F1 and PR-AUC too.
    noise = np.random.random(len(y)) + torch.rand(len(y)).numpy()
    return y, y * (1.0 + model.offset) + noise, {}, {}


class TestSameWindowSameScore(unittest.TestCase):
    """A (window, detector) pair is scored once per run, under a seed, and
    every scenario reads that score."""

    def setUp(self):
        # Short windows, two scenarios: the range-based F1 sweeps 1000
        # thresholds over every sample, once per signal per window.
        Y = series(90, channels=1)
        self.windows = []
        for i in (0, 1, 2):
            labels = np.zeros(90)
            labels[20 * i + 10:20 * i + 18] = 1
            self.windows.append(OnlineWindow(index=i, start=100 * i, end=100 * i + 90,
                                             data=scoring.window_dataset(Y, labels, None)))
        self.models = {"A": FakeModel(0.0), "B": FakeModel(0.4)}
        rng = np.random.default_rng(1)
        y_train = (rng.random(400) > 0.85).astype(int)
        X_train = np.column_stack([y_train + 0.3 * rng.random(400),
                                   1.4 * y_train + 0.3 * rng.random(400)])
        self.selection = scoring.fit_selection(["A", "B"], "A", "lr", {"A": 0, "B": 0},
                                               X_train, y_train)
        patcher = mock.patch.object(scoring, "evaluate_model_consistently",
                                    stochastic_evaluate)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_fake_detector_really_is_stochastic(self):
        a = stochastic_evaluate(self.windows[0].data, self.models["A"], "A")[1]
        b = stochastic_evaluate(self.windows[0].data, self.models["A"], "A")[1]
        self.assertFalse(np.allclose(a, b))

    def test_every_scenario_reads_the_same_window_scores(self):
        import copy
        import tempfile
        from Model_Selection.online_phase import OnlineResultsWriter, run_online
        from Model_Selection.online_policy import PolicyParams

        scorer = scoring.WindowScorer(self.models, ("f1", "pr_auc"), seed=13)
        keys = ("fixed", "periodic")
        with tempfile.TemporaryDirectory() as tmp:
            writer = OnlineResultsWriter(tmp, {"strategy": "all"}, keys)

            def start(window, selection, key):
                # A re-optimisation that picks the same detectors again: the
                # periodic scenario then deploys exactly what `fixed` does.
                return CompletedReopt(copy.deepcopy(selection), 1.0, reused_from=None)

            run_online(keys, PolicyParams(update_interval=1, reopt_delay=1), self.windows,
                       lambda key: copy.deepcopy(self.selection), scorer, start, writer)
            docs = {k: json.load(open(os.path.join(tmp, f"{k}.json"))) for k in keys}
        series_of = {k: [(w["i"], w["ensemble_score"], w["single_score"])
                         for w in docs[k]["windows"]] for k in keys}
        self.assertEqual(series_of["fixed"], series_of["periodic"])
        self.assertTrue(all(e is not None and s is not None
                            for _, e, s in series_of["fixed"]))
        # The periodic scenario did switch selections; the scores held anyway.
        self.assertTrue(any(w["reopt_applied"] for w in docs["periodic"]["windows"]))
        # Two detectors (the single model is also a member) on three windows.
        self.assertEqual(scorer.computed, 6)
        self.assertGreater(scorer.reused, 0)

    def test_scores_are_reproducible_from_the_seed(self):
        w = self.windows[0]
        a = scoring.WindowScorer(self.models, ("f1",), seed=13)._detector_scores(w, "A")[1]
        b = scoring.WindowScorer(self.models, ("f1",), seed=13)._detector_scores(w, "A")[1]
        c = scoring.WindowScorer(self.models, ("f1",), seed=14)._detector_scores(w, "A")[1]
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.allclose(a, c))

    def test_scoring_leaves_the_global_generators_as_they_were(self):
        import random as pyrandom
        import torch
        def draw():
            return pyrandom.random(), float(np.random.random()), float(torch.rand(1))
        pyrandom.seed(5); np.random.seed(5); torch.manual_seed(5)
        expected = draw()
        pyrandom.seed(5); np.random.seed(5); torch.manual_seed(5)
        scoring.WindowScorer(self.models, ("f1",), seed=13)(self.windows[0], self.selection)
        self.assertEqual(draw(), expected)

    def test_a_failure_is_remembered_not_retried(self):
        flaky = FakeModel(0.0)
        flaky.fail_first, flaky.failed_once = True, False
        scorer = scoring.WindowScorer({"F": flaky}, ("f1",), seed=13)
        sel = Selection([], "F", "lr", None)
        first = scorer(self.windows[0], sel)
        second = scorer(self.windows[0], sel)
        self.assertIsNone(first.single)
        self.assertIsNone(second.single)
        self.assertEqual(first.errors["single"], second.errors["single"])
        self.assertEqual((scorer.computed, scorer.reused), (1, 1))

    def test_callers_get_copies_of_the_remembered_scores(self):
        scorer = scoring.WindowScorer(self.models, ("f1",), seed=13)
        y, s = scorer._detector_scores(self.windows[0], "A")
        kept = s.copy()
        s[:] = -1
        y[:] = 7
        y2, s2 = scorer._detector_scores(self.windows[0], "A")
        np.testing.assert_array_equal(s2, kept)
        self.assertNotIn(7, y2)


class TestIsolatedModels(unittest.TestCase):

    def test_copies_share_nothing_with_the_originals(self):
        original = FakeModel(0.3)
        original.state = [1, 2]
        copies = scoring.isolated_models({"A": original})
        self.assertIsNot(copies["A"], original)
        copies["A"].window_step = 25
        copies["A"].state.append(3)
        self.assertFalse(hasattr(original, "window_step"))
        self.assertEqual(original.state, [1, 2])

    def test_an_uncopyable_model_gets_a_shallow_copy_and_a_warning(self):
        import threading
        original = FakeModel(0.3)
        original.lock = threading.Lock()          # deepcopy refuses locks
        with self.assertLogs("Model_Selection.online_scoring", level="WARNING") as logs:
            copies = scoring.isolated_models({"L": original})
        self.assertIn("shallow copy", "\n".join(logs.output))
        self.assertIsNot(copies["L"], original)
        copies["L"].window_step = 25
        self.assertFalse(hasattr(original, "window_step"))


class TestStreamAndReoptimisationThreads(unittest.TestCase):
    """The real RunningMean, scored at the same time by the stream (100-sample
    windows, through WindowScorer on its own copy) and by a re-optimisation
    thread (the 944-sample sliding series, on the original). RM ships
    window_step = -1, which `_evaluate_model` resolves per call; it used to
    write the resolved step onto the shared object."""

    def test_neither_thread_changes_the_other_s_scores_or_the_model(self):
        import threading
        import time
        from Algorithms.running_mean import RunningMean
        from Utils.model_selection_utils import evaluate_model

        rng = np.random.RandomState(0)

        def dataset(n):
            labels = np.zeros(n)
            labels[n // 2:n // 2 + 5] = 1
            return scoring.window_dataset(rng.randn(1, n).astype(np.float32), labels,
                                          np.ones((1, n)))

        originals = {"RM_1": RunningMean(n_features=1, running_window_size=4)}
        stream = scoring.isolated_models(originals)
        scorer = scoring.WindowScorer(stream, ("f1",), seed=13)
        window = OnlineWindow(index=1, start=0, end=100, data=dataset(100))
        big = dataset(944)
        ref_small = scorer._detector_scores(window, "RM_1")[1]
        ref_big = evaluate_model(big, originals["RM_1"], "RM_1")["entity_scores"].copy()

        stop, problems = threading.Event(), {"bg": [], "main": []}

        def reoptimisation():
            while not stop.is_set():
                try:
                    out = evaluate_model(big, originals["RM_1"], "RM_1")["entity_scores"]
                    if out.shape != ref_big.shape or not np.allclose(out, ref_big):
                        problems["bg"].append("wrong scores")
                except Exception as exc:
                    problems["bg"].append(repr(exc))

        worker = threading.Thread(target=reoptimisation)
        worker.start()
        try:
            deadline = time.time() + 1.5
            while time.time() < deadline:
                scorer._memo.clear()                 # force a real scoring each time
                try:
                    out = scorer._detector_scores(window, "RM_1")[1]
                    if out.shape != ref_small.shape or not np.allclose(out, ref_small):
                        problems["main"].append("wrong scores")
                except Exception as exc:
                    problems["main"].append(repr(exc))
        finally:
            stop.set()
            worker.join()
        self.assertEqual(problems, {"bg": [], "main": []})
        self.assertEqual(originals["RM_1"].window_step, -1)
        self.assertEqual(stream["RM_1"].window_step, -1)
        self.assertIsNot(stream["RM_1"], originals["RM_1"])


class TestScoreMetaModelMatchesFitness(unittest.TestCase):

    def test_fitness_function_is_score_meta_model_on_its_own_meta_model(self):
        """The refactor that lets the online phase reuse the GA's fitness must
        not have changed the fitness."""
        rng = np.random.default_rng(3)
        y_tr = (rng.random(300) > 0.8).astype(int)
        y_te = (rng.random(200) > 0.8).astype(int)
        X_tr = np.column_stack([y_tr + rng.normal(0, s, 300) for s in (0.3, 0.6, 1.0)])
        X_te = np.column_stack([y_te + rng.normal(0, s, 200) for s in (0.3, 0.6, 1.0)])
        f1, pr_auc, fitness, y_scores, y_true, meta = fitness_function(
            ["A", "C"], None, None, None, None, X_tr, ["A", "B", "C"], X_te, y_tr, y_te,
            meta_model_type="lr", metric=("f1", "pr_auc"))
        again = score_meta_model(meta, X_te[:, [0, 2]], y_true, metric=("f1", "pr_auc"))
        self.assertAlmostEqual(again["f1"], f1)
        self.assertAlmostEqual(again["pr_auc"], pr_auc)
        self.assertAlmostEqual(again["fitness"], fitness)
        np.testing.assert_allclose(again["y_scores"], y_scores)


if __name__ == "__main__":
    unittest.main()
