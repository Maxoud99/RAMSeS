"""A checkpoint that cannot score the data is excluded at load, not in the GA.

The case that prompted this, reproduced on SKAB/1 with the Jan-2026
checkpoints: LOF_1/LOF_2/CBLOF_1/CBLOF_2 unpickle cleanly, then the GA dies
with "X has 576 features, but LocalOutlierFactor is expecting 1 features".
Those checkpoints carry window_size=64 from the old grid, and their estimator
was fitted the pre-`Algorithms/windowed.py` way — every reading reshaped to
its own one-feature sample — so `detector_.n_features_in_ == 1`. Today's
scoring hands the estimator one row per window, 9 channels x 64 steps = 576
features. (The "'LocalOutlierFactor' object has no attribute '_score_samples'"
printed above it is not the cause: it is PyOD's own fallback chain, caught
inside `LOF.decision_function`, and scikit-learn 1.7 raises it for every LOF,
fresh or stale.)

`_stale_lof` below builds exactly that object, so these tests exercise the
real failure rather than a stand-in for it.
"""

import io
import os
import random
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import torch as t
    from Utils import model_selection_utils as msu
    from Algorithms.lof import TsadLof
    from Datasets.dataset import Dataset, Entity
    from Loaders.loader import Loader
except Exception:                                          # pragma: no cover
    msu = None

N_CHANNELS = 9


def _data(n_time, seed=0):
    rng = np.random.RandomState(seed)
    Y = rng.normal(size=(N_CHANNELS, n_time))
    entity = Entity(Y=Y, name="1", labels=np.zeros((1, n_time)), verbose=False)
    return Dataset(entities=[entity], name="skab", verbose=False)


def _loader(data, window):
    return Loader(dataset=data, batch_size=32, window_size=window,
                  window_step=window, shuffle=False, padding_type='right',
                  sample_with_replace=False, verbose=False,
                  mask_position='None', n_masked_timesteps=0)


def _stale_lof():
    """A Jan-2026 LOF checkpoint: window 64, estimator fitted on 1-feature rows."""
    model = TsadLof(window_size=64, window_step=64, n_neighbors=5)
    windows = _loader(_data(400), 64).Y_windows
    n_batches, n_features, n_time = windows.shape
    rows = windows.reshape(n_batches * n_features * n_time, -1).reshape(-1, 1)
    model.model.fit(X=rows.cpu().numpy())                 # the original TsadLof.fit
    return model


def _current_lof():
    model = TsadLof(window_size=1, window_step=1, n_neighbors=5)
    model.fit(_loader(_data(400), 1))
    return model


@unittest.skipIf(msu is None, "torch/numpy/pyod not installed in this interpreter")
class TestProbeDetector(unittest.TestCase):

    def setUp(self):
        msu.reset_timed_out()

    def tearDown(self):
        msu.reset_timed_out()

    def test_the_stale_checkpoint_really_fails_the_way_the_ga_did(self):
        """Guards the fixture: without the probe this is the GA's traceback."""
        with self.assertRaisesRegex(ValueError, "576 features.*expecting 1"):
            msu.evaluate_model(_data(229), _stale_lof(), "LOF_1")

    def test_a_stale_checkpoint_is_reported_not_raised(self):
        reason = msu.probe_detector(_data(229), _stale_lof(), "LOF_1")
        self.assertIsNotNone(reason)
        self.assertIn("ValueError", reason)
        self.assertIn("576 features", reason)
        self.assertIn("expecting 1", reason)
        self.assertNotIn("\n", reason, "the reason must fit on one log line")

    def test_a_current_checkpoint_passes(self):
        self.assertIsNone(msu.probe_detector(_data(229), _current_lof(), "LOF_1"))
        self.assertIsNone(msu.probe_detector(_data(3000), _current_lof(), "LOF_1"))

    def test_the_probe_scores_a_short_head_of_a_long_series(self):
        """Normal runs must not pay for a second full scoring pass."""
        seen = []
        real = msu.evaluate_model

        def recording(data, model, model_name, *a, **k):
            seen.append(data.entities[0].Y.shape[1])
            return real(data, model, model_name, *a, **k)

        with mock.patch.object(msu, "evaluate_model", recording):
            self.assertIsNone(msu.probe_detector(_data(5000), _current_lof(), "LOF_1"))
        self.assertEqual(seen, [msu.PROBE_TIMESTEPS])

    def test_the_head_covers_two_windows_of_a_wide_detector(self):
        seen = []
        model = TsadLof(window_size=200, window_step=200)

        def recording(data, *a, **k):
            seen.append(data.entities[0].Y.shape[1])
            return {}

        with mock.patch.object(msu, "evaluate_model", recording):
            msu.probe_detector(_data(5000), model, "LOF_1")
        self.assertEqual(seen, [400])

    def test_a_failure_on_the_head_alone_does_not_exclude(self):
        """A per-call minimum longer than the head is the probe's fault, not
        the detector's: the whole series decides."""
        seen = []

        def needs_500(data, *a, **k):
            n = data.entities[0].Y.shape[1]
            seen.append(n)
            if n < 500:
                raise IndexError(f"needs 500 rows, got {n}")
            return {}

        with mock.patch.object(msu, "evaluate_model", needs_500):
            self.assertIsNone(msu.probe_detector(_data(800), _current_lof(), "COF_1"))
        self.assertEqual(seen, [msu.PROBE_TIMESTEPS, 800])

    def test_a_failure_on_both_reports_the_whole_series_failure(self):
        def always(data, *a, **k):
            raise RuntimeError(f"broken at n={data.entities[0].Y.shape[1]}")

        with mock.patch.object(msu, "evaluate_model", always):
            reason = msu.probe_detector(_data(800), _current_lof(), "X_1")
        self.assertEqual(reason, "RuntimeError: broken at n=800")

    def test_a_long_reason_is_truncated_to_one_line(self):
        def noisy(data, *a, **k):
            raise ValueError("line one\nline two " + "x" * 1000)

        with mock.patch.object(msu, "evaluate_model", noisy):
            reason = msu.probe_detector(_data(100), _current_lof(), "X_1")
        self.assertNotIn("\n", reason)
        self.assertLessEqual(len(reason), 300)

    def test_a_probe_timeout_uses_the_existing_deadline(self):
        class _Slow(TsadLof):
            def window_anomaly_score(self, *a, **k):
                time.sleep(30)

        original = msu.SCORING_TIMEOUT_SECONDS
        msu.SCORING_TIMEOUT_SECONDS = 1
        try:
            t0 = time.time()
            reason = msu.probe_detector(_data(5000), _Slow(), "SLOW_1")
            elapsed = time.time() - t0
        finally:
            msu.SCORING_TIMEOUT_SECONDS = original
        self.assertIn("deadline", reason)
        self.assertIn("SLOW_1", msu.timed_out_detectors())
        self.assertLess(elapsed, 5, "a timed-out head must not be retried in full")

    def test_the_probe_leaves_every_random_generator_where_it_was(self):
        """The probe runs before anomaly injection, which draws from numpy's
        global generator; consuming from it would move every injected anomaly."""
        class _Drawing(TsadLof):
            def window_anomaly_score(self, *a, **k):
                np.random.rand(10)
                random.random()
                t.randn(10)
                return super().window_anomaly_score(*a, **k)

        model = _Drawing(window_size=1, window_step=1, n_neighbors=5)
        model.fit(_loader(_data(400), 1))
        np.random.seed(7); random.seed(7); t.manual_seed(7)
        expected = (np.random.rand(3), random.random(), t.randn(3))
        np.random.seed(7); random.seed(7); t.manual_seed(7)
        self.assertIsNone(msu.probe_detector(_data(300), model, "LOF_1"))
        got = (np.random.rand(3), random.random(), t.randn(3))
        self.assertTrue(np.array_equal(expected[0], got[0]))
        self.assertEqual(expected[1], got[1])
        self.assertTrue(t.equal(expected[2], got[2]))

    def test_the_probe_does_not_touch_the_data(self):
        data = _data(600)
        before = data.entities[0].Y.copy()
        msu.probe_detector(data, _current_lof(), "LOF_1")
        msu.probe_detector(data, _stale_lof(), "LOF_2")
        self.assertTrue(np.array_equal(before, data.entities[0].Y))

    def test_detector_warnings_are_silenced_only_during_the_probe(self):
        from loguru import logger
        from Algorithms import windowed
        model = _current_lof()
        model.model.decision_function = lambda X: np.full(len(X), np.nan)
        messages = []
        sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            msu.probe_detector(_data(300), model, "LOF_1")
            self.assertFalse([m for m in messages if "non-finite" in m])
            windowed.score_windows(model.model, t.zeros(4, N_CHANNELS, 1))
            self.assertTrue([m for m in messages if "non-finite" in m],
                            "Algorithms logging was not re-enabled after the probe")
        finally:
            logger.remove(sink)


try:
    import dill
    import app
except Exception:                                          # pragma: no cover
    app = None


@unittest.skipIf(msu is None or app is None, "the pipeline does not import here")
class TestLoadTrainedModelsExcludes(unittest.TestCase):

    def setUp(self):
        msu.reset_timed_out()
        self.dir = tempfile.TemporaryDirectory()
        for name, model in (("LOF_1", _stale_lof()), ("LOF_2", _stale_lof()),
                            ("LOF_3", _current_lof()), ("LOF_4", _current_lof())):
            with open(os.path.join(self.dir.name, f"{name}.pth"), "wb") as f:
                t.save(model, f, pickle_module=dill)
        self.stamps = {n: (os.stat(os.path.join(self.dir.name, n)).st_mtime_ns,
                           open(os.path.join(self.dir.name, n), "rb").read())
                       for n in os.listdir(self.dir.name)}

    def tearDown(self):
        self.dir.cleanup()
        msu.reset_timed_out()

    def test_one_warning_per_excluded_detector_and_nothing_is_rewritten(self):
        from WebUI import markers
        names = ["LOF_1", "LOF_2", "LOF_3", "LOF_4", "LOF_9"]
        with self.assertLogs("app", level="WARNING") as logs:
            trained, excluded = app.load_trained_models(
                names, self.dir.name, probe_data=_data(229))
        self.assertEqual(sorted(trained), ["LOF_3", "LOF_4"])
        self.assertEqual(sorted(excluded), ["LOF_1", "LOF_2"])
        self.assertNotIn("LOF_9", excluded, "a missing checkpoint is not 'excluded'")

        lines = [l for l in logs.output if "Incompatible checkpoint" in l]
        self.assertEqual(len(lines), 2)
        codes = set()
        for name, line in zip(("LOF_1", "LOF_2"), lines):
            self.assertIn(f"Incompatible checkpoint {name} excluded", line)
            self.assertIn("576 features", line)
            self.assertIn("--overwrite true", line)
            self.assertNotIn("..", line)
            event = markers.classify(line)
            self.assertEqual(event["type"], "warning")
            self.assertIn(name, event["text"])
            self.assertIn("--overwrite true", event["text"])
            codes.add(event["code"])
        self.assertEqual(len(codes), 2, "each detector needs its own banner")

        for n, (mtime, content) in self.stamps.items():
            path = os.path.join(self.dir.name, n)
            self.assertEqual(os.stat(path).st_mtime_ns, mtime, n)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), content, n)

    def test_without_probe_data_nothing_is_probed(self):
        trained, excluded = app.load_trained_models(["LOF_1", "LOF_3"], self.dir.name)
        self.assertEqual(sorted(trained), ["LOF_1", "LOF_3"])
        self.assertEqual(excluded, {})

    def _write_unreadable(self):
        """The two kinds of unreadable checkpoint the real store holds.

        NN_1 names a module that no longer exists, as the pickles from when
        this code was the `algorithm` package do (whole entities of the smd
        and anomaly_archive trees). NN_2 is a sound checkpoint cut short, as a
        save that dies partway through Logger.save_torch_model's in-place
        write leaves it. NN_3 is the extreme of that: nothing was written.
        """
        orphan = types.ModuleType("ramses_removed_package")
        exec("class Detector:\n    window_size = 1\n", orphan.__dict__)
        sys.modules[orphan.__name__] = orphan
        try:
            with open(os.path.join(self.dir.name, "NN_1.pth"), "wb") as f:
                t.save(orphan.Detector(), f, pickle_module=dill)
        finally:
            del sys.modules[orphan.__name__]
        whole = io.BytesIO()
        t.save(_current_lof(), whole, pickle_module=dill)
        with open(os.path.join(self.dir.name, "NN_2.pth"), "wb") as f:
            f.write(whole.getvalue()[:3000])
        open(os.path.join(self.dir.name, "NN_3.pth"), "wb").close()
        return {n: (os.stat(os.path.join(self.dir.name, n)).st_mtime_ns,
                    open(os.path.join(self.dir.name, n), "rb").read())
                for n in ("NN_1.pth", "NN_2.pth", "NN_3.pth")}

    def test_an_unreadable_checkpoint_is_excluded_not_raised(self):
        """Before, the first of these ended the run at load with a traceback
        that named no detector, and LOF_3 — loaded fine just before it — went
        down with it."""
        from WebUI import markers
        stamps = self._write_unreadable()
        names = ["LOF_3", "NN_1", "NN_2", "LOF_4", "NN_3"]
        with self.assertLogs("app", level="WARNING") as logs:
            trained, excluded = app.load_trained_models(
                names, self.dir.name, probe_data=_data(229))
        self.assertEqual(sorted(trained), ["LOF_3", "LOF_4"])
        self.assertEqual(sorted(excluded), ["NN_1", "NN_2", "NN_3"])
        self.assertEqual(excluded["NN_1"],
                         "ModuleNotFoundError: No module named 'ramses_removed_package'")
        self.assertRegex(excluded["NN_2"], r"^RuntimeError: .*zip archive")
        self.assertRegex(excluded["NN_3"], r"^EOFError")

        lines = [l for l in logs.output if "Incompatible checkpoint" in l]
        self.assertEqual(len(lines), 3, logs.output)
        codes = set()
        for name, line in zip(("NN_1", "NN_2", "NN_3"), lines):
            self.assertTrue(line.startswith("WARNING:"), line)
            self.assertIn(f"Incompatible checkpoint {name} excluded: it cannot "
                          f"be loaded — {excluded[name].rstrip('.')}. Remedy: "
                          f"retrain it with --overwrite true.", line)
            self.assertNotIn("\n", line)
            event = markers.classify(line)
            self.assertEqual(event["type"], "warning")
            self.assertIn(name, event["text"])
            self.assertIn("--overwrite true", event["text"])
            codes.add(event["code"])
        self.assertEqual(len(codes), 3, "each detector needs its own banner")

        for n, (mtime, content) in {**self.stamps, **stamps}.items():
            path = os.path.join(self.dir.name, n)
            self.assertEqual(os.stat(path).st_mtime_ns, mtime, n)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), content, n)

    def test_an_unreadable_checkpoint_is_excluded_without_probe_data_too(self):
        """Reading is not part of the probe: no caller gets a traceback for it."""
        self._write_unreadable()
        with self.assertLogs("app", level="WARNING"):
            trained, excluded = app.load_trained_models(
                ["NN_1", "LOF_1", "NN_2"], self.dir.name)
        self.assertEqual(sorted(trained), ["LOF_1"])
        self.assertEqual(sorted(excluded), ["NN_1", "NN_2"])

    def test_interrupting_a_load_still_stops_the_run(self):
        """Only Exception is caught: Ctrl-C during a slow load must not read
        as one more excluded detector."""
        with mock.patch.object(app, "load_detector", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                app.load_trained_models(["LOF_3"], self.dir.name)


if __name__ == "__main__":
    unittest.main()
