"""The scoring deadline: a detector that cannot score in time leaves the run.

Exercised against the real `evaluate_model` with a stub model, so the timeout
and the refusal-on-repeat are tested where the pipeline actually calls them.
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    from Utils import model_selection_utils as msu
except Exception:                                          # pragma: no cover
    msu = None


class _SlowModel:
    """Sleeps in Python, so SIGALRM can land between bytecodes."""

    window_size = 1
    window_step = 1

    def __init__(self, seconds):
        self.seconds = seconds

    def eval(self):
        return self

    def window_anomaly_score(self, *a, **k):
        time.sleep(self.seconds)
        raise AssertionError("should have been killed before returning")


@unittest.skipIf(msu is None, "torch/numpy not installed in this interpreter")
class TestScoringDeadline(unittest.TestCase):

    def setUp(self):
        msu.reset_timed_out()

    def tearDown(self):
        msu.reset_timed_out()

    def test_a_slow_call_is_killed_at_the_deadline(self):
        t0 = time.time()
        with self.assertRaises(msu.ScoringTimeout):
            with msu._score_deadline("SLOW_1", seconds=1):
                time.sleep(30)
        self.assertLess(time.time() - t0, 5, "deadline did not fire")

    def test_a_fast_call_is_untouched(self):
        with msu._score_deadline("FAST_1", seconds=5):
            result = sum(range(1000))
        self.assertEqual(result, 499500)

    def test_the_alarm_is_cancelled_afterwards(self):
        """A deadline left armed would fire during an unrelated later call."""
        with msu._score_deadline("FAST_1", seconds=1):
            pass
        time.sleep(1.5)   # would raise here if the timer were still running

    def test_an_error_that_is_not_the_deadline_passes_through(self):
        with self.assertRaises(ZeroDivisionError):
            with msu._score_deadline("FAST_1", seconds=10):
                1 / 0

    def test_off_the_main_thread_there_is_no_deadline(self):
        """SIGALRM is main-thread only; --parallel must not crash."""
        import threading
        out = {}

        def worker():
            try:
                with msu._score_deadline("X_1", seconds=1):
                    out["ran"] = True
            except Exception as e:                          # pragma: no cover
                out["error"] = f"{type(e).__name__}: {e}"

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertTrue(out.get("ran"), out.get("error"))

    def test_a_timed_out_detector_is_registered_and_then_refused(self):
        """The cost is paid at most once: the second call raises immediately."""
        msu.SCORING_TIMEOUT_SECONDS, original = 1, msu.SCORING_TIMEOUT_SECONDS
        try:
            data = _entity(np.zeros((1, 40)))
            with self.assertRaises(msu.ScoringTimeout):
                msu.evaluate_model(data=data, model=_SlowModel(30),
                                   model_name="SLOW_1")
            self.assertIn("SLOW_1", msu.timed_out_detectors())
            t0 = time.time()
            with self.assertRaises(msu.ScoringTimeout):
                msu.evaluate_model(data=data, model=_SlowModel(30),
                                   model_name="SLOW_1")
            self.assertLess(time.time() - t0, 0.5, "second call re-ran the model")
        finally:
            msu.SCORING_TIMEOUT_SECONDS = original

    def test_an_untouched_detector_stays_out_of_the_register(self):
        self.assertEqual(msu.timed_out_detectors(), frozenset())


@unittest.skipIf(msu is None, "torch/numpy not installed in this interpreter")
class TestThompsonCandidateFilter(unittest.TestCase):
    """Thompson stops OFFERING a killed arm but never removes it from `models`:
    every per-model history must keep one entry per window."""

    def setUp(self):
        msu.reset_timed_out()

    def tearDown(self):
        msu.reset_timed_out()

    def _pool(self, n=12):
        return {f"M_{i}": object() for i in range(n)}

    def test_filter_is_inert_when_nothing_was_killed(self):
        pool = self._pool()
        slow = msu.timed_out_detectors()
        self.assertEqual(list({m: v for m, v in pool.items() if m not in slow}),
                         list(pool))

    def test_a_killed_arm_is_no_longer_offered(self):
        pool = self._pool()
        msu._TIMED_OUT.add("M_3")
        slow = msu.timed_out_detectors()
        offered = {m: v for m, v in pool.items() if m not in slow}
        self.assertNotIn("M_3", offered)
        self.assertEqual(len(offered), len(pool) - 1)
        # still present in the pool the histories are keyed on
        self.assertIn("M_3", pool)

    def test_selection_is_unchanged_while_the_filter_is_inert(self):
        """A reordered candidate dict would silently change every draw."""
        import random
        from Model_Selection.Thompson_Sampling import sample_model
        pool = self._pool(20)
        d = 6
        means = {m: np.random.RandomState(i).randn(d, 1) for i, m in enumerate(pool)}
        covs = {m: np.eye(d) for m in pool}
        ctx = np.ones(d) / np.sqrt(d)

        def draws(candidates, seed, n=100):
            random.seed(seed); np.random.seed(seed)
            return [sample_model(candidates, means, covs, 0.3, ctx) for _ in range(n)]

        same = {m: v for m, v in pool.items() if m not in msu.timed_out_detectors()}
        self.assertEqual(draws(pool, 5), draws(same, 5))

    def test_an_empty_candidate_set_is_guarded_not_raised(self):
        """random.choice raises IndexError, which the loop's `except ValueError`
        would not catch — hence the explicit guard."""
        import random
        from Model_Selection.Thompson_Sampling import sample_model
        random.seed(0)
        with self.assertRaises(IndexError):
            sample_model({}, {}, {}, 1.0, np.ones(4))
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, os.pardir, "Model_Selection",
                               "Thompson_Sampling.py")) as f:
            source = f.read()
        self.assertIn("if not candidates:", source)


def _entity(Y):
    from Datasets.dataset import Entity, Dataset
    ent = Entity(Y=Y, name="t", labels=np.zeros((1, Y.shape[1])), verbose=False)
    return Dataset(entities=[ent], name="t", verbose=False)


if __name__ == "__main__":
    unittest.main()
