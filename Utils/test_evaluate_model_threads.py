"""`evaluate_model` must not change the model it scores.

RM and RNN ship `window_size = window_step = -1` ("one window over the whole
series"), which `_evaluate_model` resolves per call from the data's length.
It used to write the resolved step onto the model and restore it afterwards,
so two threads scoring the SAME object — the online stream and a background
re-optimisation, or the --parallel pipeline's workers — read each other's
step mid-call and restored in the wrong order. The model kept a stale step
for the rest of the run: wrong scores on one series length, an AssertionError
from `de_unfold` on another, and no log line about either.

Exercised with the real RunningMean and the real `evaluate_model`.
"""

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    from Algorithms.running_mean import RunningMean
    from Datasets.dataset import Dataset, Entity
    from Utils.model_selection_utils import evaluate_model
except Exception:                                          # pragma: no cover
    RunningMean = None


def _dataset(n, rng):
    labels = np.zeros((1, n))
    labels[0, n // 2:n // 2 + 5] = 1
    entity = Entity(Y=rng.randn(1, n).astype(np.float32), name=f"d{n}",
                    labels=labels, mask=np.ones((1, n)), verbose=False)
    return Dataset(entities=[entity], name=f"d{n}", verbose=False)


@unittest.skipIf(RunningMean is None, "torch/numpy not installed in this interpreter")
class TestEvaluateModelLeavesTheModelAlone(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(0)
        self.model = RunningMean(n_features=1, running_window_size=4)
        # The stream's window length and the re-optimisation's series length.
        self.small, self.big = _dataset(100, rng), _dataset(944, rng)

    def test_a_single_call_leaves_window_step_unresolved(self):
        evaluate_model(self.big, self.model, "RM_1")
        self.assertEqual(self.model.window_step, -1)
        self.assertEqual(self.model.window_size, -1)

    def test_two_threads_scoring_one_model(self):
        ref_small = evaluate_model(self.small, self.model, "RM_1")["entity_scores"].copy()
        ref_big = evaluate_model(self.big, self.model, "RM_1")["entity_scores"].copy()
        problems = {"worker": [], "main": []}
        stop = threading.Event()

        def check(side, data, ref):
            try:
                out = evaluate_model(data, self.model, "RM_1")["entity_scores"]
                if out.shape != ref.shape or not np.allclose(out, ref):
                    problems[side].append(f"wrong scores {out.shape}")
            except Exception as exc:
                problems[side].append(repr(exc))

        def worker():
            while not stop.is_set():
                check("worker", self.big, ref_big)

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            deadline = time.time() + 1.5
            while time.time() < deadline:
                check("main", self.small, ref_small)
        finally:
            stop.set()
            thread.join()

        self.assertEqual({k: v[:3] for k, v in problems.items()},
                         {"worker": [], "main": []})
        self.assertEqual(self.model.window_step, -1)
        # And afterwards, single-threaded, both lengths still score as before.
        np.testing.assert_allclose(
            evaluate_model(self.big, self.model, "RM_1")["entity_scores"], ref_big)
        np.testing.assert_allclose(
            evaluate_model(self.small, self.model, "RM_1")["entity_scores"], ref_small)


if __name__ == "__main__":
    unittest.main()
