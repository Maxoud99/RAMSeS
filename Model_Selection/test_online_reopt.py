"""Re-optimisations in child processes (Model_Selection/online_reopt.py).

Real child interpreters, running small targets defined below instead of the
model-selection pipeline, so each test takes a few seconds. Pinned: the result
comes back pickled; the child is seeded from the seed it is given, so the
same seed gives the same result and the parent's generators are untouched;
its results root is redirected; a failure comes back with its message and
traceback; a timeout and a kill stop the process; a reused result is a copy.
"""

import os
import random
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model_Selection.online_phase import ReoptFailed, ReoptTimedOut, Selection  # noqa: E402
from Model_Selection.online_reopt import CompletedReopt, ProcessReopt  # noqa: E402

HERE = "Model_Selection.test_online_reopt"


# ── Targets the child runs ───────────────────────────────────────────────────

def seeded_draw(window):
    """A 'selection' made from every generator a pipeline draws from."""
    import torch
    picks = [random.random(), float(np.random.random()), float(torch.rand(1))]
    return Selection(ensemble=[f"E{window}", f"{picks[0]:.12f}"],
                     single=f"{picks[1]:.12f}|{picks[2]:.12f}",
                     origin="reoptimization", origin_window=window)


def write_a_result_file(name):
    from Utils.paths import results_dir
    path = os.path.join(results_dir("robust_aggregated", "ds", "1"), name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("x")
    return path


def explode(message):
    raise ValueError(message)


def sleep_for(seconds):
    time.sleep(seconds)
    return "slept"


def die_without_a_result(code):
    os._exit(code)


# ── Tests ────────────────────────────────────────────────────────────────────

class ChildCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.handles = []

    def tearDown(self):
        for h in self.handles:
            h.close()

    def start(self, target, kwargs, seed=13, timeout=None, name="w", **kw):
        handle = ProcessReopt(f"{HERE}:{target}", kwargs,
                              os.path.join(self._tmp.name, name), seed=seed,
                              timeout=timeout, **kw)
        self.handles.append(handle)
        return handle

    def finish(self, handle, limit=120):
        self.assertTrue(handle.wait(limit), "the child did not finish in time")
        return handle


class TestResults(ChildCase):

    def test_the_selection_comes_back_and_on_success_sees_it_once(self):
        seen = []
        h = self.finish(self.start("seeded_draw", {"window": 5},
                                   on_success=lambda v, d: seen.append((v, d))))
        sel = h.result()
        self.assertIsInstance(sel, Selection)
        self.assertEqual((sel.origin, sel.origin_window, sel.ensemble[0]),
                         ("reoptimization", 5, "E5"))
        h.result()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0].single, sel.single)
        self.assertGreater(h.duration_s, 0)

    def test_same_seed_same_result_and_the_parent_generators_untouched(self):
        random.seed(1)
        np.random.seed(1)
        expected = (random.random(), float(np.random.random()))
        random.seed(1)
        np.random.seed(1)
        a = self.start("seeded_draw", {"window": 3}, seed=16, name="a")
        b = self.start("seeded_draw", {"window": 3}, seed=16, name="b")
        c = self.start("seeded_draw", {"window": 3}, seed=17, name="c")
        results = [self.finish(h).result() for h in (a, b, c)]
        self.assertEqual((results[0].ensemble, results[0].single),
                         (results[1].ensemble, results[1].single))
        self.assertNotEqual(results[0].single, results[2].single)
        self.assertEqual((random.random(), float(np.random.random())), expected)

    def test_the_child_writes_under_its_own_results_root(self):
        h = self.finish(self.start("write_a_result_file", {"name": "out.txt"}))
        path = h.result()
        self.assertTrue(path.startswith(os.path.join(h.workdir, "results")), path)
        h.close()
        self.assertFalse(os.path.exists(h.workdir))


class TestFailures(ChildCase):

    def test_an_exception_comes_back_with_its_traceback(self):
        h = self.finish(self.start("explode", {"message": "no ensemble, a, b"}))
        with self.assertRaises(ReoptFailed) as ctx:
            h.result()
        self.assertEqual(str(ctx.exception), "ValueError: no ensemble, a, b")
        self.assertIn("Traceback", ctx.exception.detail)
        self.assertNotIsInstance(ctx.exception, ReoptTimedOut)

    def test_a_timeout_kills_the_child(self):
        h = self.start("sleep_for", {"seconds": 60}, timeout=3)
        began = time.monotonic()
        self.assertTrue(h.wait(30))
        self.assertLess(time.monotonic() - began, 20)
        self.assertIsNotNone(h.proc.poll())                 # really gone
        with self.assertRaises(ReoptTimedOut) as ctx:
            h.result()
        self.assertIn("timeout", str(ctx.exception))

    def test_done_enforces_the_deadline_without_waiting(self):
        h = self.start("sleep_for", {"seconds": 60}, timeout=2)
        time.sleep(2.5)
        self.assertTrue(h.done())
        self.assertIsNotNone(h.proc.poll())
        with self.assertRaises(ReoptTimedOut):
            h.result()

    def test_kill_stops_a_running_child(self):
        h = self.start("sleep_for", {"seconds": 60})
        self.assertFalse(h.wait(0.5))
        h.kill()
        self.assertTrue(h.done())
        self.assertIsNotNone(h.proc.poll())
        with self.assertRaises(ReoptFailed):
            h.result()

    def test_a_child_that_dies_without_a_result_is_a_failure(self):
        h = self.finish(self.start("die_without_a_result", {"code": 3}))
        with self.assertRaises(ReoptFailed) as ctx:
            h.result()
        self.assertIn("status 3", str(ctx.exception))


class TestReused(unittest.TestCase):

    def test_a_reused_result_is_a_copy_with_the_original_duration(self):
        sel = Selection(["A", "B"], "A", origin="reoptimization", origin_window=4)
        h = CompletedReopt(sel, 81.5, reused_from="periodic")
        self.assertTrue(h.done() and h.wait(0))
        first, second = h.result(), h.result()
        self.assertIsNot(first, sel)
        self.assertIsNot(first, second)
        first.ensemble.append("X")
        self.assertEqual(sel.ensemble, ["A", "B"])
        self.assertEqual((h.duration_s, h.reused_from), (81.5, "periodic"))


if __name__ == "__main__":
    unittest.main()
