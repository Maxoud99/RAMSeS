"""The online scenario runner and its results files, with fake scoring and
fake re-optimisation handles.

What is pinned here is the SCHEDULING: a re-optimisation triggered at window t
is switched in at exactly window t + L, whether it finished early (no wait) or
late (the stream waits there, with a recorded waiting state and heartbeats);
a trigger while one is pending is suppressed rather than queued; a failure or
a timeout keeps the old selection and baseline; one due after the last window
is killed and recorded as not applied; a scenario that fails while one is
pending kills it and records it as aborted; every scenario starts from its own
copy of the same selection. Plus the files: valid JSON after every window,
NaN as null, no temporary files left behind, full summary rows from the start,
and an earlier run's scenario files removed.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model_Selection.online_phase import (  # noqa: E402
    COUNT_FIELDS, OnlineResultsWriter, OnlineWindow, ReoptFailed, ReoptTimedOut,
    STATUS_WAITING, Selection, WindowScores, jsonable, run_online, run_scenario,
    write_json_atomic)
from Model_Selection.online_policy import PolicyParams  # noqa: E402


def windows(n, first=0):
    return [OnlineWindow(index=i, start=100 + 5 * i, end=200 + 5 * i,
                         online_start=5 * i, online_end=5 * i + 100,
                         segments=[{"copy": 0, "data_start": 0, "data_end": 100,
                                    "copy_start": 5 * i, "copy_end": 5 * i + 100}])
            for i in range(first, first + n)]


class Clock:
    """Counts scored windows; fake handles finish by it."""

    def __init__(self):
        self.scored = 0


class FakeScorer:
    """Scores from a table keyed by (selection origin, window index).

    The default is a steady 0.8 / 0.6, so only the cases a test writes down
    can move a policy.
    """

    def __init__(self, clock, table=None, default=(0.8, 0.6)):
        self.clock, self.table, self.default = clock, table or {}, default
        self.calls = []

    def __call__(self, window, selection):
        self.calls.append((window.index, selection.origin, selection.origin_window))
        ens, single = self.table.get((selection.origin, window.index), self.default)
        self.clock.scored += 1
        return WindowScores(ensemble=ens, single=single,
                            metrics={"ensemble": {"f1": ens}, "single": {"f1": single}},
                            misclassified={"ensemble": 1, "single": 2})


class FakeHandle:
    """A re-optimisation that finishes once `after` more windows are scored.

    `wait()` stands in for time passing: each call that cannot finish it yet
    counts as one heartbeat interval, and after `wait_calls` of them it
    finishes (as a real process eventually would).
    """

    def __init__(self, clock, window, after, fail=None, timeout=False, wait_calls=1,
                 on_wait=None):
        self.clock, self.window = clock, window
        self.due_at = clock.scored + after
        self.fail, self.timed_out = fail, timeout
        self.wait_calls, self.on_wait = wait_calls, on_wait
        self.waits = 0
        self.killed = self.closed = False
        self.reused_from = None
        self.duration_s = 12.5
        self._forced = False

    def done(self):
        return self._forced or self.killed or self.clock.scored >= self.due_at

    def wait(self, seconds):
        self.waits += 1
        if self.on_wait:
            self.on_wait(self)
        if self.waits >= self.wait_calls:
            self._forced = True
        return self.done()

    def result(self):
        assert self.done()
        if self.killed:
            raise ReoptFailed("killed")
        if self.timed_out:
            raise ReoptTimedOut("timeout: exceeded the 5s re-optimisation limit")
        if self.fail:
            raise ReoptFailed(self.fail)
        return Selection(ensemble=[f"E{self.window}"], single=f"S{self.window}",
                         origin="reoptimization", origin_window=self.window)

    def kill(self):
        if not self.done():
            self.killed = True

    def close(self):
        self.closed = True


class Starter:
    """start_reopt that records where it ran and hands out FakeHandles."""

    def __init__(self, clock, after=0, fail_at=(), timeout_at=(), wait_calls=1,
                 on_wait=None, raise_at=()):
        self.clock, self.after = clock, after
        self.fail_at, self.timeout_at, self.raise_at = fail_at, timeout_at, raise_at
        self.wait_calls, self.on_wait = wait_calls, on_wait
        self.started, self.handles = [], []

    def __call__(self, window, selection, key):
        if window.index in self.raise_at:
            raise OSError("disk full")
        self.started.append({"window": window.index, "from": selection.origin_window})
        handle = FakeHandle(self.clock, window.index, self.after,
                            fail=(f"boom at {window.index}" if window.index in self.fail_at
                                  else None),
                            timeout=window.index in self.timeout_at,
                            wait_calls=self.wait_calls, on_wait=self.on_wait)
        self.handles.append(handle)
        return handle


class RunnerCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self._tmp.name, "online", "ds", "1")
        self.clock = Clock()

    def tearDown(self):
        self._tmp.cleanup()

    def writer(self, keys=("periodic",), header=None, params=None, planned=None):
        return OnlineResultsWriter(self.dir, header or {"dataset": "ds", "entity": "1",
                                                        "strategy": "all"}, keys,
                                   scenario_params=params, n_windows_planned=planned)

    def run_one(self, key, n, params, table=None, starter=None, scorer=None, **kw):
        writer = self.writer((key,))
        recorder = writer.recorder(key, {}, n)
        starter = starter or Starter(self.clock)
        scorer = scorer or FakeScorer(self.clock, table=table)
        stats = run_scenario(key, params, windows(n), Selection(["A", "B"], "A"),
                             scorer, starter, recorder, heartbeat_s=0, **kw)
        doc = self.load(f"{key}.json")
        return stats, doc, starter, scorer

    def load(self, name):
        with open(os.path.join(self.dir, name)) as f:
            return json.load(f)


def rows(doc):
    return {w["i"]: w for w in doc["windows"]}


def events(doc, kind=None):
    return [e for e in doc["events"] if kind is None or e["type"] == kind]


class TestSwitchInAtTriggerPlusDelay(RunnerCase):

    def test_applied_exactly_l_windows_after_the_trigger_when_ready_early(self):
        # Finishes at once; still switched in only at 2 + 2 = 4.
        stats, doc, starter, scorer = self.run_one(
            "periodic", 7, PolicyParams(update_interval=2, reopt_delay=2),
            starter=Starter(self.clock, after=0))
        r = rows(doc)
        self.assertEqual(r[2]["trigger"]["started"], True)
        self.assertEqual(r[2]["trigger"]["apply_window"], 4)
        self.assertEqual(r[3]["reopt_pending"], {"started_at_window": 2, "apply_window": 4})
        self.assertIsNone(r[3]["reopt_applied"])
        self.assertEqual(r[3]["deployed_single"], "A")            # old one until 4
        self.assertEqual(r[4]["reopt_applied"]["started_at_window"], 2)
        self.assertEqual(r[4]["reopt_applied"]["apply_window"], 4)
        self.assertEqual(r[4]["deployed_single"], "S2")
        self.assertEqual(r[4]["waited_s"], 0.0)                   # no wait: it was done
        self.assertEqual(events(doc, "waiting"), [])
        # Window 4 is due AND a multiple of 2: the switch-in frees the slot,
        # so it starts the next one.
        self.assertTrue(r[4]["trigger"]["started"])
        self.assertEqual([s["window"] for s in starter.started], [2, 4, 6])
        self.assertEqual([s["from"] for s in starter.started], [None, 2, 4])
        self.assertEqual((stats["n_reopts_started"], stats["n_reopts_applied"]), (3, 2))
        self.assertEqual(stats["n_suppressed"], 0)

    def test_triggers_before_the_apply_window_are_suppressed(self):
        stats, doc, _, _ = self.run_one(
            "periodic", 8, PolicyParams(update_interval=1, reopt_delay=3),
            starter=Starter(self.clock, after=0))
        r = rows(doc)
        self.assertEqual([i for i, w in r.items()
                          if w["trigger"] and w["trigger"]["started"]], [1, 4, 7])
        self.assertEqual([i for i, w in r.items()
                          if w["trigger"] and w["trigger"]["suppressed_because"]],
                         [2, 3, 5, 6])
        self.assertEqual([i for i, w in r.items() if w["reopt_applied"]], [4, 7])
        self.assertEqual(stats["n_suppressed"], 4)

    def test_the_stream_waits_at_the_apply_window_and_says_so(self):
        seen = []

        def look(handle):
            # What a reader polling the files sees while the stream waits.
            summary = self.load("summary.json")
            doc = self.load("periodic.json")
            seen.append((summary["scenarios"][0]["status"],
                         summary["scenarios"][0]["waiting"], doc["status"],
                         summary["updated_at"]))

        starter = Starter(self.clock, after=99, wait_calls=3, on_wait=look)
        stats, doc, _, _ = self.run_one("periodic", 5,
                                        PolicyParams(update_interval=2, reopt_delay=1),
                                        starter=starter)
        r = rows(doc)
        self.assertEqual(r[3]["reopt_applied"]["started_at_window"], 2)
        self.assertGreaterEqual(r[3]["waited_s"], 0.0)
        self.assertEqual(len(events(doc, "waiting")), 1)
        self.assertEqual(events(doc, "waiting")[0],
                         {**events(doc, "waiting")[0], "window": 3,
                          "started_at_window": 2, "apply_window": 3})
        self.assertEqual(len(seen), 3)
        for status, waiting, doc_status, _ in seen:
            self.assertEqual(status, STATUS_WAITING)
            self.assertEqual(doc_status, STATUS_WAITING)
            self.assertEqual((waiting["since_window"], waiting["until_window"]), (2, 3))
        # Heartbeats: updated_at moved between the polls.
        self.assertLess(seen[0][3], seen[-1][3])
        self.assertEqual(stats["n_waits"], 1)
        self.assertEqual(stats["status"], "complete")
        self.assertIsNone(stats["waiting"])

    def test_a_failure_keeps_the_selection_and_the_baseline(self):
        table = {("offline", 0): (0.8, 0.6), ("offline", 1): (0.5, 0.6)}
        starter = Starter(self.clock, fail_at=(1,))
        stats, doc, _, _ = self.run_one("threshold", 5,
                                        PolicyParams(drop_pct=20, reopt_delay=2),
                                        table=table, starter=starter)
        r = rows(doc)
        self.assertEqual(r[1]["trigger"]["reason"], "relative_drop")
        self.assertIsNone(r[3]["reopt_applied"])
        self.assertEqual(r[3]["reopt_failed"]["error"], "ReoptFailed: boom at 1")
        self.assertEqual(r[3]["deployed_single"], "A")
        self.assertEqual(r[3]["baselines"]["ensemble"], 0.8)          # not reset
        failed = events(doc, "failed")
        self.assertEqual([(e["window"], e["started_at_window"], e["timed_out"])
                          for e in failed], [(3, 1, False)])
        self.assertEqual((stats["n_reopts_failed"], stats["n_reopts_timed_out"]), (1, 0))

    def test_a_timeout_is_a_failure_marked_as_such(self):
        starter = Starter(self.clock, timeout_at=(2,))
        stats, doc, _, _ = self.run_one("periodic", 5,
                                        PolicyParams(update_interval=2, reopt_delay=2),
                                        starter=starter)
        failed = events(doc, "failed")[0]
        self.assertTrue(failed["timed_out"])
        self.assertIn("timeout", failed["error"])
        self.assertEqual((stats["n_reopts_failed"], stats["n_reopts_timed_out"]), (1, 1))
        self.assertEqual(rows(doc)[4]["deployed_single"], "A")

    def test_one_due_after_the_last_window_is_killed_not_waited_for(self):
        starter = Starter(self.clock, after=99)
        stats, doc, _, _ = self.run_one("periodic", 5,
                                        PolicyParams(update_interval=3, reopt_delay=2),
                                        starter=starter)
        handle = starter.handles[0]
        self.assertTrue(handle.killed and handle.closed)
        self.assertEqual(handle.waits, 0)
        last = events(doc)[-1]
        self.assertEqual((last["type"], last["window"], last["started_at_window"],
                          last["apply_window"], last["reason"], last["completed"]),
                         ("not_applied", None, 3, 5, "stream_ended", False))
        self.assertEqual(stats["n_reopts_not_applied"], 1)
        self.assertEqual(stats["n_waits"], 0)

    def test_a_finished_one_due_after_the_end_is_still_not_applied(self):
        stats, doc, starter, _ = self.run_one(
            "periodic", 5, PolicyParams(update_interval=4, reopt_delay=2),
            starter=Starter(self.clock, after=0))
        self.assertEqual(events(doc)[-1]["type"], "not_applied")
        self.assertTrue(events(doc)[-1]["completed"])
        self.assertTrue(all(w["deployed_single"] == "A" for w in doc["windows"]))

    def test_delay_zero_rescores_the_triggering_window(self):
        table = {("offline", 0): (0.8, 0.6), ("offline", 1): (0.5, 0.6),
                 ("reoptimization", 1): (0.7, 0.6), ("reoptimization", 2): (0.55, 0.6)}
        stats, doc, _, scorer = self.run_one("threshold", 3,
                                             PolicyParams(drop_pct=20, reopt_delay=0),
                                             table=table)
        r = rows(doc)
        self.assertTrue(r[1]["rescored"])
        self.assertEqual(r[1]["trigger"]["scores"], {"ensemble": 0.5, "single": 0.6})
        self.assertEqual(r[1]["ensemble_score"], 0.7)            # the new selection's
        self.assertEqual(r[1]["deployed_single"], "S1")
        self.assertEqual(r[1]["baselines"]["ensemble"], 0.7)
        self.assertAlmostEqual(r[2]["thresholds"]["ensemble"], 0.56)
        self.assertTrue(r[2]["trigger"]["started"])              # 0.55 < 0.56
        self.assertEqual([c[:2] for c in scorer.calls[:3]],
                         [(0, "offline"), (1, "offline"), (1, "reoptimization")])

    def test_a_scenario_failing_while_one_is_pending_aborts_it(self):
        class Exploding(FakeScorer):
            def __call__(self, window, selection):
                if window.index == 3:
                    raise RuntimeError("scorer bug")
                return super().__call__(window, selection)

        writer = self.writer(("periodic",))
        starter = Starter(self.clock, after=99)
        with self.assertRaises(RuntimeError):
            run_scenario("periodic", PolicyParams(update_interval=2, reopt_delay=3),
                         windows(6), Selection(["A"], "A"), Exploding(self.clock),
                         starter, writer.recorder("periodic", {}, 6), heartbeat_s=0)
        doc = self.load("periodic.json")
        self.assertEqual([e["type"] for e in doc["events"]],
                         ["trigger", "started", "aborted"])
        aborted = doc["events"][-1]
        self.assertEqual((aborted["window"], aborted["started_at_window"],
                          aborted["apply_window"]), (3, 2, 5))
        self.assertIn("scorer bug", aborted["error"])
        self.assertTrue(starter.handles[0].killed)
        self.assertEqual(doc["status"], "failed")
        row = self.load("summary.json")["scenarios"][0]
        self.assertEqual((row["status"], row["n_reopts_aborted"]), ("failed", 1))

    def test_a_start_that_raises_is_a_failure_and_leaves_nothing_pending(self):
        starter = Starter(self.clock, raise_at=(2,))
        stats, doc, _, _ = self.run_one("periodic", 5,
                                        PolicyParams(update_interval=2, reopt_delay=1),
                                        starter=starter)
        r = rows(doc)
        self.assertFalse(r[2]["trigger"]["started"])
        self.assertIsNone(r[3]["reopt_pending"])
        self.assertTrue(r[4]["trigger"]["started"])
        self.assertIn("disk full", events(doc, "failed")[0]["error"])
        trigger_2 = [e for e in events(doc, "trigger") if e["window"] == 2][0]
        self.assertFalse(trigger_2["started"])
        self.assertEqual(stats["n_reopts_failed"], 1)

    def test_every_started_reoptimization_is_closed_exactly_once(self):
        starter = Starter(self.clock, after=1, fail_at=(4,))
        _, doc, _, _ = self.run_one("periodic", 9,
                                    PolicyParams(update_interval=2, reopt_delay=2),
                                    starter=starter)
        started = [e["window"] for e in events(doc, "started")]
        closed = [e["started_at_window"] for e in doc["events"]
                  if e["type"] in ("applied", "failed", "not_applied", "aborted")]
        self.assertEqual(sorted(started), sorted(closed))
        self.assertTrue(all(h.closed for h in starter.handles))

    def test_a_reused_result_is_counted(self):
        class Reusing(Starter):
            def __call__(self, window, selection, key):
                handle = super().__call__(window, selection, key)
                handle.reused_from = "periodic"
                return handle

        stats, doc, _, _ = self.run_one("threshold", 4, PolicyParams(drop_pct=20),
                                        table={("offline", 1): (0.1, 0.6)},
                                        starter=Reusing(self.clock))
        self.assertEqual(events(doc, "started")[0]["reused_from"], "periodic")
        self.assertEqual(stats["n_reopts_reused"], 1)

    def test_fixed_never_starts_anything(self):
        writer = self.writer(("fixed",))
        stats = run_scenario("fixed", PolicyParams(), windows(4), Selection(["A"], "A"),
                             FakeScorer(self.clock, default=(0.0, 0.0)), None,
                             writer.recorder("fixed", {}, 4))
        self.assertEqual(stats["n_triggers"], 0)
        doc = self.load("fixed.json")
        self.assertTrue(all(w["threshold_mode"] is None and w["trigger"] is None
                            for w in doc["windows"]))

    def test_window_records_carry_the_contract_fields(self):
        _, doc, _, _ = self.run_one("periodic", 3, PolicyParams(update_interval=2))
        row = doc["windows"][0]
        for name in ("i", "start", "end", "online_start", "online_end", "segments",
                     "data_digest", "n_samples", "n_anomalous", "scorable", "t",
                     "ensemble_score", "single_score", "metrics", "thresholds",
                     "threshold_modes", "threshold_mode", "baselines", "below",
                     "trigger", "reopt_pending", "reopt_running", "reopt_applied",
                     "reopt_failed", "waited_s", "rescored", "deployed_ensemble",
                     "deployed_single", "misclassified", "errors"):
            self.assertIn(name, row)
        self.assertEqual((row["i"], row["start"], row["end"]), (0, 100, 200))
        self.assertEqual((row["online_start"], row["online_end"]), (0, 100))
        self.assertEqual(row["threshold_mode"], "periodic")
        self.assertEqual(doc["windows"][2]["trigger"],
                         {"reason": "periodic", "text": "window 2 is a multiple of 2",
                          "signals": [], "scores": {"ensemble": 0.8, "single": 0.6},
                          "started": True, "apply_window": 4,
                          "suppressed_because": None})
        self.assertEqual([e["type"] for e in doc["events"][:2]], ["trigger", "started"])


class LabelScorer(FakeScorer):
    """What WindowScorer does with real labels: a window whose true labels are
    one class gets no score, the others the table's (or default) scores."""

    def __call__(self, window, selection):
        if not window.scorable:
            self.calls.append((window.index, selection.origin, selection.origin_window))
            self.clock.scored += 1
            return WindowScores(errors={"ensemble": "ValueError: the window's labels are all one class",
                                        "single": "ValueError: the window's labels are all one class"})
        return super().__call__(window, selection)


def labelled_windows(n, anomalous):
    """`windows(n)` with true-label counts: windows in `anomalous` hold 7 of
    their 100 samples anomalous, the others none."""
    out = windows(n)
    for w in out:
        w.n_samples = 100
        w.n_anomalous = 7 if w.index in anomalous else 0
    return out


class TestRealLabelWindows(RunnerCase):
    """Unscorable windows (one-class true labels): a gap, never a trigger,
    never in the adaptive history."""

    def run_labelled(self, key, n, anomalous, params, table=None, starter=None):
        writer = self.writer((key,))
        recorder = writer.recorder(key, {}, n)
        starter = starter or Starter(self.clock)
        scorer = LabelScorer(self.clock, table=table)
        stats = run_scenario(key, params, labelled_windows(n, anomalous),
                             Selection(["A", "B"], "A"), scorer, starter, recorder,
                             heartbeat_s=0)
        return stats, self.load(f"{key}.json"), starter

    def test_unscorable_windows_are_recorded_as_gaps(self):
        stats, doc, _ = self.run_labelled("fixed", 6, {0, 1, 4}, PolicyParams())
        r = rows(doc)
        self.assertEqual([r[i]["scorable"] for i in range(6)],
                         [True, True, False, False, True, False])
        for i in (2, 3, 5):
            self.assertIsNone(r[i]["ensemble_score"])
            self.assertIsNone(r[i]["single_score"])
            self.assertEqual(r[i]["n_anomalous"], 0)
            self.assertIn("one class", r[i]["errors"]["ensemble"])
        self.assertEqual(r[4]["n_anomalous"], 7)
        self.assertEqual((stats["n_windows"], stats["n_windows_scorable"],
                          stats["n_windows_scored"]), (6, 3, 3))
        # Aggregates are over the scored windows only, not zeros for the gaps.
        self.assertAlmostEqual(stats["mean_ensemble_score"], 0.8)

    def test_triggers_follow_the_true_label_scores_and_never_a_gap(self):
        # Baseline 0.8 at window 0; windows 1-3 hold no anomaly (no score),
        # window 4 drops to 0.5 < 0.8 * 0.8 and triggers.
        stats, doc, starter = self.run_labelled(
            "threshold", 7, {0, 4, 5, 6}, PolicyParams(drop_pct=20, reopt_delay=2),
            table={("offline", 4): (0.5, 0.6)})
        r = rows(doc)
        self.assertEqual([i for i in r if r[i]["trigger"]], [4])
        self.assertEqual([s["window"] for s in starter.started], [4])
        for i in (1, 2, 3):
            self.assertIsNone(r[i]["trigger"])
            self.assertEqual(r[i]["below"], [])
        self.assertEqual(r[4]["thresholds"]["ensemble"], 0.8 * 0.8)
        self.assertEqual(r[4]["trigger"]["scores"]["ensemble"], 0.5)
        self.assertEqual(stats["n_triggers"], 1)

    def test_a_gap_does_not_enter_the_adaptive_history(self):
        # N = 3 scorable scores of 0.8, 0.7, 0.9 with gaps between them; the
        # threshold at window 6 must be computed from exactly those three.
        table = {("offline", 0): (0.8, 0.6), ("offline", 2): (0.7, 0.6),
                 ("offline", 4): (0.9, 0.6), ("offline", 6): (0.79, 0.6)}
        _, doc, _ = self.run_labelled("adaptive_threshold", 7, {0, 2, 4, 6},
                                      PolicyParams(adaptive_window=3, adaptive_k=1.0),
                                      table=table)
        r = rows(doc)
        scores = [0.8, 0.7, 0.9]
        mean = sum(scores) / 3
        std = (sum((x - mean) ** 2 for x in scores) / 3) ** 0.5
        self.assertEqual(r[6]["threshold_modes"]["ensemble"], "adaptive")
        self.assertAlmostEqual(r[6]["thresholds"]["ensemble"], mean - std)
        for i in (1, 3, 5):
            self.assertIsNone(r[i]["trigger"])

    def test_a_baseline_on_an_unscorable_window_waits_for_a_scorable_one(self):
        _, doc, _ = self.run_labelled("threshold", 4, {2, 3}, PolicyParams(drop_pct=20),
                                      table={("offline", 3): (0.1, 0.6)})
        r = rows(doc)
        self.assertEqual(r[2]["baselines"]["ensemble"], 0.8)
        self.assertTrue(r[3]["trigger"])

    def test_a_reopt_that_cannot_start_is_recorded_and_the_stream_goes_on(self):
        """A real-label re-optimisation with no labelled anomaly in what it
        would select on is refused before it starts (require_labelled_anomaly)."""
        class Refusing(Starter):
            def __call__(self, window, selection, key):
                raise ValueError("no labelled anomaly in the data seen so far")

        stats, doc, _ = self.run_labelled("periodic", 5, {0, 1, 2, 3, 4},
                                          PolicyParams(update_interval=2, reopt_delay=1),
                                          starter=Refusing(self.clock))
        self.assertEqual(stats["status"], "complete")
        self.assertEqual(stats["n_reopts_failed"], 2)
        self.assertEqual(stats["n_reopts_started"], 0)
        self.assertIn("no labelled anomaly", events(doc, "failed")[0]["error"])


class TestRunOnline(RunnerCase):

    def test_every_scenario_replays_the_same_windows_from_its_own_copy(self):
        handed_out = []

        def initial(key):
            sel = Selection(["A", "B"], "A")
            # The object itself, not its id(): a finished scenario's selection
            # can be freed and its id reused by the next one.
            handed_out.append((key, sel))
            return sel

        scorer = FakeScorer(self.clock)
        keys = ("fixed", "periodic", "threshold", "adaptive_threshold")
        writer = self.writer(keys)
        results = run_online(keys, PolicyParams(update_interval=3), windows(5),
                             initial, scorer, Starter(self.clock), writer, heartbeat_s=0)
        self.assertEqual([r["key"] for r in results], list(keys))
        self.assertEqual([k for k, _ in handed_out], list(keys))
        self.assertEqual(len({id(s) for _, s in handed_out}), 4)
        per_scenario = [scorer.calls[i:i + 5] for i in range(0, 20, 5)]
        for calls in per_scenario:
            self.assertEqual([c[0] for c in calls], [0, 1, 2, 3, 4])
        summary = self.load("summary.json")
        self.assertEqual(summary["status"], "complete")
        self.assertEqual([s["key"] for s in summary["scenarios"]], list(keys))
        for row in summary["scenarios"]:
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["n_windows"], 5)
        self.assertEqual(summary["scenarios"][1]["params"],
                         {"update_interval": 3, "reopt_delay": 2})
        self.assertAlmostEqual(summary["scenarios"][0]["mean_ensemble_score"], 0.8)

    def test_pending_rows_have_every_field_from_the_start(self):
        params = {"fixed": {}, "periodic": {"update_interval": 5, "reopt_delay": 2}}
        self.writer(("fixed", "periodic"), params=params, planned=12)
        summary = self.load("summary.json")
        self.assertEqual(summary["status"], "running")
        self.assertEqual(summary["pid"], os.getpid())
        row = summary["scenarios"][1]
        self.assertEqual((row["status"], row["params"], row["n_windows"],
                          row["n_windows_planned"]),
                         ("pending", {"update_interval": 5, "reopt_delay": 2}, 0, 12))
        for name in COUNT_FIELDS:
            self.assertEqual(row[name], 0, name)
        # The same shape as a finished row.
        writer = self.writer(("fixed",))
        run_online(("fixed",), PolicyParams(), windows(2),
                   lambda k: Selection(["A"], "A"), FakeScorer(self.clock), None, writer)
        finished = self.load("summary.json")["scenarios"][0]
        self.assertEqual(set(finished), set(row))

    def test_a_failing_scenario_does_not_stop_the_others(self):
        class Exploding(FakeScorer):
            def __call__(self, window, selection):
                if self.current == "periodic":
                    raise RuntimeError("scorer bug")
                return super().__call__(window, selection)

        scorer = Exploding(self.clock)
        keys = ("fixed", "periodic", "threshold")

        def initial(key):
            scorer.current = key
            return Selection(["A"], "A")

        writer = self.writer(keys)
        results = run_online(keys, PolicyParams(), windows(3), initial, scorer,
                             Starter(self.clock), writer)
        self.assertEqual([r["key"] for r in results], ["fixed", "threshold"])
        summary = self.load("summary.json")
        self.assertEqual(summary["status"], "complete")
        self.assertIn("periodic", summary["error"])
        statuses = {s["key"]: s["status"] for s in summary["scenarios"]}
        self.assertEqual(statuses, {"fixed": "complete", "periodic": "failed",
                                    "threshold": "complete"})
        self.assertIn("scorer bug", self.load("periodic.json")["error"])

    def test_every_scenario_failing_fails_the_phase(self):
        def broken(window, selection):
            raise RuntimeError("nope")
        writer = self.writer(("fixed",))
        run_online(("fixed",), PolicyParams(), windows(2),
                   lambda k: Selection(["A"], "A"), broken, None, writer)
        self.assertEqual(self.load("summary.json")["status"], "failed")


class TestFiles(RunnerCase):

    def test_nan_and_numpy_become_plain_json(self):
        import numpy as np
        value = jsonable({"a": float("nan"), "b": np.float32(0.5), "c": np.array([1, 2]),
                          "d": (np.int64(3), float("inf")), "e": np.bool_(True)})
        self.assertEqual(value, {"a": None, "b": 0.5, "c": [1, 2], "d": [3, None],
                                 "e": True})
        path = os.path.join(self.dir, "x.json")
        write_json_atomic(path, {"score": float("nan")})
        self.assertEqual(json.load(open(path)), {"score": None})

    def test_no_temporary_file_survives_a_write(self):
        self.run_one("periodic", 3, PolicyParams(update_interval=2))
        self.assertEqual(sorted(os.listdir(self.dir)), ["periodic.json", "summary.json"])

    def test_scenarios_not_in_this_run_are_cleared(self):
        os.makedirs(self.dir, exist_ok=True)
        for stale in ("threshold.json", "fixed_random.json", "notes.txt"):
            open(os.path.join(self.dir, stale), "w").write("{}")
        self.writer(("periodic",))
        self.assertEqual(sorted(os.listdir(self.dir)), ["notes.txt", "summary.json"])

    def test_skipping_closes_out_pending_rows_with_their_params(self):
        writer = self.writer(("fixed", "periodic"),
                             params={"periodic": {"update_interval": 5, "reopt_delay": 2}})
        summary = self.load("summary.json")
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual([s["status"] for s in summary["scenarios"]], ["pending", "pending"])
        writer.finish("skipped", "too short")
        summary = self.load("summary.json")
        self.assertEqual((summary["status"], summary["error"]), ("skipped", "too short"))
        self.assertEqual([s["status"] for s in summary["scenarios"]], ["skipped", "skipped"])
        self.assertEqual(summary["scenarios"][1]["params"],
                         {"update_interval": 5, "reopt_delay": 2})
        self.assertEqual(summary["scenarios"][1]["error"], "too short")


class TestLogLines(RunnerCase):

    def test_one_window_line_per_window_with_parsable_fields(self):
        records = []

        class Capture:
            def info(self, msg):
                records.append(msg)

            def error(self, msg):
                records.append(msg)

        writer = self.writer(("periodic",))
        # Windows 0..4: 2 is applied at 3 after a wait, 4 is due after the end.
        run_scenario("periodic", PolicyParams(update_interval=2, reopt_delay=1), windows(5),
                     Selection(["A", "B"], "A"), FakeScorer(self.clock),
                     Starter(self.clock, after=99, wait_calls=2),
                     writer.recorder("periodic", {}, 5), log=Capture(), heartbeat_s=0)
        lines = [r for r in records if r.startswith("ONLINE WINDOW:")]
        self.assertEqual(len(lines), 5)
        self.assertIn("window=3/5", lines[2])
        self.assertIn("trigger=started", lines[2])
        self.assertIn("deployed_ensemble=A+B", lines[0])
        self.assertTrue(any(r.startswith("ONLINE SCENARIO START:") for r in records))
        self.assertTrue(any(r.startswith("ONLINE SCENARIO COMPLETE:") for r in records))
        self.assertTrue(any("event=waiting, window=3, started_at_window=2" in r
                            for r in records))
        self.assertTrue(any("event=applied, window=3" in r for r in records))
        self.assertTrue(any("event=not_applied" in r and r.endswith("reason=stream_ended")
                            for r in records))


if __name__ == "__main__":
    unittest.main()
