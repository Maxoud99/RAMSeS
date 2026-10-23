"""The online phase's trigger rules and switch-in timing, as pure functions
of a score series.

Every case is "given these window scores, which windows trigger, what
threshold was in effect, and where is each re-optimisation switched in", with
re-optimisations applied or pending at chosen windows. Nothing here touches a
model, a file or the clock.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model_Selection.online_policy import (  # noqa: E402
    MODE_ADAPTIVE, MODE_PERIODIC, MODE_RELATIVE, PendingReopt, PolicyParams,
    REASON_ADAPTIVE, REASON_PERIODIC, REASON_RELATIVE, ReoptSchedule,
    SUPPRESSED_PENDING, TriggerPolicy, adaptive_threshold, apply_window,
    relative_threshold, simulate, simulate_timeline)


def ens(*values):
    """A series where only the ensemble signal moves (single stays at 0.5)."""
    return [{"ensemble": v, "single": 0.5} for v in values]


def fired(decisions):
    return [d.window for d in decisions if d.fire]


class TestHelpers(unittest.TestCase):

    def test_relative_threshold_is_baseline_minus_x_percent(self):
        self.assertAlmostEqual(relative_threshold(0.8, 20), 0.64)
        self.assertAlmostEqual(relative_threshold(0.5, 50), 0.25)

    def test_adaptive_threshold_uses_population_std(self):
        # mean 2, population std sqrt(2/3); the sample std would be 1.
        self.assertAlmostEqual(adaptive_threshold([1.0, 2.0, 3.0], 1.0),
                               2.0 - math.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(adaptive_threshold([0.7, 0.7, 0.7], 2.0), 0.7)

    def test_unknown_scenario_is_refused(self):
        with self.assertRaises(ValueError):
            TriggerPolicy("sometimes")

    def test_params_name_only_what_the_scenario_uses(self):
        p = PolicyParams(update_interval=3, drop_pct=15, adaptive_window=4, adaptive_k=1.5,
                         reopt_delay=1)
        self.assertEqual(TriggerPolicy("fixed", p).params_dict(), {})
        self.assertEqual(TriggerPolicy("periodic", p).params_dict(),
                         {"update_interval": 3, "reopt_delay": 1})
        self.assertEqual(TriggerPolicy("threshold", p).params_dict(),
                         {"drop_pct": 15, "reopt_delay": 1})
        self.assertEqual(TriggerPolicy("adaptive_threshold", p).params_dict(),
                         {"adaptive_window": 4, "adaptive_k": 1.5, "drop_pct": 15,
                          "reopt_delay": 1})
        self.assertEqual(TriggerPolicy("fixed_random", p).params_dict(), {})

    def test_the_default_delay_is_two_windows(self):
        self.assertEqual(PolicyParams().reopt_delay, 2)


class TestFixed(unittest.TestCase):

    def test_never_triggers_and_reports_no_threshold(self):
        out = simulate("fixed", series=ens(0.9, 0.1, 0.0, 0.0, 0.0))
        self.assertEqual(fired(out), [])
        for d in out:
            self.assertIsNone(d.mode)
            self.assertEqual(d.thresholds, {"ensemble": None, "single": None})

    def test_the_random_baseline_never_triggers_either(self):
        self.assertEqual(fired(simulate("fixed_random", series=ens(0.9, 0.0, 0.0))), [])


class TestPeriodic(unittest.TestCase):

    def test_every_mth_window_index(self):
        out = simulate("periodic", ens(*[0.5] * 12), PolicyParams(update_interval=5))
        self.assertEqual([d.window for d in out][:2], [0, 1])     # window 0 is replayed
        self.assertEqual(fired(out), [5, 10])
        self.assertTrue(all(d.reason == REASON_PERIODIC for d in out if d.fire))
        self.assertTrue(all(d.mode == MODE_PERIODIC for d in out))

    def test_window_0_never_triggers(self):
        """The first online window is scored with the selection the offline
        phase has just made; re-optimising there would redo it."""
        self.assertEqual(fired(simulate("periodic", ens(*[0.5] * 3),
                                        PolicyParams(update_interval=1))), [1, 2])

    def test_indices_not_positions(self):
        out = simulate("periodic", ens(*[0.5] * 4), PolicyParams(update_interval=2),
                       first_window=3)
        self.assertEqual(fired(out), [4, 6])

    def test_scores_do_not_matter(self):
        out = simulate("periodic", ens(0.9, 0.0, 0.9, 0.0, 0.0), PolicyParams(update_interval=4))
        self.assertEqual(fired(out), [4])

    def test_a_trigger_while_one_runs_is_suppressed_not_queued(self):
        out = simulate("periodic", ens(*[0.5] * 7), PolicyParams(update_interval=2),
                       pending_at=[4, 5])
        by_window = {d.window: d for d in out}
        self.assertTrue(by_window[2].start)
        self.assertTrue(by_window[4].fire)
        self.assertFalse(by_window[4].start)
        self.assertEqual(by_window[4].suppressed_because, SUPPRESSED_PENDING)
        self.assertTrue(by_window[6].start)       # nothing carried over from 4


class TestRelativeThreshold(unittest.TestCase):

    P = PolicyParams(drop_pct=20)

    def test_baseline_is_the_first_window_and_has_no_threshold(self):
        out = simulate("threshold", first_window=1, series=ens(0.8, 0.79, 0.65, 0.63, 0.7), params=self.P)
        self.assertIsNone(out[0].thresholds["ensemble"])
        self.assertEqual(out[0].baselines["ensemble"], 0.8)
        for d in out[1:]:
            self.assertAlmostEqual(d.thresholds["ensemble"], 0.64)
            self.assertEqual(d.modes["ensemble"], MODE_RELATIVE)
        self.assertEqual(fired(out), [4])
        self.assertEqual(out[3].reason, REASON_RELATIVE)
        self.assertEqual(out[3].below, ["ensemble"])

    def test_more_than_x_percent_means_strictly_below(self):
        out = simulate("threshold", first_window=1, series=ens(0.5, 0.4, 0.3999), params=self.P)
        self.assertEqual(fired(out), [3])

    def test_either_signal_triggers(self):
        series = [{"ensemble": 0.8, "single": 0.6},
                  {"ensemble": 0.8, "single": 0.45},     # single: 0.45 < 0.48
                  {"ensemble": 0.6, "single": 0.6}]      # ensemble: 0.6 < 0.64
        out = simulate("threshold", first_window=1, series=series, params=self.P)
        self.assertEqual(fired(out), [2, 3])
        self.assertEqual(out[1].below, ["single"])
        self.assertEqual(out[2].below, ["ensemble"])
        self.assertAlmostEqual(out[1].thresholds["single"], 0.48)

    def test_both_below_names_both(self):
        series = [{"ensemble": 0.8, "single": 0.6}, {"ensemble": 0.1, "single": 0.1}]
        d = simulate("threshold", first_window=1, series=series, params=self.P)[1]
        self.assertEqual(d.below, ["ensemble", "single"])
        self.assertIn("ensemble", d.text)
        self.assertIn("single", d.text)

    def test_no_cooldown_a_persisting_drop_keeps_firing(self):
        out = simulate("threshold", first_window=1, series=ens(0.8, 0.5, 0.5, 0.5), params=self.P)
        self.assertEqual(fired(out), [2, 3, 4])

    def test_pending_suppresses_without_touching_the_baseline(self):
        out = simulate("threshold", first_window=1, series=ens(0.8, 0.5, 0.5, 0.5), params=self.P, pending_at=[3, 4])
        self.assertEqual([d.start for d in out], [False, True, False, False])
        self.assertEqual([d.suppressed_because for d in out],
                         [None, None, SUPPRESSED_PENDING, SUPPRESSED_PENDING])
        self.assertTrue(all(d.baselines["ensemble"] == 0.8 for d in out))

    def test_an_applied_reoptimization_resets_the_baseline(self):
        # Applied before window 3 is scored: window 3 is the new baseline.
        out = simulate("threshold", first_window=1, series=ens(0.8, 0.5, 0.5, 0.45, 0.39), params=self.P,
                       applied_at=[3])
        self.assertIsNone(out[2].thresholds["ensemble"])
        self.assertEqual(out[2].baselines["ensemble"], 0.5)
        self.assertAlmostEqual(out[3].thresholds["ensemble"], 0.4)
        self.assertEqual(fired(out), [2, 5])

    def test_unscorable_windows_are_skipped_and_do_not_set_the_baseline(self):
        series = [{"ensemble": None, "single": 0.5},
                  {"ensemble": float("nan"), "single": 0.5},
                  {"ensemble": 0.8, "single": 0.5},
                  {"ensemble": None, "single": 0.5},
                  {"ensemble": 0.6, "single": 0.5}]
        out = simulate("threshold", first_window=1, series=series, params=self.P)
        self.assertEqual([d.baselines["ensemble"] for d in out],
                         [None, None, 0.8, 0.8, 0.8])
        self.assertIsNone(out[2].thresholds["ensemble"])    # 0.8 set the baseline
        self.assertEqual(out[3].below, [])                  # None is not compared
        self.assertEqual(fired(out), [5])


class TestAdaptiveThreshold(unittest.TestCase):

    P = PolicyParams(drop_pct=20, adaptive_window=3, adaptive_k=1.0)

    def test_warm_up_uses_the_relative_rule(self):
        out = simulate("adaptive_threshold", first_window=1, series=ens(0.8, 0.7, 0.6, 0.5), params=self.P)
        self.assertEqual([d.modes["ensemble"] for d in out],
                         [MODE_RELATIVE, MODE_RELATIVE, MODE_RELATIVE, MODE_ADAPTIVE])
        self.assertIsNone(out[0].thresholds["ensemble"])
        self.assertAlmostEqual(out[1].thresholds["ensemble"], 0.64)
        self.assertAlmostEqual(out[2].thresholds["ensemble"], 0.64)
        # Window 3 (0.6 < 0.64) fires under the warm-up rule.
        self.assertEqual(out[2].reason, REASON_RELATIVE)
        expected = adaptive_threshold([0.8, 0.7, 0.6], 1.0)
        self.assertAlmostEqual(out[3].thresholds["ensemble"], expected)
        self.assertEqual(fired(out), [3, 4])
        self.assertEqual(out[3].reason, REASON_ADAPTIVE)

    def test_the_threshold_comes_from_the_last_n_scores_before_the_window(self):
        scores = [0.9, 0.1, 0.5, 0.5, 0.5, 0.2]
        out = simulate("adaptive_threshold", first_window=1, series=ens(*scores), params=self.P)
        # Window 6 is judged on windows 3..5 only; window 6's own score and the
        # older windows 1..2 play no part.
        self.assertAlmostEqual(out[5].thresholds["ensemble"], 0.5)
        self.assertEqual(out[5].below, ["ensemble"])

    def test_a_steady_series_never_fires(self):
        out = simulate("adaptive_threshold", first_window=1, series=ens(*[0.7] * 10), params=self.P)
        self.assertEqual(fired(out), [])

    def test_history_restarts_after_an_applied_reoptimization(self):
        out = simulate("adaptive_threshold", first_window=1, series=ens(*[0.7] * 4, 0.5, 0.5, 0.5, 0.5),
                       params=self.P, applied_at=[5])
        modes = [d.modes["ensemble"] for d in out]
        self.assertEqual(modes[3], MODE_ADAPTIVE)
        # Window 5 is the new baseline; the relative rule until 3 scores exist.
        self.assertEqual(modes[4:7], [MODE_RELATIVE] * 3)
        self.assertIsNone(out[4].thresholds["ensemble"])
        self.assertEqual(out[4].baselines["ensemble"], 0.5)
        self.assertEqual(modes[7], MODE_ADAPTIVE)
        self.assertEqual(fired(out), [])

    def test_mixed_modes_read_as_relative(self):
        """One signal warmed up, the other not: the warm-up still governs part
        of the window, and the per-signal modes carry the detail."""
        series = [{"ensemble": 0.7, "single": None}] + \
                 [{"ensemble": 0.7, "single": 0.7}] * 3
        d = simulate("adaptive_threshold", first_window=1, series=series, params=self.P)[3]
        self.assertEqual(d.modes, {"ensemble": MODE_ADAPTIVE, "single": MODE_RELATIVE})
        self.assertEqual(d.mode, MODE_RELATIVE)

    def test_pending_suppresses_but_the_history_keeps_growing(self):
        out = simulate("adaptive_threshold", first_window=1, series=ens(0.8, 0.8, 0.8, 0.1, 0.1), params=self.P,
                       pending_at=[4, 5])
        self.assertEqual(fired(out), [4, 5])
        self.assertFalse(any(d.start for d in out))
        policy = TriggerPolicy("adaptive_threshold", self.P)
        for d in ens(0.8, 0.8, 0.8, 0.1, 0.1):
            policy.decide(1, d, pending=True)
        self.assertEqual(len(policy.signals["ensemble"].history), 5)


def started(steps):
    return [(st.started.started_at_window, st.started.apply_window)
            for st in steps if st.started]


def applied(steps):
    return [(st.applied.started_at_window, st.window) for st in steps if st.applied]


def suppressed(steps):
    return [st.window for st in steps
            if st.decision.suppressed_because == SUPPRESSED_PENDING]


class TestReoptSchedule(unittest.TestCase):

    def test_apply_window_is_trigger_plus_delay(self):
        self.assertEqual(apply_window(5, 2), 7)
        self.assertEqual(apply_window(5, 0), 5)

    def test_pending_from_trigger_up_to_its_apply_window(self):
        schedule = ReoptSchedule(2)
        self.assertFalse(schedule.is_pending)
        self.assertEqual(schedule.start(5), PendingReopt(5, 7))
        self.assertIsNone(schedule.due(6))
        self.assertEqual(schedule.due(7), PendingReopt(5, 7))
        self.assertEqual(schedule.resolve(), PendingReopt(5, 7))
        self.assertFalse(schedule.is_pending)

    def test_one_at_a_time_and_valid_delays_only(self):
        schedule = ReoptSchedule(1)
        schedule.start(1)
        with self.assertRaises(RuntimeError):
            schedule.start(2)
        with self.assertRaises(RuntimeError):
            ReoptSchedule(1).resolve()
        with self.assertRaises(ValueError):
            ReoptSchedule(-1)


class TestTimeline(unittest.TestCase):
    """The runner's timeline: every re-optimisation switched in exactly L
    windows after its trigger, triggers in between suppressed."""

    def test_periodic_triggers_at_multiples_and_applies_l_later(self):
        steps = simulate_timeline("periodic", ens(*[0.5] * 13),
                                  PolicyParams(update_interval=5, reopt_delay=2))
        self.assertEqual(started(steps), [(5, 7), (10, 12)])
        self.assertEqual(applied(steps), [(5, 7), (10, 12)])
        self.assertEqual(suppressed(steps), [])

    def test_suppression_covers_trigger_up_to_apply(self):
        steps = simulate_timeline("periodic", ens(*[0.5] * 10),
                                  PolicyParams(update_interval=2, reopt_delay=3))
        self.assertEqual(started(steps), [(2, 5), (6, 9)])
        self.assertEqual(suppressed(steps), [4, 8])
        self.assertEqual(applied(steps), [(2, 5), (6, 9)])
        # Pending exactly on [trigger, apply): the apply window may start anew.
        steps = simulate_timeline("periodic", ens(*[0.5] * 8),
                                  PolicyParams(update_interval=2, reopt_delay=2))
        self.assertEqual(started(steps), [(2, 4), (4, 6), (6, 8)])
        self.assertEqual(applied(steps), [(2, 4), (4, 6)])
        self.assertEqual(suppressed(steps), [])

    def test_threshold_resets_at_the_apply_window_not_at_the_trigger(self):
        P = PolicyParams(drop_pct=20, reopt_delay=2)
        steps = simulate_timeline("threshold", ens(0.8, 0.8, 0.5, 0.5, 0.6, 0.6), P)
        self.assertEqual(started(steps), [(2, 4)])
        self.assertEqual(suppressed(steps), [3])                 # 0.5 < 0.64, pending
        self.assertEqual(applied(steps), [(2, 4)])
        self.assertEqual(steps[3].decision.baselines["ensemble"], 0.8)
        self.assertIsNone(steps[4].decision.thresholds["ensemble"])  # new baseline
        self.assertEqual(steps[4].decision.baselines["ensemble"], 0.6)
        self.assertAlmostEqual(steps[5].decision.thresholds["ensemble"], 0.48)

    def test_adaptive_history_restarts_at_the_apply_window(self):
        P = PolicyParams(drop_pct=20, adaptive_window=2, adaptive_k=1.0, reopt_delay=1)
        steps = simulate_timeline("adaptive_threshold",
                                  ens(0.8, 0.8, 0.8, 0.3, 0.7, 0.7, 0.7), P)
        self.assertEqual(steps[3].decision.modes["ensemble"], MODE_ADAPTIVE)
        self.assertEqual(started(steps), [(3, 4)])
        self.assertEqual(applied(steps), [(3, 4)])
        self.assertEqual([st.decision.modes["ensemble"] for st in steps[4:]],
                         [MODE_RELATIVE, MODE_RELATIVE, MODE_ADAPTIVE])

    def test_delay_zero_switches_in_at_the_trigger_and_rebaselines_there(self):
        P = PolicyParams(drop_pct=20, reopt_delay=0)
        steps = simulate_timeline("threshold", ens(0.8, 0.5, 0.45, 0.39), P)
        self.assertEqual(started(steps), [(1, 1), (3, 3)])
        self.assertTrue(steps[1].rescored)
        self.assertIsNone(steps[1].applied)       # not applied BEFORE scoring
        # Window 1's (re-)score is the new baseline: 0.45 is not below 0.4,
        # 0.39 is.
        self.assertAlmostEqual(steps[2].decision.thresholds["ensemble"], 0.4)
        self.assertEqual(suppressed(steps), [])

    def test_a_failed_reoptimization_still_suppresses_until_its_apply_window(self):
        P = PolicyParams(drop_pct=20, reopt_delay=2)
        steps = simulate_timeline("threshold", ens(0.8, 0.5, 0.5, 0.5, 0.5), P,
                                  fails=[1])
        self.assertEqual(started(steps), [(1, 3), (3, 5)])
        # 2 is suppressed by the failing one, 4 by the one started at 3.
        self.assertEqual(suppressed(steps), [2, 4])
        self.assertEqual(applied(steps), [])
        # The baseline survives the failure: window 3 is still judged on 0.8.
        self.assertEqual(steps[3].decision.baselines["ensemble"], 0.8)

    def test_one_due_after_the_last_window_is_never_applied(self):
        steps = simulate_timeline("periodic", ens(*[0.5] * 12),
                                  PolicyParams(update_interval=5, reopt_delay=2))
        self.assertEqual(started(steps), [(5, 7), (10, 12)])
        self.assertEqual(applied(steps), [(5, 7)])               # 12 > last (11)

    def test_fixed_has_no_timeline(self):
        steps = simulate_timeline("fixed", ens(0.9, 0.0, 0.0), PolicyParams())
        self.assertEqual((started(steps), applied(steps), suppressed(steps)), ([], [], []))


class TestObserve(unittest.TestCase):

    def test_observe_records_without_judging(self):
        policy = TriggerPolicy("threshold", PolicyParams(drop_pct=20))
        policy.observe({"ensemble": 0.7, "single": None})
        self.assertEqual(policy.baselines(), {"ensemble": 0.7, "single": None})
        d = policy.decide(1, {"ensemble": 0.5, "single": 0.5}, pending=False)
        self.assertEqual(d.below, ["ensemble"])

    def test_observe_is_a_no_op_for_rules_without_history(self):
        for key in ("fixed", "periodic"):
            policy = TriggerPolicy(key)
            policy.observe({"ensemble": 0.7, "single": 0.7})
            self.assertEqual(policy.baselines(), {"ensemble": None, "single": None})


if __name__ == "__main__":
    unittest.main()
