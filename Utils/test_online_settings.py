"""The online phase's settings: CLI flag, else config key, else default.

Also the online split, window geometry and feasibility verdict that app.py
and the web UI share, the results root the CLI pins from the config it was
given, and the per-thread results redirection.
"""

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from Utils import paths as ramses_paths  # noqa: E402
from Utils import pipeline_spec as spec  # noqa: E402


class TestStrategy(unittest.TestCase):

    def test_all_is_the_four_scenarios_in_comparison_order(self):
        self.assertEqual(spec.parse_online_strategy("all"),
                         ("fixed", "periodic", "threshold", "adaptive_threshold"))

    def test_old_spellings_still_work(self):
        self.assertEqual(spec.parse_online_strategy("adaptive"), ("periodic",))
        self.assertEqual(spec.parse_online_strategy("fixed-best"), ("fixed",))
        self.assertEqual(spec.parse_online_strategy("fixed-random"), ("fixed_random",))
        self.assertEqual(spec.parse_online_strategy(" Adaptive-Threshold "),
                         ("adaptive_threshold",))

    def test_default_is_what_adaptive_always_was(self):
        self.assertEqual(spec.parse_online_strategy(None), ("periodic",))
        self.assertEqual(spec.parse_online_strategy(""), ("periodic",))

    def test_the_random_baseline_is_cli_only(self):
        with self.assertRaisesRegex(ValueError, "command-line baseline"):
            spec.parse_online_strategy("fixed-random", allow_baselines=False)

    def test_unknown_names_the_valid_ones(self):
        with self.assertRaisesRegex(ValueError, "adaptive_threshold, all"):
            spec.parse_online_strategy("sometimes")

    def test_every_scenario_has_a_label(self):
        for key in spec.ONLINE_SCENARIOS + spec.ONLINE_BASELINE_SCENARIOS:
            self.assertIn(key, spec.ONLINE_SCENARIO_LABELS)


class TestNumbers(unittest.TestCase):

    def test_bounds(self):
        for parse, good, bad in (
            (spec.parse_update_interval, ("1", 5, "5.0"), (0, -1, "2.5", "x", True)),
            (spec.parse_drop_pct, ("20", 0.5, "99.9", "20%"), (0, 100, -5, "nan", "inf")),
            (spec.parse_adaptive_window, (2, "10"), (1, 0, "3.5")),
            (spec.parse_adaptive_k, (0, "2", 0.5), (-0.1, "nan")),
            (spec.parse_max_online_windows, (1, "15"), (0, -3, "1.5")),
            (spec.parse_reopt_delay, (0, "2", 7), (-1, "1.5", True, "x")),
            (spec.parse_reopt_timeout, ("10", 0.5, 0, "none"), (-5, "x", True)),
        ):
            for value in good:
                parse(value)
            for value in bad:
                with self.assertRaises(ValueError, msg=(parse.__name__, value)):
                    parse(value)

    def test_unset_means_the_default(self):
        self.assertEqual(spec.parse_update_interval(None), 5)
        self.assertEqual(spec.parse_drop_pct(None), 20.0)
        self.assertEqual(spec.parse_adaptive_window(None), 10)
        self.assertEqual(spec.parse_adaptive_k(None), 2.0)
        self.assertIsNone(spec.parse_max_online_windows(None))
        self.assertIsNone(spec.parse_max_online_windows("null"))
        self.assertEqual(spec.parse_reopt_delay(None), 2)
        self.assertEqual(spec.parse_reopt_timeout(""), 3600.0)

    def test_a_zero_or_null_timeout_means_no_limit(self):
        for value in (0, "0", None, "none", "null", "off"):
            self.assertIsNone(spec.parse_reopt_timeout(value), value)
        self.assertEqual(spec.parse_reopt_timeout("90"), 90.0)

    def test_messages_name_the_flag(self):
        with self.assertRaisesRegex(ValueError, "^--drop_pct:"):
            spec.parse_drop_pct(0)


class TestResolution(unittest.TestCase):

    def test_flag_beats_config_beats_default(self):
        config = {"online_strategy": "threshold", "drop_pct": 15, "update_interval": 7}
        out = spec.resolve_online_settings({"update_interval": "3"}, config)
        self.assertEqual(out["online_scenarios"], ("threshold",))
        self.assertEqual(out["strategy"], "threshold")
        self.assertEqual(out["update_interval"], 3)          # flag
        self.assertEqual(out["drop_pct"], 15.0)              # config
        self.assertEqual(out["adaptive_window"], 10)         # default
        self.assertIsNone(out["max_online_windows"])

    def test_reopt_delay_and_timeout_resolve_like_the_others(self):
        out = spec.resolve_online_settings({}, {})
        self.assertEqual((out["reopt_delay"], out["reopt_timeout"]), (2, 3600.0))
        out = spec.resolve_online_settings({}, {"reopt_delay": 0, "reopt_timeout": 120})
        self.assertEqual((out["reopt_delay"], out["reopt_timeout"]), (0, 120.0))
        out = spec.resolve_online_settings({"reopt_delay": "4", "reopt_timeout": "0"},
                                           {"reopt_delay": 1, "reopt_timeout": 120})
        self.assertEqual((out["reopt_delay"], out["reopt_timeout"]), (4, None))
        # An explicit null in the config means no limit; an omitted key the default.
        self.assertIsNone(spec.resolve_online_settings({}, {"reopt_timeout": None})
                          ["reopt_timeout"])

    def test_strategy_all_and_aliases_are_canonical(self):
        self.assertEqual(spec.resolve_online_settings({"strategy": "all"}, {})["strategy"], "all")
        self.assertEqual(spec.resolve_online_settings({"strategy": "adaptive"}, {})["strategy"],
                         "periodic")

    def test_a_bad_config_value_is_an_error_too(self):
        with self.assertRaisesRegex(ValueError, "--adaptive_window"):
            spec.resolve_online_settings({}, {"adaptive_window": 1})

    def test_the_shipped_config_carries_the_defaults(self):
        with open(Path(spec.__file__).resolve().parent.parent / "Configs" / "config.yml") as f:
            config = yaml.safe_load(f)
        out = spec.resolve_online_settings({}, config)
        self.assertEqual(out["online_scenarios"], (spec.DEFAULT_ONLINE_STRATEGY,))
        self.assertEqual((out["update_interval"], out["drop_pct"], out["adaptive_window"],
                          out["adaptive_k"], out["max_online_windows"], out["reopt_delay"],
                          out["reopt_timeout"]),
                         (spec.DEFAULT_UPDATE_INTERVAL, spec.DEFAULT_DROP_PCT,
                          spec.DEFAULT_ADAPTIVE_WINDOW, spec.DEFAULT_ADAPTIVE_K, None,
                          spec.DEFAULT_REOPT_DELAY, spec.DEFAULT_REOPT_TIMEOUT))
        for key in spec.ONLINE_CONFIG_KEYS.values():
            self.assertIn(key, config)


class TestCommandLine(unittest.TestCase):
    """get_args_from_cmdline end to end, against a temporary config file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = self.root / "config.yml"
        self.write_config()

    def tearDown(self):
        ramses_paths.reset_cache()
        self._tmp.cleanup()

    def write_config(self, **extra):
        base = {"dataset_path": str(self.root / "data"),
                "trained_model_path": str(self.root / "models"),
                "results_path": str(self.root / "elsewhere"),
                "overwrite": False, **extra}
        self.config.write_text(yaml.safe_dump(base))

    def args(self, *flags):
        from Utils.utils import get_args_from_cmdline
        argv = ["app.py", "--config_file_path", str(self.config), *flags]
        with mock.patch.object(sys, "argv", argv):
            return get_args_from_cmdline()

    def test_defaults_without_flags_or_keys(self):
        args = self.args()
        self.assertEqual(args["online_scenarios"], ("periodic",))
        self.assertEqual(args["update_interval"], 5)
        self.assertFalse(args["enable_online"])

    def test_config_keys_and_flags(self):
        self.write_config(online_strategy="all", adaptive_k=1.5, max_online_windows=12)
        args = self.args("--enable_online", "--drop_pct", "30", "--adaptive_window", "4")
        self.assertTrue(args["enable_online"])
        self.assertEqual(args["strategy"], "all")
        self.assertEqual(len(args["online_scenarios"]), 4)
        self.assertEqual((args["drop_pct"], args["adaptive_window"], args["adaptive_k"],
                          args["max_online_windows"]), (30.0, 4, 1.5, 12))

    def test_backward_compatible_strategy_flags(self):
        self.assertEqual(self.args("--strategy", "adaptive")["online_scenarios"], ("periodic",))
        self.assertEqual(self.args("--strategy", "fixed-best")["online_scenarios"], ("fixed",))
        self.assertEqual(self.args("--strategy", "fixed-random")["online_scenarios"],
                         ("fixed_random",))

    def test_reopt_flags(self):
        args = self.args("--reopt_delay", "0", "--reopt_timeout", "45")
        self.assertEqual((args["reopt_delay"], args["reopt_timeout"]), (0, 45.0))
        self.write_config(reopt_delay=3, reopt_timeout=None)
        args = self.args()
        self.assertEqual((args["reopt_delay"], args["reopt_timeout"]), (3, None))

    def test_bad_values_stop_the_cli(self):
        with mock.patch("sys.stderr"):
            for flags in (("--strategy", "sometimes"), ("--update_interval", "0"),
                          ("--drop_pct", "100"), ("--max_online_windows", "0"),
                          ("--reopt_delay", "-1"), ("--reopt_timeout", "-3")):
                with self.assertRaises(SystemExit, msg=flags):
                    self.args(*flags)

    def test_online_labels_flag_and_config_key(self):
        self.assertIsNone(self.args()["online_labels"])
        self.assertEqual(self.args("--online_labels", "real")["online_labels"], "real")
        self.write_config(online_labels="injected")
        self.assertEqual(self.args()["online_labels"], "injected")
        # An explicit `auto` on the command line overrides the config's choice
        # and leaves the decision to the dataset, as the flag's help says.
        self.assertIsNone(self.args("--online_labels", "auto")["online_labels"])
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                self.args("--online_labels", "fake")

    def test_results_go_where_the_given_config_says(self):
        """results_dir used to read Configs/config.yml whatever config was passed."""
        self.args()
        self.assertTrue(ramses_paths.results_dir("online", "ds", "1").startswith(
            str(self.root / "elsewhere")))


class TestFeasibility(unittest.TestCase):
    """The split, the windows and the verdict app.py and the web UI share."""

    def test_split_is_80_20(self):
        self.assertEqual(spec.online_split(1180), (944, 236))
        self.assertEqual(spec.online_split(230), (184, 46))

    def test_window_geometry(self):
        self.assertEqual(spec.online_window_geometry(236, 5), (100, 5))     # the floor
        self.assertEqual(spec.online_window_geometry(5000, 5), (1000, 50))
        self.assertEqual(spec.online_window_geometry(5000, 0), (1000, 50))
        self.assertEqual(spec.online_window_geometry(1000, 100), (100, 5))

    def test_count_windows_matches_the_thompson_windowing(self):
        import numpy as np
        from Model_Selection.Thompson_Sampling import initialize_sliding_windows
        for n, w, step in ((236, 100, 5), (100, 100, 5), (99, 100, 5), (1000, 200, 10)):
            data = np.zeros((1, n))
            if n >= w:
                expected = initialize_sliding_windows(data, np.zeros(n), np.ones((1, n)),
                                                      w, step)[3]
            else:
                expected = 0
            self.assertEqual(spec.count_windows(n, w, step), expected, (n, w, step))

    def test_a_short_series_is_refused_with_the_reason(self):
        out = spec.online_feasibility(230, 5)                  # a SKAB entity
        self.assertFalse(out["ok"])
        self.assertEqual((out["n_windows"], out["online_size"], out["window_size"]),
                         (0, 46, 100))
        self.assertIn("46 samples", out["reason"])
        self.assertIn("at least 100", out["reason"])
        # A larger --iteration does not help: the floor is the floor.
        self.assertFalse(spec.online_feasibility(230, 50)["ok"])

    def test_a_long_enough_series_runs(self):
        out = spec.online_feasibility(1180, 5)                 # UCR 028, downsampling 5
        self.assertTrue(out["ok"])
        self.assertIsNone(out["reason"])
        self.assertEqual((out["n_windows"], out["window_size"], out["step_size"]),
                         (28, 100, 5))
        # The regime-shift stream is 8 copies of the online part.
        self.assertEqual(out["regime_n_windows"], spec.count_windows(8 * 236, 100, 5))
        self.assertTrue(spec.online_feasibility(500, 5)["ok"])     # exactly 100 online

    def test_injection_copies(self):
        self.assertEqual(spec.injection_copies(["spikes"]), 1)
        self.assertEqual(spec.injection_copies(["speedup"]), 4)
        self.assertEqual(spec.injection_copies(["cutoff"]), 2)
        self.assertEqual(spec.injection_copies(spec.REGIME_SHIFT_TYPES), 8)


class TestOnlineLabels(unittest.TestCase):
    """--online_labels: real (no online injection) or injected."""

    def test_parse(self):
        for text in (None, "", "auto", " Default "):
            self.assertIsNone(spec.parse_online_labels(text), text)
        self.assertEqual(spec.parse_online_labels(" REAL "), "real")
        self.assertEqual(spec.parse_online_labels("injected"), "injected")
        for bad in ("synthetic", "true", True, 1):
            with self.assertRaisesRegex(ValueError, "--online_labels", msg=bad):
                spec.parse_online_labels(bad)

    def test_default_is_real_wherever_the_data_has_labels(self):
        self.assertEqual(spec.resolve_online_labels(None, True), "real")
        self.assertEqual(spec.resolve_online_labels(None, False), "injected")
        self.assertEqual(spec.resolve_online_labels("injected", True), "injected")

    def test_real_labels_need_labelled_data(self):
        with self.assertRaisesRegex(ValueError, "SKAB provides no labelled test anomalies"):
            spec.resolve_online_labels("real", False, dataset="skab")

    def test_real_labels_refuse_the_online_regime_shift(self):
        """Real-label mode injects nothing online; the regime shift does."""
        with self.assertRaisesRegex(ValueError, "inject_online_regime"):
            spec.resolve_online_labels("real", True, inject_online_regime=True)
        with self.assertRaisesRegex(ValueError, "so they are the default"):
            spec.resolve_online_labels(None, True, inject_online_regime=True, dataset="tsb-ad-u")
        self.assertEqual(spec.resolve_online_labels("injected", True,
                                                     inject_online_regime=True), "injected")
        self.assertEqual(spec.resolve_online_labels(None, False,
                                                     inject_online_regime=True), "injected")

    def test_which_datasets_have_real_labels(self):
        for key in ("tsb-ad-u", "TSB-AD-M", "anomaly_archive", "smd", "msl", "smap"):
            self.assertTrue(spec.dataset_has_real_labels(key), key)
        # SKAB's loader labels with the changepoint column, not the anomalies.
        for key in ("skab", "apple", "synthetic"):
            self.assertFalse(spec.dataset_has_real_labels(key), key)

    def test_settings_flag_beats_config(self):
        self.assertIsNone(spec.resolve_online_settings({}, {})["online_labels"])
        self.assertEqual(spec.resolve_online_settings(
            {}, {"online_labels": "injected"})["online_labels"], "injected")
        self.assertEqual(spec.resolve_online_settings(
            {"online_labels": "real"}, {"online_labels": "injected"})["online_labels"], "real")
        with self.assertRaisesRegex(ValueError, "--online_labels"):
            spec.resolve_online_settings({}, {"online_labels": "maybe"})


class TestRealLabelFeasibility(unittest.TestCase):
    """online_feasibility with the online part's real labels."""

    def labels(self, total, anomalies):
        out = [0] * total
        for lo, hi in anomalies:
            out[lo:hi] = [1] * (hi - lo)
        return out

    def test_counts_windows_with_an_anomaly_and_scorable_ones(self):
        # 1180 samples: online part 236, windows of 100 every 5 (28 windows).
        online = self.labels(236, [(120, 130)])
        out = spec.online_feasibility(1180, 5, online_labels=online)
        self.assertTrue(out["ok"])
        self.assertEqual(out["label_source"], "real")
        # Window i covers [5i, 5i + 100): it holds 120..129 for 5i + 100 > 120
        # and 5i < 130, i.e. i = 5..25.
        self.assertEqual(out["n_windows_with_anomaly"], 21)
        self.assertEqual(out["n_windows_scorable"], 21)
        self.assertEqual(out["n_anomalous_online"], 10)

    def test_no_labelled_anomaly_online_is_not_feasible(self):
        out = spec.online_feasibility(1180, 5, online_labels=[0] * 236)
        self.assertFalse(out["ok"])
        self.assertEqual(out["n_windows_with_anomaly"], 0)
        self.assertIn("no labelled anomaly in the online part", out["reason"].lower())
        self.assertIn("nothing to score", out["reason"])

    def test_windows_anomalous_throughout_are_not_scorable(self):
        out = spec.online_feasibility(1180, 5, online_labels=[1] * 236)
        self.assertFalse(out["ok"])
        self.assertEqual((out["n_windows_with_anomaly"], out["n_windows_scorable"]), (28, 0))

    def test_injected_mode_has_no_label_counts(self):
        out = spec.online_feasibility(1180, 5)
        self.assertEqual(out["label_source"], "injected")
        self.assertIsNone(out["n_windows_with_anomaly"])

    def test_window_label_counts_cap(self):
        counts = spec.window_label_counts(self.labels(236, [(0, 3)]), 100, 5, n_windows=4)
        self.assertEqual((counts["n_windows"], counts["n_windows_with_anomaly"]), (4, 1))

    def test_a_too_short_series_keeps_its_length_reason(self):
        out = spec.online_feasibility(230, 5, online_labels=[1] + [0] * 45)
        self.assertFalse(out["ok"])
        self.assertIn("46 samples", out["reason"])


class TestRedirectedResults(unittest.TestCase):

    def tearDown(self):
        ramses_paths.reset_cache()

    def test_only_the_calling_thread_is_redirected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ramses_paths.set_results_root(os.path.join(tmp, "main"))
            inside, outside = {}, {}
            entered, release = threading.Event(), threading.Event()

            def worker():
                with ramses_paths.redirected_results(os.path.join(tmp, "scratch")):
                    inside["dir"] = ramses_paths.results_dir("Thomposon", "ds", "1")
                    entered.set()
                    release.wait(5)
                inside["after"] = ramses_paths.results_dir("x")

            t = threading.Thread(target=worker)
            t.start()
            entered.wait(5)
            outside["dir"] = ramses_paths.results_dir("online", "ds", "1")
            release.set()
            t.join(5)
            self.assertTrue(inside["dir"].startswith(os.path.join(tmp, "scratch")))
            self.assertTrue(outside["dir"].startswith(os.path.join(tmp, "main")))
            self.assertTrue(inside["after"].startswith(os.path.join(tmp, "main")))


if __name__ == "__main__":
    unittest.main()
