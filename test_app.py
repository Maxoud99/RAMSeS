"""
Tests for app.py: the GAN robustness stage fails soft in both pipelines.

The real `app` module is imported, with real TensorFlow, rank aggregation and
explanation layer. Only the stage functions are replaced: GA, Thompson,
Off-by and Monte Carlo return fixed rankings, and `run_Gan` raises the way
TensorFlow does when it rejects torch's cuDNN. What is under test is what the
pipeline does next. It must log one warning the web UI shows, mark the stage
skipped, and finish the other stages and the rank aggregation exactly as
--skip_gan would.
"""

import json
import logging
import os
import pathlib
import shutil
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import app
from Explainability.ir import NOT_AVAILABLE, assemble_global_ir
from Model_Selection.rank_aggregation import enhanced_markov_chain_rank_aggregator_text
from Utils import paths as ramses_paths
from WebUI import artifacts, jobs, markers
from WebUI import paths as webui_paths

DETECTORS = ["LOF_1", "CBLOF_1", "MD_1"]


def _scored(names):
    """Stage output in the (name, [{'f1', 'pr_auc'}]) shape the report reads."""
    return [(n, [{"f1": 0.9 - 0.1 * i, "pr_auc": 0.8 - 0.1 * i}])
            for i, n in enumerate(names)]


BORDERLINE_F1 = ["MD_1", "LOF_1", "CBLOF_1"]
BORDERLINE_PR = ["LOF_1", "MD_1", "CBLOF_1"]
MC_F1 = ["CBLOF_1", "MD_1", "LOF_1"]
MC_PR = ["MD_1", "CBLOF_1", "LOF_1"]
GAN_F1 = ["CBLOF_1", "LOF_1", "MD_1"]
GAN_PR = ["CBLOF_1", "MD_1", "LOF_1"]
THOMPSON = ["LOF_1", "MD_1", "CBLOF_1"]

# The first line is what TensorFlow 2.18 logs when torch's cuDNN 9.1 is the
# one in the process. The second stands in for the rest of its long message.
TF_ERROR = ("Loaded runtime CuDNN library: 9.1.0 but source was compiled with: "
            "9.3.0.\nCuDNN library needs to have matching major version and "
            "equal or higher minor version.")


def _gan_raises(*args, **kwargs):
    raise RuntimeError(TF_ERROR)


def _gan_ok(*args, **kwargs):
    return _scored(GAN_F1), _scored(GAN_PR), list(GAN_F1), list(GAN_PR), []


def _test_data():
    entity = types.SimpleNamespace(Y=np.zeros((1, 60)), labels=np.zeros((1, 60)))
    return types.SimpleNamespace(entities=[entity])


class _PipelineCase(unittest.TestCase):
    """Stubs the stages around the GAN and points results at a temp dir."""

    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        self.results = pathlib.Path(tmp) / "results"
        ramses_paths._RESULTS_ROOT = self.results
        self.addCleanup(ramses_paths.reset_cache)
        ga = (["LOF_1", "MD_1"], 0.5, 0.4, 0.45, None, None, None, None, None, "rf", None)
        patches = mock.patch.multiple(
            app,
            # The 12th item (the fitted meta-model) only when asked for, as
            # the real genetic_algorithm does.
            genetic_algorithm=lambda *a, **k: ga + ((None,) if k.get("return_meta_model") else ()),
            run_linear_thompson_sampling=lambda *a, **k: list(THOMPSON),
            run_off_by_threshold=lambda *a, **k: (
                _scored(BORDERLINE_F1), _scored(BORDERLINE_PR),
                list(BORDERLINE_F1), list(BORDERLINE_PR), []),
            run_monte_carlo_simulation=lambda *a, **k: (list(MC_F1), list(MC_PR), []),
            vus_window=lambda *a, **k: 8,
        )
        patches.start()
        self.addCleanup(patches.stop)
        models = mock.patch.object(app, "trained_models",
                                   {d: object() for d in DETECTORS}, create=True)
        models.start()
        self.addCleanup(models.stop)

    def run_pipeline(self, which, gan, **kw):
        """Run pipeline 1 (sequential) or 2 (parallel); returns (result, log records)."""
        with mock.patch.object(app, "run_Gan", gan), \
                self.assertLogs(app.logger, level="INFO") as logs:
            if which == 1:
                result = app.run_model_selection_algorithms_1(
                    None, _test_data(), "skab", "1", 0, model_list=list(DETECTORS), **kw)
            else:
                result = app.run_model_selection_algorithms_2(
                    None, _test_data(), "skab", "1", 0, app.trained_models,
                    model_list=list(DETECTORS), **kw)
        return result, logs.records

    @staticmethod
    def as_log_lines(records):
        """The lines the web UI reads: stdlib's basicConfig format, traceback included."""
        fmt = logging.Formatter(logging.BASIC_FORMAT)
        return [line for r in records for line in fmt.format(r).splitlines()]

    def gan_warnings(self, records):
        return [r for r in records
                if r.levelno == logging.WARNING and app.GAN_FAILED_WARNING in r.getMessage()]

    def robust_without_gan(self):
        return enhanced_markov_chain_rank_aggregator_text(
            [BORDERLINE_F1, BORDERLINE_PR, MC_F1, MC_PR])[1]


class TestFailSoftWrapper(unittest.TestCase):

    def test_exception_becomes_none_and_one_warning(self):
        with mock.patch.object(app, "run_Gan", _gan_raises), \
                self.assertLogs(app.logger, level="WARNING") as logs:
            self.assertIsNone(app._run_gan_or_skip("data", {}, DETECTORS, "skab", "1"))
        self.assertEqual(len(logs.records), 1)
        message = logs.records[0].getMessage()
        self.assertIn(app.GAN_FAILED_WARNING, message)
        self.assertIn("RuntimeError: Loaded runtime CuDNN library: 9.1.0", message)
        # One line: TensorFlow's long tail goes to the traceback, not the warning.
        self.assertNotIn("\n", message)
        self.assertIsNotNone(logs.records[0].exc_info)

    def test_success_passes_the_result_through(self):
        with mock.patch.object(app, "run_Gan", _gan_ok):
            self.assertEqual(app._run_gan_or_skip()[2], GAN_F1)

    def test_keyboard_interrupt_still_stops_the_run(self):
        def interrupted(*a, **k):
            raise KeyboardInterrupt
        with mock.patch.object(app, "run_Gan", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                app._run_gan_or_skip()


class TestSequentialPipeline(_PipelineCase):

    def test_full_run_finishes_without_the_gan_ranking(self):
        result, records = self.run_pipeline(1, _gan_raises)
        extra = result[10]
        self.assertEqual(extra["gan"]["f1_names"], [])
        self.assertEqual(extra["gan"]["pr_auc_names"], [])
        self.assertEqual(extra["gan"]["best_model"], "N/A")
        self.assertEqual(extra["borderline"]["f1_names"], BORDERLINE_F1)
        self.assertEqual(extra["monte_carlo"]["f1_names"], MC_F1)
        # Rank aggregation ran, on the four rankings that exist.
        self.assertEqual(result[1], self.robust_without_gan())
        self.assertTrue(result[2], "final aggregation is empty")
        self.assertEqual(len(self.gan_warnings(records)), 1)
        self.assertGreater(extra["timing"]["3_GAN_Robustness"], 0.0)

    def test_failure_is_handled_exactly_like_skip_gan(self):
        failed, _ = self.run_pipeline(1, _gan_raises)
        skipped, records = self.run_pipeline(1, _gan_ok, skip_gan=True)
        self.assertEqual(self.gan_warnings(records), [])
        self.assertEqual(failed[1], skipped[1])
        self.assertEqual(failed[2], skipped[2])
        self.assertEqual({k: v for k, v in failed[10]["gan"].items()},
                         {k: v for k, v in skipped[10]["gan"].items()})

    def test_successful_gan_still_feeds_the_consensus(self):
        result, records = self.run_pipeline(1, _gan_ok)
        self.assertEqual(result[10]["gan"]["f1_names"], GAN_F1)
        self.assertEqual(self.gan_warnings(records), [])
        self.assertEqual(result[1], enhanced_markov_chain_rank_aggregator_text(
            [GAN_F1, GAN_PR, BORDERLINE_F1, BORDERLINE_PR, MC_F1, MC_PR])[1])

    def test_partial_robustness_run_keeps_off_by_and_monte_carlo(self):
        result, records = self.run_pipeline(
            1, _gan_raises, stages={"gan", "offby", "montecarlo"})
        self.assertEqual(result[10]["gan"]["f1_names"], [])
        self.assertEqual(result[10]["borderline"]["pr_auc_names"], BORDERLINE_PR)
        self.assertEqual(result[10]["monte_carlo"]["pr_auc_names"], MC_PR)
        self.assertEqual(len(self.gan_warnings(records)), 1)

    def test_web_ui_shows_the_warning_and_a_skipped_stage(self):
        """Feed the run's real log lines through the web UI's own Job."""
        _, records = self.run_pipeline(1, _gan_raises)
        job = jobs.Job("t", [], {})
        for line in self.as_log_lines(records):
            job._record(line)
        self.assertEqual([w["code"] for w in job.warnings], ["gan_failed"])
        self.assertEqual(job.stages["gan"]["status"], "skipped")
        # The traceback that follows the warning must not read as a crash.
        self.assertFalse(job.saw_fatal)
        for key in ("offby", "montecarlo", "aggregation"):
            self.assertEqual(job.stages[key]["status"], "done", key)

    def test_explanation_leaves_the_failed_gan_out(self):
        self.run_pipeline(1, _gan_raises, explain=True)
        ir_dir = self.results / "explanations_ir" / "skab" / "1"
        with open(ir_dir / "ir_rank_aggregation_robust_0.json") as fh:
            robust = json.load(fh)
        self.assertEqual(robust["output"]["sources"], sorted(
            ["Borderline_F1", "Borderline_PR_AUC", "MonteCarlo_F1", "MonteCarlo_PR_AUC"]))
        self.assertTrue((ir_dir / "ir_rank_aggregation_final_0.json").exists())


class TestParallelPipeline(_PipelineCase):

    def test_full_run_finishes_without_the_gan_ranking(self):
        result, records = self.run_pipeline(2, _gan_raises)
        self.assertEqual(result[10]["gan"]["f1_names"], [])
        self.assertEqual(result[10]["gan"]["best_model"], "N/A")
        self.assertEqual(result[1], self.robust_without_gan())
        self.assertTrue(result[2])
        self.assertEqual(len(self.gan_warnings(records)), 1)
        self.assertTrue(any("GAN: SKIPPED (the stage failed" in r.getMessage()
                            for r in records))

    def test_parallel_matches_sequential_after_a_gan_failure(self):
        sequential, _ = self.run_pipeline(1, _gan_raises)
        parallel, _ = self.run_pipeline(2, _gan_raises)
        self.assertEqual(sequential[1], parallel[1])
        self.assertEqual(sequential[2], parallel[2])

    def test_successful_parallel_run_reaches_rank_aggregation(self):
        """The off-by result used to be unpacked into seven names, so every
        full --parallel run raised ValueError before rank aggregation."""
        result, _ = self.run_pipeline(2, _gan_ok)
        self.assertEqual(result[10]["gan"]["f1_names"], GAN_F1)
        self.assertEqual(result[10]["borderline"]["f1_names"], BORDERLINE_F1)
        self.assertEqual(result[10]["borderline"]["best_model"], BORDERLINE_F1[0])
        self.assertTrue(result[2])

    def test_explanation_leaves_the_failed_gan_out(self):
        self.run_pipeline(2, _gan_raises, explain=True)
        with open(self.results / "explanations_ir" / "skab" / "1"
                  / "ir_rank_aggregation_robust_0.json") as fh:
            robust = json.load(fh)
        self.assertEqual(robust["output"]["n_sources"], 4)
        self.assertFalse(any(s.startswith("GAN") for s in robust["output"]["sources"]))


EARLIER_PICK = "OLD_RUN_PICK"


class TestEarlierGanExplanation(_PipelineCase):
    """An --explain run with no GAN result must not present an earlier run's
    GAN explanation as its own.

    Before the fix, the earlier run's ir_gan.json was still on disk after a
    GAN failure. The global IR listed it as an "ok" stage with that run's top
    pick, and the result page showed its card next to the warning that the
    GAN test had failed.
    """

    def setUp(self):
        super().setUp()
        self.ir_gan = self.results / "explanations_ir" / "skab" / "1" / "ir_gan.json"
        self.nl_gan = self.results / "explanations_nl" / "skab" / "1" / "nl_gan.txt"

    def write_earlier_explanation(self):
        """The pair an earlier, successful --explain run leaves behind."""
        self.ir_gan.parent.mkdir(parents=True, exist_ok=True)
        self.nl_gan.parent.mkdir(parents=True, exist_ok=True)
        sentence = f"The GAN test ranked {EARLIER_PICK} first."
        self.ir_gan.write_text(json.dumps({
            "stage": "gan", "dataset": "skab", "entity": "1",
            "output": {"top_pick": EARLIER_PICK},
            "evidence": [{"id": "gn.output.top", "type": "stage_output",
                          "subject": EARLIER_PICK, "value": {}, "text": sentence}],
            "required_atom_ids": ["gn.output.top"], "caveats": []}))
        self.nl_gan.write_text(sentence + "\n")

    def earlier_explanation_exists(self):
        return self.ir_gan.exists(), self.nl_gan.exists()

    def global_ir(self, result):
        """The global IR as run_app assembles it, from this run's GAN result."""
        gan = result[10]["gan"]
        path = assemble_global_ir(
            {"gan_robustness": {"f1_names": gan["f1_names"],
                                "pr_auc_names": gan["pr_auc_names"]},
             "final_decision": {}},
            "skab", "1", app.OFFLINE_ITERATION)
        with open(path) as fh:
            return json.load(fh)

    def test_a_failed_gan_removes_it_in_both_pipelines(self):
        for which in (1, 2):
            with self.subTest(pipeline=which):
                self.write_earlier_explanation()
                result, records = self.run_pipeline(which, _gan_raises, explain=True)
                self.assertEqual(self.earlier_explanation_exists(), (False, False))
                self.assertEqual(len(self.gan_warnings(records)), 1)
                g = self.global_ir(result)
                self.assertEqual(g["stages"]["gan"]["status"], NOT_AVAILABLE)
                self.assertNotIn(EARLIER_PICK, json.dumps(g))

    def test_skip_gan_removes_it_too(self):
        for which in (1, 2):
            with self.subTest(pipeline=which):
                self.write_earlier_explanation()
                result, _ = self.run_pipeline(which, _gan_ok, skip_gan=True, explain=True)
                self.assertEqual(self.earlier_explanation_exists(), (False, False))
                self.assertEqual(self.global_ir(result)["stages"]["gan"]["status"],
                                 NOT_AVAILABLE)

    def test_result_page_shows_no_gan_card_after_a_failure(self):
        """Through the web UI's own payload builder, which reads ir_gan.json and
        nl_gan.txt directly rather than through the global IR."""
        self.write_earlier_explanation()
        result, _ = self.run_pipeline(1, _gan_raises, explain=True)
        self.global_ir(result)
        with mock.patch.multiple(
                webui_paths, RESULTS=self.results,
                EXPLANATIONS_IR=self.results / "explanations_ir",
                EXPLANATIONS_NL=self.results / "explanations_nl",
                COMPREHENSIVE=self.results / "comprehensive"):
            payload = artifacts.build_payload("skab", "1")
        self.assertNotIn("gan", [s["key"] for s in payload["stages"]])
        missing = {m["key"]: m for m in payload["missing_stages"]}
        self.assertIn("failed or was skipped", missing["gan"]["note"])
        self.assertNotIn(EARLIER_PICK, json.dumps(payload))

    def test_kept_when_this_run_did_not_replace_it(self):
        """No --explain, or a partial run that leaves GAN out of --stages:
        these runs write no GAN explanation, and like every stage they did not
        run they keep the earlier files. A successful GAN keeps them as well;
        the stub writes nothing here, so they would only go if deleted."""
        cases = (
            ("no --explain", _gan_raises, {}),
            ("partial run without gan", _gan_raises,
             {"stages": {"offby", "montecarlo"}, "explain": True}),
            ("successful gan", _gan_ok, {"explain": True}),
        )
        for label, gan, kw in cases:
            with self.subTest(label):
                self.write_earlier_explanation()
                self.run_pipeline(1, gan, **kw)
                self.assertEqual(self.earlier_explanation_exists(), (True, True))

    def test_a_file_that_cannot_be_removed_does_not_stop_the_run(self):
        self.write_earlier_explanation()
        with mock.patch.object(app.os, "remove",
                               side_effect=PermissionError("read-only")):
            result, records = self.run_pipeline(1, _gan_raises, explain=True)
        self.assertTrue(result[2], "final aggregation is empty")
        self.assertTrue(any("Could not remove the earlier run's GAN explanation"
                            in r.getMessage() for r in records
                            if r.levelno == logging.WARNING))
        # The global IR does not rely on the removal: it reads this run's
        # empty GAN ranking and ignores the file still on disk.
        self.assertEqual(self.earlier_explanation_exists(), (True, True))
        self.assertEqual(self.global_ir(result)["stages"]["gan"]["status"],
                         NOT_AVAILABLE)


class TestRealLabelThompson(_PipelineCase):
    """A real-label re-optimisation asks Thompson Sampling to skip one-class
    windows; every other call keeps the offline behaviour."""

    def thompson_kwargs(self, **kw):
        seen = {}

        def spy(*a, **k):
            seen.update(k)
            return list(THOMPSON)

        with mock.patch.object(app, "run_linear_thompson_sampling", spy):
            self.run_pipeline(1, _gan_ok, skip_gan=True, **kw)
        return seen

    def test_default_is_off(self):
        self.assertFalse(self.thompson_kwargs()["skip_one_class_windows"])

    def test_real_labels_turn_it_on(self):
        self.assertTrue(self.thompson_kwargs(real_labels=True)["skip_one_class_windows"])


def _run_app_args(**over):
    args = {"dataset": "skab", "entity": "1", "parallel": False, "enable_online": True,
            "iteration": 5, "strategy": "all", "inject_online_regime": False,
            "max_online_windows": None, "explain": False, "stages": set(app.ALL_STAGES),
            "skip_gan": True, "anomaly_types": ["spikes"], "anomaly_rate": None,
            "decision_metrics": app.DEFAULT_DECISION_METRICS, "detectors": list(DETECTORS),
            "dataset_path": "/nonexistent", "online_labels": None,
            "downsampling": None, "min_length": None}
    args.update(over)
    return args


class TestOnlineLabelResolution(unittest.TestCase):
    """run_app settles --online_labels before it loads or trains anything."""

    def run_app(self, **over):
        args = _run_app_args(**over)
        loads = mock.Mock(side_effect=RuntimeError("stop after the label check"))
        with mock.patch.object(app, "get_args_from_cmdline", return_value=args), \
                mock.patch.object(app, "load_data", loads), \
                self.assertLogs(app.logger, level="INFO") as logs:
            try:
                app.run_app(list(DETECTORS), list(DETECTORS))
            except RuntimeError:
                pass
        return args, loads, [r.getMessage() for r in logs.records]

    def test_real_labels_on_a_dataset_without_them_stop_the_run(self):
        args, loads, lines = self.run_app(dataset="skab", online_labels="real")
        loads.assert_not_called()
        self.assertTrue(any("SKAB provides no labelled test anomalies" in l for l in lines))

    def test_real_labels_and_the_regime_shift_stop_the_run(self):
        _, loads, lines = self.run_app(dataset="smd", entity="machine-1-1",
                                       inject_online_regime=True)
        loads.assert_not_called()
        self.assertTrue(any("inject_online_regime" in l for l in lines))

    def test_the_default_follows_the_dataset(self):
        args, loads, lines = self.run_app(dataset="smd", entity="machine-1-1")
        self.assertEqual(args["online_label_source"], "real")
        loads.assert_called()
        self.assertTrue(any("online_labels=real" in l for l in lines))
        args, _, _ = self.run_app(dataset="skab")
        self.assertEqual(args["online_label_source"], "injected")
        args, _, _ = self.run_app(dataset="smd", online_labels="injected")
        self.assertEqual(args["online_label_source"], "injected")

    def test_not_resolved_without_the_online_phase(self):
        args, loads, _ = self.run_app(dataset="skab", online_labels="real",
                                      enable_online=False)
        self.assertIsNone(args["online_label_source"])
        loads.assert_called()

    def test_a_tsb_ad_entity_is_named_by_its_id(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "TSB-AD-U"))
            pathlib.Path(root, "TSB-AD-U", "007_YAHOO_id_7_Synthetic_tr_400_1st_600.csv") \
                .write_text("Data,Label\n1.0,0\n")
            args, _, lines = self.run_app(dataset="tsb-ad-u", dataset_path=root,
                                          entity="007_YAHOO_id_7_Synthetic_tr_400_1st_600.csv")
        self.assertEqual(args["entity"], "007_YAHOO_id_7_Synthetic")
        self.assertTrue(any("entity=007_YAHOO_id_7_Synthetic," in l for l in lines))


class TestRealLabelOnlinePhase(unittest.TestCase):
    """run_online_phase with real labels: no injection, the raw test slices
    with their true labels, re-optimisations fed the true labels seen."""

    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        ramses_paths._RESULTS_ROOT = pathlib.Path(tmp) / "results"
        self.addCleanup(ramses_paths.reset_cache)
        rng = np.random.default_rng(0)
        self.off_Y = rng.normal(size=(1, 944))
        self.off_labels = np.zeros(944)
        self.off_labels[50:55] = 1                   # slides out after 55 online samples
        self.on_Y = rng.normal(size=(1, 236)) + 3.0
        self.on_labels = np.zeros(236)
        self.on_labels[150:156] = 1

    def run_phase(self, label_source="real"):
        from Model_Selection import online_reopt, online_scoring
        from Model_Selection.online_phase import Selection
        captured = {}

        def fake_run_online(scenarios, params, windows, initial_selection, scorer,
                            start_reopt, writer, log=None, on_window=None):
            captured["on_window"] = on_window
            captured["windows"] = windows
            captured["summary"] = dict(writer.summary)
            captured["starts"] = {}
            for index in (5, 27):
                try:
                    captured["starts"][index] = start_reopt(windows[index],
                                                            initial_selection("periodic"),
                                                            "periodic")
                except ValueError as e:
                    captured["starts"][index] = e
            writer.finish("complete")
            return []

        class FakeProcess:
            def __init__(self, target, kwargs, workdir, seed, timeout=None, on_success=None):
                self.kwargs = kwargs

        def no_injection(*a, **k):
            raise AssertionError("real-label mode must not inject online")

        args = {"online_scenarios": ("periodic",), "update_interval": 5, "drop_pct": 20.0,
                "adaptive_window": 10, "adaptive_k": 2.0, "reopt_delay": 2,
                "reopt_timeout": None, "random_seed": 13, "max_online_windows": None,
                "inject_online_regime": False, "strategy": "periodic",
                "online_label_source": label_source, "labels_available": True,
                "decision_metrics": app.DEFAULT_DECISION_METRICS}
        selection = Selection(["LOF_1", "MD_1"], "LOF_1", meta_model=object())
        feasibility = app.online_feasibility(1180, 5, online_labels=(
            self.on_labels if label_source == "real" else None))
        patches = [mock.patch("Model_Selection.online_phase.run_online", fake_run_online),
                   mock.patch.object(online_scoring, "fit_selection",
                                     lambda *a, **k: selection),
                   mock.patch.object(online_reopt, "ProcessReopt", FakeProcess)]
        if label_source == "real":
            patches.append(mock.patch.object(online_scoring, "build_online_stream",
                                             no_injection))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        app.run_online_phase(
            args=args, dataset="tsb-ad-u", entity="007_X", train_data=None,
            trained_models={"LOF_1": object(), "MD_1": object()},
            loaded_model_names=["LOF_1", "MD_1"], models_dir="/nowhere",
            offline_copies=[], offline_raw=(self.off_Y, self.off_labels,
                                            np.ones_like(self.off_Y)),
            stream_raw=(self.on_Y, self.on_labels, np.ones_like(self.on_Y)),
            feasibility=feasibility, window_size=100, step_size=5, total_samples=1180,
            offline_size=944, online_size=236, anomaly_list=["spikes"], anomaly_rate=None,
            best_ensemble=["LOF_1", "MD_1"], best_single_model="LOF_1",
            meta_model_type="rf", ga_meta_model=None, individual_predictions=None,
            base_model_predictions_train=None, y_true_train=None, offline_summary={},
            vus_win=None, iteration=5, offline_time=1.0)
        return captured

    def test_windows_are_the_raw_test_slices_with_true_labels(self):
        out = self.run_phase()
        self.assertEqual(out["summary"]["label_source"], "real")
        self.assertIsNone(out["summary"]["injection"])
        for w in out["windows"]:
            lo, hi = w.online_start, w.online_end
            np.testing.assert_array_equal(w.data.entities[0].Y, self.on_Y[:, lo:hi])
            np.testing.assert_array_equal(np.asarray(w.data.entities[0].labels).ravel(),
                                          self.on_labels[lo:hi])
        scorable = sum(1 for w in out["windows"] if w.scorable)
        self.assertEqual(out["summary"]["n_windows_scorable"], scorable)
        self.assertEqual(scorable, out["summary"]["feasibility"]["n_windows_scorable"])

    def test_reopt_inputs_are_the_true_labels_seen_so_far(self):
        out = self.run_phase()
        # Window 5 ends at online sample 125: the offline anomaly (50..54)
        # has slid out and the online one (150..155) is not seen yet.
        self.assertIsInstance(out["starts"][5], ValueError)
        self.assertIn("no labelled anomaly", str(out["starts"][5]))
        kwargs = out["starts"][27].kwargs
        self.assertEqual(kwargs["label_source"], "real")
        seen = out["windows"][27].online_end                 # 235
        sliding = kwargs["sliding"].entities[0]
        expected_Y = np.concatenate([self.off_Y, self.on_Y[:, :seen]], axis=1)[:, -944:]
        expected_labels = np.concatenate([self.off_labels, self.on_labels[:seen]])[-944:]
        np.testing.assert_array_equal(sliding.Y, expected_Y)
        np.testing.assert_array_equal(np.asarray(sliding.labels).ravel(), expected_labels)
        np.testing.assert_array_equal(kwargs["sliding_raw"].entities[0].Y, expected_Y)

    def test_the_live_monitor_stream_is_written(self):
        import json as _json
        out = self.run_phase()
        self.assertIsNotNone(out["on_window"])
        directory = pathlib.Path(app.results_dir("online", "tsb-ad-u", "007_X"))
        meta = _json.loads((directory / "stream.json").read_text())
        self.assertEqual((meta["label_source"], meta["offline_size"], meta["total_length"]),
                         ("real", 944, 1180))
        self.assertTrue(meta["has_real_labels"])
        self.assertFalse(meta["has_injected_labels"])
        with np.load(directory / "stream.npz") as z:
            np.testing.assert_array_equal(z["online"][0], self.on_Y.astype(np.float32))
            np.testing.assert_array_equal(z["real_labels"],
                                          np.concatenate([self.off_labels, self.on_labels]) > 0)

    def test_injected_mode_still_injects(self):
        out = self.run_phase("injected")
        self.assertEqual(out["summary"]["label_source"], "injected")
        self.assertEqual(out["summary"]["injection"]["n_copies"], 1)


if __name__ == "__main__":
    unittest.main()
