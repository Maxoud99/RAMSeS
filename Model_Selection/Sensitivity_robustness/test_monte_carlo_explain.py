"""
Standalone unit tests for the Monte Carlo robustness-explainability layer.
Mocks the module's heavy imports (Metrics.metrics, Utils.model_selection_utils,
loguru) and loads Monte_Carlo_Simulation.py by file path. numpy / scipy /
matplotlib are real; sklearn is intentionally NOT mocked so the surrogate
integration test can `importorskip` it (runs where sklearn is installed).
"""

import os
import sys
import types
import importlib.util
import tempfile
import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")


# ── Mock the module's heavy imports ─────────────────────────────────────────
def _mk(name):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


for _n in ("Metrics", "Metrics.metrics", "Utils", "Utils.model_selection_utils", "loguru"):
    _mk(_n)
# See test_off_by_explain: `Utils.pipeline_spec` is stdlib-only and is wanted for
# real, so the stand-in gets a __path__ while the mocked sibling still shadows.
sys.modules["Utils"].__path__ = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "Utils")]
sys.modules["Metrics.metrics"].range_based_precision_recall_f1_auc = lambda *a, **k: (0, 0, 0.5, 0.5, None)
sys.modules["Metrics.metrics"].prauc = lambda *a, **k: 0.5
sys.modules["Metrics.metrics"].f1_score = lambda *a, **k: (0.5,) * 7
sys.modules["Metrics.metrics"].f1_soft_score = lambda *a, **k: (0.5,) * 7
sys.modules["Metrics.metrics"].rank_key = lambda v: v
sys.modules["Metrics.metrics"].vus_score = lambda *a, **k: 0.5
sys.modules["Metrics.metrics"].vus_window = lambda *a, **k: 8
sys.modules["Utils.model_selection_utils"].evaluate_model = lambda *a, **k: {}
sys.modules["Utils.model_selection_utils"].ScoringTimeout = type(
    "ScoringTimeout", (Exception,), {})


class _Logger:
    def __getattr__(self, _):
        return lambda *a, **k: None


sys.modules["loguru"].logger = _Logger()

# ── Load Monte_Carlo_Simulation.py by path ──────────────────────────────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "mc_sim", os.path.join(_THIS, "Monte_Carlo_Simulation.py"))
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _fake_data(n=60, two_class=True):
    labels = np.zeros((1, n))
    if two_class:
        labels[0, :max(1, n // 6)] = 1
    ent = types.SimpleNamespace(labels=labels, Y=np.zeros((3, n)))
    return types.SimpleNamespace(entities=[ent])


# ════════════════════════════════════════════════════════════════════════════
# 1.  monte_carlo_noise_sweep
# ════════════════════════════════════════════════════════════════════════════

class TestSweep(unittest.TestCase):

    def test_shapes_and_grid(self):
        models = ["A", "B"]
        levels = np.linspace(0.0, 0.5, 6)

        def ev(m, level):
            return (0.9 - level if m == "A" else 0.2 + level, 0.5)

        out = mc.monte_carlo_noise_sweep(_fake_data(), {}, models,
                                         noise_levels=levels, repeats=4, evaluate_fn=ev)
        self.assertIsNotNone(out)
        self.assertEqual(out["F1"].shape, (6 * 4, 2))
        self.assertEqual(out["noise"].shape, (6 * 4,))
        self.assertTrue(np.allclose(np.unique(out["noise"]), levels))

    def test_infeasible_returns_none(self):
        # too small
        self.assertIsNone(mc.monte_carlo_noise_sweep(_fake_data(n=20), {}, ["A"],
                                                     evaluate_fn=lambda m, l: (0.5, 0.5)))
        # single class
        self.assertIsNone(mc.monte_carlo_noise_sweep(_fake_data(two_class=False), {}, ["A"],
                                                     evaluate_fn=lambda m, l: (0.5, 0.5)))

    def test_nan_on_model_failure(self):
        def ev(m, level):
            if m == "bad":
                raise RuntimeError("boom")
            return (0.7, 0.6)
        out = mc.monte_carlo_noise_sweep(_fake_data(), {}, ["good", "bad"],
                                         noise_levels=[0.0, 0.1], repeats=2, evaluate_fn=ev)
        self.assertTrue(np.all(np.isnan(out["F1"][:, 1])))      # bad column all NaN
        self.assertFalse(np.any(np.isnan(out["F1"][:, 0])))     # good column finite

    def test_f1_fixed_two_tuple_mirrors_f1(self):
        # A 2-tuple evaluate_fn carries no fixed value → F1_fixed == F1.
        out = mc.monte_carlo_noise_sweep(_fake_data(), {}, ["A"],
                                         noise_levels=[0.0, 0.25], repeats=2,
                                         evaluate_fn=lambda m, l: (0.9 - l, 0.5))
        self.assertEqual(out["F1_fixed"].shape, out["F1"].shape)
        self.assertTrue(np.allclose(out["F1_fixed"], out["F1"]))

    def test_f1_fixed_three_tuple_used(self):
        # A 3-tuple evaluate_fn routes its 3rd element into F1_fixed.
        out = mc.monte_carlo_noise_sweep(_fake_data(), {}, ["A"],
                                         noise_levels=[0.0, 0.25], repeats=1,
                                         evaluate_fn=lambda m, l: (0.9 - l, 0.5, 0.3 - l))
        self.assertTrue(np.allclose(out["F1"].flatten(), [0.9, 0.65]))
        self.assertTrue(np.allclose(out["F1_fixed"].flatten(), [0.3, 0.05]))

    def test_grid_sorted_ascending(self):
        out = mc.monte_carlo_noise_sweep(_fake_data(), {}, ["A"],
                                         noise_levels=[0.5, 0.0, 0.25], repeats=1,
                                         evaluate_fn=lambda m, l: (0.5, 0.5))
        self.assertTrue(np.allclose(out["grid_levels"], [0.0, 0.25, 0.5]))


class TestFixedThresholdHelpers(unittest.TestCase):

    def test_f1_at_cut_nan(self):
        self.assertTrue(np.isnan(mc._f1_at_cut(np.array([0, 1]), np.array([0.1, 0.9]), float('nan'))))

    def test_best_f1_and_cut_returns_in_range_cut(self):
        # f1_score is mocked to a constant, so the first quantile wins; the cut must
        # be a real score-range value (not NaN) and best_f1 the mocked constant.
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])
        best_f1, cut = mc._best_f1_and_cut(y_true, y_scores)
        self.assertFalse(np.isnan(cut))
        self.assertTrue(y_scores.min() <= cut <= y_scores.max())
        self.assertGreater(best_f1, 0.0)


# ════════════════════════════════════════════════════════════════════════════
# 2.  compute_noise_curves
# ════════════════════════════════════════════════════════════════════════════

class TestNoiseCurves(unittest.TestCase):

    def test_crossover_winregions_breakdown(self):
        models = ["A", "B"]
        grid = np.array([0.0, 0.2, 0.4, 0.6])
        # A leads at low noise then collapses below 0.5; B stays robust (always ≥0.5).
        # scores: A = [0.9, 0.7, 0.3, 0.1] ; B = [0.55, 0.65, 0.6, 0.8]
        noise = grid.copy()
        score = np.array([[0.9, 0.55], [0.7, 0.65], [0.3, 0.6], [0.1, 0.8]])
        cur = mc.compute_noise_curves(noise, grid, score, models, breakdown_threshold=0.5)
        # winner flips A→B between 0.2 and 0.4.
        self.assertEqual(len(cur["crossovers"]), 1)
        self.assertEqual(cur["crossovers"][0]["from_model"], "A")
        self.assertEqual(cur["crossovers"][0]["to_model"], "B")
        self.assertAlmostEqual(cur["crossovers"][0]["noise"], 0.4)
        # A's win-region covers the low-noise band.
        self.assertEqual(cur["win_regions"]["A"], [(0.0, 0.2)])
        # A breaks down (first mean <0.5) at 0.4; B never.
        self.assertAlmostEqual(cur["breakdown_points"]["A"], 0.4)
        self.assertIsNone(cur["breakdown_points"]["B"])


# ════════════════════════════════════════════════════════════════════════════
# 3.  compute_ranking_stability
# ════════════════════════════════════════════════════════════════════════════

class TestRankingStability(unittest.TestCase):

    def test_constant_ranking_tau_one(self):
        models = ["A", "B", "C"]
        grid = np.array([0.0, 0.3])
        # A>B>C at every level → τ ≈ 1 everywhere.
        noise = np.array([0.0, 0.3])
        score = np.array([[0.9, 0.5, 0.1], [0.8, 0.4, 0.05]])
        st = mc.compute_ranking_stability(noise, grid, score, models)
        self.assertTrue(np.all(st["tau_per_level"] > 0.99))

    def test_reversed_ranking_tau_drops(self):
        models = ["A", "B", "C"]
        grid = np.array([0.0, 0.3])
        # low noise A>B>C ; high noise C>B>A (full reversal) → τ negative at high.
        noise = np.array([0.0, 0.3])
        score = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
        st = mc.compute_ranking_stability(noise, grid, score, models)
        self.assertLess(st["tau_per_level"][1], 0.0)


# ════════════════════════════════════════════════════════════════════════════
# 4.  winner-label / single-class surrogate path (no sklearn)
# ════════════════════════════════════════════════════════════════════════════

class TestWinnerLabelSingleClass(unittest.TestCase):

    def test_single_winner_no_sklearn(self):
        # A wins every row → single-class path returns without importing sklearn.
        models = ["A", "B"]
        noise = np.array([0.0, 0.1, 0.2, 0.3])
        score = np.array([[0.9, 0.1]] * 4)
        info = mc.train_noise_winner_surrogate(noise, score, models)
        self.assertTrue(info["feasible"])
        self.assertEqual(info["classes"], ["A"])
        self.assertEqual(info["win_rates"]["A"], 1.0)
        self.assertIn("Always A", info["rules_text"])


# ════════════════════════════════════════════════════════════════════════════
# 5.  Integration — needs sklearn (skips where mocked/missing)
# ════════════════════════════════════════════════════════════════════════════

class TestExplainMonteCarloIntegration(unittest.TestCase):

    @staticmethod
    def _flip_eval(m, level):
        # A leads below s≈0.35, B above (0.9−s == 0.2+s ⇒ s = 0.35); same for F1 & PR.
        s = (0.9 - level) if m == "A" else (0.2 + level)
        return (s, s)

    def test_winner_surrogate_threshold(self):
        import importlib
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("scikit-learn not installed")
        models = ["A", "B"]
        sweep = mc.monte_carlo_noise_sweep(_fake_data(), {}, models,
                                           noise_levels=np.linspace(0.0, 0.5, 15),
                                           repeats=6, evaluate_fn=self._flip_eval)
        info = mc.train_noise_winner_surrogate(sweep["noise"], sweep["F1"], models)
        self.assertTrue(info["feasible"])
        self.assertGreater(info["train_accuracy"], 0.9)
        self.assertIsNotNone(info["root_threshold"])
        self.assertTrue(0.25 < info["root_threshold"] < 0.45)
        # Held-out fidelity (cross-validated) must be reported alongside the
        # in-sample train_accuracy, not just the optimistic in-sample number.
        self.assertIn("cv_accuracy", info)
        self.assertFalse(np.isnan(info["cv_accuracy"]))
        self.assertGreaterEqual(info["cv_accuracy"], 0.0)
        self.assertLessEqual(info["cv_accuracy"], 1.0)
        # This sweep has a clean, noise-free crossover, so the surrogate should
        # generalize almost as well as it fits in-sample.
        self.assertGreater(info["cv_accuracy"], 0.8)

    def test_permodel_surrogates_report_cv_r2(self):
        import importlib
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("scikit-learn not installed")
        models = ["A", "B"]
        sweep = mc.monte_carlo_noise_sweep(_fake_data(), {}, models,
                                           noise_levels=np.linspace(0.0, 0.5, 15),
                                           repeats=6, evaluate_fn=self._flip_eval)
        permodel = mc.train_noise_permodel_surrogates(sweep["noise"], sweep["F1"], models)
        for m in models:
            self.assertIn("cv_r2", permodel[m])
            self.assertIn("cv_method", permodel[m])

    def test_constant_target_cv_r2_guarded(self):
        """A flat curve (constant up to float noise) must report cv_r2 as NaN
        with the constant_target method — never an astronomically negative
        R^2 from a near-zero-variance fold. The guard fires before the
        sklearn import, so this test needs no sklearn."""
        sf = mc.surrogate_fidelity
        X = np.linspace(0.0, 0.2, 20).reshape(-1, 1)
        y = np.full(20, 0.618)
        out = sf.held_out_regressor_fidelity(X, y)
        self.assertTrue(np.isnan(out["cv_r2"]))
        self.assertEqual(out["method"], "constant_target")
        self.assertIn("constant", out["note"])
        # Jitter below the 1e-8 tolerance is still "constant".
        y_jit = y + np.random.RandomState(0).uniform(-1e-10, 1e-10, 20)
        self.assertEqual(sf.held_out_regressor_fidelity(X, y_jit)["method"],
                         "constant_target")

    def test_mostly_flat_target_scored_per_fold(self):
        """A mostly-flat curve (global spread > 1e-8, but some folds' test
        points all on the flat part) must yield a finite cv_r2 via the
        per-fold force_finite convention — never an astronomical negative."""
        import importlib
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("scikit-learn not installed")
        sf = mc.surrogate_fidelity
        X = np.linspace(0.0, 0.2, 20).reshape(-1, 1)
        y = np.full(20, 0.022) + np.random.RandomState(1).uniform(-1e-12, 1e-12, 20)
        y[3] = 0.024
        y[15] = 0.020
        out = sf.held_out_regressor_fidelity(X, y)
        self.assertEqual(out["method"], "kfold")
        self.assertTrue(np.isfinite(out["cv_r2"]))
        self.assertGreaterEqual(out["cv_r2"], -100.0)
        self.assertIn("force_finite", out["note"])

    def test_orchestrator_writes_report_and_plots(self):
        import importlib
        if importlib.util.find_spec("sklearn") is None:
            self.skipTest("scikit-learn not installed")
        models = ["A", "B"]
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                res = mc.explain_monte_carlo(
                    _fake_data(), {"A": object(), "B": object()}, models,
                    "TEST", "e1", noise_levels=np.linspace(0.0, 0.5, 15), repeats=5,
                    explain=True, evaluate_fn=self._flip_eval,
                )
                self.assertIsInstance(res, dict)
                for k in ("sweep", "curves_f1", "curves_f1_fixed", "stability_f1",
                          "winner_f1", "permodel_f1", "n_trials"):
                    self.assertIn(k, res)
                # F1_fixed must be present in the sweep output.
                self.assertIn("F1_fixed", res["sweep"])
                self.assertEqual(res["sweep"]["F1_fixed"].shape, res["sweep"]["F1"].shape)
                # A→B crossover should be detected on the F1 curves.
                self.assertGreaterEqual(len(res["curves_f1"]["crossovers"]), 1)
                out = os.path.join("myresults", "robustness", "MonteCarlo", "TEST", "e1")
                for fname in (
                    "TEST_e1_MonteCarlo_explainability.txt",
                    "TEST_e1_MonteCarlo_noise_curves_F1.png",
                    "TEST_e1_MonteCarlo_noise_curves_PRAUC.png",
                    "TEST_e1_MonteCarlo_noise_curves_F1_fixed.png",
                    "TEST_e1_MonteCarlo_noise_curves_F1_plain.png",
                    "TEST_e1_MonteCarlo_noise_curves_PRAUC_plain.png",
                    "TEST_e1_MonteCarlo_noise_curves_F1_fixed_plain.png",
                    "TEST_e1_MonteCarlo_ranking_stability.png",
                    "TEST_e1_MonteCarlo_surrogate_tree_F1.png",
                ):
                    self.assertTrue(os.path.exists(os.path.join(out, fname)), fname)
                # Intermediate Representation JSON is emitted alongside.
                import json
                ir_path = os.path.join("myresults", "explanations_ir", "TEST", "e1",
                                       "ir_monte_carlo.json")
                self.assertTrue(os.path.exists(ir_path), ir_path)
                with open(ir_path) as fh:
                    ir_doc = json.load(fh)
                self.assertEqual(ir_doc["stage"], "monte_carlo")
                with open(os.path.join(
                        out, "TEST_e1_MonteCarlo_explainability.txt")) as fh:
                    report_txt = fh.read()
                self.assertIn("Held-out accuracy", report_txt)
                self.assertIn("R² (held-out)", report_txt)
            finally:
                os.chdir(cwd)

    def test_explain_false_returns_none(self):
        self.assertIsNone(mc.explain_monte_carlo(
            _fake_data(), {}, ["A"], "X", "Y", explain=False))


# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestVusRanking(unittest.TestCase):
    """The production summary publishes a VUS ranking only when one was asked for."""

    @staticmethod
    def _results(vus):
        return {
            "A": {"f1_scores": [0.9], "pr_auc_scores": [0.5], "vus_scores": vus and [0.2]},
            "B": {"f1_scores": [0.5], "pr_auc_scores": [0.9], "vus_scores": vus and [0.8]},
        }

    def test_vus_ranking_orders_by_mean_vus(self):
        s = mc.summarize_results(self._results(True))
        self.assertEqual(s["ranked_by_vus"], ["B", "A"])
        self.assertEqual(s["ranked_by_f1"], ["A", "B"])

    def test_no_vus_scores_means_no_vus_ranking(self):
        s = mc.summarize_results(self._results(False))
        self.assertEqual(s["ranked_by_vus"], [])
        self.assertEqual(s["ranked_by_f1"], ["A", "B"])

    def test_an_uncomputable_vus_sorts_last(self):
        results = self._results(True)
        results["C"] = {"f1_scores": [0.7], "pr_auc_scores": [0.7],
                        "vus_scores": [float("nan")]}
        self.assertEqual(mc.summarize_results(results)["ranked_by_vus"][-1], "C")

    def test_ranking_keys_cover_every_ranking_the_summary_adds(self):
        """Whatever walks the summary must skip all of them, not two of three."""
        s = mc.summarize_results(self._results(True))
        added = [k for k, v in s.items() if isinstance(v, list)]
        self.assertEqual(set(added), set(mc.RANKING_KEYS))


class TestSweepMetricGating(unittest.TestCase):
    """The explain-only sweep computes what the run's fitness names, no more."""

    @staticmethod
    def _sweep(metrics):
        return mc.monte_carlo_noise_sweep(
            _fake_data(), {}, ["A", "B"], noise_levels=np.linspace(0.0, 0.2, 3),
            repeats=2, evaluate_fn=lambda m, l: (0.8, 0.6, 0.7, 0.55),
            metrics=metrics)

    def test_the_sweep_reports_which_metrics_it_ran(self):
        self.assertEqual(self._sweep(("f1", "vus"))["metrics"], ("f1", "vus"))

    def test_every_column_is_present_whatever_was_asked_for(self):
        """Shapes stay rectangular so nothing downstream has to branch."""
        out = self._sweep(("f1",))
        for key in ("F1", "F1_fixed", "PR", "VUS"):
            self.assertEqual(out[key].shape, (3 * 2, 2), key)

    def test_an_injected_evaluator_can_supply_a_vus_column(self):
        out = self._sweep(("f1", "pr_auc", "vus"))
        self.assertTrue(np.allclose(out["VUS"], 0.55))
        self.assertTrue(np.allclose(out["PR"], 0.6))


class TestExplainMetricGating(unittest.TestCase):
    """Only the chosen metrics produce curves, surrogates and report sections."""

    def _run(self, metrics, tmp):
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            return mc.explain_monte_carlo(
                _fake_data(), {}, ["A", "B"], "DS", "e1",
                noise_levels=np.linspace(0.0, 0.2, 3), repeats=2, explain=True,
                evaluate_fn=lambda m, l: (0.9 - l if m == "A" else 0.2 + l, 0.6, 0.7, 0.55),
                metrics=metrics)
        finally:
            os.chdir(cwd)

    def test_f1_only_computes_no_pr_auc_structures(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(("f1",), tmp)
            self.assertTrue(res["curves_f1"])
            self.assertEqual(res["curves_pr"], {})
            self.assertEqual(res["curves_vus"], {})
            self.assertEqual(res["stability_pr"], {})
            report = _read_report(tmp)
        self.assertIn("Method A · F1 curves", report)
        self.assertNotIn("PR-AUC curves", report)
        self.assertNotIn("VUS curves", report)

    def test_vus_only_computes_no_f1_structures(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(("vus",), tmp)
            self.assertTrue(res["curves_vus"])
            self.assertEqual(res["curves_f1"], {})
            self.assertEqual(res["curves_f1_fixed"], {})
            report = _read_report(tmp)
        self.assertIn("Method A · VUS curves", report)
        self.assertNotIn("Method A · F1 curves", report)

    def test_only_the_chosen_metrics_write_curve_figures(self):
        """The result page's metric switcher is built from what is on disk."""
        with tempfile.TemporaryDirectory() as tmp:
            self._run(("f1", "vus"), tmp)
            figs = os.listdir(os.path.join(tmp, "myresults", "robustness",
                                           "MonteCarlo", "DS", "e1"))
        names = " ".join(figs)
        self.assertIn("noise_curves_F1_plain.png", names)
        self.assertIn("noise_curves_VUS_plain.png", names)
        self.assertNotIn("noise_curves_PRAUC", names)
        self.assertNotIn("surrogate_tree_PRAUC", names)


def _read_report(tmp):
    directory = os.path.join(tmp, "myresults", "robustness", "MonteCarlo", "DS", "e1")
    name = [f for f in os.listdir(directory) if f.endswith("_explainability.txt")][0]
    with open(os.path.join(directory, name)) as f:
        return f.read()
