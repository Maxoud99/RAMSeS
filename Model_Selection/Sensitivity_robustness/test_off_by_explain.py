"""
Standalone unit tests for the off-by-threshold per-point explainability layer.
Mocks the module's heavy imports (Metrics.metrics, Utils.model_selection_utils,
loguru) and loads off_by_threshold_testing.py by file path. numpy / matplotlib
are real; sklearn is intentionally NOT mocked so the surrogate integration tests
can `importorskip` it (run where scikit-learn is installed).
"""

import os
import sys
import types
import importlib.util
import importlib
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
# `Utils` is a stand-in so that `Utils.model_selection_utils` can be mocked, but
# `Utils.pipeline_spec` is stdlib-only and is wanted for real — giving the
# stand-in a __path__ lets that one submodule resolve from disk while the mock
# above still shadows its sibling.
sys.modules["Utils"].__path__ = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "Utils")]
sys.modules["Metrics.metrics"].range_based_precision_recall_f1_auc = lambda *a, **k: (0, 0, 0.5, 0.5, None)
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

# ── Load off_by_threshold_testing.py by path ────────────────────────────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "offby", os.path.join(_THIS, "off_by_threshold_testing.py"))
ob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ob)


def _sklearn_or_skip(test):
    if importlib.util.find_spec("sklearn") is None:
        test.skipTest("scikit-learn not installed")


# ════════════════════════════════════════════════════════════════════════════
# 1.  Injector regression — return_records is output-preserving
# ════════════════════════════════════════════════════════════════════════════

class TestInjectorRegression(unittest.TestCase):

    def _data(self):
        rng = np.random.RandomState(123)
        data = rng.normal(size=(1, 200))
        labels = np.zeros(200)
        labels[::7] = 1
        return data, labels

    def test_return_records_preserves_outputs(self):
        data, labels = self._data()
        np.random.seed(0)
        a0, l0, n0, an0 = ob.intersperse_borderline_normal_points(data, labels, 0.1)
        np.random.seed(0)
        a1, l1, n1, an1, recs = ob.intersperse_borderline_normal_points(
            data, labels, 0.1, return_records=True)
        # First four returns must be byte-for-byte identical.
        self.assertTrue(np.array_equal(a0, a1))
        self.assertTrue(np.array_equal(l0, l1))
        self.assertEqual(n0, n1)
        self.assertEqual(an0, an1)
        # Records align with the injected-index lists and labels.
        self.assertEqual(len(recs), len(n1) + len(an1))
        rec_idx = sorted(r["index"] for r in recs)
        self.assertEqual(rec_idx, sorted(list(n1) + list(an1)))
        for r in recs:
            in_anom = r["index"] in an1
            self.assertEqual(bool(r["label"]), in_anom)
            self.assertGreaterEqual(r["scale"], 0.95)
            self.assertLessEqual(r["scale"], 1.05)


# ════════════════════════════════════════════════════════════════════════════
# 2.  build_offby_point_table
# ════════════════════════════════════════════════════════════════════════════

class TestPointTable(unittest.TestCase):

    def test_shapes_and_correctness(self):
        N = 10
        true_labels = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        recs = [
            {"index": 2, "scale": 1.02, "local_std": 0.5, "label": 1},
            {"index": 5, "scale": 0.97, "local_std": 0.3, "label": 1},
            {"index": 8, "scale": 1.04, "local_std": 0.9, "label": 1},
        ]
        # A: correct at all three; B: wrong at index 5.
        predA = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        predB = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0])
        ad = {"A": [predA], "B": [predB]}
        tbl = ob.build_offby_point_table(recs, ad, true_labels, ["A", "B"])
        self.assertIsNotNone(tbl)
        self.assertEqual(tbl["X"].shape, (3, 4))
        self.assertEqual(tbl["correct"].shape, (3, 2))
        self.assertEqual(tbl["model_names"], ["A", "B"])
        # boundary_distance = |scale - 1|
        self.assertAlmostEqual(tbl["X"][0, 0], 0.02)
        # A correct on all; B wrong on the middle injected point.
        self.assertTrue(np.all(tbl["correct"][:, 0]))
        self.assertListEqual(list(tbl["correct"][:, 1]), [True, False, True])

    def test_no_points_returns_none(self):
        self.assertIsNone(ob.build_offby_point_table([], {"A": [np.zeros(5)]}, np.zeros(5), ["A"]))

    def test_model_without_predictions_dropped(self):
        recs = [{"index": 1, "scale": 1.0, "local_std": 0.1, "label": 0}]
        tbl = ob.build_offby_point_table(recs, {"A": [np.zeros(4)], "B": []},
                                         np.zeros(4), ["A", "B"])
        self.assertEqual(tbl["model_names"], ["A"])


# ════════════════════════════════════════════════════════════════════════════
# 3.  train_offby_point_surrogates
# ════════════════════════════════════════════════════════════════════════════

class TestSurrogates(unittest.TestCase):

    def _table(self, correct, X=None, models=("A", "B")):
        n = correct.shape[0]
        if X is None:
            X = np.column_stack([
                np.linspace(0.0, 0.1, n),          # boundary_distance
                np.zeros(n),                        # is_anomaly
                np.linspace(0.1, 0.9, n),           # local_volatility
                np.linspace(0.0, 1.0, n),           # position
            ])
        return {"X": X, "feature_names": list(ob.OFFBY_FEATURE_NAMES),
                "correct": correct, "model_names": list(models),
                "indices": np.arange(n), "n_points": n}

    def test_degenerate_no_exclusive_wins_no_sklearn(self):
        # A and B both correct everywhere → winner has no exclusive wins; pure path.
        correct = np.ones((20, 2), dtype=bool)
        res = ob.train_offby_point_surrogates(self._table(correct), "A")
        info = res["per_competitor"]["B"]
        self.assertTrue(info["degenerate"])
        self.assertEqual(info["n_exclusive_wins"], 0)
        self.assertIsNone(info["clf"])
        self.assertIn("no exclusive wins", info["rules_text"])

    def test_split_on_boundary_distance(self):
        _sklearn_or_skip(self)
        n = 60
        rng = np.random.RandomState(7)
        bd = np.linspace(0.0, 0.1, n)
        # Other features are non-informative (random) so only boundary_distance separates.
        X = np.column_stack([bd, rng.randint(0, 2, n).astype(float),
                             rng.uniform(0.1, 0.9, n), rng.uniform(0, 1, n)])
        correct = np.ones((n, 2), dtype=bool)
        # B fails exactly on the small-boundary-distance points; A correct everywhere.
        correct[:, 1] = bd >= 0.05
        res = ob.train_offby_point_surrogates(self._table(correct, X=X), "A")
        info = res["per_competitor"]["B"]
        self.assertFalse(info["degenerate"])
        self.assertGreater(info["train_accuracy"], 0.95)
        top = max(info["feature_importances"].items(), key=lambda kv: kv[1])[0]
        self.assertEqual(top, "boundary_distance")
        self.assertGreater(info["n_exclusive_wins"], 0)
        # Held-out fidelity must be reported alongside the in-sample fit — a
        # clean single-feature split like this one should generalize well.
        self.assertIn("cv_accuracy", info)
        self.assertFalse(np.isnan(info["cv_accuracy"]))
        self.assertGreater(info["cv_accuracy"], 0.8)

    def test_winner_without_predictions_infeasible(self):
        correct = np.ones((5, 2), dtype=bool)
        res = ob.train_offby_point_surrogates(self._table(correct), "ZZ")
        self.assertFalse(res["feasible"])


# ════════════════════════════════════════════════════════════════════════════
# 4.  explain_off_by_threshold orchestrator (integration; needs sklearn)
# ════════════════════════════════════════════════════════════════════════════

class TestOrchestrator(unittest.TestCase):

    def _materials(self, N=60):
        true_labels = np.zeros(N, dtype=int)
        true_labels[::5] = 1
        # Injected points at a spread of indices with varying boundary distance.
        idxs = list(range(2, N, 5))
        recs = []
        for j, idx in enumerate(idxs):
            scale = 0.96 + 0.08 * (j / max(1, len(idxs) - 1))   # 0.96 → 1.04
            recs.append({"index": idx, "scale": scale,
                         "local_std": 0.2 + 0.5 * (j % 3), "label": int(true_labels[idx])})
        # A matches the truth everywhere; B fails on the low-boundary-distance injected points.
        predA = true_labels.copy()
        predB = true_labels.copy()
        for j, idx in enumerate(idxs):
            if abs(recs[j]["scale"] - 1.0) < 0.03:
                predB[idx] = 1 - true_labels[idx]   # flip → wrong
        ad = {"A": [predA], "B": [predB]}
        return recs, ad, true_labels

    def test_writes_report_and_plots(self):
        _sklearn_or_skip(self)
        recs, ad, true_labels = self._materials()
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                res = ob.explain_off_by_threshold(recs, ad, true_labels, ["A", "B"],
                                                  ["A", "B"], "TEST", "e1", explain=True)
                self.assertIsInstance(res, dict)
                self.assertEqual(res["winner"], "A")
                out = os.path.join("myresults", "robustness", "off_by", "TEST", "e1")
                for fname in (
                    "TEST_e1_off_by_explainability.txt",
                    "TEST_e1_off_by_point_tree_A_vs_B.png",
                    "TEST_e1_off_by_point_importance.png",
                ):
                    self.assertTrue(os.path.exists(os.path.join(out, fname)), fname)
                with open(os.path.join(out, "TEST_e1_off_by_explainability.txt")) as fh:
                    report_txt = fh.read()
                self.assertIn("held-out accuracy", report_txt.lower())
                # Intermediate Representation JSON is emitted alongside.
                import json
                ir_path = os.path.join("myresults", "explanations_ir", "TEST", "e1",
                                       "ir_off_by.json")
                self.assertTrue(os.path.exists(ir_path), ir_path)
                with open(ir_path) as fh:
                    ir_doc = json.load(fh)
                self.assertEqual(ir_doc["stage"], "off_by_threshold")
                self.assertEqual(ir_doc["output"]["winner"], "A")
            finally:
                os.chdir(cwd)

    def test_explain_false_returns_none(self):
        recs, ad, true_labels = self._materials()
        self.assertIsNone(ob.explain_off_by_threshold(recs, ad, true_labels, ["A", "B"],
                                                      ["A", "B"], "X", "Y", explain=False))


# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
