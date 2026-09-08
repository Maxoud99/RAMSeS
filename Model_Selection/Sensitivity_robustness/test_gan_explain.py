"""
Standalone unit tests for the GAN robustness per-point explainability layer, and
for the four production fixes that made it possible at all.

Mocks the module's heavy imports (tensorflow, Metrics.metrics,
Utils.model_selection_utils, loguru) and loads GAN_test.py by file path. numpy /
matplotlib are real; sklearn is intentionally NOT mocked so the surrogate
integration tests can `importorskip` it (run where scikit-learn is installed).

The generator and discriminator are stubbed at module level rather than trained:
these tests are about the injection contract — how many points, which ones, at
what scale, described how — not about whether an adversarial network converges.
"""

import os
import sys
import types
import importlib.util
import importlib.machinery
import importlib
import tempfile
import unittest

import numpy as np
import matplotlib
matplotlib.use("Agg")


# ── Stand in for the module's heavy imports ─────────────────────────────────
#
# Only where the real module is NOT already imported. A unittest run that names
# several modules at once shares one process and one sys.modules, so overwriting
# a real tensorflow / Utils.model_selection_utils here would break whichever
# other test module actually needs it — Utils.test_pipeline_spec resolves real
# detector classes and would fail on a stubbed tensorflow with no __spec__.
def _set(name, **attrs):
    """Fill in a stand-in, never a real module.

    A real module has a `__file__`; a stand-in built by one of these harnesses
    does not. Attributes are added only to stand-ins, and only where absent, so
    this file can run beside `test_off_by_explain` — which builds its own,
    smaller stand-in for `Metrics.metrics` — and top it up rather than either
    clobbering it or leaving this module's imports unsatisfied.

    A stand-in is also given a real ModuleSpec: `importlib.util.find_spec` raises
    ValueError rather than answering on a module whose `__spec__` is None, and
    other test modules ask exactly that question about optional dependencies.
    """
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    if getattr(module, "__file__", None) is not None:
        return module                      # the real thing — leave it alone
    if getattr(module, "__spec__", None) is None:
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for key, value in attrs.items():
        if not hasattr(module, key):
            setattr(module, key, value)
    return module


class _Logger:
    def __getattr__(self, _):
        return lambda *a, **k: None


_set("Metrics")
_set("Utils")
# `Utils.pipeline_spec` is stdlib-only and is wanted for real. A stand-in `Utils`
# has no __path__, so that submodule would not resolve; giving it one lets the
# real file load while the mocked `Utils.model_selection_utils` below still
# shadows its sibling. Harmless when `Utils` is already the real package.
if not hasattr(sys.modules["Utils"], "__path__"):
    sys.modules["Utils"].__path__ = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "Utils")]
_set("Metrics.metrics",
     range_based_precision_recall_f1_auc=lambda *a, **k: (0, 0, 0.5, 0.5, None),
     prauc=lambda *a, **k: 0.5,
     f1_score=lambda *a, **k: 0.5,
     rank_key=lambda v: v,
     vus_score=lambda *a, **k: 0.5,
     vus_window=lambda *a, **k: 8)
_set("Utils.model_selection_utils", evaluate_model=lambda *a, **k: {},
     ScoringTimeout=type("ScoringTimeout", (Exception,), {}))
_set("loguru", logger=_Logger())
_keras = _set("tensorflow.keras", layers=types.SimpleNamespace(),
              models=types.SimpleNamespace())
_set("tensorflow", keras=_keras)

# ── Load GAN_test.py by path ────────────────────────────────────────────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "gan_test_mod", os.path.join(_THIS, "GAN_test.py"))
gt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gt)


def _sklearn_or_skip(test):
    if importlib.util.find_spec("sklearn") is None:
        test.skipTest("scikit-learn not installed")


# ── Stubs for the adversarial pair ──────────────────────────────────────────

class _RecordingDiscriminator:
    """Scores a candidate from its first feature: a known, orderable verdict."""

    def __init__(self):
        self.seen = []

    def predict(self, x, verbose=0):
        x = np.asarray(x, dtype=float)
        self.seen.append(x)
        return (0.5 + 0.4 * x[:, 0]).reshape(-1, 1)


class _Harness:
    """Swaps the module's generator/discriminator/trainer for deterministic stubs.

    `train_seen` captures exactly what `train_gan` was handed, which is how the
    training-data and scaling fixes are checked.
    """

    def __init__(self, candidate_fn):
        self.candidate_fn = candidate_fn
        self.train_seen = []
        self.discriminator = _RecordingDiscriminator()
        self._saved = {}

    def __enter__(self):
        for name in ("make_generator_model", "make_discriminator_model",
                     "train_gan", "generate_borderline_points"):
            self._saved[name] = getattr(gt, name)
        gt.make_generator_model = lambda dim: types.SimpleNamespace(dim=dim)
        gt.make_discriminator_model = lambda dim: self.discriminator
        gt.train_gan = lambda g, d, data, **kw: self.train_seen.append(np.asarray(data))
        gt.generate_borderline_points = (
            lambda generator, num_samples=100, noise_dim=10:
            self.candidate_fn(num_samples, noise_dim))
        return self

    def __exit__(self, *exc):
        for name, fn in self._saved.items():
            setattr(gt, name, fn)
        return False


def _random_candidates(num_samples, noise_dim):
    """Consumes the RNG, as the real generator's noise draw does."""
    return np.tanh(np.random.normal(0, 1, (num_samples, noise_dim)))


def _ramped_candidates(num_samples, noise_dim):
    """Feature 0 sweeps [-1, 1], so discriminator scores are a known ramp."""
    out = np.zeros((num_samples, noise_dim))
    out[:, 0] = np.linspace(-1.0, 1.0, num_samples)
    return out


def _series(n=500, d=3, seed=7):
    """A [0, 1] series, the range Datasets/load.py MinMax-scales everything to."""
    rng = np.random.RandomState(seed)
    data = rng.uniform(0.0, 1.0, size=(d, n))
    labels = np.zeros((1, n))
    labels[0, ::11] = 1
    return data, labels


# ════════════════════════════════════════════════════════════════════════════
# 1.  The production fixes
# ════════════════════════════════════════════════════════════════════════════

class TestProductionFixes(unittest.TestCase):

    def test_injection_budget_is_ten_percent(self):
        """`int(0.1 * len(labels))` read the ROW count of a (1, N) array, so it
        was 0 -> forced to 1: one injected point per run whatever the series
        length, while the window loop below it was already sized for one per
        window. The budget is a fraction of the timestamps."""
        data, labels = _series(n=500)
        with _Harness(_random_candidates):
            np.random.seed(0)
            _, _, normal_idx, anom_idx, total, recs = gt.integrate_gan_with_dataset(
                data, labels, return_records=True)
        self.assertEqual(len(recs), 50)
        self.assertEqual(len(normal_idx) + len(anom_idx), 50)
        self.assertEqual(total, 550)

    def test_gan_trains_on_distinct_normal_columns(self):
        """`np.where(labels == 0)[0]` on a 2-D label array returned ROW indices —
        an array of zeros — so clean_data was n_normal copies of column 0 and the
        GAN trained on a single repeated point."""
        data, labels = _series(n=200)
        with _Harness(_random_candidates) as h:
            np.random.seed(0)
            gt.integrate_gan_with_dataset(data, labels)
        seen = h.train_seen[0]
        self.assertEqual(seen.shape[0], int((labels == 0).sum()))
        self.assertGreater(len(np.unique(seen, axis=0)), 1)
        # And it really is the normal columns, not the whole series.
        self.assertLess(seen.shape[0], data.shape[1])

    def test_only_the_most_ambiguous_candidates_are_injected(self):
        """Paper Eq. 7-8: score a candidate pool, keep the B closest to tau."""
        data, labels = _series(n=200)          # -> budget 20, pool 200
        with _Harness(_ramped_candidates) as h:
            np.random.seed(0)
            _, _, _, _, _, recs = gt.integrate_gan_with_dataset(
                data, labels, return_records=True)

        budget = 20
        pool = gt.GAN_CANDIDATE_OVERSAMPLE * budget
        self.assertEqual(h.discriminator.seen[0].shape[0], pool)
        self.assertEqual(len(recs), budget)

        # Reproduce the selection independently from the ramp.
        scores = 0.5 + 0.4 * np.linspace(-1.0, 1.0, pool)
        tau = float(np.mean(scores))
        expected = np.sort(np.abs(scores - tau))[:budget]
        self.assertTrue(np.allclose(sorted(r["ambiguity"] for r in recs), expected))
        # Every kept point is at least as ambiguous as every discarded one.
        self.assertLessEqual(max(r["ambiguity"] for r in recs),
                             float(np.sort(np.abs(scores - tau))[budget]))
        # tau is recorded so the label rule stays auditable.
        self.assertTrue(all(abs(r["tau"] - tau) < 1e-9 for r in recs))

    def test_labels_follow_equation_9(self):
        """y_hat(x) = 1[D(x) >= tau] — the paper's direction, unchanged."""
        data, labels = _series(n=200)
        with _Harness(_ramped_candidates):
            np.random.seed(0)
            _, _, _, _, _, recs = gt.integrate_gan_with_dataset(
                data, labels, return_records=True)
        for r in recs:
            self.assertEqual(r["label"], int(r["disc_score"] > r["tau"]))
        # Both classes present: the selection straddles the threshold.
        self.assertEqual({r["label"] for r in recs}, {0, 1})

    def test_training_inputs_and_injected_points_round_trip(self):
        """The generator's tanh spans [-1, 1] but the series lives in [0, 1], so
        training happens in tanh space and the points come back before injection."""
        data, labels = _series(n=200)
        with _Harness(_ramped_candidates) as h:
            np.random.seed(0)
            augmented, _, normal_idx, anom_idx, _, _ = gt.integrate_gan_with_dataset(
                data, labels, return_records=True)
        seen = h.train_seen[0]
        self.assertGreaterEqual(seen.min(), -1.0)
        self.assertLessEqual(seen.max(), 1.0)
        # The map is exactly invertible, so the training view is the series view.
        self.assertTrue(np.allclose(gt._from_tanh_space(seen), data[:, (labels == 0).flatten()].T))
        # Injected columns land inside the series' own [0, 1] range, not tanh's.
        injected = augmented[:, np.concatenate([normal_idx, anom_idx]).astype(int)]
        self.assertGreaterEqual(injected.min(), 0.0)
        self.assertLessEqual(injected.max(), 1.0)

    def test_return_records_preserves_outputs(self):
        """The explainability opt-in must not perturb the RNG stream: the four
        production return values have to be byte-for-byte identical either way,
        or the ranking this explains is not the ranking that was reported."""
        data, labels = _series(n=300)
        with _Harness(_random_candidates):
            np.random.seed(11)
            a0, l0, n0, an0, t0 = gt.integrate_gan_with_dataset(data, labels)
            np.random.seed(11)
            a1, l1, n1, an1, t1, recs = gt.integrate_gan_with_dataset(
                data, labels, return_records=True)
        self.assertTrue(np.array_equal(a0, a1))
        self.assertTrue(np.array_equal(l0, l1))
        self.assertTrue(np.array_equal(n0, n1))
        self.assertTrue(np.array_equal(an0, an1))
        self.assertEqual(t0, t1)
        # Records align with the injected-index arrays and their labels.
        self.assertEqual(len(recs), len(n1) + len(an1))
        self.assertEqual(sorted(r["index"] for r in recs),
                         sorted(list(n1) + list(an1)))
        for r in recs:
            self.assertEqual(bool(r["label"]), r["index"] in list(an1))


# ════════════════════════════════════════════════════════════════════════════
# 2.  build_gan_point_table
# ════════════════════════════════════════════════════════════════════════════

def _rec(index, ambiguity=0.01, label=1, magnitude=0.5, spread=0.2,
         context_gap=0.3, local_std=0.4):
    return {"index": index, "ambiguity": ambiguity, "tau": 0.5,
            "disc_score": 0.5 + ambiguity, "label": label, "magnitude": magnitude,
            "spread": spread, "context_gap": context_gap, "local_std": local_std}


class TestPointTable(unittest.TestCase):

    def test_shapes_and_correctness(self):
        true_labels = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        recs = [_rec(2, ambiguity=0.02), _rec(5, ambiguity=0.07), _rec(8, ambiguity=0.01)]
        predA = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        predB = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0])   # wrong at index 5
        tbl = gt.build_gan_point_table(recs, {"A": [predA], "B": [predB]},
                                       true_labels, ["A", "B"])
        self.assertIsNotNone(tbl)
        self.assertEqual(tbl["X"].shape, (3, 7))
        self.assertEqual(tbl["correct"].shape, (3, 2))
        self.assertEqual(tbl["feature_names"], gt.GAN_FEATURE_NAMES)
        self.assertAlmostEqual(tbl["X"][0, 0], 0.02)        # ambiguity
        self.assertAlmostEqual(tbl["X"][0, 6], 0.2)         # position = 2 / 10
        self.assertTrue(np.all(tbl["correct"][:, 0]))
        self.assertListEqual(list(tbl["correct"][:, 1]), [True, False, True])

    def test_no_points_returns_none(self):
        self.assertIsNone(gt.build_gan_point_table([], {"A": [np.zeros(5)]},
                                                   np.zeros(5), ["A"]))

    def test_model_without_predictions_dropped(self):
        tbl = gt.build_gan_point_table([_rec(1)], {"A": [np.zeros(4)], "B": []},
                                       np.zeros(4), ["A", "B"])
        self.assertEqual(tbl["model_names"], ["A"])


# ════════════════════════════════════════════════════════════════════════════
# 3.  train_gan_point_surrogates
# ════════════════════════════════════════════════════════════════════════════

class TestSurrogates(unittest.TestCase):

    def _table(self, correct, X=None):
        n = correct.shape[0]
        if X is None:
            X = np.zeros((n, len(gt.GAN_FEATURE_NAMES)))
        return {"X": X, "feature_names": list(gt.GAN_FEATURE_NAMES),
                "correct": correct, "model_names": ["A", "B"],
                "indices": np.arange(n), "n_points": n}

    def test_degenerate_no_exclusive_wins_no_sklearn(self):
        correct = np.ones((8, 2), dtype=bool)
        res = gt.train_gan_point_surrogates(self._table(correct), "A")
        info = res["per_competitor"]["B"]
        self.assertTrue(info["degenerate"])
        self.assertIsNone(info["clf"])
        self.assertIn("no exclusive wins", info["rules_text"])

    def test_split_on_ambiguity(self):
        _sklearn_or_skip(self)
        n = 40
        X = np.zeros((n, len(gt.GAN_FEATURE_NAMES)))
        X[:, 0] = np.linspace(0.0, 0.1, n)          # ambiguity
        # position is a decoy, so it must carry NO information about the target:
        # a plain linspace would be perfectly collinear with ambiguity and the
        # tree could split on either, which tests the fixture rather than the code.
        X[:, 6] = np.random.RandomState(3).permutation(n) / float(n)
        correct = np.ones((n, 2), dtype=bool)
        correct[X[:, 0] < 0.05, 1] = False          # B fails on the ambiguous half
        res = gt.train_gan_point_surrogates(self._table(correct, X=X), "A")
        info = res["per_competitor"]["B"]
        self.assertFalse(info["degenerate"])
        self.assertGreater(info["train_accuracy"], 0.95)
        top = max(info["feature_importances"].items(), key=lambda kv: kv[1])[0]
        self.assertEqual(top, "ambiguity")
        self.assertFalse(np.isnan(info["cv_accuracy"]))
        self.assertGreater(info["cv_accuracy"], 0.8)

    def test_winner_without_predictions_infeasible(self):
        res = gt.train_gan_point_surrogates(self._table(np.ones((4, 2), dtype=bool)), "ZZ")
        self.assertFalse(res["feasible"])
        self.assertEqual(res["per_competitor"], {})


# ════════════════════════════════════════════════════════════════════════════
# 4.  explain_gan_robustness — the orchestrator
# ════════════════════════════════════════════════════════════════════════════

class TestOrchestrator(unittest.TestCase):

    def _materials(self):
        N = 60
        true_labels = np.zeros(N, dtype=int)
        true_labels[::5] = 1
        idxs = list(range(2, N, 5))
        recs = []
        for j, idx in enumerate(idxs):
            amb = 0.002 * j
            recs.append(_rec(idx, ambiguity=amb, label=int(true_labels[idx]),
                             local_std=0.2 + 0.5 * (j % 3)))
        predA = true_labels.copy()
        predB = true_labels.copy()
        for j, idx in enumerate(idxs):
            if recs[j]["ambiguity"] < 0.01:
                predB[idx] = 1 - true_labels[idx]     # B misses the ambiguous ones
        return recs, {"A": [predA], "B": [predB]}, true_labels

    def test_writes_report_and_plots(self):
        _sklearn_or_skip(self)
        recs, ad, true_labels = self._materials()
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                res = gt.explain_gan_robustness(recs, ad, true_labels, ["A", "B"],
                                                ["A", "B"], "TEST", "e1", explain=True)
                self.assertIsInstance(res, dict)
                self.assertEqual(res["winner"], "A")
                out = os.path.join("myresults", "robustness", "GAN", "TEST", "e1")
                for fname in (
                    "TEST_e1_gan_explainability.txt",
                    "TEST_e1_gan_point_tree_A_vs_B.png",
                    "TEST_e1_gan_point_importance.png",
                ):
                    self.assertTrue(os.path.exists(os.path.join(out, fname)), fname)
                with open(os.path.join(out, "TEST_e1_gan_explainability.txt")) as fh:
                    report_txt = fh.read()
                self.assertIn("held-out accuracy", report_txt.lower())
                # Intermediate Representation JSON is emitted alongside.
                import json
                ir_path = os.path.join("myresults", "explanations_ir", "TEST", "e1",
                                       "ir_gan.json")
                self.assertTrue(os.path.exists(ir_path), ir_path)
                with open(ir_path) as fh:
                    ir_doc = json.load(fh)
                self.assertEqual(ir_doc["stage"], "gan")
                self.assertEqual(ir_doc["output"]["winner"], "A")
            finally:
                os.chdir(cwd)

    def test_explain_false_returns_none(self):
        recs, ad, true_labels = self._materials()
        self.assertIsNone(gt.explain_gan_robustness(recs, ad, true_labels, ["A", "B"],
                                                    ["A", "B"], "X", "Y", explain=False))


# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
