"""
Tests for Utils/pipeline_spec.py — the shared detector/stage vocabulary.

Loaded by file path (stdlib-only module) so the suite never imports torch via
Utils/utils.py. Run with `pytest Utils/test_pipeline_spec.py` or
`python -m unittest Utils.test_pipeline_spec`.
"""

import importlib.util
import math
import os
import unittest

_THIS = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "pipeline_spec", os.path.join(_THIS, "pipeline_spec.py"))
spec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spec)


class TestStages(unittest.TestCase):
    """parse_stages must reproduce the behaviour the inline block in
    Utils/utils.py had before it was replaced, token for token."""

    def test_every_individual_token(self):
        for tok in ("ga", "thompson", "gan", "offby", "montecarlo"):
            self.assertEqual(spec.parse_stages(tok), {tok})

    def test_groups(self):
        self.assertEqual(spec.parse_stages("all"), set(spec.ALL_STAGES))
        self.assertEqual(spec.parse_stages("robustness"),
                         {"gan", "offby", "montecarlo"})

    def test_combination_union_and_case_and_whitespace(self):
        self.assertEqual(spec.parse_stages(" GA , Thompson "), {"ga", "thompson"})
        self.assertEqual(spec.parse_stages("robustness,ga"),
                         {"gan", "offby", "montecarlo", "ga"})
        self.assertEqual(spec.parse_stages("ga,ga"), {"ga"})

    def test_empty_and_none_mean_all(self):
        for value in (None, "", "  ", ",,"):
            self.assertEqual(spec.parse_stages(value), set(spec.ALL_STAGES))

    def test_unknown_token_raises_with_the_cli_message(self):
        with self.assertRaises(ValueError) as cm:
            spec.parse_stages("ga,nope")
        msg = str(cm.exception)
        self.assertIn("unknown stage 'nope'", msg)
        self.assertIn("all, robustness", msg)


class TestAnomalies(unittest.TestCase):

    def test_type_defaults_and_normalises(self):
        self.assertEqual(spec.parse_anomaly_type(None), spec.DEFAULT_ANOMALY_TYPE)
        self.assertEqual(spec.parse_anomaly_type(" Wander "), "wander")

    def test_every_type_is_accepted(self):
        for token in spec.ALL_ANOMALY_TYPES:
            self.assertEqual(spec.parse_anomaly_type(token), token)

    def test_unknown_type_names_the_valid_ones(self):
        with self.assertRaises(ValueError) as cm:
            spec.parse_anomaly_type("spike")
        self.assertIn("unknown type 'spike'", str(cm.exception))

    def test_rate_none_keeps_the_per_type_defaults(self):
        self.assertIsNone(spec.parse_anomaly_rate(None))
        self.assertIsNone(spec.parse_anomaly_rate(""))

    def test_rate_bounds(self):
        self.assertEqual(spec.parse_anomaly_rate("0.25"), 0.25)
        self.assertEqual(spec.parse_anomaly_rate(1), 1.0)
        for bad in ("0", "-0.1", "1.01"):
            with self.assertRaises(ValueError):
                spec.parse_anomaly_rate(bad)
        with self.assertRaises(ValueError):
            spec.parse_anomaly_rate("half")


class TestDecisionMetric(unittest.TestCase):

    def test_defaults_and_normalises(self):
        self.assertEqual(spec.parse_decision_metrics(None), spec.DEFAULT_DECISION_METRICS)
        self.assertEqual(spec.parse_decision_metrics(" PR-AUC "), ("pr_auc",))
        self.assertEqual(spec.parse_decision_metrics("PR_AUC,f1"), ("f1", "pr_auc"))

    def test_order_is_canonical_whatever_the_user_typed(self):
        """Equivalent selections must produce identical argv and labels."""
        self.assertEqual(spec.parse_decision_metrics("vus,f1,pr_auc"),
                         spec.parse_decision_metrics("pr_auc,vus,f1"))

    def test_duplicates_collapse(self):
        self.assertEqual(spec.parse_decision_metrics("f1,f1"), ("f1",))

    def test_every_metric_is_accepted_and_labelled(self):
        for token in spec.DECISION_METRICS:
            self.assertEqual(spec.parse_decision_metrics(token), (token,))
            self.assertTrue(spec.decision_metric_label((token,)))

    def test_unknown_metric_names_the_valid_ones(self):
        with self.assertRaises(ValueError) as cm:
            spec.parse_decision_metrics("auroc")
        self.assertIn("auroc", str(cm.exception))

    def test_at_least_one_metric_is_required(self):
        with self.assertRaises(ValueError):
            spec.parse_decision_metrics(",")

    def test_default_is_f1_and_pr_auc(self):
        self.assertEqual(spec.DEFAULT_DECISION_METRICS, ("f1", "pr_auc"))
        self.assertEqual(spec.decision_metric_formula(spec.DEFAULT_DECISION_METRICS),
                         "0.5 * F1 + 0.5 * PR-AUC")

    def test_the_default_publishes_both_robustness_rankings(self):
        self.assertEqual(spec.ranking_metrics_for(spec.DEFAULT_DECISION_METRICS),
                         ("f1", "pr_auc"))

    def test_a_single_metric_is_its_own_fitness(self):
        scores = {"f1": 0.9, "pr_auc": 0.2, "vus": 0.99}
        for token, want in scores.items():
            self.assertEqual(spec.combine_metrics((token,), scores), want)

    def test_several_metrics_are_equally_weighted(self):
        scores = {"f1": 0.9, "pr_auc": 0.3, "vus": 0.6}
        self.assertAlmostEqual(spec.combine_metrics(("f1", "pr_auc"), scores), 0.6)
        self.assertAlmostEqual(spec.combine_metrics(("f1", "pr_auc", "vus"), scores), 0.6)

    def test_the_formula_is_what_the_report_prints(self):
        self.assertEqual(spec.decision_metric_formula(("f1",)), "F1")
        self.assertEqual(spec.decision_metric_formula(("f1", "pr_auc")),
                         "0.5 * F1 + 0.5 * PR-AUC")
        self.assertIn("0.333", spec.decision_metric_formula(("f1", "pr_auc", "vus")))

    def test_required_names_only_what_must_be_computed(self):
        """The GA skips VUS unless the spec asks for it, so this is what keeps
        an F1 run from paying for one."""
        for token in spec.DECISION_METRICS:
            self.assertEqual(spec.metrics_required((token,)), (token,))
        self.assertNotIn("vus", spec.metrics_required(("f1", "pr_auc")))

    def test_combine_needs_only_the_required_metrics(self):
        """A caller that computed just the required ones must not KeyError."""
        spec_ = ("f1", "pr_auc")
        partial = {k: 0.5 for k in spec.metrics_required(spec_)}
        self.assertEqual(spec.combine_metrics(spec_, partial), 0.5)

    def test_robustness_rankings_follow_the_fitness(self):
        """One published ranking per chosen metric, and no others."""
        for chosen in (("f1",), ("pr_auc",), ("vus",), ("f1", "pr_auc"),
                       ("f1", "vus"), ("f1", "pr_auc", "vus")):
            self.assertEqual(spec.ranking_metrics_for(chosen), chosen)
        self.assertEqual(spec.ranking_metrics_for({"f1": 0.7, "vus": 0.3}),
                         ("f1", "vus"))


class TestDecisionMetricWeights(unittest.TestCase):

    def test_weights_are_parsed_and_kept(self):
        self.assertEqual(spec.parse_decision_metrics("f1:0.5,pr_auc:0.3,vus:0.2"),
                         {"f1": 0.5, "pr_auc": 0.3, "vus": 0.2})

    def test_weights_need_not_sum_to_one(self):
        self.assertEqual(spec.parse_decision_metrics("f1:2,pr_auc:2"), ("f1", "pr_auc"))
        weights = spec.metric_weights(spec.parse_decision_metrics("f1:3,pr_auc:1"))
        self.assertAlmostEqual(weights["f1"], 0.75)
        self.assertAlmostEqual(weights["pr_auc"], 0.25)

    def test_uniform_weights_collapse_to_the_plain_spelling(self):
        """So an equally weighted run keeps producing the argv it always did."""
        self.assertEqual(spec.parse_decision_metrics("f1:0.5,pr_auc:0.5"),
                         ("f1", "pr_auc"))

    def test_an_unweighted_metric_counts_one(self):
        self.assertEqual(spec.parse_decision_metrics("f1:3,pr_auc"),
                         {"f1": 0.75, "pr_auc": 0.25})

    def test_a_zero_weight_drops_its_metric(self):
        """Otherwise ranking_metrics_for publishes both rankings for a fitness
        that is purely F1."""
        self.assertEqual(spec.parse_decision_metrics("f1:1,pr_auc:0"), ("f1",))
        self.assertEqual(spec.ranking_metrics_for({"f1": 1.0, "pr_auc": 0.0}), ("f1",))
        self.assertEqual(spec.metrics_required({"f1": 1.0, "pr_auc": 0.0}), ("f1",))

    def test_all_weights_zero_is_no_selection(self):
        with self.assertRaises(ValueError):
            spec.parse_decision_metrics("f1:0,pr_auc:0")

    def test_negative_and_unparsable_weights_are_refused(self):
        for text in ("f1:-1", "f1:x"):
            with self.assertRaises(ValueError):
                spec.parse_decision_metrics(text)

    def test_fitness_is_the_weighted_mean(self):
        weighted = {"f1": 0.5, "pr_auc": 0.3, "vus": 0.2}
        scores = {"f1": 1.0, "pr_auc": 0.5, "vus": 0.0}
        self.assertAlmostEqual(spec.combine_metrics(weighted, scores), 0.65)

    def test_an_uncomputable_metric_renormalises_the_rest(self):
        """VUS on a short window must narrow the fitness, not void it."""
        weighted = {"f1": 0.5, "pr_auc": 0.3, "vus": 0.2}
        scores = {"f1": 1.0, "pr_auc": 0.5, "vus": float("nan")}
        self.assertAlmostEqual(spec.combine_metrics(weighted, scores), 0.8125)

    def test_every_metric_missing_gives_nan(self):
        value = spec.combine_metrics(("f1", "vus"),
                                     {"f1": float("nan"), "vus": float("nan")})
        self.assertTrue(math.isnan(value))

    def test_the_formula_shows_the_weights(self):
        self.assertEqual(
            spec.decision_metric_formula({"f1": 0.5, "pr_auc": 0.3, "vus": 0.2}),
            "0.5 * F1 + 0.3 * PR-AUC + 0.2 * VUS")

    def test_format_round_trips_through_parse(self):
        for text in ("f1", "f1,pr_auc", "f1:0.5,pr_auc:0.3,vus:0.2", "f1:3,pr_auc:1"):
            parsed = spec.parse_decision_metrics(text)
            self.assertEqual(
                spec.parse_decision_metrics(spec.format_decision_metrics(parsed)),
                parsed)

    def test_restrict_keeps_the_relative_weights(self):
        narrowed = spec.restrict_metrics({"f1": 0.5, "pr_auc": 0.3, "vus": 0.2},
                                         ("f1", "pr_auc"))
        self.assertAlmostEqual(narrowed["f1"], 0.625)
        self.assertAlmostEqual(narrowed["pr_auc"], 0.375)

    def test_restrict_collapses_to_a_tuple_when_it_can(self):
        self.assertEqual(spec.restrict_metrics(("f1", "pr_auc", "vus"), ("f1", "vus")),
                         ("f1", "vus"))

    def test_a_weighted_spec_labels_and_orders_like_a_plain_one(self):
        weighted = {"vus": 0.2, "f1": 0.5, "pr_auc": 0.3}
        self.assertEqual(spec.metrics_required(weighted), ("f1", "pr_auc", "vus"))
        self.assertEqual(spec.decision_metric_label(weighted), "F1 + PR-AUC + VUS")
        self.assertEqual(list(spec.metric_weights(weighted)), ["f1", "pr_auc", "vus"])


class TestDetectors(unittest.TestCase):

    def test_none_means_all(self):
        self.assertIsNone(spec.parse_detectors(None))
        self.assertIsNone(spec.parse_detectors(""))

    def test_canonical_order_and_dedupe(self):
        # Input order and duplicates must not change the result: identical
        # selections have to produce byte-identical argv.
        self.assertEqual(spec.parse_detectors("NN_1,LOF_1"), ["LOF_1", "NN_1"])
        self.assertEqual(spec.parse_detectors("LOF_1,NN_1"), ["LOF_1", "NN_1"])
        self.assertEqual(spec.parse_detectors("NN_1,LOF_1,NN_1"), ["LOF_1", "NN_1"])
        self.assertEqual(spec.parse_detectors("CBLOF_4,NN_3,LOF_2"),
                         ["LOF_2", "NN_3", "CBLOF_4"])

    def test_case_insensitive_and_whitespace(self):
        self.assertEqual(spec.parse_detectors(" lof_1 , nn_2 "), ["LOF_1", "NN_2"])

    def test_unknown_detector_raises(self):
        with self.assertRaises(ValueError) as cm:
            spec.parse_detectors("LOF_1,NOPE_9")
        self.assertIn("NOPE_9", str(cm.exception))

    def test_validation_is_against_the_list_not_the_disk(self):
        """Names are checked against ALL_DETECTORS, never against whatever
        checkpoints happen to exist.

        RNN used to be the example here — entities carried leftover RNN_*.pth
        that were not selectable — but RNN is in the pool now, so the case needs
        a name that is genuinely absent. GMM is one: eight checkpoints on disk,
        no implementation in the repo, no place in the pool.
        """
        with self.assertRaises(ValueError) as cm:
            spec.parse_detectors("GMM_1,LOF_1")
        self.assertIn("GMM_1", str(cm.exception))

    def test_below_minimum_raises(self):
        with self.assertRaises(ValueError) as cm:
            spec.parse_detectors("LOF_1")
        self.assertIn("at least 2", str(cm.exception))


class TestFamilies(unittest.TestCase):

    def test_family_of(self):
        self.assertEqual(spec.family_of("CBLOF_2"), "CBLOF")
        self.assertEqual(spec.family_of("NN_3"), "NN")

    def test_ae_cuts_enough_windows_for_pyods_dropped_last_batch(self):
        """AutoEncoder is the only pool detector on PyOD's deep base class.

        `base_dl.fit` builds `DataLoader(batch_size=32, drop_last=True)`, so the
        window COUNT has to clear 32 or the loader yields no batches, the
        training loop body never runs, and PyOD raises `UnboundLocalError` on an
        unassigned `loss` — having trained on nothing. The count is set here, by
        `window_step`, not by anything PyOD can see: at the 64/64 the framework
        models use, SKAB's 917-step entity gave 14 windows and crashed.

        The entity lengths are the real ones after downsampling: SKAB 917,
        SMD 2848. A grid change that stopped clearing 32 on the shorter of them
        would put the crash back.
        """
        from Model_Training.hyperparameter_grids import AUTOENCODER_PARAM_GRID
        window = AUTOENCODER_PARAM_GRID["window_size"][0]
        step = AUTOENCODER_PARAM_GRID["window_step"][0]
        for n_time, name in ((917, "SKAB"), (2848, "SMD")):
            with self.subTest(entity=name):
                n_windows = (n_time - window) // step + 1
                self.assertGreaterEqual(
                    n_windows, 32,
                    f"{name}: window={window} step={step} yields {n_windows} "
                    f"windows, under PyOD's batch_size 32 with drop_last=True")

    def test_families_for_is_ordered_and_deduped(self):
        self.assertEqual(spec.families_for(["NN_1", "LOF_2", "NN_3"]), ["LOF", "NN"])
        self.assertEqual(spec.families_for(spec.ALL_DETECTORS),
                         list(spec.DETECTOR_FAMILIES))

    def test_every_detector_maps_to_a_known_family(self):
        for d in spec.ALL_DETECTORS:
            self.assertIn(spec.family_of(d), spec.DETECTOR_FAMILIES)

    def test_every_family_can_actually_be_trained(self):
        """A family listed here is offered in the UI and accepted by
        --detectors, so a name the trainer cannot build would only fail once
        someone selected it and waited for stage 3.

        Families with their own branch in TrainModels.train_models are trained
        by that branch; the TSB-AD families are routed to `train_tsbad`, which
        resolves them in the vendored package; the rest fall through to
        `train_pyod`, which asks PyOD for the class by name. This asserts every
        family reaches one of the three.
        """
        with open(os.path.join(_THIS, os.pardir, "Model_Training", "train.py")) as f:
            trainer = f.read()
        try:
            import pyod.models as pyod_models
            from Algorithms.pyod_model import create_model, get_all_module_names
            from Algorithms.tsbad_model import _class_for
        except ImportError:
            self.skipTest("pyod not installed in this interpreter")
        modules = get_all_module_names(pyod_models)
        for family in spec.DETECTOR_FAMILIES:
            if f"'{family}' == model_name" in trainer:
                continue          # has its own branch
            if family in spec.TSBAD_FAMILIES:
                _class_for(family)                         # raises if not
                continue
            create_model(family, modules, contamination=0.1)   # raises if not

    def test_every_family_scores_a_batch_independently(self):
        """A detector's score for a window must not depend on which other
        windows share its batch.

        `evaluate_model` scores in batches, so a transductive detector gives a
        window a different score depending on where the batch boundaries fall,
        and raises outright when a batch is smaller than its neighbourhood.
        PyOD's COF is exactly that — `decision_function` runs
        `distance_matrix(X, X)` over the call's own rows — which is why it is
        not in the pool. This is the guard against it, or another like it, being
        added back on the strength of the name alone.
        """
        try:
            import numpy as np
            import torch as t
            import pyod.models as pyod_models
            from Algorithms import windowed
            from Algorithms.pyod_model import create_model, get_all_module_names
        except ImportError:
            self.skipTest("torch/pyod not installed in this interpreter")

        modules = get_all_module_names(pyod_models)
        rng = np.random.default_rng(0)
        train = t.tensor(rng.normal(size=(60, 4, 16)), dtype=t.float32)
        probe = t.tensor(rng.normal(size=(1, 4, 16)), dtype=t.float32)
        # Two different sets of companions, one of them on a different scale.
        pad_a = t.tensor(rng.normal(size=(24, 4, 16)), dtype=t.float32)
        pad_b = t.tensor(rng.normal(size=(24, 4, 16)), dtype=t.float32) * 5

        class _Loader:
            Y_windows = train

        # Only the PyOD-backed families go through `windowed.score_windows`. The
        # framework's own detectors (NN, RNN, LSTMVAE, DGHL, RM, MD) implement
        # their own scoring and are not PyOD estimators at all, so `create_model`
        # cannot build them and there is nothing here to check.
        # LSTMAD cuts its own subsequences out of what it is handed, so it
        # cannot score a one-row batch at all. `evaluate_model` gives it the
        # whole series in a single batch for that reason (_WHOLE_SERIES_MODELS),
        # which removes the boundary this test is about; it was measured
        # inductive separately (14.586687 both ways).
        #
        # The transductive three fail this by definition — that is what
        # transductive means — and are admitted knowing it. They are not simply
        # dropped: TestTransductiveFamilies below asserts the three properties
        # that make the exemption safe, including that they really are still
        # transductive, so this exemption cannot quietly outlive its reason.
        #
        # The TSB-AD families are exempt for LSTMAD's reason, not COF's: each
        # cuts its own subsequence out of the call, so a one-row batch has
        # nothing to cut. They are named here rather than left to fall through
        # the `except ValueError: continue` below, which would skip them
        # silently on the accident that `create_model` cannot build them —
        # TestTSBADFamilies asserts what holds in their place.
        exempt = ({"LSTMAD"} | set(spec.TRANSDUCTIVE_FAMILIES)
                  | set(spec.TSBAD_FAMILIES))
        checked = 0
        for family in spec.DETECTOR_FAMILIES:
            if family in exempt:
                continue
            try:
                model = create_model(family, modules, contamination=0.1)
            except ValueError:
                continue
            checked += 1
            windowed.fit_windows(model, _Loader())
            alone = windowed.score_windows(model, probe)[0]
            with_a = windowed.score_windows(model, t.cat([probe, pad_a]))[0]
            with_b = windowed.score_windows(model, t.cat([probe, pad_b]))[0]
            self.assertAlmostEqual(alone, with_a, places=6, msg=family)
            self.assertAlmostEqual(with_a, with_b, places=6, msg=family)
        self.assertGreaterEqual(checked, 5, "no pyod-backed family was checked")

    def test_detector_names_match_what_the_generic_trainer_writes(self):
        """`_train_wrapped` names its checkpoints `{FAMILY}_{i}` VERBATIM, and
        the loader looks for exactly the name in this tuple.

        It used to `.upper()` the family, which was a no-op while every family
        was an acronym and would now write `SPECTRALRESIDUAL_1.pth` for a
        detector the pool calls `SpectralResidual_1`. Asserted by rebuilding
        the name from its parts rather than by restating a casing rule, since
        the rule is now "whatever the pool says"."""
        for name in spec.ALL_DETECTORS:
            family, sep, index = name.rpartition("_")
            self.assertTrue(sep, name)
            self.assertIn(family, spec.DETECTOR_FAMILIES, name)
            self.assertEqual(f"{family}_{index}", name)


class TestSpecMatchesAppPy(unittest.TestCase):
    """The spec is the single owner of these lists; app.py must consume it
    rather than re-declaring them."""

    def _app_source(self):
        with open(os.path.join(_THIS, os.pardir, "app.py")) as f:
            return f.read()

    def test_app_py_has_no_duplicate_literals(self):
        src = self._app_source()
        self.assertIn("algorithm_list_instances = list(ALL_DETECTORS)", src)
        self.assertNotIn("'LOF_1', 'LOF_2', 'LOF_3', 'LOF_4'", src)
        self.assertNotIn('ALL_STAGES = {"ga", "thompson"', src)

    def test_sequential_call_sites_use_the_filtered_list(self):
        # Regression guard for the pre-existing bug: Thompson/GAN/off-by/MC in
        # the sequential path were handed the global detector list even when
        # run_app had already narrowed it to the models that loaded.
        # `still_usable()` IS that narrowed list, minus anything a previous
        # stage killed for slowness.
        src = self._app_source()
        for fragment in ("model_names=still_usable(),",
                         "test_data_for_gan, trained_models, still_usable(),",
                         "test_data_for_borderline, trained_models, still_usable(),",
                         "test_data_for_mc, trained_models, still_usable(),"):
            self.assertIn(fragment, src)
        self.assertIn("m for m in models_to_use if m not in slow", src)


class TestTransductiveFamilies(unittest.TestCase):
    """COF, SOS and SpectralResidual are exempt from
    `test_every_family_scores_a_batch_independently` because they cannot pass
    it. These are the properties that make the exemption safe — without them,
    exempting a family would be a hole rather than a decision.
    """

    def setUp(self):
        try:
            import numpy as np
            import torch as t
            import pyod.models as pyod_models
            from Algorithms import windowed
            from Algorithms.pyod_model import create_model, get_all_module_names
        except ImportError:
            self.skipTest("torch/pyod not installed in this interpreter")
        self.np, self.t, self.windowed = np, t, windowed
        self.create_model = create_model
        self.modules = get_all_module_names(pyod_models)
        self.rng = np.random.default_rng(0)
        # 25 rows minimum anywhere a score is taken: COF raises IndexError below
        # n_neighbors + 1 (21) and SpectralResidual needs score_window (3).
        self.train = t.tensor(self.rng.normal(size=(60, 4, 16)), dtype=t.float32)
        self.probe = t.tensor(self.rng.normal(size=(25, 4, 16)), dtype=t.float32)
        self.pad_a = t.tensor(self.rng.normal(size=(25, 4, 16)), dtype=t.float32)
        self.pad_b = t.tensor(self.rng.normal(size=(25, 4, 16)), dtype=t.float32) * 5

    # POLY and Series2Graph are univariate only, so they need their own probe:
    # `windows_as_rows` must yield ONE column. Both are also shrunk from their
    # production lengths (POLY's `window` is 200, Series2Graph's smallest
    # `pattern_length` is 50) because the property under test is refit-per-call,
    # which neither length changes.
    _UNIVARIATE = {"POLY", "Series2Graph"}
    _PROBE_KWARGS = {"POLY": {"power": 3, "window": 20},
                     "Series2Graph": {"pattern_length": 10, "rate": 1}}

    def _shape_for(self, family):
        return (1, 1) if family in self._UNIVARIATE else (4, 16)

    def _sample(self, family, n, scale=1.0):
        size = (n,) + self._shape_for(family)
        return self.t.tensor(self.rng.normal(size=size), dtype=self.t.float32) * scale

    def _probes(self, family):
        """(train, probe, pad_a, pad_b) at the width this family accepts."""
        if family not in self._UNIVARIATE:
            return self.train, self.probe, self.pad_a, self.pad_b
        return (self._sample(family, 60), self._sample(family, 25),
                self._sample(family, 25), self._sample(family, 25, scale=5.0))

    def _fitted(self, family, train=None):
        """A fitted estimator, from whichever library owns the family.

        The transductive set spans both backends now — COF, SOS and SR are
        PyOD, POLY is TSB-AD — so the probe cannot assume one factory.
        """
        train = self.train if train is None else train

        class _Loader:
            Y_windows = train

        if family in spec.TSBAD_FAMILIES:
            from Algorithms.tsbad_model import _TSBADEstimator
            model = _TSBADEstimator(family, 0.1, self._PROBE_KWARGS.get(family, {}))
        else:
            model = self.create_model(family, self.modules, contamination=0.1)
        try:
            self.windowed.fit_windows(model, _Loader())
        except ImportError as exc:      # Series2Graph, not fetched — see its own test
            self.skipTest(str(exc).splitlines()[0])
        return model

    def test_one_finite_score_per_row_on_a_whole_series_call(self):
        """The invariant the whole pipeline rests on. Catches SpectralResidual
        returning three scores for one row, and COF raising below 21 rows."""
        for family in sorted(spec.TRANSDUCTIVE_FAMILIES):
            with self.subTest(family=family):
                train, probe, pad_a, pad_b = self._probes(family)
                whole = self.t.cat([probe, pad_a, pad_b])
                scores = self.windowed.score_windows(
                    self._fitted(family, train), whole)
                self.assertEqual(scores.shape, (len(whole),))
                self.assertTrue(self.np.isfinite(scores).all())

    def test_the_same_entity_scores_identically_twice(self):
        """Two independent constructions AND fits, exact equality.

        This is the property that separated these three from TimeSeriesOD and
        AnomalyTransformer, which return different scores on two runs of
        identical input and expose no seed. Fitting twice rather than scoring
        twice also catches fit-time RNG leaking into the score.
        """
        for family in sorted(spec.TRANSDUCTIVE_FAMILIES):
            with self.subTest(family=family):
                train, probe, pad_a, _ = self._probes(family)
                whole = self.t.cat([probe, pad_a])
                first = self.windowed.score_windows(self._fitted(family, train), whole)
                second = self.windowed.score_windows(self._fitted(family, train), whole)
                self.np.testing.assert_array_equal(first, second)

    def test_the_exemption_is_earned(self):
        """They really are transductive. If a future pyod makes one of them
        inductive, this fails and says so — move it back into the strict test
        rather than leaving an exemption nobody re-examines."""
        for family in sorted(spec.TRANSDUCTIVE_FAMILIES):
            with self.subTest(family=family):
                train, probe, pad_a, pad_b = self._probes(family)
                model = self._fitted(family, train)
                n = len(probe)
                with_a = self.windowed.score_windows(
                    model, self.t.cat([probe, pad_a]))[:n]
                with_b = self.windowed.score_windows(
                    model, self.t.cat([probe, pad_b]))[:n]
                # The whole probe block rather than its first row: POLY leaves
                # the first `n_initial_` scores at zero by construction, so a
                # single fixed index can read 0.0 against 0.0 and look inductive
                # when the rest of the block plainly is not.
                self.assertFalse(
                    self.np.allclose(with_a, with_b, atol=1e-6),
                    msg=f"{family} now looks inductive; move it out of "
                        f"TRANSDUCTIVE_FAMILIES and into the strict test")

    def test_the_scoring_path_routes_them_to_a_single_batch(self):
        """Exempting a family from the strict test is only safe because
        `evaluate_model` never batches it. Without this, a fourth transductive
        family could be added, skipped by the strict test, and batched anyway."""
        from Utils import model_selection_utils as msu
        self.assertTrue(spec.TRANSDUCTIVE_FAMILIES <= set(spec.DETECTOR_FAMILIES))
        self.assertEqual(msu._TRANSDUCTIVE_MODELS, spec.TRANSDUCTIVE_FAMILIES)
        self.assertFalse(spec.TRANSDUCTIVE_FAMILIES & msu._WHOLE_SERIES_MODELS)
        self.assertFalse(spec.TRANSDUCTIVE_FAMILIES & msu._SINGLE_WINDOW_MODELS)


    def test_too_few_windows_names_the_detector_instead_of_unbinding_loss(self):
        """The guard that turns PyOD's `UnboundLocalError` into an explanation.

        Whatever the grid says today, a shorter entity or a wider window can put
        the count back under `batch_size`. PyOD's own failure names neither the
        detector nor the requirement and arrives four frames down, after it has
        silently trained on nothing.
        """
        model = self.create_model("AutoEncoder", self.modules, contamination=0.1,
                                  hidden_neuron_list=[64, 32], epoch_num=10)
        rows = self.t.tensor(self.rng.normal(size=(8, 9, 64)),
                             dtype=self.t.float32)   # 8 windows, batch_size 32

        class _Loader:
            Y_windows = rows

        with self.assertRaises(ValueError) as caught:
            self.windowed.fit_windows(model, _Loader())
        message = str(caught.exception)
        self.assertIn("AutoEncoder", message)
        self.assertIn("8 window", message)
        self.assertIn("32", message)
        self.assertIn("window_step", message)

    def test_a_fitted_deep_detector_can_be_checkpointed(self):
        """AutoEncoder must survive `logging_obj.save`, which is dill through torch.save.

        PyOD's deep base leaves `self.optimizer` on the detector, and a torch
        optimiser reaches a `torch._dynamo` config module: `TypeError: cannot
        pickle 'ConfigModuleInstance' object`, raised AFTER training finished,
        so the work was done and then discarded. Dropping the training-only
        state is what makes the checkpoint writable — and it must not change a
        single score, which is the other half of this test.
        """
        import io
        import dill
        model = self.create_model("AutoEncoder", self.modules, contamination=0.1,
                                  hidden_neuron_list=[64, 32], epoch_num=2)
        rows = self.t.tensor(self.rng.normal(size=(40, 3, 8)),
                             dtype=self.t.float32)

        class _Loader:
            Y_windows = rows

        self.windowed.fit_windows(model, _Loader())
        # fit_windows released it; nothing that scores may have been touched.
        self.assertIsNone(getattr(model, "optimizer", None))
        scores = self.windowed.score_windows(model, rows)

        buffer = io.BytesIO()
        self.t.save(model, buffer, pickle_module=dill)
        buffer.seek(0)
        reloaded = self.t.load(buffer, pickle_module=dill, weights_only=False)
        self.assertTrue(self.np.array_equal(
            scores, self.windowed.score_windows(reloaded, rows)))

    def test_the_guard_ignores_detectors_that_do_not_batch(self):
        """LOF and friends have no `batch_size`, and a guard that fired on them
        would break every point-wise family in the pool."""
        model = self.create_model("LOF", self.modules, contamination=0.1)
        rows = self.t.tensor(self.rng.normal(size=(8, 9, 64)),
                             dtype=self.t.float32)

        class _Loader:
            Y_windows = rows

        self.windowed.fit_windows(model, _Loader())      # must not raise


class TestTSBADFamilies(unittest.TestCase):
    """The eight families reached through the vendored TSB-AD subset.

    They are exempt from `test_every_family_scores_a_batch_independently`
    because each cuts its own subsequence out of the call and cannot score a
    one-row batch. This class asserts what holds instead — the properties the
    pipeline actually relies on — so the exemption is a decision rather than a
    gap. POLY carries the extra restrictions and is checked separately.
    """

    def setUp(self):
        try:
            import numpy as np
            import torch as t
            from Algorithms import windowed
            from Algorithms.tsbad_model import _TSBADEstimator, _class_for
        except ImportError:
            self.skipTest("torch not installed in this interpreter")
        self.np, self.t, self.windowed = np, t, windowed
        self.estimator, self.class_for = _TSBADEstimator, _class_for

    # Enough rows for the longest subsequence any of these cuts, and small
    # enough that eight fits stay under a few seconds.
    _KWARGS = {
        "KMEANSAD": {"k": 4, "window_size": 20, "stride": 1},
        "DONUT": {"win_size": 20, "num_epochs": 1},
        "OmniAnomaly": {"win_size": 20, "epochs": 1},
        "USAD": {"win_size": 20, "epochs": 1},
        "TRANAD": {"win_size": 20, "epochs": 1},
        # FITS is the one whose two parameters are coupled: it keeps `cut_freq`
        # frequencies of a window downsampled by DSR (4), so cut_freq must not
        # exceed floor(win_size/DSR)/2 + 1 or the linear layer is built at the
        # wrong width and the matmul fails. 100/4 -> 13 bins, so 6 is safe; the
        # production grid uses the same window for the same reason.
        "FITS": {"win_size": 100, "cut_freq": 6, "epochs": 1},
        "TIMESNET": {"win_size": 20, "epochs": 1},
        # The Graph Based group's multivariate member. Series2Graph needs no
        # entry: it is univariate, so `skip` below already excludes it.
        "MTADGAT": {"win_size": 20, "epochs": 1},
    }

    def test_every_family_resolves_to_a_class(self):
        """A family in TSBAD_FAMILIES that the vendored package cannot supply
        would only fail once someone selected it and waited for training."""
        for family in sorted(spec.TSBAD_FAMILIES):
            with self.subTest(family=family):
                cls, _channel_arg, scorer = self.class_for(family)
                self.assertTrue(callable(cls))
                self.assertIn(scorer, {"decision_function", "predict", "refit"})

    def test_every_family_has_its_own_grid(self):
        """Unlike the PyOD families there is no shared default: no two of these
        detectors take the same parameters, so a missing grid is unrecoverable."""
        from Model_Training.hyperparameter_grids import (FAMILY_GRIDS,
                                                         TSBAD_MODEL_GRIDS)
        for family in sorted(spec.TSBAD_FAMILIES):
            with self.subTest(family=family):
                self.assertIn(family, TSBAD_MODEL_GRIDS)
                self.assertIs(FAMILY_GRIDS[family], TSBAD_MODEL_GRIDS[family])
                # window_size 1 is the whole arrangement: one row per timestep
                # is the raw series these expect, and their own subsequence
                # length is a `detector__` key.
                self.assertEqual(FAMILY_GRIDS[family]["window_size"], [1])

    def test_one_finite_score_per_row_on_a_whole_series_call(self):
        """The invariant the pipeline rests on, and the reason these are routed
        to a single batch: one score per row, all finite."""
        rng = self.np.random.default_rng(0)
        # 800 training rows, not 300: these detectors hold back `validation_size`
        # (0.2) of what they are fitted on, and that HOLD-OUT must itself be
        # longer than the subsequence. FITS at win_size 100 therefore needs 500
        # training rows before it can cut a single validation window. The
        # entities in use clear this (SKAB ~700, SMD ~2400); a shorter one would
        # not, which is worth knowing from the test rather than from a run.
        train = self.t.tensor(rng.normal(size=(800, 1, 5)), dtype=self.t.float32)
        probe = self.t.tensor(rng.normal(size=(120, 1, 5)), dtype=self.t.float32)

        class _Loader:
            Y_windows = train

        # The foundation models are excluded here and covered by
        # `test_foundation_models_resolve_and_are_routed` instead. They download
        # pretrained weights from the HuggingFace hub on first use and take
        # 13-15s per fit, which would turn an offline seven-second suite into a
        # network-dependent minute. Their one-score-per-row behaviour was
        # measured directly instead: OFA, TIMESFM and CHRONOS each returned
        # (300,) finite scores for a 300-row multivariate call.
        skip = spec.UNIVARIATE_FAMILIES | set(spec.DETECTOR_GROUPS["FM"])
        for family in sorted(spec.TSBAD_FAMILIES - skip):
            with self.subTest(family=family):
                model = self.estimator(family, 0.1, self._KWARGS[family])
                self.windowed.fit_windows(model, _Loader())
                scores = self.windowed.score_windows(model, probe)
                self.assertEqual(scores.shape, (len(probe),))
                self.assertTrue(self.np.isfinite(scores).all())

    def test_foundation_models_resolve_and_are_routed(self):
        """What can be asserted about the FMs without a network round-trip.

        The pool's first Foundation Models. Everything here is structural: that
        the class resolves, that the grid exists and keeps window_size 1, and
        that scoring never hands them a partial batch. Actually fitting them
        needs weights from the hub, so it is not done in this suite.
        """
        from Model_Training.hyperparameter_grids import FAMILY_GRIDS
        from Utils import model_selection_utils as msu
        families = spec.DETECTOR_GROUPS["FM"]
        self.assertEqual(sorted(families), ["CHRONOS", "OFA", "TIMESFM"])
        for family in families:
            with self.subTest(family=family):
                cls, channel_arg, scorer = self.class_for(family)
                self.assertTrue(callable(cls))
                # Each takes the channel count, which is what lets the two
                # univariate-marked ones run their per-channel loop.
                self.assertIn(channel_arg, ("enc_in", "input_c"))
                self.assertEqual(FAMILY_GRIDS[family]["window_size"], [1])
                self.assertIn(family, msu._WHOLE_SERIES_MODELS)
        # CHRONOS is ours over `chronos-forecasting`, not TSB-AD's autogluon
        # route; the dotted path in the spec is what makes that reachable.
        self.assertEqual(self.class_for("CHRONOS")[0].__module__,
                         "Algorithms.chronos_detector")

    def test_chronos_asks_for_the_reproducible_variant(self):
        """Bolt, not T5, and the difference is not cosmetic.

        Chronos-T5 forecasts by sampling: two calls on identical input measured
        1.7e-01 apart on one pipeline and 4.9e-01 across a scoring call. That is
        the same property PyOD's TimeSeriesOD and AnomalyTransformer are kept
        out of this pool for, so a T5 checkpoint would make a detector whose
        rank changes between two reads of the same run. Bolt does direct
        quantile regression with no sampling and measures 0.000e+00.

        Asserted against the source rather than by loading weights, so the guard
        costs nothing and needs no network.
        """
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, os.pardir, "Algorithms",
                               "chronos_detector.py")) as f:
            source = f.read()
        self.assertIn("amazon/chronos-bolt-", source)
        self.assertNotIn("amazon/chronos-t5-", source)

    def test_they_are_routed_to_a_single_batch(self):
        """Exempting them from the batch-independence test is only safe because
        `evaluate_model` never hands them a partial batch."""
        from Utils import model_selection_utils as msu
        self.assertTrue(spec.TSBAD_FAMILIES <= set(spec.DETECTOR_FAMILIES))
        for family in spec.TSBAD_FAMILIES:
            self.assertTrue(
                family in msu._WHOLE_SERIES_MODELS
                or family in msu._TRANSDUCTIVE_MODELS,
                f"{family} would be scored in 128-row batches, which is "
                f"shorter than the subsequence it cuts")

    def test_poly_refuses_multivariate_input_by_name(self):
        """Table I marks POLY `U`. Selecting it on SKAB or SMD must say so —
        numpy's "Polynomial must be 1d only", four frames down, does not."""
        rng = self.np.random.default_rng(0)
        wide = self.t.tensor(rng.normal(size=(60, 9, 1)), dtype=self.t.float32)

        class _Loader:
            Y_windows = wide

        model = self.estimator("POLY", 0.1, {"power": 3, "window": 20})
        with self.assertRaises(ValueError) as caught:
            self.windowed.fit_windows(model, _Loader())
        self.assertIn("POLY", str(caught.exception))
        self.assertIn("univariate", str(caught.exception))

    def test_every_checkpoint_load_goes_through_one_loader(self):
        """A save and a load must use the same pickler.

        `Logger.save_torch_model` writes with `dill`, because PyOD 3 defines
        LSTMAD's network as a class inside a function and stdlib pickle cannot
        serialise that — it raised after `fit`, leaving a truncated .pth that
        read as a trained model, which is why no LSTMAD checkpoint has ever
        existed. A `torch.load` that forgets `pickle_module` would then fail
        only on whichever of the three load sites a given user happens to hit.
        This is the same "two places must know one rule" defect that cost
        LSTMAD its training in the first place, so it gets the same treatment:
        one owner, asserted.
        """
        import re
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(here, os.pardir)
        targets = ["app.py",
                   os.path.join("Model_Selection", "model_selection.py"),
                   os.path.join("Services", "mmodel", "__init__.py")]
        raw = re.compile(r"\b(?:t|torch)\.load\s*\(")
        for rel in targets:
            with open(os.path.join(root, rel)) as f:
                source = f.read()
            with self.subTest(file=rel):
                self.assertFalse(
                    raw.search(source),
                    f"{rel} calls torch.load directly; use "
                    f"Utils.model_io.load_checkpoint so the pickler matches "
                    f"what Logger.save_torch_model wrote")
                self.assertIn("load_checkpoint", source, rel)

    def test_the_trainer_and_the_scorer_agree_on_who_needs_one_batch(self):
        """Both places that size a batch must read the same set.

        They did not, and it cost LSTMAD entirely. `evaluate_model` knew to give
        it the whole series, but `TrainModels._diagnostic_batch_size` only knew
        about the transductive and TSB-AD families — so the post-fit plotting
        loop ran at batch_size 8 against a 50-150 step window and raised
        "negative dimensions are not allowed". That raise happens BEFORE
        `logging_obj.save`, so no checkpoint was written and LSTMAD could not be
        trained on any entity. One owner, read by both, is the fix.
        """
        import os
        from Utils import model_selection_utils as msu
        self.assertEqual(msu._WHOLE_SERIES_MODELS, spec.WHOLE_SERIES_FAMILIES)
        self.assertFalse(spec.WHOLE_SERIES_FAMILIES & spec.TRANSDUCTIVE_FAMILIES,
                         "a family cannot need one batch for both reasons")
        self.assertIn("LSTMAD", spec.WHOLE_SERIES_FAMILIES)
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, os.pardir, "Model_Training", "train.py")) as f:
            trainer = f.read()
        self.assertIn("WHOLE_SERIES_FAMILIES", trainer,
                      "_diagnostic_batch_size must size from the shared set, "
                      "not from a list of its own")

    def test_poly_refuses_a_call_shorter_than_one_window(self):
        """POLY computes `N = floor(n_rows / window)` and then takes
        `n_rows % N`, so a short call raises `ZeroDivisionError: integer modulo
        by zero` from inside the vendored code — naming neither the detector nor
        the requirement. Thompson does hand it short calls on small entities."""
        rng = self.np.random.default_rng(0)
        short = self.t.tensor(rng.normal(size=(10, 1, 1)), dtype=self.t.float32)
        long = self.t.tensor(rng.normal(size=(120, 1, 1)), dtype=self.t.float32)

        class _Loader:
            Y_windows = long

        model = self.estimator("POLY", 0.1, {"power": 3, "window": 20})
        self.windowed.fit_windows(model, _Loader())
        with self.assertRaises(ValueError) as caught:
            self.windowed.score_windows(model, short)
        self.assertIn("POLY", str(caught.exception))
        self.assertIn("20", str(caught.exception))

    def test_the_univariate_restriction_is_declared_where_the_ui_reads_it(self):
        """`UNIVARIATE_FAMILIES` is what lets the run page warn before a run
        rather than after. A restriction enforced only in the adapter would be
        invisible until stage 3."""
        self.assertIn("POLY", spec.UNIVARIATE_FAMILIES)
        self.assertIn("TIMESFM", spec.UNIVARIATE_FAMILIES)
        self.assertIn("Series2Graph", spec.UNIVARIATE_FAMILIES)
        self.assertTrue(spec.UNIVARIATE_FAMILIES <= set(spec.DETECTOR_FAMILIES))

    def test_the_multivariate_restriction_mirrors_the_univariate_one(self):
        """ABOD is dropped on a 1-channel entity for the same reason POLY is
        dropped on a 38-channel one: the method cannot mean what it says
        there."""
        self.assertIn("ABOD", spec.MULTIVARIATE_FAMILIES)
        self.assertTrue(spec.MULTIVARIATE_FAMILIES <= set(spec.DETECTOR_FAMILIES))

    def test_no_family_is_restricted_to_both_widths(self):
        """A family in both sets would be droppable everywhere, leaving it
        selectable in the UI and runnable nowhere."""
        self.assertEqual(spec.MULTIVARIATE_FAMILIES & spec.UNIVARIATE_FAMILIES,
                         frozenset())

    def test_the_multivariate_restriction_is_enforced_by_dropping(self):
        """Same requirement as the univariate drop: ABOD raises nothing on 1
        channel, so nothing downstream would catch it."""
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, os.pardir, "app.py")) as f:
            source = f.read()
        self.assertIn("MULTIVARIATE_FAMILIES", source)

    def test_the_two_vocabularies_of_the_restriction_agree(self):
        """The UI reads `pipeline_spec`, the adapter enforces `tsbad_model`.

        Two independent lists, so a family added to one and not the other either
        fails at stage 3 with no warning, or warns about a run that would have
        worked. Only TSB-AD families can appear in the adapter's dict, which is
        the whole of what it can refuse.
        """
        from Algorithms import tsbad_model
        self.assertEqual(set(tsbad_model.UNIVARIATE_ONLY),
                         set(spec.UNIVARIATE_FAMILIES) & set(spec.TSBAD_FAMILIES))
        # Every entry carries its own reason: the two are restricted for
        # different causes and one shared sentence would misdescribe one of them.
        for family, reason in tsbad_model.UNIVARIATE_ONLY.items():
            with self.subTest(family=family):
                self.assertTrue(reason and reason[0].islower(),
                                "reason is spliced mid-sentence after a colon")

    def test_the_restriction_is_enforced_by_dropping_not_only_by_raising(self):
        """`app.py` must FILTER the pool, not rely on the adapter's ValueError.

        The adapter raises from inside `TrainModels.train_models`, whose family
        loop has no per-family recovery — so one univariate-only detector on a
        multivariate entity aborted training for every family after it. The
        declaration alone was inert: nothing read UNIVARIATE_FAMILIES.
        """
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, os.pardir, "app.py")) as f:
            source = f.read()
        self.assertIn("UNIVARIATE_FAMILIES", source)
        self.assertIn("family_of", source)
        # The channel count is what the decision turns on, and it is only known
        # after the test entity loads.
        self.assertIn("n_channels", source)

    def test_series2graph_is_absent_by_licence_and_says_how_to_get_it(self):
        """Series2Graph is the one pool detector whose source is NOT here.

        Its file is patent-encumbered and licensed for research use only, unlike
        the Apache-2.0 TSB-AD code vendored around it, so it is fetched rather
        than redistributed. Three things have to hold for that to be a design
        rather than a broken family: it is gitignored, the failure names the
        command that fixes it, and the failure arrives at TRAINING time — not
        three stages later when something reads `decision_scores_`.

        The test passes either way, because a developer who has run the fetch
        should not see a red suite for having done so.
        """
        import os
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
        with open(os.path.join(root, ".gitignore")) as f:
            self.assertIn("Algorithms/tsb_ad/models/Series2Graph.py", f.read())

        model = self.estimator("Series2Graph", 0.1, {"pattern_length": 20})
        series = self.np.random.default_rng(0).normal(size=(500, 1))
        try:
            model.fit(series)
        except ImportError as exc:
            self.assertIn("python -m Algorithms.tsb_ad.fetch_series2graph", str(exc))
            self.assertIn("research use", str(exc))
            return
        # Fetched: then it must behave like every other whole-series detector.
        scores = model.decision_function(series)
        self.assertEqual(scores.shape, (len(series),))
        self.assertTrue(self.np.isfinite(scores).all())

    def test_series2graph_scores_the_series_it_is_given(self):
        """The vendored `score(query_length, dataset)` never reads `dataset` —
        it re-reports the series `fit` built its graph from. Fitting and scoring
        the same length hides that, which is why this scores a different one:
        UCR splits train from test, so one score per TRAINING row reached
        `windowed.score_windows` as a shape mismatch and the family could not be
        trained at all. The 'refit' scorer is what fixes it.
        """
        model = self.estimator("Series2Graph", 0.1, {"pattern_length": 20})
        rng = self.np.random.default_rng(0)
        try:
            model.fit(rng.normal(size=(300, 1)))
        except ImportError:
            self.skipTest("Series2Graph not fetched")
        for n in (300, 700):
            with self.subTest(rows=n):
                scores = model.decision_function(rng.normal(size=(n, 1)))
                self.assertEqual(scores.shape, (n,))

    def test_timesfm_refuses_a_multivariate_call_by_name(self):
        """A cost refusal, not a capability one — TimesFM's per-channel loop
        runs fine, it just costs ~13 min per scoring call on 38 channels. The
        message has to say so without claiming the model cannot do it, and it
        must name the detector rather than surface as a timeout an hour later.
        """
        model = self.estimator("TIMESFM", 0.1, {"win_size": 64})
        probe = self.np.zeros((200, 4))
        with self.assertRaises(ValueError) as caught:
            model.fit(probe)
        message = str(caught.exception)
        self.assertIn("TIMESFM", message)
        self.assertIn("univariate only", message)
        self.assertIn("4 channels", message)
        # Refused on the way in, before any weights are fetched from the hub.
        self.assertIsNone(model._model)
        # Univariate still passes the width guard (it stops before any forecast
        # here only because fit() on a refit scorer merely constructs).
        self.assertIsNone(model._check_width(1))


class TestDetectorGroups(unittest.TestCase):
    """The paper's Table I taxonomy, which the run page's group buttons read."""

    def test_every_family_is_in_exactly_one_group(self):
        seen = [f for members in spec.DETECTOR_GROUPS.values() for f in members]
        self.assertEqual(sorted(seen), sorted(spec.DETECTOR_FAMILIES))
        self.assertEqual(len(seen), len(set(seen)), "a family is in two groups")

    def test_group_of_agrees_with_the_map(self):
        for group, members in spec.DETECTOR_GROUPS.items():
            for family in members:
                self.assertEqual(spec.group_of(family), group)
        self.assertIsNone(spec.group_of("NOT_A_FAMILY"))

    def test_the_knn_collision_is_deliberate(self):
        """Our NN family is k-Nearest Neighbors and belongs to Stat; the group
        called NN is Neural Networks. Pinned because the names invite the
        opposite assumption and a 'fix' would silently mis-file it."""
        self.assertEqual(spec.group_of("NN"), "Stat")
        self.assertNotIn("NN", spec.DETECTOR_GROUPS["NN"])

    def test_the_paper_s_three_groups_are_all_present(self):
        """The paper's three, plus one this pool adds.

        Table I has NN, Stat and FM. `Graph` is a fourth, so the taxonomy here
        is the paper's EXTENDED rather than the paper's — asserted explicitly so
        the divergence is a decision on record instead of something a reader of
        Table I has to notice.
        """
        self.assertEqual(set(spec.DETECTOR_GROUPS), {"NN", "Stat", "FM", "Graph"})
        self.assertEqual(sorted(spec.DETECTOR_GROUPS["Graph"]),
                         ["LUNAR", "MTADGAT", "Series2Graph"])

    def test_the_graph_group_keeps_upstreams_boundary(self):
        """SOS and COF are near misses that stay in Stat.

        Both are defined over something graph-shaped — SOS over an affinity
        graph, COF over a set-based nearest path — and moving them would make
        the new group look better populated than it is. PyOD's own taxonomy
        files them "Probabilistic" and "Proximity-Based" while filing LUNAR
        under "Graph-based", and following upstream beats inventing a second
        classification only this repository would use.
        """
        for family in ("SOS", "COF"):
            self.assertEqual(spec.group_of(family), "Stat", family)
        self.assertEqual(spec.group_of("LUNAR"), "Graph")


class TestGridReadback(unittest.TestCase):
    """`grid_combinations` is what tells the run page "LOF_2 is contamination
    0.15". It reimplements sklearn's ParameterGrid ordering in stdlib so the
    web UI can read it without sklearn — which is only safe while the two
    orderings agree."""

    def setUp(self):
        from Model_Training.hyperparameter_grids import (FAMILY_GRIDS,
                                                         grid_combinations,
                                                         varying_keys)
        self.FAMILY_GRIDS = FAMILY_GRIDS
        self.grid_combinations = grid_combinations
        self.varying_keys = varying_keys

    def test_every_pool_family_has_a_grid(self):
        from Utils.pipeline_spec import DETECTOR_FAMILIES
        self.assertEqual(sorted(self.FAMILY_GRIDS), sorted(DETECTOR_FAMILIES))

    def test_grid_size_matches_the_instance_count(self):
        """A family's grid must expand to exactly as many combinations as
        ALL_DETECTORS has instances, or the run page numbers them wrongly."""
        from Utils.pipeline_spec import ALL_DETECTORS, family_of
        counts = {}
        for detector in ALL_DETECTORS:
            counts[family_of(detector)] = counts.get(family_of(detector), 0) + 1
        for family, grid in self.FAMILY_GRIDS.items():
            with self.subTest(family=family):
                self.assertEqual(len(self.grid_combinations(grid)), counts[family])

    def test_ordering_matches_sklearn(self):
        """The reason the stdlib copy is allowed to exist."""
        try:
            from sklearn.model_selection import ParameterGrid
        except ImportError:
            self.skipTest("sklearn not installed in this interpreter")
        for family, grid in self.FAMILY_GRIDS.items():
            with self.subTest(family=family):
                self.assertEqual(self.grid_combinations(grid), list(ParameterGrid(grid)))

    def test_varying_keys_are_the_ones_that_differ(self):
        for family, grid in self.FAMILY_GRIDS.items():
            with self.subTest(family=family):
                varying = self.varying_keys(grid)
                for key in varying:
                    self.assertGreater(len(grid[key]), 1)
                for key, values in grid.items():
                    if len(values) == 1:
                        self.assertNotIn(key, varying)

    def test_no_family_varies_contamination_alone(self):
        """Not one family may distinguish its instances by contamination.

        It sets `threshold_` and `labels_` and never reaches
        `decision_function`, and this pipeline scores with its own threshold
        sweep — so a family sweeping it alone is one detector wearing four
        names. Measured before the fix: every instance of all twelve such
        families scored 0.000e+00 apart on both SKAB and SMD.

        Twelve families are now on a real parameter. Seven took TSB-AD's own
        sweep (LOF, CBLOF, IFOREST, HBOS, PCA, OCSVM, MCD), separating by
        3.1e-01, 1.5e+00, 6.6e-02, 2.4e+00, 1.2e+02, 2.5e+00 and 2.7e+00. The
        other five have no TSB-AD entry to copy — the name ABOD does not occur
        anywhere in that package, and its only SpectralResidual entry is a
        `periodicity` — so they took the parameter PyOD's own estimator exposes:
        ABOD `n_neighbors`, KDE `bandwidth`, COF `n_neighbors`, SOS
        `perplexity`, SpectralResidual `score_window`.

        Written as "none" rather than as a shrinking allow-list so a NEW family
        cannot be added on a contamination sweep without this failing.
        """
        contamination_only = {f for f, g in self.FAMILY_GRIDS.items()
                              if self.varying_keys(g) == ["contamination"]}
        self.assertEqual(contamination_only, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
