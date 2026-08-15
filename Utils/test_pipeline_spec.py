"""
Tests for Utils/pipeline_spec.py — the shared detector/stage vocabulary.

Loaded by file path (stdlib-only module) so the suite never imports torch via
Utils/utils.py. Run with `pytest Utils/test_pipeline_spec.py` or
`python -m unittest Utils.test_pipeline_spec`.
"""

import importlib.util
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
        """`train_pyod` names its checkpoints `{FAMILY.upper()}_{i}`, and the
        loader looks for exactly the name in this tuple. A lower- or mixed-case
        family here would train `IFOREST_1.pth` and then fail to find
        `IForest_1.pth`."""
        for family in spec.DETECTOR_FAMILIES:
            self.assertEqual(family, family.upper(), family)


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
        src = self._app_source()
        for fragment in ("model_names=models_to_use,",
                         "test_data_for_gan, trained_models, models_to_use,",
                         "test_data_for_borderline, trained_models, models_to_use,",
                         "test_data_for_mc, trained_models, models_to_use,"):
            self.assertIn(fragment, src)


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

    # POLY is univariate only, so it needs its own probe: `windows_as_rows`
    # must yield ONE column. Its production `window` is 200, which would need a
    # 200-row call to exercise, so the probe uses 20 — the property under test
    # is refit-per-call, which the window length does not change.
    _UNIVARIATE = {"POLY"}
    _PROBE_KWARGS = {"POLY": {"power": 3, "window": 20}}

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
        self.windowed.fit_windows(model, _Loader())
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
        "OMNIANOMALY": {"win_size": 20, "epochs": 1},
        "USAD": {"win_size": 20, "epochs": 1},
        "TRANAD": {"win_size": 20, "epochs": 1},
        # FITS is the one whose two parameters are coupled: it keeps `cut_freq`
        # frequencies of a window downsampled by DSR (4), so cut_freq must not
        # exceed floor(win_size/DSR)/2 + 1 or the linear layer is built at the
        # wrong width and the matmul fails. 100/4 -> 13 bins, so 6 is safe; the
        # production grid uses the same window for the same reason.
        "FITS": {"win_size": 100, "cut_freq": 6, "epochs": 1},
        "TIMESNET": {"win_size": 20, "epochs": 1},
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

        for family in sorted(spec.TSBAD_FAMILIES - spec.UNIVARIATE_FAMILIES):
            with self.subTest(family=family):
                model = self.estimator(family, 0.1, self._KWARGS[family])
                self.windowed.fit_windows(model, _Loader())
                scores = self.windowed.score_windows(model, probe)
                self.assertEqual(scores.shape, (len(probe),))
                self.assertTrue(self.np.isfinite(scores).all())

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
        self.assertTrue(spec.UNIVARIATE_FAMILIES <= set(spec.DETECTOR_FAMILIES))


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
        """FM is empty today. It stays listed so the taxonomy is visible and a
        foundation model has an obvious home."""
        self.assertEqual(set(spec.DETECTOR_GROUPS), {"NN", "Stat", "FM"})


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

    def test_contamination_only_families_are_recorded_as_such(self):
        """Nine families vary contamination alone, which does not enter
        decision_function — so their instances are identical to the pipeline.
        This pins the current state so the fix is a deliberate, visible change
        rather than something that quietly drifts."""
        contamination_only = {f for f, g in self.FAMILY_GRIDS.items()
                              if self.varying_keys(g) == ["contamination"]}
        self.assertEqual(contamination_only,
                         {"LOF", "CBLOF", "ABOD", "KDE",
                          "IFOREST", "HBOS", "PCA", "OCSVM", "MCD",
                          # Added knowing it: these three join the same pending
                          # fix rather than getting a different one first.
                          # SOS is the extreme case — Algorithms/sos.py does not
                          # even pass contamination to the estimator.
                          "COF", "SOS", "SR"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
