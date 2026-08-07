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

    def test_stale_checkpoint_names_are_not_selectable(self):
        # SMD entities carry leftover RNN_*.pth; validation is against the
        # canonical list, never against what happens to be on disk.
        with self.assertRaises(ValueError):
            spec.parse_detectors("RNN_1,LOF_1")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
