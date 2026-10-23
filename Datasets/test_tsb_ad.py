"""The TSB-AD loader: file names, entity ids, the train/test split, labels.

Built on small CSVs written to a temporary data root in the benchmark's own
layout, so the tests do not need the 1 GB download. One test reads the real
files when they are present.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Datasets import tsb_ad  # noqa: E402
from Datasets.load import load_data, provides_real_labels  # noqa: E402

REAL_ROOT = "/raid0_ssd2/maxoud/projects/RAMSeS/Datasets"


def write_series(directory, name, values, labels):
    """One TSB-AD CSV: a column per channel, then Label."""
    values = np.atleast_2d(values)
    header = ("Data" if values.shape[0] == 1
              else ",".join(str(c) for c in range(values.shape[0])))
    with open(os.path.join(directory, name), "w") as f:
        f.write(header + ",Label\n")
        for t in range(values.shape[1]):
            f.write(",".join(repr(float(v)) for v in values[:, t]) + f",{int(labels[t])}\n")


class TestNames(unittest.TestCase):

    def test_fields_of_a_file_name(self):
        meta = tsb_ad.parse_name("001_NAB_id_1_Facility_tr_1007_1st_2014.csv")
        self.assertEqual(meta["entity"], "001_NAB_id_1_Facility")
        self.assertEqual((meta["index"], meta["source"], meta["id"], meta["domain"]),
                         ("001", "NAB", 1, "Facility"))
        self.assertEqual((meta["train_end"], meta["first_anomaly"]), (1007, 2014))
        self.assertEqual(meta["stem"], "001_NAB_id_1_Facility_tr_1007_1st_2014")

    def test_underscores_inside_the_source_or_domain(self):
        meta = tsb_ad.parse_name("042_MITDB_id_3_Medical_ECG_tr_500_1st_700")
        self.assertEqual((meta["source"], meta["domain"]), ("MITDB", "Medical_ECG"))
        self.assertEqual(meta["entity"], "042_MITDB_id_3_Medical_ECG")

    def test_other_names_are_not_series(self):
        for name in ("README.txt", "001_UCR_Anomaly_x_35000_52000_52620.txt",
                     "notes.csv", "001_NAB_id_1_Facility.csv"):
            self.assertIsNone(tsb_ad.entity_from_filename(name), name)

    def test_dataset_keys_are_case_insensitive(self):
        for key in ("tsb-ad-u", "TSB-AD-U", "Tsb-Ad-M"):
            self.assertTrue(tsb_ad.is_tsb_ad(key))
        self.assertFalse(tsb_ad.is_tsb_ad("tsb-ad"))


class TestLoader(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.u = os.path.join(self.root, "TSB-AD-U")
        self.m = os.path.join(self.root, "TSB-AD-M")
        os.makedirs(self.u)
        os.makedirs(self.m)
        rng = np.random.default_rng(0)
        # Univariate: 1000 samples, train 400; anomalies at 600..609 and 900..904.
        self.u_values = rng.normal(size=1000) * 3 + 10
        self.u_labels = np.zeros(1000, dtype=int)
        self.u_labels[600:610] = 1
        self.u_labels[900:905] = 1
        write_series(self.u, "007_YAHOO_id_7_Synthetic_tr_400_1st_600.csv",
                     self.u_values, self.u_labels)
        # Multivariate: 3 channels, 800 samples, train 300; one anomaly 500..519.
        self.m_values = rng.normal(size=(3, 800)) * np.array([[1.0], [5.0], [50.0]])
        self.m_labels = np.zeros(800, dtype=int)
        self.m_labels[500:520] = 1
        write_series(self.m, "012_SMD_id_2_Facility_tr_300_1st_500.csv",
                     self.m_values, self.m_labels)

    def tearDown(self):
        self._tmp.cleanup()

    def load(self, dataset, entity, group, **kw):
        return load_data(dataset=dataset, group=group, entities=entity, root_dir=self.root,
                         verbose=False, **kw).entities[0]

    def test_train_is_before_train_end_and_test_after_it(self):
        train = self.load("tsb-ad-u", "007_YAHOO_id_7_Synthetic", "train", normalize=False)
        test = self.load("tsb-ad-u", "007_YAHOO_id_7_Synthetic", "test", normalize=False)
        np.testing.assert_allclose(train.Y[0], self.u_values[:400])
        np.testing.assert_allclose(test.Y[0], self.u_values[400:])
        self.assertEqual(train.Y.shape[1] + test.Y.shape[1], 1000)
        self.assertEqual(train.name, "007_YAHOO_id_7_Synthetic")

    def test_test_labels_are_the_label_column_of_the_test_part(self):
        test = self.load("tsb-ad-u", "007_YAHOO_id_7_Synthetic", "test")
        self.assertEqual(test.labels.shape, (1, 600))
        np.testing.assert_array_equal(test.labels[0], self.u_labels[400:])
        self.assertEqual(int(test.labels.sum()), 15)
        # The training part carries no labels, as for every other loader.
        train = self.load("tsb-ad-u", "007_YAHOO_id_7_Synthetic", "train")
        self.assertEqual(int(train.labels.sum()), 0)

    def test_scaled_with_the_training_range(self):
        train = self.load("tsb-ad-u", "007_YAHOO_id_7_Synthetic", "train")
        test = self.load("tsb-ad-u", "007_YAHOO_id_7_Synthetic", "test")
        self.assertAlmostEqual(train.Y.min(), 0.0)
        self.assertAlmostEqual(train.Y.max(), 1.0)
        lo, hi = self.u_values[:400].min(), self.u_values[:400].max()
        np.testing.assert_allclose(test.Y[0], (self.u_values[400:] - lo) / (hi - lo))

    def test_multivariate_keeps_every_channel(self):
        train = self.load("TSB-AD-M", "012_SMD_id_2_Facility", "train")
        test = self.load("TSB-AD-M", "012_SMD_id_2_Facility", "test")
        self.assertEqual((train.Y.shape, test.Y.shape), ((3, 300), (3, 500)))
        np.testing.assert_array_equal(test.labels[0], self.m_labels[300:])
        # Each channel is scaled on its own range.
        np.testing.assert_allclose(train.Y.min(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(train.Y.max(axis=1), 1.0)

    def test_entity_aliases_resolve_to_one_series(self):
        expected = self.load("tsb-ad-u", "007_YAHOO_id_7_Synthetic", "test").Y
        for alias in ("007_YAHOO_id_7_Synthetic_tr_400_1st_600",
                      "007_YAHOO_id_7_Synthetic_tr_400_1st_600.csv", "7", "007",
                      ["007_YAHOO_id_7_Synthetic"]):
            np.testing.assert_array_equal(self.load("tsb-ad-u", alias, "test").Y, expected)
            self.assertEqual(tsb_ad.canonical_entity(self.root, "tsb-ad-u", alias),
                             "007_YAHOO_id_7_Synthetic")

    def test_an_unknown_series_is_a_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "No TSB-AD series"):
            self.load("tsb-ad-u", "999_NOPE_id_1_X", "test")
        self.assertEqual(tsb_ad.canonical_entity(self.root, "tsb-ad-u", "nope"), "nope")

    def test_downsampling_takes_block_maxima_labels_included(self):
        test = self.load("tsb-ad-u", "007_YAHOO_id_7_Synthetic", "test",
                         downsampling=5, min_length=10)
        self.assertEqual(test.Y.shape[1], 120)
        self.assertEqual(test.labels.shape[1], 120)
        # 600..609 -> test 200..209 -> blocks 40, 41; 900..904 -> test 500..504 -> block 100.
        self.assertEqual(sorted(np.flatnonzero(test.labels[0])), [40, 41, 100])

    def test_min_length_turns_downsampling_off(self):
        # The training part would be 400 / 5 = 80 samples, under min_length.
        test = self.load("tsb-ad-u", "007_YAHOO_id_7_Synthetic", "test",
                         downsampling=5, min_length=100)
        self.assertEqual(test.Y.shape[1], 600)

    def test_the_loader_reports_real_labels(self):
        self.assertTrue(provides_real_labels("tsb-ad-u", "007_YAHOO_id_7_Synthetic", self.root))
        self.assertTrue(provides_real_labels("TSB-AD-M", "x", self.root))
        self.assertFalse(provides_real_labels("skab", "1", self.root))

    def test_index_lists_every_series_once(self):
        self.assertEqual(tsb_ad.index(self.u),
                         {"007_YAHOO_id_7_Synthetic": "007_YAHOO_id_7_Synthetic_tr_400_1st_600.csv"})


@unittest.skipUnless(os.path.isdir(os.path.join(REAL_ROOT, "TSB-AD-U")),
                     "the TSB-AD download is not on this machine")
class TestTheRealFiles(unittest.TestCase):

    def test_every_file_parses_and_ids_are_unique(self):
        for sub, expected in (("TSB-AD-U", 870), ("TSB-AD-M", 200)):
            directory = os.path.join(REAL_ROOT, sub)
            files = [f for f in os.listdir(directory) if f.endswith(".csv")]
            self.assertEqual(len(tsb_ad.index(directory)), len(files))
            self.assertEqual(len(files), expected)

    def test_one_series_of_each(self):
        u = load_data("tsb-ad-u", "test", "001_NAB_id_1_Facility", root_dir=REAL_ROOT,
                      verbose=False).entities[0]
        self.assertEqual(u.Y.shape, (1, 4031 - 1007))
        self.assertGreater(int(u.labels.sum()), 0)
        m = load_data("tsb-ad-m", "train", "002_MSL_id_1_Sensor", root_dir=REAL_ROOT,
                      verbose=False).entities[0]
        self.assertEqual(m.Y.shape, (55, 500))


if __name__ == "__main__":
    unittest.main()
