import os
import tempfile
import unittest

from Model_Selection.Sensitivity_robustness.plot_retention import prune_timestamped


class TestPruneTimestamped(unittest.TestCase):
    """Retention for the figures off-by and GAN mint fresh on every run.

    The alternative was overwriting in place, which would have made the WebUI's
    "n older" count dead and thrown away any comparison between runs. Keeping
    the newest few caps the growth and leaves both intact — so what matters here
    is that the grouping matches WebUI.plots.dedupe_timestamped exactly: what
    this deletes and what the page hides have to be the same set.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, name):
        with open(os.path.join(self.d, name), "wb") as f:
            f.write(b"\x89PNG")

    def _names(self):
        return sorted(os.listdir(self.d))

    def test_keeps_the_newest_three_of_a_group(self):
        for day in range(1, 8):
            self._touch(f"SKAB_7_Misclassified Anomalies_2026-05-0{day}_10-00-00.png")
        removed = prune_timestamped(self.d)
        self.assertEqual(len(removed), 4)
        self.assertEqual(self._names(), [
            "SKAB_7_Misclassified Anomalies_2026-05-05_10-00-00.png",
            "SKAB_7_Misclassified Anomalies_2026-05-06_10-00-00.png",
            "SKAB_7_Misclassified Anomalies_2026-05-07_10-00-00.png",
        ])

    def test_groups_are_independent_and_include_the_trailing_underscore(self):
        """The four naming irregularities that actually occur on disk — off-by's
        trailing underscore and its literal space, GAN's underscored form — are
        separate groups, so pruning one cannot empty another."""
        for day in range(1, 6):
            self._touch(f"Data_vs_DataWithAnomalies_2026-05-0{day}_10-00-00_.png")
            self._touch(f"SKAB_7_Data_vs_DataWithAnomalies_2026-05-0{day}_10-00-00.png")
            self._touch(f"SKAB_7_Misclassified_Anomalies_2026-05-0{day}_10-00-00_.png")
        prune_timestamped(self.d)
        self.assertEqual(len(self._names()), 9)
        for stem in ("Data_vs_DataWithAnomalies_2026",
                     "SKAB_7_Data_vs_DataWithAnomalies_2026",
                     "SKAB_7_Misclassified_Anomalies_2026"):
            self.assertEqual(sum(1 for n in self._names() if n.startswith(stem)), 3)

    def test_untimestamped_files_are_never_touched(self):
        """The same directory holds the surrogate trees and the importance
        figure, which carry no timestamp and are not one-per-run."""
        self._touch("SKAB_7_off_by_point_importance.png")
        self._touch("SKAB_7_off_by_point_tree_LOF_1_vs_NN_3.png")
        for day in range(1, 6):
            self._touch(f"Data_vs_DataWithAnomalies_2026-05-0{day}_10-00-00_.png")
        prune_timestamped(self.d)
        self.assertIn("SKAB_7_off_by_point_importance.png", self._names())
        self.assertIn("SKAB_7_off_by_point_tree_LOF_1_vs_NN_3.png", self._names())

    def test_a_group_at_or_under_the_limit_is_left_alone(self):
        for day in (1, 2, 3):
            self._touch(f"Data_vs_DataWithAnomalies_2026-05-0{day}_10-00-00_.png")
        self.assertEqual(prune_timestamped(self.d), [])
        self.assertEqual(len(self._names()), 3)

    def test_a_missing_directory_is_not_an_error(self):
        """This runs right after a figure is saved; losing that run's output to
        a bookkeeping failure would be far worse than the disk it saves."""
        self.assertEqual(prune_timestamped(os.path.join(self.d, "nope")), [])


if __name__ == "__main__":
    unittest.main()
