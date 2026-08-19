import os
import tempfile
import unittest

from Model_Selection.Sensitivity_robustness.plot_retention import (
    prune_superseded, prune_timestamped)


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




class TestPruneSuperseded(unittest.TestCase):
    """Retention for the figures whose NAME encodes the run's outcome.

    `..._point_tree_{winner}_vs_{competitor}.png` is not a stable filename: it
    changes whenever the winner changes, whenever a competitor is added or
    dropped, and — the case that produced this test — whenever a detector is
    renamed. None of those overwrite the previous run's file, so the directory
    accumulates trees from several runs and the WebUI picker offers all of them
    as if they described one.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, name):
        with open(os.path.join(self.d, name), "wb") as f:
            f.write(b"\x89PNG")
        return name

    def _names(self):
        return sorted(os.listdir(self.d))

    def test_the_rename_case_that_produced_this(self):
        """Same winner, competitors respelled: the old set must not survive.

        This is exactly what SKAB/5 held — `LUNAR_2_vs_AE_1.png` from before the
        rename sitting beside the `LUNAR_2_vs_AutoEncoder_1.png` that replaced
        it. Grouping by winner cannot separate them: the winner is identical.
        """
        for old in ("AE_1", "AE_2", "OA_1"):
            self._touch(f"skab_5_off_by_point_tree_LUNAR_2_vs_{old}.png")
        kept = [self._touch(f"skab_5_off_by_point_tree_LUNAR_2_vs_{new}.png")
                for new in ("AutoEncoder_1", "AutoEncoder_2", "OmniAnomaly_1")]

        removed = prune_superseded(self.d, "skab_5_off_by_point_tree_", kept)

        self.assertEqual(len(removed), 3)
        self.assertEqual(self._names(), sorted(kept))

    def test_a_previous_run_s_winner_is_removed_too(self):
        """The stale-winner case the WebUI already hid is now gone from disk."""
        self._touch("skab_5_off_by_point_tree_LUNAR_1_vs_DONUT_1.png")
        kept = [self._touch("skab_5_off_by_point_tree_LUNAR_2_vs_DONUT_1.png")]
        prune_superseded(self.d, "skab_5_off_by_point_tree_", kept)
        self.assertEqual(self._names(), kept)

    def test_a_competitor_that_went_degenerate_loses_its_old_tree(self):
        """No tree this run means no tree shown.

        A degenerate surrogate draws nothing, so its name is absent from
        `keep_names`. Leaving the previous run's tree would show a decision
        boundary for a comparison this run found to have none.
        """
        self._touch("skab_5_off_by_point_tree_LUNAR_2_vs_USAD_1.png")
        kept = [self._touch("skab_5_off_by_point_tree_LUNAR_2_vs_USAD_2.png")]
        prune_superseded(self.d, "skab_5_off_by_point_tree_", kept)
        self.assertEqual(self._names(), kept)

    def test_it_touches_nothing_outside_its_prefix(self):
        """The importance figure, the report and the timestamped figures share
        this directory and have their own retention (or none)."""
        others = [
            self._touch("skab_5_off_by_point_importance.png"),
            self._touch("Data_vs_DataWithAnomalies_2026-08-19_22-48-05_.png"),
            self._touch("skab_5_gan_point_tree_LUNAR_2_vs_DONUT_1.png"),
        ]
        with open(os.path.join(self.d, "skab_5_off_by_explainability.txt"), "w") as f:
            f.write("report")
        kept = [self._touch("skab_5_off_by_point_tree_LUNAR_2_vs_DONUT_1.png")]

        prune_superseded(self.d, "skab_5_off_by_point_tree_", kept)

        self.assertEqual(self._names(),
                         sorted(others + kept + ["skab_5_off_by_explainability.txt"]))

    def test_an_empty_keep_set_still_clears_the_directory(self):
        """A run whose surrogates were all degenerate writes no trees at all,
        and the previous run's must not stand in for them."""
        self._touch("skab_5_off_by_point_tree_LUNAR_2_vs_DONUT_1.png")
        removed = prune_superseded(self.d, "skab_5_off_by_point_tree_", [])
        self.assertEqual(len(removed), 1)
        self.assertEqual(self._names(), [])

    def test_a_missing_directory_is_not_an_error(self):
        """Same contract as prune_timestamped: retention never fails a run."""
        self.assertEqual(
            prune_superseded(os.path.join(self.d, "nope"), "x_", []), [])


if __name__ == "__main__":
    unittest.main()
