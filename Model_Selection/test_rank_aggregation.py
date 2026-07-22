"""
Standalone unit tests for the rank-aggregation explainability layer.
Mocks the heavy module-level imports (cvxpy / networkx / sklearn) so the test
can run in any env that has numpy + scipy + matplotlib.
"""

import os
import sys
import tempfile
import types
import unittest

import numpy as np


# ── Mock heavy module-level imports so rank_aggregation.py loads ────────────
def _make_mock_module(*names):
    for name in names:
        parts = name.split(".")
        parent = None
        for i, part in enumerate(parts):
            full = ".".join(parts[: i + 1])
            if full not in sys.modules:
                mod = types.ModuleType(full)
                sys.modules[full] = mod
                if parent is not None:
                    setattr(parent, part, mod)
            parent = sys.modules[full]


_make_mock_module(
    "cvxpy",
    "networkx",
    "sklearn",
    "sklearn.cluster",
    "sklearn.neighbors",
    "sklearn.preprocessing",
)
# rank_aggregation does `from sklearn.cluster import AgglomerativeClustering` etc.
sys.modules["sklearn.cluster"].AgglomerativeClustering = type("AgglomerativeClustering", (), {})
sys.modules["sklearn.neighbors"].NearestNeighbors = type("NearestNeighbors", (), {})
sys.modules["sklearn.preprocessing"].MinMaxScaler = type("MinMaxScaler", (), {})

# ── Import the module under test ────────────────────────────────────────────
from rank_aggregation import (
    enhanced_markov_chain_rank_aggregator_text,
    kendall_tau_restricted,
    leave_one_out_contributions,
    kendall_tau_alignments,
    borda_count_resolution,
    borda_verdict_per_source,
    prominent_contradictions,
    explain_rank_aggregation,
    explain_rank_aggregation_kendall_only,
    _ranks_from_scores,
)


# ════════════════════════════════════════════════════════════════════════════
# 1.  kendall_tau_restricted
# ════════════════════════════════════════════════════════════════════════════

class TestKendallTauRestricted(unittest.TestCase):

    def test_identical_orderings_give_plus_one(self):
        a = ["A", "B", "C", "D"]
        b = ["A", "B", "C", "D"]
        self.assertAlmostEqual(kendall_tau_restricted(a, b), 1.0)

    def test_reverse_orderings_give_minus_one(self):
        a = ["A", "B", "C", "D"]
        b = ["D", "C", "B", "A"]
        self.assertAlmostEqual(kendall_tau_restricted(a, b), -1.0)

    def test_partial_overlap_restricted_to_common_items(self):
        a = ["A", "B", "C"]
        b = ["B", "A", "Z"]   # common = {A, B}; b's order is B,A → reverse → -1
        self.assertAlmostEqual(kendall_tau_restricted(a, b), -1.0)

    def test_no_overlap_returns_zero(self):
        a = ["A", "B"]
        b = ["X", "Y"]
        self.assertEqual(kendall_tau_restricted(a, b), 0.0)


# ════════════════════════════════════════════════════════════════════════════
# 2.  leave_one_out_contributions
# ════════════════════════════════════════════════════════════════════════════

class TestLeaveOneOutContributions(unittest.TestCase):

    def test_pivotal_source_has_high_loo(self):
        """With two strongly-disagreeing sources, removing one must flip the
        consensus to match the surviving source — that source is *pivotal* and
        its LOO contribution should be near maximal. Ties in the aggregator
        may resolve asymmetrically (one source ends up matching the full
        ranking), so we assert on the maximum LOO across sources rather than
        on both."""
        sources = [
            ["A", "B", "C", "D"],
            ["D", "C", "B", "A"],
        ]
        names = ["forward", "reverse"]
        _, full = enhanced_markov_chain_rank_aggregator_text(sources)
        loo = leave_one_out_contributions(sources, names, full)
        # At least one source must be pivotal (LOO ~ 1.0 when its removal
        # leaves a single dissenting source as the sole input).
        self.assertGreater(max(loo.values()), 0.9)

    def test_single_source_returns_zero(self):
        sources = [["A", "B", "C"]]
        names = ["only"]
        _, full = enhanced_markov_chain_rank_aggregator_text(sources)
        loo = leave_one_out_contributions(sources, names, full)
        self.assertEqual(loo["only"], 0.0)

    def test_loo_score_is_in_unit_interval(self):
        sources = [
            ["A", "B", "C", "D"],
            ["B", "A", "D", "C"],
            ["D", "C", "B", "A"],
        ]
        names = ["s1", "s2", "s3"]
        _, full = enhanced_markov_chain_rank_aggregator_text(sources)
        loo = leave_one_out_contributions(sources, names, full)
        for v in loo.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# 3.  kendall_tau_alignments
# ════════════════════════════════════════════════════════════════════════════

class TestKendallTauAlignments(unittest.TestCase):

    def test_matches_kendall_tau_restricted(self):
        sources = [
            ["A", "B", "C"],
            ["C", "B", "A"],
        ]
        names = ["agree", "reverse"]
        full = ["A", "B", "C"]
        align = kendall_tau_alignments(sources, names, full)
        self.assertAlmostEqual(align["agree"], 1.0)
        self.assertAlmostEqual(align["reverse"], -1.0)


# ════════════════════════════════════════════════════════════════════════════
# 4.  borda_count_resolution  (Borda voting over LOO + Kendall rankings)
# ════════════════════════════════════════════════════════════════════════════

class TestBordaCountResolution(unittest.TestCase):

    def test_borda_count_is_sum_of_two_voter_points(self):
        """Each source's Borda count = (N - loo_rank) + (N - align_rank)."""
        # 3 sources; LOO order: s0 > s1 > s2 (ranks 1,2,3)
        #            Align order: s2 > s1 > s0 (ranks 3,2,1)
        loo   = {"s0": 0.9, "s1": 0.5, "s2": 0.1}
        align = {"s0": 0.1, "s1": 0.5, "s2": 0.9}
        counts = borda_count_resolution(loo, align)
        # n = 3. s0: (3-1) + (3-3) = 2; s1: (3-2)+(3-2) = 2; s2: (3-3)+(3-1) = 2
        self.assertAlmostEqual(counts["s0"], 2.0)
        self.assertAlmostEqual(counts["s1"], 2.0)
        self.assertAlmostEqual(counts["s2"], 2.0)

    def test_borda_promotes_doubly_top_ranked_source(self):
        """A source ranked 1st by BOTH voters gets the maximum count of 2(N-1)."""
        loo   = {"top": 0.99, "mid": 0.5, "bot": 0.1}
        align = {"top": 0.99, "mid": 0.5, "bot": 0.1}
        counts = borda_count_resolution(loo, align)
        n = 3
        self.assertAlmostEqual(counts["top"], 2 * (n - 1))
        self.assertAlmostEqual(counts["bot"], 0.0)

    def test_mismatched_keys_raises(self):
        with self.assertRaises(ValueError):
            borda_count_resolution({"a": 1.0}, {"b": 1.0})


# ════════════════════════════════════════════════════════════════════════════
# 5.  borda_verdict_per_source  (Borda count rank IS the resolution)
# ════════════════════════════════════════════════════════════════════════════

class TestBordaVerdictPerSource(unittest.TestCase):

    def test_every_source_has_full_record(self):
        loo   = {"s0": 0.9, "s1": 0.1, "s2": 0.5, "s3": 0.6}
        align = {"s0": 0.1, "s1": 0.9, "s2": 0.5, "s3": 0.6}
        verdicts = borda_verdict_per_source(loo, align)
        self.assertEqual(len(verdicts), 4)
        for v in verdicts:
            for key in ("source", "loo_rank", "align_rank",
                        "borda_count", "borda_rank", "pattern",
                        "lo_align_rank_delta"):
                self.assertIn(key, v)
            # The dropped key from the old design must not reappear.
            self.assertNotIn("borda_verdict", v)

    def test_pattern_labels(self):
        loo   = {"hi_loo": 0.9, "lo_loo": 0.1, "tied": 0.5}
        align = {"hi_loo": 0.1, "lo_loo": 0.9, "tied": 0.5}
        verdicts = {v["source"]: v for v in borda_verdict_per_source(loo, align)}
        self.assertEqual(verdicts["hi_loo"]["pattern"], "influential_disagreer")
        self.assertEqual(verdicts["lo_loo"]["pattern"], "redundant_agreer")
        self.assertEqual(verdicts["tied"]["pattern"], "consistent")

    def test_borda_rank_resolves_extreme_disagreement(self):
        """Two sources sharply contradict on LOO vs Kendall; a third is moderate.
        Borda's combined ranking must give the moderate source the best (lowest)
        rank because it's #2 under both voters while the others alternate."""
        # 3 sources: LOO order = s0 > s_mid > s1; Align order = s1 > s_mid > s0.
        loo   = {"s0": 0.9, "s_mid": 0.5, "s1": 0.1}
        align = {"s0": 0.1, "s_mid": 0.5, "s1": 0.9}
        verdicts = {v["source"]: v for v in borda_verdict_per_source(loo, align)}
        # All three have Borda count = 2 (sum of complementary ranks).
        for v in verdicts.values():
            self.assertAlmostEqual(v["borda_count"], 2.0)
        # Standard competition ranking ("1224"): three tied sources all share the
        # smallest rank in their group — rank 1, never the fractional 2.0 that
        # average ranking produced.
        self.assertEqual(verdicts["s0"]["borda_rank"], 1)
        self.assertEqual(verdicts["s_mid"]["borda_rank"], 1)
        self.assertEqual(verdicts["s1"]["borda_rank"], 1)

    def test_borda_count_breaks_an_asymmetric_disagreement(self):
        """If one source is preferred more strongly by one voter than the other
        opposes it, Borda's combined rank reflects that asymmetry."""
        # 3 sources: s_winner is LOO rank 1 and Align rank 2 → count = 3.
        #            s_mid    is LOO rank 2 and Align rank 1 → count = 3.
        #            s_loser  is LOO rank 3 and Align rank 3 → count = 0.
        loo   = {"s_winner": 0.9, "s_mid": 0.5, "s_loser": 0.1}
        align = {"s_winner": 0.5, "s_mid": 0.9, "s_loser": 0.1}
        verdicts = {v["source"]: v for v in borda_verdict_per_source(loo, align)}
        self.assertAlmostEqual(verdicts["s_winner"]["borda_count"], 3.0)
        self.assertAlmostEqual(verdicts["s_mid"]["borda_count"],    3.0)
        self.assertAlmostEqual(verdicts["s_loser"]["borda_count"],  0.0)
        # s_loser is unambiguously last under Borda.
        self.assertEqual(verdicts["s_loser"]["borda_rank"], 3.0)

    def test_hybrid_tie_handling(self):
        """Displayed ranks + pattern use competition ranking; the Borda COUNT
        uses average ranks internally (fair arithmetic). A tie on influence
        makes the two differ, which this locks in."""
        # A and B tie for top influence (0.9); agreement is all-distinct.
        loo   = {"A": 0.9, "B": 0.9, "C": 0.1}
        align = {"A": 0.9, "B": 0.5, "C": 0.1}
        verdicts = {v["source"]: v for v in borda_verdict_per_source(loo, align)}
        # Displayed influence rank: competition → A and B both 1 (ints), not 1.5.
        self.assertEqual(verdicts["A"]["loo_rank"], 1)
        self.assertEqual(verdicts["B"]["loo_rank"], 1)
        self.assertIsInstance(verdicts["A"]["loo_rank"], int)
        # Borda count uses AVERAGE loo ranks (A,B → 1.5): n=3.
        #   A = (3-1.5)+(3-1) = 3.5   B = (3-1.5)+(3-2) = 2.5   C = 0
        # (competition loo ranks would have given A=4, B=3 — the boost we avoid.)
        self.assertAlmostEqual(verdicts["A"]["borda_count"], 3.5)
        self.assertAlmostEqual(verdicts["B"]["borda_count"], 2.5)
        self.assertAlmostEqual(verdicts["C"]["borda_count"], 0.0)
        # Borda rank = competition rank of those counts.
        self.assertEqual(verdicts["A"]["borda_rank"], 1)
        self.assertEqual(verdicts["B"]["borda_rank"], 2)
        self.assertEqual(verdicts["C"]["borda_rank"], 3)
        # Patterns from competition ranks: A is tied-top on both → consistent;
        # B is rank 1 influence but rank 2 agreement → influential_disagreer.
        self.assertEqual(verdicts["A"]["pattern"], "consistent")
        self.assertEqual(verdicts["B"]["pattern"], "influential_disagreer")

    def test_ranks_from_scores_method_switch(self):
        scores = {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.1}  # b,c tie
        comp = _ranks_from_scores(scores)                   # default competition
        avg = _ranks_from_scores(scores, method="average")
        self.assertEqual(comp, {"a": 1, "b": 2, "c": 2, "d": 4})
        self.assertEqual(avg, {"a": 1.0, "b": 2.5, "c": 2.5, "d": 4.0})


# ════════════════════════════════════════════════════════════════════════════
# 6.  prominent_contradictions
# ════════════════════════════════════════════════════════════════════════════

class TestProminentContradictions(unittest.TestCase):

    def test_only_large_delta_sources_returned(self):
        loo   = {"s0": 0.9, "s1": 0.1, "s2": 0.5, "s3": 0.55}
        align = {"s0": 0.1, "s1": 0.9, "s2": 0.5, "s3": 0.55}
        verdicts = borda_verdict_per_source(loo, align)
        prom = prominent_contradictions(verdicts, rank_delta_threshold=2)
        names = {v["source"] for v in prom}
        # s0 and s1 have rank deltas of 3 and 3 → flagged; s2 and s3 have deltas of 0 / 0
        self.assertIn("s0", names)
        self.assertIn("s1", names)
        self.assertNotIn("s2", names)
        self.assertNotIn("s3", names)


# ════════════════════════════════════════════════════════════════════════════
# 7.  explain_rank_aggregation (integration smoke test)
# ════════════════════════════════════════════════════════════════════════════

class TestExplainRankAggregationIntegration(unittest.TestCase):

    def test_writes_plot_and_report_files(self):
        sources = [
            ["A", "B", "C", "D", "E"],
            ["A", "B", "C", "D", "E"],
            ["E", "D", "C", "B", "A"],
        ]
        names = ["agree1", "agree2", "outlier"]
        _, full = enhanced_markov_chain_rank_aggregator_text(sources)

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd_before = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = explain_rank_aggregation(
                    rankings=sources,
                    source_names=names,
                    full_ranking=full,
                    stage_name="robust",
                    dataset="TEST",
                    entity="e1",
                    iteration=0,
                    explain=True,
                )
                self.assertIsInstance(result, dict)
                for key in ("loo_scores", "align_scores", "borda_counts",
                            "verdicts", "prominent_contradictions", "kendall_only"):
                    self.assertIn(key, result)
                # 3 sources → the Kendall-only method does not apply.
                self.assertIsNone(result["kendall_only"])

                out_dir = os.path.join("myresults", "robust_aggregated", "TEST", "e1")
                self.assertTrue(os.path.exists(
                    os.path.join(out_dir, "aggregation_explainability_robust_0.png")))
                self.assertTrue(os.path.exists(
                    os.path.join(out_dir, "aggregation_explainability_robust_0.txt")))
                # Intermediate Representation JSON is emitted alongside.
                import json
                ir_path = os.path.join("myresults", "explanations_ir", "TEST", "e1",
                                       "ir_rank_aggregation_robust_0.json")
                self.assertTrue(os.path.exists(ir_path), ir_path)
                with open(ir_path) as fh:
                    self.assertEqual(json.load(fh)["stage"], "rank_aggregation_robust")
            finally:
                os.chdir(cwd_before)

    def test_explain_false_is_noop_and_returns_none(self):
        result = explain_rank_aggregation(
            rankings=[["A"], ["B"]],
            source_names=["s1", "s2"],
            full_ranking=["A", "B"],
            stage_name="robust",
            dataset="X", entity="Y", iteration=0,
            explain=False,
        )
        self.assertIsNone(result)


# ════════════════════════════════════════════════════════════════════════════
# 8.  explain_rank_aggregation_kendall_only  (two-source method)
# ════════════════════════════════════════════════════════════════════════════

class TestKendallOnly(unittest.TestCase):

    def test_returns_none_when_not_two_sources(self):
        for n in (1, 3):
            result = explain_rank_aggregation_kendall_only(
                rankings=[["A", "B"]] * n,
                source_names=[f"s{i}" for i in range(n)],
                full_ranking=["A", "B"],
                stage_name="final", dataset="X", entity="Y", iteration=0,
            )
            self.assertIsNone(result)

    def test_winner_is_source_with_higher_tau(self):
        # final == source A exactly → tau(A,final)=+1; B reversed → tau(B,final)=-1
        full = ["A", "B", "C", "D"]
        sources = [["A", "B", "C", "D"], ["D", "C", "B", "A"]]
        names = ["aligned", "reversed"]
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = explain_rank_aggregation_kendall_only(
                    sources, names, full, "final", "TEST", "e1", 5)
            finally:
                os.chdir(cwd)
        self.assertIsNotNone(result)
        self.assertEqual(result["winner"], "aligned")
        self.assertAlmostEqual(result["winner_tau"], 1.0)
        self.assertEqual(result["runner_up"], "reversed")
        self.assertAlmostEqual(result["runner_up_tau"], -1.0)
        self.assertAlmostEqual(result["alignment_gap"], 2.0)

    def test_writes_report_and_plot(self):
        full = ["A", "B", "C"]
        sources = [["A", "B", "C"], ["B", "A", "C"]]
        names = ["robust_agg", "thompson"]
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                explain_rank_aggregation_kendall_only(
                    sources, names, full, "final", "TEST", "e1", 5)
                out_dir = os.path.join("myresults", "robust_aggregated", "TEST", "e1")
                self.assertTrue(os.path.exists(os.path.join(
                    out_dir, "aggregation_explainability_final_kendall_only_5.txt")))
                self.assertTrue(os.path.exists(os.path.join(
                    out_dir, "aggregation_explainability_final_kendall_only_5.png")))
            finally:
                os.chdir(cwd)

    def test_explain_rank_aggregation_triggers_kendall_only_for_two_sources(self):
        full = ["A", "B", "C"]
        sources = [["A", "B", "C"], ["C", "B", "A"]]
        names = ["robust_agg", "thompson"]
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = explain_rank_aggregation(
                    sources, names, full, "final", "TEST", "e1", 5, explain=True)
                # Intermediate Representation JSON with the kendall-only block.
                import json
                ir_path = os.path.join("myresults", "explanations_ir", "TEST", "e1",
                                       "ir_rank_aggregation_final_5.json")
                self.assertTrue(os.path.exists(ir_path), ir_path)
                with open(ir_path) as fh:
                    ir_doc = json.load(fh)
                self.assertEqual(ir_doc["stage"], "rank_aggregation_final")
                self.assertIn("ra_final.kendall_only.winner",
                              [a["id"] for a in ir_doc["evidence"]])
            finally:
                os.chdir(cwd)
        # 2 sources → the nested kendall_only result is populated.
        self.assertIsNotNone(result["kendall_only"])
        self.assertIn("winner", result["kendall_only"])


# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
