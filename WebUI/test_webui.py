"""
Tests for the WebUI backend.

Follows the repo's test style (unittest, no network). Every test builds a
synthetic artifact tree in a temporary directory and repoints WebUI.paths at
it, so nothing depends on the contents of the real myresults/ and no test ever
starts the pipeline or the LLM.
"""

import json
import os
import pickle
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from WebUI import artifacts, catalog, paths, summarize  # noqa: E402

# Flask lives in the project venv (3.11) while pytest lives in the framework
# interpreter (3.13), so the route tests skip rather than fail when the suite
# is run under the one without Flask. Run them with:
#     .venv/bin/python3 -m unittest WebUI.test_webui
try:
    import flask  # noqa: F401
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


# ── Fixture builders ─────────────────────────────────────────────────────────

def _ir(stage, **kw):
    doc = {"ir_version": "1.0", "stage": stage, "dataset": "SKAB", "entity": "7",
           "output": kw.pop("output", {}), "evidence": kw.pop("evidence", []),
           "caveats": kw.pop("caveats", []), "required_atom_ids": [],
           "confidence": {}, "question": kw.pop("question", "Why?"),
           "info_footer": kw.pop("info_footer", "Glossary text.")}
    doc.update(kw)
    return doc


def _write(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, (dict, list)):
        path.write_text(json.dumps(content, indent=2))
    else:
        path.write_text(content)


def _nl(info, narrative):
    """The on-disk per-stage format: glossary first, blank line, narrative."""
    return f"INFO: {info}\n\n{narrative}\n"


RULE = "=" * 80

# A faithful shape of run_app's report: rule/title/rule around every section.
REPORT_TEXT = "\n".join([
    RULE, "RAMSeS Framework - Comprehensive Results",
    "Dataset: skab | Entity: 7 | Iteration: 5", "Timestamp: 2026-08-06 19:07:24", RULE,
    "", RULE, "COMPUTATIONAL OVERHEAD (seconds)", RULE, "",
    "Per-Module Timing:", "  1_Genetic_Algorithm            :    39.6792s", "",
    RULE, "FRAMEWORK FINAL DECISION", RULE, "",
    "Final Choice:", "  ✓ ENSEMBLE SELECTED", "",
    RULE, "END OF REPORT", RULE, "",
])


class ArtifactTreeCase(unittest.TestCase):
    """Builds a realistic tree and points WebUI.paths at it."""

    DATASET = "SKAB"
    ENTITY = "7"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.myresults = root / "myresults"
        self.ir_dir = self.myresults / "explanations_ir" / self.DATASET / self.ENTITY
        self.nl_dir = self.myresults / "explanations_nl" / self.DATASET / self.ENTITY
        self.report_dir = self.myresults / "comprehensive" / self.DATASET / self.ENTITY
        self._saved = (paths.MYRESULTS, paths.EXPLANATIONS_IR, paths.EXPLANATIONS_NL,
                       paths.COMPREHENSIVE)
        paths.MYRESULTS = self.myresults
        paths.EXPLANATIONS_IR = self.myresults / "explanations_ir"
        paths.EXPLANATIONS_NL = self.myresults / "explanations_nl"
        paths.COMPREHENSIVE = self.myresults / "comprehensive"
        artifacts.paths = paths
        self.build()

    def tearDown(self):
        (paths.MYRESULTS, paths.EXPLANATIONS_IR, paths.EXPLANATIONS_NL,
         paths.COMPREHENSIVE) = self._saved
        self._tmp.cleanup()

    def build(self):
        _write(self.ir_dir / "ir_thompson.json", _ir(
            "thompson_sampling",
            output={"top_pick": "NN_1", "n_regimes": 2, "n_windows": 173},
            evidence=[
                {"id": "ts.output.top", "type": "stage_output", "subject": "NN_1",
                 "value": {}, "text": "Thompson Sampling ranked NN_1 first."},
                {"id": "ts.regime.1", "type": "regime", "subject": "NN_2",
                 "value": {"start": 5, "end": 18, "duration": 14, "leader": "NN_2"},
                 "text": "Regime 1 (windows 5 to 18) was led by NN_2."},
                {"id": "ts.regime.0", "type": "regime", "subject": "NN_3",
                 "value": {"start": 0, "end": 4, "duration": 5, "leader": "NN_3"},
                 "text": "Regime 0 (windows 0 to 4) was led by NN_3."},
            ]))
        _write(self.nl_dir / "nl_thompson.txt",
               _nl("Expected reward is the weights applied to the window.",
                   "Thompson Sampling ranked NN_1 first."))

        # The ranking-criterion sibling: same tree, own files, own atom prefix.
        _write(self.ir_dir / "ir_thompson_ranking.json", _ir(
            "thompson_ranking",
            output={"top_pick": "NN_1", "n_regimes": 1, "n_windows": 173,
                    "n_channels": 8, "warmup_windows": 10},
            evidence=[
                {"id": "tsr.output.top", "type": "stage_output", "subject": "NN_1",
                 "value": {"top": "NN_1", "score": 0.559322, "runner_up": "NN_2",
                           "margin": 0.00041},
                 "text": "Ranked by the size of its learned weights, NN_1 scored "
                         "0.559322, ahead of NN_2 by 0.000410."},
                {"id": "tsr.winner.channels", "type": "winner_channels",
                 "subject": "NN_1",
                 "value": {"channel": 3, "total": 0.559322,
                           "per_channel": [[3, 0.229], [0, 0.101]]},
                 "text": "NN_1's score is built mostly from channel 3 (41.0%)."},
                {"id": "tsr.gap.runner_up", "type": "rank_gap", "subject": "NN_1",
                 "value": {"rivals": ["NN_2"], "runner_up": "NN_2",
                           "per_channel": [[3, 0.091], [0, -0.058]]},
                 "text": "NN_1's lead over NN_2 came mostly from channel 3."},
                {"id": "tsr.regime.0", "type": "regime", "subject": "NN_2",
                 "value": {"start": 10, "end": 71, "duration": 62, "leader": "NN_2"},
                 "text": "Regime 0 (windows 10 to 71, 62 windows) was led by NN_2."},
            ]))
        _write(self.nl_dir / "nl_thompson_ranking.txt",
               _nl("The ranking score is the sum of every squared weight.",
                   "Ranked by the size of its learned weights, NN_1 scored "
                   "0.559322, ahead of NN_2 by 0.000410. NN_1's score is built "
                   "mostly from channel 3 (41.0%). Regime 0 (windows 10 to 71, "
                   "62 windows) was led by NN_2."))

        _write(self.ir_dir / "ir_monte_carlo.json", _ir(
            "monte_carlo", output={"top_pick_f1": "LOF_1", "top_pick_pr": "LOF_1"},
            caveats=[{"id": "mc.c", "type": "caveat", "subject": "x", "value": None,
                      "text": "Degenerate folds."}]))
        _write(self.nl_dir / "nl_monte_carlo.txt",
               _nl("The sweep injects Gaussian noise.", "LOF_1 ranked first."))

        # Rank aggregation carries an iteration suffix.
        _write(self.ir_dir / "ir_rank_aggregation_robust_0.json",
               _ir("rank_aggregation_robust", output={"top_pick": "LOF_1"},
                   evidence=[
                       {"id": "ra_robust.output.top", "type": "stage_output",
                        "subject": "LOF_1", "value": "LOF_1",
                        "text": "Its first-ranked detector is LOF_1."},
                       {"id": "ra_robust.source.GAN_F1.role", "type": "source_role",
                        "subject": "GAN_F1",
                        "value": {"influence_rank": 1, "agreement_rank": 1,
                                  "borda_rank": 1, "pattern": "consistent"},
                        "text": "GAN_F1 shaped the consensus most."},
                       {"id": "ra_robust.source.GAN_PR_AUC.role", "type": "source_role",
                        "subject": "GAN_PR_AUC",
                        "value": {"influence_rank": 5, "agreement_rank": 2,
                                  "borda_rank": 2, "pattern": "redundant_agreer"},
                        "text": "GAN_PR_AUC shaped the consensus second most."},
                   ]))
        _write(self.nl_dir / "nl_rank_aggregation_robust_0.txt",
               _nl("Influence compares rankings.",
                   "Its first-ranked detector is LOF_1. GAN_F1 shaped the "
                   "consensus most. GAN_PR_AUC shaped the consensus second most."))

        _write(self.ir_dir / "ir_global_iter0.json", {
            "ir_version": "1.0", "stage": "global", "dataset": self.DATASET,
            "entity": self.ENTITY, "iteration": 0,
            "decision": {"framework_choice": "ensemble", "chosen": ["A", "B"],
                         "ensemble_f1": 0.9, "single_model": "NN_3"},
            "stage_agreement": {"thompson": {"top_pick": "NN_1",
                                             "agrees_with_final_single": False}},
            "stages": {"thompson_sampling": {"status": "ok"},
                       "thompson_ranking": {"status": "ok"},
                       "monte_carlo": {"status": "ok"},
                       "gan": {"status": "not_available",
                               "note": "no explainability layer implemented for the GAN test"}},
            "evidence": [{"id": "global.decision", "type": "decision", "subject": "d",
                          "value": None, "text": "The final decision is the ensemble {A, B}."}],
            "caveats": [], "required_atom_ids": []})

        # The pipeline's own report, in its own tree, under the CLI --iteration
        # (5) rather than the explanations' OFFLINE_ITERATION (0).
        _write(self.report_dir / f"comprehensive_results_{self.DATASET}_{self.ENTITY}_iter5.txt",
               REPORT_TEXT)

        _write(self.nl_dir / "nl_global_iter0.txt", "RAMSeS model selection — SKAB / entity 7\n")
        _write(self.nl_dir / "faithfulness_iter0.json", {
            "dataset": self.DATASET, "entity": self.ENTITY, "iteration": 0,
            "model": "qwen2.5:7b-instruct",
            "overall": {"hallucination_rate": 0.0, "omission_rate": 0.016,
                        "n_claims": 259, "n_required": 61},
            "stages": {"thompson_sampling": {"status": "ok", "words": 466,
                                             "verify": {"hallucination_rate": 0.0,
                                                        "omission_rate": 0.0,
                                                        "n_claims": 58, "n_required": 18}},
                       "global": {"status": "ok", "mode": "concat", "words": 2100}}})


# ── split_info ───────────────────────────────────────────────────────────────

class TestSplitInfo(unittest.TestCase):

    def test_glossary_leads_the_file(self):
        info, narrative = artifacts.split_info("INFO: A glossary.\n\nThe narrative.\n")
        self.assertEqual(info, "A glossary.")
        self.assertEqual(narrative, "The narrative.")

    def test_no_glossary(self):
        info, narrative = artifacts.split_info("Just prose.\n")
        self.assertIsNone(info)
        self.assertEqual(narrative, "Just prose.")

    def test_info_appearing_mid_narrative_is_not_a_glossary(self):
        raw = "The narrative mentions INFO: not as a marker.\n"
        info, narrative = artifacts.split_info(raw)
        self.assertIsNone(info)
        self.assertIn("INFO:", narrative)

    def test_crlf_and_missing_trailing_newline(self):
        info, narrative = artifacts.split_info("INFO: G.\r\n\r\nN.")
        self.assertEqual(info, "G.")
        self.assertEqual(narrative, "N.")

    def test_multi_paragraph_narrative_is_kept_whole(self):
        info, narrative = artifacts.split_info("INFO: G.\n\nPara one.\n\nPara two.")
        self.assertEqual(info, "G.")
        self.assertEqual(narrative, "Para one.\n\nPara two.")

    def test_empty_and_none(self):
        self.assertEqual(artifacts.split_info(""), (None, ""))
        self.assertEqual(artifacts.split_info(None), (None, ""))

    def test_glossary_with_no_narrative(self):
        info, narrative = artifacts.split_info("INFO: Only a glossary.\n")
        self.assertEqual(info, "Only a glossary.")
        self.assertEqual(narrative, "")


# ── build_payload ────────────────────────────────────────────────────────────

class TestBuildPayload(ArtifactTreeCase):

    def test_a_narrative_older_than_its_ir_is_flagged_stale(self):
        """`--stages X --explain` is a PARTIAL run: app.py returns before the
        narration block, so it rewrites the IR and every plot but leaves the
        prose alone. On SKAB/7 that left a narrative walking 14 regimes beside
        an IR and plots holding 10, and nothing noticed — the two files are read
        independently. Freshness is the only signal available."""
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        by = {s["key"]: s for s in p["stages"]}
        self.assertFalse(any(s["stale"] for s in p["stages"]))

        ir_file = self.ir_dir / "ir_thompson.json"
        os.utime(ir_file, (time.time() + 60, time.time() + 60))
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        by = {s["key"]: s for s in p["stages"]}
        self.assertTrue(by["thompson_sampling"]["stale"])
        # Only the stage whose IR moved; the others are untouched.
        self.assertFalse(by["monte_carlo"]["stale"])

    def test_same_run_write_order_is_not_mistaken_for_staleness(self):
        """A full run writes the IR and the narrative moments apart, in that
        order. Without slack every healthy run would raise the warning."""
        ir_file = self.ir_dir / "ir_thompson.json"
        nl_file = self.nl_dir / "nl_thompson.txt"
        now = time.time()
        os.utime(ir_file, (now, now))
        os.utime(nl_file, (now + 0.2, now + 0.2))
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        by = {s["key"]: s for s in p["stages"]}
        self.assertFalse(by["thompson_sampling"]["stale"])

    def test_assembles_stages_in_pipeline_order(self):
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        # The ranking criterion precedes the selection dynamics: it explains the
        # ordering the rest of the pipeline consumes.
        self.assertEqual([s["key"] for s in p["stages"]],
                         ["thompson_ranking", "thompson_sampling", "monte_carlo",
                          "rank_aggregation_robust"])
        self.assertEqual([s["order"] for s in p["stages"]], [3, 4, 5, 7])

    def test_stage_order_matches_the_narrator(self):
        """The merged .txt and the page must present the stages in one order.

        artifacts.STAGES and llm._GLOBAL_STAGE_ORDER are independent copies, and
        a divergence would give the global document section headings in a
        different sequence from the cards on the page.
        """
        llm_path = Path(__file__).resolve().parent.parent / "Explainability" / "llm.py"
        source = llm_path.read_text(encoding="utf-8")
        order = re.search(r"_GLOBAL_STAGE_ORDER = \((.*?)\)", source, re.S).group(1)
        narrator = re.findall(r'"([a-z_]+)"', order)
        self.assertEqual([s["key"] for s in artifacts.STAGES], narrator)

    def test_stage_titles_match_the_narrator(self):
        """Same reason: the titles are duplicated verbatim in llm.py."""
        llm_path = Path(__file__).resolve().parent.parent / "Explainability" / "llm.py"
        source = llm_path.read_text(encoding="utf-8")
        block = re.search(r"_GLOBAL_STAGE_TITLES = \{(.*?)\n\}", source, re.S).group(1)
        titles = dict(re.findall(r'"([a-z_]+)": "([^"]+)"', block))
        for stage in artifacts.STAGES:
            self.assertEqual(titles.get(stage["key"]), stage["title"], stage["key"])

    def test_the_payload_serves_the_trimmed_body_but_counts_the_real_one(self):
        """`full` is what the disclosure renders; `words` and the download stay
        on the narrative as written, which is the verbatim record."""
        p = artifacts.build_payload("SKAB", "7")
        s = next(x for x in p["stages"] if x["key"] == "thompson_ranking")
        self.assertNotIn("Regime 0", s["full"])
        self.assertIn("Ranked by the size of its learned weights", s["full"])
        self.assertGreater(s["words"], len(s["full"].split()))
        # And the regimes still carry their own narrated sentences.
        self.assertTrue(any("Regime 0" in (r.get("narrated") or r["text"])
                            for r in s["regimes"]))

    def test_case_insensitive_dataset_and_entity(self):
        # The pipeline writes the dataset string verbatim, so `skab` and `SKAB`
        # both occur; on Linux they are different directories.
        self.assertIsNotNone(artifacts.build_payload("skab", "7"))
        self.assertIsNotNone(artifacts.build_payload("SKAB", "7"))

    def test_glossary_is_separated_from_narrative(self):
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        ts = next(s for s in p["stages"] if s["key"] == "thompson_sampling")
        self.assertEqual(ts["info"], "Expected reward is the weights applied to the window.")
        self.assertEqual(ts["full"], "Thompson Sampling ranked NN_1 first.")
        self.assertNotIn("INFO:", ts["full"])

    def test_headline_pick_handles_each_stages_naming(self):
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        by = {s["key"]: s for s in p["stages"]}
        self.assertEqual(by["thompson_sampling"]["top_pick"], "NN_1")
        self.assertEqual(by["monte_carlo"]["top_pick"], "LOF_1")   # top_pick_f1

    def test_regimes_are_ordered_and_carry_their_window_range(self):
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        ts = next(s for s in p["stages"] if s["key"] == "thompson_sampling")
        self.assertEqual([r["index"] for r in ts["regimes"]], [0, 1])
        self.assertEqual(ts["regimes"][0]["leader"], "NN_3")
        self.assertEqual((ts["regimes"][1]["start"], ts["regimes"][1]["end"]), (5, 18))

    def test_iteration_suffixed_stage_is_found(self):
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        ra = next(s for s in p["stages"] if s["key"] == "rank_aggregation_robust")
        self.assertEqual(ra["top_pick"], "LOF_1")

    def test_missing_stage_carries_its_reason(self):
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        gan = next(m for m in p["missing_stages"] if m["key"] == "gan")
        self.assertEqual(gan["status"], "not_available")
        self.assertIn("no explainability layer", gan["note"])

    def test_decision_and_agreement_come_from_the_global_ir(self):
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        self.assertEqual(p["decision_text"], "The final decision is the ensemble {A, B}.")
        self.assertEqual(p["decision"]["framework_choice"], "ensemble")
        self.assertEqual(p["agreement"][0]["source"], "thompson")
        self.assertFalse(p["agreement"][0]["agrees"])
        self.assertFalse(p["degraded"])

    def test_final_consensus_is_filtered_from_agreement(self):
        """Older result trees still carry the tautological row; the reader hides
        it so existing results read correctly without re-running anything."""
        doc = json.loads((self.ir_dir / "ir_global_iter0.json").read_text())
        doc["stage_agreement"]["final_consensus"] = {
            "top_pick": "NN_3", "agrees_with_final_single": True}
        _write(self.ir_dir / "ir_global_iter0.json", doc)
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        sources = [a["source"] for a in p["agreement"]]
        self.assertNotIn("final_consensus", sources)
        self.assertIn("thompson", sources)

    def test_faithfulness_overall_and_per_stage(self):
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        self.assertEqual(p["model"], "qwen2.5:7b-instruct")
        self.assertEqual(p["faithfulness"]["n_claims"], 259)
        ts = next(s for s in p["stages"] if s["key"] == "thompson_sampling")
        self.assertEqual(ts["faithfulness"]["n_claims"], 58)
        self.assertFalse(ts["faithfulness"]["repaired"])

    def test_iteration_mismatch_takes_the_newest_global_ir(self):
        # Real on SMD/machine-1-6: the comprehensive tree uses --iteration while
        # explanations use OFFLINE_ITERATION, so both can sit side by side.
        import time
        newer = dict(json.loads((self.ir_dir / "ir_global_iter0.json").read_text()))
        newer["iteration"] = 5
        newer["decision"] = {"framework_choice": "single_model"}
        _write(self.ir_dir / "ir_global_iter5.json", newer)
        os.utime(self.ir_dir / "ir_global_iter5.json", (time.time() + 10, time.time() + 10))
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        self.assertEqual(p["iteration"], 5)
        self.assertEqual(p["decision"]["framework_choice"], "single_model")

    def test_missing_global_ir_degrades_but_still_serves_stages(self):
        (self.ir_dir / "ir_global_iter0.json").unlink()
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        self.assertTrue(p["degraded"])
        self.assertIsNone(p["decision_text"])
        self.assertTrue(p["stages"])          # per-stage prose still readable
        self.assertEqual(p["missing_stages"], [])

    def test_unknown_entity_returns_none(self):
        self.assertIsNone(artifacts.build_payload("SKAB", "999"))
        self.assertIsNone(artifacts.build_payload("NOPE", "7"))

    def test_entity_summary_and_known_entities(self):
        self.assertEqual(artifacts.known_entities(), [("SKAB", "7")])
        s = artifacts.entity_summary(self.DATASET, self.ENTITY)
        self.assertEqual(s["framework_choice"], "ensemble")
        # Both Thompson stages plus Monte Carlo; GAN is not_available.
        self.assertEqual(s["n_stages"], 3)
        self.assertEqual(s["hallucination_rate"], 0.0)


# ── the summary seam ─────────────────────────────────────────────────────────

class TestComprehensiveReport(ArtifactTreeCase):
    """The pipeline's numeric report — the binding record for those numbers."""

    def test_payload_links_the_report_without_inlining_it(self):
        payload = artifacts.build_payload("SKAB", "7")
        report = payload["comprehensive"]
        self.assertEqual(report["iteration"], 5)
        self.assertEqual(report["url"], "/report/SKAB/7")
        self.assertTrue(report["name"].endswith("_iter5.txt"))
        # The page links to it; the payload must not carry the whole file.
        self.assertNotIn("text", report)

    def test_absent_report_is_none_not_an_error(self):
        for path in self.report_dir.iterdir():
            path.unlink()
        self.assertIsNone(artifacts.comprehensive_info("SKAB", "7"))
        self.assertIsNone(artifacts.build_payload("SKAB", "7")["comprehensive"])

    def test_newest_iteration_wins(self):
        # Two iterations can coexist; the index is never derived from the name.
        newer = self.report_dir / "comprehensive_results_SKAB_7_iter9.txt"
        _write(newer, REPORT_TEXT)
        os.utime(newer, (time.time() + 10, time.time() + 10))
        self.assertEqual(artifacts.comprehensive_info("SKAB", "7")["iteration"], 9)

    def test_report_body_is_served_verbatim(self):
        report = artifacts.comprehensive_report("skab", "7")   # case-insensitive
        self.assertEqual(report["text"], REPORT_TEXT)

    def test_case_insensitive_and_unknown_entity(self):
        self.assertIsNotNone(artifacts.comprehensive_path("skab", "7"))
        self.assertIsNone(artifacts.comprehensive_path("SKAB", "999"))
        self.assertIsNone(artifacts.comprehensive_path("NOPE", "7"))


class TestSummarizeSeam(ArtifactTreeCase):

    def test_unconfigured_stage_keeps_the_whole_narrative(self):
        out = summarize.summarize("Some narrative.")
        self.assertEqual(out["summary"], "Some narrative.")
        self.assertTrue(out["is_full"])
        self.assertEqual(out["mode"], "full")

    def test_rank_aggregation_final_is_left_whole(self):
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        final = next((s for s in p["stages"]
                      if s["key"] == "rank_aggregation_final"), None)
        if final and final["full"]:
            self.assertTrue(final["summary_is_full"])
            self.assertEqual(final["summary"], final["full"])

    def test_payload_never_has_an_empty_summary_for_a_non_empty_narrative(self):
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        for s in p["stages"]:
            if s["full"]:
                self.assertTrue(s["summary"], s["key"])

    def test_swapping_the_summariser_is_one_function(self):
        """The seam's actual contract: replace `summarize` and the payload keeps
        its shape, with `summary_is_full` flipping so the frontend knows to
        offer an expand."""
        original = artifacts.summarize
        try:
            artifacts.summarize = lambda text, stage=None, ir_doc=None: {
                "summary": text.split(".")[0] + ".", "is_full": False, "mode": "stub"}
            p = artifacts.build_payload(self.DATASET, self.ENTITY)
            ts = next(s for s in p["stages"] if s["key"] == "thompson_sampling")
            self.assertEqual(ts["summary"], "Thompson Sampling ranked NN_1 first.")
            self.assertFalse(ts["summary_is_full"])
            self.assertEqual(ts["summary_mode"], "stub")
            self.assertTrue(ts["full"])       # full text still present alongside
        finally:
            artifacts.summarize = original

    def test_a_summariser_failure_degrades_to_the_full_text(self):
        """A broken summariser must never blank a stage card."""
        original = summarize._TABLE_BUILDERS["ga_combination"]
        try:
            def boom(_):
                raise RuntimeError("nope")
            summarize._TABLE_BUILDERS["ga_combination"] = boom
            out = summarize.summarize("A narrative.", stage="ga_combination",
                                      ir_doc={"evidence": []})
            self.assertEqual(out["summary"], "A narrative.")
            self.assertTrue(out["is_full"])
        finally:
            summarize._TABLE_BUILDERS["ga_combination"] = original

    def test_glossary_is_never_summarised(self):
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        ts = next(s for s in p["stages"] if s["key"] == "thompson_sampling")
        self.assertNotIn("Expected reward is", ts["summary"])


def _stage_ir(stage, atoms, output=None):
    return {"ir_version": "1.0", "stage": stage, "dataset": "DS", "entity": "e1",
            "output": output or {}, "evidence": atoms, "caveats": [],
            "required_atom_ids": [], "confidence": {}}


class TestSummaryDropsAtomClasses(unittest.TestCase):
    """The default view is the narrative minus one class of fact. Sentences are
    attributed to atoms by shared names and numbers — nothing is paraphrased,
    so every sentence shown is one the verifier already scored."""

    def test_ga_selection_moves_exclusions_to_the_extended_view(self):
        ir_doc = _stage_ir("ga_selection", [
            {"id": "o", "type": "stage_output", "subject": "best_ensemble",
             "value": ["LOF_1", "NN_1"],
             "text": "The genetic algorithm selected the ensemble {LOF_1, NN_1}."},
            {"id": "b", "type": "member_reason", "subject": "both",
             "value": {"detectors": ["LOF_1"]},
             "text": "LOF_1 was chosen for both high utility and high stability."},
            {"id": "x", "type": "excluded_group", "subject": "plain",
             "value": {"detectors": ["CBLOF_3"]},
             "text": "CBLOF_3 had low utility and low stability, and was left out."},
        ])
        narrative = ("The genetic algorithm selected the ensemble of LOF_1 and "
                     "NN_1. LOF_1 was chosen for both high utility and high "
                     "stability. CBLOF_3 was left out for low utility.")
        out = summarize.summarize(narrative, stage="ga_selection", ir_doc=ir_doc)
        self.assertFalse(out["is_full"])
        self.assertIn("LOF_1 was chosen", out["summary"])
        self.assertNotIn("CBLOF_3", out["summary"])

    def test_monte_carlo_moves_win_regions_to_the_extended_view(self):
        ir_doc = _stage_ir("monte_carlo", [
            {"id": "t", "type": "stage_output", "subject": "LOF_1", "value": {},
             "text": "In the production test, LOF_1 ranked first by F1."},
            {"id": "w", "type": "win_region", "subject": "NN_3", "value": {},
             "text": "NN_3 won by F1 at noise levels 0.042 and 0.158."},
            {"id": "r", "type": "surrogate_win_rates", "subject": "rates",
             "value": {}, "text": "Across the sweep LOF_1 led with 39.0%."},
        ])
        narrative = ("In the production test, LOF_1 ranked first by F1. NN_3 "
                     "won by F1 at noise levels 0.042 and 0.158. Across the "
                     "sweep LOF_1 led with 39.0%.")
        out = summarize.summarize(narrative, stage="monte_carlo", ir_doc=ir_doc)
        self.assertNotIn("0.042", out["summary"])
        self.assertIn("39.0%", out["summary"])

    def test_off_by_moves_both_importance_families(self):
        ir_doc = _stage_ir("off_by_threshold", [
            {"id": "w", "type": "stage_output", "subject": "LOF_1", "value": {},
             "text": "LOF_1 was the highest-ranked model of the stage."},
            {"id": "v", "type": "feature_importance", "subject": "NN_3",
             "value": {}, "text": "Against NN_3, the property that best "
                                  "separates those points is position (importance 0.71)."},
            {"id": "s", "type": "summary", "subject": "position", "value": {},
             "text": "Across all competitors, LOF_1's edge is best explained "
                     "by position (mean importance 0.87)."},
        ])
        narrative = ("LOF_1 was the highest-ranked model of the stage. Against "
                     "NN_3, the best separator is position (importance 0.71). "
                     "Across all competitors, LOF_1's edge is best explained by "
                     "position (mean importance 0.87).")
        out = summarize.summarize(narrative, stage="off_by_threshold", ir_doc=ir_doc)
        self.assertNotIn("0.71", out["summary"])
        self.assertNotIn("0.87", out["summary"])
        self.assertIn("highest-ranked", out["summary"])

    def test_an_unattributable_sentence_is_kept(self):
        """Dropping happens only on positive evidence: a sentence that matches
        no atom is shown rather than silently lost."""
        ir_doc = _stage_ir("monte_carlo", [
            {"id": "w", "type": "win_region", "subject": "NN_3", "value": {},
             "text": "NN_3 won by F1 at noise levels 0.042."},
        ])
        out = summarize.summarize(
            "Something the verifier never saw. NN_3 won at 0.042.",
            stage="monte_carlo", ir_doc=ir_doc)
        self.assertIn("Something the verifier never saw.", out["summary"])
        self.assertNotIn("0.042", out["summary"])


class TestCaveatsLeaveThePros(unittest.TestCase):
    """The card renders the caveats verbatim from the IR in a section of their
    own, so a narrated restatement is the same limitation twice — in looser
    words, and wherever the narrator chose to put it. Every stage strips them
    from every view of the prose. They are matched lexically, not by the
    name/number scorer: most carry neither a detector name nor a number, so
    that scorer cannot see them at all."""

    CAVEAT = ("Correctness is judged on thresholded predictions (the F1 side); "
              "PR-AUC has no per-point notion of correct or incorrect.")

    def _ir(self, stage):
        return {"ir_version": "1.0", "stage": stage, "dataset": "DS",
                "entity": "e1", "output": {},
                "evidence": [
                    {"id": "w", "type": "stage_output", "subject": "LOF_1",
                     "value": {}, "text": "LOF_1 was the highest-ranked model."},
                    {"id": "v", "type": "feature_importance", "subject": "NN_3",
                     "value": {}, "text": "Against NN_3 the separator is position "
                                          "(importance 0.71)."},
                ],
                "caveats": [{"id": "c", "type": "caveat", "subject": "scope",
                             "value": None, "text": self.CAVEAT}],
                "required_atom_ids": [], "confidence": {}}

    NARRATIVE = ("LOF_1 was the highest-ranked model. Against NN_3 the separator "
                 "is position (importance 0.71). Note that correctness is judged "
                 "on thresholded predictions (the F1 side), and PR-AUC has no "
                 "per-point notion of correct or incorrect.")

    def test_caveat_leaves_the_default_view(self):
        out = summarize.summarize(self.NARRATIVE, stage="off_by_threshold",
                                  ir_doc=self._ir("off_by_threshold"))
        self.assertNotIn("thresholded predictions", out["summary"])
        self.assertIn("highest-ranked model", out["summary"])

    def test_a_stage_without_a_full_text_view_still_drops_them(self):
        """thompson_sampling renders no full-text disclosure, which used to
        exempt it. It no longer needs one: the caveats section is where they
        land, and it is on every card."""
        ir_doc = self._ir("thompson_sampling")
        ir_doc["evidence"][1] = {"id": "r0", "type": "regime", "subject": "NN_3",
                                 "value": {}, "text": "Regime 0 was led by NN_3."}
        narrative = ("LOF_1 was the highest-ranked model. Regime 0 was led by "
                     "NN_3. Note that correctness is judged on thresholded "
                     "predictions (the F1 side), and PR-AUC has no per-point "
                     "notion of correct or incorrect.")
        out = summarize.summarize(narrative, stage="thompson_sampling", ir_doc=ir_doc)
        self.assertNotIn("thresholded predictions", out["summary"])
        self.assertNotIn("thresholded predictions", out["body"])
        self.assertNotIn("Regime 0", out["summary"])

    def test_a_table_lead_is_never_a_caveat(self):
        ir_doc = {"ir_version": "1.0", "stage": "rank_aggregation_robust",
                  "dataset": "DS", "entity": "e1", "output": {},
                  "evidence": [
                      {"id": "s1", "type": "source_role", "subject": "GAN_F1",
                       "value": {"influence_rank": 1, "agreement_rank": 1,
                                 "borda_rank": 1, "pattern": "consistent"},
                       "text": "GAN_F1 shaped the consensus most."}],
                  "caveats": [{"id": "c", "type": "caveat", "subject": "agg",
                               "value": None,
                               "text": "The consensus ranking is produced by "
                                       "Markov-chain rank aggregation over the "
                                       "source rankings."}],
                  "required_atom_ids": [], "confidence": {}}
        # No stage_output sentence, so the lead falls back — and must skip the
        # caveat rather than introduce the table with its own limitation.
        narrative = ("The consensus ranking is produced by Markov-chain rank "
                     "aggregation over the source rankings. GAN_F1 shaped the "
                     "consensus most.")
        out = summarize.summarize(narrative, stage="rank_aggregation_robust",
                                  ir_doc=ir_doc)
        self.assertEqual(out["summary"], "GAN_F1 shaped the consensus most.")

    def test_sharing_vocabulary_is_not_a_caveat(self):
        """An evidence sentence that happens to reuse a caveat's words must
        survive; the bar is deliberately high."""
        ir_doc = self._ir("off_by_threshold")
        hits = summarize.caveat_sentences(
            "LOF_1 was judged correct on 4 points.", ir_doc)
        self.assertEqual(hits, [])

    def test_a_stage_with_no_spec_at_all_still_drops_them(self):
        """rank_aggregation_final holds nothing else back — the stripping is
        not part of the per-stage policy, it happens before it."""
        out = summarize.summarize(self.NARRATIVE, stage="rank_aggregation_final",
                                  ir_doc=self._ir("rank_aggregation_final"))
        self.assertTrue(out["is_full"])
        self.assertNotIn("thresholded predictions", out["summary"])
        self.assertIn("highest-ranked model", out["summary"])

    def test_the_full_text_view_is_stripped_too(self):
        """`body` is what the full-text disclosure renders. Stripping only the
        summary would move the duplicate one click away, not remove it."""
        out = summarize.summarize(self.NARRATIVE, stage="off_by_threshold",
                                  ir_doc=self._ir("off_by_threshold"))
        self.assertNotIn("thresholded predictions", out["body"])
        self.assertIn("importance 0.71", out["body"])

    def test_an_all_caveat_narrative_is_never_blanked(self):
        """A matcher that swallowed the whole narrative is a matcher failure.
        A redundant card beats an empty one."""
        ir_doc = self._ir("rank_aggregation_final")
        out = summarize.summarize("Note that " + self.CAVEAT.lower(),
                                  stage="rank_aggregation_final", ir_doc=ir_doc)
        self.assertIn("thresholded predictions", out["summary"])


class TestThompsonRegimeHandling(unittest.TestCase):

    def _ir(self):
        atoms = [
            {"id": "ts.output.top", "type": "stage_output", "subject": "NN_1",
             "value": {}, "text": "Thompson Sampling ranked NN_1 first."},
            {"id": "ts.regimes.summary", "type": "regime_summary",
             "subject": "regimes", "value": {},
             "text": "The 173 windows split into 2 regimes led by 2 detectors."},
            {"id": "ts.winner.channels", "type": "winner_channels",
             "subject": "NN_1", "value": {},
             "text": "Across the regimes NN_1 led, channel 7 contributed most."},
        ]
        for i, (lead, a, b) in enumerate(((("NN_3"), 0, 4), ("NN_1", 5, 18))):
            atoms.append({"id": f"ts.regime.{i}", "type": "regime", "subject": lead,
                          "value": {"start": a, "end": b, "leader": lead},
                          "text": f"Regime {i} (windows {a} to {b}) was led by {lead}."})
        return _stage_ir("thompson_sampling", atoms)

    NARRATIVE = (
        "Thompson Sampling ranked NN_1 first. The run was divided into 2 "
        "regimes led by 2 detectors. Regime 0 (windows 0 to 4) was led by "
        "NN_3, with channel 7 raising its reward. Regime 1 (windows 5 to 18) "
        "was led by NN_1. Across the regimes NN_1 led, channel 7 contributed "
        "most.")

    # A regime's second sentence — the deviation clause — carries no detector
    # name and no window range. Its ONLY anchor is "In regime N".
    TWO_SENTENCE_NARRATIVE = (
        "Thompson Sampling ranked NN_1 first. The run was divided into 2 "
        "regimes led by 2 detectors. Regime 0 (windows 0 to 4) was led by "
        "NN_3, with channel 7 raising its reward. In regime 0, channel 4 "
        "departed furthest from its usual contribution, running below it. "
        "Regime 1 (windows 5 to 18) was led by NN_1. In regime 1, channel 7 "
        "departed furthest from its usual contribution, running above it. "
        "Across the regimes NN_1 led, channel 7 contributed most.")

    def test_regime_walk_leaves_the_default_view(self):
        out = summarize.summarize(self.NARRATIVE, stage="thompson_sampling",
                                  ir_doc=self._ir())
        self.assertNotIn("Regime 0", out["summary"])
        self.assertNotIn("Regime 1", out["summary"])

    def test_ordinal_worded_regimes_attach_to_nothing(self):
        """Why the prompt demands the literal "Regime N".

        A narrative that says "the first regime" pairs with no regime atom, and
        there is NO error: `narrated` stays empty and the disclosure quietly
        renders the IR's own wording instead. Every regime on SMD's ranking
        stage was in that state, and the page looked fine.
        """
        ir_doc = self._ir()
        ordinal = ("Thompson Sampling ranked NN_1 first. In the first regime "
                   "(windows 0 to 4), NN_3 was in charge. The second regime "
                   "(windows 5 to 18) was led by NN_1.")
        regimes = artifacts._regimes_from_ir(ir_doc)
        artifacts._attach_narrated_regimes(regimes, ordinal, ir_doc)
        self.assertTrue(all(not r.get("narrated") for r in regimes))

        numbered = ("Thompson Sampling ranked NN_1 first. Regime 0 (windows 0 "
                    "to 4) was led by NN_3. Regime 1 (windows 5 to 18) was led "
                    "by NN_1.")
        regimes = artifacts._regimes_from_ir(ir_doc)
        artifacts._attach_narrated_regimes(regimes, numbered, ir_doc)
        self.assertTrue(all(r.get("narrated") for r in regimes))

    def test_a_regimes_second_sentence_goes_with_its_regime(self):
        """Resilience, not the primary path: the IR now packs every claim about
        a regime into ONE sentence, so there is normally no second sentence to
        strand. But the narrator may still split one off, and if it does the
        clause must carry the regime number — that name is the only thing
        attribution can key on. It has no detector name, and its lone channel
        index gets captured by whichever unrelated atom holds that integer
        ("channel 4" landing on the regime summary's "4 detectors"). Unanchored,
        such sentences escaped the walk, survived the summary drop, and piled up
        as a block of context-free sentences at the end of the card.
        """
        ir_doc = self._ir()
        out = summarize.summarize(self.TWO_SENTENCE_NARRATIVE,
                                  stage="thompson_sampling", ir_doc=ir_doc)
        # Gone from the default view, with the rest of the walk.
        self.assertNotIn("departed furthest", out["summary"])
        # And filed under the right regime — channel 4 with 0, channel 7 with 1.
        regimes = artifacts._regimes_from_ir(ir_doc)
        artifacts._attach_narrated_regimes(regimes, self.TWO_SENTENCE_NARRATIVE,
                                           ir_doc)
        self.assertIn("channel 4 departed furthest", regimes[0]["narrated"])
        self.assertNotIn("channel 7 departed", regimes[0]["narrated"])
        self.assertIn("channel 7 departed furthest", regimes[1]["narrated"])
        # The roll-up says "regimes" with no index and must NOT be swallowed by
        # the last regime just because it trails the walk.
        self.assertNotIn("contributed most", regimes[1]["narrated"])

    def test_the_regime_disclosure_is_the_only_extended_view(self):
        """The regime sentences belong beside their SHAP plots, not in a second
        full-text disclosure that would repeat them without the plots."""
        out = summarize.summarize(self.NARRATIVE, stage="thompson_sampling",
                                  ir_doc=self._ir())
        self.assertEqual(out["extended_in"], "regimes")
        # Every other summarised stage still offers its own full text.
        other = summarize.summarize(
            "LOF_1 was first. Against NN_3 the separator is position (0.71).",
            stage="off_by_threshold",
            ir_doc=_stage_ir("off_by_threshold", [
                {"id": "w", "type": "stage_output", "subject": "LOF_1",
                 "value": {}, "text": "LOF_1 was first."},
                {"id": "v", "type": "feature_importance", "subject": "NN_3",
                 "value": {}, "text": "Against NN_3 the separator is position (0.71)."},
            ]))
        self.assertIsNone(other.get("extended_in"))

    def test_roll_up_sentences_survive_the_regime_walk(self):
        """These lost ties against regime atoms and vanished with them: both
        share one detector name and one channel number with some regime. A
        regime is identified by its index or not at all."""
        out = summarize.summarize(self.NARRATIVE, stage="thompson_sampling",
                                  ir_doc=self._ir())
        self.assertIn("divided into 2 regimes", out["summary"])
        self.assertIn("Across the regimes NN_1 led", out["summary"])
        self.assertIn("ranked NN_1 first", out["summary"])

    def test_each_regime_carries_its_narrated_sentence(self):
        regimes = artifacts._regimes_from_ir(self._ir())
        artifacts._attach_narrated_regimes(regimes, self.NARRATIVE, self._ir())
        self.assertEqual(regimes[0]["narrated"],
                         "Regime 0 (windows 0 to 4) was led by NN_3, with "
                         "channel 7 raising its reward.")
        self.assertIn("Regime 1", regimes[1]["narrated"])
        # The deterministic text stays as the fallback.
        self.assertTrue(regimes[0]["text"])

    def test_regimes_without_a_narrated_sentence_fall_back(self):
        regimes = artifacts._regimes_from_ir(self._ir())
        artifacts._attach_narrated_regimes(regimes, "Nothing about regimes here.",
                                           self._ir())
        self.assertNotIn("narrated", regimes[0])


class TestThompsonRankingStage(unittest.TestCase):
    """The ||mu||^2 sibling. It reuses the regime machinery rather than forking
    it, so these tests pin that the shared path really is shared."""

    def _ir(self):
        atoms = [
            {"id": "tsr.output.top", "type": "stage_output", "subject": "NN_1",
             "value": {"top": "NN_1", "score": 0.559322, "runner_up": "NN_2",
                       "margin": 0.00041},
             "text": "Ranked by the size of its learned weights, NN_1 scored "
                     "0.559322, ahead of NN_2 by 0.000410."},
            {"id": "tsr.winner.channels", "type": "winner_channels",
             "subject": "NN_1",
             "value": {"channel": 3, "total": 0.5,
                       "per_channel": [[3, 0.25], [0, 0.15], [7, 0.10]]},
             "text": "NN_1's score is built mostly from channel 3 (50.0%)."},
            {"id": "tsr.gap.runner_up", "type": "rank_gap", "subject": "NN_1",
             "value": {"rivals": ["NN_2"], "runner_up": "NN_2",
                       "per_channel": [[3, 0.091], [0, -0.058]]},
             "text": "NN_1's lead over NN_2 came mostly from channel 3 (0.091)."},
            {"id": "tsr.support", "type": "support", "subject": "NN_1",
             "value": {"winner_selections": 23, "runner_up": "NN_2"},
             "text": "NN_1 was selected in 23 of the 173 windows, against 30 "
                     "for NN_2."},
            {"id": "tsr.regimes.summary", "type": "regime_summary",
             "subject": "regimes", "value": {},
             "text": "Leadership splits into 2 regimes led by 2 detectors."},
        ]
        for i, (lead, a, b) in enumerate((("NN_2", 10, 71), ("NN_1", 72, 172))):
            atoms.append({"id": f"tsr.regime.{i}", "type": "regime", "subject": lead,
                          "value": {"start": a, "end": b, "leader": lead},
                          "text": f"Regime {i} (windows {a} to {b}) was led by {lead}."})
        return _stage_ir("thompson_ranking", atoms)

    NARRATIVE = (
        "Ranked by the size of its learned weights, NN_1 scored 0.559322, ahead "
        "of NN_2 by 0.000410. NN_1's score is built mostly from channel 3 "
        "(50.0%). NN_1's lead over NN_2 came mostly from channel 3 (0.091). "
        "NN_1 was selected in 23 of the 173 windows, against 30 for NN_2. "
        "Leadership splits into 2 regimes led by 2 detectors. Regime 0 (windows "
        "10 to 71) was led by NN_2. Regime 1 (windows 72 to 172) was led by NN_1.")

    def test_summary_stops_after_the_answer(self):
        """The default view answers the question — winner, the channels its
        score is built from, which channels decided the margin — and stops.
        The selection counts, the regime walk and the limitations are all
        supporting detail behind the click."""
        out = summarize.summarize(self.NARRATIVE, stage="thompson_ranking",
                                  ir_doc=self._ir())
        self.assertIn("built mostly from channel 3", out["summary"])
        self.assertIn("lead over NN_2", out["summary"])
        for held_back in ("Regime 0", "Regime 1", "Leadership splits",
                          "was selected in"):
            self.assertNotIn(held_back, out["summary"])
        # Unlike its sibling this stage keeps a full-text disclosure, which is
        # what makes dropping the caveats safe.
        self.assertIsNone(out.get("extended_in"))
        self.assertFalse(out["is_full"])

    def test_the_regime_walk_appears_beside_its_plots_and_nowhere_else(self):
        """The regime sentences are held back from the summary AND from the
        full-text view: they belong next to their own per-regime figures, and a
        card that also lists them under "read the full explanation" is showing
        the same eleven sentences twice, the second time without the plots."""
        out = summarize.summarize(self.NARRATIVE, stage="thompson_ranking",
                                  ir_doc=self._ir())
        extended = out["extended"]
        self.assertNotIn("Regime 0", extended)
        self.assertNotIn("Regime 1", extended)
        # Everything else the summary held back is exactly what the click buys.
        self.assertIn("Leadership splits", extended)
        self.assertIn("was selected in", extended)


    def test_a_drop_stage_can_also_carry_a_table(self):
        """The channel split is the stage's headline answer and 38 channels on
        SMD do not read as prose, so this stage holds sentences back AND
        tabulates — a combination no other stage used before."""
        out = summarize.summarize(self.NARRATIVE, stage="thompson_ranking",
                                  ir_doc=self._ir())
        table = out["table"]
        self.assertEqual(table["columns"],
                         ["Channel", "Share", "Contribution", "vs NN_2"])
        self.assertEqual([r[0] for r in table["rows"]],
                         ["channel 3", "channel 0", "channel 7"])
        self.assertEqual(table["collapse_after"], 5)
        self.assertEqual(table["rows"][0][1], "50.0%")
        self.assertEqual(table["rows"][0][3], "+0.091000")
        # A channel the gap atom does not mention renders as a blank, not a zero:
        # "no delta recorded" and "the two are level" are different claims.
        self.assertIsNone(table["rows"][2][3])

    def test_the_table_keeps_every_channel_and_folds_the_tail(self):
        """The shares only mean anything because they sum to the whole score,
        so no channel may be dropped however many there are — SMD carries 38.
        The default view shows five; the rest are folded, not discarded."""
        ir_doc = self._ir()
        atom = next(a for a in ir_doc["evidence"] if a["type"] == "winner_channels")
        atom["value"]["per_channel"] = [[c, 1.0] for c in range(1000)]
        table = summarize._ts_ranking_table(ir_doc)
        self.assertEqual(len(table["rows"]), 1000)
        self.assertEqual(table["collapse_after"], 5)

    def test_regime_atoms_are_found_despite_the_different_prefix(self):
        """One regex serves `ts.regime.N` and `tsr.regime.N`; a prefix-anchored
        one would silently find nothing here and every regime would vanish."""
        regimes = artifacts._regimes_from_ir(self._ir())
        self.assertEqual([r["index"] for r in regimes], [0, 1])
        artifacts._attach_narrated_regimes(regimes, self.NARRATIVE, self._ir())
        self.assertIn("was led by NN_2", regimes[0]["narrated"])
        self.assertIn("was led by NN_1", regimes[1]["narrated"])

    def test_no_stage_carries_a_per_stage_caveat_policy(self):
        """Caveat stripping is global and unconditional. A leftover
        `drop_caveats` key would read as though some stage still opted in."""
        for key, spec in summarize._STAGE_SUMMARY.items():
            self.assertNotIn("drop_caveats", spec, key)


class TestSummaryTables(unittest.TestCase):
    """Stages whose answer is a ranking get a deterministic table from the IR —
    never parsed out of the rendered *_explainability_*.txt reports, which are
    a display format that no test would catch changing."""

    def test_ga_combination_table_omits_raw_markov_values(self):
        ir_doc = _stage_ir("ga_combination", [
            {"id": "s", "type": "stage_output", "subject": "best_ensemble",
             "value": {}, "text": "The combination step selected 2 detectors."},
            {"id": "d1", "type": "detector_role", "subject": "LOF_1",
             "value": {"final_rank": 1, "final_rank_tied": False,
                       "markov_score": 0.1835, "mean_abs_shap_rank": 2,
                       "pfi_rank": 1, "ale_rank": 1, "sign": "positive",
                       "sign_support": []},
             "text": "LOF_1 carries the most weight."},
            {"id": "d2", "type": "detector_role", "subject": "NN_3",
             "value": {"final_rank": 2, "final_rank_tied": True,
                       "markov_score": 0.0777, "mean_abs_shap_rank": 1,
                       "pfi_rank": 2, "ale_rank": 3, "sign": "negative",
                       "sign_support": ["low_consistency"]},
             "text": "NN_3 carries the second-most weight."},
            {"id": "d3", "type": "detector_role", "subject": "CBLOF_2",
             "value": {"final_rank": 3, "final_rank_tied": False,
                       "markov_score": 0.0431, "mean_abs_shap_rank": 3,
                       "pfi_rank": 3, "ale_rank": 2, "sign": "not_available",
                       "sign_support": []},
             "text": "CBLOF_2 carries the third-most weight."},
            {"id": "sg", "type": "sign_summary", "subject": "sign", "value": {},
             "text": "LOF_1 had a positive sign, while NN_3 had negative."},
        ])
        out = summarize.summarize(
            "The combination step selected 2 detectors. LOF_1 carries the most "
            "weight. LOF_1 had a positive sign, while NN_3 had negative.",
            stage="ga_combination", ir_doc=ir_doc)
        # The per-detector walk is what the table replaces, so only those
        # sentences leave the default view.
        self.assertNotIn("carries the most weight", out["summary"])
        self.assertIn("selected 2 detectors", out["summary"])
        # The signs are a finding, not a restatement of the table, so they stay.
        self.assertIn("had a positive sign", out["summary"])

        table = out["table"]
        self.assertEqual(table["columns"],
                         ["Weight rank", "Detector", "|SHAP| rank", "PFI rank",
                          "ALE rank", "Sign"])
        self.assertEqual(table["rows"][0], [1, "LOF_1", 2, 1, 1, "positive"])
        # A thinly-evidenced sign is still SHOWN. How well it is supported is a
        # caveat and lives in the caveats section — never a second column here.
        self.assertEqual(table["rows"][1],
                         ["2 (tie)", "NN_3", 1, 2, 3, "negative"])
        self.assertNotIn("support", " ".join(table["columns"]).lower())
        # The IR's underscored state renders as words for a reader.
        self.assertEqual(table["rows"][2], [3, "CBLOF_2", 3, 3, 2, "not available"])
        # The raw stationary-distribution values never reach the page: their
        # ties are decided at the 16th decimal.
        self.assertNotIn("0.1835", json.dumps(table))

    def test_rank_aggregation_table_carries_all_three_ranks(self):
        ir_doc = _stage_ir("rank_aggregation_robust", [
            {"id": "t", "type": "stage_output", "subject": "LOF_1", "value": {},
             "text": "Its first-ranked detector is LOF_1."},
            {"id": "s1", "type": "source_role", "subject": "GAN_F1",
             "value": {"influence_rank": 1, "agreement_rank": 1,
                       "borda_rank": 1, "pattern": "consistent"},
             "text": "GAN_F1 shaped the consensus most."},
            {"id": "s2", "type": "source_role", "subject": "GAN_PR_AUC",
             "value": {"influence_rank": 5, "agreement_rank": 2,
                       "borda_rank": 2, "pattern": "redundant_agreer"},
             "text": "GAN_PR_AUC shaped the consensus second most."},
        ])
        out = summarize.summarize("Its first-ranked detector is LOF_1. GAN_F1 "
                                  "shaped the consensus most.",
                                  stage="rank_aggregation_robust", ir_doc=ir_doc)
        table = out["table"]
        self.assertEqual(table["columns"], ["Overall standing", "Source",
                                            "Influence", "Agreement", "Pattern"])
        self.assertEqual(table["rows"][0], [1, "GAN_F1", 1, 1, "consistent"])
        self.assertEqual(table["rows"][1], [2, "GAN_PR_AUC", 5, 2, "redundant agreer"])
        # The lead is the narrative's own stage-output sentence, not invented copy.
        self.assertEqual(out["summary"], "Its first-ranked detector is LOF_1.")

class TestSummaryTableThroughThePayload(ArtifactTreeCase):

    def test_table_reaches_the_payload_with_the_full_text_behind_it(self):
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        ra = next(s for s in p["stages"] if s["key"] == "rank_aggregation_robust")
        self.assertEqual(ra["summary_mode"], "table")
        self.assertFalse(ra["summary_is_full"])
        self.assertEqual([r[1] for r in ra["summary_table"]["rows"]],
                         ["GAN_F1", "GAN_PR_AUC"])
        # The narrative stays available for the disclosure.
        self.assertIn("GAN_PR_AUC shaped the consensus second most", ra["full"])


# ── catalog ──────────────────────────────────────────────────────────────────

class TestCatalog(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.models = root / "trained_models"
        self.data = root / "datasets"
        for name in ("LOF_1", "NN_2"):
            p = self.models / "SKAB" / "7" / f"{name}.pth"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        meta = {"train_hyperparameters": {"seed": 1},
                "model_hyperparameters": {"contamination": 0.1, "window_size": 64}}
        (self.models / "SKAB" / "7" / "LOF_1.meta").write_bytes(pickle.dumps(meta))
        (self.models / "SKAB" / "7" / "NN_2.meta").write_bytes(b"not a pickle")
        (self.models / "SKAB" / "7" / "RNN_1.pth").write_bytes(b"x")   # stale
        (self.models / "NASA").mkdir()                                  # not loadable

        # The data root mirrors the real layouts: SKAB as one CSV per entity,
        # UCR as suffixed .txt files, SMD split across two aliased directories
        # with a train/test/test_label structure.
        for entity in ("0", "7", "12"):
            f = self.data / "SKAB" / f"{entity}.csv"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x")
        ucr = self.data / "Anomaly_Archive"
        ucr.mkdir(parents=True)
        (ucr / "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt").write_text("x")
        for directory, machine in (("SMD", "machine-1-1"), ("ServerMachineDataset", "machine-2-1")):
            train = self.data / directory / "train"
            train.mkdir(parents=True)
            (train / f"{machine}.txt").write_text("x")
            (self.data / directory / "test").mkdir()

        paths.reset_config_cache()
        paths._CONFIG_CACHE = {"trained_model_path": str(self.models),
                               "dataset_path": str(self.data), "results_path": None,
                               "overwrite": True}
        catalog.reset_cache()

    def tearDown(self):
        paths.reset_config_cache()
        catalog.reset_cache()
        self._tmp.cleanup()

    def test_only_loadable_datasets_are_listed(self):
        names = [d["name"] for d in catalog.datasets()]
        self.assertIn("skab", names)
        self.assertNotIn("nasa", names)     # on disk but not in VALID_DATASETS
        self.assertNotIn("NASA", names)

    def test_display_labels(self):
        labels = {d["name"]: d["label"] for d in catalog.datasets()}
        self.assertEqual(labels["skab"], "SKAB")
        self.assertEqual(labels["smd"], "SMD")
        self.assertEqual(labels["anomaly_archive"], "UCR")

    def test_entities_come_from_the_data_root_not_just_trained_models(self):
        """An entity with no checkpoint is still runnable — it trains first —
        so discovery must not be limited to trained_models."""
        entities = catalog.entities_for("skab")
        self.assertEqual(entities, ["0", "7", "12"])
        self.assertEqual(catalog.trained_entities("skab"), {"7"})

    def test_aliased_directories_merge_into_one_dataset(self):
        """SMD and ServerMachineDataset are the same dataset on disk."""
        names = [d["name"] for d in catalog.datasets()]
        self.assertEqual(names.count("smd"), 1)
        self.assertEqual(catalog.entities_for("smd"), ["machine-1-1", "machine-2-1"])
        self.assertEqual(catalog.entities_for("ServerMachineDataset"),
                         ["machine-1-1", "machine-2-1"])

    def test_ucr_entity_names_drop_the_index_suffix(self):
        """The loader keys on the first four underscore fields, so the UI must
        offer the same name the trained_models tree uses."""
        self.assertEqual(catalog.entities_for("anomaly_archive"),
                         ["001_UCR_Anomaly_DISTORTED1sddb40"])

    def test_detector_availability_and_labels(self):
        dets = {d["name"]: d for d in catalog.detectors_for("SKAB", "7")}
        self.assertEqual(len(dets), 11)      # always the canonical eleven
        self.assertTrue(dets["LOF_1"]["available"])
        self.assertFalse(dets["CBLOF_4"]["available"])
        self.assertEqual(dets["LOF_1"]["params"]["label"], "contamination 0.1")
        self.assertNotIn("RNN_1", dets)      # stale checkpoints are not selectable

    def test_corrupt_meta_degrades_to_no_params(self):
        dets = {d["name"]: d for d in catalog.detectors_for("SKAB", "7")}
        self.assertTrue(dets["NN_2"]["available"])
        self.assertIsNone(dets["NN_2"]["params"])

    def test_case_insensitive_entity_lookup(self):
        self.assertTrue(any(d["available"] for d in catalog.detectors_for("skab", "7")))

    def test_overwrite_warning_is_surfaced(self):
        self.assertIn("overwrite_on", [w["code"] for w in catalog.warnings()])

    def test_valid_datasets_matches_the_loader(self):
        """The list is copied rather than imported (importing Datasets.load
        would pull in pandas/sklearn); assert the copy is still accurate."""
        src = Path(__file__).resolve().parent.parent / "Datasets" / "load.py"
        text = src.read_text()
        line = next(l for l in text.splitlines() if l.strip().startswith("VALID_DATASETS"))
        listed = {t.strip().strip("'\"") for t in
                  line.split("[", 1)[1].rsplit("]", 1)[0].split(",") if t.strip()}
        self.assertEqual(listed, set(catalog.VALID_DATASETS))


# ── plots ────────────────────────────────────────────────────────────────────

class TestPlots(unittest.TestCase):

    def setUp(self):
        from WebUI import plots
        self.plots = plots
        self._tmp = tempfile.TemporaryDirectory()
        self.myresults = Path(self._tmp.name) / "myresults"
        self._saved = paths.MYRESULTS
        paths.MYRESULTS = self.myresults

    def tearDown(self):
        paths.MYRESULTS = self._saved
        self._tmp.cleanup()

    def _touch(self, rel):
        p = self.myresults / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x89PNG")
        return p

    def test_dedupe_keeps_newest_of_each_real_pattern(self):
        """The four naming irregularities that actually occur on disk."""
        d = self.myresults / "robustness" / "off_by" / "SKAB" / "7"
        files = []
        for ts in ("2026-01-01_00-00-00", "2026-08-05_19-19-42", "2026-03-02_11-11-11"):
            # off-by: trailing underscore, and a literal SPACE in the stem
            files.append(self._touch(f"robustness/off_by/SKAB/7/Data_vs_DataWithAnomalies_{ts}_.png"))
            files.append(self._touch(f"robustness/off_by/SKAB/7/SKAB_7_Misclassified Anomalies_{ts}.png"))
            # GAN: underscore stem AND trailing underscore
            files.append(self._touch(f"robustness/off_by/SKAB/7/SKAB_7_Misclassified_Anomalies_{ts}_.png"))
        plain = self._touch("robustness/off_by/SKAB/7/SKAB_7_off_by_point_importance.png")
        out = self.plots.dedupe_timestamped(sorted(d.glob("*.png")))
        by_name = {e["path"].name: e for e in out}
        self.assertEqual(len(out), 4)          # 3 collapsed groups + 1 untouched
        for name, entry in by_name.items():
            if entry["timestamp"] is None:
                self.assertEqual(name, plain.name)
                self.assertEqual(entry["n_older"], 0)
            else:
                self.assertEqual(entry["timestamp"], "2026-08-05_19-19-42", name)
                self.assertEqual(entry["n_older"], 2, name)

    def test_glob_escaping_handles_the_bracketed_filename(self):
        """ensemble_scores_..._['spikes'].png would otherwise be read as a
        glob character class and silently never match.

        Asserted on `_ls` itself rather than through a stage manifest: the
        escaping is _ls's job, and pinning it to whichever manifest happens to
        list that file made the check disappear the moment one stopped.
        """
        bracketed = self._touch(
            "GA_Ens/SKAB/7/ensemble_scores_SKAB_7_Data_vs_anomalies_['spikes'].png")
        found = self.plots._ls(bracketed.parent, "ensemble_scores_*.png")
        self.assertEqual([p.name for p in found], [bracketed.name])

    def test_ga_selection_leads_with_one_figure(self):
        """Utility x stability answers the stage's question; LOFO and survival
        are the inputs to it and browse. The injected-anomalies figure is about
        the data rather than the selection and is not listed at all."""
        for name in ("ga_selection_archetypes_SKAB_7", "ga_selection_utility_SKAB_7",
                     "ga_selection_survival_SKAB_7", "ga_selection_survival_all_SKAB_7",
                     "ensemble_scores_SKAB_7_Data"):
            self._touch(f"GA_Ens/SKAB/7/{name}.png")
        headline, gallery = self.plots._ga_selection("SKAB", "7")
        self.assertEqual([f["title"] for f in headline], ["Utility \u00d7 stability"])
        self.assertEqual(gallery[0]["title"], "LOFO")
        names = " ".join(f["name"] for f in gallery)
        self.assertIn("survival", names)
        self.assertNotIn("ensemble_scores", names)

    def test_regime_plots_key_on_the_zero_based_index(self):
        self._touch("Thomposon/SKAB/7/expected_rewards_50.png")
        self._touch("Thomposon/SKAB/7/shap_per_regime_50/regime_00_w0-4_NN_3.png")
        self._touch("Thomposon/SKAB/7/shap_per_regime_50/regime_13_w147-172_NN_2.png")
        r = self.plots.regime_plots("SKAB", "7")
        self.assertEqual(sorted(r), [0, 13])
        self.assertIn("Windows 147\u2013172, led by NN_2", r[13]["caption"])

    def test_each_thompson_stage_gets_its_own_regime_directory(self):
        """Both stages mint the same filename shape into one tree, so the only
        thing separating their regimes is the subdirectory the caller names.
        Crossing them would put expected-reward plots beside ||mu||^2 prose."""
        self._touch("Thomposon/SKAB/7/expected_rewards_50.png")
        self._touch("Thomposon/SKAB/7/shap_per_regime_50/regime_00_w0-4_NN_3.png")
        self._touch("Thomposon/SKAB/7/ranking_per_regime_50/regime_00_w10-71_NN_2.png")
        self._touch("Thomposon/SKAB/7/ranking_per_regime_50/regime_01_w72-172_NN_1.png")
        shap = self.plots.regime_plots("SKAB", "7")
        ranking = self.plots.regime_plots("SKAB", "7", "ranking_per_regime")
        self.assertEqual(sorted(shap), [0])
        self.assertEqual(sorted(ranking), [0, 1])
        self.assertIn("led by NN_3", shap[0]["caption"])
        self.assertIn("led by NN_2", ranking[0]["caption"])

    def test_ranking_builder_reads_only_its_own_prefix(self):
        """Both Thompson stages share TREE_THOMPSON and are separated purely by
        the `ranking_` filename prefix, exactly as the two GA stages are."""
        for name in ("expected_rewards_50", "expected_rewards_smoothed_50",
                     "selection_states_50", "ranking_final_50", "ranking_gap_50",
                     "ranking_criterion_50", "ranking_channels_50",
                     "ranking_channels_all_50"):
            self._touch(f"Thomposon/SKAB/7/{name}.png")
        ranking, _ = self.plots._ts_ranking("SKAB", "7")
        names = [f.get("name") for f in ranking if "name" in f]
        self.assertNotIn("expected_rewards_50.png", names)
        self.assertNotIn("selection_states_50.png", names)
        self.assertIn("ranking_final_50.png", names)
        self.assertIn("ranking_criterion_50.png", names)
        # The all-vs-top-3 toggle is a variant group, not two separate figures.
        variants = [f for f in ranking if "variants" in f]
        self.assertEqual(len(variants), 1)
        self.assertEqual(len(variants[0]["variants"]), 2)

        thompson, _ = self.plots._thompson("SKAB", "7")
        self.assertNotIn("ranking_final_50.png",
                         [f.get("name") for f in thompson if "name" in f])

    def test_plot_groups_do_not_prefix_collide(self):
        """result.js attaches lazy galleries with id.startsWith(plot_group), so
        one group being a prefix of another makes a card claim the other's
        galleries. `thompson` vs `ts_ranking` is the pair at risk."""
        groups = [s["plot_group"] for s in artifacts.STAGES]
        for a in groups:
            for b in groups:
                if a != b:
                    self.assertFalse(a.startswith(b), f"{a} starts with {b}")

    def test_monte_carlo_defaults_to_the_plain_variant(self):
        for suffix in ("F1_plain", "F1", "PRAUC_plain"):
            self._touch(f"robustness/MonteCarlo/SKAB/7/SKAB_7_MonteCarlo_noise_curves_{suffix}.png")
        headline, _ = self.plots._monte_carlo("SKAB", "7")
        self.assertEqual(headline[0]["default"], 0)
        self.assertIn("_F1_plain.png", headline[0]["variants"][0]["name"])

    def test_ale_views_are_one_toggle_with_the_plain_curve_first(self):
        """Both ALE figures share the `ga_combination_ale` prefix, and the
        dataset name follows it — so they are split by name, not by a glob
        character class that a dataset beginning with 'b' would defeat. The
        plain curve is the default: the bin rules are for digging."""
        self._touch("GA_Ens/bats/7/ga_combination_importance_bats_7.png")
        self._touch("GA_Ens/bats/7/ga_combination_ale_bats_7.png")
        self._touch("GA_Ens/bats/7/ga_combination_ale_bins_bats_7.png")
        headline, _ = self.plots._ga_combination("bats", "7")
        toggle = next(f for f in headline if "variants" in f)
        self.assertEqual(toggle["default"], 0)
        self.assertEqual([v["title"] for v in toggle["variants"]],
                         ["Plain", "Bin edges marked"])
        self.assertEqual(toggle["variants"][0]["name"],
                         "ga_combination_ale_bats_7.png")

    def test_ale_toggle_survives_a_missing_second_view(self):
        """Entities explained before the bin view existed keep the plain curve
        rather than losing the figure to a half-built toggle."""
        self._touch("GA_Ens/SKAB/7/ga_combination_ale_SKAB_7.png")
        headline, _ = self.plots._ga_combination("SKAB", "7")
        toggle = next(f for f in headline if "variants" in f)
        self.assertEqual([v["title"] for v in toggle["variants"]], ["Plain"])

    def test_aggregation_is_glob_driven(self):
        """_kendall_only only exists for two-source aggregations, so the set is
        whatever is on disk rather than a fixed list."""
        self._touch("robust_aggregated/SKAB/7/aggregation_explainability_final_0.png")
        self._touch("robust_aggregated/SKAB/7/aggregation_explainability_final_kendall_only_0.png")
        self._touch("robust_aggregated/SKAB/7/aggregation_explainability_robust_0.png")
        final, _ = self.plots._aggregation("SKAB", "7", "final")
        robust, _ = self.plots._aggregation("SKAB", "7", "robust")
        self.assertEqual(len(final), 2)
        self.assertEqual(len(robust), 1)

    def test_large_galleries_are_described_not_listed(self):
        self._touch("Thomposon/SKAB/7/expected_rewards_50.png")
        for i in range(120):
            self._touch(f"Thomposon/SKAB/7/shap_per_window_50/window_{i:03d}.png")
        descriptors = self.plots.gallery_descriptors("SKAB", "7")
        self.assertEqual(descriptors[0]["count"], 120)
        page = self.plots.gallery_page("SKAB", "7", "thompson/shap_per_window_50", 0, 60)
        self.assertEqual(len(page["items"]), 60)
        self.assertEqual(page["total"], 120)
        page2 = self.plots.gallery_page("SKAB", "7", "thompson/shap_per_window_50", 100, 60)
        self.assertEqual(len(page2["items"]), 20)

    def test_gallery_id_cannot_traverse(self):
        for bad in ("thompson/../../etc", "thompson/.hidden", "other/x", ""):
            self.assertEqual(
                self.plots.gallery_page("SKAB", "7", bad)["items"], [])

    def test_each_thompson_stage_owns_its_per_window_gallery(self):
        """Both stages write per-window frames into the one Thompson tree, so
        the gallery id carries the plot_group. result.js attaches a descriptor
        with id.startsWith(plot_group), which is why these two group names must
        not prefix one another."""
        self._touch("Thomposon/SKAB/7/expected_rewards_50.png")
        for i in range(30):
            self._touch(f"Thomposon/SKAB/7/shap_per_window_50/window_{i:03d}.png")
            self._touch(f"Thomposon/SKAB/7/ranking_per_window_50/window_{i:03d}.png")
        ids = [d["id"] for d in self.plots.gallery_descriptors("SKAB", "7")]
        self.assertIn("thompson/shap_per_window_50", ids)
        self.assertIn("ts_ranking/ranking_per_window_50", ids)
        for gid in ids:
            group = gid.split("/")[0]
            other = "ts_ranking" if group == "thompson" else "thompson"
            self.assertFalse(gid.startswith(other), gid)
        # The ranking group used to be rejected outright by a hardcoded gate.
        page = self.plots.gallery_page("SKAB", "7", "ts_ranking/ranking_per_window_50")
        self.assertEqual(page["total"], 30)

    def test_per_regime_captions_say_which_quantity_they_show(self):
        """Three per-regime sets cover the same window range and show three
        different things — a share of the expected reward, a departure from a
        typical window, and a cumulative ranking snapshot. "Windows 10–62"
        alone under-describes all of them."""
        self._touch("Thomposon/SKAB/7/expected_rewards_50.png")
        for stem in ("reward_per_regime", "shap_per_regime", "ranking_per_regime"):
            self._touch(f"Thomposon/SKAB/7/{stem}_50/regime_00_w0-4_NN_3.png")
        cap = {stem: self.plots.regime_plots("SKAB", "7", stem)[0]["caption"]
               for stem in ("reward_per_regime", "shap_per_regime", "ranking_per_regime")}
        self.assertIn("sum to that reward", cap["reward_per_regime"])
        self.assertIn("does not sum to it", cap["shap_per_regime"])
        self.assertIn("cumulative", cap["ranking_per_regime"])
        self.assertEqual(len(set(cap.values())), 3)

    def test_regime_variants_are_ordered_with_the_default_first(self):
        """The selection-dynamics card offers a toggle: the expected-reward
        contribution by default, SHAP's deviation view one click away. Order is
        the contract — the frontend shows variants[0]."""
        self._touch("Thomposon/SKAB/7/expected_rewards_50.png")
        self._touch("Thomposon/SKAB/7/reward_per_regime_50/regime_00_w0-4_NN_3.png")
        self._touch("Thomposon/SKAB/7/reward_per_regime_50/regime_01_w5-9_NN_1.png")
        self._touch("Thomposon/SKAB/7/shap_per_regime_50/regime_00_w0-4_NN_3.png")
        variants = self.plots.regime_plot_variants(
            "SKAB", "7", ["reward_per_regime", "shap_per_regime"])
        self.assertEqual(sorted(variants), [0, 1])
        self.assertEqual([f["title"] for f in variants[0]],
                         ["Expected-reward contribution", "Deviation from a typical window"])
        # A regime present in only one set keeps that set, rather than the
        # missing one shifting the others.
        self.assertEqual([f["title"] for f in variants[1]],
                         ["Expected-reward contribution"])

    def test_channel_plot_captions_state_the_selection_rule(self):
        """These figures plot a subset — 9 channels on SKAB, 38 on SMD — and the
        bars cannot say whether a missing channel was small or just not picked.

        Both mean|SHAP| figures carry the rule. The headline captions were
        deliberately shortened and no longer repeat it, so this asserts on the
        gallery pair by name rather than on a count that any caption edit would
        move.
        """
        self._touch("Thomposon/SKAB/7/expected_rewards_50.png")
        for name in ("shap_average_top3_50", "shap_average_all_50",
                     "reward_average_top3_50", "ranking_channels_50"):
            self._touch(f"Thomposon/SKAB/7/{name}.png")
        _headline, gallery = self.plots._thompson("SKAB", "7")
        by_name = {f["name"]: f.get("caption", "") for f in gallery}
        for name in ("shap_average_top3_50.png", "shap_average_all_50.png"):
            self.assertIn("not necessarily zero", by_name.get(name, ""), name)

    def test_missing_directories_yield_empty_manifest_not_an_error(self):
        m = self.plots.manifest("NOPE", "999")
        self.assertTrue(all(v["headline"] == [] for k, v in m.items() if not k.startswith("_")))


class TestSafeMediaPath(unittest.TestCase):

    def setUp(self):
        from WebUI import plots
        self.plots = plots
        self._tmp = tempfile.TemporaryDirectory()
        self.myresults = Path(self._tmp.name) / "myresults"
        (self.myresults / "GA_Ens").mkdir(parents=True)
        self.png = self.myresults / "GA_Ens" / "plot.png"
        self.png.write_bytes(b"\x89PNG")
        self.secret = Path(self._tmp.name) / "secret.png"
        self.secret.write_bytes(b"\x89PNG")
        (self.myresults / "model.pth").write_bytes(b"x")
        (self.myresults / "report.json").write_text("{}")
        self._saved = paths.MYRESULTS
        paths.MYRESULTS = self.myresults

    def tearDown(self):
        paths.MYRESULTS = self._saved
        self._tmp.cleanup()

    def test_serves_a_png_inside_the_tree(self):
        self.assertIsNotNone(self.plots.safe_media_path("GA_Ens/plot.png"))

    def test_bracketed_filename_is_servable(self):
        odd = self.myresults / "GA_Ens" / "ensemble_scores_SKAB_7_Data_vs_anomalies_['spikes'].png"
        odd.write_bytes(b"\x89PNG")
        self.assertIsNotNone(self.plots.safe_media_path(
            "GA_Ens/ensemble_scores_SKAB_7_Data_vs_anomalies_['spikes'].png"))

    def test_rejects_traversal_and_absolute_paths(self):
        for bad in ("../secret.png", "GA_Ens/../../secret.png", "/etc/passwd",
                    "\\etc\\passwd", "", "C:/Windows/x.png"):
            self.assertIsNone(self.plots.safe_media_path(bad), bad)

    def test_rejects_a_symlink_pointing_outside(self):
        """resolve() runs before the containment check, so a symlink out of the
        tree is caught — send_from_directory alone would follow it."""
        link = self.myresults / "GA_Ens" / "escape.png"
        try:
            link.symlink_to(self.secret)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        self.assertIsNone(self.plots.safe_media_path("GA_Ens/escape.png"))

    def test_rejects_non_image_extensions(self):
        self.assertIsNone(self.plots.safe_media_path("model.pth"))
        self.assertIsNone(self.plots.safe_media_path("report.json"))

    def test_rejects_directories_and_missing_files(self):
        self.assertIsNone(self.plots.safe_media_path("GA_Ens"))
        self.assertIsNone(self.plots.safe_media_path("GA_Ens/nope.png"))


# ── markers ──────────────────────────────────────────────────────────────────

class TestMarkers(unittest.TestCase):

    def setUp(self):
        from WebUI import markers
        self.m = markers

    def test_phase_lines(self):
        e = self.m.classify("INFO:app:📂 STAGE 1/7: Loading Training Data...")
        self.assertEqual((e["type"], e["number"]), ("phase", 1))
        e = self.m.classify("INFO:app:📝 STAGE 7/7: Writing Comprehensive Results...")
        self.assertEqual(e["number"], 7)

    def test_substage_running(self):
        e = self.m.classify("INFO:app:  📊 Sub-stage 6.2: Thompson Sampling - Online model selection...")
        self.assertEqual((e["type"], e["key"], e["status"]), ("stage", "thompson", "running"))

    def test_skipped_is_not_running(self):
        """The SKIPPED rule must win, or a skipped stage lights up as running
        and never clears."""
        e = self.m.classify("INFO:app:  ⏩ Sub-stage 6.3: GAN Robustness Testing SKIPPED (partial run)")
        self.assertEqual((e["key"], e["status"]), ("gan", "skipped"))

    def test_stage_result_lines(self):
        e = self.m.classify("INFO:app:  ✓ [GA] Best ensemble=['LOF_1'] | F1=0.9000 | Time=37.0s")
        self.assertEqual((e["key"], e["status"]), ("ga", "done"))
        self.assertIn("Best ensemble", e["text"])
        for tag, key in (("Thompson", "thompson"), ("GAN", "gan"),
                         ("Borderline", "offby"), ("MonteCarlo", "montecarlo")):
            e = self.m.classify(f"INFO:app:  ✓ [{tag}] top-5: x | Time=1.0s")
            self.assertEqual(e["key"], key, tag)

    def test_completion_markers_are_distinguished(self):
        full = self.m.classify("INFO:app:🎉 EXECUTION COMPLETE! Total Time: 244.5s")
        self.assertEqual((full["type"], full["partial"]), ("complete", False))
        part = self.m.classify("INFO:app:✅ Partial run complete (stages=montecarlo).")
        self.assertEqual((part["type"], part["partial"]), ("complete", True))

    def test_banner_fields_are_parsed(self):
        e = self.m.classify("INFO:app:🚀 STARTING RAMSeS EXECUTION: dataset=SKAB, entity=7, "
                            "parallel=False, stages=ga,thompson, detectors=3/11")
        self.assertEqual(e["type"], "run_started")
        self.assertEqual(e["fields"]["dataset"], "SKAB")
        self.assertEqual(e["fields"]["detectors"], "3/11")

    def test_warnings_and_fatal_signature(self):
        w = self.m.classify("WARNING:app:LLM narration skipped — server unreachable")
        self.assertEqual(w["code"], "llm_unreachable")
        w = self.m.classify("WARNING:app:⚠ Requested detectors with no trained model (skipped): NN_2")
        self.assertEqual(w["code"], "detectors_missing")
        f = self.m.classify("INFO:app:Traceback for Entity: 7 Dataset: SKAB")
        self.assertEqual(f["type"], "fatal_marker")

    def test_noise_is_ignored(self):
        for line in ("", "   ", "INFO:app:Evaluating fitness for ensemble: ['LOF_1']"):
            self.assertIsNone(self.m.classify(line))

    def test_emoji_loss_does_not_break_progress(self):
        """Matching is on the text after the emoji, so a mangled encoding still
        advances the rail."""
        e = self.m.classify("INFO:app: Sub-stage 6.1: Genetic Algorithm...")
        self.assertEqual(e["key"], "ga")


# ── build_argv ───────────────────────────────────────────────────────────────

class TestBuildArgv(unittest.TestCase):

    def setUp(self):
        from WebUI import jobs
        self.jobs = jobs
        self.base = {"dataset": "SKAB", "entity": "7"}

    def _argv(self, **kw):
        return self.jobs.build_argv({**self.base, **kw})

    def test_minimal_command(self):
        argv = self._argv()
        self.assertEqual(argv[1:3], ["-u", "app.py"])
        self.assertIn("--dataset", argv)
        self.assertEqual(argv[argv.index("--dataset") + 1], "SKAB")
        self.assertEqual(argv[argv.index("--parallel") + 1], "false")
        self.assertIn("--explain", argv)          # the UI defaults explain on

    def test_defaults_emit_no_stages_or_detectors_flag(self):
        argv = self._argv(stages=["ga", "thompson", "gan", "offby", "montecarlo"],
                          detectors=list(self.jobs.ALL_DETECTORS))
        self.assertNotIn("--stages", argv)
        self.assertNotIn("--detectors", argv)

    def test_subsets_are_emitted_in_canonical_order(self):
        argv = self._argv(stages=["montecarlo", "ga"],
                          detectors=["NN_1", "LOF_1", "NN_1"])
        self.assertEqual(argv[argv.index("--stages") + 1], "ga,montecarlo")
        self.assertEqual(argv[argv.index("--detectors") + 1], "LOF_1,NN_1")

    def test_equivalent_selections_produce_identical_argv(self):
        a = self._argv(detectors=["NN_1", "LOF_1"])
        b = self._argv(detectors=["LOF_1", "NN_1", "LOF_1"])
        self.assertEqual(a, b)

    def test_flag_shapes(self):
        argv = self._argv(parallel=True, explain=False, enable_online=True,
                          max_online_windows=50, iteration=3, strategy="fixed-best")
        self.assertEqual(argv[argv.index("--parallel") + 1], "true")
        self.assertNotIn("--explain", argv)       # bare flag, omitted when off
        self.assertIn("--enable_online", argv)
        self.assertEqual(argv[argv.index("--max_online_windows") + 1], "50")
        self.assertEqual(argv[argv.index("--iteration") + 1], "3")
        self.assertEqual(argv[argv.index("--strategy") + 1], "fixed-best")

    def test_overwrite_is_always_explicit(self):
        """config.yml ships with overwrite: True, so omitting the flag would
        silently retrain every detector on every run."""
        argv = self._argv()
        self.assertEqual(argv[argv.index("--overwrite") + 1], "false")
        argv = self._argv(overwrite=True)
        self.assertEqual(argv[argv.index("--overwrite") + 1], "true")

    def test_llm_overrides_are_omitted_when_unset(self):
        self.assertNotIn("--llm_model", self._argv())
        argv = self._argv(llm_model="qwen2.5:7b-instruct", llm_base_url="http://x/v1")
        self.assertEqual(argv[argv.index("--llm_model") + 1], "qwen2.5:7b-instruct")

    def test_missing_dataset_or_entity_is_rejected(self):
        for bad in ({"dataset": "SKAB"}, {"entity": "7"}, {"dataset": "", "entity": "7"}):
            with self.assertRaises(ValueError):
                self.jobs.build_argv(bad)

    def test_narrate_argv(self):
        argv = self.jobs.build_narrate_argv("SKAB", "7")
        self.assertIn("Explainability.narrate", argv)
        self.assertEqual(argv[argv.index("--iteration") + 1], "0")


# ── classify_outcome: the exit-0-is-a-lie matrix ─────────────────────────────

class TestClassifyOutcome(unittest.TestCase):

    def setUp(self):
        from WebUI import jobs
        self.jobs = jobs

    def _job(self, lines=(), exit_code=0, **kw):
        job = self.jobs.Job("t", ["x"], kw.pop("params", {}))
        for line in lines:
            job._record(line)
        job.exit_code = exit_code
        for key, value in kw.items():
            setattr(job, key, value)
        return job

    FULL = "INFO:app:🎉 EXECUTION COMPLETE! Total Time: 1.0s"
    PARTIAL = "INFO:app:✅ Partial run complete (stages=montecarlo)."

    def test_full_run_with_marker_succeeds(self):
        out = self.jobs.classify_outcome(self._job([self.FULL]), expect_partial=False)
        self.assertEqual(out["status"], "succeeded")

    def test_partial_run_with_partial_marker_succeeds(self):
        out = self.jobs.classify_outcome(self._job([self.PARTIAL]), expect_partial=True)
        self.assertEqual(out["status"], "succeeded")

    def test_exit_zero_without_a_marker_is_a_failure(self):
        """run_app swallows exceptions and still exits 0 — the headline risk."""
        out = self.jobs.classify_outcome(self._job(["INFO:app:working..."]),
                                         expect_partial=False)
        self.assertEqual(out["status"], "failed")
        self.assertIn("without reaching completion", out["reason"])

    def test_traceback_signature_is_reported_as_the_reason(self):
        job = self._job(["INFO:app:Traceback for Entity: 7 Dataset: SKAB"])
        out = self.jobs.classify_outcome(job, expect_partial=False)
        self.assertEqual(out["status"], "failed")
        self.assertIn("raised an exception", out["reason"])

    def test_partial_marker_on_a_full_run_is_a_mismatch(self):
        out = self.jobs.classify_outcome(self._job([self.PARTIAL]), expect_partial=False)
        self.assertEqual(out["status"], "failed")
        self.assertIn("does not match", out["reason"])

    def test_non_zero_exit_is_a_failure(self):
        out = self.jobs.classify_outcome(
            self._job(["INFO:app:boom"], exit_code=2), expect_partial=False)
        self.assertEqual(out["status"], "failed")
        self.assertIn("code 2", out["reason"])

    def test_cancel_and_timeout_win_over_everything(self):
        job = self._job([self.FULL], cancel_requested=True)
        self.assertEqual(self.jobs.classify_outcome(job, False)["status"], "cancelled")
        job = self._job([self.FULL], status="timeout")
        self.assertEqual(self.jobs.classify_outcome(job, False)["status"], "timeout")

    def test_explain_requested_but_nothing_written(self):
        out = self.jobs.classify_outcome(self._job([self.FULL]), expect_partial=False,
                                         explain_requested=True, artifacts_written=False)
        self.assertEqual(out["status"], "succeeded_with_warnings")
        self.assertIn("no explanation artifacts", out["reason"])

    def test_error_line_is_preferred_over_the_generic_reason(self):
        job = self._job(["INFO:app:starting", "ERROR:app:Could not load models"])
        out = self.jobs.classify_outcome(job, expect_partial=False)
        self.assertIn("Could not load models", out["reason"])


# ── job lifecycle against a fake pipeline ────────────────────────────────────

_FAKE_PIPELINE = r"""
import sys, time
def log(msg): print(msg, file=sys.stderr, flush=True)
log("INFO:app:STARTING RAMSeS EXECUTION: dataset=SKAB, entity=7, detectors=11/11")
log("INFO:app:STAGE 1/7: Loading Training Data...")
log("INFO:app:  Sub-stage 6.1: Genetic Algorithm...")
log("INFO:app:  ✓ [GA] Best ensemble=['LOF_1'] | Time=1.0s")
log("INFO:app:  ⏩ Sub-stage 6.3: GAN Robustness Testing SKIPPED (partial run)")
sys.stdout.write("progress 1/3\rprogress 2/3\rprogress 3/3\n"); sys.stdout.flush()
log("MARKER")
"""


class TestJobLifecycle(unittest.TestCase):

    def setUp(self):
        from WebUI import jobs
        self.jobs = jobs
        self._tmp = tempfile.TemporaryDirectory()
        # COMPREHENSIVE too: the report decides whether a finished job links to
        # it, and a test must never read the developer's real myresults/ tree.
        self._saved = (paths.WEBUI_LOGS, paths.COMPREHENSIVE)
        paths.WEBUI_LOGS = Path(self._tmp.name) / "webui_logs"
        paths.COMPREHENSIVE = Path(self._tmp.name) / "myresults" / "comprehensive"
        self.mgr = jobs.JobManager(repo_root=Path(self._tmp.name))

    def tearDown(self):
        (paths.WEBUI_LOGS, paths.COMPREHENSIVE) = self._saved
        self._tmp.cleanup()

    def _run(self, script, params=None, timeout=30):
        job = self.mgr.start(params or {"dataset": "SKAB", "entity": "7", "explain": False},
                             argv=[sys.executable, "-u", "-c", script], timeout=timeout)
        for _ in range(300):
            if job.is_done():
                break
            time.sleep(0.02)
        return job

    def test_markers_drive_the_stage_rail(self):
        job = self._run(_FAKE_PIPELINE.replace("MARKER", "\U0001F389 EXECUTION COMPLETE! 1.0s"))
        self.assertEqual(job.status, "succeeded")
        self.assertEqual(job.stages["ga"]["status"], "done")
        self.assertEqual(job.stages["gan"]["status"], "skipped")
        self.assertEqual(job.phase["number"], 1)
        self.assertEqual(job.exit_code, 0)

    def test_report_url_is_offered_only_when_the_report_exists(self):
        """A partial run returns before the pipeline writes the report, so the
        link must follow the file on disk, not the run's success."""
        done = _FAKE_PIPELINE.replace("MARKER", "\U0001F389 EXECUTION COMPLETE! 1.0s")
        job = self._run(done)
        self.assertEqual(job.result_url, "/result/SKAB/7")
        self.assertIsNone(job.report_url)

        _write(paths.COMPREHENSIVE / "SKAB" / "7" / "comprehensive_results_SKAB_7_iter5.txt",
               REPORT_TEXT)
        job = self._run(done)
        self.assertEqual(job.report_url, "/report/SKAB/7")
        self.assertEqual(job.snapshot()["report_url"], "/report/SKAB/7")

    def test_the_rail_carries_no_stage_timings(self):
        """Wall-clock deltas between log lines disagreed with the timings the
        pipeline measures and writes to the comprehensive report; that report
        is the binding record, so the rail reports status only."""
        job = self._run(_FAKE_PIPELINE.replace("MARKER", "\U0001F389 EXECUTION COMPLETE! 1.0s"))
        for entry in job.stages.values():
            self.assertNotIn("started_at", entry)
            self.assertNotIn("finished_at", entry)

    def test_carriage_returns_are_split_into_separate_lines(self):
        """tqdm rewrites its bar with \\r; without splitting on it one progress
        bar becomes a single enormous line."""
        job = self._run(_FAKE_PIPELINE.replace("MARKER", "\U0001F389 EXECUTION COMPLETE! 1.0s"))
        text = "\n".join(job.lines)
        self.assertIn("progress 1/3", text)
        self.assertIn("progress 3/3", text)
        self.assertFalse(any(len(l) > 500 for l in job.lines))

    def test_log_is_also_written_to_disk(self):
        job = self._run(_FAKE_PIPELINE.replace("MARKER", "\U0001F389 EXECUTION COMPLETE! 1.0s"))
        self.assertTrue(job.log_path.exists())
        self.assertIn("Sub-stage 6.1", job.log_path.read_text())

    def test_exit_zero_without_completion_marker_fails(self):
        job = self._run(_FAKE_PIPELINE.replace('log("MARKER")', 'pass'))
        self.assertEqual(job.status, "failed")

    def test_tail_supports_incremental_reads(self):
        job = self._run(_FAKE_PIPELINE.replace("MARKER", "\U0001F389 EXECUTION COMPLETE! 1.0s"))
        first = job.tail(0)
        self.assertGreater(len(first["lines"]), 3)
        self.assertEqual(job.tail(first["cursor"])["lines"], [])

    def test_one_job_at_a_time(self):
        script = "import time; time.sleep(5)"
        job = self.mgr.start({"dataset": "SKAB", "entity": "7"},
                             argv=[sys.executable, "-u", "-c", script], timeout=30)
        try:
            with self.assertRaises(RuntimeError) as cm:
                self.mgr.start({"dataset": "SKAB", "entity": "8"},
                               argv=[sys.executable, "-u", "-c", script])
            self.assertEqual(str(cm.exception), job.id)   # 409 carries the active id
        finally:
            self.mgr.cancel(job.id)

    def test_cancel_kills_the_whole_process_group(self):
        """start_new_session puts the child in its own group so a cancel takes
        its worker threads/children with it."""
        script = ("import subprocess, sys, time\n"
                  "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                  "time.sleep(30)\n")
        job = self.mgr.start({"dataset": "SKAB", "entity": "7"},
                             argv=[sys.executable, "-u", "-c", script], timeout=30)
        time.sleep(0.6)
        self.assertTrue(self.mgr.cancel(job.id))
        for _ in range(200):
            if job.is_done():
                break
            time.sleep(0.02)
        self.assertEqual(job.status, "cancelled")

    def test_timeout_is_enforced(self):
        job = self._run("import time; time.sleep(30)", timeout=1)
        self.assertIn(job.status, ("timeout", "failed"))

    def test_unstartable_command_fails_cleanly(self):
        job = self.mgr.start({"dataset": "SKAB", "entity": "7"},
                             argv=["/nonexistent/binary"])
        self.assertEqual(job.status, "failed")
        self.assertIsNone(self.mgr.active())     # the slot is released


# ── routes ───────────────────────────────────────────────────────────────────

@unittest.skipUnless(HAS_FLASK, "Flask is not installed in this interpreter")
class TestRoutes(ArtifactTreeCase):
    """Exercised through Flask's test client so no server or port is needed."""

    def setUp(self):
        super().setUp()
        from WebUI import jobs, server
        jobs.reset_manager()
        self.jobs = jobs
        self.client = server.create_app(TESTING=True).test_client()

    def tearDown(self):
        self.jobs.reset_manager()
        super().tearDown()

    def test_css_and_js_are_cache_busted_by_mtime(self):
        """The server runs with use_reloader=False, so a frontend edit needs a
        restart — and without a stamp the browser then keeps the old file, which
        presents as the new feature simply not being there."""
        html = self.client.get("/result/SKAB/7").get_data(as_text=True)
        self.assertRegex(html, r'href="/static/css/ramses\.css\?v=\d+"')
        self.assertRegex(html, r'src="/static/js/result\.js\?v=\d+"')

    def test_dom_js_is_loaded_exactly_once_per_page(self):
        """Every entry script imports "./dom.js". A second, stamped <script> tag
        for it would be a different URL, so the module would evaluate twice and
        bind the theme toggle twice — one click, two theme changes."""
        for path in ("/", "/result/SKAB/7", "/report/SKAB/7"):
            html = self.client.get(path).get_data(as_text=True)
            self.assertEqual(html.count("js/dom.js"), 0, path)

    def test_pages_render(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/result/SKAB/7").status_code, 200)
        self.assertEqual(self.client.get("/report/SKAB/7").status_code, 200)
        self.assertEqual(self.client.get("/run/nope").status_code, 404)

    def test_comprehensive_report_is_served_and_downloadable(self):
        r = self.client.get("/api/comprehensive/SKAB/7")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["text"], REPORT_TEXT)
        self.assertEqual(body["iteration"], 5)

        download = self.client.get("/api/comprehensive/SKAB/7?download=1")
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.data.decode(), REPORT_TEXT)
        self.assertIn("attachment", download.headers["Content-Disposition"])

    def test_comprehensive_report_404_explains_partial_runs(self):
        r = self.client.get("/api/comprehensive/SKAB/999")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.get_json()["error"], "no_report")
        self.assertIn("partial", r.get_json()["hint"])
        self.assertEqual(
            self.client.get("/api/comprehensive/SKAB/999?download=1").status_code, 404)

    def test_catalog_lists_existing_results(self):
        r = self.client.get("/api/catalog")
        self.assertEqual(r.status_code, 200)
        results = r.get_json()["results"]
        self.assertEqual([(x["dataset"], x["entity"]) for x in results], [("SKAB", "7")])

    def test_explanations_payload_and_404(self):
        r = self.client.get("/api/explanations/SKAB/7")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn("stages", body)
        self.assertIn("plots", body)
        missing = self.client.get("/api/explanations/SKAB/999")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.get_json()["error"], "no_artifacts")

    def test_download_serves_the_verbatim_file(self):
        r = self.client.get("/api/explanations/SKAB/7/download?stage=thompson_sampling")
        self.assertEqual(r.status_code, 200)
        # The download is byte-identical to disk, glossary included.
        self.assertTrue(r.data.decode().startswith("INFO: "))
        self.assertEqual(self.client.get(
            "/api/explanations/SKAB/7/download?stage=nope").status_code, 404)

    def test_dry_run_returns_argv_without_starting_anything(self):
        r = self.client.post("/api/runs", json={"dataset": "SKAB", "entity": "7",
                                                "stages": ["montecarlo"], "dry_run": True})
        self.assertEqual(r.status_code, 200)
        self.assertIn("--stages", r.get_json()["argv"])
        self.assertIsNone(self.jobs.manager().active())

    def test_run_validation_errors(self):
        for body, fragment in (
            ({"entity": "7"}, "dataset"),
            ({"dataset": "S"}, "entity"),
            ({"dataset": "S", "entity": "7", "stages": ["nope"]}, "unknown stage"),
            ({"dataset": "S", "entity": "7", "detectors": ["LOF_1"]}, "at least two"),
        ):
            r = self.client.post("/api/runs", json=body)
            self.assertEqual(r.status_code, 400, body)
            self.assertIn(fragment, r.get_json()["error"])

    def test_second_concurrent_run_is_refused_with_the_active_id(self):
        script = "import time; time.sleep(5)"
        job = self.jobs.manager().start({"dataset": "SKAB", "entity": "7"},
                                        argv=[sys.executable, "-c", script])
        try:
            r = self.client.post("/api/runs", json={"dataset": "SKAB", "entity": "8"})
            self.assertEqual(r.status_code, 409)
            self.assertEqual(r.get_json()["active_job_id"], job.id)
        finally:
            self.jobs.manager().cancel(job.id)

    def test_sse_route_content_type(self):
        job = self.jobs.manager().start({"dataset": "SKAB", "entity": "7"},
                                        argv=[sys.executable, "-c", "pass"])
        for _ in range(200):
            if job.is_done():
                break
            time.sleep(0.02)
        r = self.client.get(f"/api/runs/{job.id}/events")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["Content-Type"].startswith("text/event-stream"))
        body = r.get_data(as_text=True)
        self.assertIn("event: hello", body)
        self.assertIn("event: status", body)     # already finished -> closes

    def test_media_route_enforces_the_path_rules(self):
        png = self.myresults / "GA_Ens" / "x.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"\x89PNG\r\n\x1a\n")
        self.assertEqual(self.client.get("/media/GA_Ens/x.png").status_code, 200)
        for bad in ("../../app.py", "GA_Ens/../../../app.py",
                    "explanations_ir/SKAB/7/ir_thompson.json"):
            self.assertEqual(self.client.get(f"/media/{bad}").status_code, 404, bad)

    def test_log_endpoint_is_incremental(self):
        job = self.jobs.manager().start(
            {"dataset": "SKAB", "entity": "7"},
            argv=[sys.executable, "-c", "import sys; print('hello', file=sys.stderr)"])
        for _ in range(200):
            if job.is_done():
                break
            time.sleep(0.02)
        first = self.client.get(f"/api/runs/{job.id}/log").get_json()
        self.assertTrue(any("hello" in l for l in first["lines"]))
        again = self.client.get(
            f"/api/runs/{job.id}/log?offset={first['cursor']}").get_json()
        self.assertEqual(again["lines"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestOnDemandRankingGap(unittest.TestCase):
    """The gap between an ARBITRARY detector pair, drawn per request.

    11 detectors is 55 unordered pairs per entity and a reader looks at one or
    two, so these are not pre-rendered. What makes that safe is that the split
    is exactly shares(a) - shares(b): the on-demand figure and the pipeline's
    own winner-vs-runner-up PNG are the same quantity, not a re-derivation.
    """

    def setUp(self):
        from WebUI import ondemand
        self.ondemand = ondemand
        self._tmp = tempfile.TemporaryDirectory()
        self.myresults = Path(self._tmp.name) / "myresults"
        self._saved = paths.MYRESULTS
        paths.MYRESULTS = self.myresults

    def tearDown(self):
        paths.MYRESULTS = self._saved
        self._tmp.cleanup()

    def _write_ir(self, shares):
        d = self.myresults / "explanations_ir" / "SKAB" / "7"
        d.mkdir(parents=True, exist_ok=True)
        doc = {"ir_version": "1.0", "stage": "thompson_ranking", "dataset": "SKAB",
               "entity": "7", "output": {}, "evidence": [], "caveats": [],
               "required_atom_ids": [], "confidence": {}}
        if shares is not None:
            doc["channel_shares"] = shares
        (d / "ir_thompson_ranking.json").write_text(json.dumps(doc))

    def test_renders_any_pair_from_the_persisted_shares(self):
        self._write_ir({"NN_1": [0.4, 0.1, 0.05], "NN_2": [0.1, 0.3, 0.05],
                        "LOF_1": [0.0, 0.0, 0.2]})
        png = self.ondemand.render_ranking_gap("SKAB", "7", "NN_1", "LOF_1")
        self.assertTrue(png.startswith(b"\x89PNG"), "not a PNG")
        # Order matters: the reversed pair is a different figure, not an error.
        self.assertIsNotNone(self.ondemand.render_ranking_gap("SKAB", "7", "LOF_1", "NN_1"))

    def test_unavailable_pairs_return_none_rather_than_raising(self):
        """Every 'cannot draw this' case answers None so the route can 404 and
        the page keeps its default pair, instead of surfacing a traceback."""
        self._write_ir({"NN_1": [0.4], "NN_2": [0.1]})
        self.assertIsNone(self.ondemand.render_ranking_gap("SKAB", "7", "NN_1", "GHOST"))
        # A detector against itself is an all-zero gap, which is not a finding.
        self.assertIsNone(self.ondemand.render_ranking_gap("SKAB", "7", "NN_1", "NN_1"))
        self.assertIsNone(self.ondemand.render_ranking_gap("NOPE", "999", "NN_1", "NN_2"))

    def test_an_ir_without_shares_offers_nothing(self):
        """Result trees written before the block existed must degrade to the
        pipeline's static figure, not to a broken control."""
        self._write_ir(None)
        self.assertEqual(self.ondemand.ranking_channel_shares("SKAB", "7"), {})
        from WebUI import plots
        self.assertIsNone(plots._ranking_pair_picker("SKAB", "7"))

    def test_the_picker_orders_detectors_so_the_default_pair_is_the_top_two(self):
        """The initial view must reproduce the static winner-vs-runner-up
        figure, or adding this control would silently change what the card
        shows on load."""
        self._write_ir({"MID": [0.2, 0.2], "TOP": [0.9, 0.1], "LOW": [0.01, 0.0]})
        from WebUI import plots
        spec = plots._ranking_pair_picker("SKAB", "7")
        self.assertEqual(spec["pair_picker"]["detectors"][:2], ["TOP", "MID"])
        self.assertIn("/api/plots/SKAB/7/ranking-gap", spec["pair_picker"]["endpoint"])


class TestOffByTreeSelector(unittest.TestCase):
    """One surrogate tree at a time, chosen by competitor.

    Every pair is already on disk, so this is a selector over files — but ten
    near-identical trees stacked down the card is ten figures to scroll past.
    """

    def setUp(self):
        from WebUI import plots
        self.plots = plots
        self._tmp = tempfile.TemporaryDirectory()
        self.myresults = Path(self._tmp.name) / "myresults"
        self._saved = paths.MYRESULTS
        paths.MYRESULTS = self.myresults

    def tearDown(self):
        paths.MYRESULTS = self._saved
        self._tmp.cleanup()

    def _touch(self, rel):
        p = self.myresults / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x89PNG")

    def test_trees_collapse_into_one_selector_named_by_competitor(self):
        base = "robustness/off_by/SKAB/7"
        self._touch(f"{base}/SKAB_7_off_by_point_importance.png")
        for competitor in ("NN_1", "NN_3", "CBLOF_2"):
            self._touch(f"{base}/SKAB_7_off_by_point_tree_LOF_1_vs_{competitor}.png")
        headline, _ = self.plots._off_by("SKAB", "7")
        picker = next(f for f in headline if "variants" in f)
        self.assertEqual(picker["title"], "Where LOF_1 uniquely wins")
        # Titled by the competitor alone — the winner is in the group title, so
        # repeating it in every option makes the choice harder to scan.
        self.assertEqual(sorted(v["title"] for v in picker["variants"]),
                         ["CBLOF_2", "NN_1", "NN_3"])
        # Past four options the frontend switches to a <select>; the named axis
        # of choice says so explicitly rather than relying on the count.
        self.assertEqual(picker["select_label"], "Compared against")
        self.assertEqual(picker["default"], 0)
        # The importance figure is a separate finding and keeps its own slot.
        self.assertEqual(sum(1 for f in headline if "variants" not in f), 1)
