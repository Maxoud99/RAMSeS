"""
Tests for the WebUI backend.

Follows the repo's test style (unittest, no network). Every test builds a
synthetic artifact tree in a temporary directory and repoints WebUI.paths at
it, so nothing depends on the contents of the real results/ and no test ever
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



def _strip_js_comments(src: str) -> str:
    """`src` with /* */ and // comments removed.

    The assertions below ban DOM constructs, not vocabulary. The prose in
    configure.js explains at length why the detector pool stopped using
    checkboxes, so a bare substring check on the source would fail on the
    explanation for the very rule it is enforcing.
    """
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


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
        self.results_root = root / "results"
        self.ir_dir = self.results_root / "explanations_ir" / self.DATASET / self.ENTITY
        self.nl_dir = self.results_root / "explanations_nl" / self.DATASET / self.ENTITY
        self.report_dir = self.results_root / "comprehensive" / self.DATASET / self.ENTITY
        self._saved = (paths.RESULTS, paths.EXPLANATIONS_IR, paths.EXPLANATIONS_NL,
                       paths.COMPREHENSIVE)
        paths.RESULTS = self.results_root
        paths.EXPLANATIONS_IR = self.results_root / "explanations_ir"
        paths.EXPLANATIONS_NL = self.results_root / "explanations_nl"
        paths.COMPREHENSIVE = self.results_root / "comprehensive"
        artifacts.paths = paths
        self.build()

    def tearDown(self):
        (paths.RESULTS, paths.EXPLANATIONS_IR, paths.EXPLANATIONS_NL,
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
                 "text": "Ranked by the size of its mean vector, NN_1 scored "
                         "0.559322, ahead of NN_2 by 0.000410."},
                {"id": "tsr.winner.channels", "type": "winner_channels",
                 "subject": "NN_1",
                 "value": {"channel": 3, "total": 0.559322,
                           "per_channel": [[3, 0.229], [0, 0.101]]},
                 "text": "NN_1's score is built mostly from context feature 3 (41.0%)."},
                {"id": "tsr.gap.runner_up", "type": "rank_gap", "subject": "NN_1",
                 "value": {"rivals": ["NN_2"], "runner_up": "NN_2",
                           "per_channel": [[3, 0.091], [0, -0.058]]},
                 "text": "NN_1's lead over NN_2 came mostly from context feature 3."},
                {"id": "tsr.regime.0", "type": "regime", "subject": "NN_2",
                 "value": {"start": 10, "end": 71, "duration": 62, "leader": "NN_2"},
                 "text": "Regime 0 (windows 10 to 71, 62 windows) was led by NN_2."},
            ]))
        _write(self.nl_dir / "nl_thompson_ranking.txt",
               _nl("The ranking score is the sum of every squared weight.",
                   "Ranked by the size of its mean vector, NN_1 scored "
                   "0.559322, ahead of NN_2 by 0.000410. NN_1's score is built "
                   "mostly from context feature 3 (41.0%). Regime 0 (windows 10 to 71, "
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
                                  "borda_rank": 1},
                        "text": "GAN_F1 shaped the consensus most."},
                       {"id": "ra_robust.source.GAN_PR_AUC.role", "type": "source_role",
                        "subject": "GAN_PR_AUC",
                        "value": {"influence_rank": 5, "agreement_rank": 2,
                                  "borda_rank": 2},
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
                               "note": "no explanation was written for this stage — the run "
                                       "either skipped it or did not use --explain"}},
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

    def test_every_stage_points_at_its_documentation_section(self):
        """The glossaries left the cards for a page of their own, so each card
        carries the section holding its own — and the two halves of one pipeline
        stage share it. A reader looking up ALE should not have to know whether
        it came from the selection card or the weighting one."""
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        by = {s["key"]: s["doc_section"] for s in p["stages"]}
        self.assertEqual(by["thompson_ranking"], "lints")
        self.assertEqual(by["thompson_sampling"], "lints")
        self.assertEqual(by["monte_carlo"], "monte-carlo")
        # Every stage the page can render has somewhere to send the reader.
        self.assertTrue(all(s["doc_section"] for s in p["stages"]), by)

    def test_every_stage_carries_its_one_line_definitions(self):
        """The strip that fills the space the glossary left. Ordered pairs, in
        the order the card's prose uses them, so a reader meets a term where it
        first matters rather than alphabetically."""
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        by = {s["key"]: s["terms"] for s in p["stages"]}
        self.assertEqual([t for t, _d in by["thompson_ranking"]],
                         ["Score", "Share", "Contribution", "Margin", "Regime"])
        self.assertEqual([t for t, _d in by["monte_carlo"]], ["Noise level"])
        self.assertTrue(all(t and d for stage in by.values() for t, d in stage), by)

    def test_both_consensus_cards_define_the_same_vocabulary(self):
        """They are one stage split across two cards, so the terms are the
        stage's. A two-source run reports no influence, and the card's own prose
        is what says so — the definition being present is not a claim that it
        applied."""
        self.assertEqual(artifacts.STAGE_TERMS["rank_aggregation_final"],
                         artifacts.STAGE_TERMS["rank_aggregation_robust"])

    def test_explained_stages_all_have_definitions(self):
        """A stage added later without a strip would leave the gap the glossary
        used to fill simply empty."""
        missing = [s["key"] for s in artifacts.STAGES
                   if not artifacts.STAGE_TERMS.get(s["key"])]
        self.assertEqual(missing, [])

    def test_documentation_is_the_whole_pipeline_in_order(self):
        """The text describes RAMSeS, not a run, so every section is present
        whichever stages this entity happened to execute — including the two
        that no card points at, the overview and the GAN test."""
        p = artifacts.documentation(self.DATASET, self.ENTITY)
        self.assertEqual([s["id"] for s in p["sections"]],
                         ["overview", "ga", "lints", "gan", "off-by",
                          "monte-carlo", "rank-aggregation"])

    def test_documentation_blocks_are_renderable(self):
        """docs.js knows four block shapes and nothing else; a fifth would be
        silently dropped from the page."""
        p = artifacts.documentation(self.DATASET, self.ENTITY)
        for section in p["sections"]:
            self.assertTrue(section["blocks"], section["id"])
            groups = [(section["id"], section["blocks"])] + [
                (sub["id"], sub["blocks"]) for sub in section["subsections"]]
            for where, blocks in groups:
                self.assertTrue(blocks, where)
                for block in blocks:
                    self.assertTrue(
                        ("list" in block) or ("formula" in block) or ("text" in block),
                        (where, block))
                    if "lead" in block:
                        self.assertIn("text", block, where)

    def test_documentation_subsection_ids_are_unique_and_addressable(self):
        """Every subsection is a fragment the sidebar links to and the page
        scrolls to, so a duplicate id would silently send two entries to one
        place."""
        p = artifacts.documentation(self.DATASET, self.ENTITY)
        ids = [s["id"] for s in p["sections"]]
        ids += [sub["id"] for s in p["sections"] for sub in s["subsections"]]
        self.assertEqual(len(ids), len(set(ids)), sorted(ids))
        self.assertTrue(all(i and " " not in i and "#" not in i for i in ids), ids)

    def test_documentation_opens_with_the_architecture(self):
        """The overview precedes the stages: a reader who has never seen the
        framework needs the two branches before any one of them."""
        p = artifacts.documentation(self.DATASET, self.ENTITY)
        first = p["sections"][0]
        self.assertEqual(first["id"], "overview")
        self.assertIn("overview-branches", [s["id"] for s in first["subsections"]])

    def test_every_stage_section_explains_its_explainability(self):
        """Each stage documents how RAMSeS works and then how that is explained.

        Keyed on the id, not the title: the titles describe their own content
        rather than announcing which half of the section they belong to, so the
        `-explained` suffix is the only durable marker of that boundary. Every
        section carries one — GAN was the last exception and gained its layer.
        """
        p = artifacts.documentation(self.DATASET, self.ENTITY)
        for section in p["sections"]:
            ids = [s["id"] for s in section["subsections"]]
            self.assertTrue(any(i.endswith("-explained") for i in ids),
                            (section["id"], ids))

    def test_documentation_sections_cover_every_stage_that_can_carry_one(self):
        """A stage missing from DOC_SECTIONS would render a card whose button
        led nowhere, which is worse than having no button."""
        covered = set(artifacts.DOC_SECTION_BY_STAGE)
        self.assertEqual(covered, {s["key"] for s in artifacts.STAGES})

    def test_assembles_stages_in_pipeline_order(self):
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        # The ranking criterion precedes the selection dynamics: it explains the
        # ordering the rest of the pipeline consumes.
        self.assertEqual([s["key"] for s in p["stages"]],
                         ["thompson_ranking", "thompson_sampling", "monte_carlo",
                          "rank_aggregation_robust"])
        # The robustness block reads Monte Carlo, off-by, GAN, so Monte Carlo
        # holds order 5 and the two aggregation stages close the page.
        self.assertEqual([s["order"] for s in p["stages"]], [3, 4, 5, 8])

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
        self.assertIn("Ranked by the size of its mean vector", s["full"])
        self.assertGreater(s["words"], len(s["full"].split()))
        # And the regimes still carry their own narrated sentences.
        self.assertTrue(any("Regime 0" in (r.get("narrated") or r["text"])
                            for r in s["regimes"]))

    def test_case_insensitive_dataset_and_entity(self):
        # The pipeline writes the dataset string verbatim, so `skab` and `SKAB`
        # both occur; on Linux they are different directories.
        self.assertIsNotNone(artifacts.build_payload("skab", "7"))
        self.assertIsNotNone(artifacts.build_payload("SKAB", "7"))

    def test_legacy_glossary_is_stripped_from_the_narrative(self):
        """Footers are no longer written — the documentation page carries the
        glossary — but files from earlier runs still lead with the marker, and
        the reader must not render one as prose."""
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        ts = next(s for s in p["stages"] if s["key"] == "thompson_sampling")
        self.assertEqual(ts["full"], "Thompson Sampling ranked NN_1 first.")
        self.assertNotIn("INFO:", ts["full"])
        self.assertNotIn("Expected reward is the weights", ts["full"])
        self.assertNotIn("info", ts)

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
        """A stage the run never explained is listed WITH a reason.

        GAN used to be permanently in this list; now it is a full stage and
        lands here only when its IR file is absent, like any other. The title
        still comes from STAGES, so the entry names the stage rather than its
        key.
        """
        p = artifacts.build_payload(self.DATASET, self.ENTITY)
        gan = next(m for m in p["missing_stages"] if m["key"] == "gan")
        self.assertEqual(gan["status"], "not_available")
        self.assertEqual(gan["title"], "Robustness: GAN")
        self.assertIn("--explain", gan["note"])

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

    def test_monte_carlo_drops_the_per_noise_walk(self):
        """Dropped by type, not by position: the narrator merges facts, so a
        count of atoms cuts in the wrong place."""
        ir_doc = _stage_ir("monte_carlo", [
            {"id": "t", "type": "stage_output", "subject": "LOF_1", "value": {},
             "text": "In the production test, LOF_1 ranked first by F1."},
            {"id": "v1", "type": "sweep_verdict", "subject": "LOF_1", "value": {},
             "text": "Measured by F1, the sweep and the production run agree."},
            {"id": "v2", "type": "sweep_verdict", "subject": "NN_3", "value": {},
             "text": "Measured by PR-AUC, the sweep and the production run do "
                     "not agree."},
            {"id": "r", "type": "surrogate_win_rates", "subject": "rates",
             "value": {}, "text": "Across the sweep LOF_1 led with 39.0%."},
            {"id": "w", "type": "win_region", "subject": "NN_3", "value": {},
             "text": "NN_3 won by F1 at noise levels 0.042 and 0.158."},
        ])
        narrative = ("In the production test, LOF_1 ranked first by F1. "
                     "Measured by F1, the sweep and the production run agree. "
                     "Measured by PR-AUC, the sweep and the production run do "
                     "not agree. Across the sweep LOF_1 led with 39.0%. NN_3 "
                     "won by F1 at noise levels 0.042 and 0.158.")
        out = summarize.summarize(narrative, stage="monte_carlo", ir_doc=ir_doc)
        self.assertEqual(out["mode"], "drop")
        self.assertFalse(out["is_full"])
        self.assertIn("39.0%", out["summary"])
        self.assertIn("PR-AUC", out["summary"])
        self.assertNotIn("0.042", out["summary"])

    def test_off_by_and_gan_keep_only_their_opening_sentences(self):
        """These two stages summarise by POSITION, not by atom type.

        Their narrative is one sentence per rival, all `exclusive_wins`, so
        there is no type to drop that would not take every rival with it. The
        card opens with the winner and its three closest competitors; the weaker
        rivals, the per-rival importances and the roll-up are what the click
        buys. The IR orders rivals hardest-first, so the four kept are the four
        that matter.
        """
        for stage in ("off_by_threshold", "gan"):
            with self.subTest(stage=stage):
                ir_doc = _stage_ir(stage, [
                    {"id": "w", "type": "stage_output", "subject": "LOF_1",
                     "value": {}, "text": "LOF_1 was the highest-ranked model."},
                ])
                narrative = ("LOF_1 was the highest-ranked model. It beat A on 1 point. "
                             "It beat B on 2 points. It beat C on 3 points. "
                             "It beat D on 9 points. "
                             "Across all competitors the edge is position (0.87).")
                out = summarize.summarize(narrative, stage=stage, ir_doc=ir_doc)
                self.assertEqual(out["mode"], "lead")
                self.assertFalse(out["is_full"])
                # Exactly the first four sentences.
                self.assertIn("highest-ranked", out["summary"])
                self.assertIn("beat C on 3 points", out["summary"])
                self.assertNotIn("beat D on 9 points", out["summary"])
                self.assertNotIn("0.87", out["summary"])
                # The full text is still available behind the disclosure.
                self.assertIn("0.87", out["body"])

    def test_a_short_narrative_is_not_cut(self):
        """Under the limit the summary IS the narrative, so the card renders it
        open and labelled 'Full text' rather than offering an empty expand."""
        ir_doc = _stage_ir("gan", [
            {"id": "w", "type": "stage_output", "subject": "LOF_1",
             "value": {}, "text": "LOF_1 was the highest-ranked model."},
        ])
        narrative = "LOF_1 was the highest-ranked model. It beat A on 1 point."
        out = summarize.summarize(narrative, stage="gan", ir_doc=ir_doc)
        self.assertTrue(out["is_full"])
        self.assertEqual(out["summary"], out["body"])

    def test_an_unattributable_sentence_is_kept(self):
        """Dropping happens only on positive evidence: a sentence that matches
        no atom is shown rather than silently lost."""
        ir_doc = _stage_ir("ga_selection", [
            {"id": "x", "type": "excluded_group", "subject": "plain",
             "value": {"detectors": ["CBLOF_3"]},
             "text": "CBLOF_3 had low utility and low stability, and was left out."},
        ])
        out = summarize.summarize(
            "Something the verifier never saw. CBLOF_3 was left out for low "
            "utility.",
            stage="ga_selection", ir_doc=ir_doc)
        self.assertIn("Something the verifier never saw.", out["summary"])
        self.assertNotIn("CBLOF_3", out["summary"])


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
                                 "borda_rank": 1},
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
             "text": "Across the regimes NN_1 led, context feature 7 contributed most."},
        ]
        for i, (lead, a, b) in enumerate(((("NN_3"), 0, 4), ("NN_1", 5, 18))):
            atoms.append({"id": f"ts.regime.{i}", "type": "regime", "subject": lead,
                          "value": {"start": a, "end": b, "leader": lead},
                          "text": f"Regime {i} (windows {a} to {b}) was led by {lead}."})
        return _stage_ir("thompson_sampling", atoms)

    NARRATIVE = (
        "Thompson Sampling ranked NN_1 first. The run was divided into 2 "
        "regimes led by 2 detectors. Regime 0 (windows 0 to 4) was led by "
        "NN_3, with context feature 7 raising its reward. Regime 1 (windows 5 to 18) "
        "was led by NN_1. Across the regimes NN_1 led, context feature 7 contributed "
        "most.")

    # A regime's second sentence — the deviation clause — carries no detector
    # name and no window range. Its ONLY anchor is "In regime N".
    TWO_SENTENCE_NARRATIVE = (
        "Thompson Sampling ranked NN_1 first. The run was divided into 2 "
        "regimes led by 2 detectors. Regime 0 (windows 0 to 4) was led by "
        "NN_3, with context feature 7 raising its reward. In regime 0, context feature 4 "
        "departed furthest from its usual contribution, running below it. "
        "Regime 1 (windows 5 to 18) was led by NN_1. In regime 1, context feature 7 "
        "departed furthest from its usual contribution, running above it. "
        "Across the regimes NN_1 led, context feature 7 contributed most.")

    def test_regime_walk_leaves_the_default_view(self):
        out = summarize.summarize(self.NARRATIVE, stage="thompson_sampling",
                                  ir_doc=self._ir())
        self.assertNotIn("Regime 0", out["summary"])
        self.assertNotIn("Regime 1", out["summary"])

    def test_a_regime_is_anchored_by_its_index_or_its_window_range(self):
        """The two forms the prompt offers, and the one it does not.

        A word-ordinal ("the first regime") is not an anchor: the narrator
        wrote it for every regime at once, so it identifies nothing. The index
        and the window range each identify one regime exactly, and accepting
        both is what lets the openings vary. Failure is quiet either way —
        `narrated` stays empty and the disclosure renders the IR's own wording.
        """
        ir_doc = self._ir()
        for narrative in (
            ("Thompson Sampling ranked NN_1 first. Regime 0 (windows 0 "
             "to 4) was led by NN_3. Regime 1 (windows 5 to 18) was led "
             "by NN_1."),
            ("Thompson Sampling ranked NN_1 first. Across windows 0 to 4, "
             "NN_3 was in charge. NN_1 then led from window 5 to 18."),
        ):
            regimes = artifacts._regimes_from_ir(ir_doc)
            artifacts._attach_narrated_regimes(regimes, narrative, ir_doc)
            self.assertTrue(all(r.get("narrated") for r in regimes), narrative)

        ordinal = ("Thompson Sampling ranked NN_1 first. In the first regime "
                   "NN_3 was in charge. The second regime was led by NN_1.")
        regimes = artifacts._regimes_from_ir(ir_doc)
        artifacts._attach_narrated_regimes(regimes, ordinal, ir_doc)
        self.assertTrue(all(not r.get("narrated") for r in regimes))

    def test_a_regimes_second_sentence_goes_with_its_regime(self):
        """Resilience, not the primary path: the IR now packs every claim about
        a regime into ONE sentence, so there is normally no second sentence to
        strand. But the narrator may still split one off, and if it does the
        clause must carry the regime number — that name is the only thing
        attribution can key on. It has no detector name, and its lone context feature
        index gets captured by whichever unrelated atom holds that integer
        ("context feature 4" landing on the regime summary's "4 detectors"). Unanchored,
        such sentences escaped the walk, survived the summary drop, and piled up
        as a block of context-free sentences at the end of the card.
        """
        ir_doc = self._ir()
        out = summarize.summarize(self.TWO_SENTENCE_NARRATIVE,
                                  stage="thompson_sampling", ir_doc=ir_doc)
        # Gone from the default view, with the rest of the walk.
        self.assertNotIn("departed furthest", out["summary"])
        # And filed under the right regime — context feature 4 with 0, context feature 7 with 1.
        regimes = artifacts._regimes_from_ir(ir_doc)
        artifacts._attach_narrated_regimes(regimes, self.TWO_SENTENCE_NARRATIVE,
                                           ir_doc)
        self.assertIn("context feature 4 departed furthest", regimes[0]["narrated"])
        self.assertNotIn("context feature 7 departed", regimes[0]["narrated"])
        self.assertIn("context feature 7 departed furthest", regimes[1]["narrated"])
        # The roll-up says "regimes" with no index and must NOT be swallowed by
        # the last regime just because it trails the walk.
        self.assertNotIn("contributed most", regimes[1]["narrated"])

    CARRIED_NARRATIVE = (
        "Thompson Sampling ranked NN_1 first. The run was divided into 2 "
        "regimes led by 2 detectors. Regime 0 (windows 0 to 4) was led by "
        "NN_3, with context feature 7 raising its reward. This regime also "
        "saw context feature 4 depart furthest from its usual contribution. "
        "Regime 1 (windows 5 to 18) was led by NN_1. Across the regimes NN_1 "
        "led, context feature 7 contributed most.")

    def test_an_unanchored_second_sentence_carries_the_regime_forward(self):
        """Only the FIRST sentence about a regime needs an anchor. The second
        may refer back ("This regime also saw ..."), which is what lets the
        walk read as prose rather than eight copies of one template.

        The veto is the roll-up that trails the last regime: it accounts for an
        atom of its own, so it breaks the carry rather than joining regime 1.
        """
        ir_doc = self._ir()
        regimes = artifacts._regimes_from_ir(ir_doc)
        artifacts._attach_narrated_regimes(regimes, self.CARRIED_NARRATIVE, ir_doc)
        self.assertIn("depart furthest", regimes[0]["narrated"])
        self.assertNotIn("depart furthest", regimes[1].get("narrated", ""))
        self.assertNotIn("contributed most", regimes[1]["narrated"])

        out = summarize.summarize(self.CARRIED_NARRATIVE,
                                  stage="thompson_sampling", ir_doc=ir_doc)
        # It leaves the default view WITH its regime, not stranded after it.
        self.assertNotIn("depart furthest", out["summary"])
        self.assertIn("Across the regimes NN_1 led", out["summary"])

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
        share one detector name and one context feature number with some regime. A
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
                         "context feature 7 raising its reward.")
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
             "text": "Ranked by the size of its mean vector, NN_1 scored "
                     "0.559322, ahead of NN_2 by 0.000410."},
            {"id": "tsr.winner.channels", "type": "winner_channels",
             "subject": "NN_1",
             "value": {"channel": 3, "total": 0.5,
                       "per_channel": [[3, 0.25], [0, 0.15], [7, 0.10]]},
             "text": "NN_1's score is built mostly from context feature 3 (50.0%)."},
            {"id": "tsr.gap.runner_up", "type": "rank_gap", "subject": "NN_1",
             "value": {"rivals": ["NN_2"], "runner_up": "NN_2",
                       "per_channel": [[3, 0.091], [0, -0.058]]},
             "text": "NN_1's lead over NN_2 came mostly from context feature 3 (0.091)."},
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
        "Ranked by the size of its mean vector, NN_1 scored 0.559322, ahead "
        "of NN_2 by 0.000410. NN_1's score is built mostly from context feature 3 "
        "(50.0%). NN_1's lead over NN_2 came mostly from context feature 3 (0.091). "
        "NN_1 was selected in 23 of the 173 windows, against 30 for NN_2. "
        "Leadership splits into 2 regimes led by 2 detectors. Regime 0 (windows "
        "10 to 71) was led by NN_2. Regime 1 (windows 72 to 172) was led by NN_1.")

    def test_summary_stops_after_the_answer(self):
        """The default view answers the question — winner, the context features its
        score is built from, which context features decided the margin — and stops.
        The selection counts, the regime walk and the limitations are all
        supporting detail behind the click."""
        out = summarize.summarize(self.NARRATIVE, stage="thompson_ranking",
                                  ir_doc=self._ir())
        self.assertIn("built mostly from context feature 3", out["summary"])
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
        """The context feature split is the stage's headline answer and 38 context features on
        SMD do not read as prose, so this stage holds sentences back AND
        tabulates — a combination no other stage used before."""
        out = summarize.summarize(self.NARRATIVE, stage="thompson_ranking",
                                  ir_doc=self._ir())
        table = out["table"]
        self.assertEqual(table["columns"],
                         ["Context feature", "Share", "Contribution", "vs NN_2"])
        self.assertEqual([r[0] for r in table["rows"]],
                         ["context feature 3", "context feature 0", "context feature 7"])
        self.assertEqual(table["collapse_after"], 5)
        self.assertEqual(table["rows"][0][1], "50.0%")
        self.assertEqual(table["rows"][0][3], "+0.091000")
        # A context feature the gap atom does not mention renders as a blank, not a zero:
        # "no delta recorded" and "the two are level" are different claims.
        self.assertIsNone(table["rows"][2][3])

    def test_the_table_keeps_every_context_feature_and_folds_the_tail(self):
        """The shares only mean anything because they sum to the whole score,
        so no context feature may be dropped however many there are — SMD carries 38.
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
                       "borda_rank": 1},
             "text": "GAN_F1 shaped the consensus most."},
            {"id": "s2", "type": "source_role", "subject": "GAN_PR_AUC",
             "value": {"influence_rank": 5, "agreement_rank": 2,
                       "borda_rank": 2},
             "text": "GAN_PR_AUC shaped the consensus second most."},
        ])
        out = summarize.summarize("Its first-ranked detector is LOF_1. GAN_F1 "
                                  "shaped the consensus most.",
                                  stage="rank_aggregation_robust", ir_doc=ir_doc)
        table = out["table"]
        self.assertEqual(table["columns"], ["Overall Rank", "Source",
                                            "Influence Rank", "Agreement Rank"])
        self.assertEqual(table["rows"][0], [1, "GAN_F1", 1, 1])
        self.assertEqual(table["rows"][1], [2, "GAN_PR_AUC", 5, 2])
        # The lead answers what this stage is asked — which source shaped the
        # consensus — and is the narrative's own sentence, not invented copy.
        # It is picked by atom type, so it holds wherever the narrator put it.
        self.assertEqual(out["summary"], "GAN_F1 shaped the consensus most.")

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

    def test_tsb_ad_datasets_and_entities(self):
        for sub, names in (("TSB-AD-U", ("001_NAB_id_1_Facility_tr_1007_1st_2014.csv",
                                         "010_NAB_id_10_WebService_tr_500_1st_271.csv",
                                         "README.md")),
                           ("TSB-AD-M", ("002_MSL_id_1_Sensor_tr_500_1st_900.csv",))):
            (self.data / sub).mkdir()
            for name in names:
                (self.data / sub / name).write_text("x")
        catalog.reset_cache()
        by_name = {d["name"]: d for d in catalog.datasets()}
        self.assertEqual(by_name["tsb-ad-u"]["label"], "TSB-AD-U")
        self.assertEqual(by_name["tsb-ad-m"]["label"], "TSB-AD-M")
        self.assertIn("univariate", by_name["tsb-ad-u"]["note"])
        self.assertEqual(catalog.entities_for("tsb-ad-u"),
                         ["001_NAB_id_1_Facility", "010_NAB_id_10_WebService"])
        self.assertEqual(catalog.entities_for("TSB-AD-M"), ["002_MSL_id_1_Sensor"])
        self.assertTrue(by_name["tsb-ad-u"]["runnable"])
        self.assertTrue(by_name["tsb-ad-u"]["labels_available"])
        self.assertEqual(by_name["tsb-ad-u"]["default_online_labels"], "real")
        # SKAB's loader labels changepoints, not anomalies: injected by default.
        self.assertFalse(by_name["skab"]["labels_available"])
        self.assertEqual(by_name["skab"]["default_online_labels"], "injected")
        self.assertTrue(by_name["smd"]["labels_available"])
        self.assertEqual([s["token"] for s in catalog.catalog(refresh=True)
                          ["online_label_sources"]], ["real", "injected"])

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
        """Always the whole canonical pool, flagged available or not — a
        detector with no checkpoint is now selectable (it trains first), so it
        has to be listed rather than dropped."""
        dets = {d["name"]: d for d in catalog.detectors_for("SKAB", "7")}
        self.assertEqual(len(dets), len(catalog.ALL_DETECTORS))
        self.assertTrue(dets["LOF_1"]["available"])
        self.assertFalse(dets["CBLOF_4"]["available"])
        # LOF's checkpoint here predates the move off the contamination sweep,
        # so its sidecar holds no `n_neighbors` and the label falls back to the
        # grid — what a retrain will give it. See `catalog._read_meta`.
        self.assertEqual(dets["LOF_1"]["params"]["label"], "k 10")
        # Availability comes from disk, the LIST never does. GMM has checkpoints
        # on some entities and no implementation in the repo, so it must not
        # appear however many .pth files are lying around.
        self.assertNotIn("GMM_1", dets)
        # The families added beyond LOF/NN/CBLOF are in the pool and, on this
        # fixture, untrained.
        for name in ("ABOD_1", "KDE_1", "IFOREST_1", "HBOS_1", "OCSVM_1"):
            self.assertIn(name, dets)
            self.assertFalse(dets[name]["available"], name)

    def test_width_unsuitable_families_are_hidden_not_disabled(self):
        """A missing checkpoint can be trained; a family that cannot mean
        anything at this width can never run, so the run page should not offer
        it at all. `app.py` drops it too — this is the UI half."""
        from Utils.pipeline_spec import UNIVARIATE_FAMILIES, MULTIVARIATE_FAMILIES
        self.assertEqual(catalog.unusable_families(38), UNIVARIATE_FAMILIES)
        self.assertEqual(catalog.unusable_families(1), MULTIVARIATE_FAMILIES)

    def test_unknown_channel_count_hides_nothing(self):
        """Failing open: hiding a usable detector is worse than showing one the
        run would drop anyway."""
        self.assertEqual(catalog.unusable_families(None), frozenset())

    def test_every_detector_carries_its_paper_group(self):
        """The run page colours chips and builds its select-all buttons from
        this, so a detector without a group would render uncoloured and be
        unreachable from any group button."""
        dets = catalog.detectors_for("SKAB", "7")
        self.assertTrue(all(d["group"] for d in dets),
                        [d["name"] for d in dets if not d["group"]])
        by_name = {d["name"]: d for d in dets}
        self.assertEqual(by_name["RNN_1"]["group"], "NN")
        self.assertEqual(by_name["LOF_1"]["group"], "Stat")
        # k-Nearest Neighbors is Statistical; the group called NN is Neural
        # Networks. Pinned here too because this is the payload the UI reads.
        self.assertEqual(by_name["NN_1"]["group"], "Stat")

    def test_catalog_lists_groups_including_the_empty_one(self):
        """FM has no detectors in the pool, so it cannot be inferred from the
        detector list — the run page needs it from here to show the button.

        `Graph` is the fourth group and is NOT one of the paper's three; see
        `Utils.pipeline_spec.DETECTOR_GROUPS`.
        """
        self.assertEqual(catalog.catalog(refresh=True)["detector_groups"],
                         ["NN", "Stat", "FM", "Graph"])

    def test_every_family_has_a_chip_colour(self):
        """`familyClass` emits chip-{family}; a family with no rule falls back
        to grey, which is the state all sixteen non-original families were in."""
        css = (Path(__file__).parent / "static" / "css" / "ramses.css").read_text()
        for family in catalog.DETECTOR_FAMILIES:
            with self.subTest(family=family):
                self.assertIn(f".chip-{family.lower()} ", css)
                self.assertIn(f"--fam-{family.lower()}:", css)

    def test_corrupt_meta_degrades_to_the_grid_not_to_nothing(self):
        """A sidecar that cannot be unpickled used to leave the chip with no
        parameters at all. The family's grid still knows what NN_2 is — the
        instance number indexes it — so the fallback answers from there and
        only a family missing from FAMILY_GRIDS yields None."""
        dets = {d["name"]: d for d in catalog.detectors_for("SKAB", "7")}
        self.assertTrue(dets["NN_2"]["available"])
        self.assertEqual(dets["NN_2"]["params"]["label"], "k 3")

    def test_untrained_detectors_still_report_their_hyperparameters(self):
        """Nothing is on disk for them, so the sidecar cannot answer; the grid
        can, and the run page needs it to tell LOF_1 from LOF_4 BEFORE deciding
        whether to train the family."""
        dets = {d["name"]: d for d in catalog.detectors_for("SKAB", "7")}
        self.assertFalse(dets["CBLOF_4"]["available"])
        self.assertEqual(dets["CBLOF_4"]["params"]["label"], "clusters 32")

    def test_every_family_names_something_that_varies(self):
        """The point of the change: RNN, LSTMVAE, DGHL and RM were blank
        because the old lookup only knew contamination and n_neighbors."""
        dets = {d["name"]: d for d in catalog.detectors_for("SKAB", "7")}
        expected = {
            "RNN_1": "input_size 32, state_hsize 128",
            "LSTMVAE_1": "hidden_size 512, latent_size 256",
            "DGHL_1": "z_iters 25, z_size 25",
            "RM_3": "running window 64",
            "LSTMAD_2": "subsequence 100",
            # The families that stopped sweeping contamination now name the
            # parameter TSB-AD varies for them.
            "IFOREST_1": "trees 25",
            "HBOS_4": "bins 30",
            "PCA_1": "components 0.25",
            "OCSVM_3": "kernel rbf",
            "MCD_4": "support fraction 0.8",
        }
        for name, label in expected.items():
            self.assertEqual(dets[name]["params"]["label"], label, name)
        # MD is a single instance: nothing varies, so there is nothing to name.
        self.assertIsNone(dets["MD_1"]["params"]["label"])

    def test_case_insensitive_entity_lookup(self):
        self.assertTrue(any(d["available"] for d in catalog.detectors_for("skab", "7")))

    def test_config_overwrite_is_not_warned_about(self):
        """`overwrite: True` in the config file never reaches a run started from
        the form — build_argv always passes --overwrite from the checkbox — so
        warning about it described a state the UI cannot produce."""
        self.assertNotIn("overwrite_on", [w["code"] for w in catalog.warnings()])

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
        self.results_root = Path(self._tmp.name) / "results"
        self._saved = paths.RESULTS
        paths.RESULTS = self.results_root

    def tearDown(self):
        paths.RESULTS = self._saved
        self._tmp.cleanup()

    def _touch(self, rel):
        p = self.results_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x89PNG")
        return p

    def test_dedupe_keeps_newest_of_each_real_pattern(self):
        """The four naming irregularities that actually occur on disk."""
        d = self.results_root / "robustness" / "off_by" / "SKAB" / "7"
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
        whatever is on disk rather than a fixed list.

        The final stage shows only its agreement-only figure: it merges two
        sources, where leave-one-out and Borda are degenerate, so the standard
        figure's influence bars carry no information there. The robust stage has
        six sources and keeps its own.
        """
        self._touch("robust_aggregated/SKAB/7/aggregation_explainability_final_0.png")
        self._touch("robust_aggregated/SKAB/7/aggregation_explainability_final_kendall_only_0.png")
        self._touch("robust_aggregated/SKAB/7/aggregation_explainability_robust_0.png")
        final, _ = self.plots._aggregation("SKAB", "7", "final")
        robust, _ = self.plots._aggregation("SKAB", "7", "robust")
        self.assertEqual([f["title"] for f in final], ["Agreement only (two sources)"])
        self.assertEqual([f["title"] for f in robust], ["Robust aggregation"])

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

    def test_context_feature_plot_captions_state_the_selection_rule(self):
        """These figures plot a subset — 9 context features on SKAB, 38 on SMD — and the
        bars cannot say whether a missing context feature was small or just not picked.

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
        self.results_root = Path(self._tmp.name) / "results"
        (self.results_root / "GA_Ens").mkdir(parents=True)
        self.png = self.results_root / "GA_Ens" / "plot.png"
        self.png.write_bytes(b"\x89PNG")
        self.secret = Path(self._tmp.name) / "secret.png"
        self.secret.write_bytes(b"\x89PNG")
        (self.results_root / "model.pth").write_bytes(b"x")
        (self.results_root / "report.json").write_text("{}")
        self._saved = paths.RESULTS
        paths.RESULTS = self.results_root

    def tearDown(self):
        paths.RESULTS = self._saved
        self._tmp.cleanup()

    def test_serves_a_png_inside_the_tree(self):
        self.assertIsNotNone(self.plots.safe_media_path("GA_Ens/plot.png"))

    def test_bracketed_filename_is_servable(self):
        odd = self.results_root / "GA_Ens" / "ensemble_scores_SKAB_7_Data_vs_anomalies_['spikes'].png"
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
        link = self.results_root / "GA_Ens" / "escape.png"
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

    def test_each_incompatible_checkpoint_gets_its_own_banner(self):
        """app.load_trained_models logs one line per excluded checkpoint, and
        Job._record keeps one banner per code — so the code must carry the
        detector, or a second stale checkpoint would vanish behind the first."""
        from WebUI.jobs import Job
        lines = [
            "WARNING:__main__:⚠ Incompatible checkpoint LOF_1 excluded: it cannot "
            "score this data — ValueError: X has 576 features, but "
            "LocalOutlierFactor is expecting 1 features as input. Remedy: retrain "
            "it with --overwrite true. The checkpoint file was left untouched.",
            "WARNING:__main__:⚠ Incompatible checkpoint CBLOF_1 excluded: it cannot "
            "score this data — ValueError: X has 576 features, but KMeans is "
            "expecting 1 features as input. Remedy: retrain it with --overwrite "
            "true. The checkpoint file was left untouched.",
        ]
        e = self.m.classify(lines[0])
        self.assertEqual((e["type"], e["code"]), ("warning", "checkpoint_incompatible:LOF_1"))
        self.assertTrue(e["text"].startswith("Incompatible checkpoint LOF_1 excluded"))
        self.assertIn("--overwrite true", e["text"])
        job = Job("j", argv=[], params={})
        for line in lines + lines:                 # repeats must not duplicate
            job._record(line)
        self.assertEqual([w["code"] for w in job.warnings],
                         ["checkpoint_incompatible:LOF_1",
                          "checkpoint_incompatible:CBLOF_1"])

    def test_a_detector_that_failed_to_train_raises_a_banner(self):
        """Both the trainer's per-instance line (loguru) and app.py's summary
        (stdlib logging) must reach the run page, as one banner — the run
        continues without the detector, so it is a warning, not a failure.
        `Model_Training/test_train` checks the trainer's real output against
        this rule as well."""
        lines = (
            "2026-10-20 17:00:06.535 | WARNING  | Model_Training.train:_record_failure:388 "
            "- ⚠ SOS_1 failed to train and is left out of this run: it exceeded "
            "the 10s training limit (training_timeout in the config) and was stopped",
            "WARNING:__main__:⚠ Left out of this run because they failed to train: "
            "SOS_1. The warnings above give each reason.",
        )
        for line in lines:
            w = self.m.classify(line)
            self.assertEqual((w["type"], w["code"]), ("warning", "training_failed"), line)

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

    def test_anomaly_flags(self):
        argv = self._argv(anomaly_type="wander", anomaly_rate=0.25)
        self.assertEqual(argv[argv.index("--anomaly_type") + 1], "wander")
        self.assertEqual(argv[argv.index("--anomaly_rate") + 1], "0.25")

    def test_default_anomaly_type_emits_no_flag(self):
        self.assertNotIn("--anomaly_type", self._argv(anomaly_type="spikes"))
        self.assertNotIn("--anomaly_type", self._argv(anomaly_types=["spikes"]))
        self.assertNotIn("--anomaly_type", self._argv())
        self.assertNotIn("--anomaly_rate", self._argv())

    def test_several_anomaly_types_become_one_comma_separated_flag(self):
        """The multi-select sends `anomaly_types`; the CLI takes them as one
        comma-separated value, in the order chosen and without repeats."""
        argv = self._argv(anomaly_types=["noise", "spikes", "noise"])
        self.assertEqual(argv[argv.index("--anomaly_type") + 1], "noise,spikes")
        self.assertEqual(argv.count("--anomaly_type"), 1)
        argv = self._argv(anomaly_type=["scale", "wander"])       # older field, list value
        self.assertEqual(argv[argv.index("--anomaly_type") + 1], "scale,wander")

    def test_decision_metric_flag(self):
        argv = self._argv(decision_metric="pr_auc")
        self.assertEqual(argv[argv.index("--decision_metric") + 1], "pr_auc")

    def test_default_decision_metric_emits_no_flag(self):
        self.assertNotIn("--decision_metric", self._argv(decision_metric="f1,pr_auc"))
        self.assertNotIn("--decision_metric", self._argv())

    def test_a_single_metric_is_no_longer_the_default(self):
        argv = self._argv(decision_metric="f1")
        self.assertEqual(argv[argv.index("--decision_metric") + 1], "f1")

    def test_weighted_decision_metrics_reach_the_flag(self):
        argv = self._argv(decision_metrics={"f1": 0.5, "pr_auc": 0.3, "vus": 0.2})
        self.assertEqual(argv[argv.index("--decision_metric") + 1],
                         "f1:0.5,pr_auc:0.3,vus:0.2")

    def test_uniform_weights_keep_the_plain_spelling(self):
        argv = self._argv(decision_metrics={"f1": 1, "vus": 1})
        self.assertEqual(argv[argv.index("--decision_metric") + 1], "f1,vus")

    def test_uniform_weights_on_the_default_metrics_emit_no_flag(self):
        self.assertNotIn("--decision_metric",
                         self._argv(decision_metrics={"f1": 1, "pr_auc": 1}))
        self.assertNotIn("--decision_metric",
                         self._argv(decision_metrics={"f1": 0.5, "pr_auc": 0.5}))

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

    def test_cancel_wins_over_everything(self):
        job = self._job([self.FULL], cancel_requested=True)
        self.assertEqual(self.jobs.classify_outcome(job, False)["status"], "cancelled")
        job = self._job(["INFO:app:working..."], cancel_requested=True, status="timeout")
        self.assertEqual(self.jobs.classify_outcome(job, False)["status"], "cancelled")

    def test_a_timeout_before_completion_is_a_timeout(self):
        job = self._job(["INFO:app:STAGE 6/7: Running Model Selection Algorithms..."],
                        status="timeout", exit_code=-9)
        out = self.jobs.classify_outcome(job, False)
        self.assertEqual(out["status"], "timeout")
        self.assertIn("No completion after", out["reason"])

    def test_a_timeout_after_the_completion_marker_keeps_the_offline_success(self):
        """The watchdog is one timer for the whole run and the online phase
        runs after the marker: the offline results stand, with a warning."""
        job = self._job([self.FULL], status="timeout", exit_code=-9)
        out = self.jobs.classify_outcome(job, False)
        self.assertEqual(out["status"], "succeeded_with_warnings")
        self.assertIn("killed by the", out["reason"])
        self.assertIn("after the offline phase finished", out["reason"])
        online = self._job([self.FULL,
                            "INFO:x:ONLINE PHASE START: strategy=all, scenarios=fixed, "
                            "windows=4"],
                           status="timeout", exit_code=-9, params={"enable_online": True})
        out = self.jobs.classify_outcome(online, False)
        self.assertEqual(out["status"], "succeeded_with_warnings")
        self.assertIn("during the online phase, whose results are incomplete", out["reason"])

    def test_a_partial_run_timing_out_after_its_marker_is_still_a_success_with_warnings(self):
        job = self._job([self.PARTIAL], status="timeout", exit_code=-9)
        out = self.jobs.classify_outcome(job, expect_partial=True)
        self.assertEqual(out["status"], "succeeded_with_warnings")

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
        # it, and a test must never read the developer's real results/ tree.
        self._saved = (paths.WEBUI_LOGS, paths.COMPREHENSIVE)
        paths.WEBUI_LOGS = Path(self._tmp.name) / "webui_logs"
        paths.COMPREHENSIVE = Path(self._tmp.name) / "results" / "comprehensive"
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
        for path in ("/", "/result/SKAB/7", "/report/SKAB/7", "/docs/SKAB/7"):
            html = self.client.get(path).get_data(as_text=True)
            self.assertEqual(html.count("js/dom.js"), 0, path)

    def test_the_detector_pool_has_no_checkboxes(self):
        """Detector, family and group all say "chosen" the same way.

        The chips used to be a <label> wrapping a checkbox while the group
        buttons filled and the family buttons showed nothing, so one screen had
        three notations for one idea. They are all buttons with aria-pressed
        now. `selectedDetectors` was always the only thing `currentBody` reads,
        so the DOM is purely a reflection of it — but a reintroduced checkbox
        would still be a second source of truth waiting to disagree.
        """
        js = (Path(__file__).parent / "static" / "js" / "configure.js").read_text()
        # renderDetectors was split into renderGroupButtons / syncGroupButtons /
        # syncDetectorButtons / familyParams / refreshDetectors, which occupy the
        # stretch from the first of them up to the training banner.
        pool = js.split("function renderGroupButtons")[1].split("function renderTrainingBanner")[0]
        self.assertNotIn("checkbox", _strip_js_comments(pool),
                         "detector chips must be buttons, not checkboxes")
        self.assertIn('"data-detector"', pool)
        self.assertIn('"aria-pressed"', pool)
        # Nothing may drive the selection through input elements any more.
        self.assertNotIn('$$("#detectors input")', js)
        # The stage chips keep theirs: those are an ordinary multi-select.
        self.assertIn('$$("#stages input:checked")', js)

    def test_chosen_chips_and_group_buttons_share_one_style(self):
        """`.is-on` is what fills a group button; the chips must use the same
        class or "chosen" would look different one row down."""
        css = (Path(__file__).parent / "static" / "css" / "ramses.css").read_text()
        self.assertIn(".toggle.is-on", css)
        self.assertIn("button.grp-nn.is-on", css)
        # Buttons do not inherit the page font, so without this the pool would
        # render in the browser's default control face.
        self.assertIn("button.toggle { font: inherit", css)

    def test_configure_page_has_the_training_banner_slot(self):
        """configure.js writes the "this run trains first" warning into
        #training-banner and returns silently when it is missing, so the element
        drifting out of the template would remove the warning without failing
        anything."""
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="training-banner"', html)

    def _tabs(self, path):
        html = self.client.get(path).get_data(as_text=True)
        nav = re.search(r'<nav class="view-tabs".*?</nav>', html, re.S)
        self.assertIsNotNone(nav, path)
        return re.findall(r'<a id="([\w-]+)"', nav.group(0)), nav.group(0)

    def test_every_view_shows_the_same_tabs(self):
        """One result, four pages, one tab set: the views that exist."""
        sets = {p: self._tabs(p)[0] for p in
                ("/result/SKAB/7", "/report/SKAB/7", "/docs/SKAB/7", "/online/SKAB/7")}
        # SKAB/7 has an explanation and a report but no online data; the
        # online page keeps its own tab.
        self.assertEqual(sets["/result/SKAB/7"], ["to-explanation", "open-report", "open-docs"])
        self.assertEqual(sets["/report/SKAB/7"], sets["/result/SKAB/7"])
        self.assertEqual(sets["/docs/SKAB/7"], sets["/result/SKAB/7"])
        self.assertEqual(sets["/online/SKAB/7"],
                         ["to-explanation", "open-report", "open-online", "open-docs"])
        for path in sets:
            _, nav = self._tabs(path)
            self.assertEqual(nav.count('aria-current="page"'), 1, path)
        self.assertEqual(self.client.get("/api/views/SKAB/7").get_json(),
                         {"explanation": True, "report": True, "online": False, "docs": True})

    def test_a_run_without_explanations_offers_no_explanation_tabs(self):
        """A report with nothing to explain: no Explanation or Docs tab leading
        to an empty page, on its report page as on its online page."""
        _write(self.results_root / "comprehensive" / "SKAB" / "8" /
               "comprehensive_results_SKAB_8_iter5.txt", REPORT_TEXT)
        self.assertEqual(self._tabs("/report/SKAB/8")[0], ["open-report"])
        self.assertEqual(self._tabs("/online/SKAB/8")[0], ["open-report", "open-online"])
        self.assertEqual(self.client.get("/api/views/SKAB/8").get_json(),
                         {"explanation": False, "report": True, "online": False, "docs": False})

    def test_pages_render(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/result/SKAB/7").status_code, 200)
        self.assertEqual(self.client.get("/report/SKAB/7").status_code, 200)
        self.assertEqual(self.client.get("/docs/SKAB/7").status_code, 200)
        # No online data yet is the page's own empty state, not a 404.
        self.assertEqual(self.client.get("/online/SKAB/7").status_code, 200)
        self.assertEqual(self.client.get("/run/nope").status_code, 404)

    def test_documentation_is_served_and_missing_entities_404(self):
        body = self.client.get("/api/docs/SKAB/7").get_json()
        self.assertEqual([s["id"] for s in body["sections"]],
                         ["overview", "ga", "lints", "gan", "off-by",
                          "monte-carlo", "rank-aggregation"])
        # Still per-entity, and still 404 for one that has nothing: the page
        # carries a link back to the entity the reader came from.
        self.assertEqual(self.client.get("/api/docs/SKAB/999").status_code, 404)

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
            ({"dataset": "S", "entity": "7", "anomaly_type": "nope"}, "unknown anomaly type"),
            ({"dataset": "S", "entity": "7", "anomaly_types": ["spikes", "nope"]},
             "unknown anomaly type: nope"),
            ({"dataset": "S", "entity": "7", "anomaly_types": []}, "at least one anomaly type"),
            ({"dataset": "S", "entity": "7", "anomaly_rate": 0}, "at most 1"),
            ({"dataset": "S", "entity": "7", "anomaly_rate": 1.5}, "at most 1"),
            ({"dataset": "S", "entity": "7", "anomaly_rate": "x"}, "must be a number"),
            ({"dataset": "S", "entity": "7", "decision_metric": "auroc"},
             "unknown decision metric"),
            ({"dataset": "S", "entity": "7", "decision_metrics": {"f1": -1}},
             "must not be negative"),
            ({"dataset": "S", "entity": "7", "decision_metrics": {"f1": "x"}},
             "must be numbers"),
            ({"dataset": "S", "entity": "7", "decision_metrics": {"f1": 0, "vus": 0}},
             "at least one decision metric"),
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
        png = self.results_root / "GA_Ens" / "x.png"
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
        self.results_root = Path(self._tmp.name) / "results"
        self._saved = paths.RESULTS
        paths.RESULTS = self.results_root

    def tearDown(self):
        paths.RESULTS = self._saved
        self._tmp.cleanup()

    def _write_ir(self, shares):
        d = self.results_root / "explanations_ir" / "SKAB" / "7"
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
        self.assertEqual(self.ondemand.ranking_context_feature_shares("SKAB", "7"), {})
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


class TestOnDemandPerWindowFrames(unittest.TestCase):
    """The nine per-window sets, drawn per request from persisted aggregates.

    These were nine folders of PNGs — over a thousand frames and 167 MB for one
    173-window entity — of which a reader opens a handful. The numbers behind
    them are three orders of magnitude smaller, so the pipeline saves those and
    the frame is rendered on demand. `_all` and `_every10` were never different
    figures, only a different top-k and stride, which is why they collapse into
    arguments here.
    """

    MODELS = ["A", "B", "C"]
    N_WINDOWS = 25
    N_CONTEXT_FEATURES = 4

    def setUp(self):
        from WebUI import ondemand, plots
        self.ondemand, self.plots = ondemand, plots
        self.ondemand._PW_CACHE.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.results_root = Path(self._tmp.name) / "results"
        self._saved = paths.RESULTS
        paths.RESULTS = self.results_root

    def tearDown(self):
        paths.RESULTS = self._saved
        self.ondemand._PW_CACHE.clear()
        self._tmp.cleanup()

    def _write_doc(self, **overrides):
        """A per-window document whose rows are distinct per (window, model)."""
        def frame(t, scale):
            return [[float(scale * (t + 1) * (i + 1) * (c + 1))
                     for c in range(self.N_CONTEXT_FEATURES)]
                    for i in range(len(self.MODELS))]
        doc = {
            "schema": 1, "n_channels": self.N_CONTEXT_FEATURES,
            "n_windows": self.N_WINDOWS, "top_k_models": 2, "top_n_channels": 3,
            "models": list(self.MODELS),
            "models_by_final_norm": list(reversed(self.MODELS)),
            "kinds": {
                "reward": {"ylabel": "y", "title_top": "R {t} top {k}",
                           "title_all": "R {t} all", "note": "note {t}",
                           "rank_by": "reward", "all_by": "final"},
                "ranking": {"ylabel": "y", "title_top": "S {t} top {k}",
                            "title_all": "S {t} all", "note": None,
                            "rank_by": "ranking", "all_by": "ranking"},
            },
            "sets": {"reward": [frame(t, 1.0) for t in range(self.N_WINDOWS)],
                     "ranking": [frame(t, 2.0) for t in range(self.N_WINDOWS)]},
        }
        doc.update(overrides)
        d = self.results_root / "Thomposon" / "SKAB" / "7"
        d.mkdir(parents=True, exist_ok=True)
        (d / "per_window_channels_50.json").write_text(json.dumps(doc))
        (d / "expected_rewards_50.png").write_bytes(b"\x89PNG")
        return d

    def test_descriptors_come_from_the_document_not_the_disk(self):
        self._write_doc()
        by_id = {d["id"]: d for d in self.plots.gallery_descriptors("SKAB", "7")}
        self.assertIn("thompson/pw:reward:top:1", by_id)
        self.assertIn("ts_ranking/pw:ranking:all:1", by_id)
        self.assertEqual(by_id["thompson/pw:reward:top:1"]["count"], self.N_WINDOWS)
        # 25 windows at stride 10 is windows 0, 10 and 20 — a count, not a
        # folder listing, so it cannot disagree with what the pages return.
        self.assertEqual(by_id["thompson/pw:reward:top:10"]["count"], 3)
        # The SHAP set is absent from this document, so it is not offered.
        self.assertNotIn("thompson/pw:shap:top:1", by_id)

    def test_pages_carry_render_urls_and_honour_the_stride(self):
        self._write_doc()
        page = self.plots.gallery_page("SKAB", "7", "thompson/pw:reward:top:1", 0, 10)
        self.assertEqual(page["total"], self.N_WINDOWS)
        self.assertEqual(len(page["items"]), 10)
        self.assertEqual(page["items"][3]["src"],
                         "/api/plots/SKAB/7/per-window?kind=reward&scope=top&t=3")
        strided = self.plots.gallery_page("SKAB", "7", "ts_ranking/pw:ranking:all:10")
        self.assertEqual([i["src"].rsplit("t=", 1)[1] for i in strided["items"]],
                         ["0", "10", "20"])

    def test_top_k_and_all_are_the_same_data_read_two_ways(self):
        """The scopes differ only in how many detectors are kept and in what
        order — the rows behind them are one stored frame."""
        self._write_doc()
        doc = self.ondemand.per_window_document("SKAB", "7")
        top = self.ondemand._select_models(doc, "reward", 4, "top")
        every = self.ondemand._select_models(doc, "reward", 4, "all")
        self.assertEqual(top[1], ["C", "B"])          # top 2 by row sum
        self.assertEqual(every[1], ["C", "B", "A"])   # final-norm order
        # The ranking set re-sorts per window instead of using the fixed order.
        self.assertEqual(
            self.ondemand._select_models(doc, "ranking", 4, "all")[1], ["C", "B", "A"])

    def test_a_stale_folder_cannot_shadow_the_persisted_numbers(self):
        """A re-run stops writing the folders but does not delete the ones an
        earlier run left, so the document has to win."""
        d = self._write_doc()
        for i in range(99):
            frame = d / "reward_per_window_50" / f"window_{i:03d}.png"
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"\x89PNG")
        ids = {x["id"]: x["count"] for x in self.plots.gallery_descriptors("SKAB", "7")}
        self.assertNotIn("thompson/reward_per_window_50", ids)
        self.assertEqual(ids["thompson/pw:reward:top:1"], self.N_WINDOWS)

    def test_older_runs_still_list_their_folders(self):
        """No document means a tree written before this existed; those frames
        are still PNGs on disk and are listed the way they always were."""
        d = self.results_root / "Thomposon" / "SKAB" / "7"
        (d / "shap_per_window_50").mkdir(parents=True)
        (d / "expected_rewards_50.png").write_bytes(b"\x89PNG")
        for i in range(5):
            (d / "shap_per_window_50" / f"window_{i:03d}.png").write_bytes(b"\x89PNG")
        ids = {x["id"]: x["count"] for x in self.plots.gallery_descriptors("SKAB", "7")}
        self.assertEqual(ids.get("thompson/shap_per_window_50"), 5)

    def test_unavailable_frames_return_none_rather_than_raising(self):
        self._write_doc()
        for kind, t, scope in (("reward", self.N_WINDOWS, "top"),
                               ("reward", -1, "top"),
                               ("shap", 0, "top")):
            self.assertIsNone(
                self.ondemand.render_per_window("SKAB", "7", kind, t, scope))
        self.assertIsNone(self.ondemand.render_per_window("SKAB", "9", "reward", 0))

    def test_a_frame_renders_to_a_png(self):
        self._write_doc()
        png = self.ondemand.render_per_window("SKAB", "7", "reward", 2, "top")
        self.assertTrue(png and png.startswith(b"\x89PNG"))

    def test_nothing_is_written_to_the_result_tree(self):
        """Same guarantee the ranking-gap figure carries: a browsing session
        cannot litter results/ or race a run writing into it."""
        self._write_doc()
        before = sorted(p.name for p in
                        (self.results_root / "Thomposon" / "SKAB" / "7").iterdir())
        self.ondemand.render_per_window("SKAB", "7", "ranking", 1, "all")
        after = sorted(p.name for p in
                       (self.results_root / "Thomposon" / "SKAB" / "7").iterdir())
        self.assertEqual(before, after)


class TestOffByTreeSelector(unittest.TestCase):
    """One surrogate tree at a time, chosen by competitor.

    Every pair is already on disk, so this is a selector over files — but ten
    near-identical trees stacked down the card is ten figures to scroll past.
    """

    def setUp(self):
        from WebUI import plots
        self.plots = plots
        self._tmp = tempfile.TemporaryDirectory()
        self.results_root = Path(self._tmp.name) / "results"
        self._saved = paths.RESULTS
        paths.RESULTS = self.results_root

    def tearDown(self):
        paths.RESULTS = self._saved
        self._tmp.cleanup()

    def _touch(self, rel):
        p = self.results_root / rel
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
        # The picker is the whole headline now: the importance figure moved to
        # the browse gallery, since it describes the comparison in general
        # rather than this entity's decision.
        self.assertEqual(sum(1 for f in headline if "variants" not in f), 0)
        _, gallery = self.plots._off_by("SKAB", "7")
        self.assertIn("Which point properties separate the winner",
                      [f["title"] for f in gallery])

    def test_only_the_latest_run_s_trees_are_offered(self):
        """Tree filenames carry the winner, not a timestamp, so a run that picks
        a different winner writes a whole new set beside the old one instead of
        overwriting it. Listing both put a previous run's competitors in the
        picker and titled the card with the OLD winner — SKAB/7 really did show
        "Where LOF_1 uniquely wins" while the run being read had chosen
        CBLOF_4."""
        import os
        base = "robustness/off_by/SKAB/7"
        for competitor in ("NN_1", "NN_2"):
            self._touch(f"{base}/SKAB_7_off_by_point_tree_LOF_1_vs_{competitor}.png")
        for competitor in ("LOF_1", "NN_1", "NN_3"):
            self._touch(f"{base}/skab_7_off_by_point_tree_CBLOF_4_vs_{competitor}.png")
        # Make the CBLOF_4 set unambiguously the newer one.
        for competitor in ("LOF_1", "NN_1", "NN_3"):
            p = self.results_root / base / f"skab_7_off_by_point_tree_CBLOF_4_vs_{competitor}.png"
            os.utime(p, (p.stat().st_atime + 60, p.stat().st_mtime + 60))
        headline, _ = self.plots._off_by("SKAB", "7")
        picker = next(f for f in headline if "variants" in f)
        self.assertEqual(picker["title"], "Where CBLOF_4 uniquely wins")
        self.assertEqual(sorted(v["title"] for v in picker["variants"]),
                         ["LOF_1", "NN_1", "NN_3"])

    def test_misclassified_points_are_written_but_not_listed(self):
        """Still produced by the stage on every run — the GAN card already shows
        the same figure under the same title, so listing it here twice made the
        off-by gallery say nothing specific to off-by."""
        base = "robustness/off_by/SKAB/7"
        self._touch(f"{base}/SKAB_7_Misclassified Anomalies_2026-08-16_13-07-59.png")
        self._touch(f"{base}/Data_vs_DataWithAnomalies_2026-08-16_13-07-31_.png")
        _, gallery = self.plots._off_by("SKAB", "7")
        titles = [f["title"] for f in gallery]
        self.assertNotIn("Misclassified points", titles)
        self.assertIn("Injected borderline points", titles)


class TestGanPlotSelector(unittest.TestCase):
    """The GAN card is off-by's sibling: same selector, its own filenames.

    Both stages explain a winner's exclusive wins over injected points, so the
    trees are the headline and everything else is one click away.
    """

    def setUp(self):
        from WebUI import plots
        self.plots = plots
        self._tmp = tempfile.TemporaryDirectory()
        self.results_root = Path(self._tmp.name) / "results"
        self._saved = paths.RESULTS
        paths.RESULTS = self.results_root

    def tearDown(self):
        paths.RESULTS = self._saved
        self._tmp.cleanup()

    def _touch(self, rel):
        p = self.results_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x89PNG")
        return p

    def test_trees_collapse_into_one_selector_named_by_competitor(self):
        base = "robustness/GAN/SKAB/7"
        for competitor in ("CBLOF_1", "LOF_1", "NN_2"):
            self._touch(f"{base}/skab_7_gan_point_tree_CBLOF_4_vs_{competitor}.png")
        self._touch(f"{base}/skab_7_gan_point_importance.png")
        headline, gallery = self.plots._gan("SKAB", "7")
        picker = next(f for f in headline if "variants" in f)
        self.assertEqual(picker["title"], "Where CBLOF_4 uniquely wins")
        self.assertEqual([v["title"] for v in picker["variants"]],
                         ["CBLOF_1", "LOF_1", "NN_2"])
        self.assertEqual(picker["select_label"], "Compared against")
        self.assertEqual(picker["default"], 0)
        # The picker IS the headline — nothing else competes with it there.
        self.assertEqual(sum(1 for f in headline if "variants" not in f), 0)
        self.assertIn("Which point properties separate the winner",
                      [f["title"] for f in gallery])

    def test_only_the_latest_run_s_trees_are_offered(self):
        """Tree filenames carry the winner, not a timestamp, so a run that picks
        a different winner writes a whole new set beside the old one."""
        base = "robustness/GAN/SKAB/7"
        for competitor in ("CBLOF_1", "NN_1"):
            self._touch(f"{base}/skab_7_gan_point_tree_LOF_1_vs_{competitor}.png")
        for competitor in ("LOF_1", "NN_1", "NN_3"):
            self._touch(f"{base}/skab_7_gan_point_tree_CBLOF_4_vs_{competitor}.png")
        for competitor in ("LOF_1", "NN_1", "NN_3"):
            p = self.results_root / base / f"skab_7_gan_point_tree_CBLOF_4_vs_{competitor}.png"
            os.utime(p, (p.stat().st_atime + 60, p.stat().st_mtime + 60))
        headline, _ = self.plots._gan("SKAB", "7")
        picker = next(f for f in headline if "variants" in f)
        self.assertEqual(picker["title"], "Where CBLOF_4 uniquely wins")
        self.assertEqual(sorted(v["title"] for v in picker["variants"]),
                         ["LOF_1", "NN_1", "NN_3"])

    def test_the_misclassified_figure_is_written_but_listed_nowhere(self):
        """Both point-injection cards drop it, so it appears on neither.

        The stage still writes it every run — it plots true against predicted
        labels for the winner alone, which says nothing about the per-rival
        comparison either card is for. off-by's copy is dropped for the same
        reason, so a reader never meets it twice under one title.
        """
        base = "robustness/GAN/SKAB/7"
        self._touch(f"{base}/skab_7_Misclassified_Anomalies_2026-08-16_13-07-05_.png")
        self._touch(f"{base}/skab_7_Data_vs_DataWithAnomalies_2026-08-16_13-07-05.png")
        _, gallery = self.plots._gan("SKAB", "7")
        titles = [f["title"] for f in gallery]
        self.assertNotIn("Misclassified points", titles)
        self.assertIn("Injected borderline points", titles)

    def test_only_the_newest_injected_points_figure_is_listed(self):
        """One entry, whichever stem it came from.

        These filenames begin with the dataset as it was typed on the command
        line, and both cases run, so an entity worked on as `--dataset SKAB` and
        again as `--dataset skab` leaves two stems. `dedupe_timestamped` keeps
        the newest of EACH stem, which listed the same figure twice; the card
        wants the newest overall, and the hidden count has to cover both stems.
        """
        base = "robustness/GAN/SKAB/7"
        self._touch(f"{base}/SKAB_7_Data_vs_DataWithAnomalies_2026-05-17_12-35-34.png")
        self._touch(f"{base}/SKAB_7_Data_vs_DataWithAnomalies_2026-05-18_09-00-00.png")
        self._touch(f"{base}/skab_7_Data_vs_DataWithAnomalies_2026-08-17_20-29-26.png")
        _, gallery = self.plots._gan("SKAB", "7")
        injected = [f for f in gallery if f["title"] == "Injected borderline points"]
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0]["name"],
                         "skab_7_Data_vs_DataWithAnomalies_2026-08-17_20-29-26.png")
        self.assertEqual(injected[0]["n_older"], 2)

    def test_a_run_without_explain_still_renders(self):
        """No trees on disk — the headline is empty and the figures the stage
        always writes are still reachable, exactly as _off_by degrades."""
        base = "robustness/GAN/SKAB/7"
        self._touch(f"{base}/skab_7_Data_vs_DataWithAnomalies_2026-08-16_13-07-05.png")
        headline, gallery = self.plots._gan("SKAB", "7")
        self.assertEqual(headline, [])
        self.assertEqual([f["title"] for f in gallery], ["Injected borderline points"])


# ── DOM insertion never receives null (configure page "…/v1null") ───────────

_JS_DIR = Path(__file__).parent / "static" / "js"
_DOM_INSERT = re.compile(r"\.(replaceChildren|append|prepend|before|after|replaceWith)\(")
# A word before `/` after which the slash opens a regex literal, not a division.
_REGEX_KEYWORDS = {"return", "typeof", "case", "in", "of", "void", "delete",
                   "throw", "new", "yield", "await"}


def _mask_js(src: str) -> str:
    """`src` with every comment, string, template literal and regex literal
    blanked to spaces, same length, so brackets can be matched on what is left.

    The checks below look at the shape of a call's arguments; a "(" inside
    `Show all ${n} rows (${hidden} more)` or inside a regex would otherwise
    throw every bracket after it out of step.
    """
    n = len(src)
    out = list(src)

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    def end_of_string(i: int) -> int:
        quote, j = src[i], i + 1
        while j < n and src[j] != quote:
            j += 2 if src[j] == "\\" else 1
        return j + 1

    def end_of_regex(i: int) -> int:
        j, in_class = i + 1, False
        while j < n:
            c = src[j]
            if c == "\\":
                j += 2
                continue
            if c == "[":
                in_class = True
            elif c == "]":
                in_class = False
            elif c == "/" and not in_class:
                break
            j += 1
        j += 1
        while j < n and src[j].isalpha():        # flags
            j += 1
        return j

    def end_of_template(i: int) -> int:
        j = i + 1
        while j < n and src[j] != "`":
            if src[j] == "\\":
                j += 2
            elif src.startswith("${", j):
                j = scan(j + 2, closer="}")
            else:
                j += 1
        return j + 1

    def scan(i: int, closer: str = "", record: bool = False) -> int:
        depth, prev = 0, "("
        j = i
        while j < n:
            c = src[j]
            end = None
            if c in "'\"":
                end = end_of_string(j)
            elif c == "`":
                end = end_of_template(j)
            elif src.startswith("//", j):
                end = src.find("\n", j)
                end = n if end < 0 else end
            elif src.startswith("/*", j):
                end = src.index("*/", j) + 2
            elif c == "/":
                after_word = prev.isalnum() or prev in "_$"
                word = re.search(r"([A-Za-z_$][\w$]*)\s*$", src[:j]) if after_word else None
                if (not after_word and prev not in ")]") or (
                        word and word.group(1) in _REGEX_KEYWORDS):
                    end = end_of_regex(j)
            if end is not None:
                if record:
                    blank(j, end)
                j, prev = end, "a"
                continue
            if c in "([{":
                depth += 1
            elif c in ")]}":
                if depth == 0 and c == closer:
                    return j + 1
                depth -= 1
            if not c.isspace():
                prev = c
            j += 1
        return j

    scan(0, record=True)
    return "".join(out)


def _matching(masked: str, open_index: int) -> int:
    """Index of the bracket closing the one at `open_index`."""
    depth = 0
    for k in range(open_index, len(masked)):
        if masked[k] in "([{":
            depth += 1
        elif masked[k] in ")]}":
            depth -= 1
            if depth == 0:
                return k
    raise AssertionError(f"unbalanced bracket at {open_index}")


def _top_level(text: str) -> str:
    """`text` with everything inside brackets removed: `a(b, c ? null : d)`
    becomes `a()`. What is left is what the expression itself evaluates to."""
    out, depth = [], 0
    for c in text:
        if c in ")]}":
            depth -= 1
        if depth == 0:
            out.append(c)
        if c in "([{":
            depth += 1
    return "".join(out)


def _split_args(text: str):
    args, depth, start = [], 0, 0
    for k, c in enumerate(text):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            args.append(text[start:k])
            start = k + 1
    args.append(text[start:])
    return [a for a in args if a.strip()]


def _nullable_dom_insertions(sources):
    """Every argument of replaceChildren/append/prepend/before/after/replaceWith
    that can evaluate to null, undefined or false.

    Unlike dom.js's el(), which skips those children, the DOM methods
    stringify them: `replaceChildren(a, b, ok ? null : c)` renders the word
    "null". Two shapes are caught — a top-level null/undefined/false or `&&`
    in the argument itself, and a direct call to a function declaration whose
    body can `return null` / `return;`. Spreads are not judged: what an array
    holds is decided where it is built.
    """
    masked = {name: _mask_js(src) for name, src in sources.items()}
    nullable = set()
    for text in masked.values():
        for m in re.finditer(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", text):
            params_end = _matching(text, m.end() - 1)
            body_open = text.index("{", params_end)
            body = text[body_open:_matching(text, body_open) + 1]
            if re.search(r"\breturn\s*(?:null\b|undefined\b)?\s*;", body):
                nullable.add(m.group(1))

    found = []
    for name, text in masked.items():
        for m in _DOM_INSERT.finditer(text):
            close = _matching(text, m.end() - 1)
            for arg in _split_args(text[m.end():close]):
                top = _top_level(arg).strip()
                if top.startswith("..."):
                    continue
                call = re.fullmatch(r"([A-Za-z_$][\w$]*)\(\)", top)
                if re.search(r"\b(?:null|undefined|false)\b|&&", top) or (
                        call and call.group(1) in nullable):
                    line = text.count("\n", 0, m.start()) + 1
                    found.append(f"{name}:{line}: .{m.group(1)}({' '.join(arg.split())[:60]})")
    return found


class TestDomInsertionNeverGetsNull(unittest.TestCase):
    """The configure page read "qwen3.5:9b-mlx at http://localhost:11434/v1null":
    renderHealth passed `llm.reachable ? null : el(...)` to replaceChildren,
    which stringifies null where el() would have skipped it. The result page
    had the same bug latent — `body.append(termsNode(stage))`, with a termsNode
    that returns null for a stage without definitions."""

    def _sources(self):
        return {p.name: p.read_text(encoding="utf-8") for p in sorted(_JS_DIR.glob("*.js"))}

    def test_the_masker_keeps_every_file_balanced(self):
        """If the masker misread a string or a regex, every bracket after it
        would be off and the check below would judge the wrong text, so a
        lexing slip has to fail loudly instead of passing quietly."""
        for name, src in self._sources().items():
            masked = _mask_js(src)
            self.assertEqual(len(masked), len(src), name)
            depth = 0
            for c in masked:
                depth += (c in "([{") - (c in ")]}")
                self.assertGreaterEqual(depth, 0, name)
            self.assertEqual(depth, 0, name)

    def test_no_dom_insertion_can_receive_null(self):
        self.assertEqual(_nullable_dom_insertions(self._sources()), [])

    def test_the_check_catches_both_original_bugs(self):
        """The two call sites as they were, so the guard above is known to
        bite rather than merely to pass."""
        before = {
            "configure.js": """
                $("#llm-health").replaceChildren(
                  el("span", { class: "badge" }, "✓ LLM server reachable"),
                  el("span", { text: ` ${llm.model} at ${llm.base_url}` }),
                  llm.reachable ? null : el("span", { class: "muted", text:
                    " — the run will finish (eventually)." }));
            """,
            "result.js": """
                function termsNode(stage) {
                  const terms = stage.terms || [];
                  if (!terms.length) return null;
                  return el("dl", { class: "terms" }, terms.flat());
                }
                function stageCard(stage) {
                  body.append(termsNode(stage));
                  body.append(proseNode(stage.summary));
                }
            """,
        }
        found = _nullable_dom_insertions(before)
        self.assertEqual(len(found), 2, found)
        self.assertTrue(any(f.startswith("configure.js") for f in found), found)
        self.assertTrue(any("termsNode" in f for f in found), found)

    def test_el_still_skips_null_children(self):
        """Every `cond ? el(...) : null` passed to el() relies on this; the
        DOM methods above are exactly the ones that do not."""
        dom = (_JS_DIR / "dom.js").read_text(encoding="utf-8")
        self.assertIn("if (child === null || child === undefined || child === false) continue;",
                      dom)


# ── [hidden] wins over component display rules ───────────────────────────────

class TestHiddenAttributeHides(unittest.TestCase):
    """`a.button { display: inline-block }` beat the browser's own
    `[hidden] { display: none }`, so a failed run showed "View explanation →"
    and "Comprehensive results" linking to results that were never written,
    and every result page offered a report link whether or not one existed."""

    def setUp(self):
        css = (Path(__file__).parent / "static" / "css" / "ramses.css").read_text(encoding="utf-8")
        self.css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    def test_hidden_is_display_none_important_at_the_top_level(self):
        m = re.search(r"(?:^|[};])\s*\[hidden\]\s*\{\s*display\s*:\s*none\s*!important\s*;?\s*\}",
                      self.css)
        self.assertIsNotNone(m, "ramses.css needs `[hidden] { display: none !important; }`")
        # Inside an @media block it would hide only on that medium.
        before = self.css[:m.start() + 1]
        self.assertEqual(before.count("{") - before.count("}"), 0,
                         "the [hidden] rule must not be nested in an at-rule")

    def test_no_screen_rule_forces_a_display_past_it(self):
        """`!important` on the [hidden] rule beats every ordinary rule, but
        another `display: … !important` with a more specific selector would
        beat it in turn. Outside print, none may exist."""
        screen = re.sub(r"@media\s+print\s*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", self.css)
        offenders = [d for d in re.findall(r"display\s*:\s*([\w-]+)\s*!important", screen)
                     if d != "none"]
        self.assertEqual(offenders, [])

    def test_the_conditional_links_start_hidden(self):
        """The scripts only ever reveal these (`link.hidden = false`) once the
        target exists, so each must start out carrying the attribute. (The
        header's view tabs are not among them: the server renders only the
        views that exist — see TestViewTabs.)"""
        templates = Path(__file__).parent / "templates"
        run_html = (templates / "run.html").read_text(encoding="utf-8")
        for html, element_id in ((run_html, "view-result"), (run_html, "view-report")):
            tag = re.search(rf'<a\b[^>]*\bid="{element_id}"[^>]*>', html)
            self.assertIsNotNone(tag, element_id)
            self.assertRegex(tag.group(0), r"\shidden(?=[\s>])", element_id)


# ── LLM health: reachable is not enough, the model has to be installed ───────

class _FakeModelsResponse:
    def __init__(self, status_code=200, payload=None, error=None):
        self.status_code = status_code
        self._payload = payload
        self._error = error

    def json(self):
        if self._error is not None:
            raise self._error
        return self._payload


def _listing(*ids):
    return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}


@unittest.skipUnless(HAS_FLASK, "Flask is not installed in this interpreter")
class TestLLMHealth(unittest.TestCase):
    """The default model (qwen3.5:9b-mlx) was not installed on the local
    Ollama, yet the configure page said "✓ LLM server reachable" — Ollama
    answers /models either way — and the run then wrote no narratives."""

    def setUp(self):
        from unittest import mock
        from WebUI import jobs, server
        self.mock = mock
        self.server = server
        # Scratch environment, so a test that sets a variable cannot leak it.
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        jobs.reset_manager()
        self.addCleanup(jobs.reset_manager)

    def _health(self, response=None, error=None, model=None, base_url=None):
        get = self.mock.Mock(return_value=response, side_effect=error)
        with self.mock.patch("requests.get", get):
            return self.server._health(model, base_url), get

    def test_a_reachable_server_without_the_model_reports_it_missing(self):
        health, _ = self._health(_FakeModelsResponse(
            payload=_listing("qwen3-4b:latest", "phi4:latest")))
        self.assertTrue(health["reachable"])
        self.assertEqual(health["model"], self.server.DEFAULT_LLM_MODEL)
        self.assertIs(health["model_available"], False)
        self.assertEqual(health["available_models"], ["qwen3-4b:latest", "phi4:latest"])
        self.assertEqual(health["n_available_models"], 2)

    def test_an_installed_model_is_available(self):
        health, _ = self._health(_FakeModelsResponse(payload=_listing("qwen3-4b:latest")),
                                 model="qwen3-4b:latest")
        self.assertIs(health["model_available"], True)
        self.assertEqual(health["model"], "qwen3-4b:latest")

    def test_a_bare_name_matches_its_latest_tag_and_nothing_else(self):
        """Ollama serves "phi4" as "phi4:latest"; a different tag, or a name that
        is merely a prefix of an installed one, is not the same model."""
        response = _FakeModelsResponse(payload=_listing("phi4:latest", "qwen3-4b:latest"))
        for model, expected in (("phi4", True), ("phi4:latest", True),
                                ("phi4:14b", False), ("qwen3", False)):
            health, _ = self._health(response, model=model)
            self.assertIs(health["model_available"], expected, model)

    def test_names_ollama_resolves_to_an_installed_model_are_installed(self):
        """Ollama 0.12.11 serves all of these as the listed model, so warning
        that they are missing would send the reader to pull what they have."""
        response = _FakeModelsResponse(payload=_listing(
            "phi4:latest", "hf.co/unsloth/Phi-4-mini-instruct-GGUF:Q8_0"))
        for model, expected in (
                ("PHI4", True), ("Phi4:LATEST", True), ("library/phi4", True),
                ("registry.ollama.ai/library/phi4:latest", True),
                ("hf.co/unsloth/phi-4-mini-instruct-gguf:q8_0", True),
                # Untagged means :latest, which this repo does not have.
                ("hf.co/unsloth/Phi-4-mini-instruct-GGUF", False),
                ("other/phi4", False)):
            health, _ = self._health(response, model=model)
            self.assertIs(health["model_available"], expected, model)
            self.assertEqual(health["model"], model)    # shown as typed

    def test_an_empty_override_falls_back_to_the_default(self):
        health, _ = self._health(_FakeModelsResponse(payload=_listing()), model="   ")
        self.assertEqual(health["model"], self.server.DEFAULT_LLM_MODEL)
        self.assertIs(health["model_available"], False)
        self.assertEqual(health["available_models"], [])

    def test_the_environment_does_not_change_what_is_checked(self):
        """The run never reads these variables (app.py falls back to the
        pipeline defaults), so a check that did said "phi4 installed" while the
        run asked for the default model, which was not."""
        from WebUI import jobs
        os.environ["RAMSES_LLM_MODEL"] = "phi4"
        os.environ["RAMSES_LLM_BASE_URL"] = "http://llm.local:1234/v1/"
        health, get = self._health(_FakeModelsResponse(payload=_listing("phi4:latest")))
        self.assertEqual(health["model"], self.server.DEFAULT_LLM_MODEL)
        self.assertIs(health["model_available"], False)
        self.assertEqual(get.call_args[0][0],
                         self.server.DEFAULT_LLM_BASE_URL.rstrip("/") + "/models")
        # ...and the run with an empty form asks for that same default.
        argv = jobs.build_argv({"dataset": "skab", "entity": "1"})
        self.assertNotIn("--llm_model", argv)
        self.assertNotIn("--llm_base_url", argv)

    def test_a_custom_server_is_not_checked_and_not_fetched(self):
        """With "LLM base URL" set, the run talks to that server. Checking the
        default one instead reported a model the custom server has as "not
        installed"; fetching the typed URL would make this GET a request relay."""
        import requests
        for response, error in ((_FakeModelsResponse(payload=_listing("phi4:latest")), None),
                                (None, requests.exceptions.ConnectionError("down"))):
            health, get = self._health(response, error=error,
                                       model="meta-llama/Llama-3.1-8B-Instruct",
                                       base_url=" http://127.0.0.1:8000/v1 ")
            get.assert_not_called()
            self.assertIs(health["checked"], False)
            self.assertIsNone(health["reachable"])        # not False: unknown
            self.assertIsNone(health["model_available"])  # not False: unknown
            self.assertEqual(health["base_url"], "http://127.0.0.1:8000/v1")
            self.assertEqual(health["model"], "meta-llama/Llama-3.1-8B-Instruct")
            self.assertEqual(health["available_models"], [])

    def test_the_default_server_typed_out_is_still_checked(self):
        """Typing the default URL, with or without the trailing slash LLMClient
        strips, names the same server, so the verdict still applies."""
        default = self.server.DEFAULT_LLM_BASE_URL
        for typed in (default, default.rstrip("/") + "/", "", "   ", None):
            health, get = self._health(_FakeModelsResponse(payload=_listing("phi4:latest")),
                                       model="phi4", base_url=typed)
            get.assert_called_once()
            self.assertIs(health["checked"], True, typed)
            self.assertIs(health["model_available"], True, typed)

    def test_the_check_keeps_its_time_budget(self):
        _, get = self._health(_FakeModelsResponse(payload=_listing()))
        self.assertEqual(get.call_args.kwargs["timeout"], 1.5)

    def test_an_unreachable_server_never_raises(self):
        import requests
        for error in (requests.exceptions.ConnectionError("refused"),
                      requests.exceptions.Timeout("slow"), RuntimeError("anything")):
            health, _ = self._health(error=error)
            self.assertIs(health["reachable"], False, error)
            self.assertIsNone(health["model_available"], error)
            self.assertEqual(health["available_models"], [], error)

    def test_a_server_error_is_unreachable(self):
        health, _ = self._health(_FakeModelsResponse(status_code=503))
        self.assertIs(health["reachable"], False)
        self.assertIsNone(health["model_available"])

    def test_an_unreadable_listing_is_unknown_not_missing(self):
        """Up, but no OpenAI-shaped list: there is nothing to compare against,
        so the page must not claim the model is missing."""
        for response in (_FakeModelsResponse(status_code=404),
                         _FakeModelsResponse(error=ValueError("not json")),
                         _FakeModelsResponse(payload=["phi4:latest"]),
                         _FakeModelsResponse(payload={"models": []}),
                         _FakeModelsResponse(payload={"data": "phi4"})):
            health, _ = self._health(response)
            self.assertIs(health["reachable"], True)
            self.assertIsNone(health["model_available"])

    def test_malformed_entries_are_skipped(self):
        health, _ = self._health(_FakeModelsResponse(
            payload={"data": [{"id": "phi4:latest"}, {"name": "x"}, "y", {"id": ""}]}),
            model="phi4")
        self.assertEqual(health["available_models"], ["phi4:latest"])
        self.assertIs(health["model_available"], True)

    def test_the_sample_of_installed_models_is_capped(self):
        ids = [f"m{i}:latest" for i in range(20)]
        health, _ = self._health(_FakeModelsResponse(payload=_listing(*ids)))
        self.assertEqual(health["available_models"], ids[:self.server.HEALTH_MODEL_SAMPLE])
        self.assertEqual(health["n_available_models"], 20)

    def test_the_route_checks_the_form_override(self):
        client = self.server.create_app(TESTING=True).test_client()
        get = self.mock.Mock(return_value=_FakeModelsResponse(payload=_listing("phi4:latest")))
        with self.mock.patch("requests.get", get):
            default = client.get("/api/health").get_json()["llm"]
            override = client.get("/api/health?model=phi4").get_json()["llm"]
            blank = client.get("/api/health?model=").get_json()["llm"]
        self.assertEqual(default["model"], self.server.DEFAULT_LLM_MODEL)
        self.assertIs(default["model_available"], False)
        self.assertEqual(override["model"], "phi4")
        self.assertIs(override["model_available"], True)
        self.assertEqual(blank["model"], self.server.DEFAULT_LLM_MODEL)

    def test_the_route_does_not_judge_a_custom_server(self):
        client = self.server.create_app(TESTING=True).test_client()
        get = self.mock.Mock(return_value=_FakeModelsResponse(payload=_listing("phi4:latest")))
        with self.mock.patch("requests.get", get):
            custom = client.get("/api/health", query_string={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "base_url": "http://127.0.0.1:8000/v1"}).get_json()["llm"]
        get.assert_not_called()
        self.assertIs(custom["checked"], False)
        self.assertIsNone(custom["model_available"])
        self.assertEqual(custom["base_url"], "http://127.0.0.1:8000/v1")

    def test_the_configure_page_warns_about_a_missing_model(self):
        """The UI half: the missing-model state has a warning of its own that
        names the model and both ways out, and the Advanced field re-checks."""
        js = (_JS_DIR / "configure.js").read_text(encoding="utf-8")
        self.assertIn("llm.model_available === false", js)
        self.assertIn("! LLM model not installed", js)
        self.assertIn("ollama pull ${llm.model}", js)
        self.assertIn('set "LLM model" under Advanced', js)
        self.assertIn('query.set("model", model)', js)
        self.assertIn('["#llm_model", "#llm_base_url"].forEach', js)

    def test_the_configure_page_drops_both_verdicts_for_a_custom_server(self):
        """A custom base URL is sent along, and an unchecked answer is handled
        before the unreachable and missing-model branches, which would
        otherwise both fire: `reachable` is null and so falsy."""
        js = (_JS_DIR / "configure.js").read_text(encoding="utf-8")
        self.assertIn('query.set("base_url", baseUrl)', js)
        unchecked = js.index("llm.checked === false")
        self.assertLess(unchecked, js.index("!llm.reachable"))
        self.assertLess(unchecked, js.index("llm.model_available === false"))
        self.assertIn("LLM server not checked", js)


# ── online phase ─────────────────────────────────────────────────────────────
#
# The fixtures are produced by the pipeline's own runner and writer
# (Model_Selection/online_phase.py) with fake scoring, so these tests pin the
# contract between what the pipeline writes and logs and what the UI reads.

def _online_run(results_root: Path, dataset="SKAB", entity="7",
                keys=("fixed", "periodic"), n=4, log=None):
    """Run the real online runner with fake scores into `results_root`.

    Re-optimisations finish at once and are switched in one window after
    their trigger (reopt_delay=1); the one triggered at the last window is
    due after it and is recorded as not applied.
    """
    from Model_Selection.online_phase import (OnlineResultsWriter, OnlineWindow,
                                              Selection, WindowScores, run_online,
                                              scenario_params)
    from Model_Selection.online_policy import PolicyParams
    from Model_Selection.online_reopt import CompletedReopt

    def score(window, selection):
        return WindowScores(ensemble=0.8 if window.index != 3 else float("nan"),
                            single=0.6, metrics={}, misclassified={})

    def start(window, selection, key):
        return CompletedReopt(Selection([f"E{window.index}"], f"S{window.index}",
                                        origin="reoptimization",
                                        origin_window=window.index), 1.5, None)

    params = PolicyParams(update_interval=2, reopt_delay=1)
    writer = OnlineResultsWriter(
        str(results_root / "online" / dataset / entity),
        {"dataset": dataset.lower(), "entity": entity, "strategy": "all",
         "window_size": 100, "step_size": 5, "n_windows_available": n},
        keys, scenario_params=scenario_params(keys, params), n_windows_planned=n)
    windows = [OnlineWindow(index=i, start=800 + 5 * i, end=900 + 5 * i,
                            online_start=5 * i, online_end=5 * i + 100)
               for i in range(1, n + 1)]
    run_online(keys, params, windows, lambda key: Selection(["A", "B"], "A"), score,
               start, writer, log=log)
    return writer


class _CaptureLog:
    """Collects what the runner logs, as the pipeline's log lines would read."""

    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(f"INFO:Model_Selection.online_phase:{msg}")

    def error(self, msg):
        self.lines.extend(f"ERROR:Model_Selection.online_phase:{part}"
                          for part in msg.splitlines())


class TestOnlineArgv(unittest.TestCase):

    def setUp(self):
        from WebUI import jobs
        self.jobs = jobs
        self.base = {"dataset": "SKAB", "entity": "7"}

    def _argv(self, **kw):
        return self.jobs.build_argv({**self.base, **kw})

    def test_online_flags_need_enable_online(self):
        argv = self._argv(online_strategy="all", update_interval=3, drop_pct=10,
                          inject_online_regime=True)
        for flag in ("--enable_online", "--strategy", "--update_interval",
                     "--drop_pct", "--inject_online_regime"):
            self.assertNotIn(flag, argv)

    def test_every_parameter_reaches_its_flag(self):
        argv = self._argv(enable_online=True, online_strategy="all", update_interval=3,
                          drop_pct=12.5, adaptive_window=8, adaptive_k=1.5,
                          max_online_windows=20, inject_online_regime=True)
        self.assertIn("--enable_online", argv)
        self.assertIn("--inject_online_regime", argv)
        for flag, value in (("--strategy", "all"), ("--update_interval", "3"),
                            ("--drop_pct", "12.5"), ("--adaptive_window", "8"),
                            ("--adaptive_k", "1.5"), ("--max_online_windows", "20")):
            self.assertEqual(argv[argv.index(flag) + 1], value, flag)

    def test_omitted_parameters_fall_back_to_the_config(self):
        """No flag means config.yml decides, so the UI never has to restate it."""
        argv = self._argv(enable_online=True)
        self.assertIn("--enable_online", argv)
        for flag in ("--strategy", "--update_interval", "--drop_pct", "--adaptive_window",
                     "--adaptive_k", "--max_online_windows", "--inject_online_regime"):
            self.assertNotIn(flag, argv)
        argv = self._argv(enable_online=True, update_interval=None, drop_pct="")
        self.assertNotIn("--update_interval", argv)
        self.assertNotIn("--drop_pct", argv)

    def test_online_labels_reach_their_flag_and_auto_sends_none(self):
        argv = self._argv(enable_online=True, online_labels="real")
        self.assertEqual(argv[argv.index("--online_labels") + 1], "real")
        argv = self._argv(enable_online=True, online_labels="Injected")
        self.assertEqual(argv[argv.index("--online_labels") + 1], "injected")
        for unset in (None, "", "auto"):
            self.assertNotIn("--online_labels", self._argv(enable_online=True,
                                                           online_labels=unset))
        self.assertNotIn("--online_labels", self._argv(online_labels="real"))
        with self.assertRaises(ValueError):
            self._argv(enable_online=True, online_labels="maybe")

    def test_reopt_delay_reaches_its_flag(self):
        argv = self._argv(enable_online=True, reopt_delay=0)
        self.assertEqual(argv[argv.index("--reopt_delay") + 1], "0")
        self.assertNotIn("--reopt_delay", self._argv(enable_online=True))
        with self.assertRaises(ValueError):
            self._argv(enable_online=True, reopt_delay=-2)

    def test_whole_number_floats_are_spelled_plainly(self):
        argv = self._argv(enable_online=True, drop_pct=20.0, adaptive_k=2.0)
        self.assertEqual(argv[argv.index("--drop_pct") + 1], "20")
        self.assertEqual(argv[argv.index("--adaptive_k") + 1], "2")

    def test_legacy_strategy_parameter_and_aliases(self):
        argv = self._argv(enable_online=True, strategy="adaptive")
        self.assertEqual(argv[argv.index("--strategy") + 1], "adaptive")
        argv = self._argv(enable_online=True, strategy="fixed-best",
                          online_strategy="threshold")
        self.assertEqual(argv[argv.index("--strategy") + 1], "threshold")

    def test_bad_values_are_refused_before_spawning(self):
        for bad in ({"online_strategy": "sometimes"}, {"update_interval": 0},
                    {"drop_pct": 100}, {"adaptive_window": 1}, {"adaptive_k": -1},
                    {"max_online_windows": 0}, {"update_interval": "2.5"}):
            with self.assertRaises(ValueError, msg=bad):
                self._argv(enable_online=True, **bad)


class TestOnlineMarkers(unittest.TestCase):

    def setUp(self):
        from WebUI import markers
        self.m = markers

    def test_every_online_line_kind(self):
        for line, kind in (
            ("ONLINE PHASE START: strategy=all, scenarios=fixed+periodic, windows=4",
             "phase_start"),
            ("ONLINE SCENARIO START: scenario=fixed, position=1/2, windows=4", "scenario_start"),
            ("ONLINE WINDOW: scenario=fixed, window=2/4, index=2, ensemble=0.8000",
             "window"),
            ("ONLINE REOPT: scenario=periodic, event=applied, window=3", "reopt"),
            ("ONLINE SCENARIO COMPLETE: scenario=fixed, windows=4", "scenario_complete"),
            ("ONLINE SCENARIO FAILED: scenario=fixed, error=boom", "scenario_failed"),
            ("ONLINE PHASE COMPLETE: scenarios=2, failed=0, wall_s=1.0", "phase_complete"),
            ("ONLINE PHASE FAILED: error=ValueError: x", "phase_failed"),
            ("ONLINE PHASE SKIPPED: reason=too short", "phase_skipped"),
        ):
            e = self.m.classify(f"INFO:__main__:{line}")
            self.assertEqual((e["type"], e["kind"]), ("online", kind), line)

    def test_fields_and_a_trailing_message_with_commas(self):
        e = self.m.classify("INFO:x:ONLINE WINDOW: scenario=threshold, window=7/20, "
                            "deployed_ensemble=CBLOF_1+LOF_1, trigger=suppressed")
        self.assertEqual(e["fields"]["window"], "7/20")
        self.assertEqual(e["fields"]["deployed_ensemble"], "CBLOF_1+LOF_1")
        e = self.m.classify("ERROR:x:❌ ONLINE PHASE FAILED: error=ValueError: a, b, c")
        self.assertEqual(e["fields"]["error"], "ValueError: a, b, c")
        e = self.m.classify("INFO:x:ONLINE REOPT: scenario=threshold, event=suppressed, "
                            "window=4, because=reoptimization_in_progress, "
                            "reason=relative_drop")
        self.assertEqual(e["fields"]["because"], "reoptimization_in_progress")
        self.assertEqual(e["fields"]["reason"], "relative_drop")

    def test_an_error_level_online_failure_is_not_a_generic_error(self):
        e = self.m.classify("ERROR:__main__:❌ ONLINE PHASE FAILED: error=boom")
        self.assertEqual(e["type"], "online")

    def test_the_thompson_stage_title_is_not_an_online_line(self):
        e = self.m.classify("INFO:app:  📊 Sub-stage 6.2: Thompson Sampling - Online model selection...")
        self.assertEqual(e["type"], "stage")


class TestOnlineJobTracking(unittest.TestCase):
    """What the run page shows for the online phase, from the runner's own lines."""

    FULL = "INFO:app:🎉 EXECUTION COMPLETE! Total Time: 1.0s"

    def setUp(self):
        from WebUI import jobs
        self.jobs = jobs
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _job(self, lines, params=None):
        job = self.jobs.Job("t", ["x"], {"enable_online": True} if params is None else params)
        for line in lines:
            job._record(line)
        job.exit_code = 0
        return job

    def _runner_lines(self, keys=("fixed", "periodic"), n=4):
        log = _CaptureLog()
        _online_run(Path(self._tmp.name), keys=keys, n=n, log=log)
        return log.lines

    def test_the_runner_lines_drive_phase_8_and_one_entry_per_scenario(self):
        job = self._job([self.FULL] + self._runner_lines())
        self.assertEqual(job.phase, {"number": 8, "title": "Online phase"})
        self.assertEqual(job.online["status"], "complete")
        self.assertEqual(job.online["order"], ["fixed", "periodic"])
        periodic = job.online["scenarios"]["periodic"]
        self.assertEqual((periodic["status"], periodic["window"], periodic["total"]),
                         ("done", 4, 4))
        # Triggers at 2 and 4; the one from 2 finished at once and applied at 3.
        self.assertEqual((periodic["triggers"], periodic["reopts_started"],
                          periodic["reopts_applied"]), (2, 2, 1))
        stage = job.stages["online_periodic"]
        self.assertEqual((stage["status"], stage["group"], stage["window"], stage["total"]),
                         ("done", "online", 4, 4))
        self.assertEqual(job.stages["online_fixed"]["status"], "done")
        snap = job.snapshot()
        self.assertEqual(snap["online"]["scenarios"]["periodic"]["window"], 4)

    def test_progress_is_visible_mid_scenario(self):
        lines = [l for l in self._runner_lines(keys=("fixed",), n=4)
                 if "ONLINE SCENARIO COMPLETE" not in l and "ONLINE PHASE COMPLETE" not in l]
        lines = [l for l in lines if "window=4/4" not in l]
        job = self._job([self.FULL] + lines)
        entry = job.stages["online_fixed"]
        self.assertEqual((entry["status"], entry["window"]), ("running", 3))
        self.assertEqual(entry["text"], "window 3/4")

    def test_reoptimization_logs_do_not_reopen_the_offline_rail(self):
        job = self._job([
            "INFO:app:  ✓ [GA] Best ensemble=['LOF_1'] | Time=1.0s",
            self.FULL,
            "INFO:__main__:  📊 Sub-stage 6.1: Genetic Algorithm (GA) - Finding best ensemble...",
        ])
        self.assertEqual(job.stages["ga"]["status"], "done")

    def test_a_succeeded_online_phase_is_a_clean_success(self):
        job = self._job([self.FULL] + self._runner_lines())
        self.assertEqual(self.jobs.classify_outcome(job, False)["status"], "succeeded")

    def test_an_online_crash_after_completion_is_a_warning_not_a_success(self):
        job = self._job([self.FULL,
                         "INFO:x:ONLINE PHASE START: strategy=all, scenarios=fixed, windows=4",
                         "ERROR:__main__:❌ ONLINE PHASE FAILED: error=ValueError: bad"])
        out = self.jobs.classify_outcome(job, False)
        self.assertEqual(out["status"], "succeeded_with_warnings")
        self.assertIn("online phase failed", out["reason"])
        self.assertIn("ValueError: bad", out["reason"])

    def test_a_traceback_after_completion_is_a_warning(self):
        """The old online loop died in run_app's catch-all, after the marker."""
        job = self._job([self.FULL, "INFO:app:Traceback for Entity: 7 Dataset: SKAB"],
                        params={})
        out = self.jobs.classify_outcome(job, False)
        self.assertEqual(out["status"], "succeeded_with_warnings")
        self.assertIn("raised an exception", out["reason"])

    def test_a_traceback_before_completion_is_still_a_failure(self):
        job = self._job(["INFO:app:Traceback for Entity: 7 Dataset: SKAB"])
        self.assertEqual(self.jobs.classify_outcome(job, False)["status"], "failed")

    def test_requested_but_never_started_or_unfinished_or_skipped(self):
        out = self.jobs.classify_outcome(self._job([self.FULL]), False)
        self.assertEqual(out["status"], "succeeded_with_warnings")
        self.assertIn("never started", out["reason"])
        job = self._job([self.FULL,
                         "INFO:x:ONLINE PHASE START: strategy=all, scenarios=fixed, windows=4"])
        self.assertIn("did not finish", self.jobs.classify_outcome(job, False)["reason"])
        job = self._job([self.FULL, "WARNING:x:⚠️  ONLINE PHASE SKIPPED: reason=too short"])
        out = self.jobs.classify_outcome(job, False)
        self.assertEqual(out["status"], "succeeded_with_warnings")
        self.assertIn("skipped: too short", out["reason"])

    def test_one_failed_scenario_is_named(self):
        lines = [self.FULL,
                 "INFO:x:ONLINE PHASE START: strategy=all, scenarios=fixed+periodic, windows=4",
                 "ERROR:x:ONLINE SCENARIO FAILED: scenario=periodic, error=boom",
                 "INFO:x:ONLINE PHASE COMPLETE: scenarios=2, failed=1, wall_s=1.0"]
        job = self._job(lines)
        self.assertEqual(job.stages["online_periodic"]["status"], "failed")
        out = self.jobs.classify_outcome(job, False)
        self.assertEqual(out["status"], "succeeded_with_warnings")
        self.assertIn("periodic", out["reason"])

    def test_online_off_or_partial_runs_are_not_judged_on_it(self):
        self.assertEqual(self.jobs.classify_outcome(
            self._job([self.FULL], params={}), False)["status"], "succeeded")
        partial = "INFO:app:✅ Partial run complete (stages=ga)."
        self.assertEqual(self.jobs.classify_outcome(
            self._job([partial]), True)["status"], "succeeded")

    def test_warnings_combine(self):
        out = self.jobs.classify_outcome(self._job([self.FULL]), False,
                                         explain_requested=True, artifacts_written=False)
        self.assertIn("no explanation artifacts", out["reason"])
        self.assertIn("never started", out["reason"])


class TestOnlineJobLinks(unittest.TestCase):
    """A finished job links to the online results only if THIS run wrote them."""

    def setUp(self):
        from WebUI import jobs
        self.jobs = jobs
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._saved = (paths.WEBUI_LOGS, paths.COMPREHENSIVE, paths.RESULTS)
        paths.WEBUI_LOGS = root / "webui_logs"
        paths.RESULTS = root / "results"
        paths.COMPREHENSIVE = paths.RESULTS / "comprehensive"
        self.mgr = jobs.JobManager(repo_root=root)

    def tearDown(self):
        (paths.WEBUI_LOGS, paths.COMPREHENSIVE, paths.RESULTS) = self._saved
        self._tmp.cleanup()

    def _run(self, script, params):
        job = self.mgr.start(params, argv=[sys.executable, "-u", "-c", script], timeout=30)
        for _ in range(300):
            if job.is_done():
                break
            time.sleep(0.02)
        return job

    def test_online_url_only_for_this_runs_results(self):
        params = {"dataset": "SKAB", "entity": "7", "explain": False, "enable_online": True}
        script = ("import sys; print('INFO:app:🎉 EXECUTION COMPLETE! 1s', file=sys.stderr)\n"
                  "print('INFO:x:ONLINE PHASE START: strategy=fixed, scenarios=fixed, "
                  "windows=1', file=sys.stderr)\n"
                  "print('INFO:x:ONLINE PHASE COMPLETE: scenarios=1, failed=0, wall_s=0', "
                  "file=sys.stderr)\n")
        job = self._run(script, params)
        self.assertIsNone(job.online_url)              # nothing on disk
        _online_run(paths.RESULTS, keys=("fixed",), n=2)
        job = self._run(script, params)
        self.assertEqual(job.status, "succeeded")
        self.assertEqual(job.online_url, "/online/SKAB/7")
        self.assertEqual(job.snapshot()["online_url"], "/online/SKAB/7")
        # A summary older than the job is an earlier run's.
        summary = paths.RESULTS / "online" / "SKAB" / "7" / "summary.json"
        doc = json.loads(summary.read_text())
        doc["generated_at"] = time.time() - 3600
        summary.write_text(json.dumps(doc))
        self.assertIsNone(self._run(script, params).online_url)


class TestOnlineReader(ArtifactTreeCase):

    def setUp(self):
        super().setUp()
        from WebUI import online
        self.online = online
        _online_run(self.results_root, keys=("fixed", "periodic"), n=4)

    def test_payload_is_the_summary_plus_each_scenarios_series(self):
        body = self.online.payload("SKAB", "7")
        self.assertEqual(body["summary"]["status"], "complete")
        self.assertEqual(set(body["scenarios"]), {"fixed", "periodic"})
        windows = body["scenarios"]["periodic"]["windows"]
        self.assertEqual([w["i"] for w in windows], [1, 2, 3, 4])
        self.assertIsNone(windows[2]["ensemble_score"])          # NaN written as null
        self.assertEqual(body["dataset_label"], "SKAB")

    def test_case_insensitive_and_absent(self):
        self.assertIsNotNone(self.online.payload("skab", "7"))
        self.assertIsNone(self.online.payload("SKAB", "999"))
        self.assertIsNone(self.online.payload("SMD", "7"))

    def test_scenario_filter_and_summary_only(self):
        self.assertEqual(set(self.online.payload("SKAB", "7", scenario="fixed")["scenarios"]),
                         {"fixed"})
        self.assertEqual(self.online.payload("SKAB", "7", summary_only=True)["scenarios"], {})
        self.assertEqual(self.online.payload("SKAB", "7", scenario="../x")["scenarios"], {})

    def test_only_known_scenario_files_are_opened(self):
        """The scenario list comes from disk; a name outside the vocabulary
        must not become a path."""
        path = self.results_root / "online" / "SKAB" / "7" / "summary.json"
        doc = json.loads(path.read_text())
        doc["scenario_order"] = ["../../../etc/passwd", "fixed"]
        path.write_text(json.dumps(doc))
        self.assertEqual(set(self.online.payload("SKAB", "7")["scenarios"]), {"fixed"})

    def test_info_and_known_entities(self):
        info = self.online.info("SKAB", "7")
        self.assertEqual(info["url"], "/api/online/SKAB/7")
        self.assertEqual(info["scenarios"], ["fixed", "periodic"])
        self.assertEqual(info["scenario_status"], {"fixed": "complete",
                                                   "periodic": "complete"})
        self.assertEqual(self.online.known_entities(), [("SKAB", "7")])

    def test_a_corrupt_summary_reads_as_absent(self):
        (self.results_root / "online" / "SKAB" / "7" / "summary.json").write_text("{")
        self.assertIsNone(self.online.payload("SKAB", "7"))
        self.assertIsNone(self.online.info("SKAB", "7"))


@unittest.skipUnless(HAS_FLASK, "Flask is not installed in this interpreter")
class TestOnlineRoutes(ArtifactTreeCase):

    def setUp(self):
        super().setUp()
        from WebUI import jobs, online, server
        jobs.reset_manager()
        self.jobs = jobs
        # No datasets: the feasibility check behind enable_online then cannot
        # read the entity and leaves the verdict to the run, so these tests do
        # not depend on what data the machine happens to have.
        self._saved_config = paths._CONFIG_CACHE
        paths._CONFIG_CACHE = {"dataset_path": str(self.results_root.parent / "no-data"),
                               "downsampling": None, "min_length": None}
        online.reset_cache()
        self.addCleanup(online.reset_cache)
        self.client = server.create_app(TESTING=True).test_client()

    def tearDown(self):
        paths._CONFIG_CACHE = self._saved_config
        self.jobs.reset_manager()
        super().tearDown()

    def test_endpoint_404_until_there_is_data(self):
        r = self.client.get("/api/online/SKAB/7")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.get_json()["error"], "no_online_results")

    def test_endpoint_serves_summary_and_series(self):
        _online_run(self.results_root)
        r = self.client.get("/api/online/skab/7")            # case-insensitive
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["Cache-Control"], "no-store")
        body = r.get_json()
        self.assertEqual([s["key"] for s in body["summary"]["scenarios"]],
                         ["fixed", "periodic"])
        self.assertEqual(len(body["scenarios"]["fixed"]["windows"]), 4)
        r = self.client.get("/api/online/SKAB/7?scenario=periodic")
        self.assertEqual(list(r.get_json()["scenarios"]), ["periodic"])
        r = self.client.get("/api/online/SKAB/7?summary=1")
        self.assertEqual(r.get_json()["scenarios"], {})

    def test_catalog_marks_results_with_online_data(self):
        body = self.client.get("/api/catalog").get_json()
        self.assertFalse(body["results"][0]["has_online"])
        self.assertIsNone(body["results"][0]["online"])
        self.assertEqual(body["online_results"], [])
        _online_run(self.results_root)
        # Online data with no explanations (a run without --explain) is listed too.
        _online_run(self.results_root, dataset="SMD", entity="machine-1-1")
        body = self.client.get("/api/catalog").get_json()
        self.assertTrue(body["results"][0]["has_online"])
        self.assertEqual(body["results"][0]["online"]["url"], "/api/online/SKAB/7")
        self.assertEqual([(o["dataset"], o["entity"]) for o in body["online_results"]],
                         [("SKAB", "7"), ("SMD", "machine-1-1")])

    def test_online_validation_errors(self):
        base = {"dataset": "S", "entity": "7", "enable_online": True}
        for extra, fragment in (
            ({"enable_online": "yes"}, "enable_online must be true or false"),
            ({"inject_online_regime": 1}, "inject_online_regime must be true or false"),
            ({"online_strategy": "sometimes"}, "online_strategy: unknown strategy"),
            ({"online_strategy": "fixed-random"}, "command-line baseline"),
            ({"strategy": "nope"}, "strategy: unknown strategy"),
            ({"online_strategy": 3}, "online_strategy must be one of"),
            ({"update_interval": 0}, "update_interval: must be at least 1"),
            ({"update_interval": 2.5}, "update_interval: expected a whole number"),
            ({"update_interval": True}, "update_interval: expected a whole number"),
            ({"drop_pct": 0}, "drop_pct: must be greater than 0"),
            ({"drop_pct": "x"}, "drop_pct: expected a number"),
            ({"adaptive_window": 1}, "adaptive_window: must be at least 2"),
            ({"adaptive_k": -0.5}, "adaptive_k: must not be negative"),
            ({"max_online_windows": 0}, "max_online_windows: must be at least 1"),
            ({"reopt_delay": -1}, "reopt_delay: must be 0 or more"),
            ({"reopt_delay": 1.5}, "reopt_delay: expected a whole number"),
            ({"iteration": "five"}, "iteration must be a whole number"),
        ):
            r = self.client.post("/api/runs", json={**base, **extra})
            self.assertEqual(r.status_code, 400, extra)
            self.assertIn(fragment, r.get_json()["error"], extra)

    def test_valid_online_run_previews_its_command(self):
        r = self.client.post("/api/runs", json={
            "dataset": "SKAB", "entity": "7", "enable_online": True,
            "online_strategy": "adaptive_threshold", "update_interval": 4,
            "drop_pct": 25, "adaptive_window": 6, "adaptive_k": 1.5,
            "max_online_windows": 15, "inject_online_regime": False, "dry_run": True})
        self.assertEqual(r.status_code, 200, r.get_json())
        argv = r.get_json()["argv"]
        self.assertEqual(argv[argv.index("--strategy") + 1], "adaptive_threshold")
        self.assertEqual(argv[argv.index("--max_online_windows") + 1], "15")
        self.assertNotIn("--inject_online_regime", argv)
        self.assertIsNone(self.jobs.manager().active())
        # The aliases from before the scenarios still validate.
        for legacy in ("adaptive", "fixed-best", "all", "fixed", "periodic", "threshold"):
            r = self.client.post("/api/runs", json={"dataset": "SKAB", "entity": "7",
                                                    "enable_online": True,
                                                    "strategy": legacy, "dry_run": True})
            self.assertEqual(r.status_code, 200, legacy)


# ── live monitor stream ─────────────────────────────────────────────────────

def _stream_run(results_root: Path, dataset="TSB", entity="e1", injected=False, n_channels=2):
    """Real StreamWriter + the real runner (fake scores with per-sample traces)
    into `results_root`: 400 offline samples, a 300-sample online part with
    one real anomaly (150..159), windows of 100 every 10, scenario fixed."""
    import numpy as np
    from Model_Selection import online_scoring as sc
    from Model_Selection.online_phase import (OnlineResultsWriter, Selection, WindowScores,
                                              run_online)
    from Model_Selection.online_policy import PolicyParams
    from Model_Selection.online_stream import StreamWriter
    rng = np.random.default_rng(0)
    off_Y = rng.normal(size=(n_channels, 400))
    on_Y = rng.normal(size=(n_channels, 300))
    on_Y[0, 200] = 25.0                                  # a one-sample spike
    labels = np.zeros(300)
    labels[150:160] = 1
    stream = (sc.build_online_stream(on_Y, labels, None, ["cutoff"], None, 13) if injected
              else sc.real_label_stream(on_Y, labels, None))
    windows = sc.build_online_windows(stream, list(range(21)), offset=400, step_size=10,
                                      window_size=100)
    directory = results_root / "online" / dataset / entity
    sw = StreamWriter(str(directory), dataset=dataset, entity=entity,
                      label_source="injected" if injected else "real",
                      offline_Y=off_Y, offline_real_labels=np.zeros(400),
                      online_copies=stream.copies, online_real_labels=labels,
                      offline_size=400, window_size=100, step_size=10, n_windows=21,
                      injected=injected)

    def score(window, selection):
        n = window.data.entities[0].Y.shape[1]
        values = np.linspace(0, 1, n)
        cut = None if not window.scorable else 0.9
        return WindowScores(
            ensemble=0.5 if window.scorable else None, single=0.4 if window.scorable else None,
            trace={"single": {"scores": values, "threshold": cut,
                              "flagged": None if cut is None else (values >= cut).astype("int8")}})

    writer = OnlineResultsWriter(str(directory), {"dataset": dataset, "entity": entity,
                                                  "strategy": "fixed"}, ("fixed",))
    run_online(("fixed",), PolicyParams(), windows, lambda k: Selection(["A"], "A"),
               score, None, writer, on_window=sw.record)
    return directory, on_Y, labels


class TestStreamDecimation(unittest.TestCase):

    def test_short_ranges_come_back_sample_by_sample(self):
        import numpy as np
        from WebUI import stream
        out = stream.minmax(np.arange(10.0), 2, 7, 100)
        self.assertEqual((out["x"], out["min"], out["max"], out["bucket"]),
                         ([2, 3, 4, 5, 6], [2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 4.0, 5.0, 6.0], 1))

    def test_buckets_keep_every_spike(self):
        import numpy as np
        from WebUI import stream
        values = np.zeros(100000)
        values[54321] = 9.0
        values[777] = -4.0
        out = stream.minmax(values, 0, 100000, 1000)
        self.assertLessEqual(len(out["x"]), 500)
        self.assertEqual(max(out["max"]), 9.0)
        self.assertEqual(min(out["min"]), -4.0)
        b = out["bucket"]
        self.assertEqual(out["max"][54321 // b], 9.0)

    def test_spans_are_exact_and_clipped(self):
        import numpy as np
        from WebUI import stream
        labels = np.zeros(50)
        labels[3] = 1
        labels[10:20] = 1
        self.assertEqual(stream.spans(labels, 0, 50), [[3, 4], [10, 20]])
        self.assertEqual(stream.spans(labels, 12, 50), [[12, 20]])
        self.assertEqual(stream.spans(labels, 400, 450, offset=400), [[403, 404], [410, 420]])
        self.assertEqual(stream.spans(None, 0, 5), [])

    def test_channels(self):
        from WebUI import stream
        self.assertEqual(stream.parse_channels(None, 3), [0])
        self.assertEqual(stream.parse_channels("2,0,2", 3), [2, 0])
        self.assertEqual(stream.parse_channels("all", 40), list(range(stream.MAX_CHANNELS)))
        for bad in ("3", "-1", "x"):
            with self.assertRaises(ValueError):
                stream.parse_channels(bad, 3)


@unittest.skipUnless(HAS_FLASK, "Flask is not installed in this interpreter")
class TestStreamRoute(ArtifactTreeCase):

    def setUp(self):
        super().setUp()
        from WebUI import jobs, server
        jobs.reset_manager()
        self.addCleanup(jobs.reset_manager)
        self.client = server.create_app(TESTING=True).test_client()

    def get(self, query="", entity="e1"):
        return self.client.get(f"/api/online/TSB/{entity}/stream{query}")

    def test_404_until_there_is_a_stream(self):
        r = self.get()
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.get_json()["error"], "no_stream")

    def test_series_labels_and_boundary(self):
        import numpy as np
        _, on_Y, labels = _stream_run(self.results_root)
        r = self.get("?channels=0,1&from=380&to=700")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["Cache-Control"], "no-store")
        body = r.get_json()
        meta = body["meta"]
        self.assertEqual((meta["offline_size"], meta["total_length"], meta["window_size"],
                          meta["step_size"], meta["label_source"]), (400, 700, 100, 10, "real"))
        self.assertTrue(meta["has_real_labels"])
        self.assertFalse(meta["has_injected_labels"])
        self.assertEqual([s["channel"] for s in body["series"]], [0, 1])
        ch0 = body["series"][0]
        self.assertEqual(ch0["bucket"], 1)
        # The online part as replayed is the raw test slice (nothing injected).
        np.testing.assert_allclose(ch0["min"][20:], np.round(on_Y[0], 6), atol=1e-5)
        self.assertEqual(body["real_spans"], [[550, 560]])
        self.assertEqual(body["injected_spans"], [])

    def test_decimated_whole_stream_keeps_the_spike_and_the_span(self):
        _stream_run(self.results_root)
        body = self.get("?max_points=40").get_json()
        ch0 = body["series"][0]
        self.assertGreater(ch0["bucket"], 1)
        self.assertLessEqual(len(ch0["x"]), 20)
        self.assertAlmostEqual(max(ch0["max"]), 25.0, places=4)
        self.assertEqual(body["real_spans"], [[550, 560]])

    def test_a_scenario_window_its_scores_threshold_and_flags(self):
        _stream_run(self.results_root)
        body = self.get("?scenario=fixed&window=6&series=0").get_json()
        self.assertNotIn("series", body)
        self.assertEqual(body["recorded_windows"], list(range(21)))
        w = body["window"]
        self.assertEqual((w["i"], w["start"], w["end"]), (6, 460, 560))
        self.assertEqual(w["single"]["threshold"], 0.9)
        self.assertEqual(len(w["single"]["x"]), 100)
        self.assertEqual(w["single"]["flagged"], [[550, 560]])
        self.assertEqual(w["single"]["n_flagged"], 10)
        self.assertIsNone(w["ensemble"])
        # An unscorable window keeps its scores, with no threshold or flags.
        w = self.get("?scenario=fixed&window=0&series=0").get_json()["window"]
        self.assertIsNone(w["single"]["threshold"])
        self.assertEqual(w["single"]["flagged"], [])
        self.assertEqual(len(w["single"]["x"]), 100)
        # Default: the latest written window.
        self.assertEqual(self.get("?scenario=fixed&series=0").get_json()["window"]["i"], 20)

    def test_injected_copies_are_named_and_their_spans_exact(self):
        _stream_run(self.results_root, entity="inj", injected=True)
        body = self.get("?copy=1", entity="inj").get_json()
        self.assertEqual([c["anomaly_type"] for c in body["meta"]["copies"]], ["cutoff", "cutoff"])
        self.assertTrue(body["meta"]["has_injected_labels"])
        self.assertTrue(body["injected_spans"])
        self.assertTrue(all(len(s) == 3 and s[2] == "cutoff" and 400 <= s[0] < s[1] <= 700
                            for s in body["injected_spans"]))
        self.assertEqual(body["real_spans"], [[550, 560]])

    def test_bad_requests(self):
        _stream_run(self.results_root)
        for query in ("?channels=5", "?from=600&to=500", "?copy=3", "?scenario=nope",
                      "?max_points=x", "?window=1.5&scenario=fixed"):
            r = self.get(query)
            self.assertEqual(r.status_code, 400, query)


# ── online feasibility, liveness and waiting ─────────────────────────────────

def _presplit_entity(root: Path, dataset: str, stem: str, n_test: int, n_train=300,
                     online_anomaly=True):
    """A two-channel entity in the pre-split layout Datasets.load reads (one
    row per timestep; a single column would load as one row). Its labels hold
    an anomaly in the middle (the offline part) and, with `online_anomaly`,
    one at 90% (the online part), where real-label mode scores."""
    import numpy as np
    base = root / dataset
    for sub in ("train", "test", "test_label"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    t = np.arange(max(n_test, n_train))
    values = np.column_stack([np.sin(t / 7.0), np.cos(t / 5.0)])
    np.savetxt(base / "train" / f"{stem}.txt", values[:n_train], delimiter=",")
    np.savetxt(base / "test" / f"{stem}.txt", values[:n_test], delimiter=",")
    labels = np.zeros(n_test)
    labels[n_test // 2:n_test // 2 + 5] = 1
    if online_anomaly:
        labels[int(n_test * 0.9):int(n_test * 0.9) + 5] = 1
    np.savetxt(base / "test_label" / f"{stem}.txt", labels)


class _FeasibilityCase(unittest.TestCase):
    """A data root of pre-split TOY entities: short (no window fits), long
    (an anomaly in the online part) and quiet (none there)."""

    def setUp(self):
        from WebUI import jobs, online, server
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        _presplit_entity(root, "TOY", "short", 230)     # like a SKAB entity
        _presplit_entity(root, "TOY", "long", 1180)     # like UCR 028 downsampled
        _presplit_entity(root, "TOY", "quiet", 1180, online_anomaly=False)
        self._saved = paths._CONFIG_CACHE
        paths._CONFIG_CACHE = {"dataset_path": str(root), "downsampling": None,
                               "min_length": None}
        self.addCleanup(setattr, paths, "_CONFIG_CACHE", self._saved)
        online.reset_cache()
        self.addCleanup(online.reset_cache)
        jobs.reset_manager()
        self.addCleanup(jobs.reset_manager)
        self.online = online
        self.client = server.create_app(TESTING=True).test_client()


@unittest.skipUnless(HAS_FLASK, "Flask is not installed in this interpreter")
class TestOnlineFeasibility(_FeasibilityCase):
    """GET /api/online/feasibility and the enable_online check behind it: the
    pipeline's own split and window functions on the length it would load."""

    def test_a_short_entity_is_not_feasible_and_says_why(self):
        r = self.client.get("/api/online/feasibility/TOY/short")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertFalse(body["ok"])
        self.assertEqual((body["n_windows"], body["window_size"], body["step_size"],
                          body["online_size"]), (0, 100, 5, 46))
        self.assertIn("46 samples", body["reason"])

    def test_a_long_entity_is_feasible(self):
        body = self.client.get("/api/online/feasibility/TOY/long").get_json()
        self.assertTrue(body["ok"])
        self.assertIsNone(body["reason"])
        self.assertEqual((body["n_windows"], body["window_size"], body["step_size"],
                          body["online_size"], body["total_samples"]),
                         (28, 100, 5, 236, 1180))
        for key in ("ok", "n_windows", "window_size", "step_size", "online_size", "reason"):
            self.assertIn(key, body)

    def test_iteration_sizes_the_windows(self):
        body = self.client.get("/api/online/feasibility/TOY/long?iteration=2").get_json()
        self.assertEqual((body["iteration"], body["window_size"], body["step_size"]),
                         (2, 118, 5))
        r = self.client.get("/api/online/feasibility/TOY/long?iteration=two")
        self.assertEqual(r.status_code, 400)

    def test_an_unreadable_entity_is_404(self):
        r = self.client.get("/api/online/feasibility/TOY/missing")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.get_json()["error"], "unknown_entity")

    def test_the_length_is_read_once(self):
        from unittest import mock
        import Datasets.load as load
        with mock.patch.object(load, "load_data", wraps=load.load_data) as spy:
            for _ in range(3):
                self.client.get("/api/online/feasibility/TOY/long?iteration=5")
            self.client.get("/api/online/feasibility/TOY/long?iteration=3")
        self.assertEqual(spy.call_count, 1)

    def test_it_agrees_with_the_pipeline_windowing(self):
        from Utils.pipeline_spec import count_windows, online_window_geometry
        body = self.online.feasibility("TOY", "long", 5)
        window, step = online_window_geometry(236, 5)
        self.assertEqual(body["n_windows"], count_windows(236, window, step))

    def test_validation_refuses_enable_online_for_an_infeasible_entity(self):
        r = self.client.post("/api/runs", json={"dataset": "TOY", "entity": "short",
                                                "enable_online": True, "dry_run": True})
        self.assertEqual(r.status_code, 400)
        self.assertIn("enable_online:", r.get_json()["error"])
        self.assertIn("46 samples", r.get_json()["error"])
        # Without the online phase the same entity is fine.
        r = self.client.post("/api/runs", json={"dataset": "TOY", "entity": "short",
                                                "enable_online": False, "dry_run": True})
        self.assertEqual(r.status_code, 200)
        r = self.client.post("/api/runs", json={"dataset": "TOY", "entity": "long",
                                                "enable_online": True, "dry_run": True})
        self.assertEqual(r.status_code, 200, r.get_json())
        # Unreadable: left to the run, which reports why.
        r = self.client.post("/api/runs", json={"dataset": "TOY", "entity": "missing",
                                                "enable_online": True, "dry_run": True})
        self.assertEqual(r.status_code, 200)


@unittest.skipUnless(HAS_FLASK, "Flask is not installed in this interpreter")
class TestRealLabelFeasibility(_FeasibilityCase):
    """Real-label mode: how many online windows hold a labelled anomaly, and
    no online phase where none does."""

    def test_real_labels_are_the_default_where_the_data_has_them(self):
        body = self.client.get("/api/online/feasibility/TOY/long").get_json()
        self.assertTrue(body["labels_available"])
        self.assertEqual((body["label_source"], body["default_label_source"]),
                         ("real", "real"))
        self.assertEqual(body["label_source_label"], "Real labels — no injection online")
        # The online anomaly is at 1062 = online position 118; window i covers
        # [5i, 5i + 100), so windows 4..24 hold it.
        self.assertEqual((body["n_windows_with_anomaly"], body["n_windows_scorable"],
                          body["n_windows"]), (21, 21, 28))
        self.assertIsNone(body["label_error"])

    def test_an_online_part_without_a_labelled_anomaly_has_nothing_to_score(self):
        body = self.client.get("/api/online/feasibility/TOY/quiet").get_json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["n_windows_with_anomaly"], 0)
        self.assertIn("No labelled anomaly in the online part", body["reason"])
        self.assertIn("nothing to score", body["reason"])
        # Injected anomalies still run it.
        body = self.client.get(
            "/api/online/feasibility/TOY/quiet?online_labels=injected").get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["label_source"], "injected")
        self.assertIsNone(body["n_windows_with_anomaly"])

    def test_real_labels_with_the_regime_shift_are_a_label_error(self):
        body = self.client.get(
            "/api/online/feasibility/TOY/long?inject_online_regime=1").get_json()
        self.assertIsNone(body["label_source"])
        self.assertIn("inject_online_regime", body["label_error"])
        r = self.client.get("/api/online/feasibility/TOY/long?online_labels=maybe")
        self.assertEqual(r.status_code, 400)

    def test_validation_follows_the_label_source(self):
        post = lambda **kw: self.client.post("/api/runs", json={
            "dataset": "TOY", "enable_online": True, "dry_run": True, **kw})
        r = post(entity="quiet")
        self.assertEqual(r.status_code, 400)
        self.assertIn("No labelled anomaly in the online part", r.get_json()["error"])
        r = post(entity="quiet", online_labels="injected")
        self.assertEqual(r.status_code, 200, r.get_json())
        argv = r.get_json()["argv"]
        self.assertEqual(argv[argv.index("--online_labels") + 1], "injected")
        r = post(entity="long", online_labels="real", inject_online_regime=True)
        self.assertEqual(r.status_code, 400)
        self.assertIn("online_labels:", r.get_json()["error"])
        r = post(entity="long", online_labels="real")
        self.assertEqual(r.status_code, 200)
        argv = r.get_json()["argv"]
        self.assertEqual(argv[argv.index("--online_labels") + 1], "real")
        r = post(entity="long", online_labels="bogus")
        self.assertEqual(r.status_code, 400)
        self.assertIn("online_labels: expected real or injected", r.get_json()["error"])


class TestOnlineLiveness(ArtifactTreeCase):
    """A summary saying "running" whose writer is gone reads (and is marked)
    "interrupted"; one that is still being written does not."""

    def setUp(self):
        super().setUp()
        from WebUI import online
        self.online = online
        self.path = self.results_root / "online" / "SKAB" / "7" / "summary.json"

    def _running(self, **changes):
        from Model_Selection.online_phase import OnlineResultsWriter, scenario_params
        from Model_Selection.online_policy import PolicyParams
        keys = ("fixed", "periodic")
        writer = OnlineResultsWriter(str(self.path.parent), {"strategy": "all"}, keys,
                                     scenario_params=scenario_params(keys, PolicyParams()),
                                     n_windows_planned=4)
        writer.recorder("fixed", {}, 4).flush()         # one scenario under way
        doc = json.loads(self.path.read_text())
        doc.update(changes)
        self.path.write_text(json.dumps(doc))
        return doc

    @staticmethod
    def _dead_pid():
        import subprocess
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        return proc.pid

    def test_a_live_writer_keeps_its_status(self):
        self._running()                               # pid = this process
        self.assertEqual(self.online.payload("SKAB", "7")["summary"]["status"], "running")

    def test_a_dead_writer_is_interrupted_and_marked_on_disk(self):
        self._running(pid=self._dead_pid())
        body = self.online.payload("SKAB", "7")
        self.assertEqual(body["summary"]["status"], "interrupted")
        self.assertIn("stopped before the online phase finished", body["summary"]["error"])
        self.assertEqual({r["key"]: r["status"] for r in body["summary"]["scenarios"]},
                         {"fixed": "interrupted", "periodic": "interrupted"})
        self.assertEqual(body["scenarios"]["fixed"]["status"], "interrupted")
        on_disk = json.loads(self.path.read_text())
        self.assertEqual(on_disk["status"], "interrupted")
        self.assertEqual(self.online.info("SKAB", "7")["status"], "interrupted")

    def test_a_live_job_owning_the_entity_is_never_judged(self):
        self._running(pid=self._dead_pid())
        body = self.online.payload("SKAB", "7", is_live=lambda d, e: True)
        self.assertEqual(body["summary"]["status"], "running")

    def test_without_a_process_of_record_only_age_counts(self):
        import time as _time
        self._running(pid=None, updated_at=_time.time() - 60)
        self.assertEqual(self.online.info("SKAB", "7")["status"], "running")
        self._running(pid=None, updated_at=_time.time() - self.online.STALE_SECONDS - 60)
        self.assertEqual(self.online.info("SKAB", "7")["status"], "interrupted")

    def test_a_stale_summary_is_interrupted_even_if_its_pid_is_alive(self):
        """The id may belong to another process by now, or the run hung."""
        import time as _time
        self._running(updated_at=_time.time() - self.online.STALE_SECONDS - 60)
        self.assertEqual(self.online.info("SKAB", "7")["status"], "interrupted")

    def test_finished_summaries_are_left_alone(self):
        self._running(pid=self._dead_pid(), status="complete")
        self.assertEqual(self.online.info("SKAB", "7")["status"], "complete")

    def test_mark_interrupted_covers_a_waiting_scenario(self):
        doc = self._running()
        doc["scenarios"][0]["status"] = "waiting_for_reopt"
        doc["scenarios"][0]["waiting"] = {"since_window": 2, "until_window": 4, "since": 1}
        self.path.write_text(json.dumps(doc))
        self.assertTrue(self.online.mark_interrupted("SKAB", "7", since=0, reason="x"))
        row = json.loads(self.path.read_text())["scenarios"][0]
        self.assertEqual((row["status"], row["waiting"]), ("interrupted", None))


class TestOnlineWaitingTracking(unittest.TestCase):
    """The run page tells a stream waiting for its re-optimisation from a
    stalled one, and counts what did not end in a switch."""

    FULL = "INFO:app:🎉 EXECUTION COMPLETE! Total Time: 1.0s"
    START = "INFO:x:ONLINE PHASE START: strategy=all, scenarios=periodic, windows=8"

    def setUp(self):
        from WebUI import jobs
        self.jobs = jobs

    def _job(self, lines):
        job = self.jobs.Job("t", ["x"], {"enable_online": True})
        for line in lines:
            job._record(line)
        job.exit_code = 0
        return job

    def test_waiting_is_visible_and_cleared_by_the_switch(self):
        lines = [self.FULL, self.START,
                 "INFO:x:ONLINE WINDOW: scenario=periodic, window=3/8, index=2, trigger=started",
                 "INFO:x:ONLINE REOPT: scenario=periodic, event=waiting, window=4, "
                 "started_at_window=2"]
        job = self._job(lines)
        stage = job.stages["online_periodic"]
        self.assertEqual(stage["status"], "running")
        self.assertEqual(stage["waiting"], {"since_window": 2, "until_window": 4})
        self.assertIn("waiting for the re-optimization started at window 2", stage["text"])
        self.assertEqual(job.online["scenarios"]["periodic"]["status"], "waiting")
        job._record("INFO:x:ONLINE REOPT: scenario=periodic, event=applied, window=4, "
                    "started_at_window=2, duration_s=80.0, waited_s=20.0, ensemble=A, single=A")
        stage = job.stages["online_periodic"]
        self.assertIsNone(stage["waiting"])
        self.assertNotIn("waiting", stage["text"])
        self.assertEqual(job.online["scenarios"]["periodic"]["reopts_applied"], 1)

    def test_not_applied_and_aborted_are_counted(self):
        job = self._job([self.FULL, self.START,
                         "INFO:x:ONLINE REOPT: scenario=periodic, event=not_applied, "
                         "started_at_window=7, apply_window=9, completed=no, "
                         "reason=stream_ended",
                         "ERROR:x:ONLINE REOPT: scenario=periodic, event=aborted, window=3, "
                         "started_at_window=2, error=RuntimeError: x"])
        prog = job.online["scenarios"]["periodic"]
        self.assertEqual((prog["not_applied"], prog["aborted"]), (1, 1))

    def test_timed_out_reoptimizations_make_a_warning(self):
        job = self._job([self.FULL, self.START,
                         "ERROR:x:ONLINE REOPT: scenario=periodic, event=failed, window=4, "
                         "started_at_window=2, timed_out=yes, error=ReoptTimedOut: timeout",
                         "INFO:x:ONLINE SCENARIO COMPLETE: scenario=periodic, windows=8",
                         "INFO:x:ONLINE PHASE COMPLETE: scenarios=1, failed=0, wall_s=1.0"])
        out = self.jobs.classify_outcome(job, False)
        self.assertEqual(out["status"], "succeeded_with_warnings")
        self.assertIn("1 re-optimization(s) failed or timed out in periodic", out["reason"])

    def test_the_real_runner_lines_with_a_wait_parse(self):
        """The runner's own lines for a wait and an end-of-stream drop."""
        from Model_Selection.online_phase import (OnlineResultsWriter, OnlineWindow,
                                                  Selection, WindowScores, run_online)
        from Model_Selection.online_policy import PolicyParams

        class Slow:
            calls = 0

            def done(self):
                return Slow.calls >= 2

            def wait(self, seconds):
                Slow.calls += 1
                return self.done()

            def result(self):
                return Selection(["B"], "B", origin="reoptimization")

            def kill(self):
                pass

            def close(self):
                pass

            duration_s, reused_from = 3.0, None

        log = _CaptureLog()
        with tempfile.TemporaryDirectory() as tmp:
            writer = OnlineResultsWriter(tmp, {"strategy": "periodic"}, ("periodic",))
            run_online(("periodic",), PolicyParams(update_interval=2, reopt_delay=2),
                       [OnlineWindow(index=i, start=i, end=i + 1) for i in range(5)],
                       lambda k: Selection(["A"], "A"),
                       lambda w, s: WindowScores(ensemble=0.5, single=0.5),
                       lambda w, s, k: Slow(), writer, log=log, heartbeat_s=0)
        job = self._job([self.FULL] + log.lines)
        prog = job.online["scenarios"]["periodic"]
        self.assertEqual((prog["status"], prog["reopts_started"], prog["reopts_applied"],
                          prog["not_applied"], prog["waiting"]), ("done", 2, 1, 1, None))
        self.assertEqual(self.jobs.classify_outcome(job, False)["status"], "succeeded")


class TestAnomalyTypeCopies(unittest.TestCase):
    """The configure page tells the user what each extra anomaly type costs;
    the count must be the number of copies `Inject` really makes."""

    def test_copies_match_the_parameter_grid(self):
        from Model_Selection.anomaly_parameters import ANOMALY_PARAM_GRID
        from Model_Selection.inject_anomalies import inject_copies
        import numpy as np
        series = np.sin(np.linspace(0, 20, 400))[None, :]
        for token in ("spikes", "scale", "cutoff"):
            with self.subTest(token=token):
                np.random.seed(0)
                self.assertEqual(catalog.anomaly_copies(token),
                                 len(inject_copies(series, [token], rate=0.1)))
        self.assertEqual(catalog.anomaly_copies("spikes"), 1)
        self.assertEqual(set(catalog.anomaly_copies(t) for t in ANOMALY_PARAM_GRID) - {1, 2, 4}, set())
