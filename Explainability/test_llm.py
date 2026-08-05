"""
Standalone unit tests for the LLM narration layer (Explainability/llm.py) and
the atom-matching faithfulness verifier (Explainability/verifier.py).
No network, no server: the client is exercised through an injected transport
and a FakeClient that echoes the prompt's fact sentences (a "perfect-copy"
model, which must score 0 hallucination / 0 omission).
"""

import importlib.util
import json
import os
import re
import tempfile
import unittest

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_THIS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ir = _load("ir")
verifier = _load("verifier")
llm = _load("llm")


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _ga_combination_result():
    return {
        "best_ensemble": ["A", "B", "C"], "feature_names": ["A", "B", "C"],
        "meta_model_type": "rf", "model_source": "captured", "baseline_f1": 0.87,
        "shap_importance": {"A": 0.4, "B": 0.2, "C": 0.1},
        "shap_signed_importance": {"A": 0.35, "B": 0.15, "C": -0.05},
        "pfi_importance": {"A": 0.2, "B": 0.05, "C": 0.08},
        "markov_scores": {"A": 0.5, "B": 0.3, "C": 0.2},
        "final_ranking": ["A", "B", "C"],
    }


def _mc_result():
    curves = {
        "grid_levels": np.array([0.0, 0.1, 0.2]),
        "win_regions": {"LOF_1": [(0.0, 0.1)], "NN_3": [(0.2, 0.2)]},
        "crossovers": [{"noise": 0.2, "from_model": "LOF_1", "to_model": "NN_3"}],
        "breakdown_points": {},
    }
    return {
        "curves_f1": curves, "curves_pr": curves, "curves_f1_fixed": curves,
        "winner_f1": {"feasible": True, "train_accuracy": 0.95, "cv_accuracy": 0.85,
                      "win_rates": {"LOF_1": 0.6, "NN_3": 0.4},
                      "rules": [{"conditions": [{"feature": "noise_level", "op": "<=",
                                                 "threshold": 0.15}],
                                 "outcome": "LOF_1", "n_samples": 10}],
                      "rules_text": "", "classes": ["LOF_1", "NN_3"],
                      "root_threshold": 0.15},
        "winner_pr": {"feasible": False},
        "permodel_f1": {"LOF_1": {"cv_r2": 0.7}}, "permodel_pr": {},
        "n_trials": 30,
    }


def _results_dict():
    return {
        "thompson": {"best_model": "LOF_1"},
        "gan_robustness": {"best_model": "LOF_1"},
        "borderline": {"best_model": "NN_3"},
        "monte_carlo": {"best_model_f1": "LOF_1"},
        "aggregation": {"robust_agg": (0.5, ["LOF_1", "NN_3"]),
                        "final_agg": (0.4, ["LOF_1", "NN_3"])},
        "final_decision": {"framework_choice": "single_model",
                           "chosen_model": "LOF_1", "ensemble": ["A", "B"],
                           "ensemble_f1": 0.8, "ensemble_pr_auc": 0.7,
                           "single_model": "LOF_1", "single_model_f1": 0.85,
                           "single_model_pr_auc": 0.75},
    }


def _tiny_ir():
    """Minimal hand-made stage IR with one required numeric atom."""
    return {
        "ir_version": "1.0", "stage": "toy", "dataset": "DS", "entity": "e1",
        "output": {"top_pick": "LOF_1"},
        "evidence": [
            {"id": "toy.score", "type": "t", "subject": "LOF_1",
             "value": 0.287, "text": "LOF_1 achieves a score of 0.287."},
            {"id": "toy.other", "type": "t", "subject": "NN_3",
             "value": 0.1, "text": "NN_3 achieves a score of 0.100."},
        ],
        "caveats": [{"id": "toy.caveat", "type": "caveat", "subject": "x",
                     "value": None, "text": "Scores are proxies."}],
        "required_atom_ids": ["toy.score"],
        "confidence": {},
    }


class FakeClient:
    """Perfect-copy model: echoes the prompt's grounded content verbatim."""
    model = "fake"

    def chat(self, system, user):
        out = []
        for line in user.splitlines():
            m = re.match(r"^\d+\.\s+(?:\[REQUIRED\]\s+)?(.*)$", line)
            if m:
                out.append(m.group(1))
                continue
            if line.startswith("- ") and ":" in line:
                out.append(line[2:].replace("[CAVEAT] ", ""))
        return " ".join(out)


# ════════════════════════════════════════════════════════════════════════════

class TestExtractNumbers(unittest.TestCase):

    def test_basic_and_percent(self):
        nums = verifier.extract_numbers("gap 0.287, share 62.5% and -0.05.")
        vals = [v for _, v in nums]
        self.assertEqual(vals, [0.287, 62.5, -0.05])

    def test_identifier_digits_excluded(self):
        self.assertEqual(verifier.extract_numbers("LOF_1 on machine-1-6"), [])

    def test_sentence_final_period(self):
        self.assertEqual([v for _, v in verifier.extract_numbers("score 0.5000.")],
                         [0.5])

    def test_digit_ordinals(self):
        self.assertEqual([v for _, v in verifier.extract_numbers("3rd and 6th and 21st")],
                         [3.0, 6.0, 21.0])

    def test_spelled_numbers(self):
        vals = [v for _, v in verifier.extract_numbers(
            "Six sources ranked first, then sixth and twentieth")]
        self.assertEqual(sorted(vals), [1.0, 6.0, 6.0, 20.0])

    def test_ambiguous_one_excluded(self):
        # "one"/"zero" cardinals are articles/pronouns here — not numeric claims.
        self.assertEqual(verifier.extract_numbers("one of the sources, leaving one out"), [])
        self.assertEqual(verifier.extract_numbers("a single source"), [])


class TestVerifier(unittest.TestCase):

    def test_faithful_narrative_scores_zero(self):
        doc = _tiny_ir()
        narrative = "LOF_1 achieves a score of 0.287, while NN_3 reaches 0.100."
        v = verifier.verify_narrative(narrative, doc)
        self.assertEqual(v["hallucination_rate"], 0.0)
        self.assertEqual(v["omission_rate"], 0.0)
        self.assertEqual(v["unsupported_numbers"], [])
        self.assertEqual(v["unsupported_entities"], [])

    def test_alien_number_is_hallucination(self):
        doc = _tiny_ir()
        v = verifier.verify_narrative("LOF_1 achieves a score of 0.287 or maybe 0.912.", doc)
        self.assertIn("0.912", v["unsupported_numbers"])
        self.assertGreater(v["hallucination_rate"], 0.0)

    def test_rounded_number_is_not_hallucination(self):
        doc = _tiny_ir()
        v = verifier.verify_narrative("LOF_1 achieves a score of 0.29.", doc)
        self.assertEqual(v["unsupported_numbers"], [])
        self.assertEqual(v["rounded_matches"], ["0.29"])
        self.assertEqual(v["hallucination_rate"], 0.0)
        # Coverage accepts the rounded number too → no omission.
        self.assertEqual(v["omission_rate"], 0.0)

    def test_missing_required_atom_is_omission(self):
        doc = _tiny_ir()
        v = verifier.verify_narrative("NN_3 achieves a score of 0.100.", doc)
        self.assertEqual(v["missing_required_ids"], ["toy.score"])
        self.assertEqual(v["omission_rate"], 1.0)

    def test_alien_entity_is_hallucination(self):
        doc = _tiny_ir()
        v = verifier.verify_narrative("LOF_1 achieves a score of 0.287; XYZ_9 wins.", doc)
        self.assertEqual(v["unsupported_entities"], ["XYZ_9"])
        self.assertGreater(v["hallucination_rate"], 0.0)

    def test_empty_narrative(self):
        doc = _tiny_ir()
        v = verifier.verify_narrative("", doc)
        self.assertEqual(v["omission_rate"], 1.0)
        self.assertEqual(v["n_claims"], 0)
        self.assertEqual(v["hallucination_rate"], 0.0)

    def test_verify_on_real_builder_output(self):
        doc = ir.build_ga_combination_ir("DS", "e1", _ga_combination_result())
        narrative = " ".join(a["text"] for a in doc["evidence"])
        v = verifier.verify_narrative(narrative, doc)
        self.assertEqual(v["hallucination_rate"], 0.0)
        self.assertEqual(v["omission_rate"], 0.0)

    def test_ordinal_conveys_required_number_no_omission(self):
        # Atom's number is 3 (digit); the narrative writes the readable "3rd".
        doc = {"ir_version": "1.0", "stage": "toy", "dataset": "D", "entity": "e",
               "output": {}, "caveats": [], "required_atom_ids": ["r"],
               "evidence": [{"id": "r", "type": "t", "subject": "LOF_1", "value": 3,
                             "text": "LOF_1 ranked 3 in influence."}]}
        v = verifier.verify_narrative("LOF_1 came 3rd in influence.", doc)
        self.assertEqual(v["omission_rate"], 0.0)
        self.assertEqual(v["missing_required_ids"], [])

    def test_spelled_number_symmetric_hallucination(self):
        # "fifth" (=5) is not an allowed number → flagged, same as a bad digit.
        doc = _tiny_ir()  # allowed numbers: 0.287, 0.1
        v = verifier.verify_narrative("LOF_1 achieves a score of 0.287, ranked fifth.", doc)
        self.assertIn("fifth", v["unsupported_numbers"])
        self.assertGreater(v["hallucination_rate"], 0.0)


def _archetype_ir():
    """Two member cards with opposite archetypes for attribution tests."""
    return {
        "ir_version": "1.0", "stage": "toy", "dataset": "DS", "entity": "e1",
        "output": {},
        "evidence": [
            {"id": "c.A", "type": "member_card", "subject": "LOF_1",
             "value": {"archetype": "LH", "utility": 0.1},
             "text": "LOF_1: archetype LH (low utility, high stability); "
                     "mean marginal contribution 0.1."},
            {"id": "c.B", "type": "member_card", "subject": "NN_3",
             "value": {"archetype": "HH", "utility": 0.4},
             "text": "NN_3: archetype HH (high utility, high stability); "
                     "mean marginal contribution 0.4."},
        ],
        "caveats": [], "required_atom_ids": [], "confidence": {},
    }


class TestVerifierAttribution(unittest.TestCase):
    """Sentence-scoped attribution (verifier v2)."""

    def test_misattributed_number_counts_as_hallucination(self):
        # 0.100 exists in the IR but belongs to NN_3; the sentence names only
        # LOF_1 → factually wrong statement built from a true value.
        v = verifier.verify_narrative("LOF_1 achieves a score of 0.100.", _tiny_ir())
        self.assertEqual(len(v["misattributed_numbers"]), 1)
        self.assertEqual(v["misattributed_numbers"][0]["number"], "0.100")
        self.assertEqual(v["misattributed_numbers"][0]["subjects"], ["lof_1"])
        self.assertGreater(v["hallucination_rate"], 0.0)
        # It is NOT double-counted as an unsupported number.
        self.assertEqual(v["unsupported_numbers"], [])

    def test_correctly_attributed_numbers_pass(self):
        v = verifier.verify_narrative(
            "LOF_1 achieves a score of 0.287, while NN_3 reaches 0.100.", _tiny_ir())
        self.assertEqual(v["misattributed_numbers"], [])
        self.assertEqual(v["hallucination_rate"], 0.0)

    def test_stage_level_numbers_allowed_next_to_any_detector(self):
        doc = _tiny_ir()
        doc["output"]["n_points"] = 40
        v = verifier.verify_narrative(
            "LOF_1 achieves a score of 0.287 across 40 points.", doc)
        self.assertEqual(v["misattributed_numbers"], [])

    def test_wrong_archetype_claim_is_warned(self):
        v = verifier.verify_narrative(
            "NN_3: archetype HH (high utility, high stability); mean marginal "
            "contribution 0.4. LOF_1 was classified as high utility with high "
            "stability.", _archetype_ir())
        warns = v["attribution_warnings"]
        self.assertEqual(len(warns), 1)
        self.assertEqual(warns[0]["subject"], "lof_1")
        self.assertEqual(warns[0]["aspect"], "utility")
        self.assertEqual(warns[0]["actual"], "L")
        # Warnings are diagnostic only — the headline rate is untouched.
        self.assertEqual(v["hallucination_rate"], 0.0)

    def test_contrast_sentence_does_not_warn(self):
        v = verifier.verify_narrative(
            "LOF_1 shows low utility and high stability while NN_3 shows high "
            "utility and high stability.", _archetype_ir())
        self.assertEqual(v["attribution_warnings"], [])

    def test_order_field_never_enters_allowed_numbers(self):
        doc = _tiny_ir()
        doc["evidence"][0]["order"] = 42
        v = verifier.verify_narrative(
            "LOF_1 achieves a score of 0.287 and 42 extras.", doc)
        self.assertIn("42", v["unsupported_numbers"])

    def test_exact_ownership_elsewhere_trumps_rounding_coincidence(self):
        # 0.104 belongs exactly to NN_3; next to LOF_1 it must be flagged even
        # though it 2dp-rounds onto LOF_1's... no — onto NN_3's own 0.1. The
        # rounded local match must not excuse a value owned elsewhere.
        doc = _tiny_ir()
        doc["evidence"][1]["value"] = 0.104
        doc["evidence"][1]["text"] = "NN_3 achieves a score of 0.104."
        doc["evidence"][0]["value"] = 0.1043
        doc["evidence"][0]["text"] = "LOF_1 achieves a score of 0.1043."
        v = verifier.verify_narrative("LOF_1 achieves a score of 0.104.", doc)
        self.assertEqual([m["number"] for m in v["misattributed_numbers"]],
                         ["0.104"])


class TestPrompts(unittest.TestCase):

    def test_stage_prompt_contains_all_atoms_and_markers(self):
        result = _mc_result()
        # MC's run-invariant notes now live in the info footer; force a
        # run-specific caveat (majority-degenerate CV) to exercise [CAVEAT].
        result["permodel_f1"] = {"A": {"cv_r2": 0.6, "cv_n_splits": 5,
                                       "cv_degenerate_folds": 4}}
        doc = ir.build_monte_carlo_ir("DS", "e1", result, ["LOF_1", "NN_3"],
                                      ["NN_3", "LOF_1"])
        self.assertTrue(doc["caveats"])  # guard: the marker test needs a caveat
        prompt = llm.build_stage_prompt(doc)
        for atom in doc["evidence"]:
            self.assertIn(atom["text"], prompt)
        self.assertEqual(prompt.count("[REQUIRED]"), len(doc["required_atom_ids"]))
        for cav in doc["caveats"]:
            self.assertIn(cav["text"], prompt)
        self.assertIn("[CAVEAT]", prompt)
        self.assertIn("120-220 words", prompt)

    def test_global_prompt_is_fact_based(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "x")
            path = ir.assemble_global_ir(_results_dict(), "DS", "e1", 3, base_dir=base)
            with open(path) as f:
                gdoc = json.load(f)
        prompt = llm.build_global_prompt(gdoc)
        self.assertIn("FACTS", prompt)
        # The decision arrives as a pre-rendered sentence, not a key:value dump.
        self.assertIn("The final decision is the single model LOF_1", prompt)
        self.assertNotIn("- framework_choice:", prompt)
        self.assertEqual(prompt.count("[REQUIRED]"), len(gdoc["required_atom_ids"]))
        self.assertIn("STAGES WITHOUT DATA", prompt)
        self.assertIn("150-300 words", prompt)

    def test_stage_task_hint_only_for_registered_stages(self):
        doc = _tiny_ir()
        doc["stage"] = "rank_aggregation_robust"
        prompt = llm.build_stage_prompt(doc)
        self.assertIn("rank is a position", prompt)
        self.assertIn("never restate a rank as 'high' or 'low'", prompt)
        # Other stages get no hint.
        doc["stage"] = "monte_carlo"
        self.assertNotIn("rank is a position", llm.build_stage_prompt(doc))

    def test_question_frames_the_prompt(self):
        doc = _tiny_ir()
        # No question → plain framing.
        self.assertNotIn("QUESTION THIS STAGE ANSWERS", llm.build_stage_prompt(doc))
        doc["question"] = "Why did LOF_1 rank first?"
        prompt = llm.build_stage_prompt(doc)
        self.assertIn("QUESTION THIS STAGE ANSWERS: Why did LOF_1 rank first?", prompt)
        self.assertIn("answers the question above, leading with the answer", prompt)

    def test_stage_prompt_budget_scales_with_atom_count(self):
        # Sparse stage (tiny IR, 2 atoms) → low floor, no 120-word padding.
        doc = _tiny_ir()
        self.assertEqual(llm._word_budget(len(doc["evidence"])), (65, 120))
        self.assertIn("65-120 words", llm.build_stage_prompt(doc))
        # Mid-size stage → the default 120-220.
        mid = dict(doc)
        mid["evidence"] = [
            {"id": f"toy.a{i}", "type": "t", "subject": "LOF_1", "value": i,
             "text": f"Fact number {i}."} for i in range(7)
        ]
        self.assertIn("120-220 words", llm.build_stage_prompt(mid))
        # Dense stage → ceiling scales above 220.
        dense = dict(doc)
        dense["evidence"] = [
            {"id": f"toy.a{i}", "type": "t", "subject": "LOF_1", "value": i,
             "text": f"Fact number {i}."} for i in range(30)
        ]
        prompt = llm.build_stage_prompt(dense)
        lo, hi = llm._word_budget(30)
        self.assertGreater(hi, 220)
        self.assertIn(f"{lo}-{hi} words", prompt)

    def test_fact_lines_follow_presentation_order(self):
        doc = _tiny_ir()
        doc["evidence"] = [
            {"id": "z.second", "type": "t", "subject": "LOF_1", "value": None,
             "text": "Second fact.", "order": 2},
            {"id": "a.first", "type": "t", "subject": "LOF_1", "value": None,
             "text": "First fact.", "order": 1},
            {"id": "m.unordered", "type": "t", "subject": "LOF_1", "value": None,
             "text": "Unordered fact."},
        ]
        doc["required_atom_ids"] = []
        lines = llm._fact_lines(doc)
        self.assertEqual(lines, ["1. First fact.", "2. Second fact.",
                                 "3. Unordered fact."])


class TestClient(unittest.TestCase):

    def test_transport_payload_and_passthrough(self):
        captured = {}

        def transport(payload):
            captured.update(payload)
            return "narrative text"

        client = llm.LLMClient(model="test-model", transport=transport)
        out = client.chat("SYS", "USER")
        self.assertEqual(out, "narrative text")
        self.assertEqual(captured["model"], "test-model")
        self.assertEqual(captured["temperature"], 0.0)
        self.assertEqual(captured["seed"], 0)
        self.assertEqual(captured["messages"][0],
                         {"role": "system", "content": "SYS"})
        self.assertEqual(captured["messages"][1]["content"], "USER")
        self.assertFalse(captured["stream"])


class TestNarrateEntity(unittest.TestCase):

    def test_end_to_end_with_fake_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "explanations_ir")
            out = os.path.join(tmp, "explanations_nl")
            # Two stage IRs + the global IR; the rest are missing on purpose.
            ir.write_stage_ir(
                ir.build_ga_combination_ir("DS", "e1", _ga_combination_result()),
                "DS", "e1", "ir_ga_combination", base_dir=base)
            ir.write_stage_ir(
                ir.build_monte_carlo_ir("DS", "e1", _mc_result(),
                                        ["LOF_1", "NN_3"], ["NN_3", "LOF_1"]),
                "DS", "e1", "ir_monte_carlo", base_dir=base)
            ir.assemble_global_ir(_results_dict(), "DS", "e1", 3, base_dir=base)

            report = llm.narrate_entity("DS", "e1", 3, FakeClient(),
                                        base_dir=base, out_dir=out)

            self.assertEqual(report["stages"]["ga_combination"]["status"], "ok")
            self.assertEqual(report["stages"]["monte_carlo"]["status"], "ok")
            self.assertEqual(report["stages"]["global"]["status"], "ok")
            self.assertEqual(report["stages"]["thompson_sampling"]["status"], "skipped")
            # Perfect-copy narratives → zero rates everywhere.
            self.assertEqual(report["overall"]["hallucination_rate"], 0.0)
            self.assertEqual(report["overall"]["omission_rate"], 0.0)
            nl_dir = os.path.join(out, "DS", "e1")
            for fname in ("nl_ga_combination.txt", "nl_monte_carlo.txt",
                          "nl_global_iter3.txt", "faithfulness_iter3.json",
                          "faithfulness_iter3.txt"):
                self.assertTrue(os.path.exists(os.path.join(nl_dir, fname)), fname)
            with open(os.path.join(nl_dir, "faithfulness_iter3.json")) as f:
                saved = json.load(f)
            self.assertEqual(saved["overall"]["omission_rate"], 0.0)

    def test_rank_agg_glob_fallback(self):
        # Rank-agg IR written under iteration 7, narration requested for 3:
        # the newest matching file must be picked up instead of skipping.
        ra_result = {"loo_scores": {"S1": 0.3, "S2": 0.1},
                     "align_scores": {"S1": 0.6, "S2": 0.8},
                     "borda_counts": {"S1": 3.0, "S2": 3.0},
                     "verdicts": [], "prominent_contradictions": [],
                     "kendall_only": None}
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "explanations_ir")
            out = os.path.join(tmp, "explanations_nl")
            ir.write_stage_ir(
                ir.build_rank_aggregation_ir("DS", "e1", "robust", 7, ra_result,
                                             ["S1", "S2"], {"S1": "A", "S2": "B"},
                                             ["A", "B"]),
                "DS", "e1", "ir_rank_aggregation_robust_7", base_dir=base)
            report = llm.narrate_entity("DS", "e1", 3, FakeClient(),
                                        base_dir=base, out_dir=out,
                                        stages=["rank_aggregation_robust"])
            self.assertEqual(report["stages"]["rank_aggregation_robust"]["status"], "ok")

    def test_stage_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "explanations_ir")
            out = os.path.join(tmp, "explanations_nl")
            ir.write_stage_ir(
                ir.build_ga_combination_ir("DS", "e1", _ga_combination_result()),
                "DS", "e1", "ir_ga_combination", base_dir=base)
            report = llm.narrate_entity("DS", "e1", 0, FakeClient(),
                                        base_dir=base, out_dir=out,
                                        stages=["ga_combination"])
            self.assertEqual(list(report["stages"].keys()), ["ga_combination"])

    def test_info_footer_leads_file_and_is_outside_verification(self):
        """The glossary heads the .txt — the reader meets the terms before the
        prose using them — but is written outside the model's output, so it
        never enters the verified narrative or the metrics."""
        ra_result = {
            "verdicts": [
                {"source": "S1", "loo_score": 0.3, "loo_rank": 1, "align_score": 0.6,
                 "align_rank": 1, "borda_rank": 1, "pattern": "consistent"},
                {"source": "S2", "loo_score": 0.1, "loo_rank": 2, "align_score": 0.8,
                 "align_rank": 2, "borda_rank": 2, "pattern": "redundant_agreer"}],
            "prominent_contradictions": [], "kendall_only": None}
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "explanations_ir")
            out = os.path.join(tmp, "explanations_nl")
            ir.write_stage_ir(
                ir.build_rank_aggregation_ir("DS", "e1", "robust", 0, ra_result,
                                             ["S1", "S2"], {"S1": "A", "S2": "B"},
                                             ["A", "B"]),
                "DS", "e1", "ir_rank_aggregation_robust_0", base_dir=base)
            report = llm.narrate_entity("DS", "e1", 0, FakeClient(),
                                        base_dir=base, out_dir=out,
                                        stages=["rank_aggregation_robust"])
            info = report["stages"]["rank_aggregation_robust"]
            self.assertEqual(info["status"], "ok")
            with open(info["narrative_path"]) as f:
                content = f.read()
            # Glossary first, then a blank line, then the narrative.
            self.assertTrue(content.startswith("INFO: "), content[:40])
            footer, _, body = content.partition("\n\n")
            self.assertIn("Influence measures how much a source moved", footer)
            # The narrative itself carries neither the marker nor the glossary.
            self.assertTrue(body.strip())
            self.assertNotIn("INFO:", body)
            self.assertNotIn("Influence measures", body)
            # Metrics are computed on the narrative alone.
            self.assertEqual(info["words"], len(body.split()))

    def test_repair_pass_fixes_violating_draft(self):
        """A draft with a hallucinated number triggers ONE verifier-guided
        retry; both metric sets are recorded and the clean rewrite is kept."""

        class RepairingClient(FakeClient):
            def __init__(self):
                self.prompts = []

            def chat(self, system, user):
                self.prompts.append(user)
                clean = FakeClient.chat(self, system, user)
                if len(self.prompts) == 1:
                    return clean + " A bogus extra value of 0.912345 appears."
                return clean

        client = RepairingClient()
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "explanations_ir")
            out = os.path.join(tmp, "explanations_nl")
            ir.write_stage_ir(
                ir.build_ga_combination_ir("DS", "e1", _ga_combination_result()),
                "DS", "e1", "ir_ga_combination", base_dir=base)
            report = llm.narrate_entity("DS", "e1", 0, client,
                                        base_dir=base, out_dir=out,
                                        stages=["ga_combination"])
        info = report["stages"]["ga_combination"]
        self.assertEqual(info["status"], "ok")
        self.assertTrue(info["repaired"])
        self.assertIn("0.912345", info["verify_initial"]["unsupported_numbers"])
        self.assertEqual(info["verify"]["unsupported_numbers"], [])
        self.assertEqual(info["verify"]["hallucination_rate"], 0.0)
        # Exactly one retry, and it carried the violation feedback.
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("PROBLEMS DETECTED IN THE DRAFT", client.prompts[1])
        self.assertIn("0.912345", client.prompts[1])

    def test_no_repair_call_for_clean_draft(self):
        class CountingClient(FakeClient):
            def __init__(self):
                self.n = 0

            def chat(self, system, user):
                self.n += 1
                return FakeClient.chat(self, system, user)

        client = CountingClient()
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "explanations_ir")
            out = os.path.join(tmp, "explanations_nl")
            ir.write_stage_ir(
                ir.build_ga_combination_ir("DS", "e1", _ga_combination_result()),
                "DS", "e1", "ir_ga_combination", base_dir=base)
            report = llm.narrate_entity("DS", "e1", 0, client,
                                        base_dir=base, out_dir=out,
                                        stages=["ga_combination"])
        info = report["stages"]["ga_combination"]
        self.assertEqual(client.n, 1)
        self.assertNotIn("repaired", info)
        self.assertNotIn("verify_initial", info)


class TestGlobalNarrativeModes(unittest.TestCase):
    """The global document has two interchangeable builders: a deterministic
    merge of the per-stage prose (default) and the original atom-based LLM
    path. Both stay working so switching back is one argument."""

    def _texts(self):
        return {
            "monte_carlo": "MC prose.",
            "ga_selection": "GA selection prose.",
            "rank_aggregation_final": "Final consensus prose.",
        }

    def _global_ir(self):
        return {
            "stage": "global", "dataset": "DS", "entity": "e1",
            "evidence": [
                {"id": "global.decision", "type": "decision", "subject": "d",
                 "value": None, "text": "The final decision is the ensemble {A, B}."},
                {"id": "global.agreement.gan", "type": "stage_agreement",
                 "subject": "gan", "value": None,
                 "text": "gan's top pick (A) differs from the final pick (B)."},
            ],
            "stages": {"monte_carlo": {"status": "ok"},
                       "ga_selection": {"status": "ok"},
                       "rank_aggregation_final": {"status": "ok"},
                       "thompson_sampling": {"status": "not_available"}},
        }

    def test_compose_orders_by_pipeline_and_flags_absent_stages(self):
        doc = llm.compose_global_narrative(
            self._texts(), self._global_ir(), dataset="DS", entity="e1", iteration=3)
        self.assertIn("RAMSeS model selection — DS / entity e1 (iteration 3)", doc)
        self.assertIn("The final decision is the ensemble {A, B}.", doc)
        self.assertIn("- gan's top pick (A) differs", doc)
        # Pipeline order, not the alphabetical order of the dict.
        self.assertLess(doc.index("GA selection prose."), doc.index("MC prose."))
        self.assertLess(doc.index("MC prose."), doc.index("Final consensus prose."))
        # A stage the run could not narrate is named, so a short document is
        # never mistaken for a complete one.
        self.assertIn("Stages without a narrative: thompson_sampling.", doc)

    def test_compose_is_verbatim_and_deterministic(self):
        texts = self._texts()
        a = llm.compose_global_narrative(texts, self._global_ir(), dataset="DS",
                                         entity="e1", iteration=3)
        b = llm.compose_global_narrative(texts, self._global_ir(), dataset="DS",
                                         entity="e1", iteration=3)
        self.assertEqual(a, b)
        for prose in texts.values():          # reused exactly, never paraphrased
            self.assertIn(prose, a)
        # Footers are opt-in here; narrate_entity always supplies them.
        self.assertNotIn("INFO:", a)
        with_footer = llm.compose_global_narrative(
            texts, self._global_ir(), dataset="DS", entity="e1", iteration=3,
            stage_footers={"monte_carlo": "Noise is Gaussian."})
        self.assertIn("INFO: Noise is Gaussian.", with_footer)
        # Glossary precedes the prose it explains, as in the per-stage files.
        self.assertLess(with_footer.index("INFO: Noise is Gaussian."),
                        with_footer.index("MC prose."))

    def _run(self, tmp, **kw):
        base = os.path.join(tmp, "explanations_ir")
        out = os.path.join(tmp, "explanations_nl")
        ir.write_stage_ir(
            ir.build_ga_combination_ir("DS", "e1", _ga_combination_result()),
            "DS", "e1", "ir_ga_combination", base_dir=base)
        ir.assemble_global_ir(_results_dict(), "DS", "e1", 3, base_dir=base)
        report = llm.narrate_entity("DS", "e1", 3, FakeClient(),
                                    base_dir=base, out_dir=out, **kw)
        return report, os.path.join(out, "DS", "e1")

    def test_concat_mode_reuses_stage_prose_and_is_not_rescored(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, nl_dir = self._run(tmp)               # concat is the default
            g = report["stages"]["global"]
            self.assertEqual(g["status"], "ok")
            self.assertEqual(g["mode"], "concat")
            self.assertEqual(g["merged_stages"], ["ga_combination"])
            # Deterministic merge adds no claims, so it carries no metrics and
            # cannot double-count the stage prose in the micro-average.
            self.assertNotIn("verify", g)
            # The merged document carries each stage's glossary too.
            with open(os.path.join(nl_dir, "nl_global_iter3.txt")) as f:
                merged = f.read()
            self.assertIn("INFO: ", merged)
            with open(os.path.join(nl_dir, "nl_global_iter3.txt")) as f:
                doc = f.read()
            with open(os.path.join(nl_dir, "nl_ga_combination.txt")) as f:
                stage = f.read().split("\nINFO:")[0].strip()
            self.assertIn(stage, doc)

    def test_llm_mode_still_narrates_and_verifies_the_global_ir(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _ = self._run(tmp, global_mode="llm")
            g = report["stages"]["global"]
            self.assertEqual(g["status"], "ok")
            self.assertNotIn("mode", g)
            self.assertIn("verify", g)
            self.assertEqual(g["verify"]["omission_rate"], 0.0)

    def test_unknown_global_mode_rejected(self):
        with self.assertRaises(ValueError):
            llm.narrate_entity("DS", "e1", 0, FakeClient(), global_mode="summarise")


if __name__ == "__main__":
    unittest.main(verbosity=2)
