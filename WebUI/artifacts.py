"""
Read-only reader over the explainability artifacts.

Assembles the page payload from the per-stage `nl_*.txt` and `ir_*.json` files
rather than parsing `nl_global_iter*.txt`. The global text is *derived* from the
same per-stage strings by `Explainability.llm.compose_global_narrative`, so
parsing it back would be a lossy round-trip through a format whose section
separators are `"=" * len(title)` — any wording change there would silently
break the reader with no test failure. Assembling instead also yields the
structured data the text cannot carry (decision, stage agreement, per-stage
outputs, caveats, confidence), which is what makes the page readable rather than
a 2,000-word wall. The global `.txt` stays available as a verbatim download.
"""

import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from WebUI import paths
from WebUI.summarize import attribute_sentences, summarize

# ── The three-vocabulary map ────────────────────────────────────────────────
#
# Three different names exist for the same stage: the CLI --stages token, the
# IR/NL stage key, and the plot directory. This tuple is the single owner of
# that mapping; plots.py, markers.py and the frontend all join on `key`.
# Order matches Explainability.llm._GLOBAL_STAGE_ORDER.
STAGES: Tuple[Dict[str, Any], ...] = (
    {"key": "ga_selection", "title": "Ensemble selection (genetic algorithm)",
     "cli": "ga", "ir": "ir_ga_selection", "nl": "nl_ga_selection",
     "plot_group": "ga_selection", "order": 1},
    {"key": "ga_combination", "title": "Ensemble weighting (meta-learner)",
     "cli": "ga", "ir": "ir_ga_combination", "nl": "nl_ga_combination",
     "plot_group": "ga_combination", "order": 2},
    # One CLI token, two stages — the same split as ga_selection/ga_combination.
    # `thompson_ranking` explains mu^T mu, the criterion the detectors are
    # ordered by; `thompson_sampling` explains mu^T x, the expected reward that
    # drove per-window selection. Neither keeps the plain name.
    #
    # `plot_group` is deliberately `ts_ranking`, not `thompson_ranking`:
    # result.js matches lazy-gallery descriptors with `id.startsWith(plot_group)`,
    # so a group prefixed by "thompson" would let one card claim the other's
    # galleries. `regimes` names the plot subdirectory whose per-regime figures
    # pair with this stage's regime atoms; its presence is what makes a stage
    # regime-bearing, replacing a hardcoded stage-key check here and in server.py.
    {"key": "thompson_ranking", "title": "Thompson Sampling: ranking criterion",
     "cli": "thompson", "ir": "ir_thompson_ranking", "nl": "nl_thompson_ranking",
     "plot_group": "ts_ranking", "order": 3,
     "regimes": ["ranking_per_regime"]},
    {"key": "thompson_sampling", "title": "Thompson Sampling: selection dynamics",
     "cli": "thompson", "ir": "ir_thompson", "nl": "nl_thompson",
     "plot_group": "thompson", "order": 4,
     "regimes": ["reward_per_regime", "shap_per_regime"]},
    {"key": "monte_carlo", "title": "Robustness: Monte Carlo noise sweep",
     "cli": "montecarlo", "ir": "ir_monte_carlo", "nl": "nl_monte_carlo",
     "plot_group": "monte_carlo", "order": 5},
    {"key": "off_by_threshold", "title": "Sensitivity: off-by-threshold test",
     "cli": "offby", "ir": "ir_off_by", "nl": "nl_off_by",
     "plot_group": "off_by", "order": 6},
    {"key": "rank_aggregation_robust", "title": "Robustness consensus",
     "cli": None, "ir": "ir_rank_aggregation_robust", "nl": "nl_rank_aggregation_robust",
     "plot_group": "rank_aggregation_robust", "order": 7, "iterated": True},
    {"key": "rank_aggregation_final", "title": "Final consensus",
     "cli": None, "ir": "ir_rank_aggregation_final", "nl": "nl_rank_aggregation_final",
     "plot_group": "rank_aggregation_final", "order": 8, "iterated": True},
)

STAGE_BY_KEY = {s["key"]: s for s in STAGES}

# The GAN test runs but has no explainability layer; the global IR records that
# explicitly. Surfacing it as a stage with a reason beats an unexplained gap.
GAN_STAGE = {"key": "gan", "title": "Robustness: GAN perturbations",
             "cli": "gan", "plot_group": "gan", "order": 9}


def split_info(raw: str) -> Tuple[Optional[str], str]:
    """`"INFO: glossary\\n\\nnarrative"` -> `("glossary", "narrative")`.

    The glossary leads the file (Explainability/llm.py writes it before the
    narrative). Anything that does not start with the marker is all narrative.
    """
    if raw is None:
        return None, ""
    text = raw.replace("\r\n", "\n").strip("\n")
    if not text.startswith("INFO:"):
        return None, text.strip()
    body = text[len("INFO:"):]
    info, sep, narrative = body.partition("\n\n")
    if not sep:
        # Glossary with no narrative after it.
        return info.strip(), ""
    return info.strip(), narrative.strip()


def _read_text(path: Path) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _newest(pattern_dir: Path, pattern: str) -> Optional[Path]:
    """Newest file matching `pattern` inside `pattern_dir`, by mtime.

    Iteration numbers are NOT consistent across artifact trees: the
    comprehensive report uses the CLI --iteration (default 5) while the
    explanations use OFFLINE_ITERATION = 0, and both can coexist in one
    directory. Always glob and take the newest; never derive the index.
    """
    matches = glob.glob(os.path.join(glob.escape(str(pattern_dir)), pattern))
    if not matches:
        return None
    return Path(max(matches, key=lambda p: os.path.getmtime(p)))


# A narrative is stale once its IR has been rewritten under it. This happens
# for real: `--stages X --explain` is a PARTIAL run, and app.py returns before
# the narration block, so it regenerates the IR and every plot but leaves the
# prose alone. The page then shows fresh figures beside sentences describing a
# run that no longer exists — on SKAB/7 the narrative walked 14 regimes while
# the IR and the plots had 10. Nothing else notices, because the two files are
# read independently. One second of slack absorbs same-run write ordering.
_STALE_SLACK_SECONDS = 1.0


def _mtime(path: Optional[Path]) -> Optional[float]:
    try:
        return path.stat().st_mtime if path is not None else None
    except OSError:
        return None


def _load_stage_files(ir_dir: Optional[Path], nl_dir: Optional[Path],
                      stage: Dict[str, Any]) -> Tuple[Optional[dict], Optional[str],
                                                      Optional[Path], bool]:
    """(ir_doc, raw_nl_text, nl_path, narrative_is_stale) for one stage."""
    ir_doc = raw = None
    ir_path = nl_path = None
    if ir_dir is not None:
        exact = ir_dir / f"{stage['ir']}.json"
        ir_path = exact if exact.exists() else _newest(ir_dir, f"{stage['ir']}*.json")
        if ir_path is not None:
            ir_doc = _read_json(ir_path)
    if nl_dir is not None:
        exact = nl_dir / f"{stage['nl']}.txt"
        nl_path = exact if exact.exists() else _newest(nl_dir, f"{stage['nl']}*.txt")
        if nl_path is not None:
            raw = _read_text(nl_path)
    ir_at, nl_at = _mtime(ir_path), _mtime(nl_path)
    stale = bool(ir_at and nl_at and nl_at + _STALE_SLACK_SECONDS < ir_at)
    return ir_doc, raw, nl_path, stale


def load_global_ir(dataset: str, entity: str) -> Optional[dict]:
    ir_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_IR, dataset, entity)
    if ir_dir is None:
        return None
    path = _newest(ir_dir, "ir_global_iter*.json")
    return _read_json(path) if path else None


def load_faithfulness(dataset: str, entity: str) -> Optional[dict]:
    nl_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_NL, dataset, entity)
    if nl_dir is None:
        return None
    path = _newest(nl_dir, "faithfulness_iter*.json")
    return _read_json(path) if path else None


def global_text_path(dataset: str, entity: str) -> Optional[Path]:
    nl_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_NL, dataset, entity)
    if nl_dir is None:
        return None
    return _newest(nl_dir, "nl_global_iter*.txt")


_ITER_RE = re.compile(r"_iter(\d+)\.txt$")


def comprehensive_path(dataset: str, entity: str) -> Optional[Path]:
    """Newest `comprehensive_results_*.txt` for this entity, or None.

    Written by the pipeline itself, not by the explainability layer, so it
    exists for runs made without `--explain` and is absent after a partial run
    (`app.py` returns before the report step). Its iteration index comes from
    `--iteration` (default 5) rather than the explanations' OFFLINE_ITERATION,
    which is exactly why this globs instead of building the filename.
    """
    report_dir = paths.resolve_entity_dir(paths.COMPREHENSIVE, dataset, entity)
    if report_dir is None:
        return None
    return _newest(report_dir, "comprehensive_results_*.txt")


def comprehensive_info(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """Metadata for the report, without reading it — the page links, not inlines."""
    path = comprehensive_path(dataset, entity)
    if path is None:
        return None
    match = _ITER_RE.search(path.name)
    try:
        stat = path.stat()
    except OSError:
        return None
    return {
        "name": path.name,
        "iteration": int(match.group(1)) if match else None,
        "bytes": stat.st_size,
        "generated_at": stat.st_mtime,
        "url": f"/report/{dataset}/{entity}",
        "download_url": f"/api/comprehensive/{dataset}/{entity}?download=1",
    }


def comprehensive_report(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """`comprehensive_info` plus the report text itself."""
    info = comprehensive_info(dataset, entity)
    if info is None:
        return None
    path = comprehensive_path(dataset, entity)
    return {**info, "text": _read_text(path) or ""}


# Matches both Thompson stages' regime atoms — `ts.regime.N` (expected-reward
# regimes) and `tsr.regime.N` (||mu||^2 leadership regimes). Anchored on the
# suffix rather than the prefix so a third producer needs no change here.
_REGIME_RE = re.compile(r"\.regime\.(\d+)$")


def _regimes_from_ir(ir_doc: dict) -> List[Dict[str, Any]]:
    """Regime atoms, ordered, ready to pair with their per-regime plots.

    The ids are 0-based and match `regime_{NN}_w{start}-{end}_{model}.png`
    exactly, so each regime sentence can be shown beside its own plot instead of
    the reader hunting through fourteen images.
    """
    out = []
    for atom in ir_doc.get("evidence", []) or []:
        m = _REGIME_RE.search(str(atom.get("id", "")))
        if not m:
            continue
        value = atom.get("value") or {}
        out.append({
            "index": int(m.group(1)),
            "start": value.get("start"),
            "end": value.get("end"),
            "duration": value.get("duration"),
            "leader": value.get("leader") or atom.get("subject"),
            "text": atom.get("text", ""),
        })
    return sorted(out, key=lambda r: r["index"])


def _attach_narrated_regimes(regimes: List[Dict[str, Any]], narrative: str,
                             ir_doc: dict) -> None:
    """Give each regime the sentence the model wrote about it.

    The per-regime disclosure used to show the IR's own atom text — correct but
    flat, and a second rendering of facts the narrative already covers. The
    narrated sentences are pulled out of the same paragraph the summary drops,
    so nothing is generated twice and nothing is lost by hiding them from the
    default view. `text` stays as the fallback when a regime's sentence cannot
    be located.
    """
    by_index: Dict[int, str] = {}
    for sentence, atom in attribute_sentences(narrative, ir_doc):
        if not atom or atom.get("type") != "regime":
            continue
        m = _REGIME_RE.search(str(atom.get("id", "")))
        if m:
            idx = int(m.group(1))
            # A regime can span two sentences; keep them in narrative order.
            by_index[idx] = (by_index.get(idx, "") + " " + sentence.strip()).strip()
    for regime in regimes:
        narrated = by_index.get(regime.get("index"))
        if narrated:
            regime["narrated"] = narrated


def _headline_pick(output: Dict[str, Any]) -> Optional[str]:
    """The one detector a stage put first, whatever the stage calls that key."""
    for key in ("top_pick", "winner", "top_pick_f1"):
        value = output.get(key)
        if isinstance(value, str) and value and value != "not_available":
            return value
    return None


def _stage_faithfulness(report: Optional[dict], key: str) -> Optional[dict]:
    if not report:
        return None
    entry = (report.get("stages") or {}).get(key)
    if not entry:
        return None
    verify = entry.get("verify") or {}
    return {
        "status": entry.get("status"),
        "words": entry.get("words"),
        "hallucination_rate": verify.get("hallucination_rate"),
        "omission_rate": verify.get("omission_rate"),
        "n_claims": verify.get("n_claims"),
        "n_required": verify.get("n_required"),
        "repaired": bool(entry.get("repaired")),
    }


def build_payload(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """The `/api/explanations/<ds>/<ent>` response, or None if nothing exists."""
    ir_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_IR, dataset, entity)
    nl_dir = paths.resolve_entity_dir(paths.EXPLANATIONS_NL, dataset, entity)
    if ir_dir is None and nl_dir is None:
        return None

    global_ir = load_global_ir(dataset, entity)
    faith = load_faithfulness(dataset, entity)
    gtext = global_text_path(dataset, entity)

    stages_out: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for stage in STAGES:
        ir_doc, raw, nl_path, stale = _load_stage_files(ir_dir, nl_dir, stage)
        if ir_doc is None and raw is None:
            continue
        info, narrative = split_info(raw or "")
        summary = summarize(narrative, stage=stage["key"], ir_doc=ir_doc)
        output = (ir_doc or {}).get("output") or {}
        entry: Dict[str, Any] = {
            "key": stage["key"],
            "title": stage["title"],
            "order": stage["order"],
            "status": "ok" if narrative else "no_narrative",
            # Stages name their headline result differently: `top_pick` for
            # most, `winner` for off-by, `top_pick_f1` for Monte Carlo (which
            # ranks on two metrics). GA selection has no single pick — it
            # chooses a set — so None there is correct, not a gap.
            "top_pick": _headline_pick(output),
            "summary": summary["summary"],
            "summary_is_full": summary["is_full"],
            "summary_mode": summary["mode"],
            "summary_table": summary.get("table"),
            "extended_in": summary.get("extended_in"),
            # What the full-text disclosure shows. Usually the whole narrative;
            # a stage that renders some of its sentences elsewhere on the card
            # (Thompson's regime walk, beside its per-regime plots) hands back a
            # trimmed body so the page never prints them twice. `words` and the
            # download stay on the real narrative — the file on disk is the
            # verbatim record, and the length is what the model actually wrote.
            # `body` is that narrative minus any sentence restating a caveat:
            # the card lists the caveats verbatim below, so the disclosure must
            # not be a second, looser copy of them.
            "full": summary.get("extended") or summary.get("body") or narrative,
            "words": len(narrative.split()) if narrative else 0,
            "info": info,
            "question": (ir_doc or {}).get("question"),
            "output": output,
            "caveats": [c.get("text") for c in ((ir_doc or {}).get("caveats") or [])],
            "faithfulness": _stage_faithfulness(faith, stage["key"]),
            "plot_group": stage["plot_group"],
            "nl_file": nl_path.name if nl_path else None,
            # The narrative predates its own IR: the numbers on this card
            # may describe a previous run. Surfaced rather than silently
            # rendered, and cleared by re-running the narrator.
            "stale": stale,
        }
        if stage.get("regimes") and ir_doc:
            entry["regimes"] = _regimes_from_ir(ir_doc)
            _attach_narrated_regimes(entry["regimes"], narrative, ir_doc)
        stages_out.append(entry)

    # Stages the global IR knows about but that produced no narrative — GAN
    # always, plus anything a partial run skipped. Carry the IR's own note so
    # the UI states a reason instead of showing an unexplained gap.
    present = {s["key"] for s in stages_out}
    for key, info_block in sorted(((global_ir or {}).get("stages") or {}).items()):
        if key in present:
            continue
        meta = STAGE_BY_KEY.get(key) or (GAN_STAGE if key == "gan" else None)
        missing.append({
            "key": key,
            "title": (meta or {}).get("title", key.replace("_", " ").capitalize()),
            "plot_group": (meta or {}).get("plot_group"),
            "status": info_block.get("status", "not_available"),
            "note": info_block.get("note"),
        })

    decision = (global_ir or {}).get("decision") or {}
    # The final consensus IS the source of the single-model pick, so comparing
    # them always reports agreement and says nothing. Newer runs no longer emit
    # the row; filtering here means older result trees read correctly too.
    agreement = [
        {"source": name,
         "top_pick": info.get("top_pick"),
         "agrees": info.get("agrees_with_final_single")}
        for name, info in sorted(((global_ir or {}).get("stage_agreement") or {}).items())
        if name != "final_consensus"
    ]
    decision_atom = next(
        (a.get("text") for a in ((global_ir or {}).get("evidence") or [])
         if a.get("id") == "global.decision"), None)

    return {
        "dataset": dataset,
        "entity": entity,
        "iteration": (global_ir or {}).get("iteration"),
        # No global IR but a global .txt on disk means an older result tree:
        # serve the raw text, never try to parse it.
        "degraded": global_ir is None,
        "decision": decision,
        "decision_text": decision_atom,
        "agreement": agreement,
        "stages": sorted(stages_out, key=lambda s: s["order"]),
        "missing_stages": missing,
        "faithfulness": (faith or {}).get("overall"),
        "model": (faith or {}).get("model"),
        "global_text": gtext.name if gtext else None,
        "generated_at": (os.path.getmtime(gtext) if gtext else None),
        # Kept beside the explanation but never merged into it: the report is
        # the pipeline's own record of what happened, in its own numbers.
        "comprehensive": comprehensive_info(dataset, entity),
    }


def entity_summary(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """Compact card for the "previous results" list — no narrative loading."""
    global_ir = load_global_ir(dataset, entity)
    gtext = global_text_path(dataset, entity)
    if global_ir is None and gtext is None:
        return None
    faith = load_faithfulness(dataset, entity)
    decision = (global_ir or {}).get("decision") or {}
    return {
        "dataset": dataset,
        "entity": entity,
        "framework_choice": decision.get("framework_choice"),
        "chosen": decision.get("chosen"),
        "n_stages": len([s for s in ((global_ir or {}).get("stages") or {}).values()
                         if s.get("status") == "ok"]),
        "hallucination_rate": ((faith or {}).get("overall") or {}).get("hallucination_rate"),
        "omission_rate": ((faith or {}).get("overall") or {}).get("omission_rate"),
        "generated_at": os.path.getmtime(gtext) if gtext else None,
    }


def known_entities() -> List[Tuple[str, str]]:
    """(dataset, entity) pairs that have explanation artifacts on disk."""
    found = []
    for root in (paths.EXPLANATIONS_NL, paths.EXPLANATIONS_IR):
        if not root.is_dir():
            continue
        for ds_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not ds_dir.is_dir():
                continue
            for ent_dir in sorted(ds_dir.iterdir(), key=lambda p: paths.natural_key(p.name)):
                if ent_dir.is_dir():
                    pair = (ds_dir.name, ent_dir.name)
                    if pair not in found:
                        found.append(pair)
    return found
