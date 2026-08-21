"""
Turns pipeline log lines into progress events.

Pure and side-effect free so it can be tested against a captured log without
running anything. Matching is on the text *after* the emoji rather than on the
emoji itself: an encoding hiccup in a terminal or a locale change must not
silently stop progress from advancing.
"""

import re
from typing import Any, Dict, Optional

# Sub-stage number -> the stage key the rest of the UI joins on. 6.6 is rank
# aggregation, which has no --stages token of its own (it only runs on a full
# run) but still deserves a row in the stage rail.
SUBSTAGE_KEYS = {
    "6.1": "ga", "6.2": "thompson", "6.3": "gan",
    "6.4": "offby", "6.5": "montecarlo", "6.6": "aggregation",
}

# The bracketed tag each stage prints when it finishes.
RESULT_TAGS = {
    "GA": "ga", "Thompson": "thompson", "GAN": "gan",
    "Borderline": "offby", "MonteCarlo": "montecarlo", "Aggregation": "aggregation",
}

PHASE_TITLES = {
    1: "Loading training data", 2: "Loading test data",
    3: "Training / loading models", 4: "Injecting synthetic anomalies",
    5: "Preparing data and visualisation", 6: "Model selection",
    7: "Writing results",
}

_PHASE_RE = re.compile(r"STAGE (\d)/7:\s*(.*)")
_SUBSTAGE_RE = re.compile(r"Sub-stage (6\.\d):\s*(.*)")
_RESULT_RE = re.compile(r"\[([A-Za-z]+)\]\s*(.*)")
_BANNER_RE = re.compile(r"STARTING RAMSeS EXECUTION:\s*(.*)")

# Completion markers. run_app swallows exceptions and still exits 0, so seeing
# one of these is the only reliable evidence a run actually finished.
FULL_COMPLETE = "EXECUTION COMPLETE!"
PARTIAL_COMPLETE = "Partial run complete"
# The signature of the swallowed-exception path in app.py.
FATAL_SIGNATURE = "Traceback for Entity:"

_WARNINGS = (
    ("LLM narration skipped", "llm_unreachable",
     "The LLM server was unreachable, so narratives were not generated."),
    ("LLM narration failed", "llm_failed", "LLM narration failed."),
    ("Global IR assembly failed", "ir_failed", "Global explanation assembly failed."),
    ("Requested detectors with no trained model", "detectors_missing",
     "Some requested detectors had no trained model and were skipped."),
    ("Only", "few_detectors", None),   # refined below
)


def _banner_fields(rest: str) -> Dict[str, str]:
    fields = {}
    for chunk in rest.split(","):
        key, sep, value = chunk.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def classify(line: str) -> Optional[Dict[str, Any]]:
    """One log line -> an event dict, or None when the line carries no signal."""
    if not line:
        return None
    text = line.strip()
    if not text:
        return None

    if FATAL_SIGNATURE in text:
        return {"type": "fatal_marker", "text": text}

    if FULL_COMPLETE in text:
        return {"type": "complete", "partial": False, "text": text}
    if PARTIAL_COMPLETE in text:
        return {"type": "complete", "partial": True, "text": text}

    m = _BANNER_RE.search(text)
    if m:
        return {"type": "run_started", "fields": _banner_fields(m.group(1))}

    m = _SUBSTAGE_RE.search(text)
    if m:
        number, tail = m.group(1), m.group(2)
        key = SUBSTAGE_KEYS.get(number)
        # The SKIPPED variant must be checked here, not after the "running"
        # rule, or a skipped stage would light up as running and never clear.
        if "SKIPPED" in text:
            return {"type": "stage", "key": key, "number": number,
                    "status": "skipped", "text": tail}
        return {"type": "stage", "key": key, "number": number,
                "status": "running", "text": tail}

    if text.lstrip().startswith("✓") or " ✓ " in text:
        m = _RESULT_RE.search(text)
        if m and m.group(1) in RESULT_TAGS:
            return {"type": "stage", "key": RESULT_TAGS[m.group(1)],
                    "status": "done", "text": m.group(2).strip()}

    m = _PHASE_RE.search(text)
    if m:
        number = int(m.group(1))
        return {"type": "phase", "number": number,
                "title": PHASE_TITLES.get(number, m.group(2).strip())}

    for needle, code, message in _WARNINGS:
        if needle in text:
            if code == "few_detectors" and "detectors available" not in text:
                continue
            return {"type": "warning", "code": code, "text": message or text}

    if "ERROR" in text or text.lstrip().startswith("❌"):
        return {"type": "error", "text": text}

    return None


def is_important(line: str) -> bool:
    """The "important lines only" filter for the log console.

    Mirrors the marker set run_testbed_comprehensive.py already filters on, so
    the console and the batch runner agree on what matters.
    """
    if classify(line) is not None:
        return True
    return any(token in line for token in
               ("STAGE", "Sub-stage", "Generation", "Final Decision",
                "Best ensemble", "WARNING", "ERROR"))
