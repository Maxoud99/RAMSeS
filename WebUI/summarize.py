"""
The summarisation seam.

Each stage card shows a summary with a disclosure to the full narrative. How to
summarise is not decided yet, so today the summary IS the full text. This module
exists so that when it is decided, exactly one function changes.

The contract that makes that true:

* `artifacts.build_payload` is the **only** caller.
* The API always returns both `summary` and `full`, plus `summary_is_full`. The
  frontend keys its disclosure off that flag and renders the same DOM either
  way, so a real summariser changes the payload and nothing else.
* `summarize` receives the **narrative only**, never the INFO glossary — the
  glossary is fixed boilerplate, identical across runs, and would dominate any
  extractive summariser.
* `stage` is in the signature from day one so a per-stage summariser can branch
  without touching callers.
* A summariser that is slow (an LLM) or fails must never break the page:
  `summarize` catches its own errors and falls back to the full text.
"""

from typing import Optional

# "full" | "first_paragraph". `first_paragraph` is not used in production; it
# exists so the contract demonstrably has a second implementation.
SUMMARY_MODE = "full"


def _first_paragraph(text: str) -> str:
    return text.split("\n\n", 1)[0].strip()


def summarize(text: str, *, stage: Optional[str] = None) -> dict:
    """Narrative text -> {"summary", "is_full", "mode"}.

    `is_full` is True when the summary is the whole narrative, which tells the
    frontend to render the disclosure pre-expanded and labelled "Full text"
    instead of offering a redundant expand.
    """
    body = (text or "").strip()
    if not body:
        return {"summary": "", "is_full": True, "mode": SUMMARY_MODE}

    try:
        if SUMMARY_MODE == "first_paragraph":
            head = _first_paragraph(body)
            return {"summary": head, "is_full": head == body,
                    "mode": "first_paragraph"}
    except Exception:
        pass  # any summariser failure degrades to the full text

    return {"summary": body, "is_full": True, "mode": "full"}
