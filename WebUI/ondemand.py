"""
Figures rendered per request rather than written by the pipeline.

One figure lives here: the Thompson ranking gap between an ARBITRARY pair of
detectors. Pre-rendering it is not an option — eleven detectors is 55 unordered
pairs per entity, and a reader looks at one or two — but the data it needs is
tiny, so the pipeline persists that instead and the picture is drawn on demand.

The contract that makes this safe:

  * The IR's `channel_shares` block is the ONLY input. The gap decomposition is
    exactly `shares(a) - shares(b)` (Thompson_Sampling.rank_gap_decomposition),
    so a pair drawn here is the same quantity the pipeline's own
    `ranking_gap_*.png` shows for the winner and runner-up — not a re-derivation
    that could drift from it.
  * Nothing is written to `myresults/`. These bytes are a response, so a
    browsing session cannot litter the result tree or race the pipeline.
  * matplotlib is imported lazily and pinned to Agg. The web process should not
    pay for it, or try to open a window, unless someone asks for a figure.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Tuple

from WebUI import paths

# Matches Thompson_Sampling.plot_ranking_gap, so the on-demand figure and the
# pipeline's own cannot disagree about which colour means what.
_AHEAD = "#2F9E44"
_BEHIND = "#C92A2A"
TOP_N_CHANNELS = 12


def _ir_path(dataset: str, entity: str, stem: str):
    root = paths.MYRESULTS / "explanations_ir"
    directory = paths.resolve_entity_dir(root, dataset, entity)
    if directory is None:
        return None
    candidate = directory / f"{stem}.json"
    return candidate if candidate.is_file() else None


def ranking_channel_shares(dataset: str, entity: str) -> Dict[str, List[float]]:
    """Per-detector per-channel shares of the ranking score, or {} if absent.

    Absent is the normal case for a result tree written before this block
    existed, so every caller treats {} as "offer nothing" rather than an error.
    """
    path = _ir_path(dataset, entity, "ir_thompson_ranking")
    if path is None:
        return {}
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return {}
    shares = doc.get("channel_shares")
    if not isinstance(shares, dict):
        return {}
    out: Dict[str, List[float]] = {}
    for name, values in shares.items():
        if isinstance(values, (list, tuple)):
            out[str(name)] = [float(v) for v in values
                              if isinstance(v, (int, float))]
    return out


def _channel_label(index: int, names: Optional[List[str]]) -> str:
    if names and 0 <= index < len(names):
        return str(names[index])
    return f"ch{index}"


def render_ranking_gap(dataset: str, entity: str, model_a: str, model_b: str,
                       channel_names: Optional[List[str]] = None,
                       top_n: int = TOP_N_CHANNELS) -> Optional[bytes]:
    """PNG bytes for `model_a` vs `model_b`, or None if the pair is unavailable.

    Returns None rather than raising for every "cannot draw this" case — an
    unknown detector, a tree with no shares, a pair of one detector against
    itself — so the route answers 404 and the page falls back to its default
    pair instead of showing a traceback.
    """
    shares = ranking_channel_shares(dataset, entity)
    if model_a not in shares or model_b not in shares or model_a == model_b:
        return None
    a, b = shares[model_a], shares[model_b]
    n = min(len(a), len(b))
    if n == 0:
        return None

    gap = [a[i] - b[i] for i in range(n)]
    total = sum(gap)
    # Largest movers either way, then re-sorted so the bars run smallest to
    # largest — the same two steps plot_ranking_gap takes.
    ranked = sorted(range(n), key=lambda i: abs(gap[i]), reverse=True)[:max(1, top_n)]
    pairs: List[Tuple[int, float]] = sorted(((i, gap[i]) for i in ranked),
                                            key=lambda cv: cv[1])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(pairs) + 1.5)))
    ax.barh([_channel_label(c, channel_names) for c, _v in pairs],
            [v for _c, v in pairs],
            color=[_AHEAD if v >= 0 else _BEHIND for _c, v in pairs])
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel(r"Contribution to the gap in $\|\mu\|^2$")
    ax.set_title(f"{model_a} vs {model_b}: where the {total:+.6f} margin came from\n"
                 f"(green: {model_a} ahead, red: {model_b} ahead)")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    if len(pairs) < n:
        fig.text(0.5, -0.02,
                 f"The {len(pairs)} channels with the largest difference, of {n}.",
                 ha="center", fontsize=9, alpha=0.8)

    buffer = io.BytesIO()
    plt.tight_layout(pad=1.2)
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()
