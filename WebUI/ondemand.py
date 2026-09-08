"""
Figures rendered per request rather than written by the pipeline.

Two families live here:

  * The Thompson ranking gap between an ARBITRARY pair of detectors. Pre-
    rendering it is not an option — eleven detectors is 55 unordered pairs per
    entity, and a reader looks at one or two — but the data it needs is tiny,
    so the pipeline persists that instead and the picture is drawn on demand.
  * The per-window frames of all three Thompson sets (reward, SHAP, ranking).
    The pipeline used to write nine folders of them, ~1,100 PNGs and 167 MB for
    one 173-window entity, of which a reader opens a handful.

The contract that makes this safe:

  * A persisted block is the ONLY input, and it holds the SAME quantity the
    pipeline computed. The gap decomposition is exactly `shares(a) - shares(b)`
    (Thompson_Sampling.rank_gap_decomposition), so a pair drawn here matches the
    pipeline's own `ranking_gap_*.png`; the per-window rows come straight out of
    reward_contribution_per_context_feature / aggregate_shap_per_context_feature /
    aggregate_squared_per_context_feature. Nothing is re-derived here that could drift.
  * Titles, axis labels and footnotes for the per-window frames travel IN the
    persisted file (`kinds`), written by the producer, so this module formats
    them rather than restating them.
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
TOP_N_CONTEXT_FEATURES = 12


def _ir_path(dataset: str, entity: str, stem: str):
    root = paths.MYRESULTS / "explanations_ir"
    directory = paths.resolve_entity_dir(root, dataset, entity)
    if directory is None:
        return None
    candidate = directory / f"{stem}.json"
    return candidate if candidate.is_file() else None


def ranking_context_feature_shares(dataset: str, entity: str) -> Dict[str, List[float]]:
    """Per-detector per-context-feature shares of the ranking score, or {} if absent.

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


def _context_feature_label(index: int, names: Optional[List[str]]) -> str:
    if names and 0 <= index < len(names):
        return str(names[index])
    return f"cf{index}"


def render_ranking_gap(dataset: str, entity: str, model_a: str, model_b: str,
                       context_feature_names: Optional[List[str]] = None,
                       top_n: int = TOP_N_CONTEXT_FEATURES) -> Optional[bytes]:
    """PNG bytes for `model_a` vs `model_b`, or None if the pair is unavailable.

    Returns None rather than raising for every "cannot draw this" case — an
    unknown detector, a tree with no shares, a pair of one detector against
    itself — so the route answers 404 and the page falls back to its default
    pair instead of showing a traceback.
    """
    shares = ranking_context_feature_shares(dataset, entity)
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
    ax.barh([_context_feature_label(c, context_feature_names) for c, _v in pairs],
            [v for _c, v in pairs],
            color=[_AHEAD if v >= 0 else _BEHIND for _c, v in pairs])
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel(r"Contribution to the gap in $\|\mu\|^2$")
    ax.set_title(f"{model_a} vs {model_b}: where the {total:+.6f} margin came from\n"
                 f"(green: {model_a} ahead, red: {model_b} ahead)")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    if len(pairs) < n:
        fig.text(0.5, -0.02,
                 f"The {len(pairs)} context features with the largest difference, of {n}.",
                 ha="center", fontsize=9, alpha=0.8)

    buffer = io.BytesIO()
    plt.tight_layout(pad=1.2)
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


# ── Per-window frames ────────────────────────────────────────────────────────

# Cached by (path, mtime): a browsing session pages 60 frames at a time out of
# one document, and re-reading a few megabytes of JSON per frame would make the
# gallery slower than the folder of PNGs it replaced. Keyed on mtime so a re-run
# invalidates it without the server being restarted.
_PW_CACHE: Dict[Tuple[str, float], Dict[str, Any]] = {}


def per_window_document(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    """The persisted per-window aggregates, or None if this run predates them.

    None is the normal case for a result tree written before the pipeline
    started saving these, so every caller falls back to whatever
    `*_per_window_*` folders are still on disk rather than treating it as an
    error.
    """
    directory = paths.resolve_entity_dir(paths.MYRESULTS / "Thomposon", dataset, entity)
    if directory is None:
        return None
    candidates = sorted(directory.glob("per_window_channels_*.json"))
    if not candidates:
        return None
    path = candidates[-1]
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return None
    if key in _PW_CACHE:
        return _PW_CACHE[key]
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("sets"), dict):
        return None
    _PW_CACHE.clear()          # one entity at a time; the document is megabytes
    _PW_CACHE[key] = doc
    return doc


def _frame_rows(doc: Dict[str, Any], kind: str, t: int) -> Optional[List[List[float]]]:
    frames = (doc.get("sets") or {}).get(kind)
    if not isinstance(frames, list) or not (0 <= t < len(frames)):
        return None
    # null is what the producer writes for a non-finite value, which only
    # happens for a window the run logged an error for. Drawn as zero.
    return [[0.0 if v is None else float(v) for v in row] for row in frames[t]]


def _select_models(doc: Dict[str, Any], kind: str, t: int,
                   scope: str) -> Optional[Tuple[List[int], List[str]]]:
    """Which detectors this frame shows, and in what order.

    Reproduces the eager plots' selection exactly, and can do so from the file
    alone because every criterion is a sum of a stored row: the top-k by
    expected reward is the row sum of the `reward` set, the top-k by score is
    the row sum of the `ranking` set. Sorting is stable over the producer's
    registry order, so a tie breaks the way it did there.
    """
    models: List[str] = list(doc.get("models") or [])
    spec = (doc.get("kinds") or {}).get(kind) or {}
    if not models:
        return None
    if scope == "all" and spec.get("all_by") == "final":
        order = [m for m in (doc.get("models_by_final_norm") or models) if m in models]
        return [models.index(m) for m in order], order
    rank_rows = _frame_rows(doc, str(spec.get("rank_by") or kind), t)
    if rank_rows is None or len(rank_rows) != len(models):
        return None
    ranked = sorted(range(len(models)), key=lambda i: -sum(rank_rows[i]))
    if scope != "all":
        ranked = ranked[:max(1, int(doc.get("top_k_models") or 3))]
    return ranked, [models[i] for i in ranked]


def render_per_window(dataset: str, entity: str, kind: str, t: int,
                      scope: str = "top") -> Optional[bytes]:
    """PNG bytes for one per-window frame, or None if it is unavailable.

    Returns None rather than raising for every "cannot draw this" case — an
    unknown set, a window past the end of the run, a tree with no persisted
    aggregates — so the route answers 404 instead of showing a traceback.
    """
    doc = per_window_document(dataset, entity)
    if doc is None or kind not in (doc.get("sets") or {}):
        return None
    rows = _frame_rows(doc, kind, t)
    selection = _select_models(doc, kind, t, scope)
    if rows is None or selection is None:
        return None
    indices, names = selection
    spec = (doc.get("kinds") or {}).get(kind) or {}
    top_n = max(1, int(doc.get("top_n_channels") or 9))
    n_context_features_total = int(doc.get("n_channels") or 0)

    # The union of each plotted detector's `top_n` largest |values| — the same
    # rule _render_shap_comparison applies, and the same one the footnote states.
    candidates: set = set()
    for i in indices:
        magnitudes = sorted(range(len(rows[i])), key=lambda c: -abs(rows[i][c]))
        candidates.update(magnitudes[:top_n])
    selected = sorted(candidates)
    if not selected:
        return None

    title = (spec.get("title_all") if scope == "all" else spec.get("title_top")) or ""
    title = title.replace("{t}", str(t)).replace(
        "{k}", str(doc.get("top_k_models") or 3))
    note = spec.get("note")
    note = note.replace("{t}", str(t)) if note else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    })
    n_models = len(indices)
    bar_width = 0.8 / max(n_models, 1)
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(selected) + 4), 5))
    for slot, (i, name) in enumerate(zip(indices, names)):
        ax.bar([c + slot * bar_width for c in range(len(selected))],
               [rows[i][c] for c in selected], bar_width, label=name,
               color=plt.cm.tab20(slot / max(n_models, 1)))

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks([c + bar_width * (n_models - 1) / 2 for c in range(len(selected))])
    ax.set_xticklabels([f"cf{c}" for c in selected], rotation=45, ha="right")
    ax.set_xlabel("Context feature")
    ax.set_ylabel(spec.get("ylabel") or "")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(1.01, 1), borderaxespad=0)

    scope_text = (f"{len(selected)} of {n_context_features_total}" if n_context_features_total
                  else f"{len(selected)}")
    rule = (f"Context features shown ({scope_text}): the union over the plotted detectors "
            f"of each one's {top_n} largest |values|. A context feature absent here was "
            f"outside every plotted detector's top {top_n}, not necessarily zero.")
    ax.text(0.0, -0.17, ((note + "  ") if note else "") + rule,
            transform=ax.transAxes, fontsize=7.5, color="dimgrey",
            va="top", ha="left", wrap=True)

    buffer = io.BytesIO()
    plt.tight_layout(pad=1.2)
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()
