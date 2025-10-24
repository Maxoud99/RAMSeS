# -*- coding: utf-8 -*-
"""
Bar chart (F1) + thumbnails row, where thumbnails can be:
- loaded from CSV/TXT (uni-/multi-variate),
- loaded from an image file (png/jpg),
- or synthesized as a fallback.

Random spikes (5–15) are injected into each line-series thumbnail.
Everything is configured in the CONFIG section below.
Run this file directly.
"""

from __future__ import annotations
import os
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import pandas as pd

# ------------------------ CONFIG ------------------------
OUTPUT_PNG = "f1_panels_from_files.png"
OUTPUT_PDF = "f1_panels_from_files.pdf"

LINE_COLOR = "#1f77b4"
SPIKE_MARK_COLOR = "red"

# Show only first N points of any series
MAX_POINTS = 500

# Hatched legend order (length must equal number of methods)
HATCHES = ["///", "|||", "++", "xx", "\\\\", "...."]

# Methods (bars on top)
METHODS = ["RNN","LSTMVAE","KNN","LOF","GMM","ABOD"]

# Categories (columns) — 4 types
CATEGORIES = ["Prices", "Medical", "Industry", "environment"]

# F1 matrix must be (len(METHODS), len(CATEGORIES)) => (6, 4)
F1S = np.array([
    [0.7, 0.62, 0.65, 0.72],   # RNN
    [0.76, 0.073, 0.6, 0.74],  # LSTMVAE
    [0.44, 0.05, 0.45, 0.44],  # KNN
    [0.62, 0.122, 0.75, 0.78], # LOF
    [0.14, 0.08, 0.77, 0.59],  # GMM
    [0.36, 0.08, 0.7, 0.64],   # ABOD
], dtype=float)

# Thumbnail sources, one per category, in order.
# If a TXT/CSV has no header, we’ll use the FIRST COLUMN as the value and x = 0..N-1.
THUMBNAILS: List[Dict] = [
    dict(kind="csv", path="/home/maxoud/projects/RAMS-TSAD/Mononito/datasets/TCPD/apple/apple.csv",
         time_col=None, value_col=None),
    dict(kind="txt", path="Mononito/datasets/Anomaly_Archive/017_UCR_Anomaly_DISTORTEDECG4_5000_17000_17100.txt",
         time_col=None, value_col=None),
    dict(kind="csv", path="/home/maxoud/projects/RAMS-TSAD/Mononito/datasets/SKAB/10.csv",
         time_col="datetime", value_col="Accelerometer1RMS"),
    dict(kind="csv", path="/home/maxoud/projects/RAMS-TSAD/Mononito/datasets/SKAB/0.csv",
         time_col="datetime", value_col="Temperature"),
]

# Figure size and layout
FIGSIZE = (11, 4.4)
TOP_HEIGHT_RATIO = 3.0
BOTTOM_HEIGHT_RATIO = 1.2

# -------- Spike Injection (global defaults; can be overridden per spec['spike_kwargs']) --------
# Number of spikes per series (inclusive)
MIN_SPIKES = 5
MAX_SPIKES = 15
# Magnitude mode: "std" uses k * std(y); "abs" uses absolute units
SPIKE_MAG_MODE = "std"
SPIKE_MAG_STD_RANGE = (1.0, 3.0)   # k in [1,3] * std
SPIKE_MAG_ABS_RANGE = (0.2, 0.8)   # used if mode == "abs"
SPIKE_DIRECTION = "both"           # 'up' | 'down' | 'both'
MARK_ALL_SPIKES = False            # set True to mark all spikes with red vlines
# -----------------------------------------------------------------------------------------------


def _read_table(path: str, kind: str) -> pd.DataFrame:
    """Robust read for CSV/TXT with unknown delimiter.
    TXT: force header=None so first row is data; CSV: let pandas infer headers."""
    if kind == "txt":
        try:
            return pd.read_csv(path, sep=None, engine="python", header=None)
        except Exception:
            return pd.read_csv(path, sep=r"\s+", engine="python", header=None)
    else:  # csv
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path)


def _to_xy_from_dataframe(df: pd.DataFrame, time_col: Optional[str], value_col: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract x (time) and y (series).
    If time_col is None or missing, x = 0..N-1.
    If value_col is None, use first numeric column; if none, coerce FIRST COLUMN."""
    # --- y selection ---
    if value_col is not None and value_col in df.columns:
        y_series = pd.to_numeric(df[value_col], errors="coerce")
    else:
        cand_cols = [c for c in df.columns if c != time_col]
        num_cols = [c for c in cand_cols if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            y_series = pd.to_numeric(df[num_cols[0]], errors="coerce")
        else:
            y_series = pd.to_numeric(df.iloc[:, 0], errors="coerce")

    y = y_series.to_numpy()
    y = y[np.isfinite(y)]
    n = len(y)

    # --- x selection ---
    if time_col is None or (time_col not in df.columns):
        x = np.arange(n, dtype=float)
    else:
        xraw = df[time_col]
        if np.issubdtype(xraw.dtype, np.datetime64):
            x = xraw.astype("datetime64[ns]").map(pd.Timestamp.toordinal).to_numpy(dtype=float)
        else:
            xnum = pd.to_numeric(xraw, errors="coerce")
            if xnum.isna().all():
                x = np.arange(n, dtype=float)
            else:
                x = xnum.fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)
                if len(x) != n:
                    x = np.arange(n, dtype=float)

    if len(x) != n:
        m = min(len(x), n)
        x, y = x[:m], y[:m]
    return x, y


def _load_series_from_spec(spec: Dict) -> Tuple[str, Union[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
    """Return ("line", (x,y)) for data series, or ("image", img_array) for image sources."""
    kind = spec.get("kind", "synthetic")
    path = spec.get("path")
    time_col = spec.get("time_col")
    value_col = spec.get("value_col")

    if kind in ("csv", "txt"):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found for spec {spec}")
        df = _read_table(path, kind)
        x, y = _to_xy_from_dataframe(df, time_col=time_col, value_col=value_col)
        return "line", (x, y)

    elif kind == "image":
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found for spec {spec}")
        img = plt.imread(path)
        return "image", img

    else:
        # synthetic fallback
        name = spec.get("name", "")
        x = np.linspace(0, 10, 220)
        rng = np.random.default_rng(0)
        if "Prices" in name:
            y = 0.5 * (1 - np.exp(-0.4 * x)) * np.sin(1.1 * x) + 0.6
        elif "Medical" in name:
            y = 0.2 * np.sin(2.5 * x) + 0.6
        elif "Industry" in name:
            y = 0.15 * np.sin(6 * x) + 0.5
        else:  # environment
            y = 0.08 * np.sin(10 * x) + 0.45 + 0.05 * rng.normal(size=x.size)
        return "line", (x, y)


def _inject_multiple_spikes(
    y: np.ndarray,
    min_spikes: int = MIN_SPIKES,
    max_spikes: int = MAX_SPIKES,
    magnitude_mode: str = SPIKE_MAG_MODE,        # "std" or "abs"
    std_range: Tuple[float, float] = SPIKE_MAG_STD_RANGE,
    abs_range: Tuple[float, float] = SPIKE_MAG_ABS_RANGE,
    direction: str = SPIKE_DIRECTION,            # 'up'|'down'|'both'
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject K random spikes (K in [min_spikes, max_spikes]) at unique indices.
    Returns (y_new, spike_indices)."""
    if rng is None:
        rng = np.random.default_rng()
    y2 = np.asarray(y, dtype=float).copy()
    n = len(y2)
    if n == 0:
        return y2, np.array([], dtype=int)

    k = int(rng.integers(min_spikes, max_spikes + 1))
    k = min(k, n)  # handle very short series
    idxs = np.unique(rng.choice(n, size=k, replace=False))

    if magnitude_mode == "std":
        base = float(np.nanstd(y2))
        base = base if base > 0 else 1.0
        mags = rng.uniform(std_range[0], std_range[1], size=idxs.size) * base
    else:
        mags = rng.uniform(abs_range[0], abs_range[1], size=idxs.size)

    if direction == "up":
        signs = np.ones_like(idxs, dtype=float)
    elif direction == "down":
        signs = -np.ones_like(idxs, dtype=float)
    else:
        signs = rng.choice([-1.0, 1.0], size=idxs.size)

    y2[idxs] = y2[idxs] + signs * mags
    return y2, idxs


def build_figure(methods: List[str], categories: List[str], f1s: np.ndarray,
                 thumbnails: List[Dict],
                 figsize=(11, 4.4), top_ratio=3.0, bottom_ratio=1.2) -> plt.Figure:
    assert len(HATCHES) == len(methods), "HATCHES length must match METHODS length"
    assert len(categories) == len(thumbnails), "CATEGORIES and THUMBNAILS must have same length"
    assert f1s.shape == (len(methods), len(categories)), \
        f"F1S shape {f1s.shape} != ({len(methods)}, {len(categories)})"

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(2, len(categories), figure=fig, height_ratios=[top_ratio, bottom_ratio])

    # ---- Top bar chart (F1) ----
    ax_bar = fig.add_subplot(gs[0, :])
    n_methods, n_cats = f1s.shape
    group_x = np.arange(n_cats)
    bar_w = 0.12
    offsets = np.linspace(-((n_methods-1)/2)*bar_w, ((n_methods-1)/2)*bar_w, n_methods)

    for m in range(n_methods):
        ax_bar.bar(group_x + offsets[m], f1s[m], width=bar_w, edgecolor="black",
                   linewidth=0.8, hatch=HATCHES[m], label=methods[m], facecolor="white")

    handles = [Patch(facecolor="white", edgecolor="black", hatch=HATCHES[i], label=methods[i])
               for i in range(n_methods)]

    winners = f1s.argmax(axis=0)
    for c in range(n_cats):
        m = winners[c]
        x = group_x[c] + offsets[m]
        y = float(f1s[m, c])
        ax_bar.plot([x], [min(1.0, y + 0.02)], marker=(3, 0, 180), markersize=12, color="red", clip_on=False)

    ax_bar.set_ylabel("F1")
    ax_bar.set_xticks(group_x)
    ax_bar.set_xticklabels([])
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.legend(handles=handles, ncol=3, frameon=False, loc="upper left")

    # ---- Bottom thumbnails ----
    for c, cat in enumerate(categories):
        spec = dict(thumbnails[c])
        spec["name"] = cat  # for synthetic generator
        ax = fig.add_subplot(gs[1, c])

        mode, payload = _load_series_from_spec(spec)

        if mode == "line":
            x, y = payload
            # keep only first MAX_POINTS
            maxn = min(len(y), MAX_POINTS)
            x, y = x[:maxn], y[:maxn]

            # Per-spec overrides for spike settings (optional)
            skw = spec.get("spike_kwargs", {}) if isinstance(spec.get("spike_kwargs"), dict) else {}
            y, spike_idxs = _inject_multiple_spikes(
                y=y,
                min_spikes=int(skw.get("min_spikes", MIN_SPIKES)),
                max_spikes=int(skw.get("max_spikes", MAX_SPIKES)),
                magnitude_mode=str(skw.get("magnitude_mode", SPIKE_MAG_MODE)),
                std_range=tuple(skw.get("std_range", SPIKE_MAG_STD_RANGE)),
                abs_range=tuple(skw.get("abs_range", SPIKE_MAG_ABS_RANGE)),
                direction=str(skw.get("direction", SPIKE_DIRECTION)),
            )

            ax.plot(x, y, linewidth=1.0, color=LINE_COLOR, marker=None)

            if MARK_ALL_SPIKES or spec.get("mark_spike", True):
                xx = x if isinstance(x, np.ndarray) else np.asarray(x)
                ax.vlines(xx[spike_idxs], ymin=np.nanmin(y), ymax=np.nanmax(y),
                          color=SPIKE_MARK_COLOR, linewidth=1.0, alpha=0.8)

        else:
            img = payload
            ax.imshow(img, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(True)

        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.set_title(cat, pad=6, fontsize=11)

    return fig


def main():
    fig = build_figure(METHODS, CATEGORIES, F1S, THUMBNAILS,
                       figsize=FIGSIZE, top_ratio=TOP_HEIGHT_RATIO, bottom_ratio=BOTTOM_HEIGHT_RATIO)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"[OK] Saved: {OUTPUT_PNG} and {OUTPUT_PDF}")


if __name__ == "__main__":
    main()