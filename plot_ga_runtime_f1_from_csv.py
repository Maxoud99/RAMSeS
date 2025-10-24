# plot_ga_runtime_f1_from_csv.py
# Reads GA results CSV and draws the 2x3 ICDE figure (Runtime row + F1 row).
# - Finds the latest output/ga_runtime_*/ga_runtime_results.csv automatically,
#   or use RESULTS_CSV to hard-set a path.
# - All y-axes start at 0 (F1 axes are capped at 1).
# - Panel A uses power-law fit (log-log) with R^2 annotation.
# - F1 row (D/E/F) includes an inset zoom (default 0.80–1.00) with red indicator.

import os
import re
import glob
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# -----------------------------
# (Optional) hard-set CSV path
# -----------------------------
RESULTS_CSV: Optional[str] = "/home/maxoud/projects/RAMSeS/output/ga_runtime_20251024-145901/ga_runtime_results.csv"

# Workloads used in the paper/experiments
WORKLOADS: Dict[str, Tuple[int, int]] = {
    "P= 5, G= 10":   (5, 10),
    "P= 50, G= 20":  (50, 20),
    "P= 100, G= 50": (100, 50),
}
META_SET = ["rf", "lr", "svm"]
MUTATIONS = [0.0, 0.05, 0.2, 0.5, 1.0]  # for ordering on the x-axis of mutation plots
P_FIX, G_FIX = 50, 10                   # default for mutation sweep; will auto-adjust if missing

# -----------------------------
# Inset (zoom) config for F1 row
# -----------------------------
INSET_ENABLED = True
INSET_YLIM = (0.75, .85)     # nominal F1 zoom band
INSET_WIDTH = "40%"           # relative to parent axes
INSET_HEIGHT = "45%"
INSET_LOC = "center right"     # 'upper left' | 'upper right' | 'lower left' | 'lower right'
INSET_BORDERPAD = 1.0
INSET_TICK_FONTSIZE = 6
INSET_DRAW_BOX = True

# Make the indicator GRAY and keep it fully INSIDE the axes
INSET_EDGE_COLOR = "gray"
INSET_EDGE_LW = 1.2
INSET_EDGE_ALPHA = 0.95
# Safety margin as a fraction of axis span so the rectangle never touches the border
INSET_PAD_FRAC_X = 0.0
INSET_PAD_FRAC_Y = 0.0

# -----------------------------
# Data model
# -----------------------------
@dataclass
class RunResult:
    dataset: str
    entity: str
    P: int
    G: int
    mu: float
    meta: str
    run_idx: int
    runtime_s: float
    best_f1: float
    best_pr_auc: float
    ensemble_size: int
    error: str = ""

# -----------------------------
# Style
# -----------------------------
def icde_style():
    plt.rcParams.update({
        "figure.figsize": (13, 8.6),   # 2 rows x 3 cols; wide but taller
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "errorbar.capsize": 3,
        "scatter.marker": "o",
    })

# -----------------------------
# Helpers
# -----------------------------
def _find_latest_csv() -> str:
    candidates = glob.glob(os.path.join("output", "ga_runtime_*", "ga_runtime_results.csv"))
    if not candidates:
        raise FileNotFoundError("Could not find any output/ga_runtime_*/ga_runtime_results.csv")
    pairs = []
    for c in candidates:
        d = os.path.dirname(c)
        try:
            m = os.path.getmtime(d)
        except OSError:
            m = 0
        pairs.append((m, c))
    pairs.sort(reverse=True)
    return pairs[0][1]

def load_results(csv_path: str) -> List[RunResult]:
    rows: List[RunResult] = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            def ffloat(key, default=np.nan):
                try:
                    return float(d[key])
                except Exception:
                    return float('nan') if default is np.nan else default
            def fint(key, default=0):
                try:
                    return int(d[key])
                except Exception:
                    return default
            rows.append(RunResult(
                dataset=d.get("dataset",""),
                entity=d.get("entity",""),
                P=fint("P"),
                G=fint("G"),
                mu=ffloat("mu"),
                meta=str(d.get("meta","")).strip(),
                run_idx=fint("run_idx"),
                runtime_s=ffloat("runtime_s"),
                best_f1=ffloat("best_f1"),
                best_pr_auc=ffloat("best_pr_auc"),
                ensemble_size=fint("ensemble_size"),
                error=d.get("error",""),
            ))
    if not rows:
        raise RuntimeError(f"No rows read from {csv_path}")
    return rows

def compute_group_stats(
    rows: List[RunResult],
    key_fields: Tuple[str, ...],
    value_selector: Callable[[RunResult], float]
):
    groups: Dict[Tuple, List[RunResult]] = {}
    for r in rows:
        k = tuple(getattr(r, f) for f in key_fields) if key_fields else ("_all",)
        groups.setdefault(k, []).append(r)

    out = []
    for k, vs in groups.items():
        vals = np.array([value_selector(v) for v in vs if math.isfinite(value_selector(v))])
        if len(vals) == 0:
            med = q25 = q75 = float('nan')
        else:
            med = float(np.median(vals))
            q25 = float(np.percentile(vals, 25))
            q75 = float(np.percentile(vals, 75))
        rec = {}
        if key_fields:
            rec.update({fld: val for fld, val in zip(key_fields, k)})
        rec.update(dict(n=len(vs), median=med, q25=q25, q75=q75))
        out.append(rec)
    return out

# --- UPDATED: accept plot kwargs and use them (for red fit, etc.) ---
def _powerlaw_fit_and_plot(ax, x, y, label=None, show_r2=True, **plot_kwargs):
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
    if good.sum() < 2:
        return (np.nan, np.nan, np.nan)
    lx, ly = np.log10(x[good]), np.log10(np.maximum(y[good], 1e-12))
    a, b = np.polyfit(lx, ly, deg=1)
    xs = np.logspace(lx.min(), lx.max(), 200)
    ys = 10 ** (a * np.log10(xs) + b)
    ax.plot(xs, ys, label=label, **plot_kwargs)

    yhat = a * lx + b
    ss_res = np.sum((ly - yhat) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)
    if show_r2:
        ax.text(0.05, 0.95, f"R²={r2:.3f}", transform=ax.transAxes, ha="left", va="top")
    return (a, 10**b, r2)

# --- UPDATED: allow per-series fit styling via 'fit_kwargs' (for red inset fit) ---
def _add_inset_zoom(ax, series_list, ylims=INSET_YLIM,
                    width=INSET_WIDTH, height=INSET_HEIGHT,
                    loc=INSET_LOC, borderpad=INSET_BORDERPAD,
                    tick_fontsize=INSET_TICK_FONTSIZE,
                    draw_box=INSET_DRAW_BOX):
    """
    Add an inset zoom to 'ax' and re-plot provided series.
    series_list: list of dicts, each with:
        - x: np.ndarray
        - y: np.ndarray
        - ylo: np.ndarray or None
        - yhi: np.ndarray or None
        - kwargs: dict for matplotlib plotting (e.g., {'fmt':'o', 'linestyle':'none'})
        - fit_powerlaw: bool (optional) -> draw power-law fit (uses the same x,y)
        - fit_kwargs: dict (optional) -> kwargs for the fit plot (e.g., {'color':'red'})
    """
    iax = inset_axes(ax, width=width, height=height, loc=loc, borderpad=borderpad)
    iax.set_xscale(ax.get_xscale())

    # Clamp inset limits to stay safely INSIDE the parent axes
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xpad = (x1 - x0) * INSET_PAD_FRAC_X
    ypad = (y1 - y0) * INSET_PAD_FRAC_Y

    # Keep same x-range to emphasize y-zoom
    iax.set_xlim(x0 + xpad, x1 - xpad)

    y_low_req, y_high_req = ylims
    y_low = max(y0 + ypad, y_low_req)
    y_high = min(y1 - ypad, y_high_req)
    if y_high <= y_low:  # fallback in pathological cases
        y_low = y0 + ypad
        y_high = y1 - ypad
    iax.set_ylim(y_low, y_high)

    iax.tick_params(labelsize=tick_fontsize)
    iax.grid(False)
    iax.set_xlabel("")
    iax.set_ylabel("")

    for s in series_list:
        x = np.asarray(s.get("x", []), float)
        y = np.asarray(s.get("y", []), float)
        ylo = s.get("ylo", None)
        yhi = s.get("yhi", None)
        kwargs = dict(s.get("kwargs", {}))
        good = np.isfinite(x) & np.isfinite(y)
        if ylo is not None and yhi is not None:
            ylo = np.asarray(ylo, float)
            yhi = np.asarray(yhi, float)
            good = good & np.isfinite(ylo) & np.isfinite(yhi)
            if good.any():
                iax.errorbar(x[good], y[good], yerr=[ylo[good], yhi[good]], **kwargs)
        else:
            if good.any():
                iax.plot(x[good], y[good], **kwargs)

        if s.get("fit_powerlaw", False) and good.sum() >= 2:
            _powerlaw_fit_and_plot(
                iax, x[good], y[good],
                show_r2=False,
                **s.get("fit_kwargs", {})  # pass inset fit style (e.g., color='red')
            )

    if draw_box:
        try:
            # Get handles and force red + clipping to keep everything inside
            pp, p1, p2 = ax.indicate_inset_zoom(
                iax,
                edgecolor=INSET_EDGE_COLOR,
                lw=INSET_EDGE_LW,
                alpha=INSET_EDGE_ALPHA,
            )
            # Patch (rectangle)
            pp.set_edgecolor(INSET_EDGE_COLOR)
            pp.set_linewidth(INSET_EDGE_LW)
            pp.set_alpha(INSET_EDGE_ALPHA)
            pp.set_clip_on(True)
            pp.set_clip_path(ax.patch)

            # Connector lines
            for line in (p1, p2):
                if line is not None:
                    line.set_color(INSET_EDGE_COLOR)
                    line.set_linewidth(INSET_EDGE_LW)
                    line.set_alpha(INSET_EDGE_ALPHA)
                    line.set_clip_on(True)
                    line.set_clip_path(ax.patch)
        except Exception:
            pass
    return iax

# -----------------------------
# Plotting
# -----------------------------
def plot_runtime_and_f1_from_rows(rows: List[RunResult], out_png: str) -> None:
    icde_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    axA, axB, axC = axes[0]
    axD, axE, axF = axes[1]

    # (A) Runtime vs Work (μ=0.2, meta=rf)
    subset_A = [r for r in rows if r.meta == "rf" and abs(r.mu - 0.2) < 1e-9]
    stats_A = compute_group_stats(subset_A, key_fields=("P","G"), value_selector=lambda r: r.runtime_s)
    stats_A = sorted(stats_A, key=lambda d: (d["P"] * d["G"], d["P"], d["G"]))
    x_work = np.array([d["P"] * d["G"] for d in stats_A], dtype=float)
    y_med  = np.array([d["median"] for d in stats_A], dtype=float)
    y_lo   = np.array([d["median"] - d["q25"] for d in stats_A], dtype=float)
    y_hi   = np.array([d["q75"] - d["median"] for d in stats_A], dtype=float)
    axA.errorbar(x_work, y_med, yerr=[y_lo, y_hi], fmt='o', linestyle='none')

    # --- RED, labeled fit so legend shows a small red line ---
    _powerlaw_fit_and_plot(
        axA, x_work, y_med,
        label="Linear-time fit",
        show_r2=True,
        color="red"
    )
    axA.set_xscale("log")
    axA.set_xlabel("Work (population × generations)")
    axA.set_ylabel("Runtime (s)")
    axA.set_title("A) Runtime vs Work (μ=0.2, meta=rf)")
    axA.set_ylim(bottom=0)

    # Legend for A (small line sample)
    axA.legend(
        title="Linear time curve in red",
        ncols=1,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=True,
        borderaxespad=0.0,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.3
    )

    # Determine (P_fix, G_fix) automatically if needed
    global P_FIX, G_FIX
    found = any((r.P == P_FIX and r.G == G_FIX) for r in rows)
    if not found:
        from collections import Counter
        cnt = Counter((r.P, r.G) for r in rows)
        (P_FIX, G_FIX), _ = cnt.most_common(1)[0]

    # (B) Runtime vs μ at fixed workload — one line per meta
    for meta in META_SET:
        sub = [r for r in rows if r.P == P_FIX and r.G == G_FIX and r.meta == meta]
        statB = compute_group_stats(sub, key_fields=("mu",), value_selector=lambda r: r.runtime_s)
        statB = sorted(statB, key=lambda d: MUTATIONS.index(d["mu"]) if d["mu"] in MUTATIONS else d["mu"])
        x_mu = np.array([d["mu"] for d in statB], float)
        y_med = np.array([d["median"] for d in statB], float)
        y_lo  = np.array([d["median"] - d["q25"] for d in statB], float)
        y_hi  = np.array([d["q75"] - d["median"] for d in statB], float)
        if len(x_mu) > 0:
            axB.errorbar(x_mu, y_med, yerr=[y_lo, y_hi], marker='o', label=meta)
    axB.set_xlabel("Mutation rate (μ)")
    axB.set_ylabel("Runtime (s)")
    axB.set_title(f"B) Runtime vs Mutation (P={P_FIX}, G={G_FIX})")
    axB.set_ylim(bottom=0)
    axB.legend(
        title="meta",
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=True,
        borderaxespad=0.0
    )

    # (C) Runtime vs meta at 3 workloads (μ=0.2) — BAR PLOT
    x_positions = np.arange(len(WORKLOADS))
    BAR_WIDTH = 0.25
    offset_map = {'rf': -BAR_WIDTH, 'lr': 0.0, 'svm': BAR_WIDTH}

    for meta in META_SET:
        ys, yerr_lo, yerr_hi = [], [], []
        for i, (label, (P, G)) in enumerate(WORKLOADS.items()):
            sub = [r for r in rows if r.P == P and r.G == G and r.meta == meta and abs(r.mu - 0.2) < 1e-9]
            statC = compute_group_stats(sub, key_fields=tuple(), value_selector=lambda r: r.runtime_s)
            if len(statC) == 1:
                s = statC[0]
                ys.append(s["median"])
                yerr_lo.append(max(0.0, s["median"] - s["q25"]))
                yerr_hi.append(max(0.0, s["q75"] - s["median"]))
            else:
                ys.append(np.nan); yerr_lo.append(0.0); yerr_hi.append(0.0)

        ys_arr = np.array(ys, float)
        ys_arr[~np.isfinite(ys_arr)] = 0.0  # avoid warnings for NaN bar heights

        axC.bar(
            x_positions + offset_map[meta], ys_arr, width=BAR_WIDTH,
            yerr=[yerr_lo, yerr_hi], capsize=3, label=meta, align='center'
        )

    axC.set_xticks(x_positions, list(WORKLOADS.keys()))
    axC.set_xlabel("Workload")
    axC.set_ylabel("Runtime (s)")
    axC.set_title("C) Runtime vs Meta at 3 Workloads (μ=0.2)")
    axC.set_ylim(bottom=0)
    axC.legend(
        title="meta",
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=True,
        borderaxespad=0.0
    )

    # (D) F1 vs Work (μ=0.2, meta=rf)
    stats_D = compute_group_stats(subset_A, key_fields=("P","G"), value_selector=lambda r: r.best_f1)
    stats_D = sorted(stats_D, key=lambda d: (d["P"] * d["G"], d["P"], d["G"]))
    xw  = np.array([d["P"] * d["G"] for d in stats_D], float)
    ym  = np.array([d["median"] for d in stats_D], float)
    ylo = np.array([d["median"] - d["q25"] for d in stats_D], float)
    yhi = np.array([d["q75"] - d["median"] for d in stats_D], float)
    axD.errorbar(xw, ym, yerr=[ylo, yhi], fmt='o', linestyle='none')

    # --- RED, labeled fit + legend (and red inset fit) ---
    _powerlaw_fit_and_plot(
        axD, xw, ym,
        label="Linear-time fit",
        show_r2=True,
        color="red"
    )
    axD.set_xscale("log")
    axD.set_xlabel("Work (population × generations)")
    axD.set_ylabel("F1")
    axD.set_title("D) F1 vs Work (μ=0.2, meta=rf)")
    axD.set_ylim(0, 1)  # cap at 1 for F1
    if INSET_ENABLED:
        _add_inset_zoom(
            axD,
            series_list=[{
                "x": xw, "y": ym, "ylo": ylo, "yhi": yhi,
                "kwargs": {"fmt": "o", "linestyle": "none"},
                "fit_powerlaw": True,
                "fit_kwargs": {"color": "red"}  # inset fit also red
            }],
        )
    axD.legend(
        title="Linear time curve in red",
        ncols=1,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=True,
        borderaxespad=0.0,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.3
    )

    # (E) F1 vs μ at fixed workload — one line per meta
    inset_series_E = []
    for meta in META_SET:
        subE = [r for r in rows if r.P == P_FIX and r.G == G_FIX and r.meta == meta]
        statE = compute_group_stats(subE, key_fields=("mu",), value_selector=lambda r: r.best_f1)
        statE = sorted(statE, key=lambda d: MUTATIONS.index(d["mu"]) if d["mu"] in MUTATIONS else d["mu"])
        x_mu = np.array([d["mu"] for d in statE], float)
        y_med = np.array([d["median"] for d in statE], float)
        y_lo  = np.array([d["median"] - d["q25"] for d in statE], float)
        y_hi  = np.array([d["q75"] - d["median"] for d in statE], float)
        if len(x_mu) > 0:
            axE.errorbar(x_mu, y_med, yerr=[y_lo, y_hi], marker='o', label=meta)
            if INSET_ENABLED:
                inset_series_E.append({
                    "x": x_mu, "y": y_med, "ylo": y_lo, "yhi": y_hi,
                    "kwargs": {"marker": "o", "linestyle": "none"}
                })
    axE.set_xlabel("Mutation rate (μ)")
    axE.set_ylabel("F1")
    axE.set_title(f"E) F1 vs Mutation (P={P_FIX}, G={G_FIX})")
    axE.set_ylim(0, 1)  # cap at 1 for F1
    axE.legend(
        title="meta",
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=True,
        borderaxespad=0.0
    )
    if INSET_ENABLED and inset_series_E:
        _add_inset_zoom(axE, series_list=inset_series_E)

    # (F) F1 vs meta at 3 workloads (μ=0.2) — BAR PLOT (+ inset bars)
    BAR_WIDTH = 0.25
    offset_map = {'rf': -BAR_WIDTH, 'lr': 0.0, 'svm': BAR_WIDTH}
    x_positions = np.arange(len(WORKLOADS))

    # Keep data for inset
    bar_Y, bar_Elo, bar_Ehi = {}, {}, {}

    for meta in META_SET:
        ys, yerr_lo, yerr_hi = [], [], []
        for i, (label, (P, G)) in enumerate(WORKLOADS.items()):
            subF = [r for r in rows if r.P == P and r.G == G and r.meta == meta and abs(r.mu - 0.2) < 1e-9]
            statF = compute_group_stats(subF, key_fields=tuple(), value_selector=lambda r: r.best_f1)
            if len(statF) == 1:
                s = statF[0]
                ys.append(s["median"])
                yerr_lo.append(max(0.0, s["median"] - s["q25"]))
                yerr_hi.append(max(0.0, s["q75"] - s["median"]))
            else:
                ys.append(np.nan); yerr_lo.append(0.0); yerr_hi.append(0.0)

        ys_arr = np.array(ys, float)
        ys_arr[~np.isfinite(ys_arr)] = 0.0

        bar_Y[meta] = ys_arr
        bar_Elo[meta] = np.array(yerr_lo, float)
        bar_Ehi[meta] = np.array(yerr_hi, float)

        axF.bar(
            x_positions + offset_map[meta], ys_arr, width=BAR_WIDTH,
            yerr=[bar_Elo[meta], bar_Ehi[meta]], capsize=3, label=meta, align='center'
        )

    axF.set_xticks(x_positions, list(WORKLOADS.keys()))
    axF.set_xlabel("Workload")
    axF.set_ylabel("F1")
    axF.set_title("F) F1 vs Meta at 3 Workloads (μ=0.2)")
    axF.set_ylim(0, 1)  # cap at 1 for F1
    axF.legend(
        title="meta",
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=True,
        borderaxespad=0.0
    )

    # Inset for F: mini grouped bars
    if INSET_ENABLED:
        iaxF = inset_axes(axF, width=INSET_WIDTH, height=INSET_HEIGHT, loc=INSET_LOC, borderpad=INSET_BORDERPAD)
        for meta in META_SET:
            iaxF.bar(
                x_positions + offset_map[meta], bar_Y[meta], width=BAR_WIDTH,
                yerr=[bar_Elo[meta], bar_Ehi[meta]], capsize=3, align='center'
            )
        iaxF.set_ylim(*INSET_YLIM)
        iaxF.set_xlim(axF.get_xlim())
        iaxF.set_xticks([])
        iaxF.tick_params(labelsize=INSET_TICK_FONTSIZE)
        iaxF.grid(False)
        try:
            pp, p1, p2 = axF.indicate_inset_zoom(
                iaxF, edgecolor=INSET_EDGE_COLOR, lw=INSET_EDGE_LW, alpha=INSET_EDGE_ALPHA
            )
            pp.set_clip_on(True); pp.set_clip_path(axF.patch)
            for line in (p1, p2):
                if line is not None:
                    line.set_clip_on(True); line.set_clip_path(axF.patch)
        except Exception:
            pass

    # Layout & save (more bottom margin to avoid legend clipping)
    plt.tight_layout(rect=[0, 0.08, 1, 1], h_pad=0.5)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    print(f"[OK] Saved figure: {out_png}")

# -----------------------------
# Entrypoint
# -----------------------------
def main():
    csv_path = RESULTS_CSV or _find_latest_csv()
    out_png = os.path.join(os.path.dirname(csv_path), "figure_runtime_f1_from_csv.pdf")
    rows = load_results(csv_path)
    plot_runtime_and_f1_from_rows(rows, out_png)

if __name__ == "__main__":
    main()
