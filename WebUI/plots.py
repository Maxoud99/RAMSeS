"""
Curated plot manifest and safe image serving.

One entity produces ~576 PNGs, of which 346 are per-window SHAP frames and ~140
are historical duplicates from directories that never clean up. Dumping that on
a page is useless, so each stage declares a small headline set and everything
else goes behind a lazy gallery.
"""

import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from WebUI import paths

# Trees the plots live under, relative to myresults/. "Thomposon" is a typo in
# the pipeline that is load-bearing — every writer uses it; myresults/Thompson/
# is a stale leftover.
TREE_GA = "GA_Ens"
TREE_THOMPSON = "Thomposon"
TREE_MC = "robustness/MonteCarlo"
TREE_OFFBY = "robustness/off_by"
TREE_GAN = "robustness/GAN"
TREE_AGG = "robust_aggregated"

# Thompson artifacts are suffixed with the iteration count (50 by default);
# discovered rather than assumed.
_IT_RE = re.compile(r"_(\d+)\.png$")

# Timestamped filenames in the GAN and off-by trees accumulate on every run.
# Zero-padded, so lexicographic order is chronological — more reliable than
# mtime, which a copy would destroy.
TS_RE = re.compile(
    r"^(?P<stem>.+?)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?P<tail>_?)\.png$")

_OFFBY_TREE_RE = re.compile(r"_off_by_point_tree_(?P<winner>.+?)_vs_(?P<competitor>.+)\.png$")


def _dir_for(tree: str, dataset: str, entity: str) -> Optional[Path]:
    """`myresults/{tree}/{dataset}/{entity}`, both name levels case-insensitive."""
    root = paths.MYRESULTS
    for part in tree.split("/"):
        root = root / part
    return paths.resolve_entity_dir(root, dataset, entity)


def _ls(directory: Optional[Path], pattern: str = "*.png") -> List[Path]:
    """Files matching `pattern`, with the directory part glob-escaped.

    Escaping matters: one real filename is
    `ensemble_scores_SKAB_7_Data_vs_anomalies_['spikes'].png`, and the brackets
    would otherwise be read as a glob character class.
    """
    if directory is None or not directory.is_dir():
        return []
    return sorted(Path(p) for p in
                  glob.glob(os.path.join(glob.escape(str(directory)), pattern)))


def dedupe_timestamped(files: List[Path]) -> List[Dict[str, Any]]:
    """Collapse accumulating timestamped files to the newest of each pattern.

    Handles the four real irregularities: off-by's trailing underscore, off-by's
    literal space in "Misclassified Anomalies", GAN's underscore-and-trailing-
    underscore form, and GAN's plain form. Files with no timestamp pass through.
    """
    groups: Dict[Any, List[Any]] = {}
    plain: List[Dict[str, Any]] = []
    for path in files:
        m = TS_RE.match(path.name)
        if not m:
            plain.append({"path": path, "timestamp": None, "n_older": 0})
            continue
        key = (m.group("stem"), m.group("tail"))
        groups.setdefault(key, []).append((m.group("ts"), path))
    out = list(plain)
    for (stem, _tail), entries in sorted(groups.items()):
        entries.sort(key=lambda pair: pair[0])
        ts, path = entries[-1]
        out.append({"path": path, "timestamp": ts, "n_older": len(entries) - 1})
    return out


def _iteration_tag(directory: Optional[Path]) -> Optional[str]:
    """The `_50` suffix Thompson plots carry, read off disk."""
    for path in _ls(directory, "expected_rewards_*.png"):
        m = _IT_RE.search(path.name)
        if m:
            return m.group(1)
    return None


def _fig(path: Path, title: str, caption: str = "", **extra) -> Dict[str, Any]:
    fig = {"title": title, "caption": caption,
           "src": "/media/" + paths.rel_to_myresults(path).replace(os.sep, "/"),
           "name": path.name}
    fig.update(extra)
    return fig


def _variants(directory, patterns, titles) -> List[Dict[str, Any]]:
    out = []
    for pattern, title in zip(patterns, titles):
        found = _ls(directory, pattern)
        if found:
            out.append(_fig(found[0], title))
    return out


# ── Per-stage manifests ──────────────────────────────────────────────────────

def _ga_selection(ds, ent):
    d = _dir_for(TREE_GA, ds, ent)
    headline, gallery = [], []
    for pattern, title, caption in (
        ("ga_selection_utility_*.png", "Utility",
         "Leave-one-out fitness change and mean marginal contribution per detector."),
        ("ga_selection_archetypes_*.png", "Utility × stability",
         "Where each detector sits on the two axes that explain its selection."),
    ):
        found = _ls(d, pattern)
        if found:
            headline.append(_fig(found[0], title, caption))
    survival = _variants(d, ["ga_selection_survival_*[!l].png", "ga_selection_survival_all_*.png"],
                         ["Ensemble highlighted", "All detectors"])
    survival = [f for f in survival if "_all_" not in f["name"]] + \
               [f for f in survival if "_all_" in f["name"]]
    if survival:
        headline.append({"title": "Survival across generations",
                         "caption": "How consistently the algorithm kept each detector.",
                         "variants": survival, "default": 0})
    newest = max((p.stat().st_mtime for p in _ls(d)), default=0)
    for path in _ls(d, "ensemble_scores_*.png"):
        gallery.append(_fig(path, "Injected anomalies", "The data the run saw."))
    for path in _ls(d, "ga_selection_*interaction_*.png"):
        # Friedman's H is disabled in the current pipeline; anything left is
        # from an older run. Badge it rather than presenting it as current.
        stale = path.stat().st_mtime < newest - 1
        gallery.append(_fig(path, "Interaction (disabled axis)",
                            "Left over from an earlier run.", stale=stale))
    return headline, gallery


def _ga_combination(ds, ent):
    d = _dir_for(TREE_GA, ds, ent)
    headline = [_fig(p, "Detector weighting",
                     "Absolute SHAP, PFI and total ALE — the three magnitude measures "
                     "that feed the Markov consensus ranking. All are magnitudes; "
                     "the sign is in the next figure.")
                for p in _ls(d, "ga_combination_importance_*.png")]
    # Both ALE figures live under the same prefix, so they are split by name
    # rather than by glob: the dataset name follows the prefix and could itself
    # begin with any letter, which rules out a character-class pattern.
    ale = _ls(d, "ga_combination_ale*.png")
    plain = [p for p in ale if not p.name.startswith("ga_combination_ale_bins_")]
    binned = [p for p in ale if p.name.startswith("ga_combination_ale_bins_")]
    variants = ([_fig(plain[0], "Plain")] if plain else []) + \
               ([_fig(binned[0], "Bin edges marked")] if binned else [])
    if variants:
        headline.append({
            "title": "How each detector moves the meta-learner",
            "caption": "One accumulated-effect curve per detector, over that "
                       "detector's own score range. Rising means higher scores "
                       "from it push the ensemble toward flagging an anomaly, "
                       "falling means toward normal. The sign is where the curve "
                       "ends; a dashed curve is one whose sign is weakly "
                       "supported. The second view marks the quantile bins the "
                       "curve is built from, which is what shows whether a turn "
                       "is structure or coarse resolution.",
            "variants": variants, "default": 0})
    return headline, []


# Every grouped-bar channel figure in both Thompson stages plots a subset —
# entities here carry 9 to 38 channels — and the bars alone cannot tell a reader
# whether a missing channel was small or simply not selected. The rule is stated
# on the figures themselves too (Thompson_Sampling._render_shap_comparison);
# this is the same sentence for the page.
CHANNEL_RULE = ("Channels shown are the union over the plotted detectors of "
                "each one's 9 largest values; a channel missing here was "
                "outside every plotted detector's top 9, not necessarily zero.")


def _thompson(ds, ent):
    d = _dir_for(TREE_THOMPSON, ds, ent)
    it = _iteration_tag(d)
    headline, gallery = [], []
    if it:
        for pattern, title, caption in (
            (f"expected_rewards_{it}.png", "Expected rewards",
             "Per-window expected reward for every detector — the raw signal."),
            (f"expected_rewards_smoothed_{it}.png", "Expected rewards (smoothed)",
             "The same signal smoothed, which is what regime detection reads."),
            (f"selection_states_{it}.png", "Selection states",
             "Exploitation, informed exploration and forced random picks over the run."),
        ):
            found = _ls(d, pattern)
            if found:
                headline.append(_fig(found[0], title, caption))
        avg = _variants(d, [f"reward_average_top3_{it}.png", f"reward_average_all_{it}.png"],
                        ["Top 3 detectors", "All detectors"])
        if avg:
            headline.append({"title": "Mean channel contribution across all windows",
                             "caption": "Each channel's own share of a detector's expected "
                                        "reward, averaged over every window. The bars sum "
                                        "to the detector's expected reward on a typical "
                                        "window. " + CHANNEL_RULE,
                             "variants": avg, "default": 0})
        for pattern, title, caption in (
            (f"history_plot_{it}.png", "Posterior history", ""),
            (f"shap_per_model_{it}.png", "Per-model channel attribution",
             "One panel per detector, each showing its own 10 largest "
             "contributions; a detector's other channels are not drawn."),
            (f"shap_comparison_{it}.png", "Channel comparison (top 3)", CHANNEL_RULE),
            (f"shap_comparison_all_{it}.png", "Channel comparison (all)", CHANNEL_RULE),
            # Demoted from the headline. mean|SHAP| measures how much a
            # channel's influence VARIES between windows — the signed average
            # is zero by construction, which is why it had to take absolute
            # values — so it is a dispersion measure, not an average share.
            (f"shap_average_top3_{it}.png", "Channel influence variability (top 3)",
             "Mean |SHAP|: how much each channel's influence varies from window "
             "to window. Not an average contribution. " + CHANNEL_RULE),
            (f"shap_average_all_{it}.png", "Channel influence variability (all)",
             "Mean |SHAP|: how much each channel's influence varies from window "
             "to window. Not an average contribution. " + CHANNEL_RULE),
        ):
            for path in _ls(d, pattern):
                gallery.append(_fig(path, title, caption))
    return headline, gallery


def _ts_ranking(ds, ent):
    """The ranking-criterion stage.

    Shares TREE_THOMPSON with `_thompson` and is separated purely by the
    `ranking_` filename prefix — the same way `_ga_combination` is separated
    from `_ga_selection` inside one GA directory. `_iteration_tag` still reads
    `expected_rewards_*.png`, which is written by the sibling stage into this
    same directory.
    """
    d = _dir_for(TREE_THOMPSON, ds, ent)
    it = _iteration_tag(d)
    headline, gallery = [], []
    if not it:
        return headline, gallery
    for pattern, title, caption in (
        (f"ranking_final_{it}.png", "Final ranking",
         "The score each detector was ranked by, with how many windows it was tried in."),
        (f"ranking_gap_{it}.png", "What decided the top spot",
         "The winner's margin over the runner-up, split channel by channel; "
         "these bars sum to the margin exactly."),
        (f"ranking_criterion_{it}.png", "Ranking score over the run",
         "Every detector's score window by window, shaded by which one led."),
    ):
        found = _ls(d, pattern)
        if found:
            headline.append(_fig(found[0], title, caption))
    channels = _variants(d, [f"ranking_channels_{it}.png", f"ranking_channels_all_{it}.png"],
                         ["Top 3 detectors", "All detectors"])
    if channels:
        headline.append({"title": "Where each detector's score comes from",
                         "caption": "Per-channel shares of the final weights. These are "
                                    "sums of squared weights, so they are never "
                                    "negative. " + CHANNEL_RULE,
                         "variants": channels, "default": 0})
    return headline, gallery


# What each per-regime figure actually shows. The stems all mint the same
# filename shape over the same window range, so without this a reader has three
# identical "windows 10–62" captions describing three different quantities.
_REGIME_SET_LABELS = {
    "reward_per_regime": (
        "Expected-reward contribution",
        " Each channel's own share of the leader's expected reward, averaged "
        "over the regime; the bars sum to that reward."),
    "shap_per_regime": (
        "Deviation from a typical window",
        " How far each channel's contribution departs from what it usually "
        "contributes. This is what separates one detector from another, but it "
        "is not a share of the reward and does not sum to it."),
    "ranking_per_regime": (
        "Ranking score",
        " Weights as at the last window of the regime; the score is cumulative, "
        "so this is the state reached by then, not what the regime itself added."),
}


def regime_plots(ds, ent, subdir_stem: str = "shap_per_regime") -> Dict[int, Dict[str, Any]]:
    """Per-regime images keyed by regime index, for ONE set.

    Filenames are `regime_{NN}_w{start}-{end}_{model}.png` and 0-based, matching
    the `*.regime.N` atom ids, so each regime sentence can be shown beside its
    own plot. `subdir_stem` selects the set; every set mints the same filename
    shape, so one regex serves all of them.
    """
    d = _dir_for(TREE_THOMPSON, ds, ent)
    it = _iteration_tag(d)
    if not it or d is None:
        return {}
    out = {}
    pattern = re.compile(r"^regime_(\d+)_w(\d+)-(\d+)_(.+)\.png$")
    label, detail = _REGIME_SET_LABELS.get(subdir_stem, ("", ""))
    for path in _ls(d / f"{subdir_stem}_{it}"):
        m = pattern.match(path.name)
        if m:
            out[int(m.group(1))] = _fig(
                path, label or f"Regime {int(m.group(1))}",
                f"Windows {m.group(2)}–{m.group(3)}, led by {m.group(4)}." + detail)
    return out


def regime_plot_variants(ds, ent, stems: List[str]) -> Dict[int, List[Dict[str, Any]]]:
    """The same regime across several sets, ready for a variant toggle.

    Returns {regime_index: [figure, ...]} in the order `stems` is given, so the
    first stem is what the card shows by default. Indices missing from a set are
    simply absent from that regime's list rather than shifting the others.
    """
    per_stem = [(stem, regime_plots(ds, ent, stem)) for stem in stems]
    out: Dict[int, List[Dict[str, Any]]] = {}
    for _stem, figures in per_stem:
        for index, figure in figures.items():
            out.setdefault(index, []).append(figure)
    return out


def _monte_carlo(ds, ent):
    d = _dir_for(TREE_MC, ds, ent)
    headline, gallery = [], []
    # Plain is the default: the un-annotated figure is the one that belongs in a
    # thesis, and the annotated version is a click away.
    curve_variants = _variants(
        d,
        ["*_MonteCarlo_noise_curves_F1_plain.png",
         "*_MonteCarlo_noise_curves_PRAUC_plain.png"],
        ["F1", "PR-AUC"])
    if curve_variants:
        headline.append({"title": "Score against noise level",
                         "caption": "Each detector's score as injected noise grows.",
                         "variants": curve_variants, "default": 0})
    # The annotated and fixed-threshold variants are browse-only: the plain
    # curves are the ones that belong in a figure, the rest are for digging.
    for pattern, title in (
            ("*_MonteCarlo_noise_curves_F1.png", "F1 (annotated)"),
            ("*_MonteCarlo_noise_curves_F1_fixed.png", "F1 at a fixed threshold"),
            ("*_MonteCarlo_noise_curves_F1_fixed_plain.png",
             "F1 at a fixed threshold (plain)"),
            ("*_MonteCarlo_noise_curves_PRAUC.png", "PR-AUC (annotated)"),
            ("*_MonteCarlo_ranking_stability.png", "Ranking stability"),
            ("*_MonteCarlo_surrogate_tree_F1.png", "Surrogate tree (F1)"),
            ("*_MonteCarlo_surrogate_tree_PRAUC.png", "Surrogate tree (PR-AUC)")):
        for path in _ls(d, pattern):
            gallery.append(_fig(path, title))
    for path in _ls(d, "*_MonteCarloResults.png"):
        gallery.append(_fig(path, path.name.split("_MonteCarloResults")[0].split("_")[-1]))
    return headline, gallery


def _off_by(ds, ent):
    d = _dir_for(TREE_OFFBY, ds, ent)
    headline, gallery = [], []
    for path in _ls(d, "*_off_by_point_importance.png"):
        headline.append(_fig(path, "Which point properties separate the winner",
                             "Feature importance across all pairwise comparisons."))
    for path in _ls(d, "*_off_by_point_tree_*.png"):
        m = _OFFBY_TREE_RE.search(path.name)
        if m:
            headline.append(_fig(
                path, f"{m.group('winner')} vs {m.group('competitor')}",
                f"Where {m.group('winner')} uniquely beat {m.group('competitor')}."))
    for entry in dedupe_timestamped(_ls(d, "Data_vs_DataWithAnomalies_*.png")
                                    + _ls(d, "*Misclassified*.png")):
        title = ("Injected borderline points" if "Data_vs" in entry["path"].name
                 else "Misclassified points")
        gallery.append(_fig(entry["path"], title,
                            timestamp=entry["timestamp"], n_older=entry["n_older"]))
    return headline, gallery


def _gan(ds, ent):
    d = _dir_for(TREE_GAN, ds, ent)
    headline = []
    for entry in dedupe_timestamped(_ls(d)):
        name = entry["path"].name
        title = ("Injected borderline points" if "Data_vs" in name
                 else "Misclassified points")
        headline.append(_fig(entry["path"], title,
                             timestamp=entry["timestamp"], n_older=entry["n_older"]))
    return headline, []


def _aggregation(ds, ent, which):
    """All aggregation plots present for `which` ('robust' or 'final').

    Glob-driven rather than a fixed list: `_kendall_only` is only emitted when
    exactly two sources feed the aggregation, so hardcoding it would either
    miss it or point at a missing file.
    """
    d = _dir_for(TREE_AGG, ds, ent)
    headline = []
    for path in _ls(d, f"aggregation_explainability_{which}_*.png"):
        kendall = "kendall_only" in path.name
        headline.append(_fig(
            path,
            "Agreement only (two sources)" if kendall else f"{which.capitalize()} aggregation",
            "With two sources, leave-one-out is undefined and only agreement is meaningful."
            if kendall else "Per-source influence and agreement behind the consensus."))
    return headline, []


_BUILDERS = {
    "ga_selection": _ga_selection,
    "ga_combination": _ga_combination,
    "thompson": _thompson,
    "ts_ranking": _ts_ranking,
    "monte_carlo": _monte_carlo,
    "off_by": _off_by,
    "gan": _gan,
    "rank_aggregation_robust": lambda ds, ent: _aggregation(ds, ent, "robust"),
    "rank_aggregation_final": lambda ds, ent: _aggregation(ds, ent, "final"),
}


def manifest(dataset: str, entity: str) -> Dict[str, Any]:
    """Headline figures plus gallery descriptors, per plot group."""
    out: Dict[str, Any] = {}
    for group, builder in _BUILDERS.items():
        try:
            headline, gallery = builder(dataset, entity)
        except OSError:
            headline, gallery = [], []
        # The whole list, not a preview: the button is labelled with the count,
        # so returning three items made every label a lie. These groups hold at
        # most a couple of dozen small dicts; the 173-frame per-window sets are
        # separate descriptors, paged on demand.
        out[group] = {"headline": headline, "gallery_count": len(gallery),
                      "gallery": gallery}
    out["_galleries"] = gallery_descriptors(dataset, entity)
    return out


# Groups whose lazy galleries live under TREE_THOMPSON. Both Thompson stages
# write into that one directory and are told apart by filename prefix, so the
# gallery id carries the plot_group and gallery_page validates against this map
# rather than against a single hardcoded name.
_GALLERY_TREES = {"thompson": TREE_THOMPSON, "ts_ranking": TREE_THOMPSON}


def gallery_descriptors(dataset: str, entity: str) -> List[Dict[str, Any]]:
    """Large per-window sets, described but never listed eagerly."""
    d = _dir_for(TREE_THOMPSON, dataset, entity)
    it = _iteration_tag(d)
    out = []
    if it and d is not None:
        for group, sub, title, caption in (
            ("thompson", f"reward_per_window_{it}",
             "Reward contribution per window (top 3)",
             "One frame per window; each detector's bars sum to its expected reward."),
            ("thompson", f"reward_per_window_all_{it}",
             "Reward contribution per window (all detectors)", ""),
            ("thompson", f"reward_per_window_every10_{it}",
             "Reward contribution, every 10th window", ""),
            ("thompson", f"reward_per_regime_all_{it}",
             "Reward contribution per regime (all detectors)", ""),
            ("thompson", f"shap_per_window_{it}",
             "Deviation per window (top 3)",
             "Departure from a typical window — not a share of the reward."),
            ("thompson", f"shap_per_window_all_{it}",
             "Deviation per window (all detectors)", ""),
            ("thompson", f"shap_per_window_every10_{it}",
             "Deviation, every 10th window", ""),
            ("thompson", f"shap_per_regime_all_{it}",
             "Deviation per regime (all detectors)", ""),
            ("ts_ranking", f"ranking_per_window_{it}",
             "Ranking score per window (top 3)",
             "One frame per window, each showing the score as it stood then."),
            ("ts_ranking", f"ranking_per_window_all_{it}",
             "Ranking score per window (all detectors)", ""),
            ("ts_ranking", f"ranking_per_window_every10_{it}",
             "Every 10th window", ""),
        ):
            count = len(_ls(d / sub))
            if count:
                out.append({"id": f"{group}/{sub}", "title": title,
                            "caption": caption, "count": count})
    return out


def gallery_page(dataset: str, entity: str, gallery_id: str,
                 offset: int = 0, limit: int = 60) -> Dict[str, Any]:
    group, _, sub = gallery_id.partition("/")
    if group not in _GALLERY_TREES or not sub or "/" in sub or sub.startswith("."):
        return {"items": [], "total": 0, "offset": offset, "limit": limit}
    d = _dir_for(_GALLERY_TREES[group], dataset, entity)
    if d is None:
        return {"items": [], "total": 0, "offset": offset, "limit": limit}
    files = _ls(d / sub)
    window = files[max(0, offset): max(0, offset) + max(1, min(limit, 200))]
    return {"items": [_fig(p, p.stem.replace("_", " ")) for p in window],
            "total": len(files), "offset": offset, "limit": limit}


# ── Serving images ───────────────────────────────────────────────────────────

ALLOWED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def safe_media_path(relpath: str) -> Optional[Path]:
    """Resolve a /media/<relpath> request, or None if it escapes myresults/.

    `resolve()` runs BEFORE the containment check so it defeats both `..` and
    symlinks pointing outside the tree (send_from_directory alone stops the
    former but not the latter). The extension allowlist means this route can
    never hand out a .pth checkpoint or a .json artifact.
    """
    if not relpath or relpath.startswith(("/", "\\")) or "\x00" in relpath:
        return None
    if len(relpath) > 3 and relpath[1] == ":":      # Windows drive-absolute
        return None
    root = paths.MYRESULTS.resolve()
    try:
        candidate = (root / relpath).resolve()
    except (OSError, RuntimeError):
        return None
    if not candidate.is_relative_to(root):
        return None
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        return None
    if not candidate.is_file():
        return None
    return candidate
