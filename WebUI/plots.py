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
_GAN_TREE_RE = re.compile(r"_gan_point_tree_(?P<winner>.+?)_vs_(?P<competitor>.+)\.png$")


def _dir_for(tree: str, dataset: str, entity: str) -> Optional[Path]:
    """`myresults/{tree}/{dataset}/{entity}`, both name levels case-insensitive."""
    root = paths.MYRESULTS
    for part in tree.split("/"):
        root = root / part
    return paths.resolve_entity_dir(root, dataset, entity)


def _ls(directory: Optional[Path], pattern: str = "*.png") -> List[Path]:
    """Files matching `pattern`, with the directory part glob-escaped.

    Escaping matters: result trees written before the anomaly type became
    selectable hold `ensemble_scores_SKAB_7_Data_vs_anomalies_['spikes'].png`,
    and the brackets would otherwise be read as a glob character class.
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
    """One headline figure; the other two are a click away.

    Utility × stability is the figure that answers the stage's question — where
    a detector sits on the two axes that decided whether it was kept. LOFO and
    the survival trace are the inputs to that placement, so they browse rather
    than lead.

    The gallery holds only those two. The injected-anomalies figure is about the
    DATA, not the selection, and the two Friedman-interaction plots are from a
    disabled axis, so a run that still has them on disk is showing leftovers.
    """
    d = _dir_for(TREE_GA, ds, ent)
    headline, gallery = [], []
    for path in _ls(d, "ga_selection_archetypes_*.png"):
        headline.append(_fig(
            path, "Utility × stability",
            "Where each detector sits on the two axes that explain its "
            "selection, split at the median of each. Both axes start at zero."))
        break
    for path in _ls(d, "ga_selection_utility_*.png"):
        gallery.append(_fig(
            path, "LOFO",
            "Leave-one-out fitness change on the chosen ensemble, and mean "
            "marginal contribution per detector."))
    survival = _variants(d, ["ga_selection_survival_*[!l].png", "ga_selection_survival_all_*.png"],
                         ["Ensemble highlighted", "All detectors"])
    survival = [f for f in survival if "_all_" not in f["name"]] + \
               [f for f in survival if "_all_" in f["name"]]
    for figure in survival:
        gallery.append(dict(figure, caption="How consistently the algorithm "
                                            "kept each detector."))
    return headline, gallery


def _ga_combination(ds, ent):
    d = _dir_for(TREE_GA, ds, ent)
    headline = [_fig(p, "Detector weighting",
                     "Absolute SHAP, PFI and total ALE — the three magnitude measures "
                     "that feed the Markov consensus ranking. All are magnitudes; "
                     "the sign is in the ALE figure below.")
                for p in _ls(d, "ga_combination_importance_*.png")]
    # Its own figure since the two were split apart; the caption is what still
    # ties it to the weighting figure above.
    headline += [_fig(p, "Consensus ranking",
                      "The Markov stationary probability each detector ends up "
                      "with, as the result of the three measures on the previous "
                      "plot.")
                 for p in _ls(d, "ga_combination_ranking_*.png")]
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
                       "detector's own score range.",
            "variants": variants, "default": 0})
    return headline, []


# Every grouped-bar context feature figure in both Thompson stages plots a subset —
# entities here carry 9 to 38 context features — and the bars alone cannot tell a reader
# whether a missing context feature was small or simply not selected. The rule is stated
# on the figures themselves too (Thompson_Sampling._render_shap_comparison);
# this is the same sentence for the page.
CONTEXT_FEATURE_RULE = ("Context features shown are the union over the plotted detectors of "
                "each one's 9 largest values; a context feature missing here was "
                "outside every plotted detector's top 9, not necessarily zero.")


def _thompson(ds, ent):
    d = _dir_for(TREE_THOMPSON, ds, ent)
    it = _iteration_tag(d)
    headline, gallery = [], []
    if it:
        # Smoothed first: it is what regime detection actually reads, so it is
        # the one the regime prose describes. The raw signal is the same
        # quantity un-smoothed, which makes it a toggle rather than a figure of
        # its own.
        rewards = _variants(d, [f"expected_rewards_smoothed_{it}.png",
                                f"expected_rewards_{it}.png"],
                            ["Smoothed", "Raw"])
        if rewards:
            headline.append({
                "title": "Expected rewards",
                "caption": "Per-window expected reward for every detector. "
                           "Smoothing is what regime detection reads; the raw "
                           "signal is the same quantity unsmoothed.",
                "variants": rewards, "default": 0})
        for pattern, title, caption in (
            (f"selection_states_{it}.png", "Selection states",
             "Exploitation, informed exploration and forced random picks over the run."),
        ):
            found = _ls(d, pattern)
            if found:
                headline.append(_fig(found[0], title, caption))
        avg = _variants(d, [f"reward_average_top3_{it}.png", f"reward_average_all_{it}.png"],
                        ["Top 3 detectors", "All detectors"])
        if avg:
            headline.append({"title": "Mean context feature contribution across all windows",
                             "caption": "Each context feature's own share of a detector's expected "
                                        "reward, averaged over every window.",
                             "variants": avg, "default": 0})
        # Only the two mean|SHAP| figures browse. The posterior history, the
        # per-model panels and the two context feature comparisons all restate what the
        # headline reward figures and the per-regime disclosure already show.
        for pattern, title, caption in (
            # Demoted from the headline rather than dropped. mean|SHAP| measures
            # how much a context feature's influence VARIES between windows — the signed
            # average is zero by construction, which is why it had to take
            # absolute values — so it is a dispersion measure, not an average
            # share, and nothing else on the card reports dispersion.
            (f"shap_average_top3_{it}.png", "Context feature influence variability (top 3)",
             "Mean |SHAP|: how much each context feature's influence varies from window "
             "to window. Not an average contribution. " + CONTEXT_FEATURE_RULE),
            (f"shap_average_all_{it}.png", "Context feature influence variability (all)",
             "Mean |SHAP|: how much each context feature's influence varies from window "
             "to window. Not an average contribution. " + CONTEXT_FEATURE_RULE),
        ):
            for path in _ls(d, pattern):
                gallery.append(_fig(path, title, caption))
    return headline, gallery


def _ranking_pair_picker(ds, ent) -> Optional[Dict[str, Any]]:
    """The gap decomposition as a PAIR PICKER rather than a fixed figure.

    Every detector's per-context-feature shares are in the IR, and the gap between any
    two is exactly the difference of their shares, so the page can ask for a
    pair and get it drawn. Returns None when the IR predates that block, and
    the caller then falls back to the pipeline's static winner-vs-runner-up
    figure — which is this control's default pair anyway.

    `detectors` arrives in the ranking's own order, so the first two are the
    winner and the runner-up and the initial view matches the static figure.
    """
    from WebUI import ondemand
    shares = ondemand.ranking_context_feature_shares(ds, ent)
    if len(shares) < 2:
        return None
    order = sorted(shares, key=lambda m: -sum(shares[m]))
    return {
        "title": "What decided the top spot",
        "caption": "The margin between two detectors, split context feature by context feature; "
                   "these bars sum to the margin exactly.",
        "pair_picker": {
            "detectors": order,
            "endpoint": f"/api/plots/{ds}/{ent}/ranking-gap",
        },
    }


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
        (f"ranking_criterion_{it}.png", "Ranking score over the run",
         "Every detector's score window by window, shaded by which one led."),
    ):
        found = _ls(d, pattern)
        if found:
            headline.append(_fig(found[0], title, caption))
    # Any pair, not just the winner and runner-up. 11 detectors is 55 unordered
    # pairs and a reader looks at one or two, so the picture is drawn per
    # request from the IR's per-detector shares; the static ranking_gap_*.png
    # the pipeline writes is exactly the default pair of this control.
    pair = _ranking_pair_picker(ds, ent)
    if pair:
        headline.append(pair)
    ranking_variants = _variants(d, [f"ranking_channels_{it}.png", f"ranking_channels_all_{it}.png"],
                         ["Top 3 detectors", "All detectors"])
    if ranking_variants:
        headline.append({"title": "Where each detector's score comes from",
                         "caption": "Per-context-feature shares of the final score.",
                         "variants": ranking_variants, "default": 0})
    return headline, gallery


# What each per-regime figure actually shows. The stems all mint the same
# filename shape over the same window range, so without this a reader has three
# identical "windows 10–62" captions describing three different quantities.
_REGIME_SET_LABELS = {
    "reward_per_regime": (
        "Expected-reward contribution",
        " Each context feature's own share of the leader's expected reward, averaged "
        "over the regime; the bars sum to that reward."),
    "shap_per_regime": (
        "Deviation from a typical window",
        " How far each context feature's contribution departs from what it usually "
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
    # One variant per metric the run swept: a metric its fitness did not name
    # writes no figure, and `_variants` drops the patterns that match nothing.
    curve_variants = _variants(
        d,
        ["*_MonteCarlo_noise_curves_F1_plain.png",
         "*_MonteCarlo_noise_curves_PRAUC_plain.png",
         "*_MonteCarlo_noise_curves_VUS_plain.png"],
        ["F1", "PR-AUC", "VUS"])
    if curve_variants:
        headline.append({"title": "Score against noise level",
                         "caption": "Each detector's score as injected noise grows.",
                         "variants": curve_variants, "default": 0})
    # Browse-only: the plain curves above are the ones that belong in a figure,
    # these are for digging.
    #
    # Three of what the stage writes are deliberately NOT offered. The pipeline
    # still generates them — they are on disk for anyone who wants them — but
    # the annotated F1 and PR-AUC curves are the same data as the plain pair in
    # the headline with labels drawn on top, and the annotated fixed-threshold
    # curve is superseded by its own plain version two lines below it. Offering
    # all seven made the reader choose between near-duplicates.
    for pattern, title in (
            ("*_MonteCarlo_noise_curves_F1_fixed_plain.png",
             "F1 at a fixed threshold"),
            ("*_MonteCarlo_ranking_stability.png", "Ranking stability"),
            ("*_MonteCarlo_surrogate_tree_F1.png", "Surrogate tree (F1)"),
            ("*_MonteCarlo_surrogate_tree_PRAUC.png", "Surrogate tree (PR-AUC)"),
            ("*_MonteCarlo_surrogate_tree_VUS.png", "Surrogate tree (VUS)")):
        for path in _ls(d, pattern):
            gallery.append(_fig(path, title))
    # The per-detector *_MonteCarloResults.png set is not listed. One figure per
    # detector repeats what the noise curves already draw together, which is the
    # comparison that matters here, and its title came out as a bare family
    # index ("1", "2") because the stem splits on the underscore inside the
    # detector name.
    return headline, gallery


def _off_by(ds, ent):
    d = _dir_for(TREE_OFFBY, ds, ent)
    headline, gallery = [], []
    # One tree at a time, chosen by competitor. Every pair is already rendered,
    # so this is a selector over files rather than anything generated on demand
    # — but ten trees stacked down the card is ten near-identical figures the
    # reader has to scroll past to reach anything else.
    #
    # ONLY THE LATEST RUN'S TREES. These filenames carry the winner, not a
    # timestamp, so a run that picks a different winner writes a whole new set
    # beside the old one instead of overwriting it: SKAB/7 holds seven
    # `LOF_1_vs_*` from one run and ten `CBLOF_4_vs_*` from the next. Listing
    # both put stale competitors in the picker and — because the group is
    # chosen by filename order, not recency — titled the card with the OLD
    # winner while the run being read had chosen another. Grouping by winner
    # and keeping whichever group holds the newest file fixes both, and needs
    # no timestamp in the name. Same-winner reruns overwrite by name already.
    by_winner = {}
    for path in _ls(d, "*_off_by_point_tree_*.png"):
        m = _OFFBY_TREE_RE.search(path.name)
        if m:
            by_winner.setdefault(m.group("winner"), []).append((path, m.group("competitor")))
    if by_winner:
        winner = max(by_winner,
                     key=lambda w: max(p.stat().st_mtime for p, _ in by_winner[w]))
        trees = [_fig(path, competitor,
                      f"Where {winner} uniquely beat {competitor}.")
                 for path, competitor in sorted(by_winner[winner], key=lambda t: t[1])]
        headline.append({
            "title": f"Where {winner} uniquely wins",
            "variants": trees, "default": 0, "select_label": "Compared against",
        })
    # Browse-only. The importance plot answers a question about the whole
    # comparison rather than about this entity's decision, so it reads as
    # background to the trees above rather than as a headline of its own.
    for path in _ls(d, "*_off_by_point_importance.png"):
        gallery.append(_fig(path, "Which point properties separate the winner",
                            "Feature importance across all pairwise comparisons."))
    # `*Misclassified*.png` is still written by the off-by stage on every run —
    # it is simply not listed, and the GAN card does not list its copy either.
    # The injected-points figure beside it is the one that says something
    # specific to this stage.
    for entry in dedupe_timestamped(_ls(d, "Data_vs_DataWithAnomalies_*.png")):
        gallery.append(_fig(entry["path"], "Injected borderline points",
                            timestamp=entry["timestamp"], n_older=entry["n_older"]))
    return headline, gallery


def _gan(ds, ent):
    d = _dir_for(TREE_GAN, ds, ent)
    headline, gallery = [], []
    # The same shape as _off_by, for the same reason: both stages explain a
    # winner's exclusive wins over injected points, so the trees are the headline
    # and everything else is one click away. Tree filenames carry the WINNER
    # rather than a timestamp, so a run that picks a different winner writes a
    # whole new set beside the old one — only the group holding the newest file
    # is offered.
    by_winner = {}
    for path in _ls(d, "*_gan_point_tree_*.png"):
        m = _GAN_TREE_RE.search(path.name)
        if m:
            by_winner.setdefault(m.group("winner"), []).append((path, m.group("competitor")))
    if by_winner:
        winner = max(by_winner,
                     key=lambda w: max(p.stat().st_mtime for p, _ in by_winner[w]))
        trees = [_fig(path, competitor, f"Where {winner} uniquely beat {competitor}.")
                 for path, competitor in sorted(by_winner[winner], key=lambda t: t[1])]
        headline.append({
            "title": f"Where {winner} uniquely wins",
            "variants": trees, "default": 0, "select_label": "Compared against",
        })
    for path in _ls(d, "*_gan_point_importance.png"):
        gallery.append(_fig(path, "Which point properties separate the winner",
                            "Feature importance across all pairwise comparisons."))
    # Only the newest, across every stem rather than one per stem.
    #
    # These names begin with the dataset as it was typed on the command line, and
    # `load_data` lowercases only for its own lookup — so one entity run as
    # `--dataset SKAB` and again as `--dataset skab` leaves two stems that
    # `dedupe_timestamped` groups apart, and the card listed the same figure
    # twice. Picking the newest of the deduped entries needs no assumption about
    # how the name was spelled, and the count of what it hides stays honest.
    injected = dedupe_timestamped(_ls(d, "*Data_vs_DataWithAnomalies_*.png"))
    if injected:
        newest = max(injected, key=lambda e: e["timestamp"] or "")
        hidden = sum(e["n_older"] + 1 for e in injected) - 1
        gallery.append(_fig(newest["path"], "Injected borderline points",
                            timestamp=newest["timestamp"], n_older=hidden))
    # `*Misclassified*.png` is written by this stage on every run and listed by
    # neither card — the same treatment off-by's copy gets. It plots true against
    # predicted labels for the winner alone, which says nothing about the
    # comparison this card is for, and the injected-points figure beside it is
    # the one specific to the stage.
    return headline, gallery


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
        # The final stage merges exactly two sources, where leave-one-out and
        # Borda are degenerate — dropping one leaves the other unchanged. Its
        # standard figure therefore plots influence bars that mean nothing, so
        # only the agreement-only companion is shown there. The robust stage has
        # six sources and keeps both.
        if which == "final" and not kendall:
            continue
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


# The per-window sets, in the order they are offered. `kind` and `scope` are
# the two arguments the on-demand renderer takes; `stride` is the third and the
# only reason `every10` was ever a separate folder.
_PER_WINDOW_SETS = (
    ("thompson", "reward", "top", 1,
     "Reward contribution per window (top 3)",
     "One frame per window; each detector's bars sum to its expected reward."),
    ("thompson", "reward", "all", 1,
     "Reward contribution per window (all detectors)", ""),
    ("thompson", "reward", "top", 10,
     "Reward contribution, every 10th window", ""),
    ("thompson", "shap", "top", 1,
     "Deviation per window (top 3)",
     "Departure from a typical window — not a share of the reward."),
    ("thompson", "shap", "all", 1,
     "Deviation per window (all detectors)", ""),
    ("thompson", "shap", "top", 10,
     "Deviation, every 10th window", ""),
    ("ts_ranking", "ranking", "top", 1,
     "Ranking score per window (top 3)",
     "One frame per window, each showing the score as it stood then."),
    ("ts_ranking", "ranking", "all", 1,
     "Ranking score per window (all detectors)", ""),
    ("ts_ranking", "ranking", "top", 10,
     "Every 10th window", ""),
)

# Per-regime sets that are still written as folders. Unlike the per-window ones
# these are read inline beside every regime sentence, so they stay on disk.
_REGIME_GALLERY_SETS = (
    ("thompson", "reward_per_regime_all",
     "Reward contribution per regime (all detectors)"),
    ("thompson", "shap_per_regime_all",
     "Deviation per regime (all detectors)"),
)

_PW_ID = re.compile(r"^pw:(?P<kind>[a-z]+):(?P<scope>top|all):(?P<stride>\d+)$")


def _per_window_descriptors(dataset: str, entity: str) -> List[Dict[str, Any]]:
    """The nine per-window sets, backed by the persisted aggregates.

    Empty when the run predates them, and the caller then falls back to whatever
    `*_per_window_*` folders that run left on disk.
    """
    from WebUI import ondemand
    doc = ondemand.per_window_document(dataset, entity)
    if not doc:
        return []
    total = int(doc.get("n_windows") or 0)
    available = doc.get("sets") or {}
    out = []
    for group, kind, scope, stride, title, caption in _PER_WINDOW_SETS:
        if kind not in available or total <= 0:
            continue
        count = (total + stride - 1) // stride
        out.append({"id": f"{group}/pw:{kind}:{scope}:{stride}",
                    "title": title, "caption": caption, "count": count})
    return out


def gallery_descriptors(dataset: str, entity: str) -> List[Dict[str, Any]]:
    """Large per-window sets, described but never listed eagerly."""
    d = _dir_for(TREE_THOMPSON, dataset, entity)
    it = _iteration_tag(d)
    out = _per_window_descriptors(dataset, entity)
    if it and d is not None:
        if not out:
            # A tree written before the aggregates existed: the frames are still
            # folders of PNGs, so list them the way they were listed then.
            for group, kind, scope, stride, title, caption in _PER_WINDOW_SETS:
                stem = {"reward": "reward_per_window", "shap": "shap_per_window",
                        "ranking": "ranking_per_window"}[kind]
                if scope == "all":
                    stem += "_all"
                if stride > 1:
                    stem += f"_every{stride}"
                sub = f"{stem}_{it}"
                count = len(_ls(d / sub))
                if count:
                    out.append({"id": f"{group}/{sub}", "title": title,
                                "caption": caption, "count": count})
        for group, stem, title in _REGIME_GALLERY_SETS:
            sub = f"{stem}_{it}"
            count = len(_ls(d / sub))
            if count:
                out.append({"id": f"{group}/{sub}", "title": title,
                            "caption": "", "count": count})
    return out


def _per_window_page(dataset: str, entity: str, sub: str,
                     offset: int, limit: int) -> Optional[Dict[str, Any]]:
    """One page of on-demand frames, or None if `sub` is not a per-window id.

    The items carry a render URL instead of a `/media/` path; everything else
    about them is what the lightbox already expects, so the page needs no
    special case for these.
    """
    m = _PW_ID.match(sub)
    if not m:
        return None
    from WebUI import ondemand
    doc = ondemand.per_window_document(dataset, entity)
    if not doc:
        return {"items": [], "total": 0, "offset": offset, "limit": limit}
    kind, scope = m.group("kind"), m.group("scope")
    stride = max(1, int(m.group("stride")))
    windows = list(range(0, int(doc.get("n_windows") or 0), stride))
    page = windows[max(0, offset): max(0, offset) + max(1, min(limit, 200))]
    endpoint = f"/api/plots/{dataset}/{entity}/per-window"
    return {
        "items": [{"title": f"window {t:03d}", "caption": "",
                   "name": f"{kind}_{scope}_window_{t:03d}.png",
                   "src": f"{endpoint}?kind={kind}&scope={scope}&t={t}"}
                  for t in page],
        "total": len(windows), "offset": offset, "limit": limit,
    }


def gallery_page(dataset: str, entity: str, gallery_id: str,
                 offset: int = 0, limit: int = 60) -> Dict[str, Any]:
    group, _, sub = gallery_id.partition("/")
    if group not in _GALLERY_TREES or not sub or "/" in sub or sub.startswith("."):
        return {"items": [], "total": 0, "offset": offset, "limit": limit}
    rendered = _per_window_page(dataset, entity, sub, offset, limit)
    if rendered is not None:
        return rendered
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
