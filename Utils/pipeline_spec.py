"""
Canonical vocabulary of the model-selection pipeline: which detectors exist,
which sub-stages exist, and how the CLI spellings of both are parsed.

Deliberately **stdlib-only**: `Utils/utils.py` cannot host these definitions
because it imports torch, matplotlib and PIL, and the web UI reads the same
vocabulary without dragging a 2 GB ML stack into a Flask process.
"""

import math
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

# Base detector instances, in the order app.py loads them. Adding a family here
# is what makes it selectable: --detectors validates against this tuple,
# `families_for` turns it into what gets trained, and the UI chips are built
# from it. Instance counts come from Model_Training/hyperparameter_grids.py.
#
# The PyOD-fallback families are spelled UPPER CASE because
# `Algorithms/pyod_model.PyodModel` names checkpoints `{FAMILY.upper()}_{i}`
# even though the PyOD classes are `IForest`, `HBOS` and so on;
# `_class_in` resolves the case.
ALL_DETECTORS = (
    "LOF_1", "LOF_2", "LOF_3", "LOF_4",
    "NN_1", "NN_2", "NN_3",
    "CBLOF_1", "CBLOF_2", "CBLOF_3", "CBLOF_4",
    "ABOD_1", "ABOD_2", "ABOD_3", "ABOD_4",
    "KDE_1", "KDE_2", "KDE_3", "KDE_4",
    "IFOREST_1", "IFOREST_2", "IFOREST_3", "IFOREST_4",
    "HBOS_1", "HBOS_2", "HBOS_3", "HBOS_4",
    "PCA_1", "PCA_2", "PCA_3", "PCA_4",
    "OCSVM_1", "OCSVM_2", "OCSVM_3", "OCSVM_4",
    "MCD_1", "MCD_2", "MCD_3", "MCD_4",
    # The transductive three: they judge a point against its neighbours, not
    # against a fitted model. See TRANSDUCTIVE_FAMILIES.
    "COF_1", "COF_2", "COF_3", "COF_4",
    "SOS_1", "SOS_2", "SOS_3", "SOS_4",
    "SpectralResidual_1", "SpectralResidual_2",
    "SpectralResidual_3", "SpectralResidual_4",
    # RM and MD are the framework's own code, grouped under Stat by what they
    # compute (a moving average, a per-channel mean) rather than how.
    "RM_1", "RM_2", "RM_3",
    "MD_1",
    # Table I's remaining Stat rows, from the vendored TSB-AD subset. POLY is
    # UNIVARIATE ONLY — see UNIVARIATE_FAMILIES.
    "KMEANSAD_1", "KMEANSAD_2", "KMEANSAD_3",
    "POLY_1", "POLY_2", "POLY_3",
    # AutoEncoder is the one Table I neural row PyOD ships, so it takes the
    # generic `train_pyod` path. Three instances, one per encoder shape in
    # TSB-AD's own sweep.
    "AutoEncoder_1", "AutoEncoder_2", "AutoEncoder_3",
    "RNN_1", "RNN_2", "RNN_3", "RNN_4",
    "LSTMVAE_1", "LSTMVAE_2", "LSTMVAE_3", "LSTMVAE_4",
    "DGHL_1", "DGHL_2", "DGHL_3", "DGHL_4",
    # The only one of PyOD 3's seven time-series models that survived vetting.
    # The other five are out for reasons worth not re-testing: MatrixProfile
    # raises NotImplementedError by design; KShape and SAND cost 13-14 min per
    # SMD entity for ~half LOF's F1; TimeSeriesOD and AnomalyTransformer score
    # identical input differently across runs and expose no seed.
    "LSTMAD_1", "LSTMAD_2", "LSTMAD_3",
    # Table I's six remaining neural rows, from the vendored TSB-AD subset. Each
    # varies its own subsequence length rather than `contamination`: most TSB-AD
    # constructors do not accept one, and where they do it moves a threshold
    # this pipeline replaces with its own sweep.
    #
    # TIMESNET is Table I's "TimeNet [87]"; TSB-AD ships TimesNet.py and no
    # TimeNet.py, so the table's spelling is taken to be a typo.
    #
    # DONUT_1 is a 30-step instance this project added: upstream's sweep is
    # [60, 90, 120], every value of which exceeds SMD's 37-row Thompson window,
    # so the family scored nothing there (posterior norm 0.000000, measured).
    "DONUT_1", "DONUT_2", "DONUT_3",
    "OmniAnomaly_1", "OmniAnomaly_2",
    "USAD_1", "USAD_2",
    "TRANAD_1", "TRANAD_2",
    "FITS_1", "FITS_2",
    "TIMESNET_1", "TIMESNET_2",
    # Foundation Models. The paper EXCLUDES these from the RAMSeS pool ("they
    # showed inconsistent performance"), so this pool is a superset of the
    # paper's — worth knowing when comparing numbers.
    #
    # All three are pretrained and frozen: fitting learns nothing and exists
    # only so they checkpoint like everything else. CHRONOS comes from
    # `chronos-forecasting`, not TSB-AD's autogluon route: 17 packages vs 69.
    "OFA_1", "OFA_2",
    "TIMESFM_1", "TIMESFM_2",
    "CHRONOS_1", "CHRONOS_2",
    # Graph Based, distinguished by WHAT the graph is over:
    #   LUNAR   — over SAMPLES. `detector__random_state` is NOT optional:
    #             unseeded it scores 3.039 apart on two fits of identical input,
    #             the same fault TimeSeriesOD and AnomalyTransformer were
    #             refused for.
    #   Series2Graph — over SUBSEQUENCES. UNIVARIATE ONLY, and the one detector
    #             not in the repository: see `Algorithms/tsb_ad/models/
    #             README_Series2Graph.md` for how to fetch it.
    #   MTADGAT — over CHANNELS. The only member saying anything about
    #             inter-sensor structure, which is what keeps the group
    #             non-degenerate on SKAB and SMD.
    "LUNAR_1", "LUNAR_2", "LUNAR_3", "LUNAR_4",
    "Series2Graph_1", "Series2Graph_2", "Series2Graph_3",
    "MTADGAT_1", "MTADGAT_2",
)

DETECTOR_FAMILIES = ("LOF", "NN", "CBLOF", "ABOD", "KDE",
                     "IFOREST", "HBOS", "PCA", "OCSVM", "MCD",
                     "COF", "SOS", "SpectralResidual", "RM", "MD",
                     "KMEANSAD", "POLY",
                     "AutoEncoder", "RNN", "LSTMVAE", "DGHL", "LSTMAD",
                     "DONUT", "OmniAnomaly", "USAD", "TRANAD", "FITS",
                     "TIMESNET", "OFA", "TIMESFM", "CHRONOS",
                     "LUNAR", "Series2Graph", "MTADGAT")

# Families reached through `Algorithms.tsbad_model.TSBADModel` rather than PyOD.
# The set is "detectors with TSB-AD's whole-series interface" — `(n_timesteps,
# n_channels)` in, one score per timestep out — not "detectors vendored in
# Algorithms/tsb_ad": CHRONOS and MTADGAT live in `Algorithms/` and
# Series2Graph is fetched. AutoEncoder is deliberately absent; PyOD ships it.
TSBAD_FAMILIES: FrozenSet[str] = frozenset({
    "KMEANSAD", "POLY", "DONUT", "OmniAnomaly", "USAD", "TRANAD", "FITS",
    "TIMESNET", "OFA", "TIMESFM", "CHRONOS", "Series2Graph", "MTADGAT"})

# Families that cut their own subsequences out of whatever call they are given,
# so a call shorter than that subsequence has nothing to cut. All INDUCTIVE —
# the same row scores identically whatever it travels with — so a whole-series
# batch changes no result, it only removes the boundary. (Contrast
# TRANSDUCTIVE_FAMILIES, where one call DEFINES the score.)
#
# Two places need this answer and drifted apart when only one had it:
# `model_selection_utils` sizes the scoring batch, `TrainModels.
# _diagnostic_batch_size` the post-fit plotting loop.
WHOLE_SERIES_FAMILIES: FrozenSet[str] = (
    frozenset({"LSTMAD"}) | (TSBAD_FAMILIES - frozenset({"POLY", "Series2Graph"})))

# Usable on UCR, dropped on SKAB (9 channels) and SMD (38). Declared here so the
# web UI can hide them before a run rather than fail during one; `app.py` drops
# them too, and `Algorithms.tsbad_model.UNIVARIATE_ONLY` carries the refusal.
#   POLY    — CANNOT. `np.polyfit` raises "Polynomial must be 1d only".
#   TIMESFM — CAN, but must not: ~13 min per scoring call on 38 channels
#             (measured) against ~0.6 s for Chronos-Bolt, seven calls per run.
#   Series2Graph — CANNOT. Its wrapper opens with `data.squeeze()` and embeds a
#             scalar subsequence into a 2-D phase space. This is why MTADGAT is
#             worth having: it keeps the Graph group runnable on SKAB and SMD.
UNIVARIATE_FAMILIES: FrozenSet[str] = frozenset(
    {"POLY", "TIMESFM", "Series2Graph"})

# The mirror image, dropped on a 1-channel entity.
#   ABOD — CAN, but must not. Angles need >1 dimension; at d=1 every difference
#          vector is collinear, so only the magnitude denominator varies. It
#          does not raise — measured spread runs 0.18 at d=5 to 2.7e10 at d=1,
#          and 2.0e21 on UCR 028, while still ranking plausibly.
MULTIVARIATE_FAMILIES: FrozenSet[str] = frozenset({"ABOD"})

# Families whose score is a function of the CALL'S OWN ROWS, not of what `fit`
# saw. COF, SOS and SpectralResidual score against their companions in the same
# call (COF's is literally `distance_matrix(X, X)`); POLY and Series2Graph reach
# it the other way, by ignoring the argument and scoring what they last fitted,
# so the adapter refits per call.
#
# The consequence is the same either way: a row's score would depend on where
# `eval_batch_size` happened to cut — the same window scored 1.003744 and
# 0.966958 under COF in two batches. `Utils/model_selection_utils` therefore
# hands these families the WHOLE series in one call, making the score a
# deterministic function of (entity, row). `Utils/test_pipeline_spec` asserts
# both the routing and the determinism, since the two models excluded for
# irreproducibility would otherwise look identical to these.
#
# Estimator limits, not pipeline ones: COF raises IndexError below its
# `n_neighbors` (20) rows, so Thompson — whose windows are `n_timesteps * 0.8 /
# iterations` — needs a ~1,250-timestep entity before COF can compete.
# SpectralResidual needs `score_window` (3) rows and returns three scores for
# one row.
TRANSDUCTIVE_FAMILIES: FrozenSet[str] = frozenset(
    {"COF", "SOS", "SpectralResidual", "POLY", "Series2Graph"})

# The paper's Table I taxonomy, EXTENDED with a fourth group — worth stating
# plainly when the thesis reproduces Table I. Keys are the identifier everything
# keys off (API value, CSS suffix); GROUP_LABELS carries what a reader sees.
#
# Two mappings invite the opposite guess:
#   * The NN FAMILY is k-Nearest Neighbors and belongs to the Stat GROUP; the
#     group called NN is Neural Networks. So the Neural Networks button does not
#     select the NN detector. The collision is the paper's.
#   * MD is an nn.Module but learns one mean per channel, so it is grouped by
#     what it computes, not how it is implemented.
#
# A detector earns `Graph` by reading its score off a graph quantity (degree,
# path, message passing) — PyOD's own criterion. SOS and COF are near misses
# that stay in Stat because PyOD files them Probabilistic and Proximity-Based,
# and following upstream beats a classification only this repo would use.
DETECTOR_GROUPS: Dict[str, tuple] = {
    "NN": ("AutoEncoder", "RNN", "LSTMVAE", "DGHL", "LSTMAD", "DONUT",
           "OmniAnomaly", "USAD", "TRANAD", "FITS", "TIMESNET"),
    "Stat": ("LOF", "NN", "CBLOF", "ABOD", "KDE", "IFOREST", "HBOS", "PCA",
             "OCSVM", "MCD", "COF", "SOS", "SpectralResidual", "RM", "MD",
             "KMEANSAD", "POLY"),
    "FM": ("OFA", "TIMESFM", "CHRONOS"),
    "Graph": ("LUNAR", "Series2Graph", "MTADGAT"),
}


# What a reader sees. Spelled out rather than abbreviated because "NN" next to
# a detector family also called NN is the misreading this taxonomy most invites.
GROUP_LABELS: Dict[str, str] = {
    "NN": "Neural Networks",
    "Stat": "Statistical",
    "FM": "Foundation",
    "Graph": "Graph Based",
}


# ── Abbreviations ───────────────────────────────────────────────────────────
#
# The pool name is CANONICAL — checkpoint filenames, `--detectors` tokens, IR
# atoms, result trees — and is also what the reader sees by default. This map
# runs the other way: a shortening for the two figures that cannot fit a full
# name. NOTHING MAY JOIN ON IT; an abbreviation is ink, never a key.
#
# Only these four shorten. Every other family already IS its published acronym
# (LOF, HBOS, USAD, FITS, LUNAR, OFA) or the framework's own short name.
FAMILY_ABBREVIATIONS: Dict[str, str] = {
    "SpectralResidual": "SR",
    "AutoEncoder": "AE",
    "OmniAnomaly": "OA",
    "Series2Graph": "S2G",
}

_ABBREV_BY_UPPER = {k.upper(): v for k, v in FAMILY_ABBREVIATIONS.items()}


def family_abbrev(family: str) -> str:
    """'OmniAnomaly' -> 'OA'. A family with no entry comes back unchanged."""
    return _ABBREV_BY_UPPER.get(str(family).upper(), str(family))


def abbreviate_detector(name: str) -> str:
    """'OmniAnomaly_2' -> 'OA_2'. The instance suffix is kept verbatim.

    Total on any input: callers are figure labels that also carry non-detector
    text ("ensemble", a channel name), which must pass through untouched.
    """
    text = str(name)
    family, sep, suffix = text.partition("_")
    if not sep:
        return family_abbrev(text)
    short = family_abbrev(family)
    return f"{short}{sep}{suffix}" if short != family else text


def abbreviation_legend(names) -> Dict[str, str]:
    """{'SR_1': 'SpectralResidual_1'}, for the names that actually shorten.

    Short to long, the direction the key is READ. Empty when nothing shortens,
    so a caller can skip the note rather than draw an empty box.
    """
    out = {}
    for name in names or ():
        short = abbreviate_detector(name)
        if short != str(name):
            out[short] = str(name)
    return out


def group_of(family: str) -> Optional[str]:
    """'RNN' -> 'NN', 'LOF' -> 'Stat'. None for a family in no group, which
    `Utils/test_pipeline_spec` forbids."""
    for group, members in DETECTOR_GROUPS.items():
        if family in members:
            return group
    return None

# Sub-stages of the model-selection phase (pipeline stage 6).
ALL_STAGES: FrozenSet[str] = frozenset({"ga", "thompson", "gan", "offby", "montecarlo"})

STAGE_GROUPS: Dict[str, FrozenSet[str]] = {
    "all": ALL_STAGES,
    "robustness": frozenset({"gan", "offby", "montecarlo"}),
}

# Synthetic anomalies injectable at pipeline stage 4, mirroring
# InjectAnomalies._VALID_ANOMALY_TYPES, which is the authority on what injects.
ALL_ANOMALY_TYPES: Tuple[str, ...] = (
    "spikes", "contextual", "flip", "speedup", "noise", "cutoff",
    "scale", "wander", "average",
)

DEFAULT_ANOMALY_TYPE = "spikes"

# Metrics the fitness function is built from. One or more may be chosen, each
# with a weight; the fitness is their weighted mean, and it is what the GA,
# Thompson and the final ensemble-vs-single comparison all maximise.
#
# A spec is either a sequence of metric names (equal weights) or a
# name -> weight mapping. Everything downstream goes through `metrics_required`
# and `metric_weights`, which accept both.
DECISION_METRICS: Dict[str, str] = {"f1": "F1", "pr_auc": "PR-AUC", "vus": "VUS"}

DEFAULT_DECISION_METRICS: Tuple[str, ...] = ("f1", "pr_auc")

# Iteration number the explainability artifacts are written under. Deliberately
# distinct from the CLI --iteration (which sizes the online windows), so IR/NL
# filenames stay stable across online configurations.
OFFLINE_ITERATION = 0

# Minimum detectors a run can be meaningful with: GA fitness, Markov rank
# aggregation and the off-by pairwise surrogates are all vacuous with one.
MIN_DETECTORS = 2

# The narrator. Owned here, not in Explainability/llm.py, because the web UI
# reports which model produced a set of explanations and a second copy would
# eventually name a model that never saw them.
DEFAULT_LLM_MODEL = "qwen3.5:9b-mlx"
DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"


# How a dataset is SHOWN; the CLI, directories and paths keep the real key.
# Owned here rather than in `WebUI/catalog.py` because the report and the web UI
# disagreed, naming one run two ways. `anomaly_archive` is why this is a table
# and not `.upper()`.
DATASET_LABELS: Dict[str, str] = {
    "skab": "SKAB", "smd": "SMD", "anomaly_archive": "UCR",
    "msl": "MSL", "smap": "SMAP", "apple": "Apple",
}


def dataset_label(key: str) -> str:
    """'skab' -> 'SKAB', 'anomaly_archive' -> 'UCR'. Unknown keys upper-case."""
    return DATASET_LABELS.get(str(key).lower(), str(key).upper())


def family_of(detector: str) -> str:
    """'CBLOF_2' -> 'CBLOF'."""
    return str(detector).rsplit("_", 1)[0]


def families_for(detectors: Sequence[str]) -> List[str]:
    """The architecture families needed to train `detectors`, in canonical order.

    Note the granularity mismatch: training is per FAMILY, so asking for NN_1
    alone still trains NN_1..NN_3 (the family's whole hyperparameter grid).
    """
    wanted = {family_of(d) for d in detectors}
    return [f for f in DETECTOR_FAMILIES if f in wanted]


def parse_anomaly_type(text: Optional[str]) -> str:
    """One anomaly type name -> its canonical spelling."""
    if text is None:
        return DEFAULT_ANOMALY_TYPE
    tok = str(text).strip().lower()
    if tok not in ALL_ANOMALY_TYPES:
        raise ValueError(
            f"--anomaly_type: unknown type '{text}'. Valid types: "
            f"{', '.join(ALL_ANOMALY_TYPES)}")
    return tok


def parse_anomaly_rate(text) -> Optional[float]:
    """Target fraction of timesteps to label anomalous, or None for the
    per-type defaults in Model_Selection/anomaly_parameters.py."""
    if text is None or str(text).strip() == "":
        return None
    try:
        rate = float(text)
    except (TypeError, ValueError):
        raise ValueError(f"--anomaly_rate: '{text}' is not a number")
    if not 0.0 < rate <= 1.0:
        raise ValueError(
            f"--anomaly_rate: must be greater than 0 and at most 1, got {rate}")
    return rate


def parse_decision_metrics(text) -> Union[Tuple[str, ...], Dict[str, float]]:
    """Metric names, optionally weighted -> a fitness spec.

    Accepts 'f1', 'f1,pr_auc' and 'f1:0.5,pr_auc:0.3,vus:0.2'. A metric with no
    explicit weight counts 1; weights are normalised, so they need not sum to 1.
    A zero weight drops its metric, which is what keeps `ranking_metrics_for`
    honest. Uniform weights collapse back to a plain tuple so the common case
    stays one spelling.
    """
    if text is None or (isinstance(text, str) and not text.strip()):
        return DEFAULT_DECISION_METRICS
    if isinstance(text, str):
        tokens = text.split(",")
    elif isinstance(text, dict):
        tokens = [f"{k}:{v}" for k, v in text.items()]
    else:
        tokens = list(text)
    weights: Dict[str, float] = {}
    unknown = []
    for raw in tokens:
        tok = str(raw).strip()
        if not tok:
            continue
        name, sep, weight = tok.partition(":")
        name = name.strip().lower().replace("-", "_")
        if name not in DECISION_METRICS:
            unknown.append(tok)
            continue
        if sep:
            try:
                value = float(weight)
            except ValueError:
                raise ValueError(
                    f"--decision_metric: '{weight.strip()}' is not a weight for {name}")
            if value < 0:
                raise ValueError(
                    f"--decision_metric: weight for {name} must not be negative")
        else:
            value = 1.0
        weights[name] = weights.get(name, 0.0) + value
    if unknown:
        raise ValueError(
            f"--decision_metric: unknown metric(s) {', '.join(unknown)}. "
            f"Valid metrics: {', '.join(DECISION_METRICS)}")
    chosen = {m: weights[m] for m in DECISION_METRICS if weights.get(m, 0.0) > 0}
    if not chosen:
        raise ValueError("--decision_metric: choose at least one metric")
    if len(set(chosen.values())) == 1:
        return tuple(chosen)
    total = sum(chosen.values())
    return {m: w / total for m, w in chosen.items()}


def format_decision_metrics(spec) -> str:
    """A spec -> the --decision_metric spelling that parses back to it."""
    weights = metric_weights(spec)
    if len(set(weights.values())) == 1:
        return ",".join(weights)
    return ",".join(f"{m}:{round(w, 4):g}" for m, w in weights.items())


def decision_metric_label(spec) -> str:
    """('f1','pr_auc') -> 'F1 + PR-AUC'."""
    chosen = metrics_required(spec)
    return " + ".join(DECISION_METRICS.get(m, m) for m in chosen)


def decision_metric_formula(spec) -> str:
    """('f1','pr_auc') -> '0.5 * F1 + 0.5 * PR-AUC'; a single metric is itself."""
    weights = metric_weights(spec)
    if len(weights) == 1:
        name = next(iter(weights))
        return DECISION_METRICS.get(name, name)
    return " + ".join(f"{round(w, 3):g} * {DECISION_METRICS.get(m, m)}"
                      for m, w in weights.items())


def metrics_required(spec) -> Tuple[str, ...]:
    """The raw metrics a fitness spec needs computed, in canonical order.

    Single choke point with `combine_metrics`, so callers that skip expensive
    metrics (VUS) stay correct without knowing what the spec is.
    """
    if isinstance(spec, str):
        spec = [spec]
    if isinstance(spec, dict):
        # A zero weight means "not chosen", so it must not reach
        # `ranking_metrics_for` as a metric the run cares about.
        chosen = {str(m).lower().replace("-", "_")
                  for m, w in spec.items() if float(w) > 0}
    else:
        chosen = {str(m).lower().replace("-", "_") for m in spec}
    return tuple(m for m in DECISION_METRICS if m in chosen)


def metric_weights(spec) -> Dict[str, float]:
    """A fitness spec -> metric -> normalised weight, in canonical order.

    A sequence of names weights them equally; a mapping is normalised, so
    {'f1': 2, 'pr_auc': 1} and {'f1': 0.667, 'pr_auc': 0.333} mean the same.
    """
    chosen = metrics_required(spec)
    if not chosen:
        raise ValueError("no metric chosen")
    if isinstance(spec, dict):
        given = {str(k).lower().replace("-", "_"): float(v) for k, v in spec.items()}
        raw = {m: given[m] for m in chosen}
        if any(w < 0 for w in raw.values()):
            raise ValueError("decision metric weights must not be negative")
    else:
        raw = {m: 1.0 for m in chosen}
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("decision metric weights must not all be zero")
    return {m: w / total for m, w in raw.items()}


def restrict_metrics(spec, metrics) -> Union[Tuple[str, ...], Dict[str, float]]:
    """The same spec over `metrics` only, keeping the relative weights."""
    keep = metrics_required(metrics)
    if not keep:
        raise ValueError("no metric chosen")
    weights = metric_weights(spec)
    kept = {m: weights[m] for m in keep if m in weights}
    if not kept:
        raise ValueError("no metric chosen")
    if len(set(kept.values())) == 1:
        return tuple(kept)
    total = sum(kept.values())
    return {m: w / total for m, w in kept.items()}


def combine_metrics(spec, scores: Dict[str, float]) -> float:
    """Raw metric values -> the single number every search maximises.

    A metric that could not be computed drops out and the remaining weights are
    renormalised, so one unavailable term (VUS on a short window) narrows the
    fitness instead of voiding it. All of them missing gives nan.
    """
    weights = metric_weights(spec)
    usable = {}
    for m, w in weights.items():
        value = float(scores[m])
        if not math.isnan(value):
            usable[m] = w
    total = sum(usable.values())
    if total <= 0:
        return float("nan")
    return sum(w * float(scores[m]) for m, w in usable.items()) / total


def ranking_metrics_for(spec) -> Tuple[str, ...]:
    """Which rankings the robustness stages publish: one per chosen metric."""
    return metrics_required(spec)


def parse_stages(text: Optional[str]) -> Set[str]:
    """Comma-separated stage tokens (plus the group names) -> a set of stages.

    Raises ValueError with the message the CLI surfaces via parser.error().
    """
    if text is None:
        return set(ALL_STAGES)
    selected: Set[str] = set()
    for tok in (t.strip().lower() for t in str(text).split(",") if t.strip()):
        if tok in STAGE_GROUPS:
            selected |= STAGE_GROUPS[tok]
        elif tok in ALL_STAGES:
            selected.add(tok)
        else:
            raise ValueError(
                f"--stages: unknown stage '{tok}'. Valid tokens: "
                f"{', '.join(sorted(ALL_STAGES))}, all, robustness")
    return selected if selected else set(ALL_STAGES)


def parse_detectors(text: Optional[str]) -> Optional[List[str]]:
    """Comma-separated detector names -> canonical-order list, or None for all.

    Returning canonical order (not the user's order) and de-duplicating means an
    equivalent selection always produces byte-identical argv, which keeps the
    web UI's command preview and the argv tests stable.

    Validation is against ALL_DETECTORS, never against what happens to be on
    disk: some entities carry stale checkpoints (e.g. RNN_*.pth under SMD) that
    are not selectable models.
    """
    if text is None:
        return None
    requested = [t.strip() for t in str(text).split(",") if t.strip()]
    if not requested:
        return None
    canonical = {d.lower(): d for d in ALL_DETECTORS}
    seen, unknown = set(), []
    for tok in requested:
        key = tok.lower()
        if key in canonical:
            seen.add(canonical[key])
        else:
            unknown.append(tok)
    if unknown:
        raise ValueError(
            f"--detectors: unknown detector(s) {', '.join(unknown)}. "
            f"Valid names: {', '.join(ALL_DETECTORS)}")
    if len(seen) < MIN_DETECTORS:
        raise ValueError(
            f"--detectors: need at least {MIN_DETECTORS} detectors to run "
            f"model selection, got {len(seen)}")
    return [d for d in ALL_DETECTORS if d in seen]
