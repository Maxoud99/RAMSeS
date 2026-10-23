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
    "tsb-ad-u": "TSB-AD-U", "tsb-ad-m": "TSB-AD-M",
}

# Datasets whose loader gives the TEST split its real anomaly labels; on these
# the online phase scores on them instead of injecting (--online_labels real,
# the default there). Checked loader by loader in Datasets/load.py:
#   tsb-ad-u / tsb-ad-m  the file's Label column, test part           (yes)
#   anomaly_archive      the anomaly range in the UCR file name        (yes)
#   smd                  test_label/<machine>.txt (presplit layout)    (yes)
#   msl / smap           NASA labeled_anomalies.csv sequences          (yes)
#   skab                 load_csv_file labels with the LAST column, which in
#                        SKAB is `changepoint`, not `anomaly` (the anomaly
#                        column is read as a channel)                  (no)
# Any dataset in the presplit layout (train/ test/ test_label/) has them too;
# Datasets.load.provides_real_labels and the web UI's catalog add that case.
REAL_LABEL_DATASETS = frozenset({"tsb-ad-u", "tsb-ad-m", "anomaly_archive",
                                 "smd", "servermachinedataset", "msl", "smap"})


def dataset_has_real_labels(key) -> bool:
    return str(key).lower() in REAL_LABEL_DATASETS


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


def parse_anomaly_types(value) -> List[str]:
    """One or more anomaly types -> their canonical spellings, in the order
    given and without repeats.

    Accepts None (the default type), a comma-separated string ("spikes,noise",
    which is how the CLI and the web UI's command carry it) or a list. Every
    type is injected: `Inject` makes one copy of the series per type and per
    parameter combination of that type, and the stages evaluate on all of them.
    """
    if value is None:
        return [DEFAULT_ANOMALY_TYPE]
    parts = value if isinstance(value, (list, tuple)) else str(value).split(",")
    tokens = [str(p).strip() for p in parts if str(p).strip()]
    if not tokens:
        raise ValueError("--anomaly_type: select at least one anomaly type. "
                         f"Valid types: {', '.join(ALL_ANOMALY_TYPES)}")
    types: List[str] = []
    for tok in tokens:
        canonical = parse_anomaly_type(tok)
        if canonical not in types:
            types.append(canonical)
    return types


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


# ── Online phase ────────────────────────────────────────────────────────────
#
# The online phase replays the last 20% of the test series window by window
# and compares re-optimisation POLICIES. The scenario keys are owned here
# because four places spell them — the CLI (`--strategy`), the web UI's
# validation, the results writer's file names and the run page's log parser —
# and a second spelling anywhere would split one scenario into two.
ONLINE_SCENARIOS: Tuple[str, ...] = (
    "fixed", "periodic", "threshold", "adaptive_threshold")

# A random selection kept for the whole stream. A CLI-only baseline: the UI's
# comparison is between policies applied to the SAME offline selection, and
# this one replaces the selection instead.
ONLINE_BASELINE_SCENARIOS: Tuple[str, ...] = ("fixed_random",)

ONLINE_SCENARIO_LABELS: Dict[str, str] = {
    "fixed": "Fixed (no re-optimization)",
    "periodic": "Periodic re-optimization",
    "threshold": "Re-optimization below a threshold",
    "adaptive_threshold": "Re-optimization below an adaptive threshold",
    "fixed_random": "Fixed random selection (baseline)",
}

# `--strategy all` runs the four policies on one replay of the same windows.
ONLINE_STRATEGY_ALL = "all"

# Spellings from before the four scenarios existed. `adaptive` WAS the periodic
# policy (re-optimise every --update_interval windows); keeping it as an alias
# rather than an error means old commands and saved run configs still work.
# Hyphens are folded to underscores before the lookup, so `fixed-best` and
# `fixed_best` are one token, as are `fixed-random` and `fixed_random`.
ONLINE_STRATEGY_ALIASES: Dict[str, str] = {
    "adaptive": "periodic",
    "fixed_best": "fixed",
}

# Defaults for the policy parameters. Each is also a key in Configs/config.yml
# (where the CLI falls back to when a flag is omitted) and a field the web UI
# may send, so all three agree on what an unset value means.
DEFAULT_ONLINE_STRATEGY = "periodic"   # what `--strategy adaptive` always did
DEFAULT_UPDATE_INTERVAL = 5            # M: periodic re-optimises every M windows
DEFAULT_DROP_PCT = 20.0                # X: relative drop below the baseline, in %
DEFAULT_ADAPTIVE_WINDOW = 10           # N: scores in the adaptive rolling window
DEFAULT_ADAPTIVE_K = 2.0               # k: threshold = mean - k * std
DEFAULT_REOPT_DELAY = 2                # L: a re-optimisation started at window t
                                       #    is switched in at window t + L
DEFAULT_REOPT_TIMEOUT = 3600.0         # seconds one re-optimisation may run


# ── Online windows: the split, the window geometry, and feasibility ─────────
#
# app.py cuts the online windows with these functions and the web UI's
# feasibility check calls the same ones, so the two cannot disagree about
# whether an entity can run the online phase.

# The last 20% of the test series is the online part (as per the paper).
OFFLINE_FRACTION = 0.8
# Windows are never shorter than this, however large --iteration is: shorter
# windows hold too few samples for the range-based metrics to mean anything.
MIN_WINDOW_SIZE = 100
MIN_STEP_SIZE = 5
# Window 0 is replayed, so one window is enough for the phase to run (it is
# then the baseline and nothing can trigger, but every scenario is scored).
MIN_ONLINE_WINDOWS = 1
# --inject_online_regime runs these through Inject: one shifted copy of the
# online part per parameter combination, concatenated into a longer stream.
REGIME_SHIFT_TYPES: Tuple[str, ...] = ("scale", "wander")

# What the online windows are scored on (--online_labels). "real": the test
# series' own labels, with nothing injected online — the default wherever the
# loader provides them (REAL_LABEL_DATASETS). "injected": synthetic anomalies
# injected once into the online stream, the behaviour from before real labels
# were used, kept for comparison and for datasets without labels.
ONLINE_LABELS_REAL = "real"
ONLINE_LABELS_INJECTED = "injected"
ONLINE_LABEL_SOURCES: Tuple[str, ...] = (ONLINE_LABELS_REAL, ONLINE_LABELS_INJECTED)
ONLINE_LABEL_SOURCE_LABELS: Dict[str, str] = {
    ONLINE_LABELS_REAL: "Real labels — no injection online",
    ONLINE_LABELS_INJECTED: "Injected anomalies",
}
NO_LABELLED_ANOMALY_REASON = ("No labelled anomaly in the online part — nothing to "
                              "score with real labels.")


def online_split(total_samples: int) -> Tuple[int, int]:
    """(offline_size, online_size) of a test series of `total_samples`."""
    total = int(total_samples)
    offline = int(total * OFFLINE_FRACTION)
    return offline, total - offline


def online_window_geometry(online_size: int, iteration: int) -> Tuple[int, int]:
    """(window_size, step_size) for an online part of `online_size` samples.

    window_size = online_size / iteration, never below MIN_WINDOW_SIZE; the
    step is 5% of the window (95% overlap), never below MIN_STEP_SIZE.
    """
    online_size, iteration = int(online_size), int(iteration)
    window = (max(int(online_size / iteration), MIN_WINDOW_SIZE) if iteration > 0
              else max(int(online_size * 0.2), MIN_WINDOW_SIZE))
    return window, max(int(window * 0.05), MIN_STEP_SIZE)


def count_windows(length: int, window_size: int, step_size: int) -> int:
    """Windows of `window_size` at stride `step_size` that fit in `length`,
    the first starting at 0 (Thompson_Sampling.initialize_sliding_windows
    counts the same way)."""
    length, window_size, step_size = int(length), int(window_size), int(step_size)
    if window_size <= 0 or step_size <= 0 or length < window_size:
        return 0
    return (length - window_size) // step_size + 1


def injection_copies(anomaly_types: Sequence[str]) -> int:
    """How many copies of a series `Inject(data, anomaly_types)` concatenates:
    one per parameter combination of each type's grid (speedup, scale and
    wander have 4, cutoff and average 2, the others 1)."""
    from Model_Selection.anomaly_parameters import ANOMALY_PARAM_GRID
    total = 0
    for name in anomaly_types:
        combos = 1
        for values in ANOMALY_PARAM_GRID[name].values():
            combos *= len(values)
        total += combos
    return total


def window_label_counts(labels: Sequence, window_size: int, step_size: int,
                        n_windows: Optional[int] = None) -> Dict[str, int]:
    """How the online windows over `labels` (the online part's real labels)
    split: how many hold a labelled anomaly, and how many are SCORABLE — hold
    both classes, so F1 and PR-AUC are defined. A window that is anomalous
    throughout holds an anomaly and is still unscorable.

    Windows as `count_windows` places them; `n_windows` caps the count.
    """
    flags = [1 if float(v) > 0 else 0 for v in labels]
    window_size, step_size = int(window_size), int(step_size)
    total = count_windows(len(flags), window_size, step_size)
    if n_windows is not None:
        total = min(total, int(n_windows))
    prefix = [0]
    for f in flags:
        prefix.append(prefix[-1] + f)
    with_anomaly = scorable = 0
    for i in range(total):
        lo = i * step_size
        k = prefix[lo + window_size] - prefix[lo]
        with_anomaly += k > 0
        scorable += 0 < k < window_size
    return {"n_windows": total, "n_windows_with_anomaly": with_anomaly,
            "n_windows_scorable": scorable, "n_anomalous_online": prefix[-1]}


def online_feasibility(total_samples: int, iteration: int,
                       online_labels: Optional[Sequence] = None) -> Dict[str, object]:
    """Can an entity whose test series has `total_samples` run the online phase?

    Judged on the plain online part, before any injection: anomaly copies are
    cut at the same positions (so they never add windows), and the regime-shift
    stream of --inject_online_regime is a synthetic stretch of that same part,
    which does not make a too-short entity long enough. `regime_n_windows` is
    how many windows that longer stream would hold.

    `online_labels`, the online part's REAL labels, asks the question for
    --online_labels real: windows are then scored on those labels, so an
    online part with no scorable window (no labelled anomaly, or anomalous
    throughout) has nothing to score and is not feasible. Without them the
    label fields are None and `label_source` is "injected".
    """
    offline_size, online_size = online_split(total_samples)
    window_size, step_size = online_window_geometry(online_size, iteration)
    n_windows = count_windows(online_size, window_size, step_size)
    ok = n_windows >= MIN_ONLINE_WINDOWS
    reason = None
    if not ok:
        reason = (f"The online part has {online_size} samples (the last "
                  f"{100 - int(OFFLINE_FRACTION * 100)}% of the {int(total_samples)}-sample "
                  f"test series), but online windows are at least "
                  f"{MIN_WINDOW_SIZE} samples long, so not one window of "
                  f"{window_size} fits.")
    counts = {"n_windows_with_anomaly": None, "n_windows_scorable": None,
              "n_anomalous_online": None}
    if online_labels is not None:
        counts = window_label_counts(online_labels, window_size, step_size)
        del counts["n_windows"]
        if ok and counts["n_windows_scorable"] == 0:
            ok = False
            reason = (NO_LABELLED_ANOMALY_REASON if counts["n_windows_with_anomaly"] == 0
                      else "Every online window that holds a labelled anomaly is "
                           "anomalous throughout, so no window has both classes — "
                           "nothing to score.")
    return {
        "ok": ok,
        "n_windows": n_windows,
        "window_size": window_size,
        "step_size": step_size,
        "online_size": online_size,
        "offline_size": offline_size,
        "total_samples": int(total_samples),
        "iteration": int(iteration),
        "min_window_size": MIN_WINDOW_SIZE,
        "regime_n_windows": count_windows(
            online_size * injection_copies(REGIME_SHIFT_TYPES), window_size, step_size),
        "label_source": (ONLINE_LABELS_REAL if online_labels is not None
                         else ONLINE_LABELS_INJECTED),
        **counts,
        "reason": reason,
    }


def parse_online_strategy(text, allow_baselines: bool = True) -> Tuple[str, ...]:
    """'all' -> the four scenarios; one scenario or alias -> (key,).

    `allow_baselines=False` is the web UI's spelling of "CLI-only": the random
    baseline is refused there with a message saying why.
    """
    if text is None or str(text).strip() == "":
        return (DEFAULT_ONLINE_STRATEGY,)
    tok = str(text).strip().lower().replace("-", "_")
    if tok == ONLINE_STRATEGY_ALL:
        return ONLINE_SCENARIOS
    tok = ONLINE_STRATEGY_ALIASES.get(tok, tok)
    if tok in ONLINE_SCENARIOS:
        return (tok,)
    if tok in ONLINE_BASELINE_SCENARIOS:
        if not allow_baselines:
            raise ValueError(
                f"--strategy: '{text}' is a command-line baseline, not one of "
                f"the compared scenarios")
        return (tok,)
    valid = list(ONLINE_SCENARIOS) + [ONLINE_STRATEGY_ALL]
    if allow_baselines:
        valid.append("fixed-random")
    raise ValueError(
        f"--strategy: unknown strategy '{text}'. Valid strategies: "
        f"{', '.join(valid)} (aliases: adaptive = periodic, fixed-best = fixed)")


def _parse_number(text, flag: str, kind):
    """`text` as an int or a finite float, or ValueError naming the flag.

    Booleans are refused outright: JSON `true` would otherwise read as 1, a
    valid update interval, and a form bug would run instead of failing.
    """
    what = "a whole number" if kind is int else "a number"
    if isinstance(text, bool):
        raise ValueError(f"{flag}: expected {what}, got {text!r}")
    try:
        value = float(str(text).strip().rstrip("%"))
    except (TypeError, ValueError):
        raise ValueError(f"{flag}: expected {what}, got {text!r}")
    if not math.isfinite(value):
        raise ValueError(f"{flag}: expected a finite number, got {text!r}")
    if kind is int:
        if not value.is_integer():
            raise ValueError(f"{flag}: expected {what}, got {text!r}")
        return int(value)
    return value


def parse_update_interval(text) -> int:
    """M, the periodic policy's gap in windows. At least 1."""
    if text is None or str(text).strip() == "":
        return DEFAULT_UPDATE_INTERVAL
    value = _parse_number(text, "--update_interval", int)
    if value < 1:
        raise ValueError(f"--update_interval: must be at least 1, got {value}")
    return value


def parse_drop_pct(text) -> float:
    """X, in percent of the baseline. Strictly between 0 and 100: at 0 every
    dip re-optimises, and at 100 the threshold is 0, which no score falls below."""
    if text is None or str(text).strip() == "":
        return DEFAULT_DROP_PCT
    value = _parse_number(text, "--drop_pct", float)
    if not 0.0 < value < 100.0:
        raise ValueError(
            f"--drop_pct: must be greater than 0 and less than 100, got {value:g}")
    return value


def parse_adaptive_window(text) -> int:
    """N, the adaptive threshold's rolling window. At least 2: the spread of a
    single score is zero, so N=1 would reduce the rule to 'any dip'."""
    if text is None or str(text).strip() == "":
        return DEFAULT_ADAPTIVE_WINDOW
    value = _parse_number(text, "--adaptive_window", int)
    if value < 2:
        raise ValueError(f"--adaptive_window: must be at least 2, got {value}")
    return value


def parse_adaptive_k(text) -> float:
    """k, how many standard deviations below the rolling mean. Not negative."""
    if text is None or str(text).strip() == "":
        return DEFAULT_ADAPTIVE_K
    value = _parse_number(text, "--adaptive_k", float)
    if value < 0:
        raise ValueError(f"--adaptive_k: must not be negative, got {value:g}")
    return value


def parse_max_online_windows(text) -> Optional[int]:
    """How many online windows to process, or None for all of them."""
    if text is None or str(text).strip().lower() in ("", "none", "null"):
        return None
    value = _parse_number(text, "--max_online_windows", int)
    if value < 1:
        raise ValueError(f"--max_online_windows: must be at least 1, got {value}")
    return value


def parse_reopt_delay(text) -> int:
    """L, the windows between a trigger and the switch to its result. 0 means
    the triggering window itself is re-scored with the new selection."""
    if text is None or str(text).strip() == "":
        return DEFAULT_REOPT_DELAY
    value = _parse_number(text, "--reopt_delay", int)
    if value < 0:
        raise ValueError(f"--reopt_delay: must be 0 or more, got {value}")
    return value


def parse_reopt_timeout(text) -> Optional[float]:
    """Seconds one re-optimisation may run, or None for no limit.

    '' (nothing set) is the default. None (a config `null`), 0, 'none',
    'null' and 'off' turn the limit off, as they do for `training_timeout`;
    resolve_online_settings tells an omitted config key from a null one."""
    if text is None or str(text).strip().lower() in ("none", "null", "off"):
        return None
    if str(text).strip() == "":
        return DEFAULT_REOPT_TIMEOUT
    value = _parse_number(text, "--reopt_timeout", float)
    if value < 0:
        raise ValueError(f"--reopt_timeout: must be 0 (no limit) or more, got {value:g}")
    return None if value == 0 else value


def parse_online_labels(text) -> Optional[str]:
    """'real' | 'injected', or None for the default ('auto', '' or unset):
    real labels wherever the dataset provides them (resolve_online_labels)."""
    if text is None or str(text).strip().lower() in ("", "auto", "default"):
        return None
    if isinstance(text, bool):
        raise ValueError(f"--online_labels: expected real or injected, got {text!r}")
    token = str(text).strip().lower()
    if token not in ONLINE_LABEL_SOURCES:
        raise ValueError(f"--online_labels: expected real or injected (or auto), "
                         f"got {text!r}")
    return token


def resolve_online_labels(requested: Optional[str], labels_available: bool,
                          inject_online_regime: bool = False,
                          dataset: Optional[str] = None) -> str:
    """The label source a run's online phase uses. Raises ValueError naming
    the flag when the request cannot be honoured.

    Unset means real labels when the dataset has them, injected otherwise.
    --inject_online_regime injects regime shifts INTO the online stream, and
    real-label mode injects nothing online, so the two are refused together
    (also when real is only the default) rather than one silently winning.
    """
    name = dataset_label(dataset) if dataset else "This dataset"
    source = requested or (ONLINE_LABELS_REAL if labels_available else ONLINE_LABELS_INJECTED)
    if source == ONLINE_LABELS_REAL and not labels_available:
        raise ValueError(
            f"--online_labels: {name} provides no labelled test anomalies, so real "
            f"labels cannot be used; pass --online_labels injected")
    if source == ONLINE_LABELS_REAL and inject_online_regime:
        raise ValueError(
            f"--online_labels: real-label mode injects nothing into the online stream, "
            f"but --inject_online_regime injects regime shifts into it; pass "
            f"--online_labels injected to use the regime shift"
            + ("" if requested else f" ({name} has real labels, so they are the default)"))
    return source


# CLI flag -> config.yml key, for the settings that may come from either. The
# strategy's key is spelled `online_strategy` because a bare `strategy` in a
# config file full of unrelated settings says nothing about which phase it is.
ONLINE_CONFIG_KEYS: Dict[str, str] = {
    "strategy": "online_strategy",
    "update_interval": "update_interval",
    "drop_pct": "drop_pct",
    "adaptive_window": "adaptive_window",
    "adaptive_k": "adaptive_k",
    "max_online_windows": "max_online_windows",
    "reopt_delay": "reopt_delay",
    "reopt_timeout": "reopt_timeout",
    "online_labels": "online_labels",
}


def resolve_online_settings(cli: Dict[str, object],
                            config: Dict[str, object]) -> Dict[str, object]:
    """The online phase's settings: each flag, else its config key, else the
    default above. Raises ValueError naming the flag on a bad value.

    `strategy` comes back as the canonical spelling the user meant ('all', or
    the scenario key an alias resolves to) and `online_scenarios` as the tuple
    of scenario keys to replay, in comparison order.
    """
    def pick(flag):
        value = cli.get(flag)
        return config.get(ONLINE_CONFIG_KEYS[flag]) if value is None else value

    # The timeout distinguishes an omitted key (the default) from an explicit
    # null (no limit), as `training_timeout` does, so it cannot use `pick`.
    timeout = cli.get("reopt_timeout")
    if timeout is None:
        key = ONLINE_CONFIG_KEYS["reopt_timeout"]
        timeout = config[key] if key in config else ""

    scenarios = parse_online_strategy(pick("strategy"))
    return {
        "strategy": (ONLINE_STRATEGY_ALL if len(scenarios) > 1 else scenarios[0]),
        "online_scenarios": scenarios,
        "update_interval": parse_update_interval(pick("update_interval")),
        "drop_pct": parse_drop_pct(pick("drop_pct")),
        "adaptive_window": parse_adaptive_window(pick("adaptive_window")),
        "adaptive_k": parse_adaptive_k(pick("adaptive_k")),
        "max_online_windows": parse_max_online_windows(pick("max_online_windows")),
        "reopt_delay": parse_reopt_delay(pick("reopt_delay")),
        "reopt_timeout": parse_reopt_timeout(timeout),
        # None = the default, resolved against the dataset in run_app.
        "online_labels": parse_online_labels(pick("online_labels")),
    }
