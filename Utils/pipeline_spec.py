"""
Canonical vocabulary of the model-selection pipeline: which detectors exist,
which sub-stages exist, and how the CLI spellings of both are parsed.

This module is deliberately **stdlib-only**. `Utils/utils.py` cannot host these
definitions because it imports torch, matplotlib and PIL, and the web UI has to
read the same vocabulary without dragging a 2 GB ML stack into a Flask process.
Keeping one definition here also retires the duplicate stage sets that used to
live in both `app.py` and `Utils/utils.py`.
"""

from typing import Dict, FrozenSet, List, Optional, Sequence, Set

# Base detector instances, in the order app.py loads them. Instance counts come
# from Model_Training/hyperparameter_grids.py: NN varies k (3 instances), every
# other family varies contamination over four values.
#
# Three groups, and the difference between them is only how they are trained:
#
#   * LOF, NN, CBLOF have bespoke classes in Algorithms/ and their own grids.
#   * ABOD and KDE also have bespoke classes and grids, and a branch each in
#     TrainModels.train_models.
#   * IFOREST, HBOS, PCA, OCSVM, MCD, SR have none of that. They
#     reach PyOD through the generic `train_pyod` fallback and
#     `Algorithms/pyod_model.PyodModel`, which names its checkpoints
#     `{FAMILY.upper()}_{i}` — which is why they are spelled in upper case here
#     even though the PyOD classes are `IForest`, `HBOS`, `SpectralResidual` and
#     so on. `Algorithms.pyod_model._class_in` resolves the case.
#
# Adding a family here is what makes it selectable: --detectors validates
# against this tuple, `families_for` turns it into what gets trained, and the
# web UI's chips are built from it.
#
# COF, SOS and SR are TRANSDUCTIVE and are admitted on that
# understanding — see TRANSDUCTIVE_FAMILIES below for what it costs and what
# makes it safe.
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
    # The transductive three. Placed here, between the point-wise PyOD families
    # and the framework's temporal models, because that is what they are: they
    # judge a point against its neighbours, not against a fitted model. COF and
    # SOS have bespoke classes in Algorithms/ and their own training branches;
    # SR reaches PyOD 3 through the generic `train_pyod` fallback
    # (its module is mapped in `Algorithms.pyod_model._TS_MODULES`).
    "COF_1", "COF_2", "COF_3", "COF_4",
    "SOS_1", "SOS_2", "SOS_3", "SOS_4",
    "SR_1", "SR_2", "SR_3", "SR_4",
    # RM and MD close out the Statistical group. They are the framework's own
    # code rather than PyOD's, but what they compute — a moving average and a
    # per-channel mean — is what the paper files under Stat, so they belong
    # beside the rest of that group rather than among the neural networks.
    "RM_1", "RM_2", "RM_3",
    "MD_1",
    # Table I's two remaining Statistical rows, which PyOD does not ship. They
    # come from the vendored TSB-AD subset through `Algorithms.tsbad_model`.
    # KMEANSAD is distance-to-centroid over sliding subsequences. POLY is local
    # polynomial residuals and is UNIVARIATE ONLY — see UNIVARIATE_FAMILIES.
    "KMEANSAD_1", "KMEANSAD_2", "KMEANSAD_3",
    "POLY_1", "POLY_2", "POLY_3",
    # The framework's own implementations, and the only detectors here with a
    # temporal model. Each was unusable until its own bug was fixed: RNN
    # reassembled its windows with the raw `window_step = -1` instead of the
    # step the loader had resolved it to, and LSTMVAE and DGHL pickled
    # `device = cuda` as state, so they allocated CUDA tensors on a CPU-only
    # machine. These are the detectors the paper's SKAB 1-1 example needs.
    # Table I's first Neural Network row, and the only one of its ten that PyOD
    # ships. No wrapper needed: `pyod.models.auto_encoder.AutoEncoder` is
    # reachable through `Algorithms.pyod_model._module_for`'s underscore-
    # insensitive fallback, so it trains on the generic `train_pyod` path. It is
    # the pool's fourth subsequence detector (window_size 64), which is what
    # Table I's AE means. Three instances, one per encoder shape in TSB-AD's own
    # AutoEncoder sweep.
    "AE_1", "AE_2", "AE_3",
    "RNN_1", "RNN_2", "RNN_3", "RNN_4",
    "LSTMVAE_1", "LSTMVAE_2", "LSTMVAE_3", "LSTMVAE_4",
    "DGHL_1", "DGHL_2", "DGHL_3", "DGHL_4",
    # PyOD 3's LSTM prediction-error detector — the only one of that library's
    # seven time-series models that survived vetting, and the only detector here
    # that models a subsequence through the shared wrapper. It does its own
    # windowing, so the framework hands it one row per timestep and its own
    # subsequence length is `detector__window_size` in its grid.
    #
    # Five of PyOD 3's seven time-series models remain out. MatrixProfile
    # raises NotImplementedError on decision_function by design. KShape and
    # SAND take 13-14 minutes per entity on SMD for roughly half LOF's F1.
    # TimeSeriesOD and AnomalyTransformer return DIFFERENT scores on two runs
    # of identical input and expose no seed parameter, which no amount of
    # batching fixes — AnomalyTransformer nearly slipped through the
    # transductivity check too: at 200 rows both companion sets scored 0.000000
    # and looked identical, and the dependence only showed at finer resolution
    # (0.000001 vs 0.000702).
    "LSTMAD_1", "LSTMAD_2", "LSTMAD_3",
    # The six Neural Network rows of Table I that PyOD does not ship, from the
    # vendored TSB-AD subset through `Algorithms.tsbad_model`. Like LSTMAD they
    # cut their own subsequences, so the framework hands them one row per
    # timestep and their own length is `detector__win_size`.
    #
    # Each varies that length rather than `contamination`: six of the eight
    # TSB-AD constructors do not accept a contamination at all, and where one is
    # accepted it moves a threshold this pipeline replaces with its own sweep.
    #
    # TIMESNET is Table I's "TimeNet [87] — temporal-variation features". The
    # description is TimesNet's own and TSB-AD ships TimesNet.py with no
    # TimeNet.py, so the table's spelling is taken to be a typo.
    # Three instances, not two: DONUT's upstream sweep is [60, 90, 120], whose
    # every value exceeds SMD's 37-row Thompson window, so on that entity the
    # family scored nothing at all (posterior norm 0.000000, measured). DONUT_1
    # is a 30-step instance added to give the family one arm Thompson can
    # actually pull; 60 and 90 remain upstream's.
    "DONUT_1", "DONUT_2", "DONUT_3",
    "OA_1", "OA_2",
    "USAD_1", "USAD_2",
    "TRANAD_1", "TRANAD_2",
    "FITS_1", "FITS_2",
    "TIMESNET_1", "TIMESNET_2",
    # Table I's Foundation Models, and the first members the FM group has had.
    # The paper excludes FMs from the RAMSeS candidate pool ("they showed
    # inconsistent performance") and reports them only in the TSB-AutoAD
    # setting, so these make the pool a superset of the paper's rather than a
    # match. Deliberate, and worth knowing when comparing numbers.
    #
    # All three are pretrained and frozen: fitting learns nothing, and training
    # exists only so they checkpoint like every other detector. OFA runs GPT-2
    # over patched windows; TIMESFM and CHRONOS forecast one step ahead and
    # score the squared error. CHRONOS comes from `chronos-forecasting` rather
    # than TSB-AD's autogluon route — 17 packages against 69 for the same model.
    "OFA_1", "OFA_2",
    "TIMESFM_1", "TIMESFM_2",
    "CHRONOS_1", "CHRONOS_2",
)

DETECTOR_FAMILIES = ("LOF", "NN", "CBLOF", "ABOD", "KDE",
                     "IFOREST", "HBOS", "PCA", "OCSVM", "MCD",
                     "COF", "SOS", "SR", "RM", "MD", "KMEANSAD", "POLY",
                     "AE", "RNN", "LSTMVAE", "DGHL", "LSTMAD",
                     "DONUT", "OA", "USAD", "TRANAD", "FITS",
                     "TIMESNET", "OFA", "TIMESFM", "CHRONOS")

# Families reached through `Algorithms.tsbad_model.TSBADModel` over the vendored
# code in `Algorithms/tsb_ad`, rather than through PyOD. Single owner of the
# fact, so `TrainModels.train_models` can route them and the tests can check
# that every one has a grid. AE is deliberately NOT here: PyOD ships it,
# and taking TSB-AD's fork instead would add a second copy of the same idea.
TSBAD_FAMILIES: FrozenSet[str] = frozenset({
    "KMEANSAD", "POLY", "DONUT", "OA", "USAD", "TRANAD", "FITS",
    "TIMESNET", "OFA", "TIMESFM", "CHRONOS"})

# Families that cut their own subsequences out of whatever call they are given,
# so a call shorter than that subsequence has nothing to cut. They are all
# INDUCTIVE — the same row scores identically whatever it travels with — so
# handing them the whole series in one batch changes no result, it only removes
# the boundary. (Contrast TRANSDUCTIVE_FAMILIES, where one call is the
# definition of the score rather than a convenience.)
#
# Owned here because TWO places need the same answer and drifted apart when only
# one of them knew it: `model_selection_utils` sizes the scoring batch, and
# `TrainModels._diagnostic_batch_size` sizes the post-fit plotting loop. The
# second knew about the transductive and TSB-AD families but not about LSTMAD,
# whose plot loop therefore ran at batch_size 8 against a 50-150 step window and
# raised "negative dimensions are not allowed" — before `logging_obj.save`, so
# LSTMAD could not be trained on any entity at all.
WHOLE_SERIES_FAMILIES: FrozenSet[str] = (
    frozenset({"LSTMAD"}) | (TSBAD_FAMILIES - frozenset({"POLY"})))

# Families that cannot see more than one channel. POLY fits `np.polyfit` to the
# raw series and raises "Polynomial must be 1d only" on anything wider, which is
# what Table I's `U` marking records. It is therefore usable on UCR and
# genuinely unavailable on SKAB (9 channels) and SMD (38): selecting it there
# fails with an explanation naming the detector, rather than a numpy error from
# four frames down. Declared here, where the vocabulary lives, so the web UI can
# say so before a run starts instead of after.
UNIVARIATE_FAMILIES: FrozenSet[str] = frozenset({"POLY"})

# Families whose `decision_function` scores each row against the OTHER ROWS OF
# THE SAME CALL rather than against what `fit` saw. COF's scoring is literally
# `distance_matrix(X, X)` and reads none of the state `fit` stored; measured,
# all three score a row identically no matter what they were fitted on
# (fit-on-X versus fit-on-unrelated-data: max difference 0.000000).
#
# That has one hard consequence: a row's score depends on which other rows share
# its call, so batching would make it depend on where `eval_batch_size` happened
# to cut. The same window scored 1.003744 and 0.966958 under COF in two
# different batches. `Utils/model_selection_utils` therefore hands these
# families the WHOLE series in one call, which makes the score a deterministic
# function of (entity, row) — and `Utils/test_pipeline_spec` asserts both that
# they are routed that way and that they are still deterministic, since the two
# models excluded for irreproducibility would otherwise look identical to these.
#
# Two limits that follow from the estimators, not from this pipeline: COF raises
# IndexError on any call holding fewer rows than its `n_neighbors` (20), and
# SpectralResidual needs at least `score_window` (3) rows and returns three
# scores for a single row. Both are fine on a whole-series call; in Thompson,
# whose windows are `n_timesteps * 0.8 / iterations`, COF needs an entity of
# ~1,250 timesteps before its windows clear 20 rows.
# POLY joins them for a different mechanism with the same consequence. Its
# `decision_function` needs a `measure` object the class never stores and scores
# against the `estimation`/`n_train_` its last fit left behind, so it cannot
# score data it was not just fitted on; the adapter therefore refits on whatever
# it is handed. Measured: `fit(B)` then `fit(A)` gives byte-identical scores to
# `fit(A)` alone (0.000e+00), i.e. fit state is fully replaced. So the score is
# a function of the call's rows alone, which is the property this set names.
TRANSDUCTIVE_FAMILIES: FrozenSet[str] = frozenset({"COF", "SOS", "SR", "POLY"})

# The paper's Table I taxonomy: "Base models grouped by family: Neural Networks
# (NN), Statistical (Stat) or Foundation Models (FM)". Keys are the paper's
# short labels and are the identifier everything else keys off — the API value,
# the CSS class suffix — so they stay short and slug-safe. GROUP_LABELS below
# carries what a reader sees; the run page builds one select-all button per
# entry here, so a group added here appears there with no UI change.
#
# Two mappings are worth stating because the names invite the opposite guess:
#
#   * Our NN family is k-Nearest Neighbors, and the paper puts it in Stat as
#     "(Sub)-KNN — kNN distance score". The group also called NN is Neural
#     Networks. So the Neural Networks button does NOT select the NN detector.
#     The collision is the paper's; the UI spells the group out in full for
#     exactly this reason.
#   * MD is an nn.Module trained by SGD, but what it learns is one mean per
#     channel, which is the same kind of quantity the paper files under Stat as
#     RM ("simple moving-average residuals"). Grouped by what it computes, not
#     by what it is implemented with.
#
# FM is empty: the paper's foundation models (OFA, Lag-Llama, Chronos, TimesFM,
# MOMENT) are none of them in this pool. It is listed anyway so the taxonomy is
# visible and a future FM detector has an obvious home.
DETECTOR_GROUPS: Dict[str, tuple] = {
    "NN": ("AE", "RNN", "LSTMVAE", "DGHL", "LSTMAD", "DONUT",
           "OA", "USAD", "TRANAD", "FITS", "TIMESNET"),
    "Stat": ("LOF", "NN", "CBLOF", "ABOD", "KDE", "IFOREST", "HBOS", "PCA",
             "OCSVM", "MCD", "COF", "SOS", "SR", "RM", "MD", "KMEANSAD",
             "POLY"),
    "FM": ("OFA", "TIMESFM", "CHRONOS"),
}


# What a reader sees. Spelled out rather than abbreviated because "NN" next to
# a detector family also called NN is the misreading this taxonomy most invites.
GROUP_LABELS: Dict[str, str] = {
    "NN": "Neural Networks",
    "Stat": "Statistical",
    "FM": "Foundation",
}


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

# Iteration number the explainability artifacts are written under. Deliberately
# distinct from the CLI --iteration (which sizes the online windows), so IR/NL
# filenames stay stable across online configurations.
OFFLINE_ITERATION = 0

# Minimum detectors a run can be meaningful with: GA fitness, Markov rank
# aggregation and the off-by pairwise surrogates are all vacuous with one.
MIN_DETECTORS = 2

# The narrator. Owned here rather than in Explainability/llm.py because the web
# UI reports which model produced a set of explanations, and a second copy of
# the string would eventually disagree with the one that actually ran — telling
# the reader their narratives came from a model that never saw them.
DEFAULT_LLM_MODEL = "qwen2.5:14b-instruct"
DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"


# How a dataset is SHOWN. The CLI, the directory names and every path keep the
# real key; this is presentation only.
#
# Owned here rather than in `WebUI/catalog.py` because two sides need the same
# answer and disagreed: the pipeline writes "Dataset: skab" into the
# comprehensive report from the raw `--dataset` argument, while the web UI
# showed "SKAB" from the directory name — the same run named two ways on two
# pages. `anomaly_archive` is why this is a table and not `.upper()`.
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
