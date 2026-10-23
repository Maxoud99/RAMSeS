"""
Scoring the deployed selection on online windows, and building those windows.

Kept apart from `online_phase.py` because it needs the model stack (torch,
sklearn, the metric code) while the runner and the trigger policy do not, and
their tests should not have to import it.

The online stream
-----------------
The synthetic anomalies are injected ONCE into the whole online part
(`build_online_stream`, seeded, with numpy's global generator restored), and
every window is cut from that one injected stream. Windows overlap by 95%, so
injecting per window gave one sample a different value and label in each of
the twenty windows that contain it; now a sample has one value and one label
everywhere. A re-optimisation sees the same injected samples
(`sliding_series`).

Anomaly types whose injection produces several copies of the series (speedup,
scale and wander give 4, cutoff and average 2) are injected once per copy, as
the offline phase does, and each window holds the SAME time slice of every
copy, concatenated in copy order. The number of windows is therefore the
plain online part's. `speedup` resamples its anomalous segment, so its copies
are longer or shorter than the series; a window then takes, from each copy,
the samples that cover the same original time slice
(`InjectedCopy.position`), and the window's `segments` say where each copy's
part lies.

Real labels (--online_labels real)
----------------------------------
When the test series carries real labels, NOTHING is injected online: the
stream is the online part itself, as one identity "copy" (`real_label_stream`),
so every window is the raw test slice with its true labels, and the code that
cuts, fingerprints and scores windows is the same as for injected copies. A
re-optimisation then selects on the real labels seen so far: the offline
part's raw samples (`identity_copy`) slid forward by the online samples seen,
each once (`sliding_series`), with no injection anywhere — a supervised
re-selection, which assumes the labels are fed back as the stream goes.
`require_labelled_anomaly` refuses one whose data holds no labelled anomaly,
since every selection stage would then have nothing to select on.

A window whose true labels are one class (most windows of most series: an
anomaly is rare) has no score; it is drawn as a gap, cannot trigger, and does
not enter the adaptive history (online_policy). `OnlineWindow.scorable` says
which windows these are, from the labels alone.

How a window is scored
----------------------
* single model — range-based best F1 and PR-AUC from
  `range_based_precision_recall_f1_auc` (what `evaluate_individual_models`
  uses offline), plus VUS when the decision metric includes it;
* ensemble — the GA's fitness (`Metrics.Ensemble_GA.score_meta_model`): the
  members' scores on the window go through the selection's meta-model, which
  is the very object the GA fitted when it chose the ensemble.

Both are combined with the run's decision metric. VUS uses ONE window length
for every online window (the offline phase's, computed once on the injected
offline series), so consecutive windows and their baseline are measured on one
scale. A window whose labels are all one class has no defined F1 or PR-AUC;
both signals are then None and the trigger policy skips the window.

A window score is not on the offline scores' scale, and is never compared with
them: the range-based metrics' tolerance grows with the series (about a tenth
of it), so the offline series of ~1000 samples is judged with a tolerance near
100 and a 100-sample window with 10. Each signal is compared only with its own
online baseline.

Same window, same score, in every scenario
------------------------------------------
A detector's scores on a window depend only on the detector and the window:
the models are frozen and the windows are generated once. `WindowScorer`
therefore scores each (window, detector) pair ONCE per run and every scenario
reads that result. Without this, LSTMVAE and DGHL, which draw from torch's
global generator whenever they score, gave the same window different scores in
different scenarios, and a trigger could differ by RNG noise rather than by
policy. The first scoring runs under a seed derived from the run's seed and
the window index, with every global generator put back afterwards, so the
stream is reproducible from run to run. Re-optimisations run in their own
processes and cannot draw from these generators.

The stream's own models
-----------------------
Scoring is not free of side effects on the model object (AnomalyTransformer
keeps its attention maps on its layers, LSTMVAE switches itself to train mode).
`isolated_models` gives the stream its own copies, so the pipeline's
`trained_models` stay exactly as the offline phase left them. Between
scenarios the memo above already keeps state from carrying over: a detector
scores each window once per run.
"""

import contextlib
import copy
import hashlib
import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from Datasets.dataset import Dataset, Entity
from Metrics.Ensemble_GA import (evaluate_model_consistently, fit_meta_model,
                                 inject_synthetic_anomalies, score_meta_model)
from Metrics.metrics import range_based_precision_recall_f1_auc, vus_score
from Model_Selection.inject_anomalies import InjectedCopy, inject_copies
from Model_Selection.online_phase import OnlineWindow, Selection, WindowScores
from Utils.pipeline_spec import combine_metrics, metrics_required

DEFAULT_META_MODEL = "rf"

logger = logging.getLogger(__name__)


# ── Randomness and model isolation ───────────────────────────────────────────

def _torch():
    try:
        import torch
        return torch
    except ImportError:                                  # pragma: no cover
        return None


def seed_all(seed: int) -> None:
    """Seed every generator a detector or the selection pipeline draws from:
    Python's `random`, numpy's global generator and torch's."""
    seed = int(seed) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch = _torch()
    if torch is not None:
        torch.manual_seed(seed)


@contextlib.contextmanager
def seeded_rngs(seed: int):
    """Run the body under `seed_all(seed)`, then put the generators back.

    Restoring keeps the stream's scoring from moving the rest of the run's
    random sequence, as `build_online_windows` does for the injection.
    """
    torch = _torch()
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state() if torch is not None else None
    try:
        seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch is not None:
            torch.set_rng_state(torch_state)


def isolated_models(trained_models: Dict[str, Any]) -> Dict[str, Any]:
    """Private copies of the detectors, for the stream.

    A deep copy, because scoring writes state into a model's sub-modules as
    well as onto the model itself. A detector that cannot be deep-copied gets
    a shallow copy, which still owns its own attributes, and a warning.
    Memory grows by roughly the size of the loaded models while the online
    phase runs.
    """
    copies = {}
    for name, model in trained_models.items():
        try:
            copies[name] = copy.deepcopy(model)
        except Exception as exc:
            logger.warning(f"Could not deep-copy {name} for the online stream "
                           f"({type(exc).__name__}: {exc}); it gets a shallow copy, "
                           f"which shares its sub-modules with the pipeline's original.")
            copies[name] = copy.copy(model)
    return copies


# ── Windows ──────────────────────────────────────────────────────────────────

def window_dataset(Y, labels, mask, name: str = "online") -> Dataset:
    """A one-entity Dataset over copies of the given arrays.

    Built fresh rather than by mutating a shared Dataset, which is what let the
    old loop's background task deep-copy a Dataset the stream was rewriting.
    """
    Y = np.array(Y, copy=True)
    labels = np.array(labels, copy=True).reshape(1, -1)
    mask = np.array(mask, copy=True) if mask is not None else None
    if mask is not None and mask.shape != Y.shape:
        mask = None                                    # Entity defaults to all ones
    entity = Entity(Y=Y, name=name, labels=labels, mask=mask, verbose=False)
    return Dataset(entities=[entity], name=name, verbose=False)


class OnlineStream:
    """The online part as every scenario replays it, injected ONCE.

    `raw_*` is the online part before the synthetic anomalies (after the
    regime shift of --inject_online_regime, when given); `copies` holds one
    injected copy per anomaly parameter combination.
    """

    def __init__(self, raw_Y, raw_labels, raw_mask, copies: Sequence[InjectedCopy]):
        self.raw_Y = np.asarray(raw_Y)
        self.raw_labels = np.asarray(raw_labels).reshape(-1)
        self.raw_mask = (np.asarray(raw_mask) if raw_mask is not None
                         else np.ones_like(self.raw_Y))
        self.copies = list(copies)

    @property
    def length(self) -> int:
        """Samples in the stream, in the original time positions windows use."""
        return int(self.raw_Y.shape[1])

    def copy_params(self) -> List[Dict[str, Any]]:
        return [{"copy": i, "anomaly_type": c.anomaly_type, **c.params,
                 "length": c.length} for i, c in enumerate(self.copies)]

    def cut(self, start: int, end: int):
        """(Y, labels, segments) of original positions [start, end) from every
        copy, concatenated in copy order."""
        parts_Y, parts_labels, segments, at = [], [], [], 0
        for i, c in enumerate(self.copies):
            lo, hi = c.position(start), c.position(end)
            parts_Y.append(c.Y[:, lo:hi])
            parts_labels.append(np.asarray(c.labels).reshape(-1)[lo:hi])
            segments.append({"copy": i, "data_start": at, "data_end": at + hi - lo,
                             "copy_start": lo, "copy_end": hi})
            at += hi - lo
        return (np.concatenate(parts_Y, axis=1), np.concatenate(parts_labels),
                segments)


def identity_copy(Y, labels) -> InjectedCopy:
    """`Y` with its own `labels` as one uninjected copy: what a stream or a
    sliding series is made of when nothing is injected (real-label mode)."""
    Y = np.array(Y, copy=True)
    labels = np.asarray(labels, dtype=float).reshape(-1).copy()
    if labels.size != Y.shape[1]:
        raise ValueError(f"{labels.size} labels for {Y.shape[1]} samples")
    return InjectedCopy(Y, labels, np.zeros(labels.size), None, {}, None, Y.shape[1])


def real_label_stream(raw_Y, raw_labels, raw_mask) -> OnlineStream:
    """The online part with its REAL labels and nothing injected.

    One identity copy, so a window is exactly the raw test slice and its true
    labels, cut and scored by the same code as an injected stream.
    """
    return OnlineStream(raw_Y, raw_labels, raw_mask, [identity_copy(raw_Y, raw_labels)])


def require_labelled_anomaly(dataset: Dataset, what: str = "the data") -> None:
    """Raise ValueError unless `dataset`'s labels hold both classes.

    A real-label re-optimisation selects on true labels only. Without a
    labelled anomaly the GA would fall back to labelling random points
    anomalous (fitness_function), the off-by-threshold and Monte Carlo stages
    skip themselves, and Thompson Sampling learns nothing: no stage has
    anything to select on, so the re-optimisation is refused up front and
    recorded as failed, and the current selection stays.
    """
    labels = np.asarray(dataset.entities[0].labels).reshape(-1)
    n = int(np.sum(labels > 0))
    if n == 0:
        raise ValueError(f"no labelled anomaly in {what}: a real-label "
                         f"re-optimisation has nothing to select on")
    if n == labels.size:
        raise ValueError(f"every sample of {what} is labelled anomalous: a "
                         f"real-label re-optimisation has nothing to select on")


def build_online_stream(raw_Y, raw_labels, raw_mask, anomaly_list: Sequence[str],
                        anomaly_rate: Optional[float], seed: int) -> OnlineStream:
    """Inject the synthetic anomalies into the whole online part, once.

    `inject_copies` draws from numpy's global generator (and reseeds it), so
    the generator is seeded here for a reproducible stream and restored
    afterwards, leaving the rest of the run's randomness as it was.
    """
    state = np.random.get_state()
    np.random.seed(int(seed) % (2 ** 32))
    try:
        copies = inject_copies(np.array(raw_Y, copy=True), list(anomaly_list),
                               rate=anomaly_rate)
    finally:
        np.random.set_state(state)
    return OnlineStream(raw_Y, raw_labels, raw_mask, copies)


def build_online_windows(stream: OnlineStream, indices: Sequence[int], offset: int,
                         step_size: int, window_size: int,
                         name: str = "online") -> List[OnlineWindow]:
    """The online windows, cut from the injected stream.

    Window i covers stream positions [i * step, i * step + window_size);
    `start`/`end` add `offset` (the offline part's length), so without a
    regime shift they are positions in the test series.
    """
    windows = []
    for i in indices:
        lo = int(i) * int(step_size)
        hi = lo + int(window_size)
        if hi > stream.length:
            raise ValueError(f"window {i} ends at {hi}, past the {stream.length}-sample "
                             f"online stream")
        Y, labels, segments = stream.cut(lo, hi)
        n_anomalous = int(np.sum(np.asarray(labels) > 0))
        windows.append(OnlineWindow(index=int(i), start=int(offset) + lo,
                                    end=int(offset) + hi, online_start=lo, online_end=hi,
                                    segments=segments,
                                    data=window_dataset(Y, labels, None, name=name),
                                    digest=window_digest(Y, labels),
                                    n_samples=int(np.asarray(labels).size),
                                    n_anomalous=n_anomalous))
    return windows


def window_digest(Y, labels) -> str:
    """sha256 (first 16 hex digits) of a window's data and labels."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(Y, dtype=np.float64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(labels, dtype=np.int64).reshape(-1)).tobytes())
    return h.hexdigest()[:16]


def sliding_series(offline_copies: Sequence[InjectedCopy], offline_raw: Tuple,
                   stream: OnlineStream, seen_end: int, name: str = "sliding"):
    """What a re-optimisation selects on: the offline data slid forward by
    the online samples seen so far, each sample once.

    Returns (injected, raw) one-entity Datasets. `injected` holds, per copy,
    the offline phase's injected copy followed by the stream's injected copy
    up to `seen_end`, cut to that offline copy's length, concatenated in copy
    order: what the offline GA and Thompson selected on, moved forward in time.
    `raw` is the same window before injection (offline raw part plus raw
    stream), which the robustness stages score, as they do offline.
    `offline_raw` is (Y, labels, mask) of the un-injected offline part.
    """
    if len(offline_copies) != len(stream.copies):
        raise ValueError(f"{len(offline_copies)} offline copies but "
                         f"{len(stream.copies)} online copies")
    parts_Y, parts_labels = [], []
    for off, on in zip(offline_copies, stream.copies):
        seen = on.position(seen_end)
        n = off.length
        parts_Y.append(np.concatenate([off.Y, on.Y[:, :seen]], axis=1)[:, -n:])
        parts_labels.append(np.concatenate([
            np.asarray(off.labels).reshape(-1),
            np.asarray(on.labels).reshape(-1)[:seen]])[-n:])
    injected = window_dataset(np.concatenate(parts_Y, axis=1),
                              np.concatenate(parts_labels), None, name=name)

    raw_Y, raw_labels, raw_mask = offline_raw
    n = int(np.asarray(raw_Y).shape[1])
    raw = window_dataset(
        np.concatenate([raw_Y, stream.raw_Y[:, :seen_end]], axis=1)[:, -n:],
        np.concatenate([np.asarray(raw_labels).reshape(-1),
                        stream.raw_labels[:seen_end]])[-n:],
        np.concatenate([raw_mask, stream.raw_mask[:, :seen_end]], axis=1)[:, -n:],
        name=f"{name}_raw")
    return injected, raw


# ── Selections ───────────────────────────────────────────────────────────────

def _member_columns(members, header, train_predictions, train_data, trained_models):
    """The training-set score columns of `members`, in `members` order.

    The GA's training matrix has one column per detector of its pool, in the
    order of `individual_predictions` (the pool after unscorable detectors were
    dropped). When that alignment cannot be trusted — a detector that failed on
    the training data is skipped without leaving a column — the members are
    scored on the training data again instead of guessing.
    """
    if (train_predictions is not None and header is not None
            and np.ndim(train_predictions) == 2
            and np.shape(train_predictions)[1] == len(header)
            and all(m in header for m in members)):
        return np.asarray(train_predictions)[:, [header.index(m) for m in members]], None
    if train_data is None or trained_models is None:
        raise ValueError("the selection's training predictions do not line up with "
                         "its pool, and no training data was given to rescore them")
    y_true, matrix, _, by_name = evaluate_model_consistently(
        copy.deepcopy(train_data), trained_models, list(members), is_ensemble=True)
    missing = [m for m in members if m not in by_name]
    if missing:
        raise ValueError(f"could not score {', '.join(missing)} on the training data")
    return np.column_stack([by_name[m] for m in members]), y_true


def _pool_order(members, header):
    """`members` in the order of the GA's pool (`header`), which is the column
    order of the GA's meta-model; sorted when the pool is unknown."""
    members = list(dict.fromkeys(members or []))
    if header:
        known = [m for m in header if m in members]
        return known + sorted(m for m in members if m not in known)
    return sorted(members)


def _takes(meta_model, n_features: int) -> bool:
    expected = getattr(meta_model, "n_features_in_", None)
    return expected is None or int(expected) == int(n_features)


def fit_selection(ensemble: Optional[Sequence[str]], single: Optional[str],
                  meta_model_type: Optional[str],
                  individual_predictions: Optional[Dict[str, Any]],
                  base_model_predictions_train, y_true_train,
                  origin: str = "offline", origin_window: Optional[int] = None,
                  train_data=None, trained_models=None, meta_model=None,
                  seed: int = 0) -> Selection:
    """A deployable Selection.

    `meta_model` is the GA's own fitted meta-model of this ensemble
    (genetic_algorithm(return_meta_model=True)); deployed as is, the online
    ensemble is the very object the GA chose. Only when it is missing (an
    ensemble the GA did not fit, such as the random baseline) or does not take
    this many members is one fitted here, on the GA's training predictions and
    labels, under `seed` — the random forest draws from numpy's global
    generator, so an unseeded refit would differ from run to run.
    """
    header = list(individual_predictions.keys()) if individual_predictions else None
    members = _pool_order(ensemble, header)
    meta_type = meta_model_type or DEFAULT_META_MODEL
    source = None
    if members and meta_model is not None and _takes(meta_model, len(members)):
        source = "ga"
    elif members:
        X, y_rescored = _member_columns(members, header, base_model_predictions_train,
                                        train_data, trained_models)
        y = np.array(y_rescored if y_rescored is not None else y_true_train, copy=True).flatten()
        with seeded_rngs(seed):
            if len(np.unique(y)) < 2:
                # The GA's own rule for a one-class training set (fitness_function).
                y = inject_synthetic_anomalies(y)
            meta_model = fit_meta_model(X, y, meta_type)
        source = "refit"
    else:
        meta_model = None
    return Selection(ensemble=members, single=single, meta_model_type=meta_type,
                     meta_model=meta_model, origin=origin, origin_window=origin_window,
                     meta_model_source=source)


def random_selection(pool: Sequence[str], seed: int, meta_model_type: Optional[str],
                     individual_predictions, base_model_predictions_train,
                     y_true_train, train_data=None, trained_models=None) -> Selection:
    """The fixed-random baseline: a random single model and a random ensemble of
    3 to 6 detectors (fewer when the pool is smaller), drawn from the GA's pool
    with their own generator so the draw is reproducible."""
    pool = list(pool)
    rng = random.Random(seed)
    single = rng.choice(pool)
    low, high = min(3, len(pool)), min(6, len(pool))
    ensemble = rng.sample(pool, rng.randint(low, high))
    return fit_selection(ensemble, single, meta_model_type, individual_predictions,
                         base_model_predictions_train, y_true_train, origin="random",
                         train_data=train_data, trained_models=trained_models, seed=seed)


# ── Scoring ──────────────────────────────────────────────────────────────────

class WindowScorer:
    """Scores a Selection on an OnlineWindow with the run's decision metric.

    One instance serves every scenario of a run: it remembers each detector's
    scores per window (see the module docstring), so scenarios that deploy the
    same detector on the same window read identical numbers. Called from the
    stream's thread only, which is why the memo has no lock.
    """

    def __init__(self, trained_models: Dict[str, Any], decision_metric, seed: int = 0,
                 vus_win: Optional[int] = None):
        self.trained_models = trained_models
        self.metric = decision_metric
        self.needed = metrics_required(decision_metric)
        self.seed = int(seed)
        # One VUS window for every online window, as the offline phase uses
        # one for both candidates: a window length found per window jumped
        # between ~35 and its 125 fallback on 95%-overlapping windows, moving
        # VUS for reasons other than the detectors.
        self.vus_win = int(vus_win) if vus_win else None
        # (window index, detector) -> ("ok", y_true, y_scores) | ("error", exc)
        self._memo: Dict[Tuple[int, str], tuple] = {}
        self.computed = 0          # detector scorings actually run
        self.reused = 0            # answered from the memo

    def _detector_scores(self, window: OnlineWindow, name: str):
        """(y_true, y_scores) of one detector on the window, scored once per run.

        A failure is remembered too, so a detector that cannot score a window
        fails it the same way in every scenario instead of being retried.
        """
        key = (int(window.index), name)
        entry = self._memo.get(key)
        if entry is None:
            model = self.trained_models.get(name)
            if model is None:
                raise KeyError(f"{name} is not a loaded model")
            self.computed += 1
            try:
                # A copy per detector: the window is replayed by every scenario,
                # so nothing a scorer does to it may leak into the next replay.
                with seeded_rngs(self.seed + int(window.index)):
                    y_true, y_scores, _, _ = evaluate_model_consistently(
                        copy.deepcopy(window.data), model, name)
                entry = ("ok", np.asarray(y_true).flatten().astype(int),
                         np.asarray(y_scores, dtype=float).flatten())
            except Exception as exc:
                entry = ("error", exc)
            self._memo[key] = entry
        else:
            self.reused += 1
        if entry[0] == "error":
            raise entry[1].with_traceback(None)
        # Copies: a caller must not be able to change what the next scenario reads.
        return entry[1].copy(), entry[2].copy()

    def _combine(self, f1, pr_auc, vus):
        value = combine_metrics(self.metric, {"f1": f1, "pr_auc": pr_auc, "vus": vus})
        return None if value is None or not np.isfinite(value) else float(value)

    def __call__(self, window: OnlineWindow, selection: Selection) -> WindowScores:
        out = WindowScores()
        vus_win = self.vus_win if "vus" in self.needed else None

        # Single model: the offline phase's evaluate_individual_models metric.
        if selection.single:
            try:
                y_true, y_scores = self._detector_scores(window, selection.single)
                # Kept for the live monitor whether or not the window can be
                # scored: the scores exist, only F1 and PR-AUC do not.
                out.trace["single"] = {"scores": y_scores, "threshold": None, "flagged": None}
                if len(np.unique(y_true)) < 2:
                    raise ValueError("the window's labels are all one class")
                _, _, f1, pr_auc, y_pred, cut = range_based_precision_recall_f1_auc(
                    y_true, y_scores, return_threshold=True)
                out.trace["single"].update(
                    threshold=cut, flagged=(y_scores >= cut).astype(np.int8))
                vus = (vus_score(y_scores, y_true, vus_win)
                       if vus_win is not None else float("nan"))
                out.single = self._combine(f1, pr_auc, vus)
                out.metrics["single"] = {"f1": float(f1), "pr_auc": float(pr_auc),
                                         "vus": float(vus)}
                out.misclassified["single"] = int(np.sum(
                    np.asarray(y_pred).astype(int) != y_true))
            except Exception as exc:
                out.errors["single"] = f"{type(exc).__name__}: {exc}"
        else:
            out.errors["single"] = "no single model is deployed"

        # Ensemble: the GA's fitness through the selection's own meta-model.
        if selection.ensemble and selection.meta_model is not None:
            try:
                columns, y_true = [], None
                for name in selection.ensemble:
                    member_true, member_scores = self._detector_scores(window, name)
                    if y_true is None:
                        y_true = member_true
                    if len(member_scores) != len(y_true):
                        raise ValueError(f"{name} scored {len(member_scores)} samples, "
                                         f"expected {len(y_true)}")
                    columns.append(member_scores)
                matrix = np.column_stack(columns)
                if len(np.unique(y_true)) < 2:
                    # The meta-model's probabilities, for the live monitor,
                    # cleaned as score_meta_model cleans its input.
                    clean = np.nan_to_num(np.asarray(matrix, dtype=np.float32),
                                          nan=0.0, posinf=1.0, neginf=0.0)
                    try:
                        out.trace["ensemble"] = {
                            "scores": selection.meta_model.predict_proba(clean)[:, 1],
                            "threshold": None, "flagged": None}
                    except Exception:
                        pass        # the window's error below is the one to report
                    raise ValueError("the window's labels are all one class")
                scored = score_meta_model(selection.meta_model, matrix,
                                          y_true, metric=self.metric, vus_win=vus_win)
                out.trace["ensemble"] = {
                    "scores": np.asarray(scored["y_scores"], dtype=float),
                    "threshold": scored.get("threshold"),
                    "flagged": np.asarray(scored["y_pred"]).astype(np.int8)}
                fitness = scored["fitness"]
                out.ensemble = float(fitness) if np.isfinite(fitness) else None
                out.metrics["ensemble"] = {"f1": float(scored["f1"]),
                                           "pr_auc": float(scored["pr_auc"]),
                                           "vus": float(scored["vus"])}
                out.misclassified["ensemble"] = int(np.sum(
                    np.asarray(scored["y_pred"]).astype(int) != y_true))
            except Exception as exc:
                out.errors["ensemble"] = f"{type(exc).__name__}: {exc}"
        else:
            out.errors["ensemble"] = "no ensemble is deployed"
        return out
