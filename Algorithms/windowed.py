"""
Shared window handling for the PyOD-backed detectors.

Every one of these detectors (LOF, CBLOF, ABOD, COF, KDE, SOS, ALAD and the
generic PyodModel) used to fit on

    Y_windows.reshape(n_batches * n_features * n_time, -1).reshape(-1, 1)

which turns a window into one sample PER NUMBER, with a single feature: its own
value. On SKAB a window is 9 channels x 64 timesteps, so 576 readings became 576
unrelated one-dimensional samples. Time was gone, channel identity was gone, and
the window the loader had just cut was thrown away — a detector could only ever
answer "is this value rare in the pooled distribution of all values", which
reaches point anomalies and nothing else. Contextual and collective anomalies,
two of the three types the framework is built to find, were invisible by
construction, and every detector fitted this way was estimating the same 1-D
marginal, so they agreed with each other far more than their names suggest.

The fix is the shape `nearest_neighbors.py` already used: one sample per window,
with the flattened window as its feature vector. That is the subsequence form the
literature means by "(Sub)-LOF" and "(Sub)-KNN".

One consequence is unavoidable and is the reason this lives in one place. Fitted
on windows, a detector produces one score per WINDOW, not one per reading, so
`window_anomaly_score` broadcasts that score across the window — again as
nearest_neighbors.py does. The old per-element score was
`(mask * (Y - Y * s))**2`, an outlier score pushed through a fabricated
reconstruction; the score here is the detector's own `decision_function` output.

Checkpoints fitted the old way carry an estimator expecting one feature and will
raise when handed a window. That is deliberate: a silent shape coercion would
have kept the old behaviour alive behind the new code. Retrain.
"""

import numpy as np
import torch as t
from loguru import logger


def windows_as_rows(Y_windows):
    """(n_windows, n_features, window_size) -> (n_windows, n_features*window_size).

    Accepts a torch tensor or a numpy array and always returns float64 numpy on
    the CPU, which is what every PyOD estimator wants. The tensors arrive
    float32, which underflows SOS to all-non-finite and CBLOF's Cython path
    rejects outright.
    """
    if isinstance(Y_windows, t.Tensor):
        Y_windows = Y_windows.detach().cpu().numpy()
    Y_windows = np.asarray(Y_windows, dtype=np.float64)
    return Y_windows.reshape(len(Y_windows), -1)


# Attributes a deep PyOD detector keeps for TRAINING only, dropped once `fit`
# returns. The same list, for the same reason, as
# `Algorithms.tsbad_model._TRAINING_ONLY_ATTRS` — the two adapters meet the same
# problem from opposite directions and should be recognisable as one fix.
_TRAINING_ONLY_ATTRS = ("model_optim", "optimizer", "scheduler")


def fit_windows(model, train_dataloader):
    """Fit `model` on one row per window."""
    rows = windows_as_rows(train_dataloader.Y_windows)
    _check_enough_windows(model, len(rows))
    model.fit(X=rows)
    _release_training_state(model)


def _release_training_state(model) -> None:
    """Drop optimiser/scheduler state so the fitted detector can be pickled.

    Not an optimisation — a checkpoint cannot be written otherwise. PyOD's deep
    base class leaves `self.optimizer` on the detector after `fit`, and a torch
    optimiser reaches a `torch._dynamo` config module through its reference
    graph: `TypeError: cannot pickle 'ConfigModuleInstance' object`, raised from
    `logging_obj.save` AFTER training has already finished, so the work is done
    and then thrown away.

    Verified on AutoEncoder, the pool's only such detector: scores before and
    after the
    drop are bit-identical (max |diff| 0.0), the checkpoint then writes, and a
    reloaded model scores the same again. Nothing here is read by
    `decision_function`; PyOD rebuilds the optimiser in `training_prepare` on
    any later `fit`, so refitting a reloaded detector still works.
    """
    for attr in _TRAINING_ONLY_ATTRS:
        if getattr(model, attr, None) is not None:
            setattr(model, attr, None)


def _check_enough_windows(model, n_rows: int) -> None:
    """A deep PyOD detector must get at least `batch_size` windows.

    `pyod.models.base_dl.BaseDeepLearningDetector.fit` builds its loader with
    `drop_last=True`, so a call holding fewer windows than `batch_size` yields
    ZERO batches: the training loop body never runs, `loss` is never assigned,
    and the next line reads it — `UnboundLocalError: cannot access local
    variable 'loss'`, four frames inside PyOD, naming neither the detector nor
    the requirement, after silently training on nothing.

    AutoEncoder is the pool's only such detector, and the window COUNT is set
    by the
    grid's `window_step`: at the 64/64 its neighbours use, a 917-step SKAB
    entity produced 14 windows. Same role as the minimum-length notes
    `score_windows` carries for COF and SpectralResidual, and as
    `_TSBADEstimator._check_length` — say which detector, what it got, and what
    it needed, at the call that can still name them.
    """
    batch_size = getattr(model, "batch_size", None)
    if not isinstance(batch_size, int) or batch_size <= 0:
        return                      # not a batching detector — nothing to check
    if n_rows >= batch_size:
        return
    name = (getattr(model, "detector_name", None)
            or type(model).__name__)
    raise ValueError(
        f"{name} was fitted on {n_rows} window(s) but its batch_size is "
        f"{batch_size}, and PyOD drops the last partial batch — so it would "
        f"train on nothing and fail inside PyOD with an unbound 'loss'. The "
        f"window count comes from the grid: lower `window_step` (or "
        f"`window_size`) for this family, or raise the entity length.")


def score_windows(model, Y, clip=None):
    """One anomaly score per window, as a (n_windows,) float array.

    A stray non-finite score is replaced rather than propagated: one NaN would
    poison the whole de-unfolded series, and downstream a 0 reads as "no
    evidence" rather than "no answer". `clip` caps the score when a detector
    reports an unbounded distance; None leaves the detector's own scale alone.

    When EVERY score is non-finite that substitution stops being repair and
    starts being concealment — it hands back a detector that scores everything
    identically, which no metric can distinguish from a detector that simply
    found nothing. PyOD's PCA does this on SMD/machine-1-6: five of its
    thirty-eight channels have zero variance, PCA divides by their eigenvalues,
    and every score comes back +inf. So that case is logged, loudly, with the
    detector named.
    """
    rows = windows_as_rows(Y)
    try:
        scores = np.asarray(model.decision_function(X=rows), dtype=float)
    except ValueError as e:
        # Estimators fitted on float32 keep float32 state; sklearn's Cython
        # kernels then reject the float64 rows (CBLOF's KMeans centres).
        if "dtype" not in str(e):
            raise
        scores = np.asarray(model.decision_function(X=rows.astype(np.float32)),
                            dtype=float)
    # The TSB-AD adapter is one wrapper around many detectors, so its class name
    # would say `_TSBADEstimator` for all of them; it sets `detector_name` to the
    # family instead. PyOD estimators have no such attribute and keep their own.
    name = getattr(model, "detector_name", type(model).__name__)
    # One score per window, or the caller is about to reshape something that is
    # not what it thinks. PyOD 3's SpectralResidual returns THREE scores for a
    # single row — `np.convolve(..., mode='same')` yields `max(len(A),
    # score_window)` — and broadcasts silently rather than raising. Caught here,
    # named, instead of surfacing three frames later as a reshape error.
    if scores.shape != (len(rows),):
        raise ValueError(
            f"{name} returned {scores.shape} scores for {len(rows)} window(s); "
            f"expected exactly one score per window. Some PyOD estimators have a "
            f"minimum input length (SpectralResidual needs score_window rows, COF "
            f"needs more than n_neighbors) and misbehave below it.")
    finite = np.isfinite(scores)
    if scores.size and not finite.any():
        logger.warning(
            f"{name} scored every one of {scores.size} windows non-finite; the "
            f"scores below are all zero and this detector cannot separate "
            f"anything on this entity. A constant column (zero variance) in the "
            f"input is the usual cause.")
    elif not finite.all():
        logger.warning(f"{name} produced {int((~finite).sum())} non-finite "
                       f"score(s) of {scores.size}; substituting 0.")
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.abs(scores)
    if clip is not None:
        scores = np.minimum(scores, clip)
    return scores


def broadcast_to_window(scores, n_batches, n_features, n_time):
    """A per-window score, spread over that window's readings.

    The detector saw the window as one object, so it has nothing finer to say
    about which reading inside it was anomalous. Repeating the score is honest
    about that; `final_anomaly_score` then de-unfolds overlapping windows and a
    reading covered by several windows still receives their average.
    """
    scores = np.asarray(scores, dtype=float).reshape(n_batches, 1, 1)
    return t.from_numpy(np.broadcast_to(scores, (n_batches, n_features, n_time)).copy())
