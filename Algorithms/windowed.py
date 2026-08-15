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

    Accepts a torch tensor or a numpy array and always returns numpy on the CPU,
    which is what every PyOD estimator wants.
    """
    if isinstance(Y_windows, t.Tensor):
        Y_windows = Y_windows.detach().cpu().numpy()
    Y_windows = np.asarray(Y_windows)
    return Y_windows.reshape(len(Y_windows), -1)


def fit_windows(model, train_dataloader):
    """Fit `model` on one row per window."""
    model.fit(X=windows_as_rows(train_dataloader.Y_windows))


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
    scores = np.asarray(model.decision_function(X=rows), dtype=float)
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
