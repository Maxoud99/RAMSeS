"""Held-out fidelity estimation for post-hoc surrogate models.

In-sample fidelity (fitting a surrogate and scoring it on the same rows) is
optimistic for a high-capacity, low-bias model class like a shallow decision
tree: a tree can fit small idiosyncrasies in the observed sample that would
not reproduce on a fresh draw, so a high in-sample accuracy/R^2 does not by
itself show the surrogate has captured a generalizable pattern rather than
memorized the specific points it saw. Molnar (2022, *Interpretable Machine
Learning*, ch. on surrogate models) makes this point directly: fidelity is a
claim about how well the surrogate approximates the target model's behavior
in general, and that claim needs a held-out (or cross-validated) estimate to
be meaningful, not just a fit-quality number computed on the training rows.

These helpers wrap the classifier/regressor surrogates used throughout the
robustness explainability layer (Monte Carlo Method B, off-by-threshold /
SBA per-competitor trees) with a cross-validated fidelity estimate. The
in-sample fit is still produced and reported (it is what actually generates
the exported tree / rule text used for the human-readable explanation), but
the *fidelity number* used to judge whether the explanation is accurate
should be this held-out estimate, reported alongside it.

Both helpers degrade gracefully on the small samples typical of this layer
(a handful of noise levels x repeats, or a modest number of injected
points): they shrink the fold count to what the data can support and fall
back to leave-one-out for very small samples, rather than raising.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict

import numpy as np


def _resolve_fold_plan(n_samples: int, requested_splits: int, min_class_count: int | None = None):
    """Pick a feasible (n_splits, method) pair for the given sample size.

    Returns method in {"stratified_kfold", "kfold", "leave_one_out", "infeasible"}.
    """
    if n_samples < 3:
        return 0, "infeasible"
    cap = min_class_count if min_class_count is not None else n_samples
    n_splits = min(requested_splits, n_samples, cap)
    if n_splits >= 2:
        method = "stratified_kfold" if min_class_count is not None else "kfold"
        return n_splits, method
    return n_samples, "leave_one_out"


def held_out_classifier_fidelity(X, y, max_depth: int = 3, random_state: int = 0,
                                 n_splits: int = 5) -> Dict[str, Any]:
    """Cross-validated accuracy of DecisionTreeClassifier(max_depth) on (X, y).

    Grounds the surrogate's reported fidelity in held-out performance rather
    than the in-sample fit alone (see module docstring / Molnar 2022).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y)
    n = len(y)
    classes = sorted(Counter(y).items())
    min_class_count = min(c for _, c in classes) if classes else 0

    if len(classes) < 2:
        return {"feasible": False, "cv_accuracy": float("nan"), "cv_accuracy_std": float("nan"),
                "n_splits": 0, "method": "n/a",
                "note": "single-class target — held-out accuracy is undefined."}

    n_splits_used, method = _resolve_fold_plan(n, n_splits, min_class_count)
    if method == "infeasible":
        return {"feasible": False, "cv_accuracy": float("nan"), "cv_accuracy_std": float("nan"),
                "n_splits": 0, "method": "n/a",
                "note": f"too few samples (n={n}) for any held-out estimate."}

    from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    if method == "stratified_kfold":
        cv = StratifiedKFold(n_splits=n_splits_used, shuffle=True, random_state=random_state)
    elif method == "kfold":
        cv = KFold(n_splits=n_splits_used, shuffle=True, random_state=random_state)
    else:
        cv = LeaveOneOut()

    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return {"feasible": True, "cv_accuracy": float(np.mean(scores)),
            "cv_accuracy_std": float(np.std(scores)), "n_splits": n_splits_used, "method": method,
            "note": ""}


def held_out_regressor_fidelity(X, y, max_depth: int = 3, random_state: int = 0,
                                n_splits: int = 5) -> Dict[str, Any]:
    """Cross-validated R^2 of DecisionTreeRegressor(max_depth) on (X, y)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    n = len(y)

    n_splits_used, method = _resolve_fold_plan(n, n_splits, min_class_count=None)
    if method == "infeasible":
        return {"feasible": False, "cv_r2": float("nan"), "cv_r2_std": float("nan"),
                "n_splits": 0, "method": "n/a",
                "note": f"too few samples (n={n}) for any held-out estimate."}

    # Guard the force_finite family only: a target constant up to float noise
    # (scores are bounded metrics, so a spread below 1e-8 is numerical jitter,
    # not signal) makes per-fold SS_tot ~ 0, and R^2 either explodes to
    # astronomically negative values or is silently forced to 0.0/1.0 by
    # sklearn. Ordinary negative R^2 on varying targets is informative and
    # passes through untouched.
    if float(np.ptp(y)) < 1e-8:
        return {"feasible": True, "cv_r2": float("nan"), "cv_r2_std": float("nan"),
                "n_splits": 0, "method": "constant_target",
                "note": "target is (near-)constant across the sweep; held-out "
                        "R^2 is undefined (no variance to explain)."}

    from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
    from sklearn.tree import DecisionTreeRegressor

    reg = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)

    # R^2 is undefined for a single left-out point (LeaveOneOut); fall back to
    # negative-MSE in that regime and note it, rather than emitting NaNs.
    if method != "kfold":
        scores = cross_val_score(reg, X, y, cv=LeaveOneOut(),
                                 scoring="neg_mean_squared_error")
        return {"feasible": True, "cv_r2": float("nan"), "cv_mse": float(-np.mean(scores)),
                "cv_r2_std": float("nan"), "n_splits": n_splits_used, "method": method,
                "note": "sample too small for K-fold R^2; reporting leave-one-out MSE instead."}

    # Per-fold R^2 with the force_finite convention extended to float-noise
    # variance: a mostly-flat curve passes the global constant guard, but a
    # shuffled fold whose TEST points all sit on the flat part has SS_tot at
    # float-epsilon scale and its R^2 explodes to astronomical negatives.
    # sklearn's force_finite only covers exactly-zero variance; here any fold
    # with test spread < 1e-8 is scored the same way it would be (1.0 when
    # predictions match within the tolerance, else 0.0). Folds with genuine
    # variance keep their true — possibly negative — R^2.
    cv = KFold(n_splits=n_splits_used, shuffle=True, random_state=random_state)
    fold_scores = []
    n_degenerate = 0
    for tr, te in cv.split(X):
        f = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        f.fit(X[tr], y[tr])
        pred = f.predict(X[te])
        y_te = y[te]
        if float(np.ptp(y_te)) < 1e-8:
            n_degenerate += 1
            fold_scores.append(
                1.0 if float(np.max(np.abs(pred - y_te))) < 1e-8 else 0.0)
        else:
            ss_res = float(np.sum((y_te - pred) ** 2))
            ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
            fold_scores.append(1.0 - ss_res / ss_tot)
    note = ("" if n_degenerate == 0 else
            f"{n_degenerate} of {n_splits_used} folds had (near-)constant test "
            f"targets and were scored by the force_finite convention (0/1).")
    return {"feasible": True, "cv_r2": float(np.mean(fold_scores)),
            "cv_r2_std": float(np.std(fold_scores)),
            "n_splits": n_splits_used, "method": method, "note": note}
