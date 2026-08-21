"""Exclusive-win surrogate trees, shared by the point-injection robustness stages.

Two stages ask the same question of their own injected points: *the winner got
these right and this rival did not — what were those points like?* Off-by-threshold
injects local-statistics-scaled noise around the decision boundary; the GAN test
injects generated points near the discriminator's threshold. They differ only in
the features that describe a point, so everything downstream of the feature matrix
is the same code and lives here:

  * the join from per-point records to a per-model correctness matrix,
  * one DecisionTreeClassifier per competitor over the exclusive-win target,
  * the held-out fidelity estimate that accompanies the in-sample fit,
  * the two figures (per-competitor tree, mean importance).

Nothing in this module names a feature. The caller builds `X` and
`feature_names`; this module never inspects either beyond their shape, which is
what lets one implementation serve both stages and keeps their prose readable as
siblings rather than as two drifting copies.
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger



def _surrogate_fidelity_module():
    """Import surrogate_fidelity.py, tolerating standalone by-path loading of this
    module (e.g. via importlib in test harnesses) where the Model_Selection
    package itself may not be on sys.path — falls back to loading the sibling
    file directly by its own location, the same trick those harnesses use."""
    try:
        from Model_Selection.Sensitivity_robustness import surrogate_fidelity as _sf
        return _sf
    except ModuleNotFoundError:
        import importlib.util
        _here = os.path.dirname(os.path.abspath(__file__))
        _spec = importlib.util.spec_from_file_location(
            "surrogate_fidelity", os.path.join(_here, "surrogate_fidelity.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod


def join_predictions(indices, X, feature_names, adjusted_y_pred_dict, true_labels,
                     model_names, stage_label: str = "explain") -> Optional[Dict[str, Any]]:
    """
    Join per-injected-point features to each model's production predictions.

    `indices[i]` is where point i landed in the augmented series and `X[i]` its
    feature row; the two must be aligned and the same length. `correct[i, m]` is
    (model m's production prediction at that index == the point's true label).
    No model inference is run here — the production run's predictions are reused,
    so the ranking this explains is the ranking that was reported.

    Returns {X, feature_names, correct (n x M bool), model_names, indices,
    n_points} or None when nothing usable survives.
    """
    true_labels = np.asarray(true_labels).flatten().astype(int)
    n = len(true_labels)
    if n == 0:
        return None

    idxs = np.asarray(indices, dtype=int).flatten()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(len(idxs), -1) if len(idxs) else X.reshape(0, 0)
    if idxs.size == 0 or X.shape[0] != idxs.size:
        return None

    # Drop any record whose index falls outside the augmented series (defensive).
    keep = np.flatnonzero((idxs >= 0) & (idxs < n))
    if keep.size == 0:
        return None
    idxs = idxs[keep]
    X = X[keep]

    # Keep only models that produced a full-length prediction vector.
    valid_models: List[str] = []
    preds: List[np.ndarray] = []
    for m in model_names:
        pred_list = adjusted_y_pred_dict.get(m)
        if pred_list is None or (hasattr(pred_list, "__len__") and len(pred_list) == 0):
            continue
        pred = np.asarray(pred_list).flatten().astype(int)
        if pred.shape[0] != n:
            logger.warning(f"{stage_label}: prediction length {pred.shape[0]} != {n} "
                           f"for {m}; skipping.")
            continue
        valid_models.append(m)
        preds.append(pred)
    if not valid_models:
        return None

    true_at = true_labels[idxs]
    correct = np.zeros((idxs.size, len(valid_models)), dtype=bool)
    for mi, pred in enumerate(preds):
        correct[:, mi] = pred[idxs] == true_at

    return {
        "X": X,
        "feature_names": list(feature_names),
        "correct": correct,
        "model_names": valid_models,
        "indices": idxs,
        "n_points": int(idxs.size),
    }


def train_exclusive_win_surrogates(table, winner, max_depth: int = 3,
                                   random_state: int = 0) -> Dict[str, Any]:
    """
    For the winner, fit one DecisionTreeClassifier per competitor `k` predicting
    the winner's *exclusive wins*: y_i = winner_correct_i AND NOT k_correct_i.

    Per competitor returns export_text rules, feature_importances, train_accuracy,
    n_exclusive_wins, exclusive_win_rate, and the fitted clf. Single-class targets
    (winner never / always strictly beats k) are recorded as degenerate without
    importing sklearn. sklearn is lazy-imported only when a real tree is fit.
    """
    models = list(table["model_names"])
    feature_names = table["feature_names"]
    X = table["X"]
    correct = table["correct"]
    if winner not in models:
        return {"feasible": False, "winner": winner, "feature_names": feature_names,
                "per_competitor": {}, "note": f"winner {winner} has no valid predictions"}
    w = models.index(winner)
    winner_correct = correct[:, w]

    per_competitor: Dict[str, Any] = {}
    for k in models:
        if k == winner:
            continue
        ki = models.index(k)
        y = winner_correct & ~correct[:, ki]
        n_pos = int(y.sum())
        rate = float(n_pos) / float(len(y)) if len(y) else 0.0
        if n_pos == 0:
            per_competitor[k] = {
                "degenerate": True, "clf": None, "feature_importances": {},
                "train_accuracy": float('nan'), "n_exclusive_wins": 0, "exclusive_win_rate": 0.0,
                "rules_text": f"{winner} has no exclusive wins over {k} "
                              f"({k} matches the winner on all injected points).",
            }
            continue
        if n_pos == len(y):
            per_competitor[k] = {
                "degenerate": True, "clf": None, "feature_importances": {},
                "train_accuracy": float('nan'), "n_exclusive_wins": n_pos, "exclusive_win_rate": 1.0,
                "rules_text": f"{winner} beats {k} on every injected point.",
            }
            continue
        from sklearn.tree import DecisionTreeClassifier, export_text
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        y_int = y.astype(int)
        clf.fit(X, y_int)
        importances = {fn: float(im) for fn, im in zip(feature_names, clf.feature_importances_)}
        # train_accuracy is the in-sample fit used to generate the exported
        # rules below; it is not a generalization claim. cv_accuracy is a
        # cross-validated fidelity estimate (see surrogate_fidelity.py,
        # grounded in Molnar 2022's point that surrogate fidelity should be
        # assessed as a held-out property, not read off the training fit).
        cv = _surrogate_fidelity_module().held_out_classifier_fidelity(
            X, y_int, max_depth=max_depth, random_state=random_state)
        per_competitor[k] = {
            "degenerate": False, "clf": clf, "feature_importances": importances,
            "train_accuracy": float(clf.score(X, y_int)),
            "cv_accuracy": cv["cv_accuracy"], "cv_accuracy_std": cv["cv_accuracy_std"],
            "cv_method": cv["method"], "cv_note": cv["note"],
            "n_exclusive_wins": n_pos, "exclusive_win_rate": rate,
            # These features live on narrow ranges (off-by's boundary_distance in
            # [0, 0.05], the GAN's ambiguity compressed around the discriminator
            # threshold by construction), and default 2-decimal printing collapses
            # distinct thresholds to the same value — so print with finer precision.
            "rules_text": export_text(clf, feature_names=list(feature_names), decimals=4),
        }
    return {"feasible": True, "winner": winner, "feature_names": feature_names,
            "per_competitor": per_competitor}


# ── Plots ────────────────────────────────────────────────────────────────────

def explain_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "serif", "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    })


def plot_exclusive_win_tree(info, winner, feature_names, *, directory: str,
                            filename: str, title: str):
    """Plot one winner-vs-competitor exclusive-win surrogate tree.

    Returns the filename written, or None when the surrogate is degenerate and
    there is no tree to draw. The caller collects those names so
    `plot_retention.prune_superseded` can remove whatever an earlier run left
    behind — a competitor this run has no tree for must not keep the tree some
    previous run drew for it.
    """
    if info is None or info.get("clf") is None:
        return None
    from sklearn.tree import plot_tree
    explain_rcparams()
    fig, ax = plt.subplots(figsize=(13, 8))
    plot_tree(info["clf"], feature_names=list(feature_names),
              class_names=[f"not {winner}-only", f"{winner}-only win"],
              filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(directory, filename), dpi=300)
    plt.close(fig)
    return filename


def plot_exclusive_win_importance(per_competitor, feature_names, *, directory: str,
                                  filename: str, title: str) -> None:
    """Bar chart of mean feature importance across all (non-degenerate) competitor trees."""
    imp_rows = [info["feature_importances"] for info in per_competitor.values()
                if not info.get("degenerate") and info.get("feature_importances")]
    if not imp_rows:
        return
    means = [float(np.mean([row.get(fn, 0.0) for row in imp_rows])) for fn in feature_names]
    explain_rcparams()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = np.argsort(means)
    ax.barh([feature_names[i] for i in order], [means[i] for i in order], color="#4477aa")
    ax.set_xlabel("Mean importance across competitor surrogates")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(directory, filename), dpi=300)
    plt.close(fig)
