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
from Explainability import ir
from Model_Selection.Sensitivity_robustness import surrogate_fidelity
from Model_Selection.Sensitivity_robustness.plot_retention import prune_superseded



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
        cv = surrogate_fidelity.held_out_classifier_fidelity(
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


def explain_exclusive_win_stage(
    point_records, adjusted_y_pred_dict, true_labels, ranked_f1_names,
    model_names, dataset, entity, explain: bool = False, *,
    stage_label: str,
    build_table,
    plot_tree_fn,
    plot_importance_fn,
    explain_dir_fn,
    tree_prefix: str,
    report_stem: str,
    report_heading: str,
    points_label: str,
    build_ir,
    ir_stem: str,
):
    """The orchestrator both point-injection stages run.

    Off-by-threshold and the GAN test ask the same question of their own
    injected points — which model got the ones the others missed, and what
    distinguishes them — so everything between "build the table" and "write the
    IR" was the same 108 lines twice, character for character apart from the
    stage's name. It is written once here; each stage supplies only what is
    genuinely its own.

    Returns the result dict, or None when `explain` is off or the table is
    infeasible.
    """
    if not explain:
        return None
    table = build_table(point_records, adjusted_y_pred_dict, true_labels, model_names)
    if table is None:
        logger.warning(f"{stage_label} explainability skipped: no injected points "
                       f"/ valid predictions.")
        return None

    models = table["model_names"]
    # Winner = highest-ranked F1 model that actually has predictions; else first valid.
    winner = next((m for m in (ranked_f1_names or []) if m in models), models[0])
    ranked_valid = [m for m in (ranked_f1_names or []) if m in models]
    runnerup = next((m for m in ranked_valid if m != winner), None)

    surrogate_note = ""
    res = {"feasible": False, "winner": winner,
           "feature_names": table["feature_names"], "per_competitor": {}}
    try:
        res = train_exclusive_win_surrogates(table, winner)
    except ImportError:
        surrogate_note = "scikit-learn unavailable — per-competitor surrogates skipped."
        logger.warning(f"{stage_label} explainability: {surrogate_note}")

    per_competitor = res.get("per_competitor", {})
    if not surrogate_note:
        # Plot every generated surrogate tree, one per competitor (degenerate
        # ones, whose clf is None, draw nothing and return None).
        written = [plot_tree_fn(info, winner, k, dataset, entity,
                                table["feature_names"])
                   for k, info in per_competitor.items()]
        plot_importance_fn(per_competitor, dataset, entity, table["feature_names"])
        # Whatever an earlier run left here describes a different outcome —
        # a different winner, or the same winner against differently-spelled
        # competitors — and the picker cannot tell the two apart. Pruned
        # AFTER the new set is on disk, so a run that dies mid-plot leaves
        # the previous set rather than deleting it and not replacing it.
        prune_superseded(explain_dir_fn(dataset, entity),
                         f"{dataset}_{entity}_{tree_prefix}",
                         [n for n in written if n])

    directory = explain_dir_fn(dataset, entity)
    report_path = os.path.join(directory, f"{dataset}_{entity}_{report_stem}.txt")
    with open(report_path, "w") as f:
        f.write(f"=== {report_heading} ===\n")
        f.write(f"Dataset: {dataset}  |  Entity: {entity}\n")
        f.write(f"Models with predictions ({len(models)}): {', '.join(models)}\n")
        f.write(f"{points_label}: {table['n_points']}\n")
        f.write(f"Features: {', '.join(table['feature_names'])}\n")
        f.write(f"F1 winner (production ranking): {winner}\n")
        f.write("(Explains the actual production run; correctness is F1/prediction-side — "
                "PR-AUC has no per-point correct/incorrect. The production ranking is unchanged.)\n\n")

        if surrogate_note:
            f.write(surrogate_note + "\n")
        elif not per_competitor:
            f.write("No competitors to compare against (winner is the only model with predictions).\n")
        else:
            order = [m for m in ranked_valid if m in per_competitor] + \
                    [m for m in per_competitor if m not in ranked_valid]
            agg: Dict[str, List[float]] = {fn: [] for fn in table["feature_names"]}
            for k in order:
                info = per_competitor[k]
                f.write(f"--- {winner} vs {k} ---\n")
                f.write(f"Exclusive wins: {info['n_exclusive_wins']} "
                        f"({info['exclusive_win_rate']:.2%} of injected points)\n")
                if info.get("degenerate"):
                    f.write(f"    {info['rules_text']}\n\n")
                    continue
                f.write(f"Surrogate train accuracy (in-sample fit): {info['train_accuracy']:.3f}\n")
                cv_acc = info.get("cv_accuracy", float('nan'))
                if not np.isnan(cv_acc):
                    f.write(f"Surrogate held-out accuracy ({info.get('cv_method', 'cv')}, "
                            f"{info.get('cv_accuracy_std', float('nan')):.3f} std): {cv_acc:.3f}\n")
                elif info.get("cv_note"):
                    f.write(f"Surrogate held-out accuracy: not estimated ({info['cv_note']})\n")
                imps = sorted(info["feature_importances"].items(), key=lambda kv: kv[1], reverse=True)
                f.write("Feature importances: "
                        + ", ".join(f"{fn} {im:.2f}" for fn, im in imps if im > 0) + "\n")
                for fn, im in info["feature_importances"].items():
                    agg[fn].append(im)
                f.write("Rules (1 = point the winner uniquely gets right):\n")
                for line in info["rules_text"].rstrip().splitlines():
                    f.write(f"    {line}\n")
                f.write("\n")
            mean_imp = {fn: (float(np.mean(v)) if v else 0.0) for fn, v in agg.items()}
            if any(mean_imp.values()):
                top = max(mean_imp.items(), key=lambda kv: kv[1])
                f.write(f"Across competitors, the winner's edge is best explained by: "
                        f"{top[0]} (mean importance {top[1]:.2f}).\n")

    result = {"table": table, "winner": winner, "runnerup": runnerup,
              "surrogates": res, "n_points": table["n_points"]}

    # ── Intermediate Representation (grounded LLM input; non-fatal) ─────────
    try:
        ir.write_stage_ir(build_ir(dataset, entity, result, ranked_f1_names),
                          dataset, entity, ir_stem)
    except Exception as e:
        logger.error(f"{stage_label} IR emission failed (non-fatal): {e}")

    return result
