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

# Base detector instances, in the order app.py loads them. The families come
# from Model_Training/hyperparameter_grids.py: NN varies k (3 instances), LOF
# and CBLOF vary contamination (4 each).
ALL_DETECTORS = (
    "LOF_1", "LOF_2", "LOF_3", "LOF_4",
    "NN_1", "NN_2", "NN_3",
    "CBLOF_1", "CBLOF_2", "CBLOF_3", "CBLOF_4",
)

DETECTOR_FAMILIES = ("LOF", "NN", "CBLOF")

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
