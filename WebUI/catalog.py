"""
What is available to run: datasets, entities, and which detectors are trained.

Discovery is driven by `trained_model_path` from Configs/config.yml — a run
needs trained checkpoints, so that tree is the authoritative answer to "what
can I run right now". Results in `myresults/` are a separate question, answered
by artifacts.known_entities().
"""

import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Utils/__init__.py is empty and pipeline_spec is stdlib-only, so this import
# stays cheap — it does not drag torch/matplotlib in the way Utils.utils would.
from Utils.pipeline_spec import (ALL_DETECTORS, DATASET_LABELS, DECISION_METRICS,
                                 DETECTOR_FAMILIES,
                                 DETECTOR_GROUPS, GROUP_LABELS,
                                 MULTIVARIATE_FAMILIES, UNIVARIATE_FAMILIES,
                                 dataset_label, family_of,
                                 group_of)
# hyperparameter_grids.py is plain dict literals with no imports at all, and
# Model_Training/__init__.py is empty, so this reaches the grids without
# pulling torch or sklearn into the Flask process.
from Model_Training.hyperparameter_grids import (FAMILY_GRIDS, grid_combinations,
                                                 varying_keys)
from WebUI import paths

# Copied from Datasets/load.py VALID_DATASETS. Importing that module would pull
# in pandas + sklearn for the sake of eight strings; a test asserts this list
# still matches the source.
VALID_DATASETS = ("msl", "smap", "smd", "anomaly_archive", "swat",
                  "synthetic", "skab", "apple")

# Present in VALID_DATASETS but raise NotImplementedError in the loader.
UNRUNNABLE = frozenset({"swat", "synthetic"})

# On-disk directory names that are the same dataset. `load.py` aliases
# servermachinedataset -> smd, and the data root carries both spellings.
DIRECTORY_ALIASES = {"servermachinedataset": "smd"}

# How each dataset is shown. The CLI still receives the real directory name.
# Defined in `Utils.pipeline_spec` so the pipeline's own report header and this
# page cannot disagree about what to call a run; re-exported under the old name
# because that is what this module's callers import.
DISPLAY_NAMES = DATASET_LABELS

# Files that hold one entity each, per dataset layout.
_ENTITY_SUFFIXES = (".csv", ".txt")


def display_name(key: str) -> str:
    return dataset_label(key)


def _entity_from_filename(name: str, dataset_key: str) -> Optional[str]:
    """Filename -> entity id, following the loader's own convention.

    UCR files carry trailing index fields the loader strips:
    `001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt` is entity
    `001_UCR_Anomaly_DISTORTED1sddb40` (Datasets/load.py joins the first four
    underscore-separated fields).
    """
    stem, dot, suffix = name.rpartition(".")
    if not dot or f".{suffix.lower()}" not in _ENTITY_SUFFIXES:
        return None
    if dataset_key == "anomaly_archive":
        parts = stem.split("_")
        return "_".join(parts[:4]) if len(parts) >= 4 else stem
    return stem

_CACHE: Dict[str, Any] = {"at": 0.0, "value": None}
_TTL_SECONDS = 30.0


# Grid keys whose own name reads badly in a tooltip. Anything absent is shown
# under the name it has in the grid, which is what a reader would grep for.
_SHOWN_AS = {
    "n_neighbors": "k",
    "detector__window_size": "subsequence",
    "running_window_size": "running window",
    # The TSB-AD and AutoEncoder keys. Without these a tooltip would read
    # "detector__win_size 30", which shows the plumbing rather than the
    # parameter; `detector__` only exists to keep the estimator's names apart
    # from the framework's, and a reader does not need to see that.
    "detector__win_size": "subsequence",
    "detector__window": "window",
    "detector__k": "clusters",
    "detector__stride": "stride",
    "detector__power": "degree",
    "detector__cut_freq": "frequencies",
    "detector__hidden_neuron_list": "hidden layers",
    "detector__lr": "learning rate",
    # The families that stopped sweeping contamination.
    "n_clusters": "clusters",
    "detector__n_estimators": "trees",
    "detector__max_features": "max features",
    "detector__n_bins": "bins",
    "detector__n_components": "components",
    "detector__kernel": "kernel",
    "detector__support_fraction": "support fraction",
    "detector__epoch_num": "epochs",
    "detector__num_epochs": "epochs",
    "detector__epochs": "epochs",
}


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) \
        else str(value)


def _label_from(source: Dict[str, Any], keys: List[str]) -> Optional[str]:
    """"contamination 0.15" / "input_size 64, state_hsize 256" — the values of
    the varying keys, in grid order, or None when none of them are present."""
    parts = [f"{_SHOWN_AS.get(k, k)} {_fmt(source[k])}" for k in keys if k in source]
    return ", ".join(parts) if parts else None


def _grid_params(name: str) -> Optional[dict]:
    """What this instance WOULD be trained as, read from its family's grid.

    Nothing is on disk for an untrained detector, so the sidecar cannot answer;
    the grid can, because `TrainModels` names the i-th combination of it
    `{FAMILY}_{i+1}`. This is what lets an untrained chip still say what it is
    rather than only that it is missing.
    """
    grid = FAMILY_GRIDS.get(family_of(name))
    if not grid:
        return None
    try:
        index = int(name.rsplit("_", 1)[1]) - 1
    except (IndexError, ValueError):
        return None
    combinations = grid_combinations(grid)
    if not 0 <= index < len(combinations):
        return None
    combination = combinations[index]
    return {"label": _label_from(combination, varying_keys(grid)),
            "window_size": combination.get("window_size"),
            "window_step": combination.get("window_step")}


def _read_meta(pth: Path, name: str) -> Optional[dict]:
    """Hyperparameters recorded beside a checkpoint, reduced to what a chip needs.

    The sidecar nests `{train_hyperparameters, model_hyperparameters}`; only the
    model side is interesting, and within it only the values that separate
    LOF_1 from LOF_2 — which the family's grid defines, so the keys to show come
    from there rather than from a hardcoded pair.

    The sidecar wins over the grid for a trained detector even when the two
    disagree, because it describes the checkpoint that will actually run. They
    do disagree: the w=64 LOF and CBLOF checkpoints predate the move to
    window_size 1, and a chip claiming "window 1" for a file trained at 64 would
    hide exactly the staleness worth seeing.

    A corrupt or unreadable sidecar degrades to the grid's answer rather than
    breaking the catalog. So does a sidecar that holds NONE of the keys the
    grid currently varies — which is every checkpoint written before a family
    stopped sweeping `contamination`, since those sidecars record a
    contamination and no `n_neighbors`, `n_clusters` or `n_estimators`. Taking
    the sidecar literally there produces an empty label, and a chip that names
    no hyperparameter at all is less use than one naming what a retrain will
    give it. Retraining makes the two agree again.
    """
    meta_path = pth.with_suffix(".meta")
    if not meta_path.is_file():
        return _grid_params(name)
    try:
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return _grid_params(name)
    if not isinstance(data, dict):
        return _grid_params(name)
    model = data.get("model_hyperparameters")
    if not isinstance(model, dict):
        return _grid_params(name)
    grid = FAMILY_GRIDS.get(family_of(name))
    keys = varying_keys(grid) if grid else []
    label = _label_from(model, keys)
    if label is None:
        return _grid_params(name)
    return {"label": label,
            "window_size": model.get("window_size"),
            "window_step": model.get("window_step")}


_CHANNELS: Dict[tuple, Optional[int]] = {}


def channels_for(dataset: str, entity: str) -> Optional[int]:
    """How many channels this entity has, or None if it cannot be read.

    Loads through `Datasets.load` rather than counting columns per format, so
    the answer is whatever the pipeline itself would see. 6-600 ms, cached
    because the run page asks on every entity change. The import is local:
    `WebUI` is deliberately importable without torch, and load_data pulls it.
    """
    key = (dataset, entity)
    if key in _CHANNELS:
        return _CHANNELS[key]
    n = None
    try:
        from Datasets.load import load_data
        data = load_data(dataset=dataset, group="train", entities=entity,
                         downsampling=10, min_length=256,
                         root_dir=paths.config().get("dataset_path"),
                         normalize=True, verbose=False)
        n = int(data.entities[0].Y.shape[0])
    except Exception:
        n = None
    _CHANNELS[key] = n
    return n


def unusable_families(n_channels: Optional[int]) -> frozenset:
    """Families that cannot mean anything at this width. Empty when the channel
    count is unknown: hiding a usable detector is worse than showing one that
    the run would drop anyway."""
    if n_channels is None:
        return frozenset()
    return UNIVARIATE_FAMILIES if n_channels > 1 else MULTIVARIATE_FAMILIES


def detectors_for(dataset: str, entity: str) -> List[Dict[str, Any]]:
    """The 11 canonical detectors, each flagged available or not for this entity.

    Availability comes from disk, but the list itself is always the canonical
    eleven: some entities carry stale checkpoints (RNN_*.pth under SMD) that are
    not selectable, and a detector missing here should show as disabled rather
    than vanish.
    """
    root = paths.config().get("trained_model_path")
    ent_dir = None
    if root:
        ent_dir = paths.resolve_entity_dir(Path(root), dataset, entity)
    # Width-unsuitable families are omitted, not disabled: a missing checkpoint
    # can be trained, but no run on this entity can ever use these.
    unusable = unusable_families(channels_for(dataset, entity))
    out = []
    for name in ALL_DETECTORS:
        if family_of(name) in unusable:
            continue
        pth = (ent_dir / f"{name}.pth") if ent_dir else None
        available = bool(pth and pth.is_file())
        out.append({
            "name": name,
            "family": family_of(name),
            # There is no separate display name. `name` is canonical AND what
            # the reader sees — it is what the CLI takes, what the checkpoint is
            # called, and what every selection is submitted as. This payload
            # used to carry `display`/`family_display` alongside it, from when
            # four families were spelled short in the pool and long on screen.
            # The paper's Table I group, so the run page can colour a chip and
            # build its select-all buttons without a second copy of the taxonomy.
            "group": group_of(family_of(name)),
            "available": available,
            # Untrained detectors get their grid's answer, not nothing: the run
            # page lists what a family WOULD train, so the reader can tell
            # LOF_1 from LOF_4 before deciding to train either.
            "params": _read_meta(pth, name) if available else _grid_params(name),
        })
    return out


def _dataset_dirs() -> Dict[str, List[Path]]:
    """dataset key -> every directory that holds it, across both roots.

    Entities are discovered from the DATA root, not just from trained_models:
    an entity with no checkpoints is still runnable, it simply trains first.
    Aliased directories (SMD / ServerMachineDataset) merge into one entry.
    """
    config = paths.config()
    out: Dict[str, List[Path]] = {}
    for root in (config.get("dataset_path"), config.get("trained_model_path")):
        if not root or not Path(root).is_dir():
            continue
        try:
            children = sorted(Path(root).iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            key = DIRECTORY_ALIASES.get(child.name.lower(), child.name.lower())
            if key not in VALID_DATASETS:
                continue      # NASA/, TCPD/ — present but not loadable
            out.setdefault(key, []).append(child)
    return out


def _entities_in(directory: Path, dataset_key: str) -> List[str]:
    """Entities inside one dataset directory, whatever its layout.

    Three layouts occur: subdirectories per entity (trained_models), one file
    per entity (SKAB's 0.csv, UCR's .txt), and SMD's train/test/test_label
    split where the entity names live inside `train/`.
    """
    names = set()
    try:
        children = list(directory.iterdir())
    except OSError:
        return []
    split_dir = next((c for c in children if c.is_dir() and c.name == "train"), None)
    if split_dir is not None:
        try:
            for item in split_dir.iterdir():
                entity = _entity_from_filename(item.name, dataset_key)
                if entity:
                    names.add(entity)
        except OSError:
            pass
    for child in children:
        if child.is_dir():
            if child.name in ("train", "test", "test_label"):
                continue
            names.add(child.name)
        else:
            entity = _entity_from_filename(child.name, dataset_key)
            if entity:
                names.add(entity)
    return sorted(names, key=paths.natural_key)


def entities_for(dataset: str) -> List[str]:
    key = DIRECTORY_ALIASES.get(str(dataset).lower(), str(dataset).lower())
    names = set()
    for directory in _dataset_dirs().get(key, []):
        names.update(_entities_in(directory, key))
    return sorted(names, key=paths.natural_key)


def trained_entities(dataset: str) -> set:
    """Entities that already have checkpoints — the rest must train first."""
    root = paths.config().get("trained_model_path")
    if not root:
        return set()
    ds_dir = paths.resolve_child(Path(root), dataset)
    if ds_dir is None:
        return set()
    try:
        return {p.name for p in ds_dir.iterdir() if p.is_dir()}
    except OSError:
        return set()


def datasets() -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    for key, dirs in sorted(_dataset_dirs().items()):
        entities = entities_for(key)
        found.append({
            # `name` is what goes to --dataset; the loader lowercases and
            # aliases it, so the canonical key is always safe to pass.
            "name": key,
            "key": key,
            "label": display_name(key),
            "runnable": key not in UNRUNNABLE,
            "n_entities": len(entities),
            "directories": [d.name for d in dirs],
        })
    return sorted(found, key=lambda d: d["label"].lower())


def warnings() -> List[Dict[str, str]]:
    """Configuration facts worth stating on the run form before a long wait."""
    out = []
    cfg = paths.config()
    if not cfg.get("trained_model_path"):
        out.append({"code": "no_model_path",
                    "text": "Configs/config.yml has no trained_model_path — no datasets "
                            "can be discovered."})
    elif not Path(cfg["trained_model_path"]).is_dir():
        out.append({"code": "model_path_missing",
                    "text": f"trained_model_path does not exist: {cfg['trained_model_path']}"})
    # No warning for `overwrite: True` in the config file. The run form owns that
    # choice — `build_argv` always passes --overwrite explicitly from the
    # checkbox, so the config value never reaches a run started here, and the
    # Options section already says what the checkbox costs.
    return out


def catalog(refresh: bool = False) -> Dict[str, Any]:
    """Everything the run form needs, cached briefly (directory scans are cheap
    but the form polls)."""
    now = time.time()
    if not refresh and _CACHE["value"] is not None and now - _CACHE["at"] < _TTL_SECONDS:
        return _CACHE["value"]

    value = {
        "datasets": [
            {**d, "entities": entities_for(d["name"]),
             "trained": sorted(trained_entities(d["name"]), key=paths.natural_key)}
            for d in datasets()
        ],
        "detector_families": list(DETECTOR_FAMILIES),
        # The paper's Table I group names, in order. Sent even when a group has
        # no detectors in the pool (FM today) so the run page can still show it,
        # which it cannot infer from the per-entity detector list alone.
        "detector_groups": list(DETECTOR_GROUPS),
        # Spelled out for display. The keys stay short because they are the
        # identifier (API value, CSS suffix); "NN" as a label sat next to a
        # detector family also called NN and read as the same thing.
        "group_labels": dict(GROUP_LABELS),
        "all_detectors": list(ALL_DETECTORS),
        "stages": [
            {"token": "ga", "label": "Genetic algorithm (ensemble)"},
            {"token": "thompson", "label": "Thompson Sampling (single model)"},
            {"token": "gan", "label": "GAN perturbations"},
            {"token": "offby", "label": "Off-by-threshold sensitivity"},
            {"token": "montecarlo", "label": "Monte Carlo noise"},
        ],
        "stage_groups": {"all": ["ga", "thompson", "gan", "offby", "montecarlo"],
                         "robustness": ["gan", "offby", "montecarlo"]},
        # `symbol` is what the fitness formula is written with, taken from
        # DECISION_METRICS so the preview and the run's report agree.
        "decision_metrics": [
            {"token": "f1", "label": "F1 score", "symbol": DECISION_METRICS["f1"]},
            {"token": "pr_auc", "label": "PR-AUC", "symbol": DECISION_METRICS["pr_auc"]},
            {"token": "vus", "label": "VUS-ROC", "symbol": DECISION_METRICS["vus"]},
        ],
        # `note` warns about types measured to behave poorly; see README.
        "anomaly_types": [
            {"token": "spikes", "label": "Spikes (scattered points)"},
            {"token": "contextual", "label": "Contextual (affine shift)"},
            {"token": "flip", "label": "Flip (time-reversed segment)"},
            {"token": "speedup", "label": "Speedup (resampled segment)",
             "note": "Resamples, so the series changes length and the realised "
                     "rate runs above the target."},
            {"token": "noise", "label": "Noise (additive Gaussian)",
             "note": "Default noise_std=0.05 is below the step-to-step variation "
                     "of some entities."},
            {"token": "cutoff", "label": "Cutoff (constant segment)"},
            {"token": "scale", "label": "Scale (amplified segment)"},
            {"token": "wander", "label": "Wander (baseline drift)",
             "note": "Shifts the whole series after the segment but labels only "
                     "the segment."},
            {"token": "average", "label": "Average (smoothed segment)",
             "note": "Perturbation measures below the step-to-step variation on "
                     "downsampled data."},
        ],
        "warnings": warnings(),
        "config": {k: paths.config().get(k) for k in ("dataset_path", "trained_model_path")},
    }
    _CACHE.update(at=now, value=value)
    return value


def reset_cache() -> None:
    """Test hook."""
    _CACHE.update(at=0.0, value=None)
