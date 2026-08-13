"""
What is available to run: datasets, entities, and which detectors are trained.

Discovery is driven by `trained_model_path` from Configs/config.yml — a run
needs trained checkpoints, so that tree is the authoritative answer to "what
can I run right now". Results in `myresults/` are a separate question, answered
by artifacts.known_entities().
"""

import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Utils/__init__.py is empty and pipeline_spec is stdlib-only, so this import
# stays cheap — it does not drag torch/matplotlib in the way Utils.utils would.
from Utils.pipeline_spec import ALL_DETECTORS, DETECTOR_FAMILIES, family_of
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
DISPLAY_NAMES = {"skab": "SKAB", "smd": "SMD", "anomaly_archive": "UCR",
                 "msl": "MSL", "smap": "SMAP", "apple": "Apple"}

# Files that hold one entity each, per dataset layout.
_ENTITY_SUFFIXES = (".csv", ".txt")


def display_name(key: str) -> str:
    return DISPLAY_NAMES.get(str(key).lower(), str(key).upper())


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


# What actually distinguishes the instances within a family: LOF/CBLOF vary
# contamination, NN varies k. Everything else in the sidecar is shared.
_DISTINGUISHING = (("contamination", "contamination"), ("n_neighbors", "k"))


def _read_meta(pth: Path) -> Optional[dict]:
    """Hyperparameters recorded beside a checkpoint, reduced to what a chip needs.

    The sidecar nests `{train_hyperparameters, model_hyperparameters}`; only the
    model side is interesting, and within it only the value that separates
    LOF_1 from LOF_2. A corrupt or unreadable sidecar degrades to "no parameters
    shown" rather than breaking the catalog.
    """
    meta_path = pth.with_suffix(".meta")
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    model = data.get("model_hyperparameters")
    if not isinstance(model, dict):
        return None
    label = None
    for key, shown_as in _DISTINGUISHING:
        if key in model:
            value = model[key]
            label = f"{shown_as} {value:g}" if isinstance(value, (int, float)) \
                else f"{shown_as} {value}"
            break
    return {"label": label,
            "window_size": model.get("window_size"),
            "window_step": model.get("window_step")}


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
    out = []
    for name in ALL_DETECTORS:
        pth = (ent_dir / f"{name}.pth") if ent_dir else None
        available = bool(pth and pth.is_file())
        out.append({
            "name": name,
            "family": family_of(name),
            "available": available,
            "params": _read_meta(pth) if available else None,
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
        "warnings": warnings(),
        "config": {k: paths.config().get(k) for k in ("dataset_path", "trained_model_path")},
    }
    _CACHE.update(at=now, value=value)
    return value


def reset_cache() -> None:
    """Test hook."""
    _CACHE.update(at=0.0, value=None)
