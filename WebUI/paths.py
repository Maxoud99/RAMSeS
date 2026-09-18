"""
Where everything lives, and how to resolve a dataset/entity directory.

The pipeline writes `results/{tree}/{dataset}/{entity}/…` using the dataset
string exactly as it was typed on the command line, so `--dataset skab` creates
`skab/` while an earlier `--dataset SKAB` created `SKAB/`. That only appears to
work because macOS is case-insensitive; on Linux the two are different
directories. Every lookup here therefore resolves case-insensitively.
"""

import os
import re
from pathlib import Path
from typing import Optional

from Utils import paths as ramses_paths

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = ramses_paths.results_root()
CONFIG_YML = REPO_ROOT / "Configs" / "config.yml"
WEBUI_LOGS = RESULTS / "webui_logs"

# Artifact trees the UI reads.
EXPLANATIONS_IR = RESULTS / "explanations_ir"
EXPLANATIONS_NL = RESULTS / "explanations_nl"
# The pipeline's own numeric report: timings, memory, per-stage rankings and the
# final decision. Written by run_app, independent of the explainability layer.
COMPREHENSIVE = RESULTS / "comprehensive"

_CONFIG_CACHE: Optional[dict] = None


def config() -> dict:
    """The handful of `Configs/config.yml` values the UI needs.

    Uses PyYAML (already a pipeline dependency) but tolerates a missing or
    unparseable file: the UI degrades to "no datasets discovered" rather than
    refusing to start.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    data = {}
    try:
        import yaml
        with open(CONFIG_YML) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = {}
    # Resolved, not raw: the config ships relative paths so nobody has to edit
    # it before a first run, and they are relative to the repository root rather
    # than to wherever the Flask process happens to have been started.
    _CONFIG_CACHE = {
        "dataset_path": str(ramses_paths.resolve(
            data.get("dataset_path"), ramses_paths.DEFAULTS["dataset_path"])),
        "trained_model_path": str(ramses_paths.resolve(
            data.get("trained_model_path"),
            ramses_paths.DEFAULTS["trained_model_path"])),
        "results_path": str(ramses_paths.resolve(
            data.get("results_path"), ramses_paths.DEFAULTS["results_path"])),
        # Surfaced as a run-form warning: with overwrite on, every run retrains
        # all base detectors, which is most of the wall-clock time.
        "overwrite": bool(data.get("overwrite", False)),
    }
    return _CONFIG_CACHE


def reset_config_cache() -> None:
    """Test hook."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def resolve_child(parent: Path, name: str) -> Optional[Path]:
    """`parent/name`, matched case-insensitively; None when absent.

    Returns the exact path when it exists, so callers keep the real on-disk
    casing rather than the user's spelling.
    """
    if not name:
        return None
    exact = parent / name
    if exact.is_dir():
        return exact
    try:
        lowered = name.lower()
        for child in parent.iterdir():
            if child.is_dir() and child.name.lower() == lowered:
                return child
    except OSError:
        return None
    return None


def resolve_entity_dir(root: Path, dataset: str, entity: str) -> Optional[Path]:
    """`root/{dataset}/{entity}` with both levels resolved case-insensitively."""
    ds_dir = resolve_child(root, str(dataset))
    if ds_dir is None:
        return None
    return resolve_child(ds_dir, str(entity))


_NUM_CHUNK = re.compile(r"(\d+)")


def natural_key(name: str):
    """Sort key so entity lists read 2, 3, 10 rather than 10, 2, 3."""
    return [int(part) if part.isdigit() else part.lower()
            for part in _NUM_CHUNK.split(str(name))]


def rel_to_results(path: Path) -> str:
    """Path relative to the results root, for building /media URLs."""
    return os.path.relpath(str(Path(path).resolve()), str(RESULTS.resolve()))
