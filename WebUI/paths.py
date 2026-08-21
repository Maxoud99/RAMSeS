"""
Where everything lives, and how to resolve a dataset/entity directory.

The pipeline writes `myresults/{tree}/{dataset}/{entity}/…` using the dataset
string exactly as it was typed on the command line, so `--dataset skab` creates
`skab/` while an earlier `--dataset SKAB` created `SKAB/`. That only appears to
work because macOS is case-insensitive; on Linux the two are different
directories. Every lookup here therefore resolves case-insensitively.
"""

import os
import re
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MYRESULTS = REPO_ROOT / "myresults"
CONFIG_YML = REPO_ROOT / "Configs" / "config.yml"
WEBUI_LOGS = MYRESULTS / "webui_logs"

# Artifact trees the UI reads.
EXPLANATIONS_IR = MYRESULTS / "explanations_ir"
EXPLANATIONS_NL = MYRESULTS / "explanations_nl"
# The pipeline's own numeric report: timings, memory, per-stage rankings and the
# final decision. Written by run_app, independent of the explainability layer.
COMPREHENSIVE = MYRESULTS / "comprehensive"

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
    _CONFIG_CACHE = {
        "dataset_path": data.get("dataset_path"),
        "trained_model_path": data.get("trained_model_path"),
        "results_path": data.get("results_path"),
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


def rel_to_myresults(path: Path) -> str:
    """Path relative to `myresults/`, for building /media URLs."""
    return os.path.relpath(str(Path(path).resolve()), str(MYRESULTS.resolve()))
