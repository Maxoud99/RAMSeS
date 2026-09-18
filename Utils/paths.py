import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YML = REPO_ROOT / "Configs" / "config.yml"

# used when the config is missing.
DEFAULTS = {
    "dataset_path": "Datasets",
    "trained_model_path": "trained_models",
    "results_path": "results",
}

_RESULTS_ROOT: Optional[Path] = None


def resolve(path, default: str = "") -> Path:
    value = str(path).strip() if path not in (None, "") else default
    expanded = Path(os.path.expanduser(os.path.expandvars(value)))
    return expanded if expanded.is_absolute() else (REPO_ROOT / expanded).resolve()


def _read_config() -> dict:
    try:
        import yaml
        with open(CONFIG_YML) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def results_root() -> Path:
    global _RESULTS_ROOT
    if _RESULTS_ROOT is None:
        _RESULTS_ROOT = resolve(_read_config().get("results_path"),
                                DEFAULTS["results_path"])
    return _RESULTS_ROOT


def results_dir(*parts) -> str:
    return f"{results_root().joinpath(*(str(p) for p in parts))}{os.sep}"


def reset_cache() -> None:
    global _RESULTS_ROOT
    _RESULTS_ROOT = None
