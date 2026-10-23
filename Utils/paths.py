import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YML = REPO_ROOT / "Configs" / "config.yml"

# used when the config is missing.
DEFAULTS = {
    "dataset_path": "Datasets",
    "trained_model_path": "trained_models",
    "results_path": "results",
}

_RESULTS_ROOT: Optional[Path] = None

# Per-THREAD override of the results root, for code that re-runs part of the
# pipeline next to a live run in the same process: that pipeline writes
# fixed-name artifacts (the Thompson history plot, the Monte Carlo summary) and
# prunes timestamped ones (off-by-threshold) under the same trees the web UI
# reads the run's figures from, and a process-wide switch would also move the
# live run's own writes. (The online phase's re-optimisations used to be such
# a thread; they now run in their own processes, which call set_results_root.)
_LOCAL = threading.local()


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
    override = getattr(_LOCAL, "root", None)
    if override is not None:
        return override
    if _RESULTS_ROOT is None:
        _RESULTS_ROOT = resolve(_read_config().get("results_path"),
                                DEFAULTS["results_path"])
    return _RESULTS_ROOT


def results_dir(*parts) -> str:
    return f"{results_root().joinpath(*(str(p) for p in parts))}{os.sep}"


def set_results_root(path) -> Path:
    """Pin the results root to `path` for the whole process.

    The CLI calls this with the `results_path` of the config file it was GIVEN.
    Without it every writer read Configs/config.yml instead, so
    `--config_file_path other.yml` changed the dataset and model paths but left
    the results in the repository's tree.
    """
    global _RESULTS_ROOT
    _RESULTS_ROOT = resolve(path, DEFAULTS["results_path"])
    return _RESULTS_ROOT


@contextmanager
def redirected_results(root) -> Iterator[Path]:
    """Send this thread's `results_dir()` writes under `root` until exit.

    Only the calling thread is affected; threads it starts are not, because
    `threading.local` is not inherited. Code run inside must therefore stay on
    this thread — which is why the online re-optimisation runs the sequential
    pipeline.
    """
    previous = getattr(_LOCAL, "root", None)
    _LOCAL.root = Path(root)
    try:
        yield _LOCAL.root
    finally:
        _LOCAL.root = previous


def reset_cache() -> None:
    global _RESULTS_ROOT
    _RESULTS_ROOT = None
