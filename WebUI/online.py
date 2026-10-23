"""
Reads the online phase's results for the API.

The pipeline writes `results/online/<dataset>/<entity>/summary.json` and one
`<scenario>.json` per replayed scenario; the schema is documented once, in
Model_Selection/online_phase.py, where the writer lives. This module only finds
and returns those files — it adds nothing the pipeline did not write, so the
UI and the files cannot disagree.

Both are rewritten after every window (temporary file + rename), so a run in
progress is readable at any moment; `summary.status` says whether it is still
running. The one exception to "read only" is `mark_interrupted`, for a run
whose process died before it could finish its own summary: the job manager
calls it when a run it owns ends, and `payload`/`info` call it for a summary
that says "running" although its writer is gone (`writer_gone`).

`feasibility` answers, before a run, whether an entity's online part is long
enough for one window, with the pipeline's own split and window functions,
and — for real labels — how many of its windows hold a labelled anomaly.
"""

import json
import os
import socket
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from Utils.pipeline_spec import (ONLINE_BASELINE_SCENARIOS, ONLINE_LABELS_REAL,
                                 ONLINE_LABEL_SOURCE_LABELS, ONLINE_SCENARIOS,
                                 dataset_has_real_labels, dataset_label,
                                 online_feasibility, online_split,
                                 resolve_online_labels)
from WebUI import paths

SUMMARY_FILE = "summary.json"

# A summary that says "running" but has not been written for this long, with no
# process of record alive and no live job owning it, is taken as interrupted.
# The pipeline rewrites it after every window and every 15 s while it waits
# for a re-optimisation; a single window can still take minutes on a large
# entity, so this is generous.
STALE_SECONDS = 30 * 60

# Statuses of a phase, or of a scenario row, that only a live process can hold.
_LIVE_STATUSES = ("running", "pending", "waiting_for_reopt")

# Only these names are ever opened. The scenario list comes from a file on
# disk, and a key is turned into a filename, so it is checked against the
# vocabulary rather than trusted.
_KNOWN_KEYS = tuple(ONLINE_SCENARIOS) + tuple(ONLINE_BASELINE_SCENARIOS)


def _root() -> Path:
    # Read from paths.RESULTS at call time, not import time, so a test (or a
    # results root changed after import) is honoured.
    return paths.RESULTS / "online"


def online_dir(dataset: str, entity: str) -> Optional[Path]:
    return paths.resolve_entity_dir(_root(), dataset, entity)


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_summary(dataset: str, entity: str) -> Optional[Dict[str, Any]]:
    directory = online_dir(dataset, entity)
    if directory is None:
        return None
    return _read_json(directory / SUMMARY_FILE)


def _scenario_keys(summary: Dict[str, Any]) -> List[str]:
    order = summary.get("scenario_order")
    if not isinstance(order, list):
        order = [s.get("key") for s in summary.get("scenarios") or []
                 if isinstance(s, dict)]
    return [k for k in order if k in _KNOWN_KEYS]


def _pid_alive(pid) -> Optional[bool]:
    """Is process `pid` alive on this host? None when that cannot be told."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                       # exists, owned by someone else
    except OSError:
        return None
    return True


def writer_gone(summary: Dict[str, Any], live: bool = False,
                now: Optional[float] = None) -> Optional[str]:
    """Why a summary still saying "running" can no longer be finished, or None.

    `live` says a job of this web UI owns the dataset/entity right now; such a
    run is never judged here (its job manager marks it when it ends).
    Otherwise: the summary names the process writing it, and if that process
    is gone the run was interrupted. And a summary not rewritten for
    STALE_SECONDS is taken as interrupted whatever its process id says (the
    id may have been reused, or the process hung, or it ran on another host):
    the pipeline rewrites it after every window and every 15 s while it waits.
    Should a live pipeline write again after all, its next write restores its
    own status.
    """
    if summary.get("status") not in _LIVE_STATUSES or live:
        return None
    now = time.time() if now is None else now
    same_host = summary.get("host") in (None, socket.gethostname())
    if same_host and _pid_alive(summary.get("pid")) is False:
        return ("The pipeline process stopped before the online phase finished "
                "(cancelled, killed by the run timeout, or crashed).")
    updated = summary.get("updated_at") or summary.get("generated_at") or 0
    if now - float(updated) > STALE_SECONDS:
        return (f"No update for {int((now - float(updated)) // 60)} minutes and no "
                f"live run owns these results; the pipeline most likely stopped "
                f"before the online phase finished.")
    return None


def _settle(dataset: str, entity: str, summary: Dict[str, Any],
            is_live: Optional[Callable[[str, str], bool]]) -> Dict[str, Any]:
    """`summary`, marked interrupted (on disk too) when its writer is gone."""
    reason = writer_gone(summary, live=bool(is_live and is_live(dataset, entity)))
    if reason is None:
        return summary
    try:
        mark_interrupted(dataset, entity, since=0, reason=reason)
    except OSError:
        pass
    settled = load_summary(dataset, entity)
    if settled is None or settled.get("status") in _LIVE_STATUSES:
        # Could not be written (read-only results): report it all the same.
        settled = dict(summary, status="interrupted",
                       error=summary.get("error") or reason)
    return settled


def payload(dataset: str, entity: str, scenario: Optional[str] = None,
            summary_only: bool = False,
            is_live: Optional[Callable[[str, str], bool]] = None
            ) -> Optional[Dict[str, Any]]:
    """Summary plus each scenario's per-window series, or None when absent.

    `scenario` narrows the series to one key (an unknown or absent key yields
    an empty `scenarios`); `summary_only` leaves the series out, for cheap
    polling while a run is in progress. `is_live(dataset, entity)` tells
    whether a job of this web UI owns the entity; a "running" summary nobody
    is writing any more is reported (and marked) "interrupted".
    """
    directory = online_dir(dataset, entity)
    if directory is None:
        return None
    summary = _read_json(directory / SUMMARY_FILE)
    if summary is None:
        return None
    summary = _settle(dataset, entity, summary, is_live)
    series: Dict[str, Any] = {}
    if not summary_only:
        for key in _scenario_keys(summary):
            if scenario is not None and key != scenario:
                continue
            doc = _read_json(directory / f"{key}.json")
            if doc is not None:
                series[key] = doc
    return {
        "dataset": dataset,
        "entity": entity,
        "dataset_label": dataset_label(summary.get("dataset") or dataset),
        "summary": summary,
        "scenarios": series,
    }


def info(dataset: str, entity: str,
         is_live: Optional[Callable[[str, str], bool]] = None) -> Optional[Dict[str, Any]]:
    """A compact pointer for the catalog: is there online data, and how far."""
    summary = load_summary(dataset, entity)
    if summary is None:
        return None
    summary = _settle(dataset, entity, summary, is_live)
    rows = [s for s in summary.get("scenarios") or [] if isinstance(s, dict)]
    return {
        "dataset": dataset,
        "dataset_label": dataset_label(summary.get("dataset") or dataset),
        "entity": entity,
        "url": f"/api/online/{dataset}/{entity}",
        # The page that draws it; "url" stays the data endpoint.
        "page_url": f"/online/{dataset}/{entity}",
        "status": summary.get("status"),
        "strategy": summary.get("strategy"),
        "scenarios": _scenario_keys(summary),
        "scenario_status": {s.get("key"): s.get("status") for s in rows},
        "n_windows": summary.get("n_windows"),
        "generated_at": summary.get("generated_at"),
        "updated_at": summary.get("updated_at"),
    }


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Temporary file + rename, as the pipeline writes these files, so a
    reader polling them never sees half a document."""
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, allow_nan=False)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def mark_interrupted(dataset: str, entity: str, since: float, reason: str) -> bool:
    """Mark a summary still saying "running" as "interrupted". True if it did.

    The one thing the web UI writes into these files. A pipeline killed by the
    run timeout or a cancel (SIGKILL/SIGTERM) cannot finish its own summary,
    which would otherwise read "running" forever. Only a summary generated at
    or after `since` (the job's start) is touched, so an earlier run's files
    are left alone. Its unfinished scenarios — the summary rows and their own
    files — become "interrupted" too.
    """
    directory = online_dir(dataset, entity)
    if directory is None:
        return False
    summary = _read_json(directory / SUMMARY_FILE)
    if (summary is None or summary.get("status") not in _LIVE_STATUSES
            or (summary.get("generated_at") or 0) < since):
        return False
    now = time.time()
    for row in summary.get("scenarios") or []:
        if isinstance(row, dict) and row.get("status") in _LIVE_STATUSES:
            row["status"] = "interrupted"
            row["error"] = row.get("error") or reason
            row["waiting"] = None
    for key in _scenario_keys(summary):
        path = directory / f"{key}.json"
        doc = _read_json(path)
        if doc is not None and doc.get("status") in _LIVE_STATUSES:
            doc["status"] = "interrupted"
            doc["error"] = doc.get("error") or reason
            doc["waiting"] = None
            _write_json_atomic(path, doc)
    summary["status"] = "interrupted"
    summary["error"] = summary.get("error") or reason
    summary["updated_at"] = now
    _write_json_atomic(directory / SUMMARY_FILE, summary)
    return True


def known_entities() -> List[Tuple[str, str]]:
    """(dataset, entity) pairs with an online summary on disk."""
    root = _root()
    found: List[Tuple[str, str]] = []
    if not root.is_dir():
        return found
    for ds_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not ds_dir.is_dir():
            continue
        for ent_dir in sorted(ds_dir.iterdir(), key=lambda p: paths.natural_key(p.name)):
            if (ent_dir / SUMMARY_FILE).is_file():
                found.append((ds_dir.name, ent_dir.name))
    return found


# ── Feasibility ──────────────────────────────────────────────────────────────

# (dataset, entity, data root, downsampling, min_length) -> the test split's
# labels as 0/1 bytes, or None. Bounded: TSB-AD alone has 1,070 entities, and
# a long series' labels are up to 900,000 samples.
_TEST_LABELS: "OrderedDict[tuple, Any]" = OrderedDict()
_TEST_LABELS_MAX = 32


def test_labels(dataset: str, entity: str):
    """The entity's test labels (a 1-D 0/1 numpy array) as the pipeline loads
    them, or None when the series cannot be read. Cached: the run form asks on
    every entity change.

    Loads through `Datasets.load` with the config's dataset path, downsampling
    and min_length, exactly as run_app does, so length and labels are the
    run's. The import is local: `WebUI` is importable without torch, and
    load_data pulls it in.
    """
    cfg = paths.config()
    key = (str(dataset).lower(), str(entity), cfg.get("dataset_path"),
           cfg.get("downsampling"), cfg.get("min_length"))
    if key in _TEST_LABELS:
        _TEST_LABELS.move_to_end(key)
        return _TEST_LABELS[key]
    labels = None
    try:
        import numpy as np
        from Datasets.load import load_data
        data = load_data(dataset=str(dataset).lower(), group="test", entities=str(entity),
                         downsampling=cfg.get("downsampling"),
                         min_length=cfg.get("min_length"),
                         root_dir=cfg.get("dataset_path"), normalize=True, verbose=False)
        if data.entities:
            labels = (np.asarray(data.entities[0].labels).reshape(-1) > 0).astype("int8")
    except Exception:
        labels = None
    _TEST_LABELS[key] = labels
    while len(_TEST_LABELS) > _TEST_LABELS_MAX:
        _TEST_LABELS.popitem(last=False)
    return labels


def test_length(dataset: str, entity: str) -> Optional[int]:
    """Samples in the entity's test series as the pipeline loads it, or None."""
    labels = test_labels(dataset, entity)
    return None if labels is None else int(labels.size)


def labels_available(dataset: str, entity: str) -> bool:
    """Does the dataset's loader give the test split real anomaly labels?

    The answer of Datasets.load.provides_real_labels, without importing it:
    the named datasets, or any dataset in the presplit layout.
    """
    if dataset_has_real_labels(dataset):
        return True
    root = paths.config().get("dataset_path")
    if not root:
        return False
    directory = paths.resolve_child(Path(root), str(dataset))
    stem = str(entity)[:-4] if str(entity).endswith(".txt") else str(entity)
    return bool(directory and (directory / "train" / f"{stem}.txt").is_file()
                and (directory / "test_label").is_dir())


def feasibility(dataset: str, entity: str, iteration: int = 5,
                online_labels: Optional[str] = None,
                inject_online_regime: bool = False) -> Optional[Dict[str, Any]]:
    """Can this entity run the online phase with this --iteration, and on
    which labels? None when its test series cannot be read (the run itself
    would then say why).

    The verdict is Utils.pipeline_spec.online_feasibility — the function
    run_app uses to size the windows and decide whether to skip — applied to
    the series run_app would load. `online_labels` is the request's label
    source (None = the default); it resolves as run_app resolves it, and in
    real-label mode the verdict also counts the online windows that hold a
    labelled anomaly, and is not ok when none can be scored. A label source
    the data cannot honour is reported in `label_error` (and `label_source`
    is then None), apart from `ok`, which still answers for the length.
    """
    labels = test_labels(dataset, entity)
    if labels is None:
        return None
    available = labels_available(dataset, entity)
    label_error = None
    try:
        source = resolve_online_labels(online_labels, available,
                                       inject_online_regime=bool(inject_online_regime),
                                       dataset=dataset)
    except ValueError as e:
        source, label_error = None, str(e).partition(": ")[2] or str(e)
    offline_size, _ = online_split(labels.size)
    verdict = online_feasibility(
        int(labels.size), int(iteration),
        online_labels=labels[offline_size:] if source == ONLINE_LABELS_REAL else None)
    if source is None:
        verdict["label_source"] = None
    return {"dataset": dataset, "entity": entity, **verdict,
            "labels_available": available,
            "default_label_source": (ONLINE_LABELS_REAL if available else "injected"),
            "label_source_label": ONLINE_LABEL_SOURCE_LABELS.get(verdict["label_source"]),
            "label_error": label_error}


def reset_cache() -> None:
    """Test hook."""
    _TEST_LABELS.clear()
