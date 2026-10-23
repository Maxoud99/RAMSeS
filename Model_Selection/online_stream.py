"""
The live monitor's stream files: the series the online phase replays, its
labels, and what each scenario's deployed detectors said about every window.

Written beside summary.json, under `results/online/<dataset>/<entity>/`:

stream.json — written once, when the stream is built ::

    {"schema_version": 1, "dataset": "tsb-ad-u", "entity": "...",
     "label_source": "real" | "injected",
     "n_channels": 1, "channel_names": ["0"],
     "offline_size": 1200,               # samples [0, offline_size) are the offline part
     "online_length": 300,               # the online stream, in original time positions
     "total_length": 1500,               # offline_size + online_length
     "window_size": 100, "step_size": 5, "n_windows": 41,
     "window_coordinates": "test_series" | "online_stream",
     "has_real_labels": true,            # real_labels is in stream.npz
     "has_injected_labels": false,       # injected_labels is in stream.npz
     "copies": [{"copy": 0, "anomaly_type": "spikes"}],
                                         # one per online copy; a single uninjected
                                         # copy (anomaly_type null) with real labels
     "updated_at": 1760000000.0}

stream.npz — the arrays, written once:
    offline          (C, offline_size) float32  the offline part as loaded (scaled)
    online           (K, C, online_length) float32  each online copy as replayed, at
                                        the original time positions (`speedup`
                                        copies resampled by their position map)
    real_labels      (total_length,) int8   the TRUE labels, offline and online,
                                        when the dataset has them (and the online
                                        part is not a regime-shifted stream)
    injected_labels  (K, online_length) int8  each copy's injected labels, when
                                        the online stream was injected

stream/<scenario>/w<index>.npz — one per scored window, written as it is scored:
    start, end           the window's [start, end) in total positions
    single_scores        float32  the deployed single model's anomaly scores on
                                  the window's samples (copy 0's part)
    single_threshold     float    the score they were cut at; NaN when the
                                  window could not be scored (one-class labels)
    single_flagged       int8     scores >= threshold (all 0 when unscorable)
    ensemble_*           the same for the deployed ensemble's meta-model
                         probabilities
    (an array is absent when that signal produced no scores at all)

Every file is written to a temporary name and renamed into place, so a reader
polling the directory never sees half a file. The decimated views the web UI
draws are computed from these by WebUI/stream.py.
"""

import json
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

SCHEMA_VERSION = 1
META_FILE = "stream.json"
DATA_FILE = "stream.npz"
WINDOWS_DIR = "stream"
SIGNALS = ("single", "ensemble")


def _atomic(path: str, write) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=os.path.splitext(path)[1], dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_npz_atomic(path: str, **arrays) -> None:
    # Uncompressed: the web UI reads one array of a large series per request.
    _atomic(path, lambda f: np.savez(f, **arrays))


def write_json_atomic(path: str, payload) -> None:
    _atomic(path, lambda f: f.write(json.dumps(payload, indent=1, allow_nan=False).encode()))


def copy_at_original_positions(copy, length: int):
    """(Y, labels) of one online copy at the original time positions 0..length-1.

    Equal-length copies are returned as they are. A `speedup` copy resamples
    its anomalous segment; each original position takes the copy's sample at
    `copy.position(t)`, so the monitor draws every copy on one time axis.
    """
    Y = np.asarray(copy.Y)
    labels = None if copy.labels is None else np.asarray(copy.labels).reshape(-1)
    if Y.shape[1] == length:
        return Y, labels
    idx = np.clip([copy.position(t) for t in range(length)], 0, Y.shape[1] - 1)
    return Y[:, idx], (None if labels is None else labels[idx])


class StreamWriter:
    """Writes stream.json and stream.npz once, then one file per scored
    window per scenario (`record`, called by the scenario runner)."""

    def __init__(self, directory: str, *, dataset: str, entity: str, label_source: str,
                 offline_Y, offline_real_labels, online_copies: Sequence,
                 online_real_labels, offline_size: int, window_size: int, step_size: int,
                 n_windows: int, window_coordinates: str = "test_series",
                 injected: bool = True):
        self.directory = directory
        self.windows_dir = os.path.join(directory, WINDOWS_DIR)
        # An earlier run's windows would read as this run's.
        shutil.rmtree(self.windows_dir, ignore_errors=True)
        offline_Y = np.asarray(offline_Y, dtype=np.float32)
        length = int(online_copies[0].original_length) if online_copies else 0
        online, injected_labels, types = [], [], []
        for c in online_copies:
            Y, labels = copy_at_original_positions(c, length)
            online.append(np.asarray(Y, dtype=np.float32))
            injected_labels.append((np.asarray(labels) > 0).astype(np.int8)
                                   if labels is not None else np.zeros(length, np.int8))
            types.append(c.anomaly_type)
        arrays = {"offline": offline_Y, "online": np.stack(online)}
        has_real = offline_real_labels is not None and online_real_labels is not None
        if has_real:
            arrays["real_labels"] = (np.concatenate([
                np.asarray(offline_real_labels).reshape(-1),
                np.asarray(online_real_labels).reshape(-1)[:length]]) > 0).astype(np.int8)
        if injected:
            arrays["injected_labels"] = np.stack(injected_labels)
        write_npz_atomic(os.path.join(directory, DATA_FILE), **arrays)
        self.meta = {
            "schema_version": SCHEMA_VERSION, "dataset": dataset, "entity": entity,
            "label_source": label_source,
            "n_channels": int(offline_Y.shape[0]),
            "channel_names": [str(c) for c in range(offline_Y.shape[0])],
            "offline_size": int(offline_size), "online_length": length,
            "total_length": int(offline_size) + length,
            "window_size": int(window_size), "step_size": int(step_size),
            "n_windows": int(n_windows), "window_coordinates": window_coordinates,
            "has_real_labels": bool(has_real), "has_injected_labels": bool(injected),
            "copies": [{"copy": i, "anomaly_type": t} for i, t in enumerate(types)],
            "updated_at": time.time(),
        }
        write_json_atomic(os.path.join(directory, META_FILE), self.meta)

    def window_path(self, key: str, index: int) -> str:
        return os.path.join(self.windows_dir, key, f"w{int(index):05d}.npz")

    def record(self, key: str, window, scores) -> None:
        """One scored window of scenario `key`: copy 0's part of each signal's
        scores, the threshold it was cut at, and the flagged samples."""
        seg = (window.segments or [{}])[0]
        lo = int(seg.get("data_start", 0))
        hi = seg.get("data_end")
        arrays: Dict[str, Any] = {"start": np.int64(window.start), "end": np.int64(window.end)}
        for signal in SIGNALS:
            trace = (getattr(scores, "trace", None) or {}).get(signal)
            if not trace or trace.get("scores") is None:
                continue
            values = np.asarray(trace["scores"], dtype=np.float32).reshape(-1)
            values = values[lo:(int(hi) if hi is not None else None)]
            threshold = trace.get("threshold")
            flagged = trace.get("flagged")
            flagged = (np.zeros(values.size, np.int8) if flagged is None else
                       np.asarray(flagged, dtype=np.int8).reshape(-1)[lo:(int(hi) if hi is not None else None)])
            arrays[f"{signal}_scores"] = values
            arrays[f"{signal}_threshold"] = np.float64(np.nan if threshold is None else threshold)
            arrays[f"{signal}_flagged"] = flagged
        write_npz_atomic(self.window_path(key, window.index), **arrays)


def recorded_windows(directory: str, key: str) -> List[int]:
    """Indices of the windows of scenario `key` with a stream file."""
    try:
        names = os.listdir(os.path.join(directory, WINDOWS_DIR, key))
    except OSError:
        return []
    out = []
    for name in names:
        if name.startswith("w") and name.endswith(".npz"):
            try:
                out.append(int(name[1:-4]))
            except ValueError:
                pass
    return sorted(out)


def read_meta(directory: str) -> Optional[Dict[str, Any]]:
    try:
        with open(os.path.join(directory, META_FILE), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
