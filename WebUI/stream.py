"""
The live monitor's view of an online run: decimated slices of the stream files
the pipeline writes (Model_Selection/online_stream.py documents them).

A request names a range of positions and a point budget. A range that fits is
returned sample by sample; a longer one is cut into equal buckets, each giving
its minimum and maximum, so a one-sample spike survives any zoom. Labels never
go through the buckets: anomalies come back as exact [start, end) spans, so
decimation cannot hide one.

numpy only (no torch, no pandas); only the requested channels and range are
copied out of the arrays.
"""

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

META_FILE = "stream.json"
DATA_FILE = "stream.npz"
WINDOWS_DIR = "stream"
SIGNALS = ("single", "ensemble")
DEFAULT_POINTS = 2000
MAX_POINTS = 20000
MAX_CHANNELS = 16


def _clean(values) -> List[Optional[float]]:
    return [None if not math.isfinite(v) else round(float(v), 6) for v in np.asarray(values)]


def minmax(values, lo: int, hi: int, max_points: int) -> Dict[str, Any]:
    """Positions lo..hi-1 of a 1-D array in at most ~max_points points.

    {"x", "min", "max", "bucket"}: `x` is each bucket's first position, and
    with bucket 1 min and max are the samples themselves. NaN becomes None.
    """
    values = np.asarray(values, dtype=np.float64)[lo:hi]
    n = values.size
    if n == 0:
        return {"x": [], "min": [], "max": [], "bucket": 1}
    # Two numbers per bucket, so half the budget in buckets.
    bucket = max(1, math.ceil(n / max(1, int(max_points) // 2)))
    if bucket == 1:
        out = _clean(values)
        return {"x": list(range(lo, lo + n)), "min": out, "max": out, "bucket": 1}
    blocks = np.pad(values, (0, (-n) % bucket), constant_values=np.nan).reshape(-1, bucket)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # an all-NaN bucket
        mins, maxs = np.nanmin(blocks, axis=1), np.nanmax(blocks, axis=1)
    return {"x": list(range(lo, lo + blocks.shape[0] * bucket, bucket)),
            "min": _clean(mins), "max": _clean(maxs), "bucket": bucket}


def spans(labels, lo: int, hi: int, offset: int = 0) -> List[List[int]]:
    """[start, end) runs of nonzero labels overlapping lo..hi, clipped to it,
    in positions; `offset` is the position of labels[0]."""
    if labels is None:
        return []
    labels = np.asarray(labels).reshape(-1)
    a, b = max(lo - offset, 0), max(min(hi - offset, labels.size), 0)
    if b <= a:
        return []
    edges = np.diff(np.concatenate([[0], (labels[a:b] > 0).astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [[int(s + a + offset), int(e + a + offset)] for s, e in zip(starts, ends)]


def _int(value, name: str, default=None) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return default
    try:
        number = float(str(value).strip())
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if not number.is_integer():
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    return int(number)


def parse_channels(text, n_channels: int) -> List[int]:
    """'0,2' -> [0, 2]; 'all' -> the first MAX_CHANNELS. ValueError when out
    of range, so the API answers 400 rather than an empty chart."""
    if text is None or str(text).strip() == "":
        return [0]
    if str(text).strip().lower() == "all":
        return list(range(min(n_channels, MAX_CHANNELS)))
    out: List[int] = []
    for part in str(text).split(","):
        if not part.strip():
            continue
        c = _int(part, "channels")
        if not 0 <= c < n_channels:
            raise ValueError(f"channel {c} is out of range (0..{n_channels - 1})")
        if c not in out:
            out.append(c)
    if len(out) > MAX_CHANNELS:
        raise ValueError(f"at most {MAX_CHANNELS} channels per request")
    return out or [0]


def load_meta(directory: Optional[Path]) -> Optional[Dict[str, Any]]:
    if directory is None:
        return None
    try:
        with open(directory / META_FILE, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) and (directory / DATA_FILE).is_file() else None


def recorded(directory: Path, scenario: str) -> List[int]:
    """Window indices of `scenario` with a stream file, in order."""
    try:
        names = [p.name for p in (directory / WINDOWS_DIR / scenario).iterdir()]
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


def window_view(directory: Path, scenario: str, index: int, max_points: int):
    """One scored window of a scenario: each signal's scores decimated like
    the series, its threshold, and its flagged samples as exact spans. None
    when that window has not been written (yet)."""
    path = directory / WINDOWS_DIR / scenario / f"w{int(index):05d}.npz"
    try:
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
    except (OSError, ValueError):
        return None
    start, end = int(data["start"]), int(data["end"])
    out: Dict[str, Any] = {"i": int(index), "start": start, "end": end}
    for signal in SIGNALS:
        scores = data.get(f"{signal}_scores")
        if scores is None:
            out[signal] = None
            continue
        scores = np.asarray(scores, dtype=np.float64)
        flagged = np.asarray(data.get(f"{signal}_flagged", np.zeros(scores.size)))
        if 0 < scores.size != end - start:
            # A `speedup` copy's part of the window is resampled; it is drawn
            # over the window's own positions.
            idx = np.clip(np.round(np.linspace(0, scores.size - 1, end - start)).astype(int),
                          0, scores.size - 1)
            scores, flagged = scores[idx], flagged[idx]
        view = minmax(scores, 0, scores.size, max_points)
        view["x"] = [start + x for x in view["x"]]
        threshold = float(data.get(f"{signal}_threshold", np.nan))
        view["threshold"] = threshold if math.isfinite(threshold) else None
        view["flagged"] = spans(flagged, start, end, offset=start)
        view["n_flagged"] = int(np.sum(flagged > 0))
        out[signal] = view
    return out


def view(directory: Optional[Path], args: Dict[str, Any]) -> Dict[str, Any]:
    """The /stream response for the query `args`. Raises ValueError on a bad
    parameter and FileNotFoundError when there are no stream files.

    Parameters: channels ('0,2' | 'all'), from, to (positions, default the
    whole stream), max_points (default 2000), copy (an injected copy, 0),
    series ('0' leaves the series out), scenario and window (that scenario's
    window, default its latest written one).
    """
    meta = load_meta(directory)
    if meta is None:
        raise FileNotFoundError("no stream files")
    total = int(meta["total_length"])
    offline = int(meta["offline_size"])
    max_points = min(max(_int(args.get("max_points"), "max_points", DEFAULT_POINTS), 10),
                     MAX_POINTS)
    lo = max(0, _int(args.get("from"), "from", 0))
    hi = min(total, _int(args.get("to"), "to", total))
    if hi <= lo:
        raise ValueError("`to` must be greater than `from`")
    channels = parse_channels(args.get("channels"), int(meta["n_channels"]))
    copies = meta.get("copies") or [{"copy": 0, "anomaly_type": None}]
    copy = _int(args.get("copy"), "copy", 0)
    if not 0 <= copy < len(copies):
        raise ValueError(f"copy {copy} does not exist (0..{len(copies) - 1})")
    out: Dict[str, Any] = {"meta": meta, "range": {"from": lo, "to": hi},
                           "channels": channels, "copy": copy}
    if str(args.get("series", "1")) != "0":
        with np.load(directory / DATA_FILE) as z:
            offline_Y, online_Y = z["offline"], z["online"][copy]
            real = z["real_labels"] if "real_labels" in z.files else None
            injected = z["injected_labels"][copy] if "injected_labels" in z.files else None
        out["series"] = [{"channel": c, **minmax(np.concatenate([offline_Y[c], online_Y[c]]),
                                                 lo, hi, max_points)} for c in channels]
        out["real_spans"] = spans(real, lo, hi)
        kind = copies[copy].get("anomaly_type")
        out["injected_spans"] = [[s, e, kind] for s, e in spans(injected, lo, hi, offset=offline)]
    scenario = args.get("scenario")
    if scenario:
        written = recorded(directory, str(scenario))
        out["recorded_windows"] = written
        index = _int(args.get("window"), "window", written[-1] if written else None)
        out["window"] = (window_view(directory, str(scenario), index, max_points)
                         if index is not None else None)
    return out
