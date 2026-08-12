"""
Retention for the timestamped figures the robustness stages mint per run.

`off_by_threshold_testing` and `GAN_test` both save their evaluation figures
under a fresh `{stem}_{YYYY-MM-DD_HH-MM-SS}.png` on every run and never remove
the old ones, so a directory accumulates one copy per run forever — 450 files
and 300 MB across the tree here, of which the WebUI shows exactly one per stem
(`WebUI.plots.dedupe_timestamped` collapses each group to its newest and reports
the rest only as a count).

Rather than overwrite in place, the newest few are kept. Overwriting would make
that "n older" machinery dead and throw away the only means of comparing one
run's injected points against another's; keeping three caps the growth while
leaving both intact.

The grouping key is deliberately the same `(stem, tail)` pair
`WebUI.plots.TS_RE` uses, so what this prunes and what the page hides are the
same set. Zero-padded timestamps mean lexicographic order is chronological,
which beats mtime — a copy would destroy that.
"""

from __future__ import annotations

import os
import re
from typing import List

from loguru import logger

KEEP_NEWEST = 3

_TS_RE = re.compile(
    r"^(?P<stem>.+?)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?P<tail>_?)\.png$")


def prune_timestamped(directory: str, keep: int = KEEP_NEWEST) -> List[str]:
    """Delete all but the `keep` newest figures in each timestamped group.

    Returns the paths removed. Never raises: this runs immediately after a
    figure is saved, and losing a run's output because an old file could not be
    unlinked would be a far worse failure than the disk it was meant to save.
    """
    if keep < 1:
        return []
    groups = {}
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    for name in names:
        m = _TS_RE.match(name)
        if m:
            groups.setdefault((m.group("stem"), m.group("tail")), []).append(
                (m.group("ts"), name))

    removed: List[str] = []
    for _key, entries in groups.items():
        if len(entries) <= keep:
            continue
        entries.sort(key=lambda pair: pair[0])
        for _ts, name in entries[:-keep]:
            path = os.path.join(directory, name)
            try:
                os.remove(path)
                removed.append(path)
            except OSError as e:
                logger.warning(f"Could not prune {path}: {e}")
    if removed:
        logger.info(f"Pruned {len(removed)} superseded figure(s) in {directory}, "
                    f"keeping the newest {keep} of each.")
    return removed
