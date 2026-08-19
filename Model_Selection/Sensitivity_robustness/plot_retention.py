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


def prune_superseded(directory: str, prefix: str, keep_names) -> List[str]:
    """Delete `{prefix}*.png` in `directory` that this run did not write.

    A DIFFERENT retention problem from `prune_timestamped` above, and the two do
    not overlap. Those filenames carry a timestamp, so every run mints a new one
    and keeping the newest three is a choice. These carry the run's OUTCOME —
    `..._point_tree_{winner}_vs_{competitor}.png` — so a run whose outcome
    differs writes a new file *beside* the old one rather than over it, and the
    directory ends up describing two runs at once with no way to tell which is
    which. The winner-grouping in `WebUI.plots` catches a changed winner; it
    cannot catch a changed COMPETITOR, because those files sit in the same
    group.

    That is not hypothetical: renaming the four abbreviated families left
    `LUNAR_2_vs_AE_1.png` on disk beside the `LUNAR_2_vs_AutoEncoder_1.png`
    that replaced it, and the picker offered both.

    Called AFTER the new figures are written, with the names that were written,
    so a run that dies mid-plot leaves the previous set intact rather than
    deleting it and failing to replace it. Never raises, for the same reason
    `prune_timestamped` does not.
    """
    keep = {os.path.basename(n) for n in (keep_names or ())}
    removed: List[str] = []
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    for name in names:
        if not name.startswith(prefix) or not name.endswith(".png"):
            continue
        if name in keep:
            continue
        path = os.path.join(directory, name)
        try:
            os.remove(path)
            removed.append(path)
        except OSError as e:
            logger.warning(f"Could not prune {path}: {e}")
    if removed:
        logger.info(f"Removed {len(removed)} figure(s) in {directory} left by an "
                    f"earlier run with a different outcome.")
    return removed
