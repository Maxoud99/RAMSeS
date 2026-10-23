"""
The online phase: replays the online windows under each scenario, switches in
re-optimisations at fixed windows, and writes what happened for the web UI.

Model-free on purpose. Scoring a window and starting a re-optimisation are
handed in as callables (`score_window`, `start_reopt`), so the scheduling —
when a re-optimisation starts, when its result is switched in, what is
suppressed, what the stream waits for — can be tested with fakes, and app.py
keeps only the wiring.

How one scenario runs (`run_scenario`)
--------------------------------------
For every window, in order:

1. If a re-optimisation is due at this window (it was triggered at window t
   and this is window t + L, --reopt_delay), it is switched in now: this window
   is the first one the new selection scores, and the trigger policy's
   baselines and histories restart from it. If it has not finished, the stream
   WAITS for it here (recorded: a `waiting` event, the scenario's status
   `waiting_for_reopt`, and `waited_s` on the window). One that failed or ran
   out of time is recorded as failed and the current selection stays.
2. The deployed selection scores the window (ensemble fitness and single-model
   score, both on the run's decision metric), on the window's labels: the
   test series' TRUE labels with --online_labels real (nothing is injected
   online), the injected ones otherwise. A window whose labels are one class
   has no score (null) — drawn as a gap, never a trigger, never part of the
   adaptive history. In app.py the scorer remembers
   each detector's scores per window, so every scenario that deploys a
   detector on a window reads the same numbers (online_scoring.WindowScorer).
3. The scenario's policy (`online_policy.TriggerPolicy`) judges the scores.
   A trigger starts a re-optimisation in its own child process unless one is
   pending, in which case it is suppressed — recorded, never queued. A
   re-optimisation is pending from its trigger up to its switch-in window,
   whether it finishes early, fails or times out meanwhile, so the timeline
   is the same on a fast machine and a slow one (online_policy.ReoptSchedule).
   With L = 0 the result is switched in at the triggering window itself, which
   is then scored again with the new selection (`rescored`).

A re-optimisation due after the last window is never switched in: at the end
of the stream its process is killed and it is recorded as `not_applied`
(reason `stream_ended`). Nothing is waited for after the last window. If the
scenario itself fails while one is pending, it is killed and recorded as
`aborted`, so every `started` has exactly one closing event (`applied`,
`failed`, `not_applied` or `aborted`).

Results files
-------------
Written under `results/online/<dataset>/<entity>/` after EVERY window, and
every `HEARTBEAT_SECONDS` while the stream waits, each as a temporary file
renamed into place, so a reader polling them never sees a half-written
document and can tell a waiting run (`updated_at` moves) from a dead one.
Every time is seconds since the epoch; every score is a float in [0, 1] or
null when it could not be computed.

Beside them, the live monitor's stream files (stream.json, stream.npz and
stream/<scenario>/w<index>.npz: the replayed series, its real and injected
labels, and per window the deployed detectors' scores, threshold and flagged
samples) are written through the `on_window` hook; their schema is in
Model_Selection/online_stream.py.

summary.json ::

    {
      "schema_version": 2,
      "dataset": "anomaly_archive", "dataset_label": "UCR", "entity": "...",
      "generated_at": 1760000000.0,      # when this online phase started
      "updated_at":   1760000123.4,      # last write (also refreshed while waiting)
      "pid": 12345, "host": "node1",     # the process writing these files
      "status": "running" | "complete" | "failed" | "skipped" | "interrupted",
                                         # "interrupted": the process died before it
                                         # could finish this file (run timeout,
                                         # cancel, crash); written by the web UI
      "error": null | "message",         # why it failed, was skipped or interrupted
      "strategy": "all" | "<scenario key>",
      "scenario_order": ["fixed", "periodic", "threshold", "adaptive_threshold"],
      "decision_metric": {"spec": "f1,pr_auc", "label": "F1 + PR-AUC",
                          "formula": "0.5 * F1 + 0.5 * PR-AUC"},
      "window_size": 100, "step_size": 5,
      "n_windows": 20,                   # windows each scenario processes
      "n_windows_available": 28,         # in the online stream, before --max_online_windows
      "first_window_index": 0,           # window 0, the first online window, is replayed
      "total_samples": 1180, "offline_size": 944, "online_size": 236,
      "online_samples": 236,             # length of the replayed online stream: 8x
                                         # online_size with --inject_online_regime
      "window_coordinates": "test_series" | "online_stream",
                                         # what start/end of a window count in (below)
      "inject_online_regime": false,
      "label_source": "real" | "injected",
                                         # what the windows are scored on
                                         # (--online_labels): "real" = the test
                                         # series' own labels, nothing injected
                                         # online (`injection` is null), and a
                                         # re-optimisation selects on the real
                                         # labels seen so far; "injected" = the
                                         # synthetic anomalies below
      "label_source_label": "Real labels — no injection online",
      "n_windows_scorable": 9,           # windows whose labels hold both classes
                                         # (of n_windows); the rest have no score
      "n_windows_with_anomaly": 9,       # windows holding a (true or injected)
                                         # anomalous label
      "anomaly_type": "spikes", "anomaly_rate": null,
                                         # the offline phase's injection; repeated
                                         # online only when label_source is "injected"
      "injection": null | {"seed": 13, "n_copies": 1,
                    "copies": [{"copy": 0, "anomaly_type": "spikes", "length": 236}]},
                                         # null with real labels; else one entry per
                                         # injected copy; a multi-copy
                                         # type adds its varying parameter, e.g.
                                         # {"copy": 2, "anomaly_type": "scale",
                                         #  "amplitude_scaling": 2, "length": 236}
      "seed": 13,                        # the run's seed (injection, scoring, re-opts)
      "vus_window": null | 37,           # the one VUS window all windows use, if needed
      "reopt_delay": 2,                  # L
      "reopt_timeout": 3600.0 | null,    # seconds one re-optimisation may run
      "feasibility": {"ok": true, "n_windows": 28, "window_size": 100, "step_size": 5,
                      "online_size": 236, "offline_size": 944, "total_samples": 1180,
                      "iteration": 5, "min_window_size": 100, "regime_n_windows": 359,
                      "label_source": "real", "n_windows_with_anomaly": 9,
                      "n_windows_scorable": 9, "n_anomalous_online": 12,
                                         # the three counts are null when injected;
                                         # with real labels an online part with no
                                         # scorable window is not ok ("No labelled
                                         # anomaly in the online part — nothing to
                                         # score with real labels.")
                      "reason": null},   # Utils.pipeline_spec.online_feasibility
      "offline_selection": {
        "framework_choice": "ensemble" | "single_model",
        "decision_formula": "...",       # what the offline scores below used
        "ensemble": {"members": [...], "meta_model_type": "rf",
                     "meta_model_source": "ga" | "refit",
                     "scores": {"f1": .., "pr_auc": .., "vus": .., "score": ..}},
        "single":   {"model": "LOF_1",
                     "scores": {"f1": .., "pr_auc": .., "vus": .., "score": ..}},
        "comparable_with_windows": false,
        "comparability_note": "..."      # why the offline scores are not on the
                                         # window scores' scale
      },
      "scenarios": [                     # one per scenario, in scenario_order, with
        {"key": "periodic",              # every field from the start (zero counts,
         "label": "Periodic re-optimization",     # null aggregates while pending)
         "params": {"update_interval": 5, "reopt_delay": 2},
         "status": "pending" | "running" | "waiting_for_reopt" | "complete"
                   | "failed" | "skipped" | "interrupted",
         "error": null | "message",
         "waiting": null | {"since_window": 5, "until_window": 7, "since": 1760000040.0},
                                         # while the stream waits at window 7 for the
                                         # re-optimisation triggered at window 5
         "n_windows": 12, "n_windows_planned": 20,
         "n_windows_scorable": 5,        # of those, windows whose labels hold both classes
         "n_windows_scored": 5,          # windows with at least one non-null score
         "n_triggers": 3,                # windows whose rule fired (incl. suppressed)
         "n_reopts_started": 2, "n_reopts_applied": 1, "n_suppressed": 1,
         "n_reopts_failed": 0,           # incl. timed out
         "n_reopts_timed_out": 0,
         "n_reopts_not_applied": 1,      # due after the last window, killed
         "n_reopts_aborted": 0,          # killed because the scenario failed
         "n_reopts_reused": 0,           # result reused from an earlier scenario
         "n_waits": 1, "total_wait_s": 212.4,
         "mean_ensemble_score": .., "min_ensemble_score": ..,
         "mean_single_score": .., "min_single_score": ..,
         "total_reopt_seconds": 384.2,   # run time of resolved re-optimisations
         "stream_time_s": 40.1,          # wall time minus waits
         "wall_time_s": 252.5,
         "final_ensemble": [...], "final_single": "...",
         "started_at": .., "finished_at": ..}
      ]
    }

<scenario>.json (fixed.json, periodic.json, threshold.json,
adaptive_threshold.json; fixed_random.json for the CLI baseline) ::

    {
      "schema_version": 2, "key": "threshold", "label": "...",
      "params": {"drop_pct": 20.0, "reopt_delay": 2}, "status": "...",
      "error": null, "waiting": null | {...as in the summary row},
      "updated_at": 1760000060.0,
      "n_windows_planned": 20,
      "windows": [
        {"i": 7,                         # window index; 0 is the first online window
         "start": 979, "end": 1079,      # [start, end) = offline_size + online_start ...
         "online_start": 35, "online_end": 135,
                                         # positions in the online stream; start/end are
                                         # test-series samples when window_coordinates is
                                         # "test_series", and run past the test series with
                                         # --inject_online_regime ("online_stream")
         "segments": [{"copy": 0, "data_start": 0, "data_end": 100,
                       "copy_start": 35, "copy_end": 135}],
                                         # where each injected copy's slice of the SAME
                                         # time positions lies in the scored window data
                                         # (data_*) and in that copy (copy_*); one entry
                                         # per summary.injection.copies
         "data_digest": "3f2a...",       # sha256 (16 hex) of the window's data and labels:
                                         # equal in every scenario for the same i
         "n_samples": 100, "n_anomalous": 41,   # scored samples, anomalous labels
                                         # (the TRUE labels with label_source "real")
         "scorable": true,               # 0 < n_anomalous < n_samples: F1 and PR-AUC
                                         # are defined; false = both scores null, a gap
         "t": 1760000050.2,              # when it was scored
         "ensemble_score": 0.71, "single_score": 0.64,
         "metrics": {"ensemble": {"f1": .., "pr_auc": .., "vus": ..},
                     "single":   {"f1": .., "pr_auc": .., "vus": ..}},
         "thresholds": {"ensemble": 0.60, "single": null},
         "threshold_modes": {"ensemble": "relative", "single": "relative"},
         "threshold_mode": "relative" | "adaptive" | "periodic" | null,
         "baselines": {"ensemble": 0.75, "single": 0.64},
         "below": ["ensemble"],          # signals under their threshold
         "trigger": null | {"reason": "periodic" | "relative_drop" | "adaptive_drop",
                            "text": "ensemble 0.5512 < 0.6000 (relative)",
                            "signals": ["ensemble"],
                            "scores": {"ensemble": 0.55, "single": 0.64},
                                         # what the rule judged (before a re-score)
                            "started": true,
                            "apply_window": 9 | null,     # where it will be switched in
                            "suppressed_because": null | "reoptimization_in_progress"},
         "reopt_pending": null | {"started_at_window": 5, "apply_window": 7},
                                         # pending while this window scored
         "reopt_running": false,         # = reopt_pending is not null
         "reopt_applied": null | {"started_at_window": 5, "apply_window": 7,
                                  "duration_s": 61.2, "waited_s": 20.4,
                                  "new_ensemble": [...], "new_single": "...",
                                  "reused_from": null | "periodic"},
         "reopt_failed": null | {"started_at_window": 5, "apply_window": 7,
                                 "error": "...", "timed_out": false},
         "waited_s": 20.4,               # the stream waited this long before scoring it
         "rescored": false,              # reopt_delay 0: re-scored after a switch-in
         "deployed_ensemble": [...], "deployed_single": "LOF_1",
         "misclassified": {"ensemble": 3, "single": 5},
         "errors": {}                    # signal -> why its score is null (an
                                         # unscorable window: "...all one class")
        }, ...
      ],
      "events": [                        # in order of occurrence; each has type and t
        {"type": "trigger", "window": 7, "reason": .., "signals": [..],
         "started": true, "suppressed_because": null},
        {"type": "started", "window": 7, "apply_window": 9, "reused_from": null},
        {"type": "waiting", "window": 9, "started_at_window": 7, "apply_window": 9},
        {"type": "applied", "window": 9, "started_at_window": 7, "apply_window": 9,
         "duration_s": 61.2, "waited_s": 20.4, "new_ensemble": [...],
         "new_single": "...", "reused_from": null},
        {"type": "failed", "window": 9, "started_at_window": 7, "apply_window": 9,
         "duration_s": .., "waited_s": .., "error": "...", "timed_out": true},
        {"type": "not_applied", "window": null, "started_at_window": 18,
         "apply_window": 20, "reason": "stream_ended", "completed": false,
         "duration_s": 12.0},            # due after the last window; killed
        {"type": "aborted", "window": 12, "started_at_window": 11,
         "apply_window": 13, "error": "..."}     # window: where the scenario failed
      ]
    }

`thresholds` holds the threshold IN EFFECT for the window (computed from the
scores before it); it is null for fixed and periodic, and on the window that
sets a signal's baseline. `trigger` is null unless the scenario's rule fired.

Log lines
---------
One parsable line per window per scenario, plus phase, scenario and
re-optimisation lines, all `KEY: field=value, field=value` with `+` joining
list members (WebUI/markers.py splits on commas; a trailing error= or reason=
is taken verbatim):

    ONLINE PHASE START: strategy=all, scenarios=fixed+periodic+..., windows=20,
                        ..., scorable=9, labels=real, reopt_delay=2
    ONLINE SCENARIO START: scenario=periodic, position=2/4, windows=20, ...
    ONLINE WINDOW: scenario=periodic, window=3/20, index=3, ensemble=0.7123, ...
    ONLINE REOPT: scenario=periodic, event=started|suppressed|waiting|applied|
                  failed|not_applied|aborted, window=5, ...
    ONLINE SCENARIO COMPLETE: scenario=periodic, windows=20, triggers=4, ...
    ONLINE SCENARIO FAILED: scenario=periodic, error=...
    ONLINE PHASE COMPLETE: scenarios=4, wall_s=...
    ONLINE PHASE FAILED: error=...
    ONLINE PHASE SKIPPED: reason=...
"""

import json
import logging
import math
import os
import socket
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from Model_Selection.online_policy import (PolicyParams, ReoptSchedule, SIGNALS,
                                           TriggerPolicy)
from Utils.pipeline_spec import (ONLINE_BASELINE_SCENARIOS, ONLINE_SCENARIO_LABELS,
                                 ONLINE_SCENARIOS)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
SUMMARY_FILE = "summary.json"

# While the stream waits for a re-optimisation, the files are rewritten this
# often so `updated_at` keeps moving: a live reader can then tell a run that is
# waiting from one whose process died.
HEARTBEAT_SECONDS = 15.0

STATUS_WAITING = "waiting_for_reopt"


# ── Data handed in by the caller ─────────────────────────────────────────────

@dataclass
class OnlineWindow:
    """One online window as every scenario sees it."""
    index: int                 # position in the online window sequence (0 first)
    start: int                 # first sample: offline_size + online_start
    end: int                   # one past the last sample
    data: Any = None           # what the deployed models score (injected)
    online_start: Optional[int] = None    # position in the online stream
    online_end: Optional[int] = None
    segments: List[Dict[str, int]] = field(default_factory=list)
    # A fingerprint of what is scored, so a reader can check from the results
    # files alone that every scenario saw the same window: a digest of the
    # data and labels, the number of samples and of anomalous labels.
    digest: Optional[str] = None
    n_samples: Optional[int] = None
    n_anomalous: Optional[int] = None

    @property
    def scorable(self) -> Optional[bool]:
        """Do the window's labels hold both classes, so its F1 and PR-AUC are
        defined? None when the counts are unknown. Decided by the labels
        alone, so it is the same in every scenario."""
        if self.n_samples is None or self.n_anomalous is None:
            return None
        return 0 < int(self.n_anomalous) < int(self.n_samples)


@dataclass
class Selection:
    """What is deployed: an ensemble (with its fitted meta-model) and a single
    model. Treated as immutable once built; a re-optimisation returns a new one.

    `ensemble` is in the meta-model's column order (the GA's pool order)."""
    ensemble: List[str]
    single: Optional[str]
    meta_model_type: Optional[str] = None
    meta_model: Any = None
    origin: str = "offline"                 # offline | reoptimization | random
    origin_window: Optional[int] = None
    meta_model_source: Optional[str] = None  # ga (the GA's own object) | refit


@dataclass
class WindowScores:
    ensemble: Optional[float] = None
    single: Optional[float] = None
    metrics: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    misclassified: Dict[str, Optional[int]] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    # signal -> {"scores": per-sample anomaly scores of the window's data,
    # "threshold": the score they were cut at (None when unscorable),
    # "flagged": per-sample 0/1 predictions (None when unscorable)}. Not
    # part of the window record; the live monitor's stream files keep it.
    trace: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ReoptFailed(Exception):
    """A re-optimisation ended without a selection."""


class ReoptTimedOut(ReoptFailed):
    """A re-optimisation outlived its time limit and was killed."""


# What `start_reopt` returns: a handle on one running re-optimisation.
# online_reopt.ProcessReopt is the real one; the tests use fakes.
#
#   done() -> bool          non-blocking; True once it finished, failed, timed
#                           out (the handle enforces its own deadline) or was
#                           killed
#   wait(seconds) -> bool   block for up to `seconds`; returns done()
#   result() -> Selection   once done; raises ReoptFailed / ReoptTimedOut
#   kill()                  stop it now; a no-op once done
#   close()                 release what it holds (files, the process)
#   duration_s              seconds the computation ran, or None
#   reused_from             the scenario whose identical result it reuses, or None


# ── JSON helpers ─────────────────────────────────────────────────────────────

def jsonable(value):
    """`value` with numpy scalars unwrapped and non-finite floats as None.

    JSON has no NaN; Python's encoder would write the bare token `NaN`, which
    the browser's JSON.parse rejects, and the whole file would be unreadable
    because one window had no score.
    """
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, bytes)):
        return value if isinstance(value, str) else value.decode("utf-8", "replace")
    if hasattr(value, "tolist"):            # numpy arrays and scalars alike
        return jsonable(value.tolist())
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def write_json_atomic(path: str, payload) -> None:
    """Write `payload` to `path` via a temporary file and a rename."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(jsonable(payload), f, indent=1, allow_nan=False)
        # mkstemp creates owner-only files; results are as readable as the
        # rest of the results tree.
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _fmt(value, digits=4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{v:.{digits}f}" if math.isfinite(v) else "-"


def _members(names) -> str:
    return "+".join(names) if names else "-"


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def _min(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return min(vals) if vals else None


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), digits)


# ── Results writer ───────────────────────────────────────────────────────────

COUNT_FIELDS = ("n_triggers", "n_reopts_started", "n_reopts_applied", "n_suppressed",
                "n_reopts_failed", "n_reopts_timed_out", "n_reopts_not_applied",
                "n_reopts_aborted", "n_reopts_reused", "n_waits")


def scenario_row(key: str, params: Dict[str, Any], n_windows_planned: Optional[int],
                 status: str = "pending") -> Dict[str, Any]:
    """A summary row with every field, before the scenario has run.

    A pending row has the same shape as a finished one, so a reader can label
    and lay out every scenario from the first poll on.
    """
    return {
        "key": key,
        "label": ONLINE_SCENARIO_LABELS.get(key, key),
        "params": dict(params or {}),
        "status": status,
        "error": None,
        "waiting": None,
        "n_windows": 0,
        "n_windows_scorable": 0,
        "n_windows_scored": 0,
        "n_windows_planned": n_windows_planned,
        **{name: 0 for name in COUNT_FIELDS},
        "total_wait_s": 0.0,
        "mean_ensemble_score": None, "min_ensemble_score": None,
        "mean_single_score": None, "min_single_score": None,
        "total_reopt_seconds": 0.0,
        "stream_time_s": None,
        "wall_time_s": None,
        "final_ensemble": None, "final_single": None,
        "started_at": None, "finished_at": None,
    }


class OnlineResultsWriter:
    """summary.json plus one <scenario>.json per scenario, in one directory.

    All calls come from the stream's thread; the re-optimisations run in other
    processes and never write here.
    """

    def __init__(self, directory: str, header: Dict[str, Any],
                 scenario_keys: Sequence[str],
                 scenario_params: Optional[Dict[str, Dict[str, Any]]] = None,
                 n_windows_planned: Optional[int] = None):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        # Files of scenarios this run does not replay belong to an earlier run;
        # left in place they would read as part of this one.
        for key in tuple(ONLINE_SCENARIOS) + tuple(ONLINE_BASELINE_SCENARIOS):
            try:
                os.unlink(os.path.join(directory, f"{key}.json"))
            except OSError:
                pass
        params = scenario_params or {}
        planned = header.get("n_windows") if n_windows_planned is None else n_windows_planned
        now = time.time()
        self.summary: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            **header,
            "generated_at": now,
            "updated_at": now,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "status": "running",
            "error": None,
            "scenario_order": list(scenario_keys),
            "scenarios": [scenario_row(k, params.get(k, {}), planned)
                          for k in scenario_keys],
        }
        self.write_summary()

    @property
    def summary_path(self) -> str:
        return os.path.join(self.directory, SUMMARY_FILE)

    def scenario_path(self, key: str) -> str:
        return os.path.join(self.directory, f"{key}.json")

    def write_summary(self) -> None:
        self.summary["updated_at"] = time.time()
        write_json_atomic(self.summary_path, self.summary)

    def update_scenario(self, stats: Dict[str, Any]) -> None:
        for i, entry in enumerate(self.summary["scenarios"]):
            if entry.get("key") == stats["key"]:
                self.summary["scenarios"][i] = stats
                break
        else:
            self.summary["scenarios"].append(stats)
        self.write_summary()

    def finish(self, status: str, error: Optional[str] = None) -> None:
        self.summary["status"] = status
        self.summary["error"] = error
        # A scenario the phase never reached would otherwise read as still
        # about to run, forever.
        if status in ("skipped", "failed"):
            for row in self.summary["scenarios"]:
                if row.get("status") == "pending":
                    row["status"] = status
                    row["error"] = row.get("error") or error
        self.write_summary()

    def recorder(self, key: str, params: Dict[str, Any],
                 n_windows_planned: int) -> "ScenarioRecorder":
        return ScenarioRecorder(self, key, params, n_windows_planned)


class ScenarioRecorder:
    """Accumulates one scenario's windows and events and keeps its file current."""

    def __init__(self, writer: OnlineResultsWriter, key: str,
                 params: Dict[str, Any], n_windows_planned: int):
        self.writer = writer
        self.key = key
        self.doc: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "key": key,
            "label": ONLINE_SCENARIO_LABELS.get(key, key),
            "params": dict(params),
            "status": "running",
            "error": None,
            "waiting": None,
            "updated_at": time.time(),
            "n_windows_planned": n_windows_planned,
            "windows": [],
            "events": [],
        }
        self.counts = {name: 0 for name in COUNT_FIELDS}
        self.total_reopt_seconds = 0.0
        self.total_wait_s = 0.0
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.final: Dict[str, Any] = {"ensemble": None, "single": None}

    # -- recording -----------------------------------------------------------

    def event(self, kind: str, **fields) -> Dict[str, Any]:
        entry = {"type": kind, "t": time.time(), **fields}
        self.doc["events"].append(entry)
        return entry

    def reopt_finished(self, duration_s: Optional[float]) -> None:
        if duration_s is not None:
            self.total_reopt_seconds += float(duration_s)

    def add_window(self, record: Dict[str, Any]) -> None:
        self.doc["windows"].append(record)
        self.final = {"ensemble": record.get("deployed_ensemble"),
                      "single": record.get("deployed_single")}
        self.flush()

    def set_waiting(self, waiting: Optional[Dict[str, Any]]) -> None:
        """Enter (a dict) or leave (None) the waiting state, and write it."""
        self.doc["waiting"] = waiting
        self.doc["status"] = STATUS_WAITING if waiting else "running"
        self.flush()

    def heartbeat(self) -> None:
        """Rewrite both files unchanged but for their time stamps."""
        self.flush()

    def flush(self) -> None:
        self.doc["updated_at"] = time.time()
        write_json_atomic(self.writer.scenario_path(self.key), self.doc)
        self.writer.update_scenario(self.stats())

    def finish(self, status: str, error: Optional[str] = None) -> None:
        self.doc["status"] = status
        self.doc["error"] = error
        self.doc["waiting"] = None
        self.finished_at = time.time()
        self.flush()

    # -- the summary row -----------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        windows = self.doc["windows"]
        ens = [w.get("ensemble_score") for w in windows]
        single = [w.get("single_score") for w in windows]
        scorable = [w for w in windows if w.get("scorable")]
        end = self.finished_at or time.time()
        wall = end - self.started_at
        row = scenario_row(self.key, self.doc["params"], self.doc["n_windows_planned"],
                           status=self.doc["status"])
        row.update({
            "error": self.doc["error"],
            "waiting": self.doc["waiting"],
            "n_windows": len(windows),
            "n_windows_scorable": len(scorable),
            "n_windows_scored": sum(1 for w in windows if w.get("ensemble_score") is not None
                                    or w.get("single_score") is not None),
            **self.counts,
            "total_wait_s": round(self.total_wait_s, 3),
            "mean_ensemble_score": _mean(ens),
            "min_ensemble_score": _min(ens),
            "mean_single_score": _mean(single),
            "min_single_score": _min(single),
            "total_reopt_seconds": round(self.total_reopt_seconds, 3),
            "stream_time_s": round(max(wall - self.total_wait_s, 0.0), 3),
            "wall_time_s": round(wall, 3),
            "final_ensemble": self.final["ensemble"],
            "final_single": self.final["single"],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        })
        return row


# ── The scenario runner ──────────────────────────────────────────────────────

def _error_text(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())


def _signal_scores(scores: WindowScores) -> Dict[str, Optional[float]]:
    return {"ensemble": scores.ensemble, "single": scores.single}


def run_scenario(key: str,
                 params: PolicyParams,
                 windows: Sequence[OnlineWindow],
                 selection: Selection,
                 score_window: Callable[[OnlineWindow, Selection], WindowScores],
                 start_reopt: Optional[Callable[[OnlineWindow, Selection, str], Any]],
                 recorder: ScenarioRecorder,
                 position: str = "1/1",
                 log: Optional[logging.Logger] = None,
                 heartbeat_s: float = HEARTBEAT_SECONDS,
                 on_window: Optional[Callable[[str, OnlineWindow, WindowScores], None]] = None
                 ) -> Dict[str, Any]:
    """Replay `windows` under scenario `key`, starting from `selection`.

    `on_window(key, window, scores)` is told about every window once its
    final scores are recorded (the live monitor's stream files); a failure
    there is logged once and never stops the scenario.

    `start_reopt(window, selection, key)` starts one re-optimisation from what
    has been seen up to the end of `window` and returns its handle (see
    ReoptFailed above for the protocol); it runs on THIS thread. Returns the
    scenario's summary row.
    """
    log = log or logger
    policy = TriggerPolicy(key, params)
    schedule = ReoptSchedule(params.reopt_delay)
    total = len(windows)
    handle = None
    log.info(f"ONLINE SCENARIO START: scenario={key}, position={position}, "
             f"windows={total}, label={recorder.doc['label']}, "
             f"params={' '.join(f'{k}={v}' for k, v in policy.params_dict().items()) or '-'}")

    def wait_for(h, pending, at_window) -> float:
        """Block until `h` is done, keeping the files' time stamps moving."""
        info = {"since_window": pending.started_at_window,
                "until_window": pending.apply_window, "since": time.time()}
        recorder.event("waiting", window=at_window,
                       started_at_window=pending.started_at_window,
                       apply_window=pending.apply_window)
        recorder.set_waiting(info)
        log.info(f"ONLINE REOPT: scenario={key}, event=waiting, window={at_window}, "
                 f"started_at_window={pending.started_at_window}")
        began = time.time()
        while not h.wait(heartbeat_s):
            recorder.heartbeat()
        waited = time.time() - began
        recorder.counts["n_waits"] += 1
        recorder.total_wait_s += waited
        recorder.set_waiting(None)
        return waited

    def switch_in(at_window: int):
        """Resolve the due re-optimisation at `at_window`: wait for it if it
        is not done, then switch it in or record why it could not be.
        Returns (applied info or None, failure info or None, seconds waited)."""
        nonlocal handle, selection
        pending = schedule.resolve()
        h, handle = handle, None
        waited = 0.0 if h.done() else wait_for(h, pending, at_window)
        base = {"started_at_window": pending.started_at_window,
                "apply_window": pending.apply_window}
        try:
            try:
                new_selection = h.result()
            except Exception as exc:
                duration = h.duration_s
                recorder.reopt_finished(duration)
                timed_out = isinstance(exc, ReoptTimedOut)
                recorder.counts["n_reopts_failed"] += 1
                if timed_out:
                    recorder.counts["n_reopts_timed_out"] += 1
                failure = {**base, "error": _error_text(exc), "timed_out": timed_out}
                recorder.event("failed", window=at_window, **failure,
                               duration_s=_round(duration), waited_s=round(waited, 3))
                log.error(f"ONLINE REOPT: scenario={key}, event=failed, window={at_window}, "
                          f"started_at_window={pending.started_at_window}, "
                          f"timed_out={'yes' if timed_out else 'no'}, "
                          f"error={_error_text(exc)}")
                detail = getattr(exc, "detail", None)
                if detail:
                    log.error(f"Re-optimisation traceback (scenario {key}, started at "
                              f"window {pending.started_at_window}); keeping the current "
                              f"selection:\n{detail}")
                return None, failure, waited
        finally:
            h.close()
        duration = h.duration_s
        recorder.reopt_finished(duration)
        selection = new_selection
        policy.reset()
        recorder.counts["n_reopts_applied"] += 1
        info = {**base, "duration_s": _round(duration), "waited_s": round(waited, 3),
                "new_ensemble": list(new_selection.ensemble),
                "new_single": new_selection.single,
                "reused_from": getattr(h, "reused_from", None)}
        recorder.event("applied", window=at_window, **info)
        log.info(f"ONLINE REOPT: scenario={key}, event=applied, window={at_window}, "
                 f"started_at_window={pending.started_at_window}, "
                 f"duration_s={_fmt(duration, 1)}, waited_s={waited:.1f}, "
                 f"ensemble={_members(new_selection.ensemble)}, "
                 f"single={new_selection.single}")
        return info, None, waited

    current = None
    stream_failed = False
    try:
        for n, window in enumerate(windows, 1):
            current = window.index
            applied = failed = None
            waited = 0.0
            if handle is not None:
                handle.done()             # enforces its deadline before it is due
            if schedule.due(window.index) is not None:
                applied, failed, waited = switch_in(window.index)

            scores = score_window(window, selection)
            pending_now = schedule.pending
            decision = policy.decide(window.index, _signal_scores(scores),
                                     pending=pending_now is not None)
            trigger = None
            rescored = False
            baselines = decision.baselines
            if decision.fire:
                recorder.counts["n_triggers"] += 1
                trigger = {"reason": decision.reason, "text": decision.text,
                           "signals": list(decision.below),
                           "scores": _signal_scores(scores),
                           "started": False, "apply_window": None,
                           "suppressed_because": decision.suppressed_because}
                trigger_event = recorder.event(
                    "trigger", window=window.index, reason=decision.reason,
                    signals=list(decision.below), started=decision.start,
                    suppressed_because=decision.suppressed_because)
                if decision.start:
                    pending = schedule.start(window.index)
                    try:
                        handle = start_reopt(window, selection, key)
                    except Exception as exc:
                        # Nothing runs, so nothing is pending: the next trigger
                        # may try again.
                        schedule.resolve()
                        trigger_event["started"] = False
                        recorder.counts["n_reopts_failed"] += 1
                        recorder.event("failed", window=window.index,
                                       started_at_window=window.index,
                                       apply_window=pending.apply_window,
                                       duration_s=None, waited_s=0.0,
                                       error=_error_text(exc), timed_out=False)
                        log.error(f"ONLINE REOPT: scenario={key}, event=failed, "
                                  f"window={window.index}, "
                                  f"started_at_window={window.index}, timed_out=no, "
                                  f"error=could not start: {_error_text(exc)}")
                    else:
                        trigger["started"] = True
                        trigger["apply_window"] = pending.apply_window
                        recorder.counts["n_reopts_started"] += 1
                        reused = getattr(handle, "reused_from", None)
                        if reused:
                            recorder.counts["n_reopts_reused"] += 1
                        recorder.event("started", window=window.index,
                                       apply_window=pending.apply_window,
                                       reused_from=reused)
                        log.info(f"ONLINE REOPT: scenario={key}, event=started, "
                                 f"window={window.index}, "
                                 f"apply_window={pending.apply_window}, "
                                 f"reused={reused or '-'}, reason={decision.reason}")
                        if schedule.due(window.index) is not None:      # L == 0
                            applied, failed, extra = switch_in(window.index)
                            waited += extra
                            if applied is not None:
                                scores = score_window(window, selection)
                                policy.observe(_signal_scores(scores))
                                baselines = policy.baselines()
                                rescored = True
                else:
                    recorder.counts["n_suppressed"] += 1
                    # `reason` last: the run page reads a trailing reason/error
                    # verbatim, commas and all.
                    log.info(f"ONLINE REOPT: scenario={key}, event=suppressed, "
                             f"window={window.index}, "
                             f"because={decision.suppressed_because}, "
                             f"reason={decision.reason}")

            recorder.add_window({
                "i": window.index,
                "start": window.start,
                "end": window.end,
                "online_start": window.online_start,
                "online_end": window.online_end,
                "segments": window.segments,
                "data_digest": window.digest,
                "n_samples": window.n_samples,
                "n_anomalous": window.n_anomalous,
                "scorable": window.scorable,
                "t": time.time(),
                "ensemble_score": scores.ensemble,
                "single_score": scores.single,
                "metrics": scores.metrics,
                "thresholds": decision.thresholds,
                "threshold_modes": decision.modes,
                "threshold_mode": decision.mode,
                "baselines": baselines,
                "below": list(decision.below),
                "trigger": trigger,
                "reopt_pending": ({"started_at_window": pending_now.started_at_window,
                                   "apply_window": pending_now.apply_window}
                                  if pending_now is not None else None),
                "reopt_running": pending_now is not None,
                "reopt_applied": applied,
                "reopt_failed": failed,
                "waited_s": round(waited, 3),
                "rescored": rescored,
                "deployed_ensemble": list(selection.ensemble),
                "deployed_single": selection.single,
                "misclassified": scores.misclassified,
                "errors": scores.errors,
            })
            if on_window is not None and not stream_failed:
                try:
                    on_window(key, window, scores)
                except Exception as exc:
                    stream_failed = True
                    log.warning(f"Live-monitor stream files for scenario {key} stopped at "
                                f"window {window.index}: {_error_text(exc)}")
            state = ("started" if trigger and trigger["started"]
                     else "suppressed" if trigger and decision.suppressed_because
                     else "failed" if trigger else "-")
            log.info(f"ONLINE WINDOW: scenario={key}, window={n}/{total}, "
                     f"index={window.index}, ensemble={_fmt(scores.ensemble)}, "
                     f"single={_fmt(scores.single)}, "
                     f"threshold_ensemble={_fmt(decision.thresholds['ensemble'])}, "
                     f"threshold_single={_fmt(decision.thresholds['single'])}, "
                     f"mode={decision.mode or '-'}, "
                     f"below={_members(decision.below)}, trigger={state}, "
                     f"applied={'yes' if applied else 'no'}, "
                     f"running={'yes' if pending_now is not None else 'no'}, "
                     f"waited_s={waited:.1f}, "
                     f"deployed_single={selection.single or '-'}, "
                     f"deployed_ensemble={_members(selection.ensemble)}")

        if schedule.is_pending:
            # Due after the last window: never switched in, and not waited for.
            pending = schedule.resolve()
            h, handle = handle, None
            completed = h.done()
            h.kill()
            duration = h.duration_s
            h.close()
            recorder.counts["n_reopts_not_applied"] += 1
            recorder.event("not_applied", window=None,
                           started_at_window=pending.started_at_window,
                           apply_window=pending.apply_window, reason="stream_ended",
                           completed=completed, duration_s=_round(duration))
            log.info(f"ONLINE REOPT: scenario={key}, event=not_applied, "
                     f"started_at_window={pending.started_at_window}, "
                     f"apply_window={pending.apply_window}, "
                     f"completed={'yes' if completed else 'no'}, reason=stream_ended")
        recorder.finish("complete")
    except Exception as exc:
        if handle is not None:
            pending = schedule.pending
            try:
                handle.kill()
                handle.close()
            except Exception:
                pass
            handle = None
            recorder.counts["n_reopts_aborted"] += 1
            recorder.event("aborted", window=current,
                           started_at_window=getattr(pending, "started_at_window", None),
                           apply_window=getattr(pending, "apply_window", None),
                           error=_error_text(exc))
            log.error(f"ONLINE REOPT: scenario={key}, event=aborted, "
                      f"window={current if current is not None else '-'}, "
                      f"started_at_window={getattr(pending, 'started_at_window', '-')}, "
                      f"error={_error_text(exc)}")
        recorder.finish("failed", error=_error_text(exc))
        log.error(f"ONLINE SCENARIO FAILED: scenario={key}, error={_error_text(exc)}")
        raise
    finally:
        # A KeyboardInterrupt (or anything else the except above lets through)
        # must not leave a re-optimisation process running on its own.
        if handle is not None:
            try:
                handle.kill()
                handle.close()
            except Exception:
                pass

    stats = recorder.stats()
    log.info(f"ONLINE SCENARIO COMPLETE: scenario={key}, windows={stats['n_windows']}, "
             f"triggers={stats['n_triggers']}, started={stats['n_reopts_started']}, "
             f"applied={stats['n_reopts_applied']}, suppressed={stats['n_suppressed']}, "
             f"failed={stats['n_reopts_failed']}, "
             f"not_applied={stats['n_reopts_not_applied']}, "
             f"waits={stats['n_waits']}, "
             f"mean_ensemble={_fmt(stats['mean_ensemble_score'])}, "
             f"mean_single={_fmt(stats['mean_single_score'])}, "
             f"wall_s={stats['wall_time_s']:.1f}")
    return stats


def scenario_params(scenarios: Sequence[str], params: PolicyParams) -> Dict[str, Dict]:
    """Each scenario's own parameters, as its results row names them."""
    return {key: TriggerPolicy(key, params).params_dict() for key in scenarios}


def run_online(scenarios: Sequence[str],
               params: PolicyParams,
               windows: Sequence[OnlineWindow],
               initial_selection: Callable[[str], Selection],
               score_window: Callable[[OnlineWindow, Selection], WindowScores],
               start_reopt,
               writer: OnlineResultsWriter,
               log: Optional[logging.Logger] = None,
               heartbeat_s: float = HEARTBEAT_SECONDS,
               on_window=None) -> List[Dict[str, Any]]:
    """Replay the same `windows` under each scenario in turn.

    `initial_selection(key)` must return an independent copy of the starting
    selection, so no scenario can see another's state. A scenario that raises
    is recorded as failed and the next one still runs; the phase's status is
    'failed' only when every scenario failed.
    """
    log = log or logger
    started = time.time()
    summary = writer.summary
    log.info(f"ONLINE PHASE START: strategy={summary.get('strategy', '-')}, "
             f"scenarios={_members(scenarios)}, windows={len(windows)}, "
             f"window_size={summary.get('window_size', '-')}, "
             f"step_size={summary.get('step_size', '-')}, "
             f"available={summary.get('n_windows_available', '-')}, "
             f"scorable={summary.get('n_windows_scorable', '-')}, "
             f"labels={summary.get('label_source', '-')}, "
             f"reopt_delay={params.reopt_delay}")
    results, failures = [], []
    per_scenario = scenario_params(scenarios, params)
    for position, key in enumerate(scenarios, 1):
        recorder = writer.recorder(key, per_scenario[key], len(windows))
        try:
            results.append(run_scenario(
                key, params, windows, initial_selection(key), score_window,
                start_reopt, recorder, position=f"{position}/{len(scenarios)}",
                log=log, heartbeat_s=heartbeat_s, on_window=on_window))
        except Exception as exc:
            failures.append(f"{key}: {_error_text(exc)}")
            log.error(f"Scenario {key} failed:\n{traceback.format_exc()}")
    wall = time.time() - started
    if failures and len(failures) == len(scenarios):
        writer.finish("failed", "; ".join(failures))
        log.error(f"ONLINE PHASE FAILED: error=every scenario failed "
                  f"({'; '.join(failures)})")
    else:
        # A partial failure keeps the phase 'complete' — the other scenarios'
        # results stand — but the error is carried, and the failed scenario's
        # own row says 'failed'.
        writer.finish("complete", "; ".join(failures) or None)
        log.info(f"ONLINE PHASE COMPLETE: scenarios={len(scenarios)}, "
                 f"failed={len(failures)}, wall_s={wall:.1f}")
    return results
