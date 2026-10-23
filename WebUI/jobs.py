"""
Runs the pipeline as a subprocess and streams its progress.

The pipeline cannot be imported: `run_app` parses sys.argv, blocks for minutes,
and pulls in torch/tensorflow/ray at module load. So it is spawned as a child
process and its merged output is pumped into a ring buffer that SSE subscribers
read from.

The subtlety worth knowing: **`run_app` exits 0 even when it fails.** Its whole
body sits in a try/except that logs a traceback and returns normally, so the
exit code alone says nothing. `classify_outcome` therefore requires a
completion marker before it will call a run successful.
"""

import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from Utils.pipeline_spec import (ALL_DETECTORS, ALL_STAGES,
                                 DEFAULT_ANOMALY_TYPE, DEFAULT_DECISION_METRICS,
                                 format_decision_metrics, OFFLINE_ITERATION,
                                 ONLINE_SCENARIO_LABELS, parse_adaptive_k,
                                 parse_adaptive_window, parse_anomaly_types,
                                 parse_decision_metrics,
                                 parse_drop_pct, parse_max_online_windows,
                                 parse_online_labels,
                                 parse_online_strategy, parse_reopt_delay,
                                 parse_update_interval)
from WebUI import artifacts, markers, online, paths

LOG_RING = 20000            # lines kept in memory; the full log also goes to disk
DEFAULT_TIMEOUT = 14400     # 4 hours. A 38-channel SMD entity with the neural
                            # networks and foundation models in the pool spends
                            # well over an hour before narration even starts.
_LINE_SPLIT = re.compile(r"[\r\n]")   # tqdm rewrites lines with \r

# Online-phase run parameter -> (CLI flag, parser). The parsers are the CLI's
# own (Utils.pipeline_spec), so a value the pipeline would refuse after a
# fork-and-load is refused here instead. An omitted parameter emits no flag and
# the pipeline falls back to Configs/config.yml.
ONLINE_PARAMS = (
    ("update_interval", "--update_interval", parse_update_interval),
    ("drop_pct", "--drop_pct", parse_drop_pct),
    ("adaptive_window", "--adaptive_window", parse_adaptive_window),
    ("adaptive_k", "--adaptive_k", parse_adaptive_k),
    ("max_online_windows", "--max_online_windows", parse_max_online_windows),
    ("reopt_delay", "--reopt_delay", parse_reopt_delay),
)


def _unset(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _online_flags(params: Dict[str, Any]) -> List[str]:
    """The online-phase flags of a run. Raises ValueError on a bad value."""
    flags = ["--enable_online"]
    # `strategy` is the parameter's spelling from before the four scenarios;
    # `online_strategy` wins when both are sent.
    strategy = params.get("online_strategy")
    if _unset(strategy):
        strategy = params.get("strategy")
    if not _unset(strategy):
        parse_online_strategy(strategy)
        flags += ["--strategy", str(strategy).strip()]
    for key, flag, parse in ONLINE_PARAMS:
        if _unset(params.get(key)):
            continue
        value = parse(params[key])
        if value is not None:
            flags += [flag, str(value) if isinstance(value, int) else format(value, ".10g")]
    # Unset ("auto") emits no flag: the pipeline then uses real labels where
    # the dataset has them, exactly as the configure page previews.
    labels = parse_online_labels(params.get("online_labels"))
    if labels is not None:
        flags += ["--online_labels", labels]
    if params.get("inject_online_regime"):
        flags.append("--inject_online_regime")
    return flags


def build_argv(params: Dict[str, Any]) -> List[str]:
    """Run parameters -> the exact argv to spawn. Pure, so it can be previewed
    in the UI and asserted in tests; the command shown to the user and the
    command executed come from this one function."""
    dataset = str(params.get("dataset") or "").strip()
    entity = str(params.get("entity") or "").strip()
    if not dataset or not entity:
        raise ValueError("dataset and entity are required")

    argv = [sys.executable, "-u", "app.py",
            "--config_file_path", str(paths.CONFIG_YML),
            "--dataset", dataset,
            "--entity", entity,
            "--parallel", "true" if params.get("parallel") else "false"]

    # One type or several: `anomaly_types` (a list) from the multi-select, or
    # the older single `anomaly_type`, which may itself be a list or "a,b".
    raw_types = params.get("anomaly_types", params.get("anomaly_type"))
    anomaly_types = (parse_anomaly_types(raw_types) if raw_types not in (None, "")
                     else [DEFAULT_ANOMALY_TYPE])
    if anomaly_types != [DEFAULT_ANOMALY_TYPE]:
        argv += ["--anomaly_type", ",".join(anomaly_types)]
    if params.get("anomaly_rate") is not None:
        argv += ["--anomaly_rate", str(float(params["anomaly_rate"]))]

    metrics = params.get("decision_metrics") or params.get("decision_metric")
    if isinstance(metrics, str):
        metrics = [metrics]
    if metrics:
        try:
            spec = parse_decision_metrics(metrics)
        except ValueError:
            spec = DEFAULT_DECISION_METRICS
        if spec != DEFAULT_DECISION_METRICS:
            argv += ["--decision_metric", format_decision_metrics(spec)]

    if params.get("iteration") is not None:
        argv += ["--iteration", str(int(params["iteration"]))]
    if params.get("explain", True):
        argv.append("--explain")
    # Always explicit: the config file ships with overwrite: True, so omitting
    # the flag would silently retrain every detector on every run.
    argv += ["--overwrite", "true" if params.get("overwrite") else "false"]
    if params.get("enable_online"):
        argv += _online_flags(params)

    stages = params.get("stages")
    if stages:
        selected = {str(s).strip().lower() for s in stages if str(s).strip()}
        if selected and selected != set(ALL_STAGES):
            argv += ["--stages", ",".join(sorted(selected))]

    detectors = params.get("detectors")
    if detectors:
        chosen = {str(d).strip() for d in detectors if str(d).strip()}
        if chosen and chosen != set(ALL_DETECTORS):
            # Canonical order so an equivalent selection always yields the same
            # command string.
            argv += ["--detectors", ",".join(d for d in ALL_DETECTORS if d in chosen)]

    if params.get("llm_model"):
        argv += ["--llm_model", str(params["llm_model"])]
    if params.get("llm_base_url"):
        argv += ["--llm_base_url", str(params["llm_base_url"])]
    return argv


def build_narrate_argv(dataset: str, entity: str,
                       llm_model: Optional[str] = None,
                       llm_base_url: Optional[str] = None) -> List[str]:
    """Narration-only pass.

    Needed because a partial run never reaches the narration block: the
    `is_partial` early return in run_app happens before it, so `--stages ga
    --explain` writes IR files but no prose. Also serves as the "regenerate
    narratives" action for when the LLM server was down during the run.
    """
    argv = [sys.executable, "-u", "-m", "Explainability.narrate",
            "--dataset", str(dataset), "--entity", str(entity),
            "--iteration", str(OFFLINE_ITERATION)]
    if llm_model:
        argv += ["--model", str(llm_model)]
    if llm_base_url:
        argv += ["--base-url", str(llm_base_url)]
    return argv


class Job:
    """One pipeline run and everything observers need to follow it."""

    def __init__(self, job_id: str, argv: List[str], params: Dict[str, Any],
                 timeout: int = DEFAULT_TIMEOUT):
        self.id = job_id
        self.argv = argv
        self.params = params
        self.timeout = timeout
        self.status = "starting"       # starting|running|succeeded|failed|cancelled|timeout
        self.exit_code: Optional[int] = None
        self.failure_reason: Optional[str] = None
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.lines: deque = deque(maxlen=LOG_RING)
        self.cursor = 0                # total lines ever seen (SSE resume point)
        self.phase: Optional[Dict[str, Any]] = None
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.warnings: List[Dict[str, str]] = []
        self.saw_complete = False
        self.saw_partial_complete = False
        self.saw_fatal = False
        # A traceback AFTER the completion marker. The online phase runs after
        # "EXECUTION COMPLETE!", so a crash there used to read as a clean run.
        self.saw_fatal_after_complete = False
        # The online phase as the log tells it: None until it starts, then
        # {"status", "strategy", "order", "scenarios": {key: progress}, "error"}.
        self.online: Optional[Dict[str, Any]] = None
        self.cancel_requested = False
        self.log_path: Optional[Path] = None
        self.result_url: Optional[str] = None
        self.report_url: Optional[str] = None
        self.online_url: Optional[str] = None
        self.condition = threading.Condition()
        self._proc: Optional[subprocess.Popen] = None

    # -- observation ---------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self.condition:
            return {
                "job_id": self.id,
                "status": self.status,
                "argv": self.argv,
                "params": self.params,
                "exit_code": self.exit_code,
                "failure_reason": self.failure_reason,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed": (self.finished_at or time.time()) - self.started_at,
                "phase": self.phase,
                "stages": dict(self.stages),
                "warnings": list(self.warnings),
                "cursor": self.cursor,
                "result_url": self.result_url,
                "report_url": self.report_url,
                "online": _copy_online(self.online),
                "online_url": self.online_url,
            }

    def tail(self, offset: int = 0) -> Dict[str, Any]:
        """Lines from `offset` onwards, plus the new cursor."""
        with self.condition:
            total, buffered = self.cursor, len(self.lines)
            first_buffered = total - buffered
            start = max(offset, first_buffered)
            slice_from = start - first_buffered
            return {"from": start, "lines": list(self.lines)[slice_from:],
                    "cursor": total}

    def is_done(self) -> bool:
        return self.status in ("succeeded", "failed", "cancelled", "timeout",
                               "succeeded_with_warnings")

    # -- internal ------------------------------------------------------------

    def _record(self, line: str) -> None:
        event = markers.classify(line)
        with self.condition:
            self.lines.append(line)
            self.cursor += 1
            if event:
                kind = event["type"]
                if kind == "phase":
                    self.phase = {"number": event["number"], "title": event["title"]}
                # `not self.saw_complete`: after the completion marker no
                # offline sub-stage can run. The online phase's background
                # re-optimisations re-run the offline pipeline and log its
                # "Sub-stage 6.x" lines, which would otherwise flip the finished
                # offline rail back to "running".
                elif kind == "stage" and event.get("key") and not self.saw_complete:
                    # Status only, deliberately no timing. The pipeline measures
                    # each module itself and writes those numbers to the
                    # comprehensive report; wall-clock deltas between log lines
                    # observed here disagree with them (they include the
                    # surrounding logging and, under --parallel, overlap), and
                    # two different durations for one stage is worse than none.
                    entry = self.stages.setdefault(event["key"], {})
                    entry["status"] = event["status"]
                    entry["text"] = event.get("text") or entry.get("text")
                elif kind == "online":
                    self._record_online(event)
                elif kind == "warning":
                    if not any(w["code"] == event["code"] for w in self.warnings):
                        self.warnings.append({"code": event["code"], "text": event["text"]})
                elif kind == "complete":
                    self.saw_complete = True
                    self.saw_partial_complete = bool(event.get("partial"))
                elif kind == "fatal_marker":
                    self.saw_fatal = True
                    if self.saw_complete:
                        self.saw_fatal_after_complete = True
            self.condition.notify_all()

    def _record_online(self, event: Dict[str, Any]) -> None:
        """Fold one `ONLINE ...` line into the phase rail and `self.online`.

        Each scenario is a stage entry `online_<key>` (group "online") carrying
        its window progress, so the existing stage stream delivers it; the rail
        moves to phase 8 when the online phase starts. Called with the
        condition held.
        """
        kind, fields = event["kind"], event.get("fields") or {}

        def scenario_entry(key):
            if self.online is None:
                return None
            return self.online["scenarios"].get(key)

        def sync_stage(key):
            prog = scenario_entry(key)
            if prog is None:
                return
            text = f"window {prog['window']}/{prog['total']}" if prog["total"] else None
            if prog["reopts_applied"]:
                text = f"{text} · {prog['reopts_applied']} re-optimization(s) applied"
            waiting = prog.get("waiting")
            if waiting:
                # Waiting is not stalling: say what for, so a long pause reads
                # as the re-optimisation it is.
                text = (f"{text} · waiting for the re-optimization started at "
                        f"window {waiting['since_window']} (due at window "
                        f"{waiting['until_window']})")
            if prog.get("error"):
                text = prog["error"]
            # The rail knows running/done/failed; a waiting scenario is still
            # running, and its text and `waiting` say what it waits for.
            self.stages[f"online_{key}"] = {
                "status": "running" if prog["status"] == "waiting" else prog["status"],
                "text": text or prog["label"],
                "group": "online", "label": prog["label"],
                "window": prog["window"], "total": prog["total"],
                "waiting": dict(waiting) if waiting else None}

        if kind == "phase_start":
            order = [k for k in (fields.get("scenarios") or "").split("+") if k]
            total = _to_int(fields.get("windows"))
            self.online = {"status": "running", "strategy": fields.get("strategy"),
                           "order": order, "scenarios": {}, "error": None}
            self.phase = {"number": markers.ONLINE_PHASE_NUMBER,
                          "title": markers.PHASE_TITLES[markers.ONLINE_PHASE_NUMBER]}
            for key in order:
                self.online["scenarios"][key] = {
                    "status": "pending", "label": ONLINE_SCENARIO_LABELS.get(key, key),
                    "window": 0, "total": total or 0, "triggers": 0,
                    "reopts_started": 0, "reopts_applied": 0, "suppressed": 0,
                    "failed": 0, "not_applied": 0, "aborted": 0, "waiting": None,
                    "error": None}
                sync_stage(key)
            return
        if kind == "phase_skipped":
            self.online = {"status": "skipped", "strategy": None, "order": [],
                           "scenarios": {}, "error": fields.get("reason")}
            return
        if kind in ("phase_failed", "phase_complete"):
            if self.online is None:
                self.online = {"status": None, "strategy": None, "order": [],
                               "scenarios": {}, "error": None}
            self.online["status"] = "failed" if kind == "phase_failed" else "complete"
            if kind == "phase_failed":
                self.online["error"] = fields.get("error") or event.get("text")
            return

        key = fields.get("scenario")
        prog = scenario_entry(key)
        if prog is None:
            return
        if kind == "scenario_start":
            prog["status"] = "running"
        elif kind == "window":
            done, _, total = (fields.get("window") or "").partition("/")
            prog["status"] = "running"
            prog["window"] = _to_int(done) or prog["window"]
            prog["total"] = _to_int(total) or prog["total"]
            trigger = fields.get("trigger")
            if trigger in ("started", "suppressed"):
                prog["triggers"] += 1
                prog["reopts_started" if trigger == "started" else "suppressed"] += 1
        elif kind == "reopt":
            what = fields.get("event")
            if what == "waiting":
                prog["status"] = "waiting"
                started = _to_int(fields.get("started_at_window"))
                prog["waiting"] = {"since_window": started,
                                   "until_window": _to_int(fields.get("window"))}
            elif what in ("applied", "failed"):
                if prog.get("waiting"):
                    prog["status"] = "running"
                prog["waiting"] = None
                prog["reopts_applied" if what == "applied" else "failed"] += 1
            elif what == "not_applied":
                prog["not_applied"] += 1
            elif what == "aborted":
                prog["aborted"] += 1
        elif kind == "scenario_complete":
            prog["status"] = "done"
            prog["waiting"] = None
        elif kind == "scenario_failed":
            prog["status"] = "failed"
            prog["waiting"] = None
            prog["error"] = fields.get("error")
        sync_stage(key)

    def _finish(self, status: str, reason: Optional[str] = None) -> None:
        with self.condition:
            self.status = status
            self.failure_reason = reason
            self.finished_at = time.time()
            self.condition.notify_all()


def _to_int(value) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _copy_online(online: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if online is None:
        return None
    return {**online, "order": list(online.get("order") or []),
            "scenarios": {k: dict(v) for k, v in (online.get("scenarios") or {}).items()}}


def online_problem(job: Job) -> Optional[str]:
    """Why the online phase of a finished run is not a clean success, or None.

    The online phase runs after "EXECUTION COMPLETE!", so the completion marker
    alone cannot vouch for it. The offline results are already written by then
    and stay valid, which is why a problem here is a warning on the run rather
    than a failure of it.
    """
    if job.saw_fatal_after_complete:
        detail = next((l for l in reversed(job.lines) if markers.FATAL_SIGNATURE in l), "")
        return ("The offline phase finished, but the pipeline raised an exception "
                f"after it. {detail}").strip()
    online_state = job.online
    if not job.params.get("enable_online"):
        return None
    if online_state is None:
        return "The online phase was requested but never started."
    status = online_state.get("status")
    if status == "skipped":
        return f"The online phase was skipped: {online_state.get('error') or 'no reason given'}"
    if status == "failed":
        return ("The offline phase finished, but the online phase failed: "
                f"{online_state.get('error') or 'see the log'}")
    if status != "complete":
        return "The online phase did not finish."
    scenarios = online_state.get("scenarios") or {}
    problems = []
    failed = [k for k, v in scenarios.items() if v.get("status") == "failed"]
    if failed:
        problems.append(f"Online scenario(s) failed: {', '.join(failed)}.")
    # A failed re-optimisation does not fail its scenario, which keeps
    # streaming with the selection it had. Left unreported, a scenario whose
    # every re-optimisation died would read as a clean run while behaving like
    # `fixed`, which is how the old NameError went unnoticed.
    for key, prog in scenarios.items():
        n = prog.get("failed") or 0
        if n and key not in failed:
            problems.append(f"{n} re-optimization(s) failed or timed out in {key} "
                            f"(see the log); the scenario kept its previous selection.")
    return " ".join(problems) or None


def classify_outcome(job: Job, expect_partial: bool,
                     explain_requested: bool = False,
                     artifacts_written: Optional[bool] = None) -> Dict[str, Any]:
    """Decide what actually happened, in priority order.

    Exit code 0 is NOT sufficient evidence of success: run_app catches every
    exception, logs it and returns normally, so a crashed run still exits 0.
    A completion marker in the log is the discriminator — for the OFFLINE
    phase. The online phase runs after it, so its problems are judged
    separately (`online_problem`) and turn a success into one with warnings.
    """
    expected_marker_seen = (job.saw_partial_complete if expect_partial
                            else job.saw_complete and not job.saw_partial_complete)
    no_artifacts = ("The run finished but wrote no explanation artifacts — "
                    "the LLM server was most likely unreachable.")

    if job.cancel_requested:
        return {"status": "cancelled", "reason": "Cancelled from the UI."}
    if job.status == "timeout":
        if job.saw_complete and expected_marker_seen:
            # The watchdog is one absolute timer for the whole run, and the
            # online phase runs after the completion marker. The offline
            # results are on disk and valid, so the run keeps its result and
            # report links; only what came after the marker was cut short.
            during_online = (job.online or {}).get("status") == "running"
            where = ("during the online phase, whose results are incomplete"
                     if during_online else "after the offline phase finished")
            problems = [f"The offline phase finished, but the run was killed by the "
                        f"{job.timeout}s run timeout {where}."]
            if explain_requested and artifacts_written is False:
                problems.append(no_artifacts)
            return {"status": "succeeded_with_warnings", "reason": " ".join(problems)}
        return {"status": "timeout",
                "reason": f"No completion after {job.timeout}s; the process was killed."}
    if job.exit_code not in (0, None):
        last = next((l for l in reversed(job.lines) if l.strip()), "")
        return {"status": "failed",
                "reason": f"Exited with code {job.exit_code}. {last}".strip()}

    if job.saw_complete and expected_marker_seen:
        problems = []
        if explain_requested and artifacts_written is False:
            problems.append(no_artifacts)
        # A partial run stops before the online phase by design.
        problem = None if expect_partial else online_problem(job)
        if problem:
            problems.append(problem)
        if problems:
            return {"status": "succeeded_with_warnings", "reason": " ".join(problems)}
        return {"status": "succeeded", "reason": None}

    if job.saw_complete and not expected_marker_seen:
        kind = "partial" if job.saw_partial_complete else "full"
        return {"status": "failed",
                "reason": f"The run reported a {kind} completion, which does not match "
                          f"the requested stages."}

    if job.saw_fatal:
        detail = next((l for l in reversed(job.lines) if markers.FATAL_SIGNATURE in l), "")
        return {"status": "failed",
                "reason": f"The pipeline raised an exception. {detail}".strip()}

    last = next((l for l in reversed(job.lines) if "ERROR" in l), None)
    return {"status": "failed",
            "reason": last or "The process exited without reaching completion."}


class JobManager:
    """One run at a time.

    Not an arbitrary cap: the pipeline peaks around 2 GB RSS, `--parallel`
    already saturates several threads, and every output path is a function of
    (dataset, entity) — two runs on one entity would interleave writes into the
    same directories and corrupt the artifact set a reader is walking.
    """

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = Path(repo_root or paths.REPO_ROOT)
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._active: Optional[str] = None

    # -- accessors -----------------------------------------------------------

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(self._active) if self._active else None

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            ids = list(reversed(self._order))[:limit]
            return [self._jobs[i].snapshot() for i in ids if i in self._jobs]

    # -- lifecycle -----------------------------------------------------------

    def start(self, params: Dict[str, Any], argv: Optional[List[str]] = None,
              timeout: int = DEFAULT_TIMEOUT) -> Job:
        argv = argv or build_argv(params)
        with self._lock:
            current = self._jobs.get(self._active) if self._active else None
            if current is not None and not current.is_done():
                raise RuntimeError(current.id)     # caller turns this into a 409
            job = Job(uuid.uuid4().hex[:12], argv, params, timeout=timeout)
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._active = job.id

        paths.WEBUI_LOGS.mkdir(parents=True, exist_ok=True)
        job.log_path = paths.WEBUI_LOGS / f"{job.id}.log"

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"      # the child buffers stdout otherwise
        try:
            job._proc = subprocess.Popen(
                argv,
                cwd=str(self.repo_root),   # every output path is relative to it
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # the pipeline logs to STDERR
                text=True,
                errors="replace",          # the logs are full of emoji
                bufsize=1,
                env=env,
                start_new_session=True,    # own group, so cancel kills children
            )
        except OSError as e:
            job._finish("failed", f"Could not start the pipeline: {e}")
            with self._lock:
                self._active = None
            return job

        job.status = "running"
        threading.Thread(target=self._pump, args=(job,), daemon=True).start()
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.is_done() or job._proc is None:
            return False
        job.cancel_requested = True
        proc = job._proc
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except OSError:
                return False
        return True

    # -- the pump ------------------------------------------------------------

    def _pump(self, job: Job) -> None:
        proc = job._proc
        assert proc is not None
        log_file = None
        try:
            log_file = open(job.log_path, "w", encoding="utf-8", errors="replace")
        except OSError:
            log_file = None

        # The timeout runs on its own timer, NOT inside the read loop: a silent
        # child blocks in read() indefinitely, so a deadline checked after each
        # chunk would never fire for exactly the hang it is meant to catch.
        # Killing the process closes the pipe, which unblocks the reader.
        def _on_timeout():
            if not job.is_done():
                job.status = "timeout"
                job._record(f"[webui] no output or exit after {job.timeout}s — killing the run")
                self._kill(proc)

        watchdog = threading.Timer(job.timeout, _on_timeout)
        watchdog.daemon = True
        watchdog.start()

        pending = ""
        try:
            while True:
                chunk = proc.stdout.read(4096) if proc.stdout else ""
                if chunk == "":
                    break
                pending += chunk
                # Split on \r too: tqdm rewrites its bar in place, and treating
                # a whole progress bar as one line would blow the buffer.
                parts = _LINE_SPLIT.split(pending)
                pending = parts.pop()
                for line in parts:
                    if line.strip():
                        job._record(line)
                        if log_file:
                            log_file.write(line + "\n")
                if log_file:
                    log_file.flush()
        except Exception as e:                       # never leave a job hanging
            job._record(f"[webui] log reader stopped: {e}")
        finally:
            watchdog.cancel()
            if pending.strip():
                job._record(pending)
                if log_file:
                    log_file.write(pending + "\n")
            if log_file:
                log_file.close()

        try:
            job.exit_code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._kill(proc)
            job.exit_code = proc.poll()

        expect_partial = bool(job.params.get("stages")) and \
            set(job.params["stages"]) != set(ALL_STAGES)
        outcome = classify_outcome(
            job, expect_partial,
            explain_requested=bool(job.params.get("explain", True)),
            artifacts_written=self._artifacts_written(job))
        dataset, entity = job.params.get("dataset"), job.params.get("entity")
        if job.params.get("enable_online"):
            # A process killed by the timeout or a cancel (or one that crashed)
            # cannot say so in its own files, and its summary would read
            # "running" forever. Only THIS run's summary is touched.
            try:
                online.mark_interrupted(
                    dataset or "", entity or "", since=job.started_at - 5,
                    reason=f"The run ended ({outcome['status']}) before the online "
                           f"phase finished.")
            except OSError as e:
                job._record(f"[webui] could not mark the online results interrupted: {e}")
        if outcome["status"] in ("succeeded", "succeeded_with_warnings"):
            job.result_url = f"/result/{dataset}/{entity}"
            # Only link the report when it is actually on disk: partial runs
            # return before the pipeline writes it.
            if artifacts.comprehensive_path(dataset, entity) is not None:
                job.report_url = f"/report/{dataset}/{entity}"
            # Likewise the online results, and only THIS run's: an older run's
            # summary for the same entity must not be offered as this one's.
            summary = online.info(dataset, entity) if job.params.get("enable_online") else None
            if summary and (summary.get("generated_at") or 0) >= job.started_at - 5:
                job.online_url = summary["page_url"]
        job._finish(outcome["status"], outcome["reason"])

        with self._lock:
            if self._active == job.id:
                self._active = None

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass

    def _artifacts_written(self, job: Job) -> Optional[bool]:
        """Did this run actually produce narratives? None when not applicable.

        Both failure modes that leave an empty results page (LLM unreachable,
        per-stage narration error) are non-fatal inside the pipeline, so the
        only way to notice is to look for the files.
        """
        if not job.params.get("explain", True):
            return None
        nl_dir = paths.resolve_entity_dir(
            paths.EXPLANATIONS_NL, job.params.get("dataset", ""),
            job.params.get("entity", ""))
        if nl_dir is None:
            return False
        try:
            return any(p.stat().st_mtime >= job.started_at - 5
                       for p in nl_dir.glob("nl_*.txt"))
        except OSError:
            return False


_MANAGER: Optional[JobManager] = None


def manager() -> JobManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = JobManager()
    return _MANAGER


def reset_manager() -> None:
    """Test hook."""
    global _MANAGER
    _MANAGER = None
