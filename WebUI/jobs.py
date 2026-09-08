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
                                 parse_decision_metrics)
from WebUI import artifacts, markers, paths

LOG_RING = 20000            # lines kept in memory; the full log also goes to disk
DEFAULT_TIMEOUT = 14400     # 4 hours. A 38-channel SMD entity with the neural
                            # networks and foundation models in the pool spends
                            # well over an hour before narration even starts.
_LINE_SPLIT = re.compile(r"[\r\n]")   # tqdm rewrites lines with \r


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

    anomaly_type = str(params.get("anomaly_type") or "").strip().lower()
    if anomaly_type and anomaly_type != DEFAULT_ANOMALY_TYPE:
        argv += ["--anomaly_type", anomaly_type]
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
    if params.get("strategy"):
        argv += ["--strategy", str(params["strategy"])]
    if params.get("explain", True):
        argv.append("--explain")
    # Always explicit: the config file ships with overwrite: True, so omitting
    # the flag would silently retrain every detector on every run.
    argv += ["--overwrite", "true" if params.get("overwrite") else "false"]
    if params.get("enable_online"):
        argv.append("--enable_online")
        if params.get("max_online_windows"):
            argv += ["--max_online_windows", str(int(params["max_online_windows"]))]

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
        self.cancel_requested = False
        self.log_path: Optional[Path] = None
        self.result_url: Optional[str] = None
        self.report_url: Optional[str] = None
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
                elif kind == "stage" and event.get("key"):
                    # Status only, deliberately no timing. The pipeline measures
                    # each module itself and writes those numbers to the
                    # comprehensive report; wall-clock deltas between log lines
                    # observed here disagree with them (they include the
                    # surrounding logging and, under --parallel, overlap), and
                    # two different durations for one stage is worse than none.
                    entry = self.stages.setdefault(event["key"], {})
                    entry["status"] = event["status"]
                    entry["text"] = event.get("text") or entry.get("text")
                elif kind == "warning":
                    if not any(w["code"] == event["code"] for w in self.warnings):
                        self.warnings.append({"code": event["code"], "text": event["text"]})
                elif kind == "complete":
                    self.saw_complete = True
                    self.saw_partial_complete = bool(event.get("partial"))
                elif kind == "fatal_marker":
                    self.saw_fatal = True
            self.condition.notify_all()

    def _finish(self, status: str, reason: Optional[str] = None) -> None:
        with self.condition:
            self.status = status
            self.failure_reason = reason
            self.finished_at = time.time()
            self.condition.notify_all()


def classify_outcome(job: Job, expect_partial: bool,
                     explain_requested: bool = False,
                     artifacts_written: Optional[bool] = None) -> Dict[str, Any]:
    """Decide what actually happened, in priority order.

    Exit code 0 is NOT sufficient evidence of success: run_app catches every
    exception, logs it and returns normally, so a crashed run still exits 0.
    A completion marker in the log is the discriminator.
    """
    if job.cancel_requested:
        return {"status": "cancelled", "reason": "Cancelled from the UI."}
    if job.status == "timeout":
        return {"status": "timeout",
                "reason": f"No completion after {job.timeout}s; the process was killed."}
    if job.exit_code not in (0, None):
        last = next((l for l in reversed(job.lines) if l.strip()), "")
        return {"status": "failed",
                "reason": f"Exited with code {job.exit_code}. {last}".strip()}

    expected_marker_seen = (job.saw_partial_complete if expect_partial
                            else job.saw_complete and not job.saw_partial_complete)
    if job.saw_complete and expected_marker_seen:
        if explain_requested and artifacts_written is False:
            return {"status": "succeeded_with_warnings",
                    "reason": "The run finished but wrote no explanation artifacts — "
                              "the LLM server was most likely unreachable."}
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
        if outcome["status"] in ("succeeded", "succeeded_with_warnings"):
            job.result_url = f"/result/{dataset}/{entity}"
            # Only link the report when it is actually on disk: partial runs
            # return before the pipeline writes it.
            if artifacts.comprehensive_path(dataset, entity) is not None:
                job.report_url = f"/report/{dataset}/{entity}"
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
