"""
Each re-optimisation of the online phase runs in a child process of its own.

Why a separate process. A re-optimisation re-runs the offline model-selection
pipeline (GA, Thompson, off-by-threshold, Monte Carlo) for minutes. Run in a
thread of the pipeline process, as it used to be, it shared the process-wide
random generators with the stream (so neither was reproducible), shared the
GIL (the stream slowed two- to four-fold while one ran), could not be stopped
(the stream had to wait for it, however long it took), and wrote into the
offline run's result trees unless every writer was redirected per thread. A
child process has its own generators, its own interpreter lock and its own
results root, and it can be killed: `ProcessReopt` gives it a deadline
(`reopt_timeout`) and SIGKILLs it there, the pattern of
Model_Training/deadline.py, whose `die_with_parent` it also uses so that an
orphaned child does not run on after the pipeline dies.

Why a fresh interpreter rather than os.fork(). deadline.py forks, and that is
safe there because nothing has scored yet. By the online phase the offline
pipeline has run scikit-learn's OpenMP code (KMeans, which CBLOF scores
through), and the GNU libgomp that scikit-learn bundles (OpenMP 4.5 here, with
no omp_pause_resource_all to release its threads) keeps a thread pool that
does not survive fork. Measured on this machine: after the parent had run
KMeans, KMeans in a forked child segfaulted with torch loaded (4 of 4 trials)
and hung without it (every variant tried, `predict` and `fit` alike), also
with OpenMP limited to one thread in the child. So the child is started as `python -m
Model_Selection.online_reopt <payload> <result>`, imports what it needs (about
5 s, against minutes of selection) and receives its inputs as a pickle; its
detectors are loaded from the same checkpoint files the pipeline loaded.

Determinism. The child seeds Python's, numpy's and torch's generators with the
seed it is given (the run's seed plus the triggering window's index) before
it does anything else, and starts from a clean interpreter, so the same
window gives the same re-optimisation — in another scenario, or another run.

The protocol: the payload is {"target": "module:function", "kwargs": {...},
"seed": int, "results_root": path}; the child calls the target with the kwargs
and pickles {"ok": True, "value": <return value>} or {"ok": False, "error":
"<one line>", "detail": "<traceback>"} to the result path (temporary file plus
rename, so a half-written result is never read).
"""

import copy
import importlib
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from Model_Selection.online_phase import ReoptFailed, ReoptTimedOut

REPO_ROOT = Path(__file__).resolve().parent.parent

PAYLOAD_FILE = "payload.pkl"
RESULT_FILE = "result.pkl"


class ProcessReopt:
    """One re-optimisation in a child interpreter: the parent's handle on it.

    Implements the handle protocol of online_phase.run_scenario: done(),
    wait(seconds), result(), kill(), close(), duration_s, reused_from.
    `on_success(value, duration_s)` is called once, when a successful result
    is first read.
    """

    def __init__(self, target: str, kwargs: Dict[str, Any], workdir: str, seed: int,
                 timeout: Optional[float] = None, python: str = sys.executable,
                 on_success: Optional[Callable[[Any, float], None]] = None):
        self.workdir = str(workdir)
        os.makedirs(self.workdir, exist_ok=True)
        self.payload_path = os.path.join(self.workdir, PAYLOAD_FILE)
        self.result_path = os.path.join(self.workdir, RESULT_FILE)
        with open(self.payload_path, "wb") as f:
            pickle.dump({"target": target, "kwargs": kwargs, "seed": int(seed),
                         "results_root": os.path.join(self.workdir, "results")},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("MPLBACKEND", "Agg")
        self.timeout = float(timeout) if timeout else None
        self.reused_from: Optional[str] = None
        self._on_success = on_success
        self._outcome: Optional[tuple] = None
        self._ended: Optional[float] = None
        self._started = time.monotonic()
        # Same process group as the pipeline, so the web UI's cancel (which
        # signals the whole group) stops it too.
        self.proc = subprocess.Popen(
            [python, "-m", "Model_Selection.online_reopt", self.payload_path,
             self.result_path],
            cwd=str(REPO_ROOT), env=env)

    # -- the handle protocol -------------------------------------------------

    def done(self) -> bool:
        if self._outcome is not None:
            return True
        code = self.proc.poll()
        if code is not None:
            self._collect(code)
            return True
        if self.timeout is not None and time.monotonic() - self._started >= self.timeout:
            self._stop()
            self._outcome = ("timeout",)
            return True
        return False

    def wait(self, seconds: float) -> bool:
        end = time.monotonic() + max(float(seconds), 0.0)
        while not self.done():
            remaining = end - time.monotonic()
            if self.timeout is not None:
                remaining = min(remaining, self._started + self.timeout - time.monotonic())
            if remaining <= 0:
                return self.done()
            try:
                # Returns as soon as the child exits.
                self.proc.wait(timeout=min(remaining, 1.0))
            except subprocess.TimeoutExpired:
                pass
        return True

    def result(self):
        if not self.done():
            raise RuntimeError("the re-optimisation has not finished")
        kind = self._outcome[0]
        if kind == "ok":
            return self._outcome[1]
        if kind == "timeout":
            raise ReoptTimedOut(f"timeout: exceeded the {self.timeout:g}s "
                                f"re-optimisation limit and was killed")
        if kind == "killed":
            raise ReoptFailed("the re-optimisation was killed before it finished")
        error = ReoptFailed(self._outcome[1])
        error.detail = self._outcome[2]
        raise error

    def kill(self) -> None:
        if self._outcome is None:
            self._stop()
            self._outcome = ("killed",)

    def close(self) -> None:
        self.kill()
        shutil.rmtree(self.workdir, ignore_errors=True)

    @property
    def duration_s(self) -> float:
        return (self._ended or time.monotonic()) - self._started

    # -- internals -----------------------------------------------------------

    def _stop(self) -> None:
        """SIGKILL: a child stuck in a C extension would not act on SIGTERM."""
        try:
            self.proc.send_signal(signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        self._ended = time.monotonic()

    def _collect(self, code: int) -> None:
        self._ended = time.monotonic()
        report = None
        try:
            with open(self.result_path, "rb") as f:
                report = pickle.load(f)
        except Exception:
            report = None
        if isinstance(report, dict) and report.get("ok"):
            self._outcome = ("ok", report.get("value"))
            if self._on_success is not None:
                self._on_success(report.get("value"), self.duration_s)
            return
        if isinstance(report, dict):
            self._outcome = ("error", report.get("error") or "the re-optimisation failed",
                             report.get("detail"))
            return
        if code < 0:
            from Model_Training.deadline import describe_signal
            why = describe_signal(-code, "the re-optimisation process")
        else:
            why = f"the re-optimisation process exited with status {code} and no result"
        self._outcome = ("error", why, None)


class CompletedReopt:
    """A handle on a result reused from an identical, earlier re-optimisation.

    The input of a re-optimisation is a function of its triggering window
    alone (the offline data slid forward to that window) and its seed is the
    run's seed plus that window, so a second scenario triggering at the same
    window would compute the very same selection; it gets a copy of the first
    one's instead. `duration_s` is the original run's.
    """

    def __init__(self, value, duration_s: Optional[float], reused_from: str):
        self._value = value
        self.duration_s = duration_s
        self.reused_from = reused_from

    def done(self) -> bool:
        return True

    def wait(self, seconds: float) -> bool:
        return True

    def result(self):
        return copy.deepcopy(self._value)

    def kill(self) -> None:
        pass

    def close(self) -> None:
        pass


# ── The child ────────────────────────────────────────────────────────────────

def _one_line(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def _resolve(target: str):
    module, _, name = target.partition(":")
    return getattr(importlib.import_module(module), name)


def _write_atomic(path: str, report: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(report, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    payload_path, result_path = argv
    from Model_Training.deadline import die_with_parent
    die_with_parent(os.getppid())
    code = 1
    try:
        with open(payload_path, "rb") as f:
            payload = pickle.load(f)
        from Utils import paths
        from Utils.set_all_seeds import set_all_seeds
        # Everything the pipeline writes goes under the scratch root the
        # parent removes afterwards, never into the offline run's trees.
        paths.set_results_root(payload["results_root"])
        set_all_seeds(int(payload["seed"]) % (2 ** 32))
        value = _resolve(payload["target"])(**payload.get("kwargs", {}))
        report, code = {"ok": True, "value": value}, 0
    except BaseException as e:                     # noqa: BLE001 — reported, not swallowed
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        report = {"ok": False, "error": _one_line(e), "detail": detail}
    try:
        _write_atomic(result_path, report)
    except Exception as e:                          # the parent reports the missing result
        print(f"Could not write the re-optimisation result: {_one_line(e)}",
              file=sys.stderr, flush=True)
        code = 1
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    # os._exit: library threads (tqdm's monitor, TensorFlow's pools) must not
    # hold the exit up once the result is on disk.
    os._exit(code)


if __name__ == "__main__":
    main()
