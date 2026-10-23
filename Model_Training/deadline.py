"""Run one detector instance's training in a child process that can be stopped.

Why a process and not the SIGALRM the scoring deadline uses
(`Utils.model_selection_utils._score_deadline`): a Python signal handler only
runs between bytecodes, so it cannot interrupt a long call into C, and
training is made of such calls — a BLAS product, an sklearn or numba kernel, a
torch step. SOS on a long entity is the reported case: its n x n matrices are
~26 GB on a full-length SMD machine, and on a machine without that memory
every numpy call over them crawls through swap, with the alarm landing only
when one returns. The only thing that reliably stops such a call is ending the
process it runs in, so the instance trains in a forked child and the parent
kills that child when the time is up.

The same boundary also contains failures no `except` clause can catch: the
kernel's OOM killer or a segfault in an extension takes down the child, and
the parent reads that as one instance failing to train rather than dying with
it.

What the child produces it writes to disk itself — the checkpoint, the meta
file, the diagnostic plot — exactly as in-process training did, and every
consumer already reads checkpoints from disk. Only a one-line error report
comes back through a pipe, so nothing has to be pickled across.

Fork, not spawn, deliberately. A forked child inherits the loaded datasets, the
imported libraries and the numpy and torch RNG state the caller seeded; a
spawned one would re-import torch and pyod (seconds per instance) and start
from an unseeded RNG. Two differences from in-process training remain, and
both concern RNG state. Every child starts from the PARENT's state, not from
what the previous instance left behind, and nothing a child draws or seeds
comes back — which is why app.py seeds before training and again after it.
And CPython re-seeds the stdlib `random` from OS entropy in every forked child
(`os.register_at_fork` in random.py); MTAD-GAT, the only training code that
uses it, seeds it itself.

Fork has two preconditions. The first cannot be detected, so the caller has to
meet it; the second is checked here.

  * The parent must not have run scikit-learn's OpenMP code (KMeans, which CBLOF
    fits) before forking. GNU libgomp keeps a thread pool that does not survive
    fork: measured here, a child's KMeans segfaulted (torch loaded) or hung
    forever (sklearn alone) whenever the parent had run one first, and was fine
    when it had not. `TrainModels` never trains in the parent while a deadline
    is set, and `app.py` trains before anything scores, so the condition holds
    there.
  * CUDA must not be initialised in the parent; a forked child cannot use it.
    `fork_unsafe_reason` detects that, and `TrainModels` trains in-process
    instead when it is set. Note what counts as initialising: not only torch's
    own lazy init but any `cuInit`, and `torch.cuda.is_available()` performs
    one (through `cudaGetDeviceCount`). So the trainer must never ask
    `is_available()` in the parent before forking — the check would cause
    the very condition it checks for.
"""

import ctypes
import os
import signal
import sys
import time
import traceback
from typing import Callable, Optional

# The child's report is one line. Capped well below the 64 KiB pipe buffer, so
# the child's single write can never block on a parent that only reads after
# the child has exited.
_REPORT_LIMIT = 4000

# How often the parent looks for the child's exit. Short enough that it adds
# nothing noticeable per instance, long enough to cost no measurable CPU.
_POLL_SECONDS = 0.05

# The CUDA driver API's answer to any call made before `cuInit`.
_CUDA_ERROR_NOT_INITIALIZED = 3


class TrainingTimeout(Exception):
    """The child outlived its time limit and was killed."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        super().__init__(f"exceeded the {seconds:g}s training limit and was stopped")


class TrainingFailed(Exception):
    """The child raised, or died without being able to say why."""


def fork_unsafe_reason() -> Optional[str]:
    """Why a training child must not be forked from this process, or None."""
    if not hasattr(os, "fork"):
        return "this platform cannot fork"
    if _cuda_initialised():
        return ("CUDA is already initialised in this process, and a forked "
                "child could not use it")
    return None


def _cuda_initialised() -> bool:
    """Has anything in this process initialised CUDA? Initialises nothing itself.

    torch's flag covers its own lazy init. The driver probe covers everything
    else (`torch.cuda.is_available()`, TensorFlow): before `cuInit`, every
    driver API call returns CUDA_ERROR_NOT_INITIALIZED, so asking for the
    device count is a read-only question. No driver library at all — a
    CPU-only machine like the one this was written on — means nothing to
    initialise.
    """
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        return True
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False
    count = ctypes.c_int()
    return libcuda.cuDeviceGetCount(ctypes.byref(count)) != _CUDA_ERROR_NOT_INITIALIZED


def run_with_deadline(fn: Callable[[], None], seconds: float) -> None:
    """Run `fn()` in a forked child; kill it if it is still running at `seconds`.

    Returns normally when `fn` returned. Raises TrainingTimeout when the child
    had to be killed, and TrainingFailed when `fn` raised (the message is the
    child's `ExceptionType: message`; the full traceback was already printed by
    the child to the shared stderr) or the child died of a signal.
    """
    read_fd, write_fd = os.pipe()
    parent_pid = os.getpid()
    # Anything still buffered would otherwise be written twice: once by the
    # parent, once by the child that inherited a copy of the buffer.
    _flush_std()
    pid = os.fork()
    if pid == 0:                                            # the child
        os.close(read_fd)
        code, report = 0, b""
        try:
            die_with_parent(parent_pid)
            fn()
        except BaseException as e:
            code = 1
            traceback.print_exc()
            report = _one_line(e).encode("utf-8", "replace")[:_REPORT_LIMIT]
        finally:
            # os._exit, never a normal exit: atexit handlers, finalizers and
            # stream buffers belong to the parent, and a child running them
            # would act on the parent's behalf — flushing its output twice,
            # tearing down shared resources.
            try:
                if report:
                    os.write(write_fd, report)
                _flush_std()
            finally:
                os._exit(code)

    os.close(write_fd)
    try:
        try:
            status = _wait_until(pid, time.monotonic() + seconds)
        except BaseException:
            # Ctrl-C, or anything else that ends the wait early: the child must
            # not be left training on its own after the parent has moved on.
            _kill_and_reap(pid)
            raise
        if status is None:
            _kill_and_reap(pid)
            raise TrainingTimeout(seconds)
        report = _read_all(read_fd).decode("utf-8", "replace").strip()
    finally:
        os.close(read_fd)

    if os.WIFSIGNALED(status):
        raise TrainingFailed(_describe_signal(os.WTERMSIG(status)))
    code = os.WEXITSTATUS(status)
    if code != 0:
        raise TrainingFailed(report or f"the training process exited with status {code}")


def _wait_until(pid: int, deadline: float):
    """The child's wait status, or None if it is still running at `deadline`."""
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return status
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(_POLL_SECONDS, remaining))


def _kill_and_reap(pid: int) -> None:
    """SIGKILL, not SIGTERM: a child stuck in C would not act on a gentler signal
    any sooner than it acts on the alarm this module exists to replace."""
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def _read_all(fd: int) -> bytes:
    chunks = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _one_line(e: BaseException) -> str:
    text = " ".join(str(e).split())
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


def describe_signal(signum: int, what: str = "the training process") -> str:
    """'<what> was killed by SIGSEGV — a crash inside a compiled extension'."""
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = f"signal {signum}"
    why = {
        signal.SIGKILL: " — most often the kernel's out-of-memory killer",
        signal.SIGSEGV: " — a crash inside a compiled extension",
    }.get(signum, "")
    return f"{what} was killed by {name}{why}"


_describe_signal = describe_signal


def _flush_std() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def die_with_parent(parent_pid: int) -> None:
    """Have the kernel SIGKILL this child if the parent dies first (Linux only).

    The web UI cancels a run by signalling its whole process group, which the
    child belongs to, so this is for the other ways a parent goes: killed on
    its own, or by the OOM killer. Without it the child would train on,
    orphaned, for as long as the deadline was meant to prevent. The online
    phase's re-optimisation processes call it too.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL(None, use_errno=True).prctl(
            PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
    except Exception:
        return
    # The parent may have died between fork and prctl, in which case the
    # request above will never fire.
    if os.getppid() != parent_pid:
        os._exit(1)


_die_with_parent = die_with_parent
