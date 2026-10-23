"""
When the online phase re-optimises: the four scenarios' trigger rules, and
when a started re-optimisation is switched in.

Pure and side-effect free — no models, no files, no clock — so the rules can be
tested as "given this series of window scores, which windows trigger, which
are suppressed, where is each result switched in, and what threshold was in
effect at each". The runner in `online_phase.py` owns time, the child process
that re-optimises and the results files; this module answers, window by
window, "does this window ask for a re-optimisation, and is one due now?".

The scenarios (keys from Utils.pipeline_spec.ONLINE_SCENARIOS):

  fixed               never re-optimises.
  periodic            re-optimises at every window index divisible by M
                      (--update_interval): windows M, 2M, ... Window 0 is the
                      first online window, scored with the offline selection
                      the offline phase has just made, so it never triggers.
  threshold           re-optimises when a score falls MORE than X% (--drop_pct)
                      below its baseline: score < baseline * (1 - X/100). The
                      baseline is the first score measured after the last
                      (re)optimisation — the first window the newly applied
                      selection scored; for the offline selection, the first
                      online window.
  adaptive_threshold  re-optimises when a score falls below mean - k * std of
                      the last N scores (--adaptive_window, --adaptive_k). The
                      history restarts whenever a re-optimisation is applied;
                      until it holds N scores again, the threshold rule above
                      (same X) stands in for it.

Two signals are tracked side by side — the deployed GA ensemble's fitness and
the deployed single model's score, both on the run's decision metric — each
with its own baseline and history. A window triggers when EITHER is below its
threshold. There is no extra cooldown.

Switching in (`ReoptSchedule`): a re-optimisation started by a trigger at
window t is switched in at exactly window t + L (--reopt_delay, default 2):
window t + L is the first window the new selection scores, and it becomes the
new baseline. The runner waits there if the result is not ready, so WHEN a
result is switched in never depends on how fast the machine is. Every trigger
from t up to (not including) t + L is suppressed — recorded, not queued. A
re-optimisation whose switch-in window lies after the last window is never
switched in. L = 0 switches the result in at the triggering window itself,
which is then scored again with the new selection (`TriggerPolicy.observe`
takes that score as the new baseline without judging it).

Choices the rules above leave open, made here once:

* A threshold is computed from the scores BEFORE the current window, then the
  current score is compared with it. A score never moves its own threshold.
* The window that sets a baseline has no threshold of its own (it would be
  compared with itself); its `threshold` is reported as None.
* The standard deviation is the population one (ddof=0), numpy's default.
* "Below" is strict and ignores float rounding (1e-12): a score equal to its
  threshold does not trigger.
* A score that could not be computed (None or NaN — e.g. a window whose labels
  are all one class) is not compared and does not enter the history; a signal
  whose baseline window was unscorable takes its next scorable window instead.
* A re-optimisation that failed (or timed out) is resolved at its switch-in
  window like any other: the triggers before it stay suppressed, the current
  selection and baselines are kept.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from Utils.pipeline_spec import (DEFAULT_ADAPTIVE_K, DEFAULT_ADAPTIVE_WINDOW,
                                 DEFAULT_DROP_PCT, DEFAULT_REOPT_DELAY,
                                 DEFAULT_UPDATE_INTERVAL, ONLINE_BASELINE_SCENARIOS,
                                 ONLINE_SCENARIOS)

# The two tracked signals, in the order a trigger reason names them.
SIGNALS = ("ensemble", "single")

MODE_RELATIVE = "relative"
MODE_ADAPTIVE = "adaptive"
MODE_PERIODIC = "periodic"

REASON_PERIODIC = "periodic"
REASON_RELATIVE = "relative_drop"
REASON_ADAPTIVE = "adaptive_drop"

# The only reason a trigger can fail to start a re-optimisation.
SUPPRESSED_PENDING = "reoptimization_in_progress"

# "Below" means below by more than rounding. A steady series has a rolling
# std of zero, so its adaptive threshold IS its mean — and the float sum behind
# that mean can land a hair above the scores it averaged, which would read
# every unchanged window as a drop.
_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PolicyParams:
    update_interval: int = DEFAULT_UPDATE_INTERVAL     # M
    drop_pct: float = DEFAULT_DROP_PCT                 # X, percent
    adaptive_window: int = DEFAULT_ADAPTIVE_WINDOW     # N
    adaptive_k: float = DEFAULT_ADAPTIVE_K             # k
    reopt_delay: int = DEFAULT_REOPT_DELAY             # L


def is_score(value) -> bool:
    """A comparable score: a finite number."""
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def relative_threshold(baseline: float, drop_pct: float) -> float:
    """The score below which a drop counts: baseline * (1 - X/100)."""
    return float(baseline) * (1.0 - float(drop_pct) / 100.0)


def adaptive_threshold(scores: Sequence[float], k: float) -> float:
    """mean - k * std of `scores` (population std)."""
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    return mean - float(k) * math.sqrt(var)


@dataclass
class SignalState:
    """One signal's scores since the last reset; the first is its baseline."""
    history: List[float] = field(default_factory=list)

    @property
    def baseline(self) -> Optional[float]:
        return self.history[0] if self.history else None

    def reset(self) -> None:
        self.history = []


@dataclass
class Decision:
    """What one window asked for, and the thresholds that decided it."""
    window: int
    thresholds: Dict[str, Optional[float]]
    modes: Dict[str, Optional[str]]
    baselines: Dict[str, Optional[float]]
    below: List[str]
    fire: bool                         # the rule asks for a re-optimisation
    start: bool                        # ...and one may start now
    reason: Optional[str]
    suppressed_because: Optional[str]
    text: Optional[str] = None         # human-readable reason, for the log

    @property
    def mode(self) -> Optional[str]:
        """The scenario-level mode for this window.

        'periodic' and None (fixed) are the scenario's own. For the threshold
        scenarios it is the signals' common mode; while one signal is still
        warming up and the other is not, the warm-up rule is still in force
        for the window, so it reads 'relative'. Per-signal modes are in
        `modes`.
        """
        values = {m for m in self.modes.values() if m is not None}
        if not values:
            return None
        if len(values) == 1:
            return values.pop()
        return MODE_RELATIVE


class TriggerPolicy:
    """The trigger rule of one scenario, with its per-signal state."""

    def __init__(self, scenario: str, params: PolicyParams = PolicyParams()):
        if scenario not in ONLINE_SCENARIOS and scenario not in ONLINE_BASELINE_SCENARIOS:
            raise ValueError(f"unknown online scenario: {scenario!r}")
        self.scenario = scenario
        self.params = params
        self.signals: Dict[str, SignalState] = {s: SignalState() for s in SIGNALS}

    # -- description ---------------------------------------------------------

    def params_dict(self) -> Dict[str, float]:
        """The parameters this scenario actually uses, for the results file."""
        p = self.params
        if self.scenario == "periodic":
            return {"update_interval": p.update_interval, "reopt_delay": p.reopt_delay}
        if self.scenario == "threshold":
            return {"drop_pct": p.drop_pct, "reopt_delay": p.reopt_delay}
        if self.scenario == "adaptive_threshold":
            return {"adaptive_window": p.adaptive_window, "adaptive_k": p.adaptive_k,
                    "drop_pct": p.drop_pct, "reopt_delay": p.reopt_delay}
        return {}

    @property
    def reoptimizes(self) -> bool:
        return self.scenario in ("periodic", "threshold", "adaptive_threshold")

    # -- the rule ------------------------------------------------------------

    def reset(self) -> None:
        """A re-optimisation was applied: baselines and histories start over."""
        for state in self.signals.values():
            state.reset()

    def observe(self, scores: Dict[str, Optional[float]]) -> None:
        """Remember `scores` without judging them.

        For a window scored a second time, with a selection switched in at
        that same window (reopt_delay = 0): its first score already had its
        say, and the second one is the new selection's baseline.
        """
        if self.scenario not in ("threshold", "adaptive_threshold"):
            return
        for name in SIGNALS:
            if is_score(scores.get(name)):
                self.signals[name].history.append(float(scores[name]))

    def baselines(self) -> Dict[str, Optional[float]]:
        return {name: self.signals[name].baseline for name in SIGNALS}

    def _threshold(self, state: SignalState):
        """(threshold, mode) for the next score of `state`, before it is seen."""
        p = self.params
        if (self.scenario == "adaptive_threshold"
                and len(state.history) >= p.adaptive_window):
            recent = state.history[-p.adaptive_window:]
            return adaptive_threshold(recent, p.adaptive_k), MODE_ADAPTIVE
        if state.baseline is None:
            return None, MODE_RELATIVE
        return relative_threshold(state.baseline, p.drop_pct), MODE_RELATIVE

    def decide(self, window: int, scores: Dict[str, Optional[float]],
               pending: bool) -> Decision:
        """Judge window `window` from its scores, then remember them.

        `pending` says a re-optimisation is still running; a trigger then
        starts nothing and is reported as suppressed.
        """
        thresholds = {s: None for s in SIGNALS}
        modes = {s: None for s in SIGNALS}
        baselines = {s: None for s in SIGNALS}
        below: List[str] = []
        fire = False
        reason = text = None

        if self.scenario == "periodic":
            modes = {s: MODE_PERIODIC for s in SIGNALS}
            # Not at window 0: the selection it would replace was made on the
            # data just before it.
            fire = window > 0 and window % self.params.update_interval == 0
            if fire:
                reason = REASON_PERIODIC
                text = f"window {window} is a multiple of {self.params.update_interval}"

        elif self.scenario in ("threshold", "adaptive_threshold"):
            parts = []
            for name in SIGNALS:
                state = self.signals[name]
                threshold, mode = self._threshold(state)
                thresholds[name], modes[name] = threshold, mode
                score = scores.get(name)
                if not is_score(score):
                    baselines[name] = state.baseline
                    continue
                score = float(score)
                if threshold is not None and score < threshold - _TOLERANCE:
                    below.append(name)
                    parts.append(f"{name} {score:.4f} < {threshold:.4f} ({mode})")
                state.history.append(score)
                baselines[name] = state.baseline
            if below:
                fire = True
                first_mode = modes[below[0]]
                reason = REASON_ADAPTIVE if first_mode == MODE_ADAPTIVE else REASON_RELATIVE
                text = "; ".join(parts)

        start = fire and not pending
        return Decision(window=window, thresholds=thresholds, modes=modes,
                        baselines=baselines, below=below, fire=fire, start=start,
                        reason=reason,
                        suppressed_because=(SUPPRESSED_PENDING if fire and pending else None),
                        text=text)


def apply_window(trigger_window: int, reopt_delay: int) -> int:
    """The window a re-optimisation started at `trigger_window` is switched in at."""
    return int(trigger_window) + int(reopt_delay)


@dataclass(frozen=True)
class PendingReopt:
    """A started re-optimisation that has not been switched in yet."""
    started_at_window: int
    apply_window: int


class ReoptSchedule:
    """Which re-optimisation is pending, and the window it is due at.

    One at a time: while one is pending a trigger starts nothing (it is
    suppressed). Pending lasts from the triggering window up to, not including,
    its switch-in window, whatever the re-optimisation itself does meanwhile —
    finishing early, failing or timing out changes nothing until then, so the
    timeline is the same on a fast machine and a slow one.
    """

    def __init__(self, reopt_delay: int = DEFAULT_REOPT_DELAY):
        if int(reopt_delay) < 0:
            raise ValueError(f"reopt_delay must be 0 or more, got {reopt_delay}")
        self.reopt_delay = int(reopt_delay)
        self.pending: Optional[PendingReopt] = None

    @property
    def is_pending(self) -> bool:
        return self.pending is not None

    def start(self, window: int) -> PendingReopt:
        if self.pending is not None:
            raise RuntimeError("a re-optimisation is already pending")
        self.pending = PendingReopt(int(window), apply_window(window, self.reopt_delay))
        return self.pending

    def due(self, window: int) -> Optional[PendingReopt]:
        """The pending re-optimisation if it is to be switched in at `window`."""
        if self.pending is not None and self.pending.apply_window <= int(window):
            return self.pending
        return None

    def resolve(self) -> PendingReopt:
        """Close the pending re-optimisation (switched in, failed or dropped)."""
        done, self.pending = self.pending, None
        if done is None:
            raise RuntimeError("no re-optimisation is pending")
        return done


@dataclass
class TimelineStep:
    """One window of `simulate_timeline`: what the rule asked and what happened."""
    window: int
    decision: Decision
    applied: Optional[PendingReopt]        # switched in BEFORE this window scored
    started: Optional[PendingReopt]        # started by this window's trigger
    rescored: bool = False                 # reopt_delay 0: switched in at this window


def simulate_timeline(scenario: str, series: Sequence[Dict[str, Optional[float]]],
                      params: PolicyParams = PolicyParams(), first_window: int = 0,
                      fails: Sequence[int] = ()) -> List[TimelineStep]:
    """The whole timeline of a scenario over a score series, every
    re-optimisation taking effect exactly `params.reopt_delay` windows after
    its trigger, as the runner does.

    The score of a window does not depend on which selection is deployed here
    (`series` is the observed series); `fails` lists trigger windows whose
    re-optimisation fails, which keeps the selection and the baselines.
    Re-optimisations still pending after the last window are not applied.
    """
    policy = TriggerPolicy(scenario, params)
    schedule = ReoptSchedule(params.reopt_delay)
    failing = set(fails)
    out: List[TimelineStep] = []
    for offset, scores in enumerate(series):
        window = first_window + offset
        applied = None
        due = schedule.due(window)
        if due is not None:
            schedule.resolve()
            if due.started_at_window not in failing:
                policy.reset()
                applied = due
        decision = policy.decide(window, scores, schedule.is_pending)
        started = rescored = None
        if decision.start:
            started = schedule.start(window)
            if schedule.due(window) is not None:          # reopt_delay == 0
                schedule.resolve()
                if window not in failing:
                    policy.reset()
                    policy.observe(scores)
                    rescored = True
        out.append(TimelineStep(window, decision, applied, started, bool(rescored)))
    return out


def simulate(scenario: str, series: Sequence[Dict[str, Optional[float]]],
             params: PolicyParams = PolicyParams(), first_window: int = 0,
             applied_at: Sequence[int] = (), pending_at: Sequence[int] = ()
             ) -> List[Decision]:
    """Run the trigger rule alone over a score series; for tests and what-ifs.

    `applied_at` lists window indices at which a re-optimisation is applied
    BEFORE that window is scored (so the window becomes the new baseline);
    `pending_at` lists windows during which one is still pending. For the
    timeline the runner actually follows, see `simulate_timeline`.
    """
    policy = TriggerPolicy(scenario, params)
    applied, pending = set(applied_at), set(pending_at)
    out = []
    for offset, scores in enumerate(series):
        window = first_window + offset
        if window in applied:
            policy.reset()
        out.append(policy.decide(window, scores, window in pending))
    return out
