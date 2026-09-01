"""TSB-AD detectors, adapted to the framework's `PyMADModel` interface.

These are the eight Table I models PyOD does not ship: the six neural networks
(Donut, OmniAnomaly, USAD, TranAD, FITS, TimesNet) and two statistical ones
(KMeansAD, POLY). The code they run is vendored under `Algorithms/tsb_ad`; see
that package's docstring for why it is vendored rather than depended on.

Three mismatches with PyOD, which is what this module exists to absorb:

  1. **Shape.** A PyOD estimator takes rows and scores rows. A TSB-AD detector
     takes the whole series as `(n_timesteps, n_channels)` and returns one score
     per TIMESTEP, cutting its own subsequences internally. That is the same
     contract PyOD 3's LSTMAD has, and it is handled the same way: the framework
     window_size stays 1 so one row IS one timestep, and the detector's own
     subsequence length is a `detector__` grid key. With window_size 1 the two
     readings of "a row" coincide and no reshaping is needed at all.

  2. **Method names.** Most spell the scorer `decision_function`; KMeansAD
     spells it `predict`; POLY has neither in a usable form (see below).

  3. **Constructor.** The channel count is a required positional-ish argument
     under three different names (`feats`, `input_c`, `enc_in`) and cannot be
     known until fit sees the data, so the estimator is built lazily in `fit`
     rather than in `__init__` as `PyodModel` does. `contamination` is passed
     only to the constructors that accept it — most of these do not have one.

POLY is the awkward one, and is admitted with two restrictions that are
properties of the detector, not of this pipeline:

  * **Univariate only.** It runs `np.polyfit` on the raw series and raises
    "Polynomial must be 1d only" on anything wider. Table I marks it `U` for
    exactly this reason. It therefore works on UCR and is unavailable on SKAB
    and SMD, and `fit` says so in those words rather than letting numpy raise.
  * **Refit per call.** Its `decision_function` needs a `measure` object the
    class never stores, and scores against `self.estimation`/`self.n_train_`
    from the last fit, so it cannot score data it was not just fitted on. Its
    real interface is `fit(X)` then read `decision_scores_`. Measured: `fit(B)`
    then `fit(A)` gives byte-identical scores to `fit(A)` alone (max difference
    0.000e+00), i.e. fit state is fully replaced. So scoring refits, which makes
    the score a function of the call's rows alone — the COF/SOS/SpectralResidual
    property, and
    why POLY joins TRANSDUCTIVE_FAMILIES.
"""

import importlib
import inspect
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import torch as t

from Algorithms.base_model import PyMADModel
from Algorithms import windowed
from Utils.utils import de_unfold


# family -> (module, class, channel-count kwarg, how to score). A bare name is
# a module under `Algorithms.tsb_ad.models`; a dotted path is taken as-is,
# which is how CHRONOS reaches `Algorithms.chronos_detector` — that one is
# written here rather than vendored, because TSB-AD's Chronos.py goes through
# autogluon's 69 packages to reach a model Amazon publishes in 17. None as the
# channel kwarg means the constructor infers the width from the data.
#
# The scorers:
#   'decision_function' — the PyOD-shaped default.
#   'predict'           — KMeansAD subclasses sklearn's OutlierMixin, not PyOD's
#                         BaseDetector, so its per-timestep scores (already
#                         reverse-windowed from its internal sliding window)
#                         come out of `predict`. It has no decision_function.
#   'refit'             — POLY. See the module docstring.
_TSBAD_SPECS = {
    "KMEANSAD":    ("KMeansAD",    "KMeansAD",    None,      "predict"),
    "POLY":        ("POLY",        "POLY",        None,      "refit"),
    "DONUT":       ("Donut",       "Donut",       "input_c", "decision_function"),
    "OmniAnomaly": ("OmniAnomaly", "OmniAnomaly", "feats",
                    "decision_function"),
    "USAD":        ("USAD",        "USAD",        "feats",   "decision_function"),
    "TRANAD":      ("TranAD",      "TranAD",      "feats",   "decision_function"),
    "FITS":        ("FITS",        "FITS",        "input_c", "decision_function"),
    "TIMESNET":    ("TimesNet",    "TimesNet",    "enc_in",  "decision_function"),
    # Table I's Foundation Models. OFA embeds the channel count natively
    # (`enc_in`), which is why Table I marks it U&M; TIMESFM and CHRONOS score
    # one channel at a time and average, which is upstream's own loop. That loop
    # is why TIMESFM is now univariate-only — see UNIVARIATE_ONLY below — while
    # CHRONOS keeps it, Bolt-tiny being fast enough that the cost never bites.
    "OFA":         ("OFA",         "OFA",         "enc_in",  "decision_function"),
    # 'refit', not 'decision_function': TSB-AD's TimesFM computes its scores in
    # `fit` and leaves `decision_function` as a bare `pass` returning None —
    # the same fit-and-read-`decision_scores_` shape POLY has. Being a frozen
    # pretrained model, refitting learns nothing; it only recomputes forecasts.
    "TIMESFM":     ("TimesFM",     "TimesFM",     "input_c", "refit"),
    "CHRONOS":     ("Algorithms.chronos_detector", "Chronos", "input_c",
                    "decision_function"),
    # The Graph Based group's two members. Both take the dotted form for the
    # same reason CHRONOS does — they are written under `Algorithms/` rather
    # than vendored — and Series2Graph's adapter is also where the "not fetched
    # yet" error lives, that one file being deliberately absent from here.
    #
    # 'refit' for Series2Graph: the vendored `score(query_length, dataset)`
    # never reads `dataset`, so it re-reports the series `fit` built its graph
    # from. See TRANSDUCTIVE_FAMILIES in Utils/pipeline_spec.py.
    "MTADGAT":     ("Algorithms.mtad_gat", "MTADGAT", "n_features",
                    "decision_function"),
    "Series2Graph": ("Algorithms.series2graph_detector",
                     "Series2GraphDetector", None, "refit"),
}

# Families that cannot see more than one channel. Checked in `fit` so the
# failure names the detector and the dataset shape instead of surfacing as
# numpy's "Polynomial must be 1d only" from four frames down.
# Families refused on multivariate entities, each with the clause saying why.
#
# A dict rather than a set because the two members are restricted for genuinely
# different reasons, and one sentence covering both would be false of one of
# them: POLY *cannot* see more than one channel, TimesFM *can* but must not.
#
# TimesFM's per-channel loop works — this is a cost refusal, not a capability
# one. It forecasts one step from every sliding window of every channel through
# a 200M-parameter model, measured at ~131 forecasts/s on CPU, so a 38-channel
# SMD entity costs ~13 minutes PER SCORING CALL and the offline pipeline makes
# seven of them: ~1.5 h for a single instance, against ~0.6 s for Chronos-Bolt
# and ~1.5 s for OFA on identical input. Table I marks it 'U' and TSB-AD lists
# it only in the univariate hyperparameter dicts, so the multivariate run was
# never a configuration the paper reports.
UNIVARIATE_ONLY = {
    "POLY": "it fits a polynomial to the raw series, so a multivariate entity "
            "has no meaning for it",
    "TIMESFM": "it forecasts each channel separately, which costs roughly 13 "
               "minutes per scoring call on a 38-channel entity — the "
               "per-channel loop runs, but it is not a configuration Table I "
               "reports",
    "Series2Graph": "it embeds a scalar subsequence into a 2-D phase space and reads a "
           "graph off that embedding, so a multivariate entity has no meaning "
           "for it — TSB-AD's own wrapper opens with `data.squeeze()` and "
           "lists it in the uni-variate pool",
}

# For the 'refit' scorers, the constructor key that sets the minimum rows per
# call, and its default. They spell it differently — POLY fits one polynomial
# per `window`, TimesFM forecasts one step from each `win_size` context — and a
# check reading the wrong key would compare against a number the detector never
# uses. Series2Graph is absent deliberately: its minimum is 2 * pattern_length,
# which no single key expresses, and it checks and reports its own.
_MIN_LENGTH_KEYS = {
    "POLY": ("window", 200),
    "TIMESFM": ("win_size", 96),
}

# Attributes these models keep for TRAINING only, dropped once `fit` returns.
#
# Not an optimisation — a checkpoint cannot be written otherwise. OFA's
# `model_optim` holds a reference reaching a torch config module, and neither
# pickle nor dill can serialise one: `TypeError: cannot pickle
# 'ConfigModuleInstance' object`, raised from `logging_obj.save` after training
# had already finished. Verified across the vendored models that every one of
# these names is touched only inside `fit` — `zero_grad`, `step`,
# `adjust_learning_rate` — and never by `decision_function`, so dropping them
# costs nothing at scoring time and makes the checkpoints smaller besides.
_TRAINING_ONLY_ATTRS = ("model_optim", "optimizer", "scheduler")


# Case- and underscore-folded index over the specs above. The pool spells every
# family exactly as its key, so the exact name always hits; this only keeps the
# lookup tolerant of a differently-cased spelling reaching it from elsewhere.
_SPECS_BY_FOLDED = {k.upper().replace("_", ""): v for k, v in _TSBAD_SPECS.items()}


def _class_for(family: str):
    """The vendored estimator class for a pool family name."""
    spec = _TSBAD_SPECS.get(family) or _SPECS_BY_FOLDED.get(
        str(family).upper().replace("_", ""))
    if spec is None:
        raise ValueError(
            f"{family} is not a TSB-AD family. Known: "
            f"{', '.join(sorted(_TSBAD_SPECS))}")
    module_name, class_name, channel_arg, scorer = spec
    if "." not in module_name:
        module_name = f"Algorithms.tsb_ad.models.{module_name}"
    module = importlib.import_module(module_name)
    return getattr(module, class_name), channel_arg, scorer


class _TSBADEstimator:
    """A PyOD-shaped `fit(X)` / `decision_function(X)` over a TSB-AD detector.

    Presenting that interface is what lets `Algorithms.windowed` handle these
    detectors unchanged — including its one-score-per-row guard and its
    non-finite handling, which are worth more here than a second scoring path
    would be. `detector_name` is read by `windowed.score_windows` so its
    messages name the detector rather than this wrapper.
    """

    def __init__(self, family: str, contamination: float, detector_kwargs: dict):
        # Canonical, not upper-cased: the pool name IS the family name, and
        # four of them are mixed case (`OmniAnomaly`, `Series2Graph`). Folding
        # here would miss every dict below that is keyed by it.
        self.family = str(family)
        self.detector_name = self.family
        self.contamination = contamination
        self.detector_kwargs = dict(detector_kwargs or {})
        self._cls, self._channel_arg, self._scorer = _class_for(family)
        self._model = None
        self.n_channels_ = None

    def _build(self, n_channels: int):
        kwargs = dict(self.detector_kwargs)
        if self._channel_arg is not None:
            kwargs[self._channel_arg] = int(n_channels)
        # Only the constructors that declare `contamination` get it. Most of
        # these detectors have none: it would set a threshold none of them use,
        # and the pipeline scores with its own threshold sweep regardless.
        if "contamination" in inspect.signature(self._cls.__init__).parameters:
            kwargs.setdefault("contamination", self.contamination)
        return self._cls(**kwargs)

    @staticmethod
    def _as_series(X) -> np.ndarray:
        Y = np.asarray(X, dtype=np.float64)
        return Y.reshape(len(Y), -1) if Y.ndim != 2 else Y

    def _check_width(self, n_channels: int):
        if self.family in UNIVARIATE_ONLY and n_channels != 1:
            raise ValueError(
                f"{self.family} is univariate only and was handed "
                f"{n_channels} channels: {UNIVARIATE_ONLY[self.family]} — "
                f"Table I marks it 'U'. Select it on a univariate dataset "
                f"(UCR); on SKAB and SMD it is unavailable.")

    def _check_length(self, n_rows: int):
        """POLY's block count must not be zero.

        It fits one polynomial per `window` rows and computes
        `N = floor(n_rows / window)`, then takes `n_rows % N` — so a call
        shorter than one window raises `ZeroDivisionError: integer modulo by
        zero` from inside the vendored code, which names neither the detector
        nor the requirement. Same role as the minimum-length note
        `windowed.score_windows` carries for COF and SpectralResidual.
        """
        if self._scorer != "refit":
            return
        spec = _MIN_LENGTH_KEYS.get(self.family)
        if spec is None:
            return
        key, default = spec
        window = int(self.detector_kwargs.get(key, default))
        if n_rows < window:
            raise ValueError(
                f"{self.family} needs at least `{key}` ({window}) rows per "
                f"call and was handed {n_rows}. Thompson cuts windows of "
                f"n_timesteps * 0.8 / iterations, so a short entity or a high "
                f"iteration count puts it below this; every other stage scores "
                f"the whole series and clears it.")

    def fit(self, X):
        """Fit on the whole training series, one row per timestep."""
        Y = self._as_series(X)
        self.n_channels_ = Y.shape[1]
        self._check_width(self.n_channels_)
        if self._scorer == "refit":
            # Nothing to carry: POLY's fit state is fully replaced by the refit
            # that scoring performs. Building once here still proves the
            # configuration is constructible, so a bad grid fails at training
            # time rather than at the first scoring call.
            with redirect_stdout(StringIO()):
                self._model = self._build(self.n_channels_)
            return self
        with redirect_stdout(StringIO()):
            # Vendored chatter — KMeansAD prints its padding length and a
            # reverse-windowing banner on every call, the torch models print
            # per-epoch lines, and `get_gpu` announces the device from the
            # constructor, which is why building happens inside here too. tqdm
            # writes to stderr, so real progress bars still reach the terminal.
            self._model = self._build(self.n_channels_)
            self._model.fit(Y)
        self._release_training_state()
        return self

    def _release_training_state(self):
        """Drop optimiser and scheduler state so the fitted model can pickle."""
        for attr in _TRAINING_ONLY_ATTRS:
            if getattr(self._model, attr, None) is not None:
                setattr(self._model, attr, None)

    def decision_function(self, X):
        """One score per row of `X`."""
        Y = self._as_series(X)
        self._check_width(Y.shape[1])
        self._check_length(len(Y))
        with redirect_stdout(StringIO()):
            if self._scorer == "refit":
                model = self._build(Y.shape[1])
                model.fit(Y)
                scores = model.decision_scores_
            elif self._scorer == "predict":
                scores = self._model.predict(Y)
            else:
                scores = self._model.decision_function(Y)
        return np.asarray(scores, dtype=float).ravel()


class TSBADModel(PyMADModel):
    """A TSB-AD detector behind the same interface as `PyodModel`.

    The constructor signature deliberately matches `PyodModel`'s so both share
    one training loop in `TrainModels._train_wrapped`.
    """

    def __init__(self, model_name: str, window_size=1, window_step=1,
                 contamination=0.1, device=None, detector_kwargs=None):
        super(TSBADModel, self).__init__()
        self.model_name = model_name
        self.contamination = contamination
        self.detector_kwargs = dict(detector_kwargs or {})
        self.model = _TSBADEstimator(model_name, contamination, self.detector_kwargs)
        self.window_size = window_size
        self.window_step = window_step
        self.device = device

    def fit(self, train_dataloader):
        windowed.fit_windows(self.model, train_dataloader)

    def forward(self, input):
        """Y_hat scaled by the window's own anomaly score.

        Kept only because the post-fit diagnostic plot asks every detector for a
        reconstruction; the ranking reads `window_anomaly_score`.
        """
        Y = input['Y']
        n_batches, n_features, n_time = Y.shape
        scores = windowed.score_windows(self.model, Y)
        Y_hat = Y.detach().cpu().numpy() * scores.reshape(n_batches, 1, 1)
        return input['Y'], t.from_numpy(Y_hat), input['mask']

    def training_step(self, input):
        Y, Y_hat, mask = self.forward(input=input)
        return t.mean((mask * (Y - Y_hat)) ** 2)

    def eval_step(self, x):
        return self.training_step(x)

    def window_anomaly_score(self, input, return_detail: bool = False):
        Y = input['Y']
        n_batches, n_features, n_time = Y.shape
        scores = windowed.score_windows(self.model, Y)
        anomaly_score = windowed.broadcast_to_window(scores, n_batches, n_features, n_time)
        if return_detail:
            return anomaly_score
        return t.mean(anomaly_score, dim=0)

    def final_anomaly_score(self, input, return_detail: bool = False):
        anomaly_scores = de_unfold(windows=input, window_step=self.window_step)
        if return_detail:
            return anomaly_scores
        return t.mean(anomaly_scores, dim=0)
