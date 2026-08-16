"""Chronos as an anomaly detector, over `chronos-forecasting` directly.

Why this is written here rather than vendored like OFA and TimesFM
------------------------------------------------------------------
TSB-AD ships a `Chronos.py`, but it reaches the model through
`autogluon.timeseries` — a 69-package dependency, against 17 for the
`chronos-forecasting` package Amazon actually publishes the model under.
Nothing in those extra 52 packages is Chronos; they are AutoGluon's own
tabular/forecasting stack. So the model comes from upstream and only the
forecast-to-score step is written here.

That step is deliberately TSB-AD's, not an invention of ours, so the detector
means the same thing it does in the benchmark:

    for each channel:
        cut sliding windows of `win_size`, target = the next point
        forecast one step from each window
        score = (target - forecast)^2
    average the per-channel scores, then left-pad the first
    `win_size + prediction_length - 1` positions with the first score

`Chronos.py:33-71` in the vendored tree is that loop; this reproduces it with
`predict_quantiles(..., quantile_levels=[0.5])` — the median forecast — where
AutoGluon returned its `'mean'` column.

Two choices worth stating
-------------------------
* **Chronos-Bolt, not Chronos-T5 — because T5 is not reproducible.** T5
  forecasts by sampling from a token distribution, and two calls on identical
  input differ: measured 1.719e-01 apart on the same fitted pipeline, and
  4.871e-01 across a whole scoring call. That is precisely the property this
  pool already refuses detectors for — `Utils.pipeline_spec` excludes PyOD's
  TimeSeriesOD and AnomalyTransformer with the words "return DIFFERENT scores
  on two runs of identical input". Bolt does direct quantile regression with no
  sampling step and measures 0.000e+00 between calls, so it is the only variant
  that can be admitted on the same terms as everything else here. It is also
  Amazon's successor to T5, published in the same package, and faster.
* **`tiny`, not `base`.** One of the sizes Bolt ships, and the only one sensible
  on CPU: ~180 one-step forecasts/second, so SKAB (229 test rows x 9 channels)
  scores in about 11 seconds. `base` is roughly an order of magnitude slower,
  and this pipeline scores every detector many times per run.
* **Multivariate is allowed**, by the per-channel loop above. Table I marks
  Chronos `U` and TSB-AD lists it only in its univariate hyperparameter dicts,
  so a multivariate run is a configuration the paper does not report — but the
  aggregation is upstream's own code, not something invented here, which is
  what separates this from POLY (whose `np.polyfit` genuinely cannot see more
  than one channel).

The class presents `fit(data)` / `decision_function(data)` over
`(n_timesteps, n_channels)`, which is the TSB-AD contract, so
`Algorithms/tsbad_model.py` adapts it with no special case.
"""

import numpy as np


class Chronos:
    """One-step-ahead forecast error from a pretrained Chronos model.

    Fitting is a formality — the model is pretrained and frozen, and nothing is
    learned from the training series. That is a property of foundation models
    rather than of this wrapper, and it is the same situation the transductive
    families are in: `fit` exists so the family trains, saves and loads like
    every other detector.
    """

    def __init__(self, win_size=100, model_size="tiny", prediction_length=1,
                 input_c=1, batch_size=128, contamination=0.1):
        self.win_size = int(win_size)
        self.model_size = str(model_size)
        self.prediction_length = int(prediction_length)
        self.input_c = int(input_c)
        self.batch_size = int(batch_size)
        self.contamination = contamination
        self.decision_scores_ = None
        self._pipeline = None

    def _pipe(self):
        """Loaded on first use, not in __init__.

        The weights are ~40 MB and come from the HuggingFace hub, so building
        the estimator to read its parameters — which `_TSBADEstimator` does on
        every construction — must not trigger a download.
        """
        if self._pipeline is None:
            import torch
            from chronos import BaseChronosPipeline
            self._pipeline = BaseChronosPipeline.from_pretrained(
                f"amazon/chronos-bolt-{self.model_size}",
                device_map="cpu", torch_dtype=torch.float32)
        return self._pipeline

    def fit(self, data, y=None):
        self.decision_scores_ = self.decision_function(data)
        return self

    def decision_function(self, data):
        import torch
        X = np.asarray(data, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]
        n_time, n_channels = X.shape
        win, horizon = self.win_size, self.prediction_length
        if n_time <= win + horizon - 1:
            raise ValueError(
                f"CHRONOS needs more than win_size + prediction_length - 1 "
                f"({win + horizon - 1}) rows per call and was handed {n_time}. "
                f"Thompson cuts windows of n_timesteps * 0.8 / iterations, so a "
                f"short entity puts it below this; every other stage scores the "
                f"whole series and clears it.")
        pipe = self._pipe()
        per_channel = []
        for channel in range(n_channels):
            series = X[:, channel]
            windows = np.lib.stride_tricks.sliding_window_view(series, win)[:-horizon]
            targets = series[win + horizon - 1:]
            forecasts = []
            for start in range(0, len(windows), self.batch_size):
                chunk = torch.tensor(np.ascontiguousarray(windows[start:start + self.batch_size]))
                quantiles, _mean = pipe.predict_quantiles(
                    chunk, prediction_length=horizon, quantile_levels=[0.5])
                # (batch, horizon, 1) -> the median forecast of the last step.
                forecasts.append(np.asarray(quantiles[:, -1, 0], dtype=float))
            predicted = np.concatenate(forecasts) if forecasts else np.zeros(0)
            per_channel.append((targets[:len(predicted)] - predicted) ** 2)
        scores = np.mean(np.asarray(per_channel), axis=0)
        # Left-pad so there is one score per timestep: the first `win` points
        # have no window behind them to forecast from.
        padded = np.zeros(n_time, dtype=float)
        pad = win + horizon - 1
        padded[:pad] = scores[0] if len(scores) else 0.0
        padded[pad:pad + len(scores)] = scores
        return padded
