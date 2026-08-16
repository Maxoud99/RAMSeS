import importlib
import pkgutil

import pyod.models as pyod_models
from Algorithms.base_model import PyMADModel
from Algorithms import windowed
import torch as t
from Utils.utils import de_unfold
import numpy as np


# this is Stochastic Outlier Selection class
def get_all_module_names(library):
    module_names = []
    for _, module_name, _ in pkgutil.walk_packages(library.__path__, prefix=library.__name__ + '.'):
        module_names.append(module_name)
    return module_names


# PyOD 3's time-series detectors live under module names that cannot be derived
# from the class name — `LSTMAD` is in `pyod.models.ts_lstm`, not
# `pyod.models.lstmad` — so the derivation below has no way to reach them. A map
# of seven entries beats importing all sixty-nine modules to search for a class,
# and `Utils/test_pipeline_spec` fails if one of these ever stops resolving.
_TS_MODULES = {
    # Not a time-series detector, but here for the same reason: the pool spells
    # it AE and no amount of case- or underscale-folding gets from "AE" to
    # "auto_encoder". `AUTOENCODER` used to reach it through the fallback below;
    # the abbreviation cannot.
    "AE": "pyod.models.auto_encoder",
    "LSTMAD": "pyod.models.ts_lstm",
    "ANOMALYTRANSFORMER": "pyod.models.ts_anomaly_transformer",
    "MATRIXPROFILE": "pyod.models.ts_matrix_profile",
    "SR": "pyod.models.ts_spectral_residual",
    "KSHAPE": "pyod.models.ts_kshape",
    "TIMESERIESOD": "pyod.models.ts_od",
    "SAND": "pyod.models.ts_sand",
}

# Families whose pool name is not the PyOD class name. `_class_in` matches
# case- and underscore-insensitively, which reaches `IForest` from `IFOREST`,
# but cannot reach `SpectralResidual` from `SR` — nothing in the string says
# so. One entry rather than a general abbreviation scheme, because a general
# one would start guessing.
_TS_CLASSES = {
    "SR": "SpectralResidual",
    "AE": "AutoEncoder",
}


def _module_for(model_name, all_pyod_modules):
    """The pyod.models module holding `model_name`, or None.

    `pyod.models.{name.lower()}` covers the acronyms (ABOD, HBOS, OCSVM) but not
    the names whose module carries an underscore the class does not:
    `AutoEncoder` lives in `pyod.models.auto_encoder`, `DeepSVDD` in
    `pyod.models.deep_svdd`. Falling back to a case- and underscore-insensitive
    match reaches those without hardcoding a table that would go stale on the
    next pyod release.
    """
    mapped = _TS_MODULES.get(model_name.upper().replace('_', ''))
    if mapped in all_pyod_modules:
        return mapped
    direct = f'pyod.models.{model_name.lower()}'
    if direct in all_pyod_modules:
        return direct
    wanted = model_name.lower().replace('_', '')
    for candidate in all_pyod_modules:
        if candidate.rsplit('.', 1)[-1].replace('_', '') == wanted:
            return candidate
    return None


def _class_in(module, model_name):
    """The estimator class inside `module`, matched without guessing its case.

    The old code did `getattr(module, model_name.upper())`, which finds ABOD and
    HBOS and misses every mixed-case name pyod ships — `IForest` is in
    `pyod.models.iforest` but is not `IFOREST`, so IForest, AutoEncoder and
    DeepSVDD were all unreachable.
    """
    aliased = _TS_CLASSES.get(model_name.upper().replace('_', ''))
    if aliased is not None:
        obj = getattr(module, aliased, None)
        if isinstance(obj, type):
            return obj
    exact = getattr(module, model_name, None)
    if isinstance(exact, type):
        return exact
    wanted = model_name.lower().replace('_', '')
    for attr in dir(module):
        obj = getattr(module, attr, None)
        if isinstance(obj, type) and attr.lower().replace('_', '') == wanted:
            return obj
    return None


def create_model(model_name, all_pyod_modules, **kwargs):
    module_name = _module_for(model_name, all_pyod_modules)
    if module_name is None:
        raise ValueError(f"Invalid model name: {model_name}")
    module = importlib.import_module(module_name)
    model_class = _class_in(module, model_name)
    if model_class is None:
        raise ValueError(f"Model class {model_name} not found in {module_name}")
    try:
        # Instantiate the model class with kwargs
        return model_class(**kwargs)
    except TypeError as e:
        raise ValueError(f"Error instantiating model {model_name} with provided arguments: {e}")


class PyodModel(PyMADModel):

    def __init__(self, model_name: str, window_size=1, window_step=1, contamination=0.1,
                 device=None, detector_kwargs=None):
        '''

        :param window_size:
        :param window_step:
        :param contamination:contaminationfloat in (0., 0.5), optional (default=0.1)
            The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function.
        :param device:
        '''
        super(PyodModel, self).__init__()

        self.contamination = contamination
        pyod_modules = get_all_module_names(pyod_models)
        # `detector_kwargs` are the PyOD estimator's own parameters, kept apart
        # from the framework's `window_size`/`window_step` because the names
        # collide: LSTMAD and AnomalyTransformer each take a `window_size` of
        # their own, meaning the subsequence they model internally, while ours
        # means how the loader cuts the series before they see it. The grids
        # spell theirs `detector__window_size`.
        self.detector_kwargs = dict(detector_kwargs or {})
        self.model = create_model(model_name, all_pyod_modules=pyod_modules,
                                  contamination=self.contamination, **self.detector_kwargs)
        self.window_size = window_size
        self.window_step = window_step
        self.device = device

    def fit(self, train_dataloader):
        windowed.fit_windows(self.model, train_dataloader)

    def forward(self, input):
        """Y_hat scaled by the window's own anomaly score.

        Kept only because `predict` asks every detector for a reconstruction;
        the ranking reads `window_anomaly_score`, not this.
        """
        Y = input['Y']
        n_batches, n_features, n_time = Y.shape
        scores = windowed.score_windows(self.model, Y)
        Y_hat = Y.detach().cpu().numpy() * scores.reshape(n_batches, 1, 1)
        return input['Y'], t.from_numpy(Y_hat), input['mask']

    def training_step(self, input):

        Y, Y_hat, mask = self.forward(input=input)

        loss = t.mean((mask * (Y - Y_hat)) ** 2)

        return loss

    def eval_step(self, x):
        self.model.eval()
        loss = self.training_step(x)
        return loss

    def window_anomaly_score(self, input, return_detail: bool = False):
        Y = input['Y']
        n_batches, n_features, n_time = Y.shape
        scores = windowed.score_windows(self.model, Y)
        anomaly_score = windowed.broadcast_to_window(scores, n_batches, n_features, n_time)
        if return_detail:
            return anomaly_score
        return t.mean(anomaly_score, dim=0)

    def final_anomaly_score(self, input, return_detail: bool = False):

        # Average anomaly score for each feature per timestamp
        anomaly_scores = de_unfold(windows=input, window_step=self.window_step)

        if return_detail:
            return anomaly_scores
        else:
            anomaly_scores = t.mean(anomaly_scores, dim=0)
            return anomaly_scores
