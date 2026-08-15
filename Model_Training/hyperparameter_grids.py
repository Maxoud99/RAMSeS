#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#######################################
# DGHL Model hyper-parameter grid
#######################################

DGHL_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'train_batch_size': [64],
    'learning_rate': [1e-3],
    'seed': [1],
    # 'max_steps': [1000],
    'max_steps': [100],
    'eval_batch_size': [128],
}

DGHL_PARAM_GRID = {
    'window_size': [64],
    'window_step': [64],
    'hidden_multiplier': [32],
    'max_filters': [256],
    'kernel_multiplier': [1],
    'a_L': [1],  # Sub-windows [1, 4]
    'z_size': [25, 50],  # Size of latent z vector [5, 25, 50]
    'z_size_up': [5],
    'z_iters': [
        25, 100
    ],  # Number of iteration in the Langevyn dynamics inference formula. [5, 25, 100] -- more the better and slower. Linear time dependence. 
    'z_iters_inference': [100],  # Higher the better -> 300, 500 better. 
    'z_sigma': [0.25],
    'z_step_size': [0.1],
    'z_with_noise': [False],
    'z_persistent': [
        False
    ],  # Can only be False currently. = True means that it will start from the last latent vector when it observed the particular window. Therefore it needs higher z_iters right now. 
    'normalize_windows': [True],
    'noise_std': [0.001],
    'random_seed': [1],
    'device': [None]
}

#######################################
# Running Mean Model hyper-parameter grid
#######################################

RM_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [1],
}

RM_PARAM_GRID = {
    'window_size': [-1],  # Works with the entire time series 
    'window_step': [-1],
    'running_window_size': [4, 16, 64],
    'device': [None]
}




#######################################
# Mean Deviation Model hyper-parameter grid
#######################################

MD_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'train_batch_size': [256],
    'learning_rate': [1e-3],
    'seed': [1],
    # 'max_steps': [5000],
    'max_steps': [100],
    'eval_batch_size': [1],
}

# window_size 1 for the same reason as LOF and the PyOD families: MD has no
# temporal model to feed. `_MeanDeviation` learns one mean per feature and
# broadcasts it, so the window is context the model never looks at.
#
# Unlike those families this was never a defect — MD scores every timestep
# inside its window, so 64 cost it no score resolution. Measured on SKAB/7,
# training at 64 and at 1 gave learned means agreeing to 0.008 and score series
# correlating at 0.99996 (max difference 0.0018 on scores spanning 0.17-0.22);
# the residual is SGD batch composition, not the window. Set to 1 because it is
# what the model actually is, not because the old value was wrong.
MD_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'device': [None]
}

#######################################
# Nearest Neighbours model hyper-parameter grid
#######################################

NN_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [128],
}

NN_PARAM_GRID = {
    'window_size': [64],
    'window_step': [64],
    'n_neighbors': [1, 3, 5]
}

#######################################
# LSTM-VAE model hyper-parameter grid
#######################################

LSTMVAE_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'train_batch_size': [256],
    'learning_rate': [0.0005],
    'seed': [1],
    # 'max_steps': [1000],
    'max_steps': [100],
    'eval_batch_size': [128],
}

LSTMVAE_PARAM_GRID = {
    'window_size': [64],
    'window_step': [64],
    'hidden_size':
    [512, 256],  # hidden_size – The number of features in the hidden state h
    'latent_size': [256, 128],  # Size of the latent z 
    'num_layers': [
        4
    ],  # Number of recurrent layers. Setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM.
    'noise_std': [0.001],
    'random_seed': [1],
    'device': [None]
}

#######################################
# RNN model hyper-parameter grid
#######################################

RNN_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'train_batch_size': [1],
    'learning_rate': [0.01],
    'seed': [1],
    'max_steps': [100],
    'eval_batch_size': [1],
}

RNN_PARAM_GRID = {
    'window_size': [-1],
    'window_step': [-1],
    'input_size': [32, 64],
    'output_size': [8],
    'sample_freq': [8],
    'n_t': [0],
    'cell_type': ['LSTM'],
    'dilations': [[[1, 2], [4, 8]]],
    'state_hsize': [128, 256],  # 128
    'add_nl_layer': [False],
    'random_seed': [1],
    'device': [None]
}

#######################################
# LOF model hyper-parameter grid
#######################################


LOF_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [128],
}

# window_size 1: one sample per TIMESTEP, its channels as the feature vector.
#
# These detectors have no temporal model — they judge a sample on its own
# values — so a 64-step window bought them nothing but cost them everything
# else. On SKAB/7 it left 14 training samples in 576 dimensions (LOF's
# n_neighbors then silently clamped from 20 to 13), and the score came out
# constant over each 64-step block: 3 distinct values across 219 timesteps,
# which collapsed ALE from 10 bins to 2 and the Monte Carlo sweep from 21
# candidate thresholds to 3. At 1 it is 876 samples of 9 dimensions and 219
# distinct scores, and the cross-channel structure the old flattening threw
# away is back.
#
# NN keeps 64 on purpose: it is the pool's subsequence detector, and its own
# grid picks n_neighbors 1/3/5, which stays feasible on few windows.
# Raising this back to 64 is a one-line change per grid.
LOF_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination':[0.1,0.15,0.2,0.25],
    'device': [None]
}


#######################################
# KDE model hyper-parameter grid
#######################################


KDE_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [128],
}

KDE_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination':[0.1,0.15,0.2,0.25],
    'device': [None]
}


#######################################
# ABOD model hyper-parameter grid
#######################################


ABOD_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [128],
}

ABOD_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination':[0.1,0.15,0.2,0.25],
    'device': [None]
}


#######################################
# SOS model hyper-parameter grid
#######################################


SOS_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [128],
}

# window_size 1 for the same reason as LOF above: SOS judges a sample on its own
# values, so a 64-step window bought it nothing but the flattening that costs
# everything else.
SOS_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination':[0.1,0.15,0.2,0.25],
    'device': [None]
}


#######################################
# ALAD model hyper-parameter grid
#######################################


ALAD_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [128],
}

ALAD_PARAM_GRID = {
    'window_size': [64],
    'window_step': [64],
    'contamination':[0.1,0.15,0.2,0.25],
    'device': [None]
}


#######################################
# PYOD model hyper-parameter grid
#######################################


PYOD_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [128],
}

PYOD_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination':[0.1,0.15,0.2,0.25],
    'device': [None]
}

#######################################
# CBLOF model hyper-parameter grid
#######################################


CBLOF_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [128],
}

CBLOF_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination':[0.1,0.15,0.2,0.25],
    'device': [None]
}


#######################################
# COF model hyper-parameter grid
#######################################


COF_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [128],
}

# window_size 1, as for LOF and SOS. Note COF's own `n_neighbors` stays at
# PyOD's default 20, which sets a floor on every call: it raises IndexError when
# handed fewer than 21 rows. A whole-series call always clears that; Thompson's
# windows do not, on any entity shorter than ~1,250 timesteps.
COF_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination':[0.1,0.15,0.2,0.25],
    'device': [None]
}

#######################################
# PyOD 3 time-series detectors
#######################################
#
# These are the only detectors in the pool that model a SUBSEQUENCE, and they do
# their own windowing internally — which is why the framework's window_size is 1
# here. One row per timestep is exactly the raw multivariate series they expect;
# their own subsequence length is `detector__window_size`, forwarded to the PyOD
# constructor by `train_pyod` (the `detector__` prefix keeps it apart from ours,
# since both are spelled window_size).
#
# Contamination is not varied. It only moves a PyOD estimator's internal
# threshold, and the pipeline scores with its own threshold sweep, so four
# contaminations would be four identical detectors. The subsequence length is
# the parameter that changes what they see.
#
# Two of PyOD 3's seven time-series models are in the pool: LSTMAD here, and
# SpectralResidual on the shared PYOD_PARAM_GRID (it needs nothing of its own —
# its constructor takes `contamination` directly and it wants one row per
# timestep, which PYOD_PARAM_GRID's window_size 1 already gives it).
# SpectralResidual is transductive and is admitted on that basis; see
# `Utils.pipeline_spec.TRANSDUCTIVE_FAMILIES`. The remaining five stay out:
# MatrixProfile raises NotImplementedError by design, KShape and SAND cost
# 13-14 minutes per entity for half LOF's F1, and TimeSeriesOD and
# AnomalyTransformer are not reproducible between runs.
LSTMAD_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [128],
}

LSTMAD_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination': [0.1],
    'device': [None],
    'detector__window_size': [25, 50],
}

#######################################
# AutoEncoder (PyOD)
#######################################
#
# Table I's first Neural Network row, and the only one of its ten that PyOD
# ships. It needs no wrapper: `Algorithms.pyod_model._module_for` already
# reaches `pyod.models.auto_encoder.AutoEncoder` through the underscore-
# insensitive fallback, and `train_pyod` trains it like any other PyOD family.
#
# window_size 64 because this is a SUBSEQUENCE autoencoder — Table I's AE, and
# the variant TSB-AD builds by applying `slidingWindow` before fitting. At
# window_size 1 it would autoencode one timestep's channels in isolation, which
# is close to a nonlinear PCA and duplicates a family already in the pool. 64/64
# matches the pool's other subsequence models (NN, LSTMVAE, DGHL) rather than
# inventing a third windowing convention; a sliding 64/1 would give far more
# training samples at the cost of being the only detector here cut that way.
#
# What varies is the architecture, not `contamination`: the encoder shape and
# the epoch budget both change the reconstruction and therefore the score, which
# `contamination` — a threshold the pipeline never reads — does not.
AUTOENCODER_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [128],
}

# The three encoder shapes are TSB-AD's own AutoEncoder sweep verbatim
# (`HP_list.Multi_algo_HP_dict['AutoEncoder']['hidden_neurons']`), including the
# [128, 64] it reports as the tuned multivariate optimum. `epoch_num` stays at
# PyOD's default.
AUTOENCODER_PARAM_GRID = {
    'window_size': [64],
    'window_step': [64],
    'contamination': [0.1],
    'device': [None],
    'detector__hidden_neuron_list': [[64, 32], [32, 16], [128, 64]],
    'detector__epoch_num': [10],
}

#######################################
# TSB-AD detectors
#######################################
#
# The eight Table I models PyOD does not ship, reached through
# `Algorithms.tsbad_model.TSBADModel` over the vendored code in
# `Algorithms/tsb_ad`. Two conventions hold across all of them:
#
#   * The framework's window_size is 1. Every one of these cuts its own
#     subsequences internally, so one row per timestep is exactly the raw series
#     they expect — the same arrangement LSTMAD uses. Their own subsequence
#     length is `detector__win_size` (or `detector__window_size` for KMeansAD,
#     which spells it the same as ours, which is what the `detector__` prefix is
#     for).
#
#   * `contamination` is fixed, not swept. Six of the eight constructors do not
#     accept it at all — `_TSBADEstimator._build` passes it only where the
#     signature declares one — and where it is accepted it moves a threshold the
#     pipeline replaces with its own sweep. What varies instead is whatever
#     changes the score: the subsequence length, the cluster count, the
#     polynomial degree, the retained frequency band.
#
# One consequence to know before selecting these: a detector whose internal
# window is 30-100 timesteps cannot score a Thompson window shorter than that.
# Thompson cuts `n_timesteps * 0.8 / iterations`, so on SKAB (3-step windows)
# these accumulate no posterior updates, exactly as LSTMAD and COF already do.
# They work in every other stage, and in Thompson on long entities.
TSBAD_TRAIN_PARAM_GRID = {
    'output_dir': [r'/output'],
    'overwrite_output_dir': [True],
    'seed': [1],
    'eval_batch_size': [128],
}

# k-Means distance-to-centroid over sliding subsequences. `stride` 1 keeps the
# subsequences overlapping, which is what makes its reverse-windowing produce
# one score per timestep rather than one per block.
KMEANSAD_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination': [0.1],
    'device': [None],
    # Cluster counts from TSB-AD's sweep [10, 20, 30, 40], keeping its tuned
    # multivariate optimum (10). `window_size` 20 and `stride` 1 are what its
    # own `run_KMeansAD` passes — the class declares no defaults for either.
    'detector__k': [10, 20, 40],
    'detector__window_size': [20],
    'detector__stride': [1],
}

# Local polynomial residuals. UNIVARIATE ONLY — it fits `np.polyfit` to the raw
# series and raises on anything wider, which is why Table I marks it `U`. It is
# selectable on UCR and unavailable on SKAB and SMD; `_TSBADEstimator._check_width`
# refuses with that explanation rather than letting numpy raise from four frames
# down. It also refits on whatever it scores (its fit state is fully replaced),
# so it is transductive in the same sense as COF/SOS/SR.
# `window` is 20 rather than TSB-AD's default of 200, and that is a deliberate
# deviation. POLY fits one polynomial per `window` block and computes
# `N = floor(n_rows / window)`, so a call shorter than `window` divides by zero
# — and Thompson's windows are `n_timesteps * 0.8 / iterations`, which measured
# 71 rows on UCR 001. At 200 the detector is silent in Thompson on every entity
# (posterior norm 0.000000 for all three instances, measured); at 20 it scores
# 51 of those 71 rows. The degree is what varies.
POLY_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination': [0.1],
    'device': [None],
    'detector__power': [1, 2, 3],
    'detector__window': [20],
}

# The six neural networks. Each varies its subsequence length, the parameter
# that decides how much context it reconstructs.
#
# Every value below comes from TSB-AD's own sweep for that detector
# (`HP_list.Multi_algo_HP_dict`), taking the two LOWEST of its range rather than
# a length chosen here. The ranges genuinely differ per detector — TranAD sweeps
# [5, 10, 50] where Donut sweeps [60, 90, 120] — so a tidy shared sweep would
# have put four of the six outside the values their authors tune over.
#
# Taking the low end is deliberate and costs one thing worth naming. Thompson
# cuts windows of `n_timesteps * 0.8 / iterations`, which is 37 rows on
# SMD/machine-1-6, and a detector cannot score a window shorter than its own
# subsequence. At the low end TranAD, USAD, OmniAnomaly and TimesNet each keep
# one instance under that bound and so keep contributing to Thompson; at the
# tuned optima (50, 100, 100, 96) none of them would. Donut and FITS have no
# such option: their ranges start at 60 and 100, so they are Thompson-silent on
# this entity at any upstream value, and contribute through GA and the
# robustness stages instead.
#
# Epoch budgets are upstream's defaults. They cost less than they look: every
# one of these ships `patience=3` early stopping, so measured against 10 epochs
# the scores at 50 correlate 0.9961 (USAD), 0.9970 (TranAD), 0.9655 (Donut) and
# 0.9996 (OmniAnomaly), and only TranAD trains materially longer.
DONUT_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination': [0.1],
    'device': [None],
    'detector__win_size': [60, 90],
    'detector__num_epochs': [50],
}

OMNIANOMALY_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination': [0.1],
    'device': [None],
    'detector__win_size': [5, 50],
    'detector__epochs': [50],
}

USAD_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination': [0.1],
    'device': [None],
    'detector__win_size': [5, 50],
    'detector__epochs': [10],
}

TRANAD_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination': [0.1],
    'device': [None],
    'detector__win_size': [5, 10],
    'detector__epochs': [50],
}

# FITS interpolates in the frequency domain, so its two parameters are coupled:
# `cut_freq` frequencies are retained out of a window downsampled by `DSR` (4),
# and the linear layer is sized from both. The constraint is
#
#     cut_freq <= floor(win_size / DSR) / 2 + 1
#
# and violating it does not raise where it is set — it builds the layer at the
# wrong width and fails later inside `forward` with "mat1 and mat2 shapes cannot
# be multiplied". At win_size 100 that ceiling is 13, so both values below fit;
# win_size 24 with cut_freq 12 does not, which is why the window stays at the
# upstream default and the retained band is what varies.
#
# The window also has to clear `validation_size` (0.2) of the TRAINING series,
# since the hold-out must itself be long enough to cut one subsequence: 100 rows
# of window need 500 of training. SKAB (~700) and SMD (~2400) clear it.
#
# The consequence FITS pays for that window, measured on SMD/machine-1-6:
# Thompson cuts 37-row windows there, so FITS raises on every one and finishes
# the stage with posterior norm 0.000000 while the other eight range 0.11-0.95.
# It is not idle — the GA picked ['FITS_1', 'LOF_1'] as the best ensemble on that
# entity (F1 0.6700) — it just contributes nowhere that scores a short window.
# Dropping win_size to ~32 would buy Thompson participation at the price of
# cut_freq <= 5, a much narrower band. Left at the default.
# TSB-AD sweeps FITS over win_size [100, 200] and lr, not over cut_freq. 200
# would need 1,000 training rows to leave a validation window (SKAB has ~700),
# so the window is held at 100 — their tuned optimum — and the learning rate is
# what varies, taking the two highest of their [1e-3, 1e-4, 1e-5]. cut_freq
# stays at its default 12, which the ceiling above admits at win_size 100.
FITS_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination': [0.1],
    'device': [None],
    'detector__win_size': [100],
    'detector__cut_freq': [12],
    'detector__lr': [1e-3, 1e-4],
    'detector__epochs': [50],
}

# Table I calls this "TimeNet [87] — temporal-variation features". That is
# TimesNet: the description is TimesNet's own, and TSB-AD ships TimesNet.py with
# no TimeNet.py. Windows are the two lowest of TSB-AD's [32, 96, 192], which
# keeps 96 — its tuned optimum — while 32 stays under Thompson's 37-row window.
TIMESNET_PARAM_GRID = {
    'window_size': [1],
    'window_step': [1],
    'contamination': [0.1],
    'device': [None],
    'detector__win_size': [32, 96],
    'detector__epochs': [10],
}

# Family -> its own grid, for the generic `train_pyod` path. Anything absent
# uses PYOD_PARAM_GRID.
PYOD_MODEL_GRIDS = {
    'LSTMAD': LSTMAD_PARAM_GRID,
    'AUTOENCODER': AUTOENCODER_PARAM_GRID,
}

# Family -> its grid, for the `train_tsbad` path. Every TSB-AD family needs an
# entry: unlike the PyOD families there is no shared default, because no two of
# these detectors take the same parameters.
TSBAD_MODEL_GRIDS = {
    'KMEANSAD': KMEANSAD_PARAM_GRID,
    'POLY': POLY_PARAM_GRID,
    'DONUT': DONUT_PARAM_GRID,
    'OMNIANOMALY': OMNIANOMALY_PARAM_GRID,
    'USAD': USAD_PARAM_GRID,
    'TRANAD': TRANAD_PARAM_GRID,
    'FITS': FITS_PARAM_GRID,
    'TIMESNET': TIMESNET_PARAM_GRID,
}

#######################################
# Reading the grids back
#######################################
#
# Which grid each family's instances come from. Mirrors the branches of
# TrainModels: the twelve families with a `train_x` method of their own, plus
# the six that fall through to `train_pyod` and share PYOD_PARAM_GRID. ALAD has
# a grid and a training branch but is not in ALL_DETECTORS, so it is
# deliberately absent.
FAMILY_GRIDS = {
    'LOF': LOF_PARAM_GRID,
    'CBLOF': CBLOF_PARAM_GRID,
    'ABOD': ABOD_PARAM_GRID,
    'KDE': KDE_PARAM_GRID,
    'NN': NN_PARAM_GRID,
    'RNN': RNN_PARAM_GRID,
    'LSTMVAE': LSTMVAE_PARAM_GRID,
    'DGHL': DGHL_PARAM_GRID,
    'RM': RM_PARAM_GRID,
    'MD': MD_PARAM_GRID,
    'LSTMAD': LSTMAD_PARAM_GRID,
    'COF': COF_PARAM_GRID,
    'SOS': SOS_PARAM_GRID,
    'IFOREST': PYOD_PARAM_GRID,
    'HBOS': PYOD_PARAM_GRID,
    'PCA': PYOD_PARAM_GRID,
    'OCSVM': PYOD_PARAM_GRID,
    'MCD': PYOD_PARAM_GRID,
    'SR': PYOD_PARAM_GRID,
    'KMEANSAD': KMEANSAD_PARAM_GRID,
    'POLY': POLY_PARAM_GRID,
    'AUTOENCODER': AUTOENCODER_PARAM_GRID,
    'DONUT': DONUT_PARAM_GRID,
    'OMNIANOMALY': OMNIANOMALY_PARAM_GRID,
    'USAD': USAD_PARAM_GRID,
    'TRANAD': TRANAD_PARAM_GRID,
    'FITS': FITS_PARAM_GRID,
    'TIMESNET': TIMESNET_PARAM_GRID,
}


def grid_combinations(grid):
    """The grid expanded in the order training numbers the instances.

    Mirrors `sklearn.model_selection.ParameterGrid`: keys sorted, then a
    cartesian product with the LAST key varying fastest. Every `train_x` method
    does `list(ParameterGrid(GRID))` and names the i-th entry
    `{FAMILY}_{i+1}`, so this is what makes "LOF_2 is contamination 0.15" a
    fact rather than a guess.

    Reimplemented here in stdlib rather than imported from sklearn because the
    web UI reads it, and `WebUI/catalog.py` must not drag sklearn into a Flask
    process. `Utils/test_pipeline_spec` asserts the two orderings still agree,
    so a change in sklearn cannot silently desynchronise the UI from training.
    """
    from itertools import product
    items = sorted(grid.items())
    if not items:
        return [{}]
    keys = [k for k, _ in items]
    values = [v for _, v in items]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def varying_keys(grid):
    """Grid keys with more than one value: what separates one instance from
    the next. Everything else is shared by the whole family and says nothing
    about which instance you are looking at."""
    return [k for k, v in sorted(grid.items()) if len(v) > 1]
