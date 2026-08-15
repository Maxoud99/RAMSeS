"""Vendored subset of TSB-AD 1.5 (Apache-2.0), the framework behind TSB-AutoAD.

Why vendored rather than a dependency
-------------------------------------
`pip install TSB-AD` resolves `numpy<2.0,>=1.24.3` and would drag this project
from numpy 2.0.0 (pinned in requirements.txt) back to 1.26.4 — the last 1.x
release, across the major-version boundary — plus nineteen other packages,
including transformers and tokenizers for the foundation models the paper
explicitly excludes from the candidate pool.

That pin is conservative packaging metadata rather than a real constraint. The
only numpy-2 incompatibility on these modules' import path is
`utils/torch_utility.py:26`, which said `np.Inf` (removed in numpy 2.0). It says
`np.inf` here. Everything else is byte-identical to the wheel, so re-syncing
against a future TSB-AD release is a diff, not a merge.

What is here, and what is not
-----------------------------
The transitive import closure of eight detectors, computed rather than guessed:

    models/  KMeansAD POLY Donut OmniAnomaly USAD TranAD FITS TimesNet
             base distance
    utils/   dataset torch_utility utility

The foundation models (OFA, Lag-Llama, Chronos, TimesFM, MOMENT) are NOT here.
The paper excludes them from the RAMSeS candidate pool for inconsistent
performance, and they are what pulls in the transformers stack.

TSB-AD's own statistical detectors are not here either, and deliberately: LOF,
IForest, HBOS, PCA, OCSVM, MCD, KNN, CBLOF and COF in TSB-AD are forks of PyOD
carrying the header "This function is adapted from [pyod] by [yzhao062]". This
pool already reaches that code through PyOD upstream, which is maintained and
numpy-2 clean. Vendoring a second, older copy would add no algorithm and two
ways to be wrong.

Interface
---------
These detectors do NOT share PyOD's API. They take the whole series as
`(n_timesteps, n_channels)` and return one score per timestep, and KMeansAD
spells its scorer `predict` while the rest use `decision_function`. Nothing in
this package is called directly — `Algorithms/tsbad_model.py` adapts it to
`PyMADModel`.

Upstream: https://github.com/TheDatumOrg/TSB-AD (Apache-2.0, LICENSE alongside).
"""
