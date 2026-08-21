"""MTAD-GAT: a graph over the channels, and a graph over the timestamps.

Zhao et al., "Multivariate Time-series Anomaly Detection via Graph Attention
Network", ICDM 2020. The pool's only detector that models the relationship
BETWEEN channels rather than treating a window as a flat feature vector, which
is what earns the Graph Based group a member that runs on SKAB and SMD —
Series2Graph is
univariate and LUNAR's graph is over samples, not sensors.

Two attention graphs over one window
------------------------------------
A window of `win_size` timesteps and `n_features` channels is read twice.

  * **Feature-oriented graph.** Each channel is a node; its feature vector is
    that channel's whole window. Every pair of channels is an edge, and the
    attention weight is learned, so the graph is complete and weighted rather
    than thresholded — a correlation graph the model fits instead of one built
    by hand.
  * **Time-oriented graph.** Each timestep is a node, its feature vector is all
    channels at that timestep, and the edges again span every pair.

Both use the paper's attention:

    e_ij  = LeakyReLU( a^T [ v_i ; v_j ] )
    a_ij  = softmax_j(e_ij)
    h_i   = sigma( sum_j a_ij v_j )

The two attended representations are concatenated with the raw window and fed
to a GRU, and two heads read the result: a forecast of the next timestep and a
reconstruction of the window's last timestep. The score is their combined error,

    s_t = (x_t - forecast_t)^2 + gamma * (x_t - recon_t)^2

which is the paper's joint objective with `gamma` balancing the two halves.

Why plain PyTorch and no torch_geometric
----------------------------------------
Both graphs are COMPLETE — every node attends to every other — so the adjacency
is implicit and the attention is a dense `(n, n)` softmax. There is no sparse
message passing to accelerate, and pulling in torch_geometric (plus, on older
versions, torch-scatter and torch-sparse) would add a heavy dependency to
express a matrix multiply. GDN, whose graph is a learned top-k sparse one, is
the detector that genuinely needs it.

Interface
---------
`fit(data)` / `decision_function(data)` over `(n_timesteps, n_channels)`,
returning one score per timestep — TSB-AD's contract, so
`Algorithms/tsbad_model.py` adapts it with no special case, exactly as it does
for `Algorithms/chronos_detector.py`. The class is reached through
`_TSBAD_SPECS`'s dotted-path form rather than being vendored, because it is
written here.

Determinism is a pool admission criterion, so `random_state` seeds torch, numpy
and python before any weight is allocated, and the loader does not shuffle.
Unseeded training would put this family in the company of TimeSeriesOD and
AnomalyTransformer, which the pool refuses for exactly that.
"""

import random

import numpy as np
import torch as t
import torch.nn as nn


class _GraphAttention(nn.Module):
    """One attention layer over a complete graph of `n_nodes` nodes.

    `embed_dim` is the length of a node's feature vector, which is `win_size`
    for the feature-oriented graph and `n_features` for the time-oriented one —
    the two readings of the same window, transposed.
    """

    def __init__(self, n_nodes: int, embed_dim: int):
        super().__init__()
        self.n_nodes = int(n_nodes)
        # One score per (i, j) pair from the concatenated pair, which is the
        # paper's `a^T [v_i ; v_j]`.
        self.attn = nn.Linear(2 * int(embed_dim), 1)
        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, n_nodes, embed_dim)
        batch, n, d = x.shape
        left = x.unsqueeze(2).expand(batch, n, n, d)
        right = x.unsqueeze(1).expand(batch, n, n, d)
        pairs = t.cat([left, right], dim=-1)          # (batch, n, n, 2d)
        e = self.leaky(self.attn(pairs)).squeeze(-1)  # (batch, n, n)
        a = t.softmax(e, dim=-1)
        return self.sigmoid(t.bmm(a, x))              # (batch, n, embed_dim)


class _Net(nn.Module):
    def __init__(self, n_features: int, win_size: int, hidden_size: int):
        super().__init__()
        self.feature_gat = _GraphAttention(n_features, win_size)
        self.time_gat = _GraphAttention(win_size, n_features)
        # The GRU consumes the raw window and both attended views, each of which
        # is n_features wide once transposed back to time-major.
        self.gru = nn.GRU(3 * n_features, hidden_size, batch_first=True)
        self.forecast = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_features))
        self.reconstruct = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_features))

    def forward(self, x):
        # x: (batch, win_size, n_features)
        # The feature graph reads channels as nodes, so the window is transposed
        # and the result transposed back; the time graph reads x as it stands.
        h_feat = self.feature_gat(x.permute(0, 2, 1)).permute(0, 2, 1)
        h_time = self.time_gat(x)
        h = t.cat([x, h_feat, h_time], dim=2)
        out, _ = self.gru(h)
        last = out[:, -1, :]
        return self.forecast(last), self.reconstruct(last)


class MTADGAT:
    """Forecast-plus-reconstruction error under two graph-attention views.

    `n_features` is set by `Algorithms.tsbad_model` from the data, like every
    other whole-series detector's channel argument.
    """

    def __init__(self, n_features=1, win_size=100, hidden_size=64, gamma=1.0,
                 epochs=10, batch_size=64, lr=1e-3, random_state=1,
                 contamination=0.1):
        self.n_features = int(n_features)
        self.win_size = int(win_size)
        self.hidden_size = int(hidden_size)
        self.gamma = float(gamma)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.random_state = int(random_state)
        self.contamination = contamination
        self.decision_scores_ = None
        self.model = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _seed(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        t.manual_seed(self.random_state)

    @staticmethod
    def _as_series(data) -> np.ndarray:
        X = np.asarray(data, dtype=np.float32)
        return X[:, None] if X.ndim == 1 else X

    def _check_length(self, n_time: int):
        """One window plus its forecast target has to fit in the call.

        Named here rather than left to torch, whose error would be a shape
        mismatch several frames down. Same role as the minimum-length guards
        POLY and CHRONOS carry.
        """
        if n_time <= self.win_size:
            raise ValueError(
                f"MTADGAT needs more than win_size ({self.win_size}) rows per "
                f"call and was handed {n_time}. Thompson cuts windows of "
                f"n_timesteps * 0.8 / iterations, so a short entity or a high "
                f"iteration count puts it below this; every other stage scores "
                f"the whole series and clears it.")

    def _windows(self, X: np.ndarray):
        """Sliding windows and the next timestep each one forecasts."""
        n_time = len(X)
        idx = np.arange(n_time - self.win_size)
        windows = np.stack([X[i:i + self.win_size] for i in idx])
        targets = X[self.win_size:]
        return t.from_numpy(windows), t.from_numpy(np.ascontiguousarray(targets))

    # ── the TSB-AD contract ──────────────────────────────────────────────────

    def fit(self, data, y=None):
        X = self._as_series(data)
        self._check_length(len(X))
        self.n_features = X.shape[1]
        self._seed()
        self.model = _Net(self.n_features, self.win_size, self.hidden_size)
        optimiser = t.optim.Adam(self.model.parameters(), lr=self.lr)
        windows, targets = self._windows(X)

        self.model.train()
        for _ in range(self.epochs):
            # Sequential batches, not a shuffled DataLoader: the shuffle is a
            # second source of run-to-run variation and seeding it buys nothing
            # a fixed order does not already give.
            for start in range(0, len(windows), self.batch_size):
                batch = windows[start:start + self.batch_size]
                target = targets[start:start + self.batch_size]
                forecast, recon = self.model(batch)
                loss = (t.mean((forecast - target) ** 2)
                        + self.gamma * t.mean((recon - batch[:, -1, :]) ** 2))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        self.decision_scores_ = self.decision_function(data)
        return self

    def decision_function(self, data):
        if self.model is None:
            raise ValueError("MTADGAT.decision_function called before fit.")
        X = self._as_series(data)
        self._check_length(len(X))
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"MTADGAT was fitted on {self.n_features} channels and handed "
                f"{X.shape[1]}.")
        windows, targets = self._windows(X)

        self.model.eval()
        errors = []
        with t.no_grad():
            for start in range(0, len(windows), self.batch_size):
                batch = windows[start:start + self.batch_size]
                target = targets[start:start + self.batch_size]
                forecast, recon = self.model(batch)
                err = (t.mean((forecast - target) ** 2, dim=1)
                       + self.gamma * t.mean((recon - batch[:, -1, :]) ** 2, dim=1))
                errors.append(err.numpy())
        scores = np.concatenate(errors) if errors else np.zeros(0, dtype=float)

        # Left-pad to one score per timestep: the first `win_size` points have
        # no window behind them, the same convention CHRONOS uses.
        padded = np.zeros(len(X), dtype=float)
        padded[:self.win_size] = scores[0] if len(scores) else 0.0
        padded[self.win_size:self.win_size + len(scores)] = scores
        return padded
