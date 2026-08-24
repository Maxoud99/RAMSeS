"""Series2Graph as a whole-series detector — over a file this repository does
not contain.

Boniol & Palpanas, "Series2Graph: Graph-based Subsequence Anomaly Detection in
Time Series", PVLDB 2020. Subsequences of length `pattern_length` are embedded
in a 2-D phase space, nodes and edges are extracted from that embedding, and a
query subsequence is scored by the degree of the path it traces through the
resulting graph. It is the only detector in the pool whose graph is over
SUBSEQUENCES, and the only true graph-based *time-series* method here — LUNAR's
graph is over samples and MTAD-GAT's is over channels.

Why the algorithm's own file is not in this repository
-------------------------------------------------------
`Series2Graph.py` ships inside the TSB-AD 1.5 wheel, which is Apache-2.0 and is
where the rest of `Algorithms/tsb_ad` was vendored from. That one file is not
Apache-2.0. Its header reads:

    # copyright retained by the authors
    # algorithms protected by patent application FR2005261
    # code provided as is, and can be used only for research purposes

"Used for research purposes" is a grant to USE. Redistribution is a separate
right, and nothing in that sentence grants it — so committing the file into a
public repository whose LICENSE says Apache-2.0 would offer third parties rights
we do not hold. Upstream has the same contradiction (its repo LICENSE is
Apache-2.0 while the file's own header is not), and inheriting a problem is not
the same as being entitled to propagate it.

So the file is fetched, never committed. It is listed in `.gitignore`, and
whoever wants this detector obtains it themselves under the authors' own terms:

    python -m Algorithms.tsb_ad.fetch_series2graph

Until then the family is selectable and fails with the message below, naming the
command — the same shape as the univariate refusal, which also refuses at fit
time rather than pretending the detector is absent from the pool.

If you use it, cite it. The header asks for:
  P. Boniol and T. Palpanas. Series2Graph: Graph-based Subsequence Anomaly
  Detection in Time Series. PVLDB, 2020.

What this adapter absorbs
-------------------------
Series2Graph has a three-call interface — `fit(ts)`, then
`score(query_length, dataset)`, then read `decision_scores_` — and returns
`len(ts) - query_length` scores rather than one per timestep. Both are dealt
with here.

`score` also accepts `dataset` and never reads it: every value it produces comes
from the graph `fit` built, so `decision_function` returns the FIT series'
scores whatever it is handed. Measured: fit on 3,000 rows, then score 9,000,
gives 2,900 scores `array_equal` to scoring the 3,000 back. That is why
`tsbad_model` reaches this class through the 'refit' scorer — a fresh graph per
call — and why the family sits in TRANSDUCTIVE_FAMILIES. It is also what the
method is: an unsupervised search over one series, not a train/test split.

Left as `decision_function`, it produced one score per TRAINING row against a
test series of a different length, which `windowed.score_windows` caught as a
shape mismatch on every UCR entity — the only dataset it runs on, being
univariate — so the family could not be trained at all.

Measured on SMD machine-1-1's first channel: deterministic across two fits
(0.000e+00) and 1.2s for a fit-and-score over 28,479 rows, so refitting per call
costs nothing the pipeline notices.
"""

import math

import numpy as np

FETCH_COMMAND = "python -m Algorithms.tsb_ad.fetch_series2graph"

_MISSING = (
    "Series2Graph is not installed. Its source file is deliberately NOT part of "
    "this repository: it is patent-encumbered and licensed for research use "
    "only, unlike the Apache-2.0 code around it, so it is fetched rather than "
    "redistributed. Run\n\n    {cmd}\n\nto obtain it from the TSB-AD wheel "
    "under the authors' own terms, or drop Series2Graph from --detectors. See "
    "Algorithms/series2graph_detector.py for the full reasoning."
).format(cmd=FETCH_COMMAND)


def _load_series2graph():
    """The vendored-on-demand class, or a message saying how to get it."""
    try:
        from Algorithms.tsb_ad.models.Series2Graph import Series2Graph
    except ImportError as exc:                       # pragma: no cover - env
        raise ImportError(_MISSING) from exc
    return Series2Graph


class Series2GraphDetector:
    """`fit(series)` / `decision_function(series)` over Series2Graph.

    `query_length` follows TSB-AD's own wrapper at `2 * pattern_length`; it is
    not a separate grid axis, because upstream never varies it independently.
    """

    def __init__(self, pattern_length=50, rate=30, contamination=0.1):
        self.pattern_length = int(pattern_length)
        self.rate = int(rate)
        self.contamination = contamination
        self.decision_scores_ = None
        self._graph = None
        # Constructing must prove the file is present, so a missing fetch fails
        # at training time with the command to run rather than at the first
        # scoring call several stages later.
        self._cls = _load_series2graph()

    @property
    def query_length(self) -> int:
        return 2 * self.pattern_length

    @staticmethod
    def _as_univariate(data) -> np.ndarray:
        X = np.asarray(data, dtype=float)
        X = X.reshape(len(X), -1) if X.ndim > 1 else X.reshape(len(X), 1)
        if X.shape[1] != 1:
            raise ValueError(
                f"Series2Graph is univariate only and was handed "
                f"{X.shape[1]} channels.")
        return X[:, 0]

    def _check_length(self, n_time: int):
        """Shorter than one query and the method returns NOTHING.

        Not an exception upstream — `score` simply produces zero rows, which
        would surface as a length mismatch inside `windowed.score_windows` with
        nothing pointing back here. Measured: 80 rows at query_length 100 gives
        0 scores.
        """
        if n_time <= self.query_length:
            raise ValueError(
                f"Series2Graph needs more than query_length "
                f"(2 * pattern_length = "
                f"{self.query_length}) rows per call and was handed {n_time}. "
                f"Thompson cuts windows of n_timesteps * 0.8 / iterations, so a "
                f"short entity or a high iteration count puts it below this; "
                f"every other stage scores the whole series and clears it.")

    def fit(self, data, y=None):
        series = self._as_univariate(data)
        self._check_length(len(series))
        self._graph = self._cls(pattern_length=self.pattern_length, rate=self.rate)
        self._graph.fit(series)
        self.decision_scores_ = self.decision_function(data)
        return self

    def decision_function(self, data):
        if self._graph is None:
            raise ValueError(
                "Series2Graph.decision_function called before fit.")
        series = self._as_univariate(data)
        self._check_length(len(series))
        self._graph.score(query_length=self.query_length, dataset=series)
        scores = np.asarray(self._graph.decision_scores_, dtype=float).ravel()

        # One score per timestep. TSB-AD's own wrapper pads by repeating the
        # end values, and the split is uneven by construction:
        # ceil(q/2) in front, floor(q/2) behind, which sums to exactly q.
        head = math.ceil(self.query_length / 2)
        tail = self.query_length // 2
        if not len(scores):
            return np.zeros(len(series), dtype=float)
        padded = np.concatenate([np.repeat(scores[0], head), scores,
                                 np.repeat(scores[-1], tail)])
        return padded[:len(series)]
