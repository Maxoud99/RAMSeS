from pyod.models.sos import SOS
from Algorithms.base_model import PyMADModel
from Algorithms import windowed
import torch as t
from Utils.utils import de_unfold
import numpy as np


# SOS is quadratic in the rows of a call, in memory and in time, and none of its
# hyperparameters changes that. PyOD's `fit` builds the full n x n dissimilarity
# matrix, then an n x n affinity matrix one row at a time (a Python loop running
# a binary search for the perplexity on every row), then the n x n binding
# probabilities, and holds them all at once. Measured on SMD rows: 4,000 rows
# took 3.9 s and 0.8 GB, 16,000 rows took 52 s and 8.3 GB peak RSS — 32 bytes a
# cell, i.e. four float64 matrices alive together. And `fit` is only half of it:
# the diagnostic plot after it scores the whole test split in one call (SOS is
# transductive, see `Utils.pipeline_spec.TRANSDUCTIVE_FAMILIES`), which repeats
# the same computation on the test rows.
#
# An SMD machine is 28,479 rows with `downsampling` null: ~26 GB and roughly
# three minutes per call on a 64-core box with memory to spare, two calls per
# instance, four instances. On a machine without that memory it swaps, which is
# the training that "hangs for hours". The training deadline
# (`training_timeout`) stops it as well, but only after the allocation has hit a
# shared machine and the whole limit has been spent, once per family; checking
# the size first refuses it before anything is built.
#
# 10,000 rows is ~3.2 GB and ~20 s a call. Every SKAB entity (under 1,000
# rows), UCR entities up to that length and an SMD machine at `downsampling: 3`
# (9,493 rows) fit under it; a full-length SMD machine does not.
SOS_MAX_ROWS = 10_000


def check_tractable(n_rows: int, split: str) -> None:
    """Refuse a call SOS cannot finish in reasonable time and memory.

    `n_rows` is what one call would hand the estimator: the number of windows
    the loader cuts, which at the grid's window_size 1 is the number of
    timesteps.
    """
    if n_rows <= SOS_MAX_ROWS:
        return
    gigabytes = 32 * n_rows ** 2 / 1e9
    raise ValueError(
        f"SOS would need about {gigabytes:.0f} GB for the {n_rows:,} {split} rows "
        f"of this entity: it holds four {n_rows:,} x {n_rows:,} matrices at once, "
        f"and its time grows with the square of the rows too. The limit is "
        f"{SOS_MAX_ROWS:,} rows (SOS_MAX_ROWS in Algorithms/sos.py). Downsample "
        f"the entity (`downsampling` in the config) or leave SOS out on it.")


# this is Stochastic Outlier Selection class
class TsadSOS(PyMADModel):

    def __init__(self, window_size=1, window_step=1,contamination=0.1, device=None, perplexity=4.5, metric='euclidean', eps=1e-05):
        '''

        :param window_size:
        :param window_step:
        :param contamination:contaminationfloat in (0., 0.5), optional (default=0.1)
            The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function.
        :param device:
        :param perplexity: A smooth measure of the effective number of neighbours. The perplexity parameter is similar to the parameter k in kNN algorithm (the number of nearest neighbors). The range of perplexity can be any real number between 1 and n-1, where n is the number of samples.
        :param metric: Metric used for the distance computation. Any metric from scipy.spatial.distance can be used.
            Valid values for metric are: ‘euclidean’
            from scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
        :param eps: Tolerance threshold for floating point errors.
        '''
        super(TsadSOS, self).__init__()


        self.contamination = contamination
        self.model = SOS(perplexity=perplexity, metric=metric, eps= eps)
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

