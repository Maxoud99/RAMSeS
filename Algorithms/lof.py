from Algorithms.base_model import PyMADModel
from Algorithms import windowed
import torch as t
import numpy as np
from pyod.models.lof import LOF
from Utils.utils import de_unfold

class TsadLof(PyMADModel):

    def __init__(self, window_size=1, window_step=1, contamination=0.1,
                 n_neighbors=20, metric='minkowski', device=None):
        """`n_neighbors` and `metric` are what separate this family's instances.

        They used to be PyOD's defaults with only `contamination` varying, which
        made the four LOF instances one detector wearing four names: it sets
        `threshold_`/`labels_` and never reaches `decision_function`, and the
        pipeline scores with its own threshold sweep. Both names and their
        values come from TSB-AD's LOF sweep, so the pool's instances are the
        ones the baseline framework tunes over.
        """
        super(TsadLof, self).__init__()

        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = LOF(contamination=self.contamination,
                         n_neighbors=self.n_neighbors, metric=self.metric)
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
        anomaly_scores = de_unfold(windows=input, window_step=self.window_step)
        if return_detail:
            return anomaly_scores
        else:
            anomaly_scores = t.mean(anomaly_scores, dim=0)
            return anomaly_scores
