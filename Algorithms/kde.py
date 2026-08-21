
from Algorithms.base_model import PyMADModel
from Algorithms import windowed
import torch as t
import numpy as np
from pyod.models.kde import KDE
from Utils.utils import de_unfold
class TsadKde(PyMADModel):

    def __init__(self, window_size=1, window_step=1, contamination=0.1,
                 bandwidth=1.0, device=None):
        super(TsadKde, self).__init__()


        self.contamination = contamination
        # `bandwidth` is what separates this family's instances: it is the
        # kernel width, so it decides how peaked the estimated density is and
        # therefore the score of every point. `contamination` only moved a
        # threshold this pipeline replaces with its own sweep.
        self.bandwidth = bandwidth
        self.model = KDE(contamination=self.contamination,
                         bandwidth=self.bandwidth)
        self.window_size = window_size
        self.window_step = window_step
        self.device =device

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








