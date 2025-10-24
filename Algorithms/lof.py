from Algorithms.base_model import PyMADModel
import torch as t
import numpy as np
from pyod.models.lof import LOF
from Utils.utils import de_unfold

class TsadLof(PyMADModel):

    def __init__(self, window_size=1, window_step=1, contamination=0.1, device=None):
        super(TsadLof, self).__init__()

        self.contamination = contamination
        self.model = LOF(contamination=self.contamination)
        self.window_size = window_size
        self.window_step = window_step
        self.device = device

    def fit(self, train_dataloader):
        n_batches, n_features, n_time = train_dataloader.Y_windows.shape

        Y_windows = train_dataloader.Y_windows.reshape(n_batches * n_features * n_time, -1).reshape(-1, 1)
        Y_windows = Y_windows.cpu().numpy()  # Ensure it's on CPU and a NumPy array

        self.model.fit(X=Y_windows)

    def forward(self, input):
        Y = input['Y'].to(self.device)
        n_batches, n_features, n_time = Y.shape

        t_Y = Y.reshape(n_batches * n_features * n_time).reshape(-1, 1).cpu().numpy()  # Ensure it's on CPU and a NumPy array
        t_Y_score = self.model.decision_function(X=t_Y)
        t_Y_score = np.array([abs(i) for i in t_Y_score])
        t_Y_score[t_Y_score > 1.5] = 1.5
        t_Y_score = t_Y_score.reshape(-1, 1)

        Y_hat = t_Y * t_Y_score
        Y_hat = Y_hat.reshape(n_batches, n_features, n_time)

        return input['Y'], t.from_numpy(Y_hat).to(self.device), input['mask'].to(self.device)

    def training_step(self, input):
        Y, Y_hat, mask = self.forward(input=input)
        loss = t.mean((mask * (Y - Y_hat)) ** 2)
        return loss

    def eval_step(self, x):
        self.model.eval()
        loss = self.training_step(x)
        return loss

    def window_anomaly_score(self, input, return_detail: bool = False):
        Y, Y_hat, mask = self.forward(input=input)
        anomaly_score = (mask * (Y - Y_hat)) ** 2
        if return_detail:
            return anomaly_score
        else:
            return t.mean(anomaly_score, dim=0)

    def final_anomaly_score(self, input, return_detail: bool = False):
        anomaly_scores = de_unfold(windows=input, window_step=self.window_step)
        if return_detail:
            return anomaly_scores
        else:
            anomaly_scores = t.mean(anomaly_scores, dim=0)
            return anomaly_scores
