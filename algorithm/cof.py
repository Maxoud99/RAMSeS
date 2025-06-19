
from algorithm.base_model import PyMADModel
import torch as t
import torch.nn as nn
import numpy as np
from pyod.models.cof import COF
from Utils.utils import de_unfold
import random
class TsadCof(PyMADModel):

    def __init__(self, window_size=1, window_step=1, contamination = 0.1,device=None):
        super(TsadCof, self).__init__()


        self.contamination = contamination
        self.model = COF(contamination=self.contamination)
        self.window_size = window_size
        self.window_step = window_step
        self.device =device

    def fit(self, train_dataloader):
        # print(f'train_dataloader.Y_windows.shape is {train_dataloader.Y_windows.shape}')
        n_batches, n_features, n_time = train_dataloader.Y_windows.shape

        Y_windows = train_dataloader.Y_windows.reshape(n_batches * n_features * n_time, -1).reshape(-1, 1)

        self.model.fit(X=Y_windows)

    def forward(self, input):
        Y = input['Y']
        n_batches, n_features, n_time = Y.shape

        t_Y = Y.reshape(n_batches * n_features * n_time).reshape(-1, 1).numpy()
        # print(f't_Y is {t_Y},shape is {t_Y.shape}')
        t_Y_score = self.model.decision_function(X=t_Y)
        t_Y_score_mean = t_Y_score.mean()
        # print(f't_Y_score_mean is {t_Y_score_mean}')
        # if t_Y_score_mean < 1 :
        #     mult = 1 / t_Y_score_mean
        #
        # t_Y_score[np.isnan(t_Y_score)]=1.1
        # t_Y_score = t_Y_score * mult
        # print(f'ori  t_Y_score is {t_Y_score}')
        t_Y_score = np.array([abs(i)for i in t_Y_score])
        t_Y_score[t_Y_score > 1.5] = 1.5
        t_Y_score = t_Y_score.reshape(-1, 1)
        # print(f't_Y_score is {t_Y_score}')
        Y_hat = t_Y * t_Y_score
        Y_hat = Y_hat.reshape(n_batches, n_features, n_time)

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

        # Forward
        Y, Y_hat, mask = self.forward(input=input)

        # Anomaly Score
        anomaly_score = (mask * (Y - Y_hat)) ** 2
        if return_detail:
            return anomaly_score
        else:
            return t.mean(anomaly_score, dim=0)

    def final_anomaly_score(self, input, return_detail: bool = False):

        # Average anomaly score for each feature per timestamp
        anomaly_scores = de_unfold(windows=input, window_step=self.window_step)

        if return_detail:
            return anomaly_scores
        else:
            anomaly_scores = t.mean(anomaly_scores, dim=0)
            return anomaly_scores








