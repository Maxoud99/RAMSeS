from pyod.models.sos import SOS
from Algorithms.base_model import PyMADModel
from Algorithms import windowed
import torch as t
from Utils.utils import de_unfold
import numpy as np



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

