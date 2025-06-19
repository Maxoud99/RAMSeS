from pyod.models.sos import SOS
from algorithm.base_model import PyMADModel
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
        n_batches, n_features, n_time = train_dataloader.Y_windows.shape

        Y_windows = train_dataloader.Y_windows.reshape(n_batches * n_features * n_time, -1).reshape(-1, 1)

        self.model.fit(X=Y_windows)

    def forward(self, input):
        Y = input['Y']
        n_batches, n_features, n_time = Y.shape

        t_Y = Y.reshape(n_batches * n_features * n_time).reshape(-1, 1).numpy()
        # print(f't_Y is {t_Y},shape is {t_Y.shape}')
        t_Y_score = self.model.decision_function(X=t_Y)
        t_Y_score[np.isnan(t_Y_score)]=1.1
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

