from core.data_types.time_series import TimeSeries
from core.analysis.metrics import my_score

import random
import numpy as np
import dtw

class TimeSeriesKMeans():
    def __init__(self, k_clusters, p=2, q=2, metric='my_score', max_iter=300, eps=0.001):
        self.k_clusters = k_clusters
        self.metric = metric
        self.p = p
        self.q = q
        self.max_iter = max_iter
        self.eps = eps

    def fit(self, X):
        pass

    def calculate_cost(self, X, centroids, clusters):
        '''
        :param X: A list of time series
        :param centroids: Fuzzy coeficients for time series representing centroids. A numpy 2-d array.
        :param clusters: Assigned class for a given time series.
        :return:
        '''
        total = 0.0
        for i, ts in enumerate(X):
            assigned_cluster = int(clusters[i]) # 0, 1, 2 ... k
            nearest_centroid = centroids[assigned_cluster, :]
            coefs, split_point = ts.f_coefs
            if self.metric == 'my_score':
                total += my_score(nearest_centroid, coefs, split_point, self.p, self.q)
            if self.metric == 'dtw':
                total += dtw.dtw(X[i].values, nearest_centroid).normalizedDistance
        return total / len(X)

    def fit(self, X):
        # Assume that all X have the same length of direct_coef_0 and direct_coef_1
        n = len(X)
        if self.metric == 'my_score':
            p = len(X[0].FTransform.direct_coefs_0) + len(X[0].FTransform.direct_coefs_1)
        if self.metric == 'dtw':
            p = len(X[0].values)
        self.clusters = np.zeros(n)
        self.centroids = np.zeros((self.k_clusters, p))
        j = 0
        for i in random.sample(list(np.arange(0, n, 1)), self.k_clusters):
            if self.metric == 'my_score':
                self.centroids[j, :] = X[i].FTransform.direct_coefs_0 + X[i].FTransform.direct_coefs_1
            if self.metric == 'dtw':
                self.centroids[j, :] = X[i].values
            j += 1

        prev_step_cost = None
        for iter in range(self.max_iter):
            for i in range(n):
                x, split_point = X[i].f_coefs
                nearest_cluster = int(self.clusters[i])
                nearest_centroid = self.centroids[nearest_cluster, :]
                if self.metric == 'my_score':
                    act_distance = my_score(x, nearest_centroid, split_point, self.p, self.q)
                if self.metric == 'dtw':
                    act_distance = dtw.dtw(X[i].values, nearest_centroid).normalizedDistance
                for k in range(self.k_clusters):
                    if self.metric == 'my_score':
                        dist_to_cluster = my_score(x, self.centroids[k, :], split_point, self.p, self.q)
                    if self.metric == 'dtw':
                        dist_to_cluster = dtw.dtw(X[i].values, self.centroids[k, :]).normalizedDistance
                    if dist_to_cluster < act_distance:
                        self.clusters[i] = k

            total_cost = self.calculate_cost(X, self.centroids, self.clusters)

            # New centroids
            for k in range(self.k_clusters):
                total = np.zeros(p)
                counter = 0
                for i in range(n):
                    if self.clusters[i] != k:
                        continue
                    if self.metric == 'my_score':
                        x, _ = X[i].f_coefs
                    if self.metric == 'dtw':
                        x = X[i].values
                    total = total + x
                    counter += 1
                if counter != 0:
                    self.centroids[k, :] = total / counter

            if prev_step_cost is not None:
                if abs(total_cost - prev_step_cost) / prev_step_cost < self.eps:
                    break

            prev_step_cost = total_cost

    def predict(self):
        return self.clusters, self.centroids


