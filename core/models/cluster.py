from core.data_types.time_series import TimeSeries
from core.analysis.metrics import my_score

import random
import numpy as np
import dtw

from sklearn.preprocessing import StandardScaler

class TimeSeriesKMeans():
    def __init__(self, k_clusters, p=2, q=2, metric='my_score', max_iter=300, eps=0.001, normalize=True, random_state=None, predict_coefs=False):
        self.k_clusters = k_clusters
        self.metric = metric
        self.p = p
        self.q = q
        self.max_iter = max_iter
        self.eps = eps
        self.normalize = normalize
        self.random_state = random_state
        self.predict_coefs = predict_coefs
        self.time = None
        self.scaler = StandardScaler()

        if self.metric == 'f_transform':
            self.normalize = False
            self.predict_coefs = True


    def fit(self, train):
        n = len(train)

        # All time series have to have a common time domain
        self.time = train[0].time

        if self.metric == 'f_transform':
            p = len(train[0].FTransform.direct_coefs_0) + len(train[0].FTransform.direct_coefs_1)
            X = np.zeros((n, p))
            split_points = np.zeros(n)
            for i in range(n):
                X[i, :], split_points[i] = train[i].f_coefs

        if self.metric == 'dtw' or self.metric == 'euclidean':
            p = len(train[0].values)
            X = np.zeros((n, p))
            for i in range(n):
                X[i, :] = train[i].values

        if self.normalize:
            X = self.scaler.fit_transform(X)

        self.clusters = np.zeros(n)
        self.centroids = np.zeros((self.k_clusters, p))

        if self.random_state is not None:
            random.seed(self.random_state)

        j = 0
        for i in random.sample(list(np.arange(0, n, 1)), self.k_clusters):
            self.centroids[j, :] = X[i, :]
            j += 1

        prev_step_cost = None
        for iter in range(self.max_iter):
            for i in range(n):
                nearest_cluster = int(self.clusters[i])
                nearest_centroid = self.centroids[nearest_cluster, :]

                if self.metric == 'f_transform':
                    act_distance = my_score(X[i, :], nearest_centroid, int(split_points[i]), self.p, self.q)
                if self.metric == 'dtw':
                    act_distance = dtw.dtw(X[i, :], nearest_centroid).normalizedDistance
                if self.metric == 'euclidean':
                    act_distance = sum(np.power(abs(X[i, :] - nearest_centroid), self.p)) ** (1 / self.p)

                for k in range(self.k_clusters):
                    if self.metric == 'f_transform':
                        dist_to_cluster = my_score(X[i, :], self.centroids[k, :], int(split_points[i]), self.p, self.q)
                    if self.metric == 'dtw':
                        dist_to_cluster = dtw.dtw(X[i, :], self.centroids[k, :]).normalizedDistance
                    if self.metric == 'euclidean':
                        dist_to_cluster = sum(np.power(abs(X[i, :] - self.centroids[k, :]), self.p)) ** (1 / self.p)
                    if dist_to_cluster < act_distance:
                        self.clusters[i] = k

            total_cost = 0.0
            for i in range(n):
                nearest_cluster = int(self.clusters[i])  # 0, 1, 2 ... k
                nearest_centroid = self.centroids[nearest_cluster, :]
                if self.metric == 'f_transform':
                    total_cost += my_score(X[i, :], nearest_centroid, int(split_points[i]), self.p, self.q)
                if self.metric == 'dtw':
                    total_cost += dtw.dtw(X[i, :], nearest_centroid).normalizedDistance
                if self.metric == 'euclidean':
                    total_cost += sum(np.power(abs(X[i, :] - nearest_centroid), self.p)) ** (1 / self.p)
            total_cost /= len(X)

            # New centroids
            for k in range(self.k_clusters):
                total = np.zeros(p)
                counter = 0
                for i in range(n):
                    if self.clusters[i] != k:
                        continue
                    total = total + X[i, :]
                    counter += 1
                if counter != 0:
                    self.centroids[k, :] = total / counter

            if prev_step_cost is not None:
                if abs(total_cost - prev_step_cost) / prev_step_cost < self.eps:
                    break

            prev_step_cost = total_cost

    def predict(self):
        if self.predict_coefs:
            if self.normalize:
                return self.clusters, self.scaler.inverse_transform(self.centroids)
            return self.clusters, self.centroids
        ts = []
        for i in range(self.centroids.shape[0]):
            if self.normalize:
                ts.append(TimeSeries(self.time, self.scaler.inverse_transform(self.centroids)[i, :], name=f'Centroid no. {i}'))
            else:
                ts.append(TimeSeries(self.time, self.centroids[i, :], name=f'Centroid no. {i}'))
        return self.clusters, ts
