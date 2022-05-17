import pandas as pd
import numpy as np
import os
import math
import seaborn as sns
import matplotlib.pyplot as plt
import copy

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from core_v2.time_series import TimeSeries


def get_fuzzy(column, h):
    coefs = []
    for ts in column:
        coefs.append(ts.fuzzy_coefs(h))
    return np.asarray(coefs)


if __name__ == '__main__':
     data = pd.read_csv('C:/Users/Antoni/Documents/Magisterka/Dane/arma.csv', index_col=0)
     data = data.sample(frac=1)

     x = []
     for i in range(data.shape[0]):
          x.append(TimeSeries(data.iloc[i, 1:].to_numpy().reshape((-1, 1))))
     x = np.asarray(x).reshape((-1, 1))
     y = data.iloc[:, 0].to_numpy()

     model = Pipeline([
          ('col_transformer', ColumnTransformer([
               ('fuzzy_coefs', Pipeline([
                    ('fuzzy_coefs', FunctionTransformer(lambda x: get_fuzzy(x, 5))),
                    ('max_scaler', MaxAbsScaler())
               ]), 0)
          ])),
          ('forest', RandomForestClassifier(max_depth=10, n_estimators=500, n_jobs=-1))
     ])

     result = cross_val_score(model, x, y, scoring='accuracy', cv=5)
     print(result, np.mean(result))


