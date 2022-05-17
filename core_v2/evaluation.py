import copy
import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import typing
import numbers
import functools

from time_series import TimeSeries
from ucr_wrapper import unpack


def knn_sscore(dataset, path_to_dir):
    train_set, train_label, test_set, test_label = unpack(dataset, path_to_dir)

    ts_len = len(train_set[0, 0])
    h0 = math.ceil(max(8, ts_len ** 0.5))
    h1 = math.ceil(h0 * 1.5)
    k = 3

    train_set = [
        {
            'deg0': ts[0].max_scaling().fuzzy_coefs(h0),
            'deg1': ts[0].max_scaling().fuzzy_slope_coefs(h1)[:, 1].flatten()
        } for ts in train_set
    ]
    test_set = [
        {
            'deg0': ts[0].max_scaling().fuzzy_coefs(h0),
            'deg1': ts[0].max_scaling().fuzzy_slope_coefs(h1)[:, 1].flatten()
        } for ts in test_set
    ]

    dist_matrix = np.zeros(shape=(len(test_set), k))

    for i in range(len(test_set)):
        furthest = 0
        for j in range(len(train_set)):
            sim_score






if __name__ == '__main__':
    knn_sscore('PowerCons', path_to_dir='../data/UCR/')