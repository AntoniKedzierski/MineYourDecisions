import numbers
import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Sequence

import numpy as np

from core.data_types.time_series import TimeSeries
from core.data_types.time_series import MultiTimeSeries
from core.analysis.transforms import FTransform


def time_series_coefs(ts : TimeSeries, h0, h1):
    ts_op =  ts.reset_time().scale_down()
    F = FTransform(ts_op.start, ts_op.end, h0, h1)
    F.fit(ts_op.time, ts_op.values)
    c0 = np.array(F.direct_coefs_0[1:-1])
    c1 = np.array(F.direct_coefs_1[1:-1])
    return c0, c1


def time_series_inv_transform(ts : TimeSeries, h0, h1, x, degree=1):
    ts_op = ts.reset_time().scale_down()
    F = FTransform(ts_op.start, ts_op.end, h0, h1)
    F.fit(ts_op.time, ts_op.values)
    if degree == 0:
        return F.inv0[x]
    elif degree == 1:
        return F.inv1[x]


def time_series_plot_inv(ts : TimeSeries, h0, h1, degree=1, plot_series=True):
    ts_op = ts.reset_time().scale_down()
    F = FTransform(ts_op.start, ts_op.end, h0, h1)
    F.fit(ts_op.time, ts_op.values)
    x = ts_op.time
    if degree == 0:
        y = F.inv0[x]
    elif degree == 1:
        y = F.inv1[x]

    plot_data = { 'x': [], 'y': [], 'label': [] }
    if plot_series:
        plot_data['x'] += list(ts_op.time)
        plot_data['y'] += list(ts_op.values)
        plot_data['label'] += [ts.name] * len(ts_op)

    plot_data['x'] += list(x)
    plot_data['y'] += list(y)
    plot_data['label'] += [f'Approx deg.{degree}'] * len(x)

    sns.lineplot(x='x', y='y', hue='label', data=plot_data)
    plt.show()


def s_score(ts1 : TimeSeries, ts2 : TimeSeries, h0, h1, kappa1, kappa2):
    ts_joined = ts1.time_join(ts2)
    ts1_coef_0, ts1_coef_1 = time_series_coefs(ts_joined[0], h0, h1)
    ts2_coef_0, ts2_coef_1 = time_series_coefs(ts_joined[1], h0, h1)
    n_elem_0 = ts1_coef_0.shape[0]
    n_elem_1 = ts1_coef_1.shape[0]
    phi = max(max(abs(ts1_coef_1)), max(abs(ts2_coef_1)))
    return max(0, 1 - kappa1 / (n_elem_0 - 1) * sum(abs(ts1_coef_0 - ts2_coef_0)) + kappa2 / (n_elem_1 - 1) / phi * sum(abs(ts1_coef_1 - ts2_coef_1)))


def my_score(ts1 : TimeSeries, ts2 : TimeSeries, h0, h1, p, q):
    ts_joined = ts1.time_join(ts2)
    ts1_coef_0, ts1_coef_1 = time_series_coefs(ts_joined[0], h0, h1)
    ts2_coef_0, ts2_coef_1 = time_series_coefs(ts_joined[1], h0, h1)
    n_elem_0 = ts1_coef_0.shape[0]
    n_elem_1 = ts1_coef_1.shape[0]
    return sum(np.power(ts1_coef_0 - ts2_coef_0, p)) ** (1 / p) + sum(np.power(ts1_coef_1 - ts2_coef_1, q)) ** (1 / q)





