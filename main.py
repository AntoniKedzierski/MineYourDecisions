from core.data_types.time_series import TimeSeries
from core.data_types.time_series import MultiTimeSeries
from core.analysis.metrics import FTransform, s_score, time_series_coefs, time_series_inv_transform, time_series_plot_inv

from core.provider.online_sources import AlphaVantageConnector
from core.provider.online_sources import dump_alphavantage

from sklearn.cluster import KMeans

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # for i, file1 in enumerate(os.listdir('data/djia_composite')):
    #     for j, file2 in enumerate(os.listdir('data/djia_composite')):
    #         if j <= i: continue
    #         ts1 = TimeSeries.read_csv(f'data/djia_composite/{file1}', name=file1.split('.')[0])
    #         ts2 = TimeSeries.read_csv(f'data/djia_composite/{file2}', name=file2.split('.')[0])
    #         print(ts1.name, ts2.name, s_score(ts1, ts2, 4, 8, 2, 2.5))

    # apple = TimeSeries.read_csv('data/djia_composite/AAPL.csv', name='Apple')
    # cat = TimeSeries.read_csv('data/djia_composite/CAT.csv', name='Caterpillar')
    #
    # time_series_plot_inv(cat, 2, 5, degree=0)

    h0 = 3
    h1 = 5
    clusters = 6

    labels = []
    series = {}
    series_coefs = []
    for file in os.listdir('data/djia_composite'):
        ts = TimeSeries.read_csv(f'data/djia_composite/{file}', name=file.split('.')[0])
        series[ts.name] = ts.values
        series['timestamp'] = ts.time
        labels.append(ts.name)
        c0, c1 = time_series_coefs(ts, h0, h1)
        series_coefs.append(np.concatenate([c0, c1]))

    series_coefs = np.asarray(series_coefs)

    kmeans = KMeans(n_clusters=clusters, max_iter=1000, random_state=42)
    classes = kmeans.fit_predict(series_coefs)

    classes = pd.DataFrame({'series' : labels, 'classes': classes})
    plot_data = pd.DataFrame(series).melt(id_vars='timestamp', var_name='series', value_vars=labels).merge(classes, left_on='series', right_on='series', how='inner')
    plot_data['timestamp'] = plot_data['timestamp'].astype('datetime64')

    for i in range(clusters):
        sns.lineplot(x='timestamp', y='value', hue='series', data=plot_data.loc[plot_data.classes == i, :])
        plt.title(f'Class no. {i}')
        plt.show()











