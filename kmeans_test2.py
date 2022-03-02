from core.data_types.time_series import TimeSeries
from core.data_types.time_series import MultiTimeSeries
from core.analysis.time_series import *

from core.provider.online_sources import AlphaVantageConnector
from core.provider.online_sources import dump_alphavantage

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy

from core.models.cluster import TimeSeriesKMeans

X = []
labels = []
series = {}
for i, file in enumerate(os.listdir('data/djia_composite')):
    ts = TimeSeries.read_csv(f'data/djia_composite/{file}', name=file.split('.')[0])
    series[ts.name] = ts.values
    series['timestamp'] = ts.time
    labels.append(ts.name)
    X.append(ts)

kmeans = TimeSeriesKMeans(k_clusters=6, metric='dtw', random_state=42, predict_coefs=False)
kmeans.fit(X)
classes, centroids = kmeans.predict()

classes = pd.DataFrame({'series': labels, 'classes': classes})
plot_data = pd.DataFrame(series).melt(id_vars='timestamp', var_name='series', value_vars=labels).merge(classes,
                                                                                                       left_on='series',
                                                                                                       right_on='series',
                                                                                                       how='inner')
plot_data['timestamp'] = plot_data['timestamp'].astype('datetime64')

for i in range(kmeans.k_clusters):
    centroids[i].plot()
    sns.lineplot(x='timestamp', y='value', hue='series', data=plot_data.loc[plot_data.classes == i, :])
    plt.title(f'Class no. {i}')
    plt.show()

# apple = TimeSeries.read_csv('data/djia_composite/AAPL.csv', name='APPLE')
# time_series_plot_inv(apple, 3, 5, degree=1)