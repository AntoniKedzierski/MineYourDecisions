from core.data_types.time_series import TimeSeries
from core.data_types.time_series import MultiTimeSeries
from core.analysis.time_series import *

from core.provider.online_sources import AlphaVantageConnector
from core.provider.online_sources import dump_alphavantage

from sklearn.cluster import KMeans

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


h0 = 3
h1 = 5
clusters = 4

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
