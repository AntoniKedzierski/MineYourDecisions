import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from core_v2.time_series import TimeSeries

def unpack(dir, path_to_dir='data/UCR/'):
    train_path = path_to_dir + dir + '/' + dir + '_TRAIN.tsv'
    test_path = path_to_dir + dir + '/' + dir + '_TEST.tsv'

    train = pd.read_csv(train_path, sep='\t', header=None)
    test = pd.read_csv(test_path, sep='\t', header=None)

    train_label = train.iloc[:, 0].to_numpy(dtype=int)
    train_set = []
    p = train.shape[1] - 1
    for i in range(train.shape[0]):
        train_set.append(TimeSeries(train.iloc[i, 1:].to_numpy().reshape((p, 1)), name=f'Obs. {i}'))
    train_set = np.asarray(train_set).reshape((-1, 1))

    test_label = test.iloc[:, 0].to_numpy(dtype=int)
    test_set = []
    p = test.shape[1] - 1
    for i in range(test.shape[0]):
        test_set.append(TimeSeries(test.iloc[i, 1:].to_numpy().reshape((p, 1)), name=f'Obs. {i}'))
    test_set = np.asarray(test_set).reshape((-1, 1))

    return train_set, train_label, test_set, test_label


def plot_dataset(dir, path_to_dir='data/UCR/'):
    train_path = path_to_dir + dir + '/' + dir + '_TRAIN.tsv'
    train = pd.read_csv(train_path, sep='\t', header=None)
    train = train.melt(id_vars=0, var_name='timestamp', ignore_index=False).reset_index().rename(columns={0: 'class', 'index': 'series'}).sort_values(by=['class', 'series', 'timestamp'])
    for c in train['class'].unique():
        sns.lineplot(x='timestamp', y='value', hue='series', data=train.loc[train['class'] == c, :])
        plt.title(f'Class {c}')
        plt.show()


def scan_lengths(path_to_dir='data/UCR/'):
    for f in os.listdir(path_to_dir):
        print(f, pd.read_csv(path_to_dir + f + '/' + f + '_TRAIN.tsv', sep='\t', nrows=1).shape[1] - 1)

if __name__ == '__main__':
    train_set, train_label, test_set, test_label = unpack('PowerCons', path_to_dir='../data/UCR/')
    train_set[2, 0].fuzzy_plot(12, title='F-transformata')
    train_set[2, 0].fuzzy_anomalies(12).plot()

