import pandas as pd
import numpy as np
import os

from core_v2.time_series import TimeSeries

def unpack(dir, path_to_dir='data/UCR/'):
    test_path = path_to_dir + dir + '/' + dir + '_TRAIN.tsv'
    train_path = path_to_dir + dir + '/' + dir + '_TEST.tsv'

    train = pd.read_csv(train_path, sep='\t')
    test = pd.read_csv(test_path, sep='\t')

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


def scan_lengths(path_to_dir='data/UCR/'):
    for f in os.listdir(path_to_dir):
        print(f, pd.read_csv(path_to_dir + f + '/' + f + '_TRAIN.tsv', sep='\t', nrows=1).shape[1] - 1)

if __name__ == '__main__':
    scan_lengths('../data/UCR/')


