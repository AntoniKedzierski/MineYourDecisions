import os
import time
import numpy as np
import pandas as pd
import math
from scipy.spatial import distance

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import dtaidistance

from core_v2.ucr_wrapper import unpack

'''
Notatki bieżące:
    1. Filip W. twierdzi, że algorytm ROCKET ma bardzo dobrą skuteczność.
    2. Rozciąganie zbiorów rozmytych - a la Dynamic Time Wrapping.
'''

def get_fuzzy(column, h):
    coefs = []
    for ts in column:
        coefs.append(ts.fuzzy_coefs(h))
    return np.asarray(coefs)


def get_values(column):
    values = []
    for ts in column:
        values.append(ts.values)
    return np.asarray(values)


def dtw_metric(x, y):
    return dtaidistance.dtw.distance_fast(x, y, window=10, use_pruning=True)


def fuzzy_score(F1, F2):
    return np.mean(np.abs(F1 - F2) / (np.abs(F1) + np.abs(F2)))


def fuzzy_similarity(ts1, ts2, h0, h1, kappa0, kappa1):
    return 1 - ts1.fuzzy_similarity(ts2, h0, h1, kappa0, kappa1)


def evaluate_ucr(datasets='all'):
    results = {'dataset': [], 'train_size': [], 'test_size': [], 'ts_length': [], 'fuzzy': [], 'dtw': [],
               'fuzzy_eval_time': [], 'dtw_eval_time': []
            }

    sets = os.listdir('data/UCR')
    if datasets != 'all':
        sets = list(set(sets) & set(datasets))

    for s in sets:
        train_set, train_label, test_set, test_label = unpack(s)
        ts_len = len(train_set[0, 0])

        if ts_len > 512:
            continue

        print(f'Set: {s}. Time series length: {ts_len}. Train/test size: ({train_set.shape[0]}, {test_set.shape[0]}).')

        h = math.ceil(max(8, ts_len ** 0.5))

        fuzzy = Pipeline([
            ('col_transformer', ColumnTransformer([
                ('fuzzy_coefs', Pipeline([
                    ('fuzzy_coefs', FunctionTransformer(lambda x: get_fuzzy(x, h))),
                    ('max_scaler', MaxAbsScaler())
                ]), 0)
            ])),
            ('forest', RandomForestClassifier(max_depth=6, n_estimators=ts_len, random_state=42, n_jobs=-1))
        ])

        fuzzy_knn = Pipeline([
            ('forest', KNeighborsClassifier(metric=fuzzy_score, n_jobs=-1))
        ])

        dtw = Pipeline([
            ('col_transformer', ColumnTransformer([
                ('get_values', Pipeline([
                    ('get_values', FunctionTransformer(get_values)),
                    ('max_scaler', MaxAbsScaler())
                ]), 0)
            ])),
            ('knn', KNeighborsClassifier(metric=dtw_metric, n_jobs=-1))
        ])

        try:
            start_fuzzy = time.time()
            fuzzy.fit(train_set, train_label)
            pred_fuzzy = fuzzy.predict(test_set)
            acc_fuzzy = accuracy_score(test_label, pred_fuzzy)
            end_fuzzy = time.time()
            print(f'  Fuzzy: {acc_fuzzy * 100:.5f}% ({end_fuzzy - start_fuzzy:.3f} s)')

            start_dtw = time.time()
            dtw.fit(train_set, train_label)
            pred_dtw = dtw.predict(test_set)
            acc_dtw = accuracy_score(test_label, pred_dtw)
            end_dtw = time.time()
            print(f'  DTW:   {acc_dtw * 100:.5f}% ({end_dtw - start_dtw:.3f} s)')


            results['dataset'].append(s)
            results['train_size'].append(train_set.shape[0])
            results['test_size'].append(test_set.shape[0])
            results['ts_length'].append(len(train_set[0, 0]))
            results['fuzzy'].append(acc_fuzzy)
            results['dtw'].append(acc_dtw)
            # results['fuzzy_dtw'].append(acc_fuzzy_dtw)
            # results['fuzzy_dtw_eval_time'].append(end_fuzzy_dtw - start_fuzzy_dtw)
            results['fuzzy_eval_time'].append(end_fuzzy - start_fuzzy)
            results['dtw_eval_time'].append(end_dtw - start_dtw)

        except:
            print('An error occured. Skipping dataset.')
            continue

    pd.DataFrame(results).to_csv('results/eval_all_2.csv')


if __name__ == '__main__':
    evaluate_ucr(datasets=[
        'PowerCons', 'Coffee', 'BME', 'SmoothSubspace', 'Wafer', 'Plane', 'Strawberry', 'ItalyPowerDemand',
        'Meat', 'GunPoint', 'CBF', 'UMD', 'BeetleFly', 'Symbols', 'MoteStrain', 'ECG200', 'Trace',
        'SwedishLeaf', 'FaceFour', 'Yoga', 'Beef', 'Wine', 'Fish', 'Fungi', 'ShapesAll', 'Ham',
        'FiftyWords', 'ElectricDevices', 'BirdChicken'
    ])