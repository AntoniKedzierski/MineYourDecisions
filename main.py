import os
import time
import numpy as np
import pandas as pd
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
    return dtaidistance.dtw.distance_fast(x, y, use_pruning=True)


if __name__ == '__main__':
    results = { 'dataset': [], 'train_size': [], 'test_size': [], 'ts_length': [], 'fuzzy': [], 'dtw': [], 'fuzzy_eval_time': [], 'dtw_eval_time': [] }
    for set in os.listdir('data/UCR'):
        train_set, train_label, test_set, test_label = unpack(set)
        print(f'Set: {set}. Time series length: {len(train_set[0, 0])}. Train/test size: ({train_set.shape[0]}, {test_set.shape[0]}).')
        h = max(10, train_set.shape[0] ** 0.5)

        fuzzy = Pipeline([
            ('col_transformer', ColumnTransformer([
                ('fuzzy_coefs', Pipeline([
                    ('fuzzy_coefs', FunctionTransformer(lambda x: get_fuzzy(x, h))),
                    ('max_scaler', MaxAbsScaler())
                ]), 0)
            ])),
            ('forest', RandomForestClassifier(max_depth=4, n_estimators=256, random_state=42, n_jobs=-1))
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
            print(f'  DTW:   {acc_dtw * 100:.5f}% ({end_dtw - start_dtw:.3f} s)\n')

            results['dataset'].append(set)
            results['train_size'].append(train_set.shape[0])
            results['test_size'].append(test_set.shape[0])
            results['ts_length'].append(len(train_set[0, 0]))
            results['fuzzy'].append(acc_fuzzy)
            results['fuzzy_eval_time'] = end_fuzzy - start_fuzzy
            results['dtw_eval_time'] = end_dtw - start_dtw
            results['dtw'].append(acc_dtw)

        except:
            print('An error occured. Skipping dataset.')
            continue

    pd.DataFrame(results).to_csv('results/eval_all_1.csv')