import functools

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
from scipy import signal
from scipy.stats import norm
import dtaidistance

from core_v2.time_series import TimeSeries


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

cool_sets = ['PowerCons', 'Coffee', 'BME', 'SmoothSubspace', 'Wafer', 'Plane', 'Strawberry', 'ItalyPowerDemand',
             'Meat', 'GunPoint', 'CBF', 'UMD', 'BeetleFly', 'Symbols', 'MoteStrain', 'ECG200', 'Trace',
             'SwedishLeaf', 'FaceFour', 'Yoga', 'Beef', 'Wine', 'Fish', 'Fungi', 'ShapesAll', 'Ham',
             'FiftyWords', 'ElectricDevices', 'BirdChicken']


def get_fuzzy(column, h):
    coefs = []
    for ts in column:
        coefs.append(ts.fuzzy_coefs(h))
    return np.asarray(coefs)


def get_slopes(column, h0, h1):
    slopes = []
    for ts in column:
        slopes.append(zip(ts.fuzzy_coefs(h0), ts.fuzzy_slope_coefs(h1)[:, 1]))
    print(slopes)


def get_values(column):
    values = []
    for ts in column:
        values.append(ts.values)
    return np.asarray(values)


def ts_max_scaling(column):
    return np.asarray([ts.max_scaling() for ts in column])


def dtw_metric(x, y):
    return dtaidistance.dtw.distance_fast(x, y, window=10, use_pruning=True)


def fuzzy_score(F1, F2):
    return np.mean(np.abs(F1 - F2) / (np.abs(F1) + np.abs(F2)))


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


def analyze_dataset(dir, path_to_dir='data/UCR/'):
    train_set, train_label, test_set, test_label = unpack(dir, path_to_dir)
    gp = pd.read_csv('../data/gorecki_piasecki.csv')
    other_methods_perf = gp.loc[gp['dataset'] == dir, :].drop('dataset', axis=1).melt(var_name='method', value_name='accuracy').sort_values(by='accuracy', ascending=False)
    config = pd.read_csv('../data/config.csv', sep=' ')

    print(f"--- Analizowanie zbioru {dir} ---")

    # Liczności klas
    classes = np.sort(np.unique(train_label))
    print(f"Zbiór zawiera {len(classes)} {'klasy' if len(classes) == 2 else 'klas'}. Rozkład liczności w klasach:\n", end='')
    for c in classes:
        print(f'{c:6}', end='')
    print()
    for c in classes:
        print(f'{train_label[train_label == c].shape[0]:6}', end='')
    print()

    # narysuj po 3 obserwacje z każdej klasy
    sample_plot_path = path_to_dir + dir + '/' + dir + '_TRAIN.tsv'
    sample_plot = pd.read_csv(sample_plot_path, sep='\t', header=None)
    sample_plot = sample_plot.groupby(by=0).head(3).reset_index().rename(columns={0: 'class', 'index': 'series'}).melt(id_vars=['series', 'class'], value_name='value', var_name='timestamp')

    for ts in sample_plot['series'].unique():
        c = sample_plot.loc[sample_plot['series'] == ts, 'class'].head(1).item()
        sns.lineplot(x='timestamp', y='value', data=sample_plot.loc[sample_plot['series'] == ts, :])
        plt.title(f'Zbiór {dir}, klasa {c}.')
        plt.savefig(f'../results/plots/samples/{dir}_class_{c}_{ts}.png')
        plt.show()
        plt.close()

    print('Wykonywanie predykcji...', end='')

    ts_len = len(train_set[0, 0])
    h = math.ceil(max(8, ts_len ** 0.5))
    max_depth = 6
    num_trees = ts_len

    if dir in list(config['dataset']):
        h = config.loc[config['dataset'] == dir, 'h'].item()
        max_depth = config.loc[config['dataset'] == dir, 'max_depth'].item()
        num_trees = config.loc[config['dataset'] == dir, 'num_trees'].item()
    else:
        config = pd.concat([config, pd.DataFrame([[dir, h, max_depth, num_trees]], columns=['dataset', 'h', 'max_depth', 'num_trees'])], axis=0)
    config.to_csv('../data/config.csv', sep=' ', index=False)

    fuzzy = Pipeline([
        ('col_transformer', ColumnTransformer([
            ('fuzzy_coefs', Pipeline([
                ('fuzzy_coefs', FunctionTransformer(lambda x: get_fuzzy(x, h))),
                ('max_scaler', MaxAbsScaler())
            ]), 0)
        ])),
        ('forest', RandomForestClassifier(max_depth=max_depth, n_estimators=num_trees, random_state=42, n_jobs=-1))
    ])

    fuzzy.fit(train_set, train_label)
    pred_fuzzy = fuzzy.predict(test_set)
    acc_fuzzy = accuracy_score(test_label, pred_fuzzy)

    best_accuracy = max(other_methods_perf['accuracy'])
    best_methods = ' '.join(other_methods_perf.loc[other_methods_perf['accuracy'] == best_accuracy, 'method'])

    print(f'\n  Dokładność lasu losowego: ${acc_fuzzy * 100:.2f}\\%$.')
    print(f'  Dokładność KNN: ${acc_fuzzy_knn * 100:.2f}\\%$.')
    print(f'  Maks. dokładność osiągnięta przez {best_methods} wynosi ${best_accuracy * 100:.2f}\\%$.')
    print(f'  Mediana dokładności klasyfikatorów: ${other_methods_perf["accuracy"].median() * 100:.2f}\\%$.')
    if best_accuracy <= 0.85:
        print('Mierne wyniki klasyfikacji dla wszystkich metod.')

    # Wyplotuj skuteczność na tym zbiorze
    plot_performance = copy.deepcopy(other_methods_perf).sort_values(by='accuracy', ascending=True)
    plot_performance['F-transformata'] = 'Nie'
    plot_performance['x'] = 'Metoda'
    plot_performance = pd.concat([
        plot_performance,
        pd.DataFrame(
            [['F-transformata', acc_fuzzy, 'Tak', 'Metoda']],
            columns=['method', 'accuracy', 'F-transformata', 'x']
        )], axis=0).reset_index(drop=True)

    ax = sns.boxplot(x='x', y='accuracy', data=plot_performance, boxprops={'alpha': 0.4})
    # ax.set(ylim=(0, 1.1))
    sns.stripplot(x='x', y='accuracy', hue='F-transformata', data=plot_performance, s=7, ax=ax)
    plt.title(f'Jakość klasyfikatorów na zbiorze {dir}')
    plt.xlabel('')
    plt.ylabel('Dokładność')
    plt.savefig(f'../results/plots/classifiers/{dir}.png', dpi=100)
    plt.show()

    return acc_fuzzy

    # Wypisz błędnie zaklasyfikowane obserwacje
    # if acc_fuzzy < 1:
    #     wrong_answers = np.c_[pred_fuzzy, test_label, test_set][pred_fuzzy != test_label]
    #     for a in wrong_answers:
    #         a[2].plot(title=f'Prawdziwa klasa: {a[1]}. Przypisana klasa: {a[0]}')


def gorecki_piasecki_build():
    part1 = pd.read_csv('../gp_part1.txt', sep=' ')
    part2 = pd.read_csv('../gp_part2.txt', sep=' ')
    gp = part1.merge(part2, left_on='dataset', right_on='dataset')
    gp.iloc[:, 1:] = gp.iloc[:, 1:].apply(lambda x: round((100 - x) / 100, 5), axis=1)
    gp.to_csv('../data/gorecki_piasecki.csv', index=False)


def best_accuracies():
    gp = pd.read_csv('../data/gorecki_piasecki.csv').melt(id_vars='dataset', var_name='method', value_name='accuracy').sort_values(by='accuracy', ascending=False).groupby(by='dataset').head(1).sort_values(by='dataset')
    print(gp)


def compare_methods():
    metrics = pd.read_csv('../results/metrics.csv')
    scores = metrics.iloc[:, 1:].melt(var_name='method', value_name='error_rate')
    scores['error_rate'] = 1 - scores['error_rate']
    methods = scores.groupby('method').agg(['mean', 'std'])
    methods.columns = ['_'.join(x) for x in methods.columns.ravel()]
    kmeans = KMeans(n_clusters=3)
    methods['class'] = kmeans.fit_predict(methods)
    methods = methods.reset_index()

    ## Mapa kolorów dla klas
    cmap = {
        0: {
            'fill': '#fe5f55',
            'edge': '#510600'
        },
        1: {
            'fill': '#bdd5ea',
            'edge': '#4589C4'
        },
        2: {
            'fill': '#7fb069',
            'edge': '#3F5E31'
        }
    }

    # Rysujemy
    plt.figure(figsize=(18, 12), dpi=240)
    for i, method in methods.iterrows():
        plt.text(
            x=method['error_rate_mean'],
            y=method['error_rate_std'],
            s=method['method'],
            size=14,
            ha='center',
            va='center',
            bbox={
                'boxstyle': 'round',
                'ec': cmap[method['class']]['edge'],
                'fc': cmap[method['class']]['fill']
            },
            alpha=0.8
        )

    plt.xlim(0, 0.63)
    plt.ylim(0.10, 0.27)
    plt.xlabel("Średni błąd")
    plt.ylabel("Odchylenie standardowe błędu")
    # plt.savefig('../results/methods_clusters.png', dpi=240)
    # plt.show()

    f_transform = metrics.loc[:, ['dataset', 'F-transformata']].rename(columns={'F-transformata': 'accuracy'})
    wilcoxon = {'method': [], 'statistics': []}
    for method in metrics.columns:
        if method == 'F-transformata' or method == 'dataset':
            continue
        diffs = f_transform['accuracy'] - metrics[method]
        ranked_diffs = diffs.reset_index().sort_values(by=0).rename(columns={'index': 'set'}).reset_index(drop=True).reset_index().rename(columns={'index': 'rank', 0: 'diff'})
        R_plus = sum(ranked_diffs.loc[ranked_diffs['diff'] > 0, 'rank']) + 0.5 * sum(ranked_diffs.loc[ranked_diffs['diff'] == 0, 'rank'])
        R_minus = sum(ranked_diffs.loc[ranked_diffs['diff'] < 0, 'rank']) + 0.5 * sum(ranked_diffs.loc[ranked_diffs['diff'] == 0, 'rank'])
        wilcoxon['method'].append(method)
        wilcoxon['statistics'].append(min(R_plus, R_minus))

    N = metrics.shape[0]
    wilcoxon = pd.DataFrame(wilcoxon)
    wilcoxon['norm_approx'] = (wilcoxon['statistics'] - 0.25 * N * (N + 1)) / math.sqrt(N * (N + 1) * (2 * N + 1) / 24)
    wilcoxon['p_value'] = norm.cdf(wilcoxon['norm_approx'])
    print(wilcoxon.loc[wilcoxon['p_value'] >= 0.005, :].sort_values(by='p_value', ascending=False))


def eval_scores():
    res = {'dataset': [], 'fuzzy': []}
    for s in cool_sets:
        res['dataset'].append(s)
        res['fuzzy'].append(analyze_dataset(s, path_to_dir='../data/UCR/'))

    gp = pd.read_csv('../data/gorecki_piasecki.csv')
    gp = gp.loc[gp['dataset'].isin(cool_sets), :]
    res = pd.DataFrame(res).merge(gp, left_on='dataset', right_on='dataset')
    res.to_csv('../results/metrics.csv', index=False)


if __name__ == '__main__':
    analyze_dataset('PowerCons', path_to_dir='../data/UCR/')



