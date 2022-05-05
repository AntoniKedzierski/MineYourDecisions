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


def get_values(column):
    values = []
    for ts in column:
        values.append(ts.values)
    return np.asarray(values)


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

    print(f'\n  Dokładność: ${acc_fuzzy * 100:.2f}\\%$.')
    print(f'  Maks. dokładność osiągnięta przez {best_methods} wynosi ${best_accuracy * 100:.2f}\\%$.')
    print(f'Mediana dokładności klasyfikatorów: ${other_methods_perf["accuracy"].median() * 100:.2f}\\%$.')
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
    gp = pd.read_csv('../data/gorecki_piasecki.csv')
    gp = gp.loc[gp['dataset'].isin(cool_sets), :].melt(id_vars='dataset', var_name='method', value_name='accuracy')
    ftrans = pd.read_csv('../results/eval_3.txt', sep=' ')[['Zbiór', 'f_trans']].rename(columns={'Zbiór': 'dataset', 'f_trans': 'accuracy'})
    ftrans['method'] = 'f-transform'
    ftrans['accuracy'] /= 100
    total = pd.concat([gp, ftrans], axis=0).reset_index(drop=True)
    sorted_index = list(total.groupby(by='method').median().sort_values(by='accuracy').index)
    sort_dict = dict(zip(sorted_index, range(len(sorted_index))))
    total['rank'] = total['method'].map(sort_dict)
    total = total.sort_values(by='rank')
    total_pvt = total.pivot(index='method', columns='dataset', values='accuracy')
    # plt.figure(figsize=(16, 20))
    # sns.boxplot(y='method', x='accuracy', data=total, orient='h')
    # plt.title('Porównanie metod klasyfikacji.')
    # plt.savefig('../results/plots/porownanie.png')
    # plt.show()
    kmeans = KMeans(4)
    pred = kmeans.fit_predict(total_pvt)
    silhouette_avg = silhouette_score(total_pvt, pred)
    print(
        "The average silhouette_score is :",
        silhouette_avg,
    )
    print(pd.DataFrame({'method': list(total_pvt.index), 'class': pred}).sort_values(by='class'))


if __name__ == '__main__':
    # train_set, train_label, test_set, test_label = unpack('PowerCons', path_to_dir='../data/UCR/')
    # train_set[2, 0].fuzzy_plot(12, title='F-transformata')
    # train_set[2, 0].fuzzy_anomalies(12).plot()
    # analyze_dataset('FiftyWords', path_to_dir='../data/UCR/')
    compare_methods()
    # best_accuracies()
    # W niektórych sytuacjach F-transformata prowadzi do zbyt dużych uogólnień, dlatego należy zwężać okno. (!)
    # Dla szczegółowych danych, różnią
    # Mało obserwacji - dużo drzew
    # Krótkie szeregi - małe okno