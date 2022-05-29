import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt

from scipy.stats import shapiro, kstest
from scipy.stats import t, norm

from time_series import TimeSeries


def find_anomalies(ts : TimeSeries, conf, h):
    anomalies_all = ts.fuzzy_anomalies(h).values
    anomalies = anomalies_all[~np.isnan(anomalies_all)]
    z_scores = (anomalies - np.mean(anomalies)) / np.std(anomalies)

    # Test Shapiro dla normalności
    # Zajrzeć do seminarium Maćka.
    # Anderson-Darling.
    # Metoda Holma-Bonferroniego.
    # Agregować p-wartości testów odrzucających obserwacje odstające.
    # _, p_value = shapiro(anomalies)
    # plt.plot(np.sort(z_scores), np.linspace(0, 1, len(z_scores), endpoint=False))
    # plt.plot(np.sort(z_scores), norm.cdf(np.sort(z_scores)))
    # plt.title(f'Dystrybuanta empiryczna anomalii vs N(0, 1), {ts.name}')
    # plt.show()
    # if p_value < conf / 2:
    #     raise AssertionError(f'Anomalies are not normally distributed: {p_value:.6f}')

    # Test z-score dla wykrycia anomalii
    # Zaznacz obserwacje spoza nadego kwantyla (dwustronie):
    is_anomaly = np.repeat(False, len(ts))
    is_anomaly[~np.isnan(anomalies_all)] = abs(z_scores) > abs(norm.ppf(conf / 4))
    ts.plot(mark_points=is_anomaly) #, save_path=f'../results/anomalies/anomalies_{ts.name}.png')


def anomalies_dow_jones():
    all_stocks = []
    for s in os.listdir('../data/djia_composite'):
        ts = TimeSeries.read_csv(f'../data/djia_composite/{s}', name=s.split('.')[0])
        try:
            find_anomalies(ts, 0.05, 7)
        except:
            continue




if __name__ == '__main__':
    ts = TimeSeries.read_csv(f'../data/djia_composite/AAPL.csv', name='AAPLE')
    # find_anomalies(ts, 0.05, 15)

    # ts.fuzzy_plot(3)
    # ts.fuzzy_plot(5)
    # ts.fuzzy_plot(7)
    # ts.fuzzy_plot(9)
    # ts.fuzzy_plot(11)
    # ts.fuzzy_plot(13)

    for i in range(3, 21, 2):
        ts.fuzzy_plot(i)
        find_anomalies(ts, 0.05, i)
