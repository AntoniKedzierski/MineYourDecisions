from scipy.optimize import minimize
import numpy as np
import seaborn as sns

from core.data_types.time_series import TimeSeries

if __name__ == '__main__':
    apple = TimeSeries.read_csv('data/djia_composite/AAPL.csv', name='Apple')
    f = apple.values
    f_dash = np.concatenate([[None], apple.diff])

    cat = TimeSeries.read_csv('data/djia_composite/CAT.csv', name='Caterpillar')
    g = cat.values
    g_dash = np.concatenate([[None], cat.diff])

    domain = np.arange(0, len(f), 1)
    n_sets = 20

    # Zaczynamy z domyślnego podziału
    f_sets = np.linspace(0, domain[-1], n_sets + 2)
    g_sets = np.linspace(0, domain[-1], n_sets + 2)

    # Trójkątny zbiór rozmyty:
    def trangf(c1, c2, c3, x):
        return max(0, min((x - c1) / (c2 - c1), (c3 - x) / (c3 - c2)))

    # Funkcja do optymalizacji
    def f_optim():
        coefs_f = np.zeros(n_sets)
        coefs_g = np.zeros(n_sets)
        for i in range(1, n_sets + 1):
            total_f = 0
            total_g = 0
            mem_f = 0
            mem_g = 0
            for j in range(len(f)):
                mem = trangf(f_sets[i - 1], f_sets[i], f_sets[i + 1], j)
                total_f += f[j] * mem
                mem_f += mem
            for j in range(len(g)):
                mem = trangf(g_sets[i - 1], g_sets[i], g_sets[i + 1], j)
                total_g += g[j] * mem
                mem_g += mem
            coefs_f[i - 1] = total_f / mem_f
            coefs_g[i - 1] = total_g / mem_g

        print(coefs_f)
        print(coefs_g)

    f_optim()

