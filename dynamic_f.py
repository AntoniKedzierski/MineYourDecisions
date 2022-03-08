from scipy.optimize import minimize
from scipy.integrate import quad

import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from core.data_types.time_series import TimeSeries

np.set_printoptions(suppress=True)


class DynamicFTransform:

    def __init__(self, n_coef):
        # Fuzzy coefs
        self.n_coef = n_coef
        self.coef1 = np.zeros(self.n_coef)
        self.coef2 = np.zeros(self.n_coef)
        
        # Split points
        self.n_splits = n_coef + 2
        self.x1 = np.zeros(self.n_splits) # Split points for fuzzy partition of ts1
        self.x2 = np.zeros(self.n_splits) # Split points for fuzzy partition of ts2
        
        # Function to evaluate
        self.f1 = None
        self.f2 = None

        # Iter
        self.optim_iter = 0


    def fit(self, ts1_, ts2_, **kwargs):
        if isinstance(ts1_, TimeSeries) and isinstance(ts2_, TimeSeries):
            # Copy to new objects
            ts1 = copy.deepcopy(ts1_)
            ts2 = copy.deepcopy(ts2_)

            # Split points
            self.x1 = np.linspace(0, ts1.enumeration[-1], self.n_splits)
            self.x2 = np.linspace(0, ts2.enumeration[-1], self.n_splits)

            # Reset time and normalize
            ts1 = ts1.reset_time().scale_down()
            ts2 = ts2.reset_time().scale_down()

            # Interpolated functions of values
            self.f1 = ts1.f_interpolate
            self.f2 = ts2.f_interpolate
        else:
            self.x1 = np.linspace(kwargs['domain_1'][0], kwargs['domain_1'][-1], self.n_splits)
            self.x2 = np.linspace(kwargs['domain_2'][0], kwargs['domain_2'][-1], self.n_splits)

            self.f1 = ts1_
            self.f2 = ts2_

        # Calculate coeficients
        self.calculate_fuzzy_coefs()

        print(self.coef1_derivative(0, 0))
        print(self.coef1_derivative(0, 1))
        print(self.coef1_derivative(0, 2))
        loss_f = self.loss_function
        return

        # Optimize task
        self.optim_iter = 0
        x = np.concatenate([self.x1, self.x2], axis=0)
        bounds = np.concatenate([[(0, self.x1[-1]) for i in range(self.n_splits)], [(0, self.x2[-1]) for i in range(self.n_splits)]], axis=0)
        # optim = minimize(lambda x: self.to_minimize(x, gradient=False), x, method='Powell', bounds=bounds)
        optim = minimize(lambda x: self.to_minimize(x), x, jac=True, method='L-BFGS-B', bounds=bounds, options={'disp': False})

        # Optimal partition
        self.x1 = optim.x[:self.n_splits]
        self.x2 = optim.x[self.n_splits:]

        # Display fit results
        print(f'Iters: {optim.nit}. Loss function decreased from {loss_f} to {optim.fun}.')

        

    def calculate_fuzzy_coefs(self):
        '''
        Calculates fuzzy partition based on split points for function f.
        :param f: Function to transform.
        :param split: Split points.
        '''
        for i in range(self.n_coef):
            a, b, c = self.x1[i], self.x1[i + 1], self.x1[i + 2]
            int_1_1, _ = quad(lambda x: (x - a) * self.f1(x), a, b, limit=100)
            int_1_2, _ = quad(lambda x: (c - x) * self.f1(x), b, c, limit=100)
            self.coef1[i] = 2 / (c - a) * (int_1_1 / (b - a) + int_1_2 / (c - b))

            a, b, c = self.x2[i], self.x2[i + 1], self.x2[i + 2]
            int_2_1, _ = quad(lambda x: (x - a) * self.f2(x), a, b, limit=100)
            int_2_2, _ = quad(lambda x: (c - x) * self.f2(x), b, c, limit=100)
            self.coef2[i] = 2 / (c - a) * (int_2_1 / (b - a) + int_2_2 / (c - b))
            
    
    def coef1_derivative(self, k, i):
        '''
        Computes the derivative of k-th coeficient w.r.t. i-th split point
        '''
        if k >= self.n_coef:
            return 0
        if i >= self.n_splits:
            return 0
        if i == k: # Partial w.r.t. a
            a, b, c = self.x1[i], self.x1[i + 1], self.x1[i + 2]
            int_1, _ = quad(lambda x: (x - b) * self.f1(x), a, b, limit=100)
            return 2 / (c - a) * (self.coef1[k] / 2 + int_1 / ((b - a) ** 2))
        if i - 1 == k: # Partial w.r.t. b
            a, b, c = self.x1[i - 1], self.x1[i], self.x1[i + 1]
            int_1, _ = quad(lambda x: (a - x) * self.f1(x), a, b, limit=100)
            int_2, _ = quad(lambda x: (c - x) * self.f1(x), b, c, limit=100)
            return 2 * (c - a) * (2 * self.f1(b) + int_1 / ((b - a) ** 2) + int_2 / ((c - b) ** 2))
        if i - 2 == k: # Partial w.r.t. c
            a, b, c = self.x1[i - 2], self.x1[i - 1], self.x1[i]
            int_1, _ = quad(lambda x: (x - b) * self.f1(x), b, c, limit=100)
            return -2 / (c - a) * (self.coef1[k] / 2 - int_1 / ((c - b) ** 2))
        return 0
    
    def coef2_derivative(self, k, i):
        '''
        Computes the derivative of k-th coeficient w.r.t. i-th split point
        '''
        if k >= self.n_coef:
            return 0
        if i >= self.n_splits:
            return 0
        if i == k:
            a, b, c = self.x2[i], self.x2[i + 1], self.x2[i + 2]
            int_1, _ = quad(lambda x: (x - b) * self.f2(x), a, b, limit=100)
            return 2 / (c - a) * (self.coef2[k] / 2 + int_1 / ((b - a) ** 2))
        if i - 1 == k:
            a, b, c = self.x2[i - 1], self.x2[i], self.x2[i + 1]
            int_1, _ = quad(lambda x: (a - x) * self.f2(x), a, b, limit=100)
            int_2, _ = quad(lambda x: (c - x) * self.f2(x), b, c, limit=100)
            return 2 * (c - a) * (2 * self.f2(b) + int_1 / ((b - a) ** 2) + int_2 / ((c - b) ** 2))
        if i - 2 == k:
            a, b, c = self.x2[i - 2], self.x2[i - 1], self.x2[i]
            int_1, _ = quad(lambda x: (x - b) * self.f2(x), b, c, limit=100)
            return -2 / (c - a) * (self.coef2[k] / 2 - int_1 / ((c - b) ** 2))
        return 0
    

    @property
    def loss_function(self):
        return sum(np.power(self.coef1 - self.coef2, 2))
    
    @property
    def gradient(self):
        result = np.zeros(2 * self.n_splits)

        for i in range(self.n_coef):
            result[i] += 2 * self.coef1_derivative(i, i) * (self.coef1[i] - self.coef2[i])
            result[i + self.n_splits] += 2 * self.coef2_derivative(i, i) * (self.coef2[i] - self.coef1[i])

        for i in range(1, self.n_coef + 1):
            result[i] += 2 * self.coef1_derivative(i - 1, i) * (self.coef1[i - 1] - self.coef2[i - 1])
            result[i + self.n_splits] += 2 * self.coef2_derivative(i - 1, i) * (self.coef2[i - 1] - self.coef1[i - 1])

        for i in range(2, self.n_coef + 2):
            result[i] += 2 * self.coef1_derivative(i - 2, i) * (self.coef1[i - 2] - self.coef2[i - 2])
            result[i + self.n_splits] += 2 * self.coef2_derivative(i - 2, i) * (self.coef2[i - 2] - self.coef1[i - 2])

        return result


    def to_minimize(self, x, gradient=True):
        self.plot_inv(self.x1)
        self.optim_iter += 1

        self.x1 = x[:self.n_splits]
        self.x2 = x[self.n_splits:]
        self.calculate_fuzzy_coefs()

        loss = self.loss_function
        print(f'Iteration {self.optim_iter}, loss: {loss}')

        if gradient:
            return loss, self.gradient
        return loss

    def inverse(self, x):
        result1 = np.zeros(len(x))
        result2 = np.zeros(len(x))

        for i in range(len(x)):
            x_arg = x[i]
            if x_arg <= self.x1[1] or x_arg >= self.x1[-2]:
                result1[i] = None
            if x_arg <= self.x2[1] or x_arg >= self.x2[-2]:
                result2[i] = None
            for j in range(1, self.n_coef - 1):
                a, b, c = self.x1[j], self.x1[j + 1], self.x1[j + 2]
                result1[i] += max(0, min((x_arg - a) / (b - a), (c - x_arg) / (c - b))) * self.coef1[j]
                a, b, c = self.x2[j], self.x2[j + 1], self.x2[j + 2]
                result2[i] += max(0, min((x_arg - a) / (b - a), (c - x_arg) / (c - b))) * self.coef2[j]

        return result1, result2

    def plot_inv(self, x):
        plot_f1, plot_f2 = self.inverse(x)
        sns.lineplot(x=x, y=plot_f1)
        sns.lineplot(x=x, y=plot_f2)
        plt.show()


if __name__ == '__main__':
    # apple = TimeSeries.read_csv('data/djia_composite/AAPL.csv', name='Apple')
    # cat = TimeSeries.read_csv('data/djia_composite/CAT.csv', name='Caterpillar')

    def f(x):
        return x

    def g(x):
        return x

    df = DynamicFTransform(n_coef=3)
    df.fit(f, g, domain_1=(0, 1), domain_2=(0, 1))

