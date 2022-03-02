import numbers
import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Sequence

import numpy as np


class FTransform:
    # ==========================================
    # IN-CLASSES (FUZZY SETS)
    # ==========================================
    class FuzzySet:
        left = None
        right = None

        def __init__(self, left, right):
            self.left = left
            self.right = right

        def __getitem__(self, item):
            if item >= self.left and item <= self.right:
                return 1
            return 0

    class TriangularFuzzySet(FuzzySet):
        peek = None
        peek_value = None

        def __init__(self, left, peek, right, peek_value=1):
            super().__init__(left, right)
            self.peek = peek
            self.peek_value = peek_value

        def __getitem__(self, item):
            if item < self.left or item > self.right:
                return 0
            return min(self.peek_value / (self.peek - self.left) * item - self.peek_value * self.left / (self.peek - self.left),
                       self.peek_value / (self.peek - self.right) * item - self.peek_value * self.right / (self.peek - self.right))

        def __str__(self):
            return f'[{self.left}, {self.peek}, {self.right}]'

    # ==========================================
    # INVERSE TRANSFORM
    # ==========================================
    class InvertTransform:
        def __init__(self, direct_transform, degree):
            self.fuzzy_sets_0 = direct_transform.fuzzy_sets_0
            self.fuzzy_sets_1 = direct_transform.fuzzy_sets_1
            self.direct_coefs_0 = direct_transform.direct_coefs_0
            self.direct_coefs_1 = direct_transform.direct_coefs_1
            self.n_elem_0 = direct_transform.n_elem_0
            self.n_elem_1 = direct_transform.n_elem_1
            self.degree = degree

        def __getitem__(self, item):
            if isinstance(item, numbers.Number):
                return self.get_value(item)
            elif isinstance(item, (Sequence, np.ndarray)):
                result = []
                for x in item:
                    y = 1
                    result.append(self.get_value(x))
                return np.array(result)

        def get_value(self, x):
            total = 0.0
            if x <= self.fuzzy_sets_0[0].right or x >= self.fuzzy_sets_0[-1].left:
                return None
            if self.degree >= 0:
                for i in range(1, self.n_elem_0 - 1):
                    total += self.fuzzy_sets_0[i][x] * self.direct_coefs_0[i]
            if self.degree >= 1:
                for i in range(1, self.n_elem_1 - 1):
                    total += self.fuzzy_sets_1[i][x] * self.direct_coefs_1[i] * (x - self.fuzzy_sets_1[i].peek)
            return total

    # ==========================================
    # CLASS METHODS
    # ==========================================
    def __init__(self, left, right, h0, h1):
        self.left = left
        self.right = right
        self.fuzzy_sets_0 = []
        self.fuzzy_sets_1 = []
        self.direct_coefs_0 = []
        self.direct_coefs_1 = []

        self.n_elem_0 = math.ceil((right - left) / h0)
        div_points_0 = np.linspace(left, right, self.n_elem_0 + 2)
        for i in range(self.n_elem_0):
            self.fuzzy_sets_0.append(FTransform.TriangularFuzzySet(div_points_0[i], div_points_0[i + 1], div_points_0[i + 2]))

        self.n_elem_1 = math.ceil((right - left) / h1)
        div_points_1 = np.linspace(left, right, self.n_elem_1 + 2)
        for i in range(self.n_elem_1):
            self.fuzzy_sets_1.append(FTransform.TriangularFuzzySet(div_points_1[i], div_points_1[i + 1], div_points_1[i + 2]))

    def __str__(self):
        result = ''
        for i in range(self.n_elem):
            result += self.fuzzy_sets[i].__str__() + f': {self.direct_coefs_0[i]};\t{self.direct_coefs_1}\n'
        return result

    def calc_membership(self, x, degree) -> float:
        if degree == 0:
            total = []
            for f_set in self.fuzzy_sets_0:
                total.append(f_set[x])
            return total
        elif degree == 1:
            total = []
            for f_set in self.fuzzy_sets_1:
                total.append(f_set[x])
            return total
        else:
            raise ValueError(f'Degree of {degree} is not supported.')

    def fit(self, x, y):
        if len(x) != len(y):
            raise ValueError('Lists x and y have different lengths.')

        for f in self.fuzzy_sets_0:
            total_y_0 = 0.0
            total_f_0 = 0.0
            for i in range(len(x)):
                total_y_0 += f[x[i]] * y[i]
                total_f_0 += f[x[i]]
            self.direct_coefs_0.append(total_y_0 / total_f_0 if total_f_0 != 0 else None)

        for f in self.fuzzy_sets_1:
            total_y_1 = 0.0
            total_f_1 = 0.0
            for i in range(len(x)):
                total_y_1 += f[x[i]] * y[i] * (x[i] - f.peek)
                total_f_1 += f[x[i]] * (x[i] - f.peek) ** 2
            self.direct_coefs_1.append(total_y_1 / total_f_1 if total_f_1 != 0 else None)

    def plot_fuzzy_sets(self):
        x_0 = np.arange(self.left, self.right + 0.1, 0.1)
        y_0 = np.zeros(len(x_0))
        for i in range(len(x_0)):
            for f_set in self.fuzzy_sets_0:
                y_0[i] += f_set[x_0[i]]

        x_1 = np.arange(self.left, self.right + 0.1, 0.1)
        y_1 = np.zeros(len(x_1))
        for i in range(len(x_1)):
            for f_set in self.fuzzy_sets_1:
                y_1[i] += f_set[x_1[i]]

        label_0 = ['0-degree'] * len(x_0)
        label_1 = ['1-degree'] * len(x_1)

        plot_data = pd.DataFrame({
            'x': np.concatenate([x_0, x_1]),
            'y': np.concatenate([y_0, y_1]),
            'label': label_0 + label_1
        })

        sns.lineplot(x='x', y='y', hue='label', data=plot_data)
        plt.show()

    # ==========================================
    # PROPERTIES
    # ==========================================
    @property
    def inv0(self):
        return FTransform.InvertTransform(self, degree=0)

    @property
    def inv1(self):
        return FTransform.InvertTransform(self, degree=1)
