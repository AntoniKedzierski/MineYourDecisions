import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import typing
import numbers


class TimeSeries:
    def __init__(self, *args, **kwargs):
        name = 'Time Series'
        value_names = ['values']
        value_columns = 'all'
        time_index_name = 'timestamp'
        time_dtype = 'datetime'
        time_column = 0
        no_time = False

        if 'name' in kwargs:
            name = kwargs['name']

        if 'value_columns' in kwargs:
            value_columns = kwargs['value_columns']

        if 'time_index_name' in kwargs:
            time_index_name = kwargs['time_index_name']

        if 'time_dtype' in kwargs:
            time_dtype = kwargs['time_dtype']

        if 'time_column' in kwargs:
            time_column = kwargs['time_column']

        if len(args) == 1:
            if isinstance(args[0], pd.DataFrame):
                if args[0].shape[1] >= 2:
                    time = np.asarray(args[0].iloc[:, time_column])
                    if value_columns == 'all':
                        val_ix = list(set(np.arange(0, args[0].shape[1], 1)) - set([time_column]))
                    elif isinstance(value_columns, int):
                        val_ix = [value_columns]
                    elif isinstance(value_columns, (pd.Series, np.ndarray, typing.Sequence)):
                        val_ix = list(value_columns)
                        if not isinstance(value_columns[0], int):
                            raise ValueError('Bad type for first element in "value_columns", %s was given.' % (type(value_columns[0])))
                    else:
                        raise ValueError('Bad type for "value_columns", %s was given.' % (type(value_columns)))
                    values = np.asarray(args[0].iloc[:, val_ix])
                    value_names = list(args[0].columns[val_ix])
                if args[0].shape[1] == 1:
                    no_time = True
                    values = np.asarray(args[0].iloc[:, 0])
                    time = np.arange(0, args[0].shape[0], 1)
                    value_names = list(args[0].columns[0])
            elif isinstance(args[0], (pd.Series, np.ndarray)):
                no_time = True
                values = np.asarray(args[0]).reshape((-1, 1))
                time = np.arange(0, args[0].shape[0], 1)
            elif isinstance(args[0], typing.Sequence):
                no_time = True
                values = np.asarray(args[0]).reshape((-1, 1))
                time = np.arange(0, len(args[0]), 1)
            else:
                raise ValueError('Unregonized data dtype of first argument. %s was given' % (type(args[0])))
        elif len(args) == 2:
            if isinstance(args[0], (pd.Series, np.ndarray, typing.Sequence)) and isinstance(args[1], (pd.Series, np.ndarray, typing.Sequence)):
                values = np.asarray(args[0]).reshape((-1, 1))
                time = np.asarray(args[1])
                if values.shape[0] != time.shape[0]:
                    raise ValueError('Time and values have different lengths: %d and %d' % (time.shape[0], values.shape[0]))
            else:
                raise ValueError('Unregonized data dtype of arguments. %s and %s were given' % (type(args[0]), type(args[1])))
        else:
            raise ValueError('Passed more than two arguments.')

        self.name = name
        self.value_names = value_names
        self.time_dtype = time_dtype
        self.time_index_name = time_index_name

        df = { self.time_index_name: time }
        for i, v_name in enumerate(self.value_names):
            df[v_name] = values[:, i]

        self.df = pd.DataFrame(df)
        if not no_time:
            self.df[self.time_index_name] = pd.to_datetime(self.df[self.time_index_name])


    # ===================
    # Python methods
    # ===================
    def __str__(self):
        return self.df.__str__()

    def __len__(self):
        return self.values.shape[0]


    # ===================
    # Properites
    # ===================
    @property
    def values(self):
        return np.asarray(self.df.iloc[:, 1])

    @property
    def timestamp(self):
        return np.asarray(self.df.iloc[:, 0], dtype=self.time_dtype)

    @property
    def enumaration(self):
        return np.asarray(self.df.index)


    # ===================
    # Utils
    # ===================
    def plot(self, features='value'):
        if features == 'value':
            sns.lineplot(x=self.time_index_name, y=self.value_names[0], data=self.df)
        elif isinstance(features, (pd.Series, np.ndarray, typing.Sequence)) or features == 'all':
            columns = []
            if features == 'all':
                columns = self.value_names
            else:
                for f in features:
                    if isinstance(f, int):
                        columns.append(self.value_names[f])
                    else:
                        columns.append(f)
            plot_data = self.df.melt(id_vars=self.time_index_name, value_vars=columns, value_name='value', var_name='series')
            sns.lineplot(x=self.time_index_name, y='value', hue='series', data=plot_data)
        plt.title(self.name)
        plt.show()


    # ===================
    # TS operations
    # ===================
    def lag(self, k=1):
        '''
        Lags all columns.
        :param k: Shift in time.
        :return: Time seriew.
        '''
        lagged = pd.concat([self.df, self.df.loc[:, self.value_names].shift(k).add_suffix('_sft')], axis=1).iloc[k:, :]
        to_drop = []
        for c in self.value_names:
            lagged[c] = lagged[c] - lagged[f"{c}_sft"]
            to_drop.append(f"{c}_sft")
        lagged = lagged.reset_index(drop=True).drop(to_drop, axis=1)
        return TimeSeries(lagged, name=f'Lagged {self.name} with {k} delay')


    # ===================
    # SQL Operations
    # ===================
    def join(self, ts, how='inner', by='index', **kwargs):
        '''
        :param ts: Time series to join with.
        :param how: Type of join. Can be inner, outer, left or right.
        :param by: Indicates which column should be used as an index - index or time.
        :return: Time series.
        '''
        suffixes = (f'_{self.name.lower().replace(" ", "_")}', f'_{ts.name.lower().replace(" ", "_")}')
        if 'suffixes' in kwargs:
            suffixes = kwargs['suffixes']

        if by == 'index':
            merged = self.df.merge(ts.df, how=how, suffixes=suffixes, left_index=True, right_index=True)
        elif by == 'time':
            merged = self.df.merge(ts.df, how=how, left_on=self.time_index_name, right_on=ts.time_index_name, suffixes=suffixes)

        return TimeSeries(merged, name=f'Merged {self.name} and {ts.name}')


    # ===================
    # Fuzzy transform
    # ===================
    def fuzzy_coefs(self, partition):
        if not isinstance(partition, (np.ndarray, pd.Series, typing.Sequence)):
            if isinstance(partition, numbers.Number):
                partition = np.arange(self.enumaration[0], self.enumaration[-1] + partition, step=partition / 2)
            else:
                raise ValueError('Partition should be an iterative sequence, but %s was given.' % type(partition))

        partition = np.asarray(partition)
        n_coefs = partition.shape[0] - 2
        coefs = np.zeros(n_coefs)
        x = self.enumaration

        for i in range(n_coefs):
            a, b, c = partition[i], partition[i + 1], partition[i + 2]
            int_1 = 0.0
            int_2 = 0.0
            for j in x[(x >= a) & (x <= b)]:
                int_1 += (j - a) * self.values[j]
            for j in x[(x >= b) & (x <= c)]:
                int_1 += (c - j) * self.values[j]
            coefs[i] = 2 / (c - a) * (int_1 / (b - a) + int_2 / (c - b))

        return coefs

    def fuzzy_plot(self, partition):
        if not isinstance(partition, (np.ndarray, pd.Series, typing.Sequence)):
            if isinstance(partition, numbers.Number):
                partition = np.arange(self.enumaration[0], self.enumaration[-1] + partition, step=partition)
            else:
                raise ValueError('Partition should be an iterative sequence, but %s was given.' % type(partition))
        coefs = self.fuzzy_coefs(partition)
        plot_x = np.linspace(partition[0], partition[-1], 100)
        plot_y = np.zeros(plot_x.shape[0])

        for i in range(plot_x.shape[0]):
            for j in range(partition.shape[0] - 2):
                a, b, c = partition[j], partition[j + 1], partition[j + 2]
                x = plot_x[i]
                plot_y[i] += max(0, min((x - a) / (b - a), (c - x) / (c - b))) * coefs[j]
            if plot_x[i] <= partition[1] or plot_x[i] >= partition[-2]:
                plot_y[i] = None

        sns.lineplot(x=self.enumaration, y=self.values, linestyle='--', alpha=0.4)
        sns.lineplot(x=plot_x, y=plot_y)
        plt.show()

    # ===================
    # Static methods
    # ===================
    @staticmethod
    def read_csv(path, **kwargs):
        return TimeSeries(pd.read_csv(path), **kwargs)


if __name__ == '__main__':

        print(apple.join(cat))