import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Sequence
import numbers
import warnings

from core.analysis.transforms import FTransform

pd.set_option('display.max_rows', 500)


class TimeSeries:
    # ==========================================
    # IN-CLASSES
    # ==========================================

    # Technicals stuff for indexing
    class TimeIndexer:
        def __init__(self, ts):
            self.ts = ts

        def __getitem__(self, t):
            return self.ts.values[np.where(self.ts.time == t)][0]

    # ==========================================
    # INITIALIZER
    # ==========================================

    # Build a series base on time and values
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], pd.DataFrame):
            if args[0].shape[1] != 2:
                raise ValueError('The data frame does not contain 2 columns.')

            if not isinstance(args[0].iloc[0, 1], (numbers.Number)):
                raise ValueError(f'Time series must have a numerical values. {type(args[0].iloc[1, 0])} was given.')

            self.time = np.array(args[0].iloc[:, 0])
            self.values = np.array(args[0].iloc[:, 1])

            if 'name' in kwargs:
                self.name = kwargs['name']
            else:
                self.name = args[0].columns[1]

        elif len(args) == 1 and isinstance(args[0], (Sequence, np.ndarray)):
            self.time = np.arange(0, args[0].__len__(), 1)
            self.values = np.array(args[0])

            if not isinstance(args[0][0], numbers.Number):
                raise ValueError('Time series must have a numerical values.')

            if 'name' in kwargs:
                self.name = kwargs['name']

        elif len(args) == 2:
            if args[0].__len__() != args[1].__len__():
                raise ValueError(f'Time and values have different length. Time: {args[0].__len__()}, values: {args[1].__len__()}')

            if not isinstance(args[1][0], numbers.Number):
                raise ValueError('Time series must have a numerical values.')

            self.time = np.array(args[0])
            self.values = np.array(args[1])

            if 'name' in kwargs:
                self.name = kwargs['name']
        else:
            return

        # Sort values, just in case
        arg_sorted = self.time.argsort()
        self.time = self.time[arg_sorted]
        self.values = self.values[arg_sorted]
        self.enumeration = np.arange(0, self.values.__len__(), 1)

        # Calculate FTransform
        h0 = 3
        h1 = 5
        if 'h0' in kwargs:
            h0 = kwargs['h0']
        if 'h1' in kwargs:
            h1 = kwargs['h1']

        self.FTransform = FTransform(0, self.enumeration[-1], h0, h1)
        self.FTransform.fit(self.enumeration, self.values)


    # To string
    def __str__(self):
        max_length = 8
        n = self.values.__len__()
        if n <= max_length:
            result = f'Series: {self.name}\n'
            for i in range(n):
                result += f'{self.time[i]}: {self.values[i]}\n'
            return result
        result = f'Series: {self.name}\n'
        for i in range(4):
            result += f'{self.time[i]}: {self.values[i]}\n'
        result += '...\n'
        for i in range(4):
            result += f'{self.time[n - 4 + i]:}: {self.values[n - 4 + i]}\n'
        return result

    # Length
    def __len__(self):
        return self.values.__len__()

    # Extracting data from time series
    def __getitem__(self, item):
        return self.values[item]

    def __setitem__(self, key, value):
        self.values[key] = value

    # At certain time point
    @property
    def at_time(self):
        return TimeSeries.TimeIndexer(self)

    # Plot
    def plot(self, title=None, labels=(None, None)):
        if title is None:
            plt.title(self.name)
        try:
            time = np.asarray(self.time, dtype='datetime64')
        except:
            time = self.time
        sns.lineplot(x = time, y = self.values)
        if labels[0] is not None:
            plt.xlabel(labels[0])
        if labels[1] is not None:
            plt.ylabel(labels[1])
        plt.show()

    # ==========================================
    # PROPERTIES
    # ==========================================
    @property
    def start(self):
        return min(self.time)

    @property
    def end(self):
        return max(self.time)

    @property
    def as_pandas(self):
        name = self.name if self.name is not None else 'values'
        return pd.DataFrame({'timestamp': self.time, name: self.values})

    @property
    def f_coefs(self):
        return np.asarray(self.FTransform.direct_coefs_0 + self.FTransform.direct_coefs_1), len(self.FTransform.direct_coefs_0)

    # ==========================================
    # TRANSFORMATIONS
    # ==========================================
    def reset_time(self, inplace=False):
        if not inplace:
            return TimeSeries(self.values, name=self.name)
        self.time = np.arange(0, self.values.__len__(), 1)

    def scale_down(self, inplace=False):
        if not inplace:
            return TimeSeries(self.time, self.values / max(abs(self.values)), name=self.name)
        self.values /= max(abs(self.values))

    def standard_scale(self, inplace=False):
        if not inplace:
            return TimeSeries(self.time, (self.values - np.mean(self.values)) / np.std(self.values), name=self.name)
        self.values = (self.values - np.mean(self.values)) / np.std(self.values)

    # ==========================================
    # JOINS
    # ==========================================

    # Join two timeseries
    def time_join(self, ts):
        '''
        Time join treats time indices as ordered sequences. It findes common start and end, then joins
        everything between those points. At the end, it approximates missing values.
        :param ts: Other time series
        :return: Multi-valued time series with two columns as two time series.
        '''
        start = max(min(self.time), min(ts.time))
        end = min(max(self.time), max(ts.time))

        time_left = list(self.time[(self.time >= start) & (self.time <= end)])
        time_right = list(ts.time[(ts.time >= start) & (ts.time <= end)])

        name_left = self.name if self.name is not None else 1
        name_right = ts.name if ts.name is not None else 2

        if name_left == name_right:
            name_left = name_left + '_1'
            name_right = name_right + '_2'

        new_time = list(sorted(set(time_left + time_right)))
        df = { 'time': new_time }
        df[name_left] = [None] * len(new_time)
        df[name_right] = [None] * len(new_time)

        for i in range(len(new_time)):
            if new_time[i] in time_left:
                df[name_left][i] = self.at_time[new_time[i]]
            else:
                df[name_left][i] = None
            if new_time[i] in time_right:
                df[name_right][i] = ts.at_time[new_time[i]]
            else:
                df[name_right][i] = None

        for i in range(1, len(new_time) - 1):
            if df[name_left][i] is None:
                gap_size = 1
                before_value = df[name_left][i - 1]
                final_value = 0
                for j in range(i + 1, len(new_time) - 1):
                    if df[name_left][j] is None:
                        gap_size += 1
                    else:
                        final_value = df[name_left][j]
                        break
                step = (final_value - before_value) / (gap_size + 1)
                for j in range(i, i + gap_size):
                    df[name_left][j] = before_value + step * j

        for i in range(1, len(new_time) - 1):
            if df[name_right][i] is None:
                gap_size = 1
                before_value = df[name_right][i - 1]
                final_value = 0
                for j in range(i + 1, len(new_time) - 1):
                    if df[name_right][j] is None:
                        gap_size += 1
                    else:
                        final_value = df[name_right][j]
                        break
                step = (final_value - before_value) / (gap_size + 1)
                for j in range(i, i + gap_size):
                    df[name_right][j] = before_value + step * (i - j + 1)

        df = pd.DataFrame(df)
        return MultiTimeSeries(df)

    # ==========================================
    # STATIC METHODS
    # ==========================================
    @staticmethod
    def read_csv(path : str, header=0, column_subset=[0, 1], parse_dates=False, name=None):
        if isinstance(column_subset, int):
            df = pd.read_csv(path, header=header)
            return TimeSeries(df.iloc[:, column_subset].to_numpy(), name=name)
        elif isinstance(column_subset, Sequence):
            if len(column_subset) != 2:
                raise ValueError('You can select only one or two columns from .csv file')
            df = pd.read_csv(path, header=header, parse_dates=parse_dates)
            return TimeSeries(df.iloc[:, column_subset], name=name)
        else:
            raise ValueError('Bad arguments were given.')






class MultiTimeSeries(TimeSeries):
    # Override values, use a list of series. It will be 2-d numpy array.
    values = np.array([[]], dtype=float)

    # Override name, use a list of names for each series
    name = []

    # Override time indexer
    class TimeIndexer:
        def __init__(self, ts):
            self.ts = ts

        def __getitem__(self, item):
            return self.ts.values[np.where(np.time == item), :]

    # This can be initialized only with pandas dataframe with index of time column equal 0
    def __init__(self, dataframe : pd.DataFrame):
        if not isinstance(dataframe.iloc[0, 1], numbers.Number):
            raise ValueError(f'Time series must have a numerical values. {dataframe.iloc[0, 1]} was given.')

        self.time = dataframe.iloc[:, 0].to_numpy()
        self.values = dataframe.iloc[:, 1:dataframe.shape[1]].to_numpy(dtype=float)
        self.name = ['timestamp'] + list(dataframe.columns[1:dataframe.shape[1]])

        # Sort values, just in case
        arg_sorted = self.time.argsort()
        self.time = self.time[arg_sorted]
        self.values = self.values[arg_sorted, :]

    # To string
    def __str__(self):
        df = {}
        df['timestamp'] = self.time
        for i, n in enumerate(self.name):
            if i == 0: continue
            df[n] = self.values[:, i - 1]
        return pd.DataFrame(df).__str__()

    # Length
    def __len__(self):
        return len(self.time)

    # Extract one series
    def get_subseries(self, name):
        idx = np.where(np.asarray([n.lower() for n in self.name]) == name.lower())[0][0] - 1
        return TimeSeries(self.time, self.values[:, idx], name=name)

    # Get item
    def __getitem__(self, item):
        return TimeSeries(self.time, self.values[:, item], name=self.name[item + 1])

    # ==========================================
    # UTILITIES
    # ==========================================
    def plot(self, series='all', title=None, labels=(None, None)):
        data = { 'timestamp': [], 'series': [], 'value': [] }
        if series == 'all':
            for i, name in enumerate(self.name):
                if i == 0: continue
                data['timestamp'] += list(self.time)
                data['value'] +=  list(self.values[:, i - 1])
                data['series'] += [name] * len(self.time)
        elif isinstance(series, (list, Sequence, np.array)):
            for name in series:
                idx = np.where(np.array(self.name) == name)[0][0]
                data['timestamp'] += list(self.time)
                data['value']  += list(self.values[:, idx - 1])
                data['series'] += [name] * len(self.time)
        else:
            raise ValueError(f'Series argument must be "all" or list-like. {typeof(series)} were given.')

        data = pd.DataFrame(data)
        try:
            data['timestamp'] = data['timestamp'].astype('datetime64')
        except:
            pass
        sns.lineplot(x='timestamp', y='value', hue='series', data=data)

        if title is not None:
            plt.title(title)
        if labels[0] is not None:
            plt.xlabel(labels[0])
        if labels[1] is not None:
            plt.ylabel(labels[1])

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def to_csv(self, path):
        self.as_dataframe.to_csv(path, index=False)

    @property
    def as_dataframe(self):
        df = {}
        for i, n in enumerate(self.name):
            if n == 'timestamp':
                df[n] = self.time
            else:
                df[n] = self.values[:, i - 1]
        return pd.DataFrame(df)

    # ==========================================
    # STATIC METHODS
    # ==========================================
    @staticmethod
    def read_csv(path : str, header=0, column_subset=None, parse_dates=False):
        if column_subset == None:
            df = pd.read_csv(path, header=header, parse_dates=parse_dates)
            return MultiTimeSeries(df)
        elif isinstance(column_subset, Sequence):
            df = pd.read_csv(path, header=header, parse_dates=parse_dates)
            return MultiTimeSeries(df.iloc[:, column_subset])
        else:
            raise ValueError('Bad arguments were given.')

