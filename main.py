from core.data_types.time_series import TimeSeries
from core.data_types.time_series import MultiTimeSeries
from core.analysis.time_series import *

from core.provider.online_sources import AlphaVantageConnector
from core.provider.online_sources import dump_alphavantage

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy

from core.models.cluster import TimeSeriesKMeans

from dtw import *


'''
Notatki bieżące:
    1. Filip W. twierdzi, że algorytm ROCKET ma bardzo dobrą skuteczność.
    2. Rozciąganie zbiorów rozmytych - a la Dynamic Time Wrapping.
'''

if __name__ == '__main__':
    apple = TimeSeries.read_csv('data/djia_composite/AAPL.csv', name='Apple')
    cat = TimeSeries.read_csv('data/djia_composite/CAT.csv', name='Caterpillar')

    print(dtw(cat.values, apple.values).normalizedDistance)
















