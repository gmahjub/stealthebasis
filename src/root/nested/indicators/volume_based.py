from root.nested.dataAccess.data_provider_interface import DataProviderInterface
from root.nested.dataAccess.WebApi.iex_trading_api import IEXTradingApi
from root.nested import get_logger
from root.nested.dataAccess.tiingo_data_object import TiingoDataObject
from root.nested.statisticalAnalysis.statistical_moments import StatisticalMoments
from root.nested.visualize.extend_bokeh import ExtendBokeh
from root.nested.SysOs.os_mux import OSMuxImpl
from root.nested.statisticalAnalysis.stats_tests import StatsTests
from root.nested.performance_analyzer import PerformanceAnalyzer

import pandas as pd
import numpy as np
import scipy.stats as stats

# import plotly.plotly as py
# import plotly.graph_objs as go
LOGGER = get_logger()


class VolumeBased:

    def __init__(self):

        self.short_sma = 30
        self.long_sma = 60
        self.sma_periodicity = 'D'
        self.pearson_corr_window = 15
        self.spearman_corr_window = 15
        self.sma_spread_window = 1

        self.short_ema = 12
        self.long_ema = 26
        self.ema_spread_window = 9
        self.ema_periodicity = 'D'
