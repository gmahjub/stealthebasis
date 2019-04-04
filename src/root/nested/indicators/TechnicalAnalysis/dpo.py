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
import math
import numpy as np
import scipy.stats as stats
# import plotly.plotly as py
# import plotly.graph_objs as go


class DetrendedPxOsc:

    def __init__(self,
                 stock_universe_filename='Russ3K_holdings',
                 use_iex_trading_symbol_universe=False,
                 sec_type_list = ['cs', 'et'],
                 daily_window_sizes = [30, 60, 90, 120, 180, 270],
                 weekly_window_sizes = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52],
                 monthly_window_sizes = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]):

        self.logger = get_logger()
        self.sm = StatisticalMoments()
        self.stock_universe_data = None
        self.stock_universe_filename = stock_universe_filename
        self.use_iex_trading_symbol_universe = use_iex_trading_symbol_universe
        self.sec_type_list = sec_type_list
        self.plotly_histograms_dir = OSMuxImpl.get_proper_path('/workspace/data/plotly/histograms/')
        self.daily_window_sizes = daily_window_sizes
        self.weekly_window_sizes = weekly_window_sizes

        self.window_size_dict = {'D': daily_window_sizes,
                                 'W': weekly_window_sizes,
                                 'M': monthly_window_sizes}

        self.index_ticker_dict = {'SPY': 'SP500'}#, 'QQQ': 'NQ100', 'IWM': 'Russell2000', 'DIA': 'Dow30'}
        self.index_ticker_df = pd.DataFrame(data=list(self.index_ticker_dict.items()), columns=['Symbol', 'Name'])

    def get_px_df(self,
                  symbol_universe_df):

        ### for testing, just do the first row -- REMEMBER to UNDO THIS!!! GM 12/9/2018
        #small_df = symbol_universe_df.head(1)
        #print (small_df)
        small_df = symbol_universe_df
        return_val_from_vectorized = small_df.apply(self.doWork, axis=1)
        return return_val_from_vectorized

    def doWork(self,
               row):

        # some parameters
        save_or_show = 'show'
        price_freq = 'W'
        ema_window = 9
        detrend_lag = math.floor(ema_window/2 + 1)
        ema_alpha = 2.0/(ema_window + 1)
        ticker = row.Symbol
        self.logger.info('DetrendedPxOsc.doWork(): pulling daily px returns for %s', ticker)
        sm = self.sm
        px_df = sm.get_pricing(ticker=ticker,
                               fields=['close','adjClose'],
                               freq=price_freq)[ticker]
        px_df['px_lagged'] = px_df.adjClose.shift(periods=detrend_lag)
        px_df['adjClose_sma'] = px_df.adjClose.rolling(window=ema_window).mean()
        px_df['adjClose_ema'] = px_df.adjClose.ewm(alpha=ema_alpha).mean()
        px_df['dpo_use_sma'] = px_df.px_lagged - px_df.adjClose_sma
        px_df['dpo_use_ema'] = px_df.px_lagged - px_df.adjClose_ema

        conditions = [
            (pd.to_numeric(px_df.dpo_use_ema.mul(1000000.0), downcast='integer') > int(min_input_vol * 1000000.0)),
            # one tick = 5000
            (pd.to_numeric(px_df.OpenSettleDelta.mul(1000000.0), downcast='integer') < int(-min_input_vol * 1000000.0))
        ]  # one tick = 5000
        ## the below, 1.0 or -1.0 multiples, tells us whether we expect reversion in next period,
        ## or autocorrelation. (-1.0,1.0) = reversion, (1.0, -1.0) = autocorrelation
        choices_settle_last = [px_df.SettleLastDelta.mul(1.0), px_df.SettleLastDelta.mul(-1.0)]
        choices_settle_nextopen = [px_df.SettleNextOpenDelta.mul(1.0), px_df.SettleNextOpenDelta.mul(-1.0)]
        choices_last_nextopen = [px_df.LastNextOpenDelta.mul(1.0), px_df.LastNextOpenDelta.mul(-1.0)]
        px_df['SettleLastTradeSelect'] = np.select(conditions, choices_settle_last, default=0.0)
        px_df['SettleNextOpenTradeSelect'] = np.select(conditions, choices_settle_nextopen, default=0.0)
        px_df['LastNextOpenTradeSelect'] = np.select(ol_delta_conditions, choices_last_nextopen, default=0.0)

        return

if __name__ == '__main__':

    dpo_obj = DetrendedPxOsc(use_iex_trading_symbol_universe=False)
    dpo_obj.get_px_df(dpo_obj.index_ticker_df)
