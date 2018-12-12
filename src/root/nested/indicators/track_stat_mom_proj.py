from root.nested.dataAccess.data_provider_interface import DataProviderInterface
from root.nested.dataAccess.WebApi.iex_trading_api import IEXTradingApi
from root.nested import get_logger
from root.nested.dataAccess.tiingo_data_object import TiingoDataObject
from root.nested.statisticalAnalysis.statistical_moments import StatisticalMoments
from root.nested.visualize.extend_bokeh import ExtendBokeh
from root.nested.SysOs.os_mux import OSMuxImpl
from root.nested.statisticalAnalysis.stats_tests import StatsTests

import pandas as pd
import numpy as np
# import plotly.plotly as py
# import plotly.graph_objs as go


class TrackStatMomProj:

    def __init__(self,
                 stock_universe_filename='Russ3K_holdings',
                 use_iex_trading_symbol_universe=False,
                 sec_type_list = ['cs', 'et']):

        self.logger = get_logger()
        self.stock_universe_data = None
        self.stock_universe_filename = stock_universe_filename
        self.use_iex_trading_symbol_universe = use_iex_trading_symbol_universe
        self.sec_type_list = sec_type_list
        self.plotly_histograms_dir = OSMuxImpl.get_proper_path('/workspace/data/plotly/histograms/')

    """ get_pricing: main function to retrieve daily price data
            The source of this data is currently Tiingo. 
        """
    def get_pricing(self,
                    ticker,
                    start_date,
                    end_date,
                    fields=['adjOpen', 'adjHigh', 'adjLow', 'adjClose']):

        symbols = [ticker]
        source = 'Tiingo'
        mdo = TiingoDataObject(start_date=start_date,
                               end_date=end_date,
                               source=source,
                               symbols=symbols)
        pricing_dict = mdo.get_px_data_df(start_date,
                                          end_date)
        pricing_df = pricing_dict[ticker]
        pricing = pricing_df[fields]
        return pricing

    def get_stock_returns(self,
                          ticker,
                          start_date='2010-01-01',
                          end_date=str(pd.to_datetime('today')).split(' ')[0],
                          px_type='adjClose'):

        symbols = [ticker]
        source = 'Tiingo'
        mdo = TiingoDataObject(start_date = start_date,
                               end_date = end_date,
                               source = source,
                               symbols = symbols)
        pricing_dict = mdo.get_px_data_df(start_date,
                                          end_date) # returned df will not include all dates up to, but not including, end_date
        pricing_df = pricing_dict[ticker] # dataframe containing the price data
        pricing = pricing_df[px_type]
        returns = pricing.pct_change()[1:]
        return returns

    def get_stock_universe(self):

        ticker_col_nm = 'Ticker' # default for Russell files
        if self.use_iex_trading_symbol_universe is False:
            dpi = DataProviderInterface()
            pd_df = dpi.get_stock_universe_file_as_df(self.stock_universe_filename)
            self.logger.info('TrackStatMomProj.get_stock_universe(): pulled stock universe from %s, dataframe columns'
                             'are %s', self.stock_universe_filename, str(pd_df.columns))
        else:
            ticker_col_nm = 'symbol'
            iex_trading_api = IEXTradingApi(sec_type_list=self.sec_type_list)
            pd_df = iex_trading_api.get_symbols_universe()
            pd_df = pd_df.set_index('symbol')
            self.logger.info('TrackStatMomProj.get_stock_universe(): pulled stock universe from IEX Trading API,'
                             'dataframe columns are %s', str(pd_df.columns))
            # with IEX the ticker column is named symbol, whereas column is named Ticker in IShares universe.
        return pd_df, ticker_col_nm

    def get_px_df(self,
                  symbol_universe_df):

        ### for testing, just do the first row -- REMEMBER to UNDO THIS!!! GM 12/9/2018
        small_df = symbol_universe_df.head(1)
        return_val_from_vectorized = small_df.apply(self.vectorized_symbols_func, axis=1)
        return return_val_from_vectorized

    def vectorized_symbols_func(self,
                                row):

        ticker = row.name
        self.logger.info('TrackStatMomProj.vectorized_symbols_func(): pulling daily px returns for %s', ticker)
        sm = StatisticalMoments()
        ext_bokeh = ExtendBokeh()
        px = sm.get_pricing(ticker=ticker, fields=['adjClose'])
        px_rets = sm.get_stock_returns(ticker=ticker)
        px_log_rets = np.log(1+px_rets)
        hist_plots = []
        column_plots = []
        px_line_plot = ext_bokeh.bokeh_px_line_plot(data=px.squeeze())
        px_rets_plot = ext_bokeh.bokeh_px_line_plot(data=px_rets)
        # column_plots.append(px_line_plot)
        # column_plots.append(px_rets_plot)
        hist_plots.append(px_line_plot)
        hist_plots.append(px_rets_plot)
        px_rets_normal_ovl, px_rets_normal_cdf = \
            ext_bokeh.bokeh_histogram_overlay_normal(px_rets)
        hist_plots.append(px_rets_normal_ovl)
        hist_plots.append(px_rets_normal_cdf)
        px_rets_lognormal_ovl, px_rets_lognormal_cdf = \
            ext_bokeh.bokeh_histogram_overlay_normal(px_log_rets,
                                                     titles=['Px Log Returns Histogram',
                                                             'Px Log Returns CDF'])
        hist_plots.append(px_rets_lognormal_ovl)
        hist_plots.append(px_rets_lognormal_cdf)
        # px_rets_gamma_ovl = ext_bokeh.bokeh_histogram_overlay_gammma(px_rets)
        # hist_plots.append(px_rets_gamma_ovl)

        # next, lets do the KS Test, to test for normality. We will do JB Test later.
        ks_test_stat_raw_rets, p_value_raw_rets = StatsTests.ks_test(rvs = px_rets, dist_size=len(px_rets), cdf='norm')
        ks_test_stat_log_rets, p_value_log_rets = StatsTests.ks_test(rvs = px_log_rets, dist_size=len(px_log_rets), cdf='norm')
        self.logger.info("TrackstatMomProj.vectorized_symbols_func(): KS Test Stat (Raw Returns) is %s and p_value is %s",
                         str(ks_test_stat_raw_rets), str(p_value_raw_rets))
        self.logger.info("TrackstatMomProj.vectorized_symbols_func(): Is KS Test Stat (Raw Returns) %s as close to 0 as possible"
                         " and p_value %s as close to 1 as possible? If p_value less than 0.05, Null Hypothesis (the"
                         "two distributions RVS and CDF are equal) is rejected!", str(ks_test_stat_raw_rets), str(p_value_raw_rets))
        self.logger.info("TrackstatMomProj.vectorized_symbols_func(): KS Test Stat (Log Returns) is %s and p_value is %s",
                         str(ks_test_stat_log_rets), str(p_value_log_rets))
        self.logger.info("TrackstatMomProj.vectorized_symbols_func(): Is KS Test Stat (Log Returns) %s as close to 0 as possible"
                         " and p_value %s as close to 1 as possible? If p_value less than 0.05, the Null Hypothesis (the"
                         "two distributions RVS and CDF are equal) is rejected!", str(ks_test_stat_log_rets), str(p_value_log_rets))

        # px_rets_weibull_ovl = ext_bokeh.bokeh_histogram_overlay_weibull(px_rets)
        # hist_plots.append(px_rets_weibull_ovl)
        html_output_file_title = ticker + '.hist.html'
        html_output_file_path = OSMuxImpl.get_proper_path('/workspace/data/bokeh/html/')
        html_output_file = html_output_file_path + html_output_file_title
        ext_bokeh.show_hist_plots(hist_plots, column_plots, html_output_file, html_output_file_title)

        # below is pyplot functionality - not using pyplot as it requires subscription
        # hist_data = [go.Histogram(y=px_rets)]
        # url = py.plot(hist_data, filename=self.plotly_histograms_dir + ticker + '_hist')
        # self.logger.info('TrackStatMomProj.vectorized_symbols_func(): histogram url for ticker %s is %s',
        #                 ticker, str(url))

        skew = sm.calc_stock_return_skew(ticker=ticker)
        kurt = sm.calc_stock_return_kurtosis(ticker=ticker)


if __name__ == '__main__':

    tsmp = TrackStatMomProj(use_iex_trading_symbol_universe=True)
    df, ticker_col_nm = tsmp.get_stock_universe()
    tsmp.get_px_df(df)
