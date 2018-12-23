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


class TrackStatMomProj:

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

        self.benchmark_ticker_list = ['SPY', 'QQQ', 'IWM']

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
        small_df = symbol_universe_df.tail(1)
        print (small_df)
        return_val_from_vectorized = small_df.apply(self.vectorized_symbols_func, axis=1)
        return return_val_from_vectorized

    @staticmethod
    def outlier_analysis(px_rets):

        no_outliers_non_norm = \
            StatisticalMoments.remove_outliers_from_non_normal_dist(px_rets)
        p_iqr_hist, p_iqr_cdf = ExtendBokeh.bokeh_histogram_overlay_normal(data=no_outliers_non_norm,
                                                                           titles=['IQR OLR - Px Rets Histogram',
                                                                                   'IQR OLR - Px Rets CDF'])
        no_outliers_assume_norm = \
            StatisticalMoments.remove_outliers_from_normal_dist(no_outliers_non_norm)
        p_iqr_3sr_hist, p_iqr_3sr_cdf = ExtendBokeh.bokeh_histogram_overlay_normal(data=no_outliers_assume_norm,
                                                                                   titles=['IQR & Sigma Rule OLR - Px Rets Histogram',
                                                                                           'IQR & Sigma Rule OLR - Px Rets CDF'])
        return p_iqr_hist, p_iqr_cdf, p_iqr_3sr_hist, p_iqr_3sr_cdf

    @staticmethod
    def calc_spread_px(ticker,
                       px,
                       benchmarks,
                       spread_ratios_df=None):

        """ This function needs some work. Right now all spread prices wille evaluate to 0.
            Ideally, we should be providing the spread ratio in spread_ratios_df (vs. each benchmark)
            TODO!!!
        """

        px = px.squeeze()
        px.name = ticker + '_' + px.name
        series_list = [px]
        for benchmark_ticker in benchmarks.keys():
            benchmark_px = benchmarks[benchmark_ticker].squeeze()
            benchmark_px.name = benchmark_ticker + '_' + benchmark_px.name
            series_list.append(benchmark_px)
            spread_ratio = px.div(benchmark_px)
            spread_price = benchmark_px*spread_ratio - px
            spread_price.name = ticker + '-' + benchmark_ticker + '_sprd'
            series_list.append(spread_price)
            df_sp = pd.concat(series_list, axis=1)
        return df_sp

    def do_excess_rets_analysis(self,
                                ticker,
                                px,
                                benchmark_px,
                                price_freq,
                                return_period):

        pa = PerformanceAnalyzer()
        sm = self.sm
        window_sizes = self.window_size_dict[price_freq]
        excess_rets_list = []
        for key in sorted(set(benchmark_px.keys())):
            benchmark_data = benchmark_px[key]
            excess_rets = pa.get_excess_returns(stock_data=px,
                                                benchmark_data=benchmark_data)

            excess_rets = excess_rets[1:].squeeze()
            excess_rets.rename(ticker + '_' + key + '_excess_rets', inplace=True)
            print ("excess_rets", excess_rets.head(10))
            excess_rets_list.append(excess_rets)
            sem = lambda excess_rets: excess_rets.std() / np.sqrt(len(excess_rets))

            rolling_excess_rets = sm.get_rolling_excess_returns(ticker=ticker,
                                                                benchmark=key,
                                                                freq=price_freq,
                                                                px_type='adjClose',
                                                                ticker_data=px,
                                                                benchmark_data=benchmark_data,
                                                                window_size=return_period,
                                                                shift_rets_series=True)
            rolling_excess_rets.rename(ticker + '_' + key + '_rolling_excess_rets', inplace=True)
            excess_rets_list.append(rolling_excess_rets)
            for window_size in window_sizes:
                er_rolling_df = excess_rets.rolling(window=window_size).agg({"mean_exc_ret": np.mean,
                                                                             "std_exc_ret": np.std,
                                                                             "sem_exc_ret": sem,
                                                                             "skew_exc_ret": stats.skew,
                                                                             "kurtosis_exc_ret": stats.kurtosis})
                er_rolling_df.columns = map(lambda col_nm: ticker + '-' + key + '_' +
                                                           str(window_size) + price_freq + '_' + col_nm,
                                            er_rolling_df.columns)
                #er_rolling_df = er_rolling_df.fillna(method='bfill')
                excess_rets_list.append(er_rolling_df)
        df_excess = pd.concat(excess_rets_list, axis=1)[2:]
        excess_rets_type_list = list(filter(lambda col_nm: ('_excess_rets' in col_nm) is True, df_excess.columns.values))
        p_iqr_hist, p_iqr_cdf, p_iqr_3sr_hist, p_iqr_3sr_cdf = TrackStatMomProj.outlier_analysis(excess_rets)
        excess_log_rets = np.log(1 + excess_rets)
        return (df_excess, excess_rets_type_list, excess_rets, excess_log_rets,
                p_iqr_hist, p_iqr_cdf, p_iqr_3sr_hist, p_iqr_3sr_cdf)

    def do_px_rets_analysis(self,
                            ticker,
                            px,
                            price_freq,
                            return_period):
        sm = self.sm
        window_sizes = self.window_size_dict[price_freq]
        rolling_px_rets = sm.get_rolling_returns(ticker=ticker,
                                                 freq=price_freq,
                                                 px_type='adjClose',
                                                 data=px,
                                                 window_size=return_period,
                                                 shift_rets_series=True)
        px_rets = px.pct_change()[1:].squeeze()
        px_rets.rename(px_rets.name + '_px_rets', inplace=True)
        rolling_px_rets.rename(px_rets.name + '_rolling', inplace=True)
        sem = lambda px_ret: px_ret.std() / np.sqrt(len(px_ret))
        all_df_to_concat = [px, px_rets, rolling_px_rets]
        for window_size in window_sizes:
            rolling_df = px_rets.rolling(window=window_size).agg({"mean_ret": np.mean,
                                                                  "std_ret": np.std,
                                                                  "sem_ret": sem,
                                                                  "skew_ret": stats.skew,
                                                                  "kurtosis_ret": stats.kurtosis})
            rolling_df.columns = map(lambda col_nm: str(window_size) + price_freq + '_' + col_nm, rolling_df.columns)
            #rolling_df = rolling_df.fillna(method='bfill')
            all_df_to_concat.append(rolling_df)
        df = pd.concat(all_df_to_concat, axis=1)[1:]
        px_ret_type_list = list(filter(lambda col_nm: ('_px_rets' in col_nm) is True, df.columns.values))
        p_iqr_hist, p_iqr_cdf, p_iqr_3sr_hist, p_iqr_3sr_cdf = TrackStatMomProj.outlier_analysis(px_rets)
        px_log_rets = np.log(1 + px_rets)

        return (df, px_ret_type_list, px_rets, px_log_rets, p_iqr_hist, p_iqr_cdf, p_iqr_3sr_hist, p_iqr_3sr_cdf )

    def vectorized_symbols_func(self,
                                row):

        # some parameters
        save_or_show = 'show'
        price_freq = 'D'
        win_price_freq = 60
        return_period = 45
        skew_filter=(-2.0, 2.0)
        benchmark_ticker_list = self.benchmark_ticker_list

        ticker = row.name
        co_nm = row['name']
        self.logger.info('TrackStatMomProj.vectorized_symbols_func(): pulling daily px returns for %s', ticker)
        sm = self.sm
        px = sm.get_pricing(ticker=ticker,
                            fields=['adjClose'],
                            freq=price_freq)[ticker]

        benchmark_px = sm.get_pricing(ticker=benchmark_ticker_list,
                                      fields=['adjClose'],
                                      freq=price_freq)

        df_excess, excess_rets_type_list, excess_rets, excess_log_rets, \
        p_iqr_hist_er, p_iqr_cdf_er, p_iqr_3sr_hist_er, p_iqr_3sr_cdf_er = \
            self.do_excess_rets_analysis(ticker,
                                         px,
                                         benchmark_px,
                                         price_freq,
                                         return_period)

        df_px, px_ret_type_list, px_rets, px_log_rets, p_iqr_hist, p_iqr_cdf, p_iqr_3sr_hist, p_iqr_3sr_cdf = \
            self.do_px_rets_analysis(ticker,
                                     px,
                                     price_freq,
                                     return_period)

        hist_plots = []
        excess_rets_hist_plots = []
        for benchmark_ticker in benchmark_ticker_list:
            spans_tuples_list_er = ExtendBokeh.bokeh_create_mean_var_spans(df_excess,
                                                                           ticker=ticker,
                                                                           benchmark_ticker=benchmark_ticker,
                                                                           freq=price_freq,
                                                                           rolling_window_size=win_price_freq,
                                                                           var_bandwidth=3.0,
                                                                           color=('red', 'green'))
            px_line_plot_er_band_breach_spans = \
                ExtendBokeh.bokeh_px_line_plot(data=df_px,
                                               ticker=ticker,
                                               benchmark_px_series=benchmark_px[benchmark_ticker],
                                               benchmark_ticker=benchmark_ticker,
                                               title=[co_nm + ' Px Chart Band Breaches'],
                                               subtitle=["Spread: " + ticker + '-' + benchmark_ticker],
                                               type_list=['adjClose', 'adjClose'],
                                               spans_list=spans_tuples_list_er,
                                               which_axis_list=[0,1])

            excess_rets_line_plot = \
                ExtendBokeh.bokeh_px_returns_plot(data=df_excess,
                                                  freq=price_freq,
                                                  title=[co_nm + ' Excess Returns'],
                                                  subtitle=["Spread: " + ticker + '-' + benchmark_ticker],
                                                  type_list=excess_rets_type_list,
                                                  scatter=False,
                                                  rolling_window_size=win_price_freq)
            ticker_bm_excess_ret = list(filter(lambda col_nm:
                                               (benchmark_ticker + '_excess_rets' in col_nm) is True,
                                               excess_rets_type_list))
            ticker_bm_rolling_excess_ret = list(filter(lambda col_nm:
                                                       (benchmark_ticker + '_rolling_excess_rets' in col_nm) is True,
                                                       excess_rets_type_list))
            print("ticker_bm_excess_ret", ticker_bm_excess_ret)
            print("ticker_bm_rolling_excess_ret", ticker_bm_rolling_excess_ret)

            excess_rets_scatter_plot = \
                ExtendBokeh.bokeh_px_returns_plot(data=df_excess,
                                                  freq=price_freq,
                                                  title=[co_nm + ' Excess Returns'],
                                                  subtitle=['Spread: ' + ticker + '-' + benchmark_ticker],
                                                  type_list=ticker_bm_excess_ret,
                                                  scatter=True,
                                                  rolling_window_size=win_price_freq)

            excess_rolling_rets_scatter_plot = \
                ExtendBokeh.bokeh_px_returns_plot(data=df_excess,
                                                  freq=price_freq,
                                                  title=[co_nm + ' ' + str(win_price_freq) + price_freq +
                                                         ' Rolling Excess Returns'],
                                                  subtitle=['Spread: ' + ticker + '-' + benchmark_ticker],
                                                  type_list=ticker_bm_rolling_excess_ret,
                                                  scatter=True,
                                                  rolling_window_size=win_price_freq)

            excess_rets_hist_plots.append(excess_rets_line_plot)
            excess_rets_hist_plots.append(excess_rets_scatter_plot)
            excess_rets_hist_plots.append(excess_rolling_rets_scatter_plot)
            excess_rets_hist_plots.append(px_line_plot_er_band_breach_spans)
            col_str = ticker + '_' + benchmark_ticker + '_excess_rets'
            print ("col_str", col_str)
            excess_rets_normal_ovl, excess_rets_normal_cdf = \
                ExtendBokeh.bokeh_histogram_overlay_normal(df_excess[col_str],
                                                           ["Excess Returns Hist",
                                                           "Excess Returns CDF"])
            excess_rets_hist_plots.append(excess_rets_normal_ovl)
            excess_rets_hist_plots.append(excess_rets_normal_cdf)
        #px_rets_lognormal_ovl, px_rets_lognormal_cdf = \
        #    ExtendBokeh.bokeh_histogram_overlay_normal(px_log_rets,
        #                                               titles=['Px Log Returns Histogram',
        #                                                       'Px Log Returns CDF'])
        #hist_plots.append(px_rets_normal_ovl)
        #hist_plots.append(px_rets_normal_cdf)
        #hist_plots.append(p_iqr_hist)
        #hist_plots.append(p_iqr_cdf)
        #hist_plots.append(p_iqr_3sr_hist)
        #hist_plots.append(p_iqr_3sr_cdf)
        #hist_plots.append(px_rets_lognormal_ovl)
        #hist_plots.append(px_rets_lognormal_cdf)

        #### below code puts out the plots for <ticker>.hist.html ####
        ##############################################################
        spans_tuples_list = ExtendBokeh.bokeh_create_mean_var_spans(df_px,
                                                                    ticker=ticker,
                                                                    freq=price_freq,
                                                                    rolling_window_size=win_price_freq,
                                                                    var_bandwidth=3.0,
                                                                    color = ('red','green'))
        px_line_plot = ExtendBokeh.bokeh_px_line_plot(data=df_px,
                                                      ticker=ticker,
                                                      title=[co_nm + ' Px Chart'],
                                                      subtitle=["Exchange Ticker: " + ticker])
        px_line_plot_band_breach_spans = ExtendBokeh.bokeh_px_line_plot(data=df_px,
                                                                        ticker=ticker,
                                                                        title=[co_nm + ' Px Chart Band Breaches'],
                                                                        subtitle=["Exchange Ticker: " + ticker],
                                                                        spans_list=spans_tuples_list)
        px_rets_line_plot = ExtendBokeh.bokeh_px_returns_plot(data=df_px,
                                                              freq=price_freq,
                                                              title=[co_nm + ' Px Returns'],
                                                              subtitle=["Exchange Ticker: " + ticker],
                                                              type_list=px_ret_type_list,
                                                              scatter=False,
                                                              rolling_window_size=win_price_freq)
        px_rets_scatter_plot = ExtendBokeh.bokeh_px_returns_plot(data=df_px,
                                                                 freq=price_freq,
                                                                 title=[co_nm + ' Px Returns'],
                                                                 subtitle=['Exchange Ticker: ' + ticker],
                                                                 type_list=px_ret_type_list,
                                                                 scatter=True,
                                                                 rolling_window_size=win_price_freq)
        hist_plots.append(px_line_plot)
        hist_plots.append(px_rets_line_plot)
        hist_plots.append(px_rets_scatter_plot)
        hist_plots.append(px_line_plot_band_breach_spans)
        px_rets_normal_ovl, px_rets_normal_cdf = \
            ExtendBokeh.bokeh_histogram_overlay_normal(px_rets)
        px_rets_lognormal_ovl, px_rets_lognormal_cdf = \
            ExtendBokeh.bokeh_histogram_overlay_normal(px_log_rets,
                                                       titles=['Px Log Returns Histogram',
                                                               'Px Log Returns CDF'])
        hist_plots.append(px_rets_normal_ovl)
        hist_plots.append(px_rets_normal_cdf)
        hist_plots.append(p_iqr_hist)
        hist_plots.append(p_iqr_cdf)
        hist_plots.append(p_iqr_3sr_hist)
        hist_plots.append(p_iqr_3sr_cdf)
        hist_plots.append(px_rets_lognormal_ovl)
        hist_plots.append(px_rets_lognormal_cdf)
        # px_rets_gamma_ovl = ext_bokeh.bokeh_histogram_overlay_gammma(px_rets)
        # hist_plots.append(px_rets_gamma_ovl)
        # overlay rolling skew and rolling returns

        #### Below code puts out the plots for <ticker>.skew.html ####
        ##############################################################
        p_rolling_pxret_skew_line, p_rolling_pxret_skew_scatter = \
            ExtendBokeh.bokeh_rolling_pxret_skew(df_px,
                                                 freq=price_freq,
                                                 title=['Rolling Returns vs. Rolling Skew'],
                                                 subtitle=[str(return_period) + price_freq + '/' +
                                                           str(win_price_freq) + price_freq + ' Window'],
                                                 type_list=['adjClose_px_rets_rolling', str(win_price_freq)+price_freq+'_skew_ret'],
                                                 rolling_window_size=win_price_freq,
                                                 skew_filter=skew_filter)
        p_pxret_rolling_skew_line, p_pxret_rolling_skew_scatter = \
            ExtendBokeh.bokeh_rolling_pxret_skew(df_px,
                                                 freq=price_freq,
                                                 title=['Returns vs. Rolling Skew'],
                                                 subtitle=[str(1) + price_freq + '/' +
                                                           str(win_price_freq) + price_freq + ' Window'],
                                                 type_list=['adjClose_px_rets', str(win_price_freq) + price_freq + '_skew_ret'],
                                                 rolling_window_size=win_price_freq,
                                                 skew_filter=skew_filter)

        #### Below code puts out the plots for <ticker>.Excess.Returns.html ####
        ########################################################################


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

        html_output_file_title_skew_analysis = ticker + '.skew.html'
        html_output_file_skew_analysis = html_output_file_path + html_output_file_title_skew_analysis

        html_output_file_title_excess_rets_analysis = ticker + '.excess.rets.html'
        html_output_file_excess_rets_analysis = html_output_file_path + html_output_file_title_excess_rets_analysis

        if (save_or_show is 'show'):
            ExtendBokeh.show_hist_plots(hist_plots,
                                        html_output_file,
                                        html_output_file_title)
            ExtendBokeh.show_hist_plots([p_rolling_pxret_skew_line, p_rolling_pxret_skew_scatter, p_pxret_rolling_skew_line, p_pxret_rolling_skew_scatter],
                                        html_output_file_skew_analysis,
                                        html_output_file_title_skew_analysis)
            ExtendBokeh.show_hist_plots(excess_rets_hist_plots,
                                        html_output_file_excess_rets_analysis,
                                        html_output_file_title_excess_rets_analysis)
        else:
            ExtendBokeh.save_html(hist_plots,
                                  html_output_file,
                                  html_output_file_title)
            ExtendBokeh.save_html([p_rolling_pxret_skew_line, p_rolling_pxret_skew_scatter],
                                  html_output_file_skew_analysis,
                                  html_output_file_title_skew_analysis)
            ExtendBokeh.save_html(excess_rets_hist_plots,
                                  html_output_file_excess_rets_analysis,
                                  html_output_file_title_excess_rets_analysis)

        # below is pyplot functionality - not using pyplot as it requires subscription
        # hist_data = [go.Histogram(y=px_rets)]
        # url = py.plot(hist_data, filename=self.plotly_histograms_dir + ticker + '_hist')
        # self.logger.info('TrackStatMomProj.vectorized_symbols_func(): histogram url for ticker %s is %s',
        #                 ticker, str(url))

        skew = sm.calc_stock_return_skew(ticker=ticker, data = px_rets)
        kurt = sm.calc_stock_return_kurtosis(ticker=ticker, data = px_rets)


if __name__ == '__main__':

    tsmp = TrackStatMomProj(use_iex_trading_symbol_universe=True)
    stock_universe_df, ticker_col_nm = tsmp.get_stock_universe()
    tsmp.get_px_df(stock_universe_df)
