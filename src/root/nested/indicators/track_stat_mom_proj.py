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


class YieldCurveRiskPricing:

    def __init__self(self,
                     yield_curve_obj):
        self.yield_curve_obj = yield_curve_obj


class TrackStatMomProj:

    def __init__(self,
                 stock_universe_filename='Russ3K_holdings',
                 use_iex_trading_symbol_universe=False,
                 sec_type_list=['cs', 'et'],
                 daily_window_sizes=[30, 60, 90, 120, 180, 270],
                 weekly_window_sizes=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52],
                 monthly_window_sizes=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]):

        self.logger = get_logger()
        self.sm = StatisticalMoments()
        self.stock_universe_data = None
        self.stock_universe_filename = stock_universe_filename
        self.use_iex_trading_symbol_universe = use_iex_trading_symbol_universe
        self.sec_type_list = sec_type_list
        self.plotly_histograms_dir = OSMuxImpl.get_proper_path('/workspace/data/plotly/histograms/')
        self.skew_bin_csv_files_dir = OSMuxImpl.get_proper_path('/workspace/data/indicators/TrackStatMom/data/')
        self.bokeh_html_file_dir = OSMuxImpl.get_proper_path('/workspace/data/bokeh/html')
        self.daily_window_sizes = daily_window_sizes
        self.weekly_window_sizes = weekly_window_sizes
        self.window_size_dict = {'D': daily_window_sizes,
                                 'W': weekly_window_sizes,
                                 'M': monthly_window_sizes}
        self.benchmark_ticker_list = ['SPY', 'QQQ', 'IWM', 'DIA']
        skew_bins = np.arange(-2.0, 2.5, 1.0)
        prev_skew_bin = -1000.0
        self.bin_range_tuple_list = []
        for skew_bin in skew_bins:
            self.bin_range_tuple_list.append((prev_skew_bin, skew_bin))
            prev_skew_bin = skew_bin
        self.bin_range_tuple_list.append((skew_bins[-1], 1000.0))
        self.bin_df_dict = dict()
        for skew_bin_range in self.bin_range_tuple_list:
            self.bin_df_dict[skew_bin_range] = \
                pd.DataFrame(columns=['Date', 'Ticker', 'Returns', 'RollingReturns', 'ExcessFlag', 'SkewType',
                                      'SkewValue'])

    def get_benchmark_ticker_list(self):
        return self.benchmark_ticker_list

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
        """ USE THIS FUNCTION TO RETRIEVE MULTIPLE SYMBOLS OF PRICE DATA
        px_type is required here, and only one px_type can be given.
        ticker is misleading. It is actually a numpy array of tickers.
        THIS IS THE FUNCTION TO USE FOR ALL PX DATA RETRIEVAL!!!! GM - 7/16/2019"""
        symbols = [ticker]
        symbols = np.asarray(symbols)[0]
        source = 'Tiingo'
        mdo = TiingoDataObject(start_date=start_date,
                               end_date=end_date,
                               source=source,
                               symbols=symbols)
        pricing_dict = mdo.get_px_data_df(start_date,
                                          end_date)
        # returned df will not include all dates up to, but not including, end_date
        series_dict = {}
        for symbol in symbols:
            pricing_df = pricing_dict[symbol]
            px_series = pricing_df[px_type]
            px_rets = px_series.pct_change()[1:]
            px_rets = px_rets.rename(symbol + '_' + px_type)
            series_dict[symbol + '_' + px_type] = px_rets
        return_df = pd.DataFrame(data=series_dict)

        return return_df

    def get_stock_universe(self):

        ticker_col_nm = 'Ticker'  # default for Russell files
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

    @staticmethod
    def autocorrelation_analysis(px_df,
                                 sample_period='W',
                                 how='last'):
        # stock prices tend to me postively autocorrelated in the long term ('monthly'),
        # yet negatively correlation in the short term ('weekly')
        # px should be a dataframe of prices, or a pd.Series of prices
        resampled_px_df = px_df.resample(sample_period, how=how)
        px_rets = resampled_px_df.pct_change()
        autocorrelation = px_rets[1:].autocorr()
        return autocorrelation

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
                                                                                   titles=[
                                                                                       'IQR & Sigma Rule OLR - Px Rets Histogram',
                                                                                       'IQR & Sigma Rule OLR - Px Rets CDF'])
        return p_iqr_hist, p_iqr_cdf, p_iqr_3sr_hist, p_iqr_3sr_cdf

    @staticmethod
    def calc_spread_px(ticker,
                       px,
                       benchmarks,
                       spread_ratios_df=None):

        """ This function needs some work. Right now all spread prices will evaluate to 0.
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
            spread_price = benchmark_px * spread_ratio - px
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
            ticker = "MSFT"
            stock_atr_df = sm.get_average_true_range(ticker=ticker, freq='W', window_size=12, start_date='2015-01-01')
            key = "SPY"
            benchmark_atr_df = sm.get_average_true_range(ticker=key, freq='W', window_size=26, start_date='2015-01-01')
            sm.get_atr_spread(stock_atr_df, benchmark_atr_df)
            exit()
            benchmark_data = benchmark_px[key]
            excess_log_rets, benchmark_log_rets, stock_log_rets = sm.get_stock_excess_return(stock_ticker=ticker,
                                                                                             benchmark_ticker=key,
                                                                                             freq='D',
                                                                                             start_date='2018-01-01')
            stock_benchmark_spread_px = sm.get_stock_benchmark_spread_px(stock_ticker=ticker,
                                                                         benchmark_ticker=key,
                                                                         hedge_ratio_calc_window=1,
                                                                         start_date='2015-01-01',
                                                                         freq='W')
            stock_log_rets = stock_log_rets[1:].squeeze()
            excess_log_rets = excess_log_rets[1:].squeeze()
            benchmark_log_rets = benchmark_log_rets[1:].squeeze()
            excess_log_rets.rename(ticker + '_' + key + '_excess_rets', inplace=True)
            benchmark_log_rets.rename(key + '_benchmark_rets', inplace=True)
            stock_log_rets.rename(ticker + '_stock_rets', inplace=True)
            excess_rets_list.append(excess_log_rets)
            rolling_excess_rets, benchmark_rolling_rets = sm.get_rolling_excess_returns(ticker=ticker,
                                                                                        benchmark=key,
                                                                                        freq=price_freq,
                                                                                        px_type='adjClose',
                                                                                        ticker_data=px,
                                                                                        benchmark_data=benchmark_data,
                                                                                        window_size=return_period,
                                                                                        shift_rets_series=True)

            rolling_excess_rets.rename(ticker + '_' + key + '_rolling_excess_rets', inplace=True)
            benchmark_rolling_rets.rename(key + '_benchmark_rolling_log_rets', inplace=True)
            excess_rets_list.append(rolling_excess_rets)
            for window_size in window_sizes:
                rolling_beta_series = pa.get_beta(stock_returns=stock_log_rets,
                                                  benchmark_returns=benchmark_log_rets,
                                                  window_size=window_size)
                rolling_beta_series.name = ticker + '_' + key + '_' + str(window_size) + '_' + \
                                           str(price_freq) + '_RollingBeta'
                """
                the_excess_returns_rolling_lpm_series = pa.get_lpm(returns=excess_log_rets,
                                                                   window_size=window_size)
                the_excess_returns_rolling_lpm_series.name = ticker + '_' + key + '_' + str(window_size) + '_' + \
                                                             str(price_freq) + '_ExcessReturnLPM'
                the_benchmark_rolling_lpm_series = pa.get_lpm(returns=benchmark_log_rets,
                                                              window_size=window_size)
                the_benchmark_rolling_lpm_series.name = key + '_' + str(window_size) + '_' + \
                                                        str(price_freq) + '_BenchmarkReturnLPM'
                the_excess_returns_rolling_hpm_series = pa.get_hpm(returns=excess_log_rets,
                                                                   window_size=window_size)
                the_excess_returns_rolling_hpm_series.name = ticker + '_' + key + '_' + str(window_size) + '_' + \
                                                             str(price_freq) + '_ExcessReturnHPM'
                the_benchmark_rolling_hpm_series = pa.get_hpm(returns=benchmark_log_rets,
                                                              window_size=window_size)
                the_benchmark_rolling_hpm_series.name = key + '_' + str(window_size) + '_' + \
                                                        str(price_freq) + '_BenchmarkReturnHPM'
                """
                er_rolling_df = excess_log_rets. \
                    rolling(window=window_size).agg({"mean_exc_ret": np.mean,
                                                     "std_exc_ret": np.std,
                                                     "sem_exc_ret": lambda x: x.std() / np.sqrt(len(x)),
                                                     "skew_exc_ret": stats.skew,
                                                     "kurtosis_exc_ret": stats.kurtosis,
                                                     "sharpe_exc_ret": lambda x: x.mean() / x.std() * np.sqrt(len(x))})
                er_rolling_df.columns = \
                    map(lambda col_nm: ticker + '-' + key + '_' + str(window_size) + price_freq + '_' + col_nm,
                        er_rolling_df.columns)
                excess_rets_list.append(er_rolling_df)
        df_excess = pd.concat(excess_rets_list, axis=1)[2:]
        excess_rets_type_list = list(
            filter(lambda col_nm: ('_excess_rets' in col_nm) is True, df_excess.columns.values))
        p_iqr_hist, p_iqr_cdf, p_iqr_3sr_hist, p_iqr_3sr_cdf = TrackStatMomProj.\
            outlier_analysis(excess_log_rets.dropna().squeeze())
        # df_excess has the following columns...
        # 1. [Stock]_[Benchmark]_excess_rets
        # 2. [Stock]_[Benchmark]_rolling_excess_rets
        # 3. [Stock]-[Benchmark]_[N]D_mean_exc_ret
        # 4. [Stock]-[Benchmark]_[N]D_std_exc_ret
        # 5. [Stock]-[Benchmark]_[N]D_sem_exc_ret
        # 6. [Stock]-[Benchmark]_[N]D_skew_exc_ret
        # 7. [Stock]-[Benchmark]_[N]D_kurtosis_exc_ret
        # ... where N is equal to 30, 60, 90, 120, 180, 270 Days (denoted by 'D')
        # ... where [Benchmark] is equal to 'QQQ', 'SPY', 'IWM', or any benchmarket index, product, 'GLD' or IEF
        # ... could be one
        return (df_excess, excess_rets_type_list, excess_log_rets, excess_log_rets,
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
                                                                  "ret_vol": lambda x: x.std() * np.sqrt(window_size),
                                                                  "sem_ret": sem,
                                                                  "skew_ret": stats.skew,
                                                                  "kurtosis_ret": stats.kurtosis})
            rolling_df.columns = map(lambda col_nm: str(window_size) + price_freq + '_' + col_nm, rolling_df.columns)
            # rolling_df = rolling_df.fillna(method='bfill')
            all_df_to_concat.append(rolling_df)
        df = pd.concat(all_df_to_concat, axis=1)[1:]
        px_ret_type_list = list(filter(lambda col_nm: ('_px_rets' in col_nm) is True, df.columns.values))
        p_iqr_hist, p_iqr_cdf, p_iqr_3sr_hist, p_iqr_3sr_cdf = TrackStatMomProj.outlier_analysis(px_rets)
        px_log_rets = np.log(1 + px_rets)
        return df, px_ret_type_list, px_rets, px_log_rets, p_iqr_hist, p_iqr_cdf, p_iqr_3sr_hist, p_iqr_3sr_cdf

    @staticmethod
    def monte_carlo_vs_bootstrapping_example():

        import datetime
        import pandas as pd
        import numpy as np
        from functools import reduce
        import pandas_datareader.data as web
        import random
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import seaborn as sns
        mpl.style.use('ggplot')
        figsize = (15, 8)

        start, end = datetime.datetime(2009, 12, 30), datetime.datetime(2019, 7, 24)
        tickers = ["^DJI", "^IXIC", "^GSPC", "^RUT"]
        asset_universe = pd.DataFrame([web.DataReader(ticker, 'yahoo', start, end).loc[:, 'Adj Close']
                                       for ticker in tickers], index=tickers).T.fillna(method='ffill')
        asset_universe = asset_universe / asset_universe.iloc[0, :]
        asset_universe.plot(figsize=figsize)

        # when we do mean along axis = 1, we are essentially doing an equally weighted portfolio,
        # simply averaging all the returns of the individual assets
        portfolio_returns = asset_universe.pct_change().dropna().mean(axis=1)
        portfolio = (asset_universe.pct_change().dropna().mean(axis=1) + 1).cumprod()
        asset_universe.plot(figsize=figsize, alpha=0.4)
        portfolio.plot(label="Portfolio", color='black')
        plt.legend()

        # bootstrapping process below
        # portfolio returns are individual return values. "random.choices" is from the random package
        # we choose a sample of size k (here is 252) rom the population (portfolio_returns)
        # basically we are choosing 1 year as our sample size.
        # we do that 1000 times (so we have 1000 samples of size 252 (1 year).
        # last step is calculate the cumulative product, and that is the line we will plot.
        portfolio_bootstrapping = (1 + pd.DataFrame([random.choices(list(portfolio_returns.values), k=252)
                                                     for i in range(1000)]).T.shift(1).fillna(0)).cumprod()
        portfolio_bootstrapping.plot(figsize=figsize, legend=False, linewidth=1, alpha=0.2, color='b')

        # the next process is also bootstrapping, but instead of creating the portfolio return,
        # it simply samples the individual returns first, and then creates the portfolio return
        # from the sampled data.
        # this should be (and is) the same as sampling the portfolio returns.
        asset_universe_returns = asset_universe.pct_change()
        portfolio_constituents_bootstrapping = pd.DataFrame([((asset_universe_returns.iloc[random.choices(
            range(len(asset_universe)), k=252)]).mean(axis=1) + 1).cumprod().values
                                                             for x in range(1000)]).T
        portfolio_constituents_bootstrapping.plot(figsize=figsize, legend=False, linewidth=1, alpha=0.2, color='purple')

        # Next we do the Monte Carlo method, which is a parametric process.
        # Here we will sample the portfolio returns, next we will sample the individual asset returns.
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()
        # the above are the two parameters are required by monte carlo if we are assuming a normal distribution
        # next we will create the distribution (normal) from the above parameters (mu, sigma)
        portfolio_mc = pd.DataFrame([(np.random.normal(loc=mu, scale=sigma, size=252) + 1)
                                     for x in range(1000)]).T.cumprod()
        portfolio_mc.plot(figsize=figsize, legend=False, linewidth=1, alpha=0.2, color='green')

        # Monte Carlo (sampling individual asset returns)
        asset_returns_dfs = []
        for asset in asset_universe_returns.mean().index:
            mu = asset_universe_returns.mean()[asset]
            sigma = asset_universe_returns.std()[asset]
            asset_mc_rets = pd.DataFrame([(np.random.normal(loc=mu,
                                                            scale=sigma,
                                                            size=252)) for x in range(1000)]).T
            asset_returns_dfs.append(asset_mc_rets)
            # so the above is the returns for each asset, taking the mean and standard deviation, as parameters
        # equal-weighted
        weighted_asset_returns_dfs = [(returns_df / len(tickers)) for returns_df in asset_returns_dfs]
        portfolio_constituents_mc = (reduce(lambda x, y: x + y, weighted_asset_returns_dfs) + 1).cumprod()
        portfolio_constituents_mc.plot(figsize=figsize, legend=False, linewidth=1, alpha=0.2, color='orange')
        plt.show()

    @staticmethod
    def plot_portfolio_correlation_matrix(px_returns_df):

        import seaborn as sns
        import matplotlib.pyplot as plt
        correlation_matrix = px_returns_df.corr(method='pearson')
        sns.heatmap(data=correlation_matrix,
                    annot=True,
                    cmap="YlGnBu",
                    linewidths=0.3,
                    annot_kws={"size": 8})

        # Plot aesthetics
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.show()

    @staticmethod
    def plot_portfolio_weighted_return(px_returns_df,
                                       portfolio_size,
                                       portfolio_weights=None,
                                       mkt_caps=False):

        """ This function takes a static portfolio weight for the entire duration of the return set.
        We would like to create a function, or modify this function, such that each day, or each week,
        or each month, we can use a different weighting based on some dynamic strategy. """

        import matplotlib.pyplot as plt
        if mkt_caps is True:
            # this means the portfolio weights are actual mkt caps of the companies, so we need to calculate
            portfolio_weights = mkt_caps / np.sum(mkt_caps)
            LOGGER.info("TrackStatMomProj.plot_portfolio_weighted_return(): using market cap weighted portfolio!")
        if portfolio_weights is None:
            # use equal weighted portfolio
            portfolio_weights = np.repeat(1 / portfolio_size, portfolio_size)
            LOGGER.info("TrackStatMomProj.plot_portfolio_weighted_return(): using equal weighted portfolio!")
        portfolio_weighted_returns_df = px_returns_df.mul(portfolio_weights, axis=1)
        portfolio_weighted_returns_df['Portfolio'] = portfolio_weighted_returns_df.sum(axis=1)
        portfolio_weighted_cumm_returns_df = ((1 + portfolio_weighted_returns_df["Portfolio"]).cumprod() - 1)
        portfolio_weighted_cumm_returns_df.plot()
        plt.show()

    @staticmethod
    def portfolio_covariance_matrix(px_returns_df,
                                    annualize=True):

        # px_returns_df is a dataframe of stock returns
        # each column is a stock returns time series
        # each row is a date
        cov_mat = px_returns_df.cov()
        if annualize is True:
            cov_mat *= 252
        return cov_mat

    @staticmethod
    def create_random_portfolios(portfolio_tickers,
                                 portfolio_size,
                                 use_dirichlet=False,
                                 how_many_portfolios=1):
        """ Two different methods for creating random weightings (that sum to 1) for testing of
        different portfolio compositions. Default is one portfolio, which should be changed. In
        reality, you would want to test potentially thousands of portfolio weightings. For more
        information on Dirichlet Distribution, see:
        https://stackoverflow.com/questions/18659858/generating-a-list-of-random-numbers-summing-to-1"""

        random_portfolios = []
        if not use_dirichlet:
            for port_num in range(how_many_portfolios):
                random_portfolio_weights = np.random.random(portfolio_size)
                random_portfolio_weights /= random_portfolio_weights.sum()
                random_portfolios.append(random_portfolio_weights)
            random_portfolios = np.array(random_portfolios)
        else:
            random_portfolios = np.random.dirichlet(np.ones(portfolio_size), size=how_many_portfolios)
        # random portfolios is of type numpy.ndarray. We would want to convert this to a pandas dataframe.
        random_portfolios_df = pd.DataFrame(random_portfolios)
        random_portfolios_df.columns = portfolio_tickers
        return random_portfolios_df

    @staticmethod
    def calculate_portfolio_sharpe_ratio(portfolio_weights_df,
                                         portfolio_returns_df):

        return 1

    @staticmethod
    def portfolio_standard_deviation(portfolio_weights,
                                     px_returns_df):

        cov_mat_annualized = TrackStatMomProj.portfolio_covariance_matrix(px_returns_df=px_returns_df)
        # portfolio weights are the weighting for each of the products in the portfolio
        portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_mat_annualized, portfolio_weights)))
        return portfolio_volatility

    @staticmethod
    def plot_efficient_frontier(simulation_results):

        import matplotlib.pyplot as plt
        # locate position of portfolio with highest Sharpe Ratio
        max_sharpe_port = simulation_results.iloc[simulation_results['Annualized Sharpe'].idxmax()]
        # locate positon of portfolio with minimum standard deviation
        min_vol_port = simulation_results.iloc[simulation_results['Annualized Vol'].idxmin()]
        # create scatter plot coloured by Sharpe Ratio
        plt.scatter(simulation_results['Annualized Vol'], simulation_results['Annualized Return'],
                    c=simulation_results['Annualized Sharpe'], cmap='RdYlBu')
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Returns')
        plt.colorbar()
        # plot red star to highlight position of portfolio with highest Sharpe Ratio
        print(max_sharpe_port)
        print(type(max_sharpe_port))
        print(max_sharpe_port['Annualized Vol'], max_sharpe_port['Annualized Return'])
        plt.scatter(max_sharpe_port['Annualized Vol'], max_sharpe_port['Annualized Return'], marker=(5, 1, 0),
                    color='r', s=1000)
        # plot green star to highlight position of minimum variance portfolio
        plt.scatter(min_vol_port['Annualized Vol'], min_vol_port['Annualized Return'], marker=(5, 1, 0), color='g',
                    s=1000)
        plt.show()

    def create_portfolio_returns(self,
                                 symbols,
                                 start_date='2010-01-01',
                                 end_date=str(pd.to_datetime('today')).split(' ')[0],
                                 px_type='adjClose'):

        import matplotlib.pyplot as plt
        portfolio_returns = self.get_stock_returns(ticker=symbols,
                                                   start_date=start_date,
                                                   end_date=end_date,
                                                   px_type=px_type)

        portfolio_size = len(portfolio_returns.columns)
        portfolio_weights = np.repeat(1 / portfolio_size, portfolio_size)
        portfolio_vol = TrackStatMomProj.portfolio_standard_deviation(portfolio_weights=portfolio_weights,
                                                                      px_returns_df=portfolio_returns)
        self.logger.info("TrackStatMomProj.create_portfolio_returns(): Portfolio Volatility = %f", portfolio_vol)
        TrackStatMomProj.plot_portfolio_correlation_matrix(px_returns_df=portfolio_returns)
        TrackStatMomProj.plot_portfolio_weighted_return(portfolio_returns, len(portfolio_returns.columns))
        RandomPortfolios = TrackStatMomProj.create_random_portfolios(symbols, portfolio_size, False, 25000)
        # now we need to vectorize a function to run through each row of portfolio weights and create
        # a return and a volatility associeated with that portfolio.
        port_perf_stats_df = RandomPortfolios.apply(self.vectorized_portfolio_calcs, axis=1,
                                                    individual_returns=portfolio_returns)
        min_sharpe, max_sharpe = port_perf_stats_df['Annualized Sharpe'].describe()[['min', 'max']]
        sorted_portfolios = port_perf_stats_df.sort_values(by=['Annualized Sharpe'], ascending=False)
        # Extract the corresponding weights for the MAX SHARPE RATIO portfolio
        MSR_weights = sorted_portfolios.iloc[0, 0:len(portfolio_returns.columns)]
        self.logger.info("TrackStatMomProj.create_portfolio_returns(): Max Sharpe Ratio Weights: %s", MSR_weights)
        # Cast the MSR weights as a numpy array
        MSR_weights_array = np.array(MSR_weights)
        # Calculate the MSR portfolio returns
        portfolio_returns['Portfolio_MSR'] = portfolio_returns.iloc[:, 0:len(portfolio_returns.columns)]. \
            mul(MSR_weights_array, axis=1).sum(axis=1)
        cumm_msr_port = ((1 + portfolio_returns['Portfolio_MSR']).cumprod() - 1)
        self.logger.info("TrackStatMomProj.create_portfolio_returns():Cummulative Return, MSR Portfolio: %f",
                         round(cumm_msr_port[-1] * 100.0, 2))
        cumm_msr_port.plot()
        plt.show()

        # GLOBAL MINIMUM VOLATILITY portfolio
        sorted_portfolios = port_perf_stats_df.sort_values(by=['Annualized Vol'], ascending=True)
        # Extract the corresponding weights
        GMV_weights = sorted_portfolios.iloc[0, 0:len(portfolio_returns.columns)]
        # print ("Global Minimum Variance Weights", GMV_weights, type(GMV_weights))
        self.logger.info("TrackStatMomProj.create_portfolio_returns(): Glob Min Var Port Weights: %s", GMV_weights)
        # Cast the GMV weights as a numpy array
        GMV_weights_array = np.array(GMV_weights)
        # Calculate the GMV portfolio returns
        portfolio_returns['Portfolio_GMV'] = portfolio_returns.iloc[:, 0:len(portfolio_returns.columns)]. \
            mul(GMV_weights_array, axis=1).sum(axis=1)
        cumm_gmv_port = ((1 + portfolio_returns['Portfolio_GMV']).cumprod() - 1)
        self.logger.info("TrackStatMomProj.create_portfolio_returns(): Cummulative Return, GMV Portfolio: %f",
                         round(cumm_gmv_port[-1] * 100.0, 2))
        cumm_gmv_port.plot()
        plt.show()

        # plot the efficient frontier
        TrackStatMomProj.plot_efficient_frontier(port_perf_stats_df)

    def vectorized_portfolio_calcs(self,
                                   row,
                                   individual_returns):

        portfolio_return = individual_returns.iloc[:, 0:len(individual_returns.columns)]. \
            mul(row.values, axis=1).sum(axis=1)
        annualized_port_ret = portfolio_return.mean() * 252
        annualized_port_vol = TrackStatMomProj.portfolio_standard_deviation(row.values, individual_returns)
        annualized_port_sr = annualized_port_ret / annualized_port_vol
        row['Annualized Return'] = annualized_port_ret
        row['Annualized Vol'] = annualized_port_vol
        row['Annualized Sharpe'] = annualized_port_sr
        return row

    def do_px_linear_regression(self,
                                symbol_universe_df):
        small_df = symbol_universe_df.head(1)
        return_val_from_vectorized = small_df.apply(self.vectorized_px_linear_regression, axis=1)
        return return_val_from_vectorized

    def vectorized_px_linear_regression(self,
                                        row,
                                        use_csv=False):
        price_freq = 'D'
        ticker = row.name
        ticker = 'SPY'
        sm = self.sm
        total_value_df = pd.DataFrame()
        px = sm.get_pricing(start_date="2017-01-01",
                            ticker=ticker,
                            fields=['adjClose'],
                            freq=price_freq)[ticker]
        LOGGER.info("TrackStatMomProj.vectorized_skew_binner(): price_df for ticker %s has size %s", ticker,
                    str(px.size))
        if px.size == 0:
            LOGGER.error("TrackStatMomProj.vectorized_skew_binner(): price_df for ticker %s has size %s", ticker,
                         str(px.size))
            return
        # TODO: We should have a specific benchmark for each of the stock tickers. Which means we need
        #       to store this in a database table, the benchmark for each ticker.
        benchmark_px = sm.get_pricing(start_date="2017-01-01",
                                      ticker=self.benchmark_ticker_list,
                                      fields=['adjClose'],
                                      freq=price_freq)
        sm.linear_regression_on_price(tgt_df=px, tgt_ticker=ticker, benchmark_df=benchmark_px)

    def do_skew_binning(self,
                        symbol_universe_df):
        small_df = symbol_universe_df.head(1)
        return_val_from_vectorized = small_df.apply(self.vectorized_skew_binner, axis=1)
        return return_val_from_vectorized

    def vectorized_skew_binner_helper(self, row, win_price_freq, is_excess, price_freq):
        if is_excess:
            # this means we have muiltiple columns to consider, one column for each benchmark
            filter_str = str(win_price_freq) + price_freq + '_skew_exc_ret'
            skew_exc_ret_columns = list(filter(lambda col_nm: (filter_str in col_nm) is True,
                                               row.index))
            filter_str = "rolling_excess_rets"
            rolling_exc_ret_columns = list(filter(lambda col_nm: (filter_str in col_nm) is True, row.index))
            filter_str = "excess_rets"
            exc_ret_columns = list(set(filter(lambda col_nm: (filter_str in col_nm) is True, row.index)) -
                                   set(rolling_exc_ret_columns))
            for skew_exc_ret_col in skew_exc_ret_columns:
                the_exc_ret_skew = row[skew_exc_ret_col]
                if np.isnan(the_exc_ret_skew):
                    return
                bool_list = [y[0] <= the_exc_ret_skew <= y[1] for y in self.bin_range_tuple_list]
                the_skew_range = ([value for key, value in zip(bool_list, self.bin_range_tuple_list) if key])
                ticker = skew_exc_ret_col.split("_")[0].replace('-', '_')
                stock_ticker = ticker.split("_")[0]
                benchmark_ticker = ticker.split("_")[1]
                rolling_exc_ret_name = list(filter(lambda col_nm: (ticker in col_nm) is True, rolling_exc_ret_columns))

                exc_ret_name = list(filter(lambda col_nm: (ticker in col_nm) is True, exc_ret_columns))
                the_rolling_exc_ret = row[rolling_exc_ret_name[0]]
                the_exc_ret = row[exc_ret_name[0]]
                insert_data_dict = {"Returns": the_exc_ret,
                                    "RollingReturns": the_rolling_exc_ret,
                                    "ExcessFlag": True,
                                    "Ticker": ticker,
                                    "Target": stock_ticker,
                                    "Hedge": benchmark_ticker,
                                    "SkewType": skew_exc_ret_col,
                                    "SkewValue": the_exc_ret_skew,
                                    "Date": row.name}
                the_bin_df = self.bin_df_dict[the_skew_range[0]]
                self.bin_df_dict[the_skew_range[0]] = the_bin_df.append(insert_data_dict, ignore_index=True)
        else:
            the_skew_ret = row[str(win_price_freq) + price_freq + "_skew_ret"]
            rolling_ret_name = list(filter(lambda col_nm: ("px_rets_rolling" in col_nm) is True, row.index))
            ret_name = list(
                set(filter(lambda col_nm: ("px_rets" in col_nm) is True, row.index)) - set(rolling_ret_name))
            bool_list = [y[0] <= the_skew_ret <= y[1] for y in self.bin_range_tuple_list]
            the_skew_range = ([value for key, value in zip(bool_list, self.bin_range_tuple_list) if key])
            insert_data_dict = {"Returns": row[ret_name[0]],
                                "RollingReturns": row[rolling_ret_name[0]],
                                "ExcessFlag": False,
                                "Ticker": row['Ticker'],
                                "Target": row['Ticker'],
                                "Hedge": "",
                                "SkewType": str(win_price_freq) + price_freq + "skew_ret",
                                "SkewValue": the_skew_ret,
                                "Date": row.name}
            the_bin_df = self.bin_df_dict[the_skew_range[0]]
            self.bin_df_dict[the_skew_range[0]] = the_bin_df.append(insert_data_dict, ignore_index=True)

    def vectorized_skew_binner(self,
                               row,
                               use_csv=False):
        price_freq = 'D'
        win_price_freq = 30
        return_period = 10
        ticker = row.name
        ticker = 'COST'
        sm = self.sm
        total_value_df = pd.DataFrame()
        if use_csv:
            self.bin_df_dict = dict()
            for bin_range_tuple in self.bin_range_tuple_list:
                csv_file_name = self.skew_bin_csv_files_dir + ticker + '_' + str(bin_range_tuple[0]) + '_' + \
                                str(bin_range_tuple[1]) + '_bin.csv'
                self.bin_df_dict[bin_range_tuple] = pd.read_csv(csv_file_name, index_col=0)
                value = self.bin_df_dict[bin_range_tuple]
                html_output_file_tickerYearGrouping = self.bokeh_html_file_dir + ticker + '_' + str(
                    bin_range_tuple[0]) + '_' + str(bin_range_tuple[1]) + '_skew_bin_analysis_TickerYearGrouping.html'
                html_output_file_tickerGrouping = self.bokeh_html_file_dir + ticker + '_' + str(
                    bin_range_tuple[0]) + '_' + str(bin_range_tuple[1]) + '_skew_bin_analysis_TickerGrouping.html'
                html_output_file_title = ticker + ' Skew vs. Returns Analysis: ' + str(bin_range_tuple) + ' Skew Range'
                ExtendBokeh.bokeh_bar_plot(value, csv_file_name,
                                           html_output_file=html_output_file_tickerYearGrouping,
                                           html_output_file_title=html_output_file_title)
                ExtendBokeh.bokeh_bar_plot_single_group(value, csv_file_name,
                                                        html_output_file=html_output_file_tickerGrouping,
                                                        html_output_file_title=html_output_file_title)
            total_csv_file_name = self.skew_bin_csv_files_dir + ticker + '_all_bins.csv'
            total_value_df = pd.read_csv(total_csv_file_name, index_col=0)
            html_output_file_title = ticker + 'Skew vs. Sharpe Analysis'
            html_output_file_binGrouping = self.bokeh_html_file_dir + ticker + '_skew_bin_analysis_BinGrouping.html'
            total_write_out_fn_path = self.skew_bin_csv_files_dir + ticker + '_all_bins.csv'
            ExtendBokeh.bokeh_bar_plot_all_bins(total_value_df, total_write_out_fn_path,
                                                html_output_file=html_output_file_binGrouping,
                                                html_output_file_title=html_output_file_title)
            return
        px = sm.get_pricing(start_date="2017-01-01",
                            ticker=ticker,
                            fields=['adjClose'],
                            freq=price_freq)[ticker]
        LOGGER.info("TrackStatMomProj.vectorized_skew_binner(): price_df for ticker %s has size %s", ticker,
                    str(px.size))
        if px.size == 0:
            LOGGER.error("TrackStatMomProj.vectorized_skew_binner(): price_df for ticker %s has size %s", ticker,
                         str(px.size))
            return
        # TODO: We should have a specific benchmark for each of the stock tickers. Which means we need
        #       to store this in a database table, the benchmark for each ticker.
        benchmark_px = sm.get_pricing(start_date="2017-01-01",
                                      ticker=self.benchmark_ticker_list,
                                      fields=['adjClose'],
                                      freq=price_freq)
        df_excess, excess_rets_type_list, excess_rets, excess_log_rets, \
        p_iqr_hist_er, p_iqr_cdf_er, p_iqr_3sr_hist_er, p_iqr_3sr_cdf_er = \
            self.do_excess_rets_analysis(ticker,
                                         px,
                                         benchmark_px,
                                         price_freq,
                                         return_period)
        excess_rets_skew_columns = list(filter(lambda col_nm: (str(win_price_freq) + price_freq +
                                                               "_skew_exc_ret" in col_nm) is True, df_excess.columns))
        rolling_excess_rets_columns = list(filter(lambda col_nm: ("rolling_excess_rets" in col_nm) is True,
                                                  df_excess.columns))
        excess_rets_columns = list(set(filter(lambda col_nm: ("excess_rets" in col_nm) is True, df_excess.columns)) -
                                   set(rolling_excess_rets_columns))
        _ = df_excess[excess_rets_skew_columns + rolling_excess_rets_columns + excess_rets_columns].apply \
            (self.vectorized_skew_binner_helper, axis=1, win_price_freq=win_price_freq, is_excess=True,
             price_freq=price_freq)
        df_px, px_ret_type_list, px_rets, px_log_rets, p_iqr_hist, p_iqr_cdf, p_iqr_3sr_hist, p_iqr_3sr_cdf = \
            self.do_px_rets_analysis(ticker,
                                     px,
                                     price_freq,
                                     return_period)
        skew_columns = list(filter(lambda col_nm: (str(win_price_freq) + price_freq + "_skew_ret" in col_nm) is True,
                                   df_px.columns))
        rolling_rets_columns = list(filter(lambda col_nm: ("rets_rolling" in col_nm) is True, df_px.columns))
        rets_columns = list(set(filter(lambda col_nm: ("rets" in col_nm) is True, df_px.columns)) -
                            set(rolling_rets_columns))
        df_px["Ticker"] = ticker
        _ = df_px[["Ticker"] + skew_columns + rolling_rets_columns + rets_columns].dropna(). \
            apply(self.vectorized_skew_binner_helper, axis=1, win_price_freq=win_price_freq,
                  is_excess=False, price_freq=price_freq)
        for key, value in self.bin_df_dict.items():
            if value.size == 0:
                LOGGER.info("TrackStatMomProj.vectorized_skew_binner(): Fwd Return Stats "
                            "ticker: %s, bin: %s, Mean: %s, Std Dev: %s,"
                            "Count: %s, Sharpe Ratio: %s", ticker, str(key), str(0.0), str(0.0), str(0), str(0.0))
                continue
            value['Bin'] = float(key[0])
            if total_value_df.empty:
                total_value_df = value
            else:
                total_value_df = total_value_df.append(value, ignore_index=True)
            value_px_rets = value[value.ExcessFlag.eq(False)]
            value_excess_rets = value[value.ExcessFlag.eq(True)]
            sharpe_rat_rolling = value_px_rets.RollingReturns.mean() / value_px_rets.RollingReturns.std() * \
                                 np.sqrt(value_px_rets.RollingReturns.size)
            sharpe_rat_excess_rolling = value_excess_rets.RollingReturns.mean() / value_excess_rets.RollingReturns. \
                std() * np.sqrt(value_excess_rets.RollingReturns.size)
            LOGGER.info("TrackStatMomProj.vectorized_skew_binner(): Fwd Px Return Stats "
                        "ticker: %s, bin: %s, Mean: %s, Std Dev: %s, "
                        "Count: %s, Sharpe Ratio: %s", ticker, str(key), str(value_px_rets.RollingReturns.mean()),
                        str(value_px_rets.RollingReturns.std()), str(value_px_rets.RollingReturns.size),
                        str(sharpe_rat_rolling))
            LOGGER.info("TrackStatMomProj.vectorized_skew_binner(): Fwd Excess Return Stats "
                        "ticker: %s, bin: %s, Mean: %s, Std Dev: %s, "
                        "Count: %s, Sharpe Ratio: %s", ticker, str(key), str(value_excess_rets.RollingReturns.mean()),
                        str(value_excess_rets.RollingReturns.std()), str(value_excess_rets.RollingReturns.size),
                        str(sharpe_rat_excess_rolling))
            write_out_fn = ticker + '_' + str(key[0]) + '_' + str(key[1]) + '_bin.csv'
            total_write_out_fn_path = self.skew_bin_csv_files_dir + write_out_fn
            value.to_csv(total_write_out_fn_path)
            html_output_file_tickerYearGrouping = self.bokeh_html_file_dir + ticker + '_' + str(key[0]) + '_' + str(
                key[1]) + \
                                                  '_skew_bin_analysis_TickerYearGrouping.html'
            html_output_file_tickerGrouping = self.bokeh_html_file_dir + ticker + '_' + str(key[0]) + '_' + str(
                key[1]) + \
                                              '_skew_bin_analysis_TickerGrouping.html'
            html_output_file_title = ticker + ' Skew vs. Returns Analysis: ' + str(key) + ' Skew Range'
            ExtendBokeh.bokeh_bar_plot(value, total_write_out_fn_path,
                                       html_output_file=html_output_file_tickerYearGrouping,
                                       html_output_file_title=html_output_file_title)
            ExtendBokeh.bokeh_bar_plot_single_group(value, total_write_out_fn_path,
                                                    html_output_file=html_output_file_tickerGrouping,
                                                    html_output_file_title=html_output_file_title)
        html_output_file_title = ticker + 'Skew vs. Sharpe Analysis'
        html_output_file_binGrouping = self.bokeh_html_file_dir + ticker + '_skew_bin_analysis_BinGrouping.html'
        html_output_file_binTickerGrouping = self.bokeh_html_file_dir + ticker + \
                                             '_skew_bin_analysis_BinGroupingInclExcess.html'
        total_write_out_fn_path = self.skew_bin_csv_files_dir + ticker + '_all_bins.csv'
        total_value_df.to_csv(total_write_out_fn_path)
        ExtendBokeh.bokeh_bar_plot_all_bins(total_value_df, total_write_out_fn_path,
                                            html_output_file=html_output_file_binGrouping,
                                            html_output_file_title=html_output_file_title)
        ExtendBokeh.bokeh_bar_plot_all_bins_by_ticker(total_value_df, total_write_out_fn_path,
                                                      html_output_file=html_output_file_binTickerGrouping,
                                                      html_output_file_title=html_output_file_title)

    def vectorized_symbols_func(self,
                                row):

        # some parameters
        save_or_show = 'show'
        price_freq = 'D'
        win_price_freq = 30
        return_period = 5
        skew_filter = (0, 0)
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
        excess_rets_olanalysis_plots = []
        first_ticker_date = df_px.iloc[0].name
        for benchmark_ticker in benchmark_ticker_list:
            filtered_excess_rets_type_list = list(filter(lambda col_nm:
                                                         (ticker + '_' + benchmark_ticker in col_nm) is True,
                                                         excess_rets_type_list))
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
                                               benchmark_px_series=benchmark_px[benchmark_ticker].loc[
                                                                   first_ticker_date:, ],
                                               benchmark_ticker=benchmark_ticker,
                                               title=[co_nm + ' Px Chart Band Breaches'],
                                               subtitle=["Spread: " + ticker + '-' + benchmark_ticker],
                                               type_list=['adjClose', 'adjClose'],
                                               spans_list=spans_tuples_list_er,
                                               which_axis_list=[0, 1])

            excess_rets_line_plot = \
                ExtendBokeh.bokeh_px_returns_plot(data=df_excess,
                                                  freq=price_freq,
                                                  title=[co_nm + ' Excess Returns'],
                                                  subtitle=["Spread: " + ticker + '-' + benchmark_ticker],
                                                  type_list=filtered_excess_rets_type_list,
                                                  scatter=False,
                                                  rolling_window_size=win_price_freq)
            ticker_bm_excess_ret = list(filter(lambda col_nm:
                                               (benchmark_ticker + '_excess_rets' in col_nm) is True,
                                               excess_rets_type_list))
            ticker_bm_rolling_excess_ret = list(filter(lambda col_nm:
                                                       (benchmark_ticker + '_rolling_excess_rets' in col_nm) is True,
                                                       excess_rets_type_list))
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
            excess_rets_normal_ovl, excess_rets_normal_cdf = \
                ExtendBokeh.bokeh_histogram_overlay_normal(df_excess[col_str],
                                                           ["Excess Returns Hist",
                                                            "Excess Returns CDF"])
            excess_rets_hist_plots.append(excess_rets_normal_ovl)
            excess_rets_hist_plots.append(excess_rets_normal_cdf)
            # px_rets_lognormal_ovl, px_rets_lognormal_cdf = \
            #    ExtendBokeh.bokeh_histogram_overlay_normal(px_log_rets,
            #                                               titles=['Px Log Returns Histogram',
            #                                                       'Px Log Returns CDF'])
        excess_rets_olanalysis_plots.append(p_iqr_hist_er)
        excess_rets_olanalysis_plots.append(p_iqr_cdf_er)
        excess_rets_olanalysis_plots.append(p_iqr_3sr_hist_er)
        excess_rets_olanalysis_plots.append(p_iqr_3sr_cdf_er)
        # hist_plots.append(px_rets_lognormal_ovl)
        # hist_plots.append(px_rets_lognormal_cdf)

        #### below code puts out the plots for <ticker>.hist.html ####
        ##############################################################
        spans_tuples_list = ExtendBokeh.bokeh_create_mean_var_spans(df_px,
                                                                    ticker=ticker,
                                                                    freq=price_freq,
                                                                    rolling_window_size=win_price_freq,
                                                                    var_bandwidth=3.0,
                                                                    color=('red', 'green'))
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
        df_px['abs_adjClose_px_rets'] = abs(df_px['adjClose_px_rets'])
        df_px['abs_adjClose_px_rets_rolling'] = abs(df_px['adjClose_px_rets_rolling'])
        p_rolling_pxret_skew_line, p_rolling_pxret_skew_scatter = \
            ExtendBokeh.bokeh_rolling_pxret_skew(df_px,
                                                 freq=price_freq,
                                                 title=['Rolling Returns vs. Rolling Skew'],
                                                 subtitle=[str(return_period) + price_freq + '/' +
                                                           str(win_price_freq) + price_freq + ' Window'],
                                                 type_list=['adjClose_px_rets_rolling',
                                                            str(win_price_freq) + price_freq + '_skew_ret'],
                                                 rolling_window_size=win_price_freq,
                                                 skew_filter=skew_filter)
        p_abs_rolling_pxret_skew_line, p_abs_rolling_pxret_skew_scatter = \
            ExtendBokeh.bokeh_rolling_pxret_skew(df_px,
                                                 freq=price_freq,
                                                 title=['Abs(Rolling Returns) vs. Rolling Skew'],
                                                 subtitle=[str(return_period) + price_freq + '/' +
                                                           str(win_price_freq) + price_freq + ' Window'],
                                                 type_list=['abs_adjClose_px_rets_rolling',
                                                            str(win_price_freq) + price_freq + '_skew_ret'],
                                                 rolling_window_size=win_price_freq,
                                                 skew_filter=skew_filter)
        p_pxret_rolling_skew_line, p_pxret_rolling_skew_scatter = \
            ExtendBokeh.bokeh_rolling_pxret_skew(df_px,
                                                 freq=price_freq,
                                                 title=['Returns vs. Rolling Skew'],
                                                 subtitle=[str(1) + price_freq + '/' +
                                                           str(win_price_freq) + price_freq + ' Window'],
                                                 type_list=['adjClose_px_rets',
                                                            str(win_price_freq) + price_freq + '_skew_ret'],
                                                 rolling_window_size=win_price_freq,
                                                 skew_filter=skew_filter)
        p_abs_pxret_rolling_skew_ling, p_abs_pxret_rolling_skew_scatter = \
            ExtendBokeh.bokeh_rolling_pxret_skew(df_px,
                                                 freq=price_freq,
                                                 title=['Abs(Returns) vs. Rolling Skew'],
                                                 subtitle=[str(1) + price_freq + '/' +
                                                           str(win_price_freq) + price_freq + ' Window'],
                                                 type_list=['abs_adjClose_px_rets',
                                                            str(win_price_freq) + price_freq + '_skew_ret'],
                                                 rolling_window_size=win_price_freq,
                                                 skew_filter=skew_filter)

        #### Below code puts out the plots for <ticker>.Excess.Returns.html ####
        ########################################################################

        # next, lets do the KS Test, to test for normality. We will do JB Test later.
        ks_test_stat_raw_rets, p_value_raw_rets = StatsTests.ks_test(rvs=px_rets, dist_size=len(px_rets), cdf='norm')
        ks_test_stat_log_rets, p_value_log_rets = StatsTests.ks_test(rvs=px_log_rets, dist_size=len(px_log_rets),
                                                                     cdf='norm')
        self.logger.info(
            "TrackstatMomProj.vectorized_symbols_func(): KS Test Stat (Raw Returns) is %s and p_value is %s",
            str(ks_test_stat_raw_rets), str(p_value_raw_rets))
        self.logger.info(
            "TrackstatMomProj.vectorized_symbols_func(): Is KS Test Stat (Raw Returns) %s as close to 0 as possible"
            " and p_value %s as close to 1 as possible? If p_value less than 0.05, Null Hypothesis (the"
            "two distributions RVS and CDF are equal) is rejected!", str(ks_test_stat_raw_rets), str(p_value_raw_rets))
        self.logger.info(
            "TrackstatMomProj.vectorized_symbols_func(): KS Test Stat (Log Returns) is %s and p_value is %s",
            str(ks_test_stat_log_rets), str(p_value_log_rets))
        self.logger.info(
            "TrackstatMomProj.vectorized_symbols_func(): Is KS Test Stat (Log Returns) %s as close to 0 as possible"
            " and p_value %s as close to 1 as possible? If p_value less than 0.05, the Null Hypothesis (the"
            "two distributions RVS and CDF are equal) is rejected!", str(ks_test_stat_log_rets), str(p_value_log_rets))

        # px_rets_weibull_ovl = ext_bokeh.bokeh_histogram_overlay_weibull(px_rets)
        # hist_plots.append(px_rets_weibull_ovl)
        html_output_file_path = OSMuxImpl.get_proper_path('/workspace/data/bokeh/html/')
        html_output_file_title = ticker + '.hist.html'
        html_output_file = html_output_file_path + html_output_file_title

        html_output_file_title_skew_analysis = ticker + '.skew.html'
        html_output_file_skew_analysis = html_output_file_path + html_output_file_title_skew_analysis

        html_output_file_title_excess_rets_analysis = ticker + '.excess.rets.html'
        html_output_file_excess_rets_analysis = html_output_file_path + html_output_file_title_excess_rets_analysis

        html_output_file_title_excess_rets_olanalysis = ticker + '.outlier.analysis.html'
        html_output_file_excess_rets_olanalysis = html_output_file_path + \
                                                  html_output_file_title_excess_rets_olanalysis

        if save_or_show is 'show':
            ExtendBokeh.show_hist_plots([p_rolling_pxret_skew_line, p_rolling_pxret_skew_scatter,
                                         p_pxret_rolling_skew_line, p_pxret_rolling_skew_scatter,
                                         p_abs_rolling_pxret_skew_line, p_abs_rolling_pxret_skew_scatter,
                                         p_abs_pxret_rolling_skew_ling, p_abs_pxret_rolling_skew_scatter],
                                        html_output_file_skew_analysis,
                                        html_output_file_title_skew_analysis)
            # ExtendBokeh.show_hist_plots(excess_rets_hist_plots,
            #                            html_output_file_excess_rets_analysis,
            #                            html_output_file_title_excess_rets_analysis)
            # ExtendBokeh.show_hist_plots(excess_rets_olanalysis_plots,
            #                            html_output_file_excess_rets_olanalysis,
            #                            html_output_file_title_excess_rets_olanalysis)
            # ExtendBokeh.show_hist_plots(hist_plots,
            #                            html_output_file,
            #                            html_output_file_title)
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
            ExtendBokeh.save_html(excess_rets_olanalysis_plots,
                                  html_output_file_excess_rets_olanalysis,
                                  html_output_file_title_excess_rets_olanalysis)

        # below is pyplot functionality - not using pyplot as it requires subscription
        # hist_data = [go.Histogram(y=px_rets)]
        # url = py.plot(hist_data, filename=self.plotly_histograms_dir + ticker + '_hist')
        # self.logger.info('TrackStatMomProj.vectorized_symbols_func(): histogram url for ticker %s is %s',
        #                 ticker, str(url))

        # Long run SKEW Value (for the entire data set)
        # Might make sense to keep a long run skew value and see how the short run skew revolves around it.
        skew = sm.calc_stock_return_skew(ticker=ticker, data=px_rets)
        # Long run Kurtosis Value (for the entire data set)
        # Makes sense to kee a long run kurtosis value and see how the short run revolves around it
        kurt = sm.calc_stock_return_kurtosis(ticker=ticker, data=px_rets)


if __name__ == '__main__':
    # TrackStatMomProj.monte_carlo_vs_bootstrapping_example()

    spyder_etfs_list = ['XLE', 'XLU', 'XLF', 'XLI', 'XLK', 'XLP', 'XTL', 'XLV', 'XLY']
    index_etfs_list = ['SPY', 'QQQ', 'IWM', 'TLT', 'ZROZ', 'GLD', 'HYG',
                       'LQD', 'SHY', 'FXE', 'FXY', 'EEM', 'FXI', 'IEF']
    index_etfs_list = ['SPY', 'QQQ', 'IWM', 'DIA']

    the_etf_list = index_etfs_list

    tsmp = TrackStatMomProj(use_iex_trading_symbol_universe=False, stock_universe_filename="NQ100")
    stock_universe_df, ticker_col_nm = tsmp.get_stock_universe()
    symbols_list = list(stock_universe_df.index)

    # the_etf_df = stock_universe_df.reindex(the_etf_list)
    # tsmp.create_portfolio_returns(the_etf_df.index, start_date="2018-01-01")
    # I like start date of 2018 for going forward because of volatility, autocorrelation of vol tells me going
    # forward we will have less drifting up markets and more volatile markets like the past 2.5 years.
    # tsmp.get_px_df(stock_universe_df)
    # tsmp.do_skew_binning(stock_universe_df)
    tsmp.do_px_linear_regression(stock_universe_df)
