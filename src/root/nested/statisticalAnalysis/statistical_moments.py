import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.stattools import jarque_bera
from root.nested.dataAccess.tiingo_data_object import TiingoDataObject
from root.nested.statisticalAnalysis.hacker_stats import HackerStats
from root.nested import get_logger

class StatisticalMoments(object):
    """description of class"""

    LOGGER = get_logger()
    MAX_TICKER_PULL_SIZE = 3

    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    @staticmethod
    def calc_skew_example():

        xs2 = np.linspace(stats.gamma.ppf(0.01, 0.7, loc = -1), stats.gamma.ppf(0.99, 0.7, loc=-1), 150) + 1
        X = stats.gamma.pdf(xs2, 1.5)
        plt.plot(xs2, X, label = 'Skew Test')
        plt.legend()
        plt.show()
        skew = stats.skew(X)
        print ("Skew: ", skew)
        if skew > 0:
            print ('The distribution is positively skewed')
        elif skew < 0:
            print ('The distribution is negatively skewed')
        else:
            print ('The distribution is symetric')

    @staticmethod
    def calc_kurtosis_example():

        xs = np.linspace(-6, 6, 300) + 2
        Y = stats.cosine.pdf(xs)
        kurtosis = stats.kurtosis(Y)
        plt.hist(xs, Y)
        if (kurtosis < 0):
            print ("Excess Kurtosis is", kurtosis, ". Because the excess kurtosis is negative, Y is platykurtic. Platykurtic distributions cluster around the mean, so large values in either direction are less likely.")
        elif (kurtosis > 0):
            print ("Excess Kurtosis is", kurtosis, ". Because the excess kurtosis is positive, Y is leptokurtic. Leptokurtic distributions have fatter tails, meaning there is more tail risk.")

    def jarque_bera_calibration(self,
                                pvalue_th):

        """Jarque-Bera test whether a distribution is normal or not. How? We run Jarque-Bera on
       distributions that we know to be normal. Because np.random.normal guarantees a small p-value,
       (guarantees normal distriubted return distribution, our threshold """

        N = 1000
        M = 1000
        pvalues = np.ndarray((N))
        for i in range(N):
            # draw M samples from a normal distribution
            X = np.random.normal(0, 1, M)
            _, pvalue, _, _ = jarque_bera(X)
            pvalues[i] = pvalue
        num_significant = len(pvalues[pvalues < pvalue_th])
        print ("% of samples that have a pvalue less than threshold : ",float(num_significant)*100.0/N, "%")

    def test_normality(self,
                       ticker,
                       start_date,
                       end_date):

        StatisticalMoments.LOGGER.info("StatisticalMoments.test_normality(): running function...")
        returns = self.get_stock_returns(ticker, start_date, end_date)
        StatisticalMoments.LOGGER.info('StatisticalMoments.test_normality(): skew is %f', stats.skew(returns))
        StatisticalMoments.LOGGER.info('StatisticalMoments.test_normality(): mean is %f', np.mean(returns))
        StatisticalMoments.LOGGER.info('StatisticalMoments.test_normality(): median is %f', np.median(returns))
        plt.hist(returns, 30)
        plt.show()
        _, pvalue, _, _ = jarque_bera(returns)
        if (pvalue > 0.05):
            StatisticalMoments.LOGGER.info('StatisticalMoments.test_normality(): '
                                           'jarques-bera test p-value of %f is within 5% error tolerance, '
                                           'and therefore the returns are likely normal.', pvalue)
            print ("jarques-bera test p-value of ", pvalue, " is within 5% error tolerance, and therefore the returns are likely normal.")
        else:
            StatisticalMoments.LOGGER.info('StatisticalMoments.test_normality():'
                                           'jarques-bera test p-value of %f is not within 5% error tolerance; '
                                           'returns are not likely normal.', pvalue)
            print ("jarques-bera test p-value of ", pvalue, " is not within 5% error tolerance; returns are not likely normal.")

    def get_stock_returns(self,
                          ticker,
                          freq='D',
                          start_date='2010-01-01',
                          end_date=str(pd.to_datetime('today')).split(' ')[0],
                          px_type='adjClose'):

        StatisticalMoments.LOGGER.info('StatisticalMoments.get_stock_returns(): running function on ticker %s', ticker)
        symbols = [ticker]
        source = 'Tiingo'
        mdo = TiingoDataObject(start_date = start_date, 
                               end_date = end_date, 
                               source = source, 
                               symbols = symbols)
        pricing_dict = mdo.get_px_data_df(start_date,
                                          end_date)
        pricing_df = pricing_dict[ticker]
        if freq is not 'D':
            pricing_df = self.down_sample_daily_price_data(pricing=pricing_df, to_freq=freq)
        pricing = pricing_df[px_type]
        returns = pricing.pct_change()[1:]
        return returns

    def get_stock_excess_return(self,
                                stock_ticker,
                                benchmark_ticker,
                                start_date='2010-01-01',
                                end_date=str(pd.to_datetime('today')).split(' ')[0],
                                px_type='adjClose'):

        StatisticalMoments.LOGGER.info('StatisticalMoments.get_stock_excess_return(): running function...')
        symbols = [stock_ticker, benchmark_ticker]
        source='Tiingo'
        mdo = TiingoDataObject(start_date = start_date,
                               end_date = end_date,
                               source = source,
                               symbols = symbols)
        pricing_dict = mdo.get_px_data_df(start_date=start_date,
                                          end_date=end_date)
        stock_pricing_df = pricing_dict[stock_ticker]
        benchmark_pricing_df = pricing_dict[benchmark_ticker]
        return (stock_pricing_df - benchmark_pricing_df)

    def get_rolling_excess_returns(self,
                                   ticker,
                                   benchmark,
                                   freq='D',
                                   start_date='2010-01-01',
                                   end_date=str(pd.to_datetime('today')).split(' ')[0],
                                   px_type='adjClose',
                                   ticker_data=None,
                                   benchmark_data=None,
                                   window_size=30,
                                   shift_rets_series=False):

        StatisticalMoments.LOGGER.info('StatisticalMoments.get_rolling_excess_returns(): running function ticker %s, '
                                       'benchmark %s', ticker, benchmark)
        ticker_roll_rets = self.get_rolling_returns(ticker=ticker,
                                                    freq=freq,
                                                    start_date=start_date,
                                                    end_date=end_date,
                                                    px_type=px_type,
                                                    data=ticker_data,
                                                    window_size=window_size,
                                                    shift_rets_series=shift_rets_series)
        benchmark_roll_rets = self.get_rolling_returns(ticker=benchmark,
                                                       freq=freq,
                                                       start_date=start_date,
                                                       end_date=end_date,
                                                       px_type=px_type,
                                                       data=benchmark_data,
                                                       window_size=window_size,
                                                       shift_rets_series=shift_rets_series)
        return (ticker_roll_rets - benchmark_roll_rets).dropna()

    def get_rolling_returns(self,
                            ticker,
                            freq='D',
                            start_date='2010-01-01',
                            end_date=str(pd.to_datetime('today')).split(' ')[0],
                            px_type='adjClose',
                            data=None,
                            window_size=30,
                            shift_rets_series=False):

        StatisticalMoments.LOGGER.info('StatisticalMoments.get_rolling_returns(): running function for ticker %s...',
                                       ticker)
        if data is None:
            symbols = [ticker]
            source = 'Tiingo'
            mdo = TiingoDataObject(start_date=start_date,
                                   end_date=end_date,
                                   source=source,
                                   symbols=symbols)
            pricing_dict = mdo.get_px_data_df(start_date,
                                          end_date)
            pricing_df = pricing_dict[ticker]
        else:
            pricing_df = data
        if freq is not 'D':
            pricing_df = self.down_sample_daily_price_data(pricing=pricing_df, to_freq=freq)
        pricing = pricing_df[px_type]
        rolling_returns = pricing.pct_change(periods = window_size)
        if shift_rets_series is True:
            rolling_returns = rolling_returns.shift(-1*window_size)
        return rolling_returns.squeeze()

    """ get_pricing: main function to retrieve daily price data
        The source of this data is currently Tiingo. 
    """
    def get_pricing(self,
                    ticker,
                    freq='D', # options are 'D', 'W', 'M'
                    start_date='2010-01-01',
                    end_date=str(pd.to_datetime('today')).split(' ')[0],
                    fields=['adjOpen', 'adjHigh', 'adjLow', 'adjClose']):

        if (type(ticker) is not list):
            symbols = [ticker]
        else:
            symbols = ticker
        if len(symbols) > 3:
            StatisticalMoments.LOGGER.error("max number of ticker pulls allowed at once is 3, "
                                            "%s given!", str(len(symbols)))
            raise ValueError("max number of ticker to pull at once is " + \
                             str(StatisticalMoments.MAX_TICKER_PULL_SIZE) + \
                             ", input list is length " + str(len(symbols)) + "!!")
            return

        source = 'Tiingo'
        mdo = TiingoDataObject(start_date=start_date,
                               end_date=end_date,
                               source=source,
                               symbols=symbols)
        pricing_dict = mdo.get_px_data_df(start_date,
                                          end_date)
        return_dict = {}
        for ticker in symbols:
            pricing_df = pricing_dict[ticker]
            if freq is not 'D':
                pricing_df=self.down_sample_daily_price_data(pricing=pricing_df, to_freq=freq)
            pricing = pricing_df[fields]
            return_dict[ticker] = pricing
        return return_dict

    def down_sample_daily_price_data(self,
                                     pricing,
                                     to_freq='W'):

        output = pricing.resample(to_freq,  # Weekly resample
                                  how={'adjOpen': take_first,
                                       'adjHigh': 'max',
                                       'adjLow': 'min',
                                       'adjClose': take_last,
                                       'adjVolume': 'sum'})
        return output

    """ Calculate the kurtosis of the returns for the specificed ticker, with specificed
        start and end dates. 
    """
    def calc_stock_return_kurtosis(self,
                                   ticker,
                                   data = None,
                                   start_date = '2010-01-01',
                                   end_date = str(pd.to_datetime('today')).split(' ')[0]):

        if data is None:
            returns = self.get_stock_returns(ticker, start_date, end_date)
        else:
            returns = data
        kurt = stats.kurtosis(returns)
        if kurt < 0:
            StatisticalMoments.LOGGER.info('StatisticalMoments.calc_stock_return_kurtosis(): Excess Kurtosis is %f.'
                                           ' Because the excess kurtosis is negative, Y is platykurtic. Platykurtic'
                                           ' distributions cluster around the mean, meaning less tail risk.', kurt)
        elif kurt > 0:
            StatisticalMoments.LOGGER.info('StatisticalMoments.calc_stock_return_kurtosis(): Excess Kurtosis is %f.'
                                           ' Because the excess kurtosis is positive, Y is leptokurtic. Leptokurtic'
                                           ' distributions have fatter tails, meaning there is more tail risk.', kurt)
        return kurt

    """Calculate the skew of the returns for the specified ticker, with specificed
        start and end dates.
    """
    def calc_stock_return_skew(self,
                               ticker,
                               data = None,
                               start_date='2010-01-01', # 2010-01-01 to present is max data offered by Tiingo
                               end_date=str(pd.to_datetime('today')).split(' ')[0], # str, e.g. "2011-01-01"
                               px_type = 'adjClose'):

        if data is None:
            returns = self.get_stock_returns(ticker, start_date, end_date, px_type=px_type)
        else:
            returns = data
        skew = stats.skew(returns)
        StatisticalMoments.LOGGER.info('StatisticalMoments.calc_stock_return_skew(): Skew = %f', skew)
        return skew
      
    """ Calculate a rolling skew, with window size "window_size", for the specified
        ticker with specified start and end dates.
    """
    def rolling_skew_of_returns(self,
                                ticker,
                                start_date,
                                end_date,
                                window_size,
                                data = None):

        if data is None:
            returns = self.get_stock_returns(ticker, start_date, end_date)
        else:
            returns = data
        roll_func = returns.rolling(window = window_size, center = False).skew()
        plt.plot(roll_func)
        plt.xlabel('Time')
        plt.ylabel(str(window_size) + " Period Rolling Skew")
        plt.show()

    """ Calculate a rolling kurtosis, with window size "window_size, for the specified
        ticker with specificed start and end dates.
    """
    def rolling_kurt_of_returns(self,
                                ticker,
                                start_date,
                                end_date,
                                window_size,
                                data = None):

        if data is None:
            returns = self.get_stock_returns(ticker, start_date, end_date)
        else:
            returns = data
        roll_func = returns.rolling(window = window_size, center = False).kurtosis()
        plt.plot(roll_func)
        plt.xlabel('Time')
        plt.ylabel(str(window_size) + " Period Rolling Kurtosis")
        plt.show()

    @staticmethod
    def remove_outliers_from_normal_dist(px_returns_series,
                                         sigma_multiplier=3.0):

        """ This function removes any returns from the dataframe that are beyond 3 sigma,
        thereby making the returns series potentially normal."""
        px_rets_series = pd.Series(px_returns_series)
        mu = px_rets_series.mean()
        sigma = px_rets_series.std()
        top = mu+sigma_multiplier*sigma
        bottom = mu-sigma_multiplier*sigma
        bool_non_outliers = (px_rets_series <= top) | (px_rets_series >= bottom)
        no_outliers = px_returns_series[bool_non_outliers].sort_index()
        return no_outliers

    @staticmethod
    def remove_outliers_from_non_normal_dist(px_returns_series,
                                             k_IQR = 1.5):

        """ Here we will use the IQR method. To classify outliers liberally, use k_IQR = 1.5
         To classify outliers very selectively, use k_IQR = 3.0"""
        num_obs = int(px_returns_series.describe()['count'])
        q25 = px_returns_series.describe()['25%']
        q75 = px_returns_series.describe()['75%']
        iqr = q75 - q25
        # calculate the outlier cutoff
        cut_off = iqr * k_IQR
        lower, upper = q25 - cut_off, q75 + cut_off
        # identify outliers
        outliers = px_returns_series[px_returns_series < lower]
        outliers = outliers.append(px_returns_series[px_returns_series > upper]).sort_index()
        StatisticalMoments.LOGGER.info('StatisticalMoments.remove_outliers_from_non_normal_dist(): '
                                       'Identified Outliers (Series): %d' % int(outliers.describe()['count']))
        percent_outliers = int(outliers.describe()['count'])/num_obs*100.0
        StatisticalMoments.LOGGER.info('StatisticalMoments.remove_outliers_from_non_normal_dist(): '
                                       'Percent Outliers: %.3f%s' % (percent_outliers, '%'))
        # remove outliers
        outliers_removed = px_returns_series[~(px_returns_series.isin(outliers))]
        StatisticalMoments.LOGGER.info('StatisticalMoments.remove_outliers_from_non_normal_dist(): '
                                       'Non-outlier observations (Series): %d' % len(outliers_removed))
        return outliers_removed

    @staticmethod
    def get_outliers_normal_dist(px_returns_series):

        px_rets_series = pd.Series(px_returns_series)
        mu = px_rets_series.mean()
        sigma = px_rets_series.std()
        top = mu+3*sigma
        bottom = mu-3*sigma
        top_outliers = px_rets_series[px_rets_series > top]
        bottom_outliers = px_rets_series[px_rets_series < bottom]

        return top_outliers.append(bottom_outliers).sort_index()

    @staticmethod
    def get_outliers_non_normal_dist(px_returns_series,
                                     k_IQR):

        q25 = px_returns_series.describe()['25%']
        q75 = px_returns_series.describe()['75%']
        iqr = q75 - q25
        # calculate the outlier cutoff
        cut_off = iqr * k_IQR
        lower, upper = q25 - cut_off, q75 + cut_off
        outliers = px_returns_series[px_returns_series < lower]

        return outliers.append(px_returns_series[px_returns_series > upper]).sort_index()

    def calc_variance_metrics_example(self,
                                      data,
                                      semivar_cutoff):

        """EXAMPLE function: Return the following types of variance metrics:
                Range, Mean Absolute Deviation, Variance, Standard Deviation, Semivariance,
                and Semideviation.
                Parameter "semivar_cutoff" specifies where in the data to split high and low. For example,
                if we want to calculate volatility for a stock separately for upside moves and downside moves,
                we would set the semivar_cutoff to 0.0 (where data is a numpy array of returns).
        """
        mu = np.mean(data)
        dist_range = np.ptp(data)
        abs_deviation = [np.abs(mu - x) for x in data]
        MAD = np.sum(abs_deviation)/len(data)
        variance = np.var(data)
        std_dev = np.std(data)
        # semi-variance
        semi_var_lows = [less_mu for less_mu in data if less_mu <= mu]
        semi_var_highs = [more_mu for more_mu in data if more_mu > mu]
        semi_var_lows = np.sum( ( semi_var_lows - mu)**2 )/len(semi_var_lows)
        semi_var_highs = np.sum( ( semi_var_highs - mu)**2 )/len(semi_var_highs)
        print ("Range: ", dist_range, '\n', 
               "Mean Abs Deviation: ", MAD, '\n', 
               "Variance: ", variance, '\n', 
               "Std Dev: ", std_dev, '\n',
               "Semivariance (downside var): ", semi_var_lows, '\n',
               "Semi-deviation (downside vol): ", np.sqrt(semi_var_lows), '\n',
               "Semivariance (upside var): ", semi_var_highs, '\n',
               "Semi-deviation (upside vol): ", np.sqrt(semi_var_highs))

    @staticmethod
    def random_numbers_test():

        hs = HackerStats()
        normal_dist_sample = hs.sample_normal_dist(0.0, 0.5, size=1000)
        hs.check_normality(normal_dist_sample)
        bins = hs.get_num_bins_hist(len(normal_dist_sample))
        hist, edges = np.histogram(normal_dist_sample, bins=bins, density=True)


if __name__ == '__main__':

    sm = StatisticalMoments()
    sm.random_numbers_test()
    ticker = 'T'
    start_date = '2016-01-01'
    end_date = '2017-01-01'
    pricing = sm.get_pricing(ticker = ticker, start_date = start_date, end_date = end_date)
    print(type(pricing))
    print(pricing.head())
    X = np.random.randint(100, size = 100)
    mu = np.mean(X)
    sm.calc_variance_metrics_example(data = X, semivar_cutoff=mu)
    StatisticalMoments.calc_skew_example()
    sm.calc_stock_return_skew(ticker='NFLX',
                              start_date='2015-01-01',
                              end_date='2016-01-01')
    sm.calc_stock_return_kurtosis(ticker='NFLX',
                                  start_date='2015-01-01',
                                  end_date='2016-01-01')
    sm.jarque_bera_calibration(0.05)
    sm.test_normality(ticker="SPY",
                      start_date='2012-01-01',
                      end_date='2015-01-01')
    print ("test in sample normaility of ticker AMC...")
    sm.test_normality(ticker="AMC",
                      start_date='2014-01-01',
                      end_date='2016-01-01')
    print ("...and now out of sample. Can we make the same conclusion about normality?")
    sm.test_normality(ticker="AMC",
                      start_date='2016-01-01',
                      end_date='2017-01-01')
    print("How about looking at rolling skew, to see how return distribution changes over time...")
    sm.rolling_skew_of_returns(ticker='AMC',
                               start_date='2015-01-01',
                               end_date='2017-01-01',
                               window_size=60)
def take_first(px_series):
    return px_series[0]

def take_last(px_series):
    return px_series[-1]
