import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.stattools import jarque_bera

from root.nested.dataAccess.tiingo_data_object import TiingoDataObject

class StatisticalMoments(object):
    """description of class"""

    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    @staticmethod
    def calc_skew_example(self):

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
    def calc_kurtosis_example(self):

        xs = np.linspace(-6, 6, 300) + 2
        Y = stats.cosine.pdf(xs)
        kurtosis = stats.kurtosis(Y)
        plt.hist(xs, Y)
        if (kurtosis < 0):
            print ("Excess Kurtosis is ", kurtosis, ".Because the excess kurtosis is negative, Y is platykurtic. Platykurtic distributions cluster around the mean, so large values in either direction are less likely.")
        elif (kurtosis > 0):
            print ("Excess Kurtosis is ", kurtosis, ".Because the excess kurtosis is positive, Y is leptokurtic. Leptokurtic distributions have fatter tails, meaning there is more tail risk.")

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

        returns = self.get_stock_returns(ticker, start_date, end_date)
        print ('Skew:', stats.skew(returns))
        print ('Mean:', np.mean(returns))
        print ('Median:', np.median(returns))
        plt.hist(returns, 30)
        plt.show()
        _, pvalue, _, _ = jarque_bera(returns)
        if (pvalue > 0.05):
            print ("jarques-bera test p-value of ", pvalue, " is within 5% error tolerance, and therefore the returns are likely normal.")
        else:
            print ("jarques-bera test p-value of ", pvalue, " is not within 5% error tolerance; returns are not likely normal.")


    def get_stock_returns(self,
                          ticker,
                          start_date = '2010-01-01',
                          end_date = str(pd.to_datetime('today')).split(' ')[0]):

        symbols = [ticker]
        source = 'Tiingo'
        mdo = TiingoDataObject(start_date = start_date, 
                               end_date = end_date, 
                               source = source, 
                               symbols = symbols)
        pricing_dict = mdo.get_px_data_df(start_date,
                                          end_date) # returned df will not include all dates up to, but not including, end_date
        pricing_df = pricing_dict[ticker] # dataframe containing the price data
        pricing = pricing_df['adjClose']
        returns = pricing.pct_change()[1:]

        return (returns)

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
        return (pricing)

    """ Calculate the kurtosis of the returns for the specificed ticker, with specificed
        start and end dates. 
    """
    def calc_stock_return_kurtosis(self,
                                   ticker,
                                   start_date = '2010-01-01',
                                   end_date = str(pd.to_datetime('today')).split(' ')[0]):

        returns = self.get_stock_returns(ticker, start_date, end_date)
        kurt = stats.kurtosis(returns)
        plt.hist(returns, 30)
        plt.show()
        if (kurt < 0):
            print ("Excess Kurtosis is ", kurt, ".Because the excess kurtosis is negative, Y is platykurtic. Platykurtic distributions cluster around the mean, meaning less tail risk.")
        elif (kurt > 0):
            print ("Excess Kurtosis is ", kurt, ".Because the excess kurtosis is positive, Y is leptokurtic. Leptokurtic distributions have fatter tails, meaning there is more tail risk.")

    """Calculate the skew of the returns for the specified ticker, with specificed
        start and end dates.
    """
    def calc_stock_return_skew(self,
                               ticker,
                               start_date='2010-01-01', # 2010-01-01 to present is max data offered by Tiingo
                               end_date=str(pd.to_datetime('today')).split(' ')[0]):  # str, e.g. "2011-01-01"

        returns = self.get_stock_returns(ticker, start_date, end_date)
        skew = stats.skew(returns)
        plt.hist(returns, 30)
        plt.show()
        print ('Skew: ', skew)
      
    """ Calculate a rolling skey, with window size "window_size", for the specified
        ticker with specified start and end dates.
    """
    def rolling_skew_of_returns(self,
                                ticker,
                                start_date,
                                end_date,
                                window_size):

        returns = self.get_stock_returns(ticker, start_date, end_date)
        roll_func = returns.rolling(window = window_size, center = False).skew()
        plt.plot(roll_func)
        plt.xlabel('Time')
        plt.ylabel(str(window_size) + " Period Rolling Skew")
        plt.show()

    """EXAMPLE function: Return the following types of variance metrics:
        Range, Mean Absolute Deviation, Variance, Standard Deviation, Semivariance,
        and Semideviation.
        Parameter "semivar_cutoff" specifies where in the data to split high and low. For example,
        if we want to calculate volatility for a stock separately for upside moves and downside moves,
        we would set the semivar_cutoff to 0.0 (where data is a numpy array of returns).
        
    """
    def calc_variance_metrics_example(self,
                                      data,
                                      semivar_cutoff):

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


sm = StatisticalMoments()
ticker = 'T'
start_date = '2016-01-01'
end_date = '2017-01-01'
pricing = sm.get_pricing(ticker = ticker, start_date = start_date, end_date = end_date)
print (type(pricing))
print (pricing.head())
X = np.random.randint(100, size = 100)
mu = np.mean(X)
sm.calc_variance_metrics_example(data = X, semivar_cutoff=mu)

sm.calc_skew_example()
sm.calc_stock_return_skew(ticker = 'NFLX',
                          start_date = '2015-01-01',
                          end_date = '2016-01-01')
sm.calc_stock_return_kurtosis(ticker = 'NFLX',
                              start_date = '2015-01-01',
                              end_date = '2016-01-01')
sm.jarque_bera_calibration(0.05)
sm.test_normality(ticker = "SPY", 
                  start_date = '2012-01-01', 
                  end_date = '2015-01-01')
print ("test in sample normaility of ticker AMC...")
sm.test_normality(ticker = "AMC",
                  start_date = '2014-01-01',
                  end_date = '2016-01-01')
print ("...and now out of sample. Can we make the same conclusion about normality?")
sm.test_normality(ticker = "AMC",
                  start_date = '2016-01-01',
                  end_date = '2017-01-01')
print("How about looking at rolling skew, to see how return distribution changes over time...")
sm.rolling_skew_of_returns(ticker = 'AMC', 
                           start_date = '2015-01-01', 
                           end_date = '2017-01-01',
                           window_size = 60)
