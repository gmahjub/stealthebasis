import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, date
import scipy.stats as stats
from statsmodels.stats.stattools import jarque_bera
from root.nested.dataAccess.tiingo_data_object import TiingoDataObject
from root.nested.statisticalAnalysis.hacker_stats import HackerStats
from root.nested import get_logger

pd.set_option('display.max_columns', 20)


class StatisticalMoments(object):
    """description of class"""

    LOGGER = get_logger()
    MAX_TICKER_PULL_SIZE = 10

    def __init__(self, **kwargs):

        return super().__init__(**kwargs)

    @staticmethod
    def calc_skew_example():

        xs2 = np.linspace(stats.gamma.ppf(0.01, 0.7, loc=-1), stats.gamma.ppf(0.99, 0.7, loc=-1), 150) + 1
        X = stats.gamma.pdf(xs2, 1.5)
        plt.plot(xs2, X, label='Skew Test')
        plt.legend()
        plt.show()
        skew = stats.skew(X)
        print("Skew: ", skew)
        if skew > 0:
            print('The distribution is positively skewed')
        elif skew < 0:
            print('The distribution is negatively skewed')
        else:
            print('The distribution is symetric')

    @staticmethod
    def calc_kurtosis_example():

        xs = np.linspace(-6, 6, 300) + 2
        Y = stats.cosine.pdf(xs)
        kurtosis = stats.kurtosis(Y)
        plt.hist(xs, Y)
        if (kurtosis < 0):
            print("Excess Kurtosis is", kurtosis,
                  ". Because the excess kurtosis is negative, Y is platykurtic. Platykurtic distributions cluster "
                  "around the mean, so large values in either direction are less likely.")
        elif (kurtosis > 0):
            print("Excess Kurtosis is", kurtosis,
                  ". Because the excess kurtosis is positive, Y is leptokurtic. Leptokurtic distributions have "
                  "fatter tails, meaning there is more tail risk.")

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
        print("% of samples that have a pvalue less than threshold : ", float(num_significant) * 100.0 / N, "%")

    def test_normality(self,
                       ticker,
                       start_date,
                       end_date):

        StatisticalMoments.LOGGER.info("StatisticalMoments.test_normality(): running function...")
        returns = self.get_stock_returns(ticker=ticker, start_date=start_date, end_date=end_date)
        StatisticalMoments.LOGGER.info('StatisticalMoments.test_normality(): skew is %f', stats.skew(returns))
        StatisticalMoments.LOGGER.info('StatisticalMoments.test_normality(): mean is %f', np.mean(returns))
        StatisticalMoments.LOGGER.info('StatisticalMoments.test_normality(): median is %f', np.median(returns))
        plt.hist(returns, 30)
        plt.show()
        _, pvalue, _, _ = jarque_bera(returns)
        if (pvalue > 0.05):
            StatisticalMoments.LOGGER.info('StatisticalMoments.test_normality(): '
                                           'jarques-bera test p-value of %f is within 5pct error tolerance, '
                                           'and therefore the returns are likely normal.', pvalue)
            print("jarques-bera test p-value of ", pvalue,
                  " is within 5% error tolerance, and therefore the returns are likely normal.")
        else:
            StatisticalMoments.LOGGER.info('StatisticalMoments.test_normality(): '
                                           'jarques-bera test p-value of %f is not within 5pct error tolerance, '
                                           'returns are not likely normal.', pvalue)
            print("jarques-bera test p-value of ", pvalue,
                  " is not within 5% error tolerance; returns are not likely normal.")

    def get_stock_returns(self,
                          ticker,
                          freq='D',
                          start_date='2010-01-01',
                          end_date=str(pd.to_datetime('today')).split(' ')[0],
                          px_type='adjClose'):

        StatisticalMoments.LOGGER.info('StatisticalMoments.get_stock_returns(): running function on ticker %s', ticker)
        symbols = [ticker]
        source = 'Tiingo'
        mdo = TiingoDataObject(start_date=start_date,
                               end_date=end_date,
                               source=source,
                               symbols=symbols)
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
                                freq='D',
                                start_date='2010-01-01',
                                end_date=str(pd.to_datetime('today')).split(' ')[0],
                                px_type='adjClose'):

        StatisticalMoments.LOGGER.info('StatisticalMoments.get_stock_excess_return(): running function...')
        symbols = [stock_ticker, benchmark_ticker]
        source = 'Tiingo'
        mdo = TiingoDataObject(start_date=start_date,
                               end_date=end_date,
                               source=source,
                               symbols=symbols)
        pricing_dict = mdo.get_px_data_df(start_date=start_date,
                                          end_date=end_date)
        stock_pricing_df = pricing_dict[stock_ticker]
        if freq is not 'D':
            stock_pricing_df = self.down_sample_daily_price_data(pricing=stock_pricing_df, to_freq=freq)
        benchmark_pricing_df = pricing_dict[benchmark_ticker]
        if freq is not 'D':
            benchmark_pricing_df = self.down_sample_daily_price_data(pricing=benchmark_pricing_df, to_freq=freq)
        stock_returns = np.log(1 + stock_pricing_df[px_type].pct_change())
        benchmark_returns = np.log(1 + benchmark_pricing_df[px_type].pct_change())
        excess_returns = stock_returns - benchmark_returns
        return excess_returns, benchmark_returns, stock_returns

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
        return (ticker_roll_rets - benchmark_roll_rets).dropna(), benchmark_roll_rets.dropna()

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
        rolling_returns = pricing.pct_change(periods=window_size)
        if shift_rets_series is True:
            rolling_returns = rolling_returns.shift(-1 * window_size)
        return rolling_returns.squeeze()

    """ get_pricing: main function to retrieve daily price data
        The source of this data is currently Tiingo. 
    """

    def get_pricing(self,
                    ticker,
                    freq='D',  # options are 'D', 'W', 'M'
                    start_date='2010-01-01',
                    end_date=str(pd.to_datetime('today')).split(' ')[0],
                    fields=['adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'close']):

        if (type(ticker) is not list):
            symbols = [ticker]
        else:
            symbols = ticker
        if len(symbols) > StatisticalMoments.MAX_TICKER_PULL_SIZE:
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
        # print(pricing_dict['SPY'].columns)
        return_dict = {}
        for ticker in symbols:
            pricing_df = pricing_dict[ticker]
            if freq is not 'D':
                pricing_df = self.down_sample_daily_price_data(pricing=pricing_df, to_freq=freq)
            pricing = pricing_df[fields]
            return_dict[ticker] = pricing
        return return_dict

    def linear_regression_on_price(self,
                                   tgt_ticker,
                                   tgt_df,
                                   benchmark_df=None,
                                   short_interval='1W-MON',
                                   long_interval='24W-MON'):
        """
        The key parameters are short_interval and long_interval. These are the intervals
        that we will calculate the linear regression on. We will split the data into those
        intervals.
        :param tgt_df:
        :param benchmark_df:
        :param short_interval:
        :param long_interval:
        :return:
        """
        long_interval_days = 5
        rolling_window = 40
        tgt_df.reset_index(inplace=True)
        short_int_returned_val = tgt_df.groupby(pd.Grouper(key='date', freq=short_interval)). \
            apply(self.get_price_line_regress, long_interval_days=long_interval_days)
        short_int_returned_val.rename(columns={'LinearModelSlope': "ShortIntervalLineRegress_Slope",
                                               'LinearModelIntercept': "ShortIntervalLineRegress_Intercept",
                                               'LinearModelSlope_95ci_Lower': "ShortIntervalLineRegress_Slope_95ci_l",
                                               'LinearModelSlope_95ci_Upper': "ShortIntervalLineRegress_Slope_95ci_u",
                                               'LinearModelIntercept_95ci_Lower':
                                                   "ShortIntervalLineRegress_Intercept_95ci_l",
                                               'LinearModelIntercept_95ci_Upper':
                                                   "ShortIntervalLineRegress_Intercept_95ci_u",
                                               'LinearModelInterval': 'ShortIntervalLineRegress_Interval',
                                               'LinearModel': 'ShortIntervalLinearModel',
                                               'OneStepPredict': "ShortIntervalOneStepPrediction"},
                                      inplace=True)
        # long_int_returned_val = tgt_df.groupby(pd.Grouper(key='date', freq=long_interval)). \
        #    apply(self.get_price_line_regress)
        # long_int_returned_val.rename(columns={'LinearModelSlope': "LongIntervalLineRegress_Slope",
        #                                      'LinearModelIntercept': "LongIntervalLineRegress_Intercept"},
        #                             inplace=True)
        tgt_df = tgt_df.set_index('date')
        total_df = pd.merge(left=tgt_df, right=short_int_returned_val, how='left', left_index=True, right_index=True)
        # print(total_df.ShortIntervalOneStepPrediction.tail(20))
        # total_df = pd.merge(left=total_df, right=long_int_returned_val, how='left', left_index=True, right_index=True)
        total_df.fillna(method='ffill', inplace=True)
        tgt_df.reset_index(inplace=True)
        total_df['X'] = tgt_df.index
        total_df['adjusted_X'] = tgt_df.index % 5
        total_df['adjusted_X'].replace(0, 5, inplace=True)
        #print(total_df.adjusted_X.head(50))
        #print(total_df.adjusted_X.tail(50))
        #exit()
        total_df.dropna(inplace=True)
        #total_df['ShortIntervalModelPrice'] = total_df.ShortIntervalLineRegress_Slope * total_df.X + \
        #                                      total_df.ShortIntervalLineRegress_Intercept
        total_df['ShortIntervalModelPrice'] = total_df.ShortIntervalLineRegress_Slope * total_df.adjusted_X + \
                                              total_df.ShortIntervalLineRegress_Intercept
        total_df['ShortIntervalModelPrice'] = total_df.ShortIntervalLineRegress_Slope * total_df.adjusted_X + total_df.the_dataAdjCloseMean
        total_df['ShortIntervalModel_IntervalPercDelta'] = total_df.ShortIntervalLineRegress_Interval / \
                                                           total_df.ShortIntervalModelPrice
        total_df['demeaned_adjClose'] = total_df.adjClose - total_df.the_dataAdjCloseMean
        #print(total_df.demeaned_adjClose)
        total_df[['demeaned_adjClose', 'ShortIntervalLineRegress_Intercept']].plot()
        plt.show()
        total_df['ShortIntervalModel_InterceptAdjClosePercDelta'] = \
            (total_df.ShortIntervalLineRegress_Intercept / total_df.demeaned_adjClose) - 1.0
        total_df['ShortIntervalModel_InterceptAdjCloseSpread'] = total_df.demeaned_adjClose - \
                                                                 total_df.ShortIntervalLineRegress_Intercept
        mean_intercept = total_df.ShortIntervalModel_InterceptAdjCloseSpread.mean()
        stdev_intercept = total_df.ShortIntervalModel_InterceptAdjCloseSpread.std()
        skew_intercept = total_df.ShortIntervalModel_InterceptAdjCloseSpread.skew()
        kurtosis_intercept = total_df.ShortIntervalModel_InterceptAdjCloseSpread.kurtosis()
        hist_title = "Mean IntAdjCloseSpread: " + str(mean_intercept) + " Stdev IntAdjCloseSpread: " + str(stdev_intercept) + \
                     " Skew IntAdjCloseSpread: " + str(skew_intercept) + " Kurtosis IntAdjCloseSpread: " + str(kurtosis_intercept)
        total_df.ShortIntervalModel_InterceptAdjCloseSpread.plot(title=hist_title)
        plt.show()
        total_df['InterAdjCloseSpread_mean'] = \
            total_df.ShortIntervalModel_InterceptAdjCloseSpread.rolling(window=rolling_window).mean()
        total_df['InterAdjCloseSpread_std'] = \
            total_df.ShortIntervalModel_InterceptAdjCloseSpread.rolling(window=rolling_window).std()
        total_df['InterAdjCloseCorr'] = total_df.ShortIntervalLineRegress_Intercept.rolling(window=rolling_window).\
            corr(total_df.demeaned_adjClose)
        mean_InterAdjCloseCorr = total_df.InterAdjCloseCorr.mean()
        stdev_InterAdjCloseCorr = total_df.InterAdjCloseCorr.std()
        skew_InterAdjCloseCorr = total_df.InterAdjCloseCorr.skew()
        hist_title = "Mean Corr: " + str(mean_InterAdjCloseCorr) + " Stdev Corr: " + \
                     str(stdev_InterAdjCloseCorr) + " Skew Corr: " + str(skew_InterAdjCloseCorr)
        total_df.InterAdjCloseCorr.hist()
        plt.suptitle(hist_title)
        plt.show()
        total_df['InterAdjCloseCorr_Change'] = total_df.InterAdjCloseCorr.diff(periods=1)
        mean_InterAdjCloseCorr_Change = total_df.InterAdjCloseCorr_Change.mean()
        stdev_InterAdjCloseCorr_Change = total_df.InterAdjCloseCorr_Change.std()
        skew_InterAdjCloseCorr_Change = total_df.InterAdjCloseCorr_Change.skew()
        hist_title = "Mean CorrChange: " + str(mean_InterAdjCloseCorr_Change) + " Stdev CorrChange: " + \
                     str(stdev_InterAdjCloseCorr_Change) + " Skew CorrChange: " + str(skew_InterAdjCloseCorr_Change)
        total_df.InterAdjCloseCorr_Change.hist()
        plt.suptitle(hist_title)
        plt.show()
        total_df['InterceptAdjCloseSpread_Sigmas'] = \
            (total_df.ShortIntervalModel_InterceptAdjCloseSpread -
             total_df.InterAdjCloseSpread_mean) / total_df.InterAdjCloseSpread_std
        total_df['InterAdjCloseCorr'].plot()
        total_df['InterceptAdjCloseSpread_Sigmas'].plot(secondary_y=True, style='g')
        plt.show()
        mean_intercept = total_df.ShortIntervalLineRegress_Intercept.mean()
        stdev_intercept = total_df.ShortIntervalLineRegress_Intercept.std()
        skew_intercept = total_df.ShortIntervalLineRegress_Intercept.skew()
        intercept_hist_title = "Mean Intercept: " + str(mean_intercept) + " Stdev Intercept: " + \
                               str(stdev_intercept) + " Skew Intercept: " + str(skew_intercept)
        total_df.ShortIntervalLineRegress_Intercept.hist()
        plt.suptitle(intercept_hist_title)
        plt.show()
        mean_interceptAdjClose_PercSpread = total_df.ShortIntervalModel_InterceptAdjClosePercDelta.mean()
        stdev_interceptAdjClose_PercSpread = total_df.ShortIntervalModel_InterceptAdjClosePercDelta.std()
        skew_interceptAdjClose_PercSpread = total_df.ShortIntervalModel_InterceptAdjClosePercDelta.skew()
        interceptAdjClose_PercSpread_hist_title = "Mean Intercept: " + str(mean_interceptAdjClose_PercSpread) + " Stdev Intercept: " + \
                               str(stdev_interceptAdjClose_PercSpread) + " Skew Intercept: " + \
                               str(skew_interceptAdjClose_PercSpread)
        total_df.ShortIntervalModel_InterceptAdjClosePercDelta.hist()
        plt.suptitle(interceptAdjClose_PercSpread_hist_title)
        plt.show()
        predicted_error_quantiles = total_df.ShortIntervalModel_IntervalPercDelta.\
            quantile([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]).to_dict()
        intercept_adjClose_quantiles = total_df.ShortIntervalModel_InterceptAdjClosePercDelta.\
            quantile([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]).to_dict()
        intercept_adjClose_sigmaBins = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        interAdjCloseCorr_bins = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
        interAdjCloseCorrChange_bins = [-0.135, -0.09, -0.045, 0.0, 0.045, 0.09, 0.135]
        interAdjCloseCorrChange_bins = [-0.123, -0.082, -0.041, 0.0, 0.041, 0.082, 0.123]
        interAdjCloseCorrChange_bins = [-0.047, 0.047]
        interAdjCloseCorrChange_bins = [-0.047]
        #intercept_adjClose_quantiles = total_df.ShortIntervalModel_InterceptAdjClosePercDelta. \
        #    quantile([0.6, 0.8]).to_dict()
        print("predicted error quantiles", predicted_error_quantiles)
        total_df['ShortIntervalModelErrorFlatPrice'] = total_df.adjClose - total_df.ShortIntervalModelPrice
        total_df['ShortIntervalModelErrorFlatPriceChange'] = total_df.ShortIntervalModelErrorFlatPrice.diff(periods=1)
        total_df['ShortIntervalModelPrice_95ul'] = total_df.ShortIntervalModelPrice + \
                                                   total_df.ShortIntervalLineRegress_Interval
        total_df['ShortIntervalModelPrice_95ll'] = total_df.ShortIntervalModelPrice - \
                                                   total_df.ShortIntervalLineRegress_Interval
        total_df["FwdAdjCloseDiff"] = total_df.adjClose.diff(periods=1).shift(periods=-1)
        #total_df[['adjClose', 'ShortIntervalModelPrice', 'ShortIntervalModelPrice_95ll',
        #          'ShortIntervalModelPrice_95ul', 'ShortIntervalLineRegress_Intercept']].plot()
        total_df[['adjClose', 'ShortIntervalModelPrice', 'ShortIntervalLineRegress_Intercept']].plot()
        print(total_df[['adjClose', 'ShortIntervalModelPrice', 'ShortIntervalModelPrice_95ll',
                        'ShortIntervalModelPrice_95ul', 'ShortIntervalLineRegress_Interval',
                        'ShortIntervalModelErrorFlatPriceChange', 'ShortIntervalLineRegress_Intercept']].tail())

        # apply entry trading (position) rule
        trade_rule_df = total_df.apply(self.price_line_regress_entry_trade_rule, axis=1)
        # apply exit trading (position) rule
        trade_rule_df = trade_rule_df.apply(self.price_line_regress_exit_trade_rule, axis=1)
        trade_rule_df.TradePnl.fillna(value=0.0, inplace=True)

        # parameters
        adjPriceAboveFairValueBy = 0.0
        adjPriceBelowFairValueBy = -0.0
        negativeSlopeDivergenceLower = 0.0
        negativeSlopeDivergenceUpper = 0.0

        print(len(total_df))
        print("------------------Stats when adjClose > every band level, Negative Slope---------------")
        print(total_df[(total_df.adjClose > total_df.ShortIntervalModelPrice_95ul) & (
                total_df.ShortIntervalLineRegress_Slope < 0.0)].FwdAdjCloseDiff.mean())
        print(total_df[(total_df.adjClose > total_df.ShortIntervalModelPrice_95ul) & (
                total_df.ShortIntervalLineRegress_Slope < 0.0)].FwdAdjCloseDiff.std())
        print(len(total_df[(total_df.adjClose > total_df.ShortIntervalModelPrice_95ul) & (
                total_df.ShortIntervalLineRegress_Slope < 0.0)].FwdAdjCloseDiff))
        print("------------------Stats when adjClose > every band level, Positive Slope---------------")
        print(total_df[(total_df.adjClose > total_df.ShortIntervalModelPrice_95ul) & (
                    total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff.mean())
        print(total_df[(total_df.adjClose > total_df.ShortIntervalModelPrice_95ul) & (
                    total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff.std())
        print(len(total_df[(total_df.adjClose > total_df.ShortIntervalModelPrice_95ul) & (
                    total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff))
        print("------------------Stats when adjClose < every band level, Negative Slope---------------")
        print(total_df[(total_df.adjClose < total_df.ShortIntervalModelPrice_95ll) & (
                total_df.ShortIntervalLineRegress_Slope < 0.0)].FwdAdjCloseDiff.mean())
        print(total_df[(total_df.adjClose < total_df.ShortIntervalModelPrice_95ll) & (
                total_df.ShortIntervalLineRegress_Slope < 0.0)].FwdAdjCloseDiff.std())
        print(len(total_df[(total_df.adjClose < total_df.ShortIntervalModelPrice_95ll) & (
                total_df.ShortIntervalLineRegress_Slope < 0.0)].FwdAdjCloseDiff))
        print("------------------Stats when adjClose < every band level, Positive Slope---------------")
        print(total_df[(total_df.adjClose < total_df.ShortIntervalModelPrice_95ll) & (
                total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff.mean())
        print(total_df[(total_df.adjClose < total_df.ShortIntervalModelPrice_95ll) & (
                total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff.std())
        print(len(total_df[(total_df.adjClose < total_df.ShortIntervalModelPrice_95ll) & (
                total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff))
        print("------------------Stats when adjClose > Intercept, Positive Slope---------------")
        print(total_df[(total_df.adjClose > total_df.ShortIntervalLineRegress_Intercept) & (
                total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff.mean())
        print(total_df[(total_df.adjClose > total_df.ShortIntervalLineRegress_Intercept) & (
                total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff.std())
        print(len(total_df[(total_df.adjClose > total_df.ShortIntervalLineRegress_Intercept) & (
                total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff))
        print("------------------Stats when adjClose > Intercept, Negative Slope---------------")
        print(total_df[(total_df.adjClose > total_df.ShortIntervalLineRegress_Intercept) & (
                total_df.ShortIntervalLineRegress_Slope < 0.0)].FwdAdjCloseDiff.mean())
        print(total_df[(total_df.adjClose > total_df.ShortIntervalLineRegress_Intercept) & (
                total_df.ShortIntervalLineRegress_Slope < 0.0)].FwdAdjCloseDiff.std())
        print(len(total_df[(total_df.adjClose > total_df.ShortIntervalLineRegress_Intercept) & (
                total_df.ShortIntervalLineRegress_Slope < 0.0)].FwdAdjCloseDiff))
        print("------------------Stats when adjClose < Intercept, Positive Slope---------------")
        print(total_df[(total_df.adjClose < total_df.ShortIntervalLineRegress_Intercept) & (
                total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff.mean())
        print(total_df[(total_df.adjClose < total_df.ShortIntervalLineRegress_Intercept) & (
                total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff.std())
        print(len(total_df[(total_df.adjClose < total_df.ShortIntervalLineRegress_Intercept) & (
                total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff))
        print("------------------Stats when adjClose < Intercept, Negative Slope---------------")
        print(total_df[(total_df.adjClose < total_df.ShortIntervalLineRegress_Intercept) & (
                total_df.ShortIntervalLineRegress_Slope < 0.0)].FwdAdjCloseDiff.mean())
        print(total_df[(total_df.adjClose < total_df.ShortIntervalLineRegress_Intercept) & (
                total_df.ShortIntervalLineRegress_Slope < 0.0)].FwdAdjCloseDiff.std())
        print(len(total_df[(total_df.adjClose < total_df.ShortIntervalLineRegress_Intercept) & (
                total_df.ShortIntervalLineRegress_Slope < 0.0)].FwdAdjCloseDiff))

        total_df_pos_slope = total_df[total_df.ShortIntervalLineRegress_Slope > 0.0]
        total_df_neg_slope = total_df[total_df.ShortIntervalLineRegress_Slope < 0.0]

        last_corr_change_value = -100.0
        interAdjCloseCorrChange_bins.append(100.0)
        for corr_change_value in interAdjCloseCorrChange_bins:
            mean = total_df[(total_df.InterAdjCloseCorr_Change < float(corr_change_value)) &
                            (total_df.InterAdjCloseCorr_Change > float(
                                last_corr_change_value)) & (total_df.InterAdjCloseCorr > -0.17)].FwdAdjCloseDiff.mean()
            stdvar = total_df[(total_df.InterAdjCloseCorr_Change < float(corr_change_value)) &
                              (total_df.InterAdjCloseCorr_Change > float(
                                  last_corr_change_value)) & (total_df.InterAdjCloseCorr > -0.17)].FwdAdjCloseDiff.std()
            sample = len(total_df[(total_df.InterAdjCloseCorr_Change < float(corr_change_value)) &
                                  (total_df.InterAdjCloseCorr_Change > float(
                                      last_corr_change_value)) & (total_df.InterAdjCloseCorr > -0.17)])
            print("Intercept-AdjClose CorrChange Analysis", corr_change_value, last_corr_change_value, mean, stdvar, sample,
                  mean / stdvar * np.sqrt(sample))
            last_corr_change_value = corr_change_value
        last_corr_value = -1.0
        interAdjCloseCorr_bins.append(1.0)
        for corr_value in interAdjCloseCorr_bins:
            mean = total_df[(total_df.InterAdjCloseCorr < float(corr_value)) &
                            (total_df.InterAdjCloseCorr > float(
                                last_corr_value))].FwdAdjCloseDiff.mean()
            stdvar = total_df[(total_df.InterAdjCloseCorr < float(corr_value)) &
                            (total_df.InterAdjCloseCorr > float(
                                last_corr_value))].FwdAdjCloseDiff.std()
            sample = len(total_df[(total_df.InterAdjCloseCorr < float(corr_value)) &
                            (total_df.InterAdjCloseCorr > float(
                                last_corr_value))])
            print("Intercept-AdjClose Corr Analysis", corr_value, last_corr_value, mean, stdvar, sample,
                  mean / stdvar * np.sqrt(sample))
            last_corr_value = corr_value
        last_sigma_value = -1000.0
        intercept_adjClose_sigmaBins.append(1000.0)
        for sigma_value in intercept_adjClose_sigmaBins:
            # total_df['InterceptAdjCloseSpread_Sigmas']
            mean_corr = total_df[(total_df.InterceptAdjCloseSpread_Sigmas < float(sigma_value)) &
                            (total_df.InterceptAdjCloseSpread_Sigmas > float(
                                last_sigma_value))].InterAdjCloseCorr.mean()
            mean = total_df[(total_df.InterceptAdjCloseSpread_Sigmas < float(sigma_value)) &
                            (total_df.InterceptAdjCloseSpread_Sigmas > float(
                                last_sigma_value))].FwdAdjCloseDiff.mean()
            stdvar = total_df[(total_df.InterceptAdjCloseSpread_Sigmas < float(sigma_value)) &
                              (total_df.InterceptAdjCloseSpread_Sigmas > float(
                                  last_sigma_value))].FwdAdjCloseDiff.std()
            sample = len(total_df[(total_df.InterceptAdjCloseSpread_Sigmas < float(sigma_value)) &
                                  (total_df.InterceptAdjCloseSpread_Sigmas > float(
                                      last_sigma_value))])
            print("Intercept-AdjClose Sigmas Analysis", sigma_value, last_sigma_value, mean, stdvar, sample,
                  mean / stdvar * np.sqrt(sample), mean_corr)
            last_sigma_value = sigma_value
        exit()
        last_value = -1000.0
        intercept_adjClose_quantiles[1.0] = 1000.0
        for key, value in intercept_adjClose_quantiles.items():
            #value = 0.0125
            #last_value = -0.005
            #from scipy import stats
            #print(value, stats.percentileofscore(total_df.ShortIntervalModel_InterceptAdjClosePercDelta.array, value))
            #print(last_value, stats.percentileofscore(total_df.ShortIntervalModel_InterceptAdjClosePercDelta.array, last_value))
            mean = total_df[(total_df.ShortIntervalModel_InterceptAdjClosePercDelta < float(value)) &
                            (total_df.ShortIntervalModel_InterceptAdjClosePercDelta > float(last_value))].FwdAdjCloseDiff.mean()
            stdvar = total_df[(total_df.ShortIntervalModel_InterceptAdjClosePercDelta < float(value)) &
                              (total_df.ShortIntervalModel_InterceptAdjClosePercDelta > float(last_value))].FwdAdjCloseDiff.std()
            sample = len(total_df[(total_df.ShortIntervalModel_InterceptAdjClosePercDelta < float(value)) &
                              (total_df.ShortIntervalModel_InterceptAdjClosePercDelta > float(
                                  last_value))])
            print("Intercept-AdjClose Analysis", key, value, last_value, mean, stdvar, sample, mean/stdvar*np.sqrt(sample))
            last_value = value
            #break
        last_value = 0.0
        predicted_error_quantiles[1.0] = 1.0
        cnt = 0
        results_dict = dict()
        for key, value in predicted_error_quantiles.items():
            print("------------------Size of Error Bands, quantile " + str(key) + "---------------")
            mean = total_df[(total_df.ShortIntervalModel_IntervalPercDelta < float(value)) &
                           (total_df.ShortIntervalModel_IntervalPercDelta > float(last_value))].FwdAdjCloseDiff.mean()
            mean_pos_slope = total_df_pos_slope[(total_df_pos_slope.ShortIntervalModel_IntervalPercDelta < float(value)) &
                           (total_df_pos_slope.ShortIntervalModel_IntervalPercDelta > float(last_value))].FwdAdjCloseDiff.mean()
            mean_neg_slope = total_df_neg_slope[
                (total_df_neg_slope.ShortIntervalModel_IntervalPercDelta < float(value)) &
                (total_df_neg_slope.ShortIntervalModel_IntervalPercDelta > float(last_value))].FwdAdjCloseDiff.mean()
            stdvar = total_df[(total_df.ShortIntervalModel_IntervalPercDelta < float(value)) &
                           (total_df.ShortIntervalModel_IntervalPercDelta > float(last_value))].FwdAdjCloseDiff.std()
            stdvar_pos_slope = total_df_pos_slope[(total_df_pos_slope.ShortIntervalModel_IntervalPercDelta < float(value)) &
                              (total_df_pos_slope.ShortIntervalModel_IntervalPercDelta > float(last_value))].FwdAdjCloseDiff.std()
            stdvar_neg_slope = total_df_neg_slope[
                (total_df_neg_slope.ShortIntervalModel_IntervalPercDelta < float(value)) &
                (total_df_neg_slope.ShortIntervalModel_IntervalPercDelta > float(last_value))].FwdAdjCloseDiff.std()
            sample = len(total_df[(total_df.ShortIntervalModel_IntervalPercDelta < float(value)) &
                               (total_df.ShortIntervalModel_IntervalPercDelta > float(last_value))].FwdAdjCloseDiff)
            sample_pos_slope = len(total_df_pos_slope[(total_df_pos_slope.ShortIntervalModel_IntervalPercDelta < float(value)) &
                                  (total_df_pos_slope.ShortIntervalModel_IntervalPercDelta > float(last_value))].FwdAdjCloseDiff)
            sample_neg_slope = len(total_df_neg_slope[(total_df_neg_slope.ShortIntervalModel_IntervalPercDelta < float(value)) &
                                   (total_df_neg_slope.ShortIntervalModel_IntervalPercDelta > float(
                                       last_value))].FwdAdjCloseDiff)
            sharpe = mean/stdvar*np.sqrt(sample)
            sharpe_neg_slope = mean_neg_slope/stdvar_neg_slope*np.sqrt(sample_neg_slope)
            sharpe_pos_slope = mean_pos_slope/stdvar_pos_slope*np.sqrt(sample_pos_slope)
            print(mean, stdvar, sample, sharpe)
            print(mean_neg_slope, stdvar_neg_slope, sample_neg_slope, sharpe_neg_slope)
            print(mean_pos_slope, stdvar_pos_slope, sample_pos_slope, sharpe_pos_slope)
            value_dict = {'mean': mean, 'mean_pos_slope': mean_pos_slope, 'mean_neg_slope': mean_neg_slope,
                          'stdvar': stdvar, 'stdvar_pos_slope': stdvar_pos_slope, 'stdvar_neg_slope': stdvar_neg_slope,
                          'sample': sample, 'sample_pos_slope': sample_pos_slope, 'sample_neg_slope': sample_neg_slope,
                          'sharpe': sharpe, 'sharpe_pos_slope': sharpe_pos_slope, 'sharpe_neg_slope': sharpe_neg_slope,
                          'param_low_range': last_value, 'param_high_range': value}
            results_dict[cnt] = value_dict
            last_value = value
            cnt += 1
        results_df = pd.DataFrame(data=results_dict).T
        results_df.set_index('param_low_range', inplace=True)
        print(results_df.head(20))
        results_df[['sharpe', 'sharpe_pos_slope', 'sharpe_neg_slope']].plot(style='o-',
                                                                            title='Sharpe Ratio - ' + tgt_ticker + ' - LR on Px')
        plt.axhline(y=0.0, color='black')
        results_df[['stdvar', 'stdvar_pos_slope', 'stdvar_neg_slope']].plot(style='o-',
                                                                            title='Std Var Rets - ' + tgt_ticker + ' - LR on Px')
        plt.axhline(y=0.0, color='black')
        results_df[['mean', 'mean_pos_slope', 'mean_neg_slope']].plot(style='o-',
                                                                      title='Mean Rets - ' + tgt_ticker + ' - LR on Px')
        plt.axhline(y=0.0, color='black')
        results_df[['sample', 'sample_pos_slope', 'sample_neg_slope']].plot(style='o-',
                                                                            title='Sample Size ' + tgt_ticker + ' - LR on Px')
        plt.axhline(y=0.0, color='black')
        plt.show()
        plt.show()
        plt.show()
        plt.show()


        #print("------------------Stats when adjClose < every band level, Positive Slope---------------")
        #print(total_df[(total_df.adjClose < total_df.ShortIntervalModelPrice_95ll) & (
        #        total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff.mean())
        #print(total_df[(total_df.adjClose < total_df.ShortIntervalModelPrice_95ll) & (
        #        total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff.std())
        #print(len(total_df[(total_df.adjClose < total_df.ShortIntervalModelPrice_95ll) & (
        #        total_df.ShortIntervalLineRegress_Slope > 0.0)].FwdAdjCloseDiff))

        """
        print(trade_rule_df[(trade_rule_df.TradePnl != 0.0) &
                            (trade_rule_df.ShortIntervalModelErrorFlatPrice > adjPriceAboveFairValueBy)].TradePnl.mean())
        print(trade_rule_df[(trade_rule_df.TradePnl != 0.0) &
                            (trade_rule_df.ShortIntervalModelErrorFlatPrice > adjPriceAboveFairValueBy)].TradePnl.std())
        print(len(trade_rule_df[(trade_rule_df.TradePnl != 0.0) &
                                (trade_rule_df.ShortIntervalModelErrorFlatPrice > adjPriceAboveFairValueBy)]))
        print(trade_rule_df[(trade_rule_df.TradePnl != 0.0) &
                            (trade_rule_df.ShortIntervalModelErrorFlatPriceChange < adjPriceBelowFairValueBy)].TradePnl.mean())
        print(trade_rule_df[(trade_rule_df.TradePnl != 0.0) &
                            (trade_rule_df.ShortIntervalModelErrorFlatPriceChange < adjPriceBelowFairValueBy)].TradePnl.std())
        print(len(trade_rule_df[(trade_rule_df.TradePnl != 0.0) &
                                (trade_rule_df.ShortIntervalModelErrorFlatPriceChange < adjPriceBelowFairValueBy)]))
        """

        #total_df.ShortIntervalModelErrorFlatPrice.hist()
        #plt.show()
        #total_df.ShortIntervalModel_IntervalPercDelta.hist()
        #plt.show()

    def price_line_regress_entry_trade_rule(self, row):
        if row['ShortIntervalLineRegress_Slope'] < 0.0 and row['adjClose'] > row['ShortIntervalModelPrice_95ul']:
            row['TradeType'] = -1
            row['TradePrice'] = row['adjClose']
        elif row['ShortIntervalLineRegress_Slope'] < 0.0 and row['adjClose'] < row['ShortIntervalModelPrice_95ll']:
            row['TradeType'] = 1
            row['TradePrice'] = row['adjClose']
        #elif row['ShortIntervalLineRegress_Slope'] > 0.0 and row['adjClose'] > row['ShortIntervalModelPrice_95ul']:
        #    row['TradeType'] = 1
        #    row['TradePrice'] = row['adjClose']
        #elif row['ShortIntervalLineRegress_Slope'] > 0.0 and row['adjClose'] < row['ShortIntervalModelPrice_95ll']:
        #    row['TradeType'] = -1
        #    row['TradePrice'] = row['adjClose']
        return row

    def price_line_regress_exit_trade_rule(self, row):
        if row['TradeType'] == -1:
            row['TradePnl'] = -1.0*row['FwdAdjCloseDiff']
        elif row['TradeType'] == 1:
            row['TradePnl'] = row['FwdAdjCloseDiff']
        return row

    def get_price_line_regress(self, the_data, long_interval_days):
        # the_data is a dataframe, with the index reset, with columns ['date', 'adjClose']
        from statsmodels import regression
        from statsmodels.sandbox.regression.predstd import wls_prediction_std
        from statsmodels.stats.outliers_influence import summary_table
        the_data_demeaned = the_data.adjClose - the_data.adjClose.mean()
        if len(the_data) < 3:
            return
        first_index = the_data.index[0]
        y_int_adjust = 0
        if first_index > long_interval_days:
            y_int_adjust = first_index - long_interval_days
        adjusted_index = the_data.index - y_int_adjust
        x_data_const = sm.add_constant(adjusted_index)
        one_step_predict_idx = [[1.0, x_data_const[-1][1] + float(x)] for x in range(1, 5)]
        #lr_model = regression.linear_model.OLS(the_data.adjClose, x_data_const).fit()
        lr_model = regression.linear_model.OLS(the_data_demeaned, x_data_const).fit()
        #sum_errs = np.sum((the_data.adjClose - lr_model.fittedvalues) ** 2)
        sum_errs = np.sum((the_data_demeaned - lr_model.fittedvalues) ** 2)
        #stdev = np.sqrt(1 / (len(the_data.adjClose) - 2) * sum_errs)
        stdev = np.sqrt(1 / (len(the_data_demeaned) - 2) * sum_errs)
        z_score = stats.norm.ppf(1 - (0.05 / 2))
        interval = z_score * stdev
        ci_df = lr_model.conf_int(0.05)
        prstd, iv_l, iv_u = wls_prediction_std(lr_model)
        #fig, ax = plt.subplots(figsize=(8,6))
        #ax.plot(adjusted_index, the_data.adjClose, 'o', label='AdjClose')
        #ax.plot(adjusted_index, lr_model.fittedvalues, 'r--', label='OLS')
        #ax.plot(adjusted_index, iv_u, 'r--')
        #ax.plot(adjusted_index, iv_l, 'r--')
        #ax.legend(loc='best')
        #plt.show()
        return_val = pd.Series()
        return_val['LinearModelMSE'] = sum_errs
        return_val['LinearModelStdDev'] = stdev
        return_val['LinearModelInterval'] = interval
        return_val['LinearModelSlope_95ci_Lower'] = ci_df.loc['x1', 0]
        return_val['LinearModelSlope_95ci_Upper'] = ci_df.loc['x1', 1]
        return_val['LinearModelIntercept_95ci_Lower'] = ci_df.loc['const', 0]
        return_val['LinearModelIntercept_95ci_Upper'] = ci_df.loc['const', 1]
        return_val['LinearModelSlope'] = lr_model.params.get('x1')
        return_val['LinearModelIntercept'] = lr_model.params.get('const') #+ the_data.adjClose.mean()*lr_model.params.get('x1')
        return_val['LinearModel'] = lr_model
        return_val['OneStepPredict'] = lr_model.predict(one_step_predict_idx)
        return_val['the_dataAdjCloseMean'] = the_data.adjClose.mean()
        #print(return_val['OneStepPredict'])
        return return_val

    def get_linear_regression(self,
                              df,
                              colnm_x,
                              colnm_y):

        pd.set_option('display.max_columns', 7)
        df = sm.add_constant(df)
        # reg1 = sm.OLS(endog=df.StockFwdLogRets, exog=df[['const', 'BenchmarkFwdLogRets']], missing='drop')
        print(df.head())
        print(df.columns)
        reg1 = sm.OLS(endog=df[colnm_y], exog=df[['const', colnm_x]], missing='drop')
        results = reg1.fit()
        # X = df['BenchmarkFwdLogRets']
        X = df[colnm_x]
        y = df[colnm_y]
        labels = df.index
        # Replace markers with country labels
        fig, ax = plt.subplots()
        # ax.scatter(X, y, marker='')
        ax.plot(X, y, 'o')
        # for i, label in enumerate(labels):
        #    ax.annotate(label, (X.iloc[i], y.iloc[i]))
        # Fit a linear trend line
        np_poly_slope, np_poly_intercept = np.polyfit(X, y, 1)
        # print ("from np polyfit for chart", slope, intercept)
        ax.plot(X, np_poly_slope * X + np_poly_intercept, color='black')
        # ax.set_xlim([3.3, 10.5])
        # ax.set_ylim([4, 10.5])
        # ax.set_xlabel('Stock Return')
        # ax.set_ylabel('Benchmark Return')
        # ax.set_title('Figure 2: Stock vs. Benchmark')
        # plt.show()
        sm_ols_y_intercept = results.params['const']
        # sm_ols_slope = results.params['BenchmarkFwdLogRets']
        sm_ols_slope = results.params[colnm_x]
        # print ("from results.params slope ", slope)
        # print ("from results.params intercept ", y_intercept)
        r_squared = results.rsquared
        # print ("R_squared is ", r_squared)
        # print ("Correlation is ", np.sqrt(r_squared))
        # spearman_rank_correl = df[['StockFwdLogRets', 'BenchmarkFwdLogRets']].corr(method='spearman')
        spearman_rank_correl = df[[colnm_y, colnm_x]].corr(method='spearman')
        # pearson_correl = df[['StockFwdLogRets', 'BenchmarkFwdLogRets']].corr(method='pearson')
        pearson_correl = df[[colnm_y, colnm_x]].corr(method='pearson')
        # print("Spearman Rank Correlation is ", spearman_rank_correl)
        # print("Pearson Correlation is ", pearson_correl)
        return_series = pd.Series()
        return_series['Pearson'] = pearson_correl
        return_series['Spearman'] = spearman_rank_correl
        return_series['r_squared'] = r_squared
        return_series['r_squared_to_slope'] = np.sqrt(r_squared)
        return_series['sm.ols.slope'] = sm_ols_slope
        return_series['sm.ols.intercept'] = sm_ols_y_intercept
        return_series['np.poly.slope'] = np_poly_slope
        return_series['np.poly.intercept'] = np_poly_intercept
        return return_series

    def get_rolling_price_linear_regression(self,
                                            df,
                                            window_size,
                                            target_name):
        """
        The point of this analysis is to see if we can identify recent trend via the slope
        of the price movement over some rolling window time period.
        We will make this a seperate page from the ATR analysis on the website.
        :param df:
        :param window_size:
        :param target_name:
        :return:
        """

    def get_rolling_linear_regression(self,
                                      df,
                                      window_size,
                                      target_name,
                                      hedge_name,
                                      autocorr_periods=0):
        """
        when autocorr_periods is greater than 2, we will take the lag of the hedge against the current of the target
        :param df:
        :param window_size:
        :param target_name:
        :param hedge_name:
        :param autocorr_periods:
        :return:
        """
        from statsmodels.regression.rolling import RollingOLS
        df_lr = sm.add_constant(df)
        df_lr[target_name + 'Rank'] = df_lr[target_name].rank()
        df_lr[hedge_name + 'Rank'] = df_lr[hedge_name].rank()
        if autocorr_periods > 2:
            for lag_p in range(1, autocorr_periods):
                df_lr['SpearmanCorr_hedge_lag' + str(lag_p)] = df_lr[target_name].rank().rolling(window=window_size). \
                    corr(df_lr[hedge_name].rank().shift(-lag_p))
                df_lr['SpearmanCorr_tgt_lag' + str(lag_p)] = df_lr[hedge_name].rank().rolling(window=window_size). \
                    corr(df_lr[target_name].rank().shift(-lag_p))
                df_lr['PearsonCorr_hedge_lag' + str(lag_p)] = df_lr[target_name]. \
                    rolling(window=window_size).corr(other=df_lr[hedge_name].shift(-lag_p))
                df_lr['PearsonCorr_tgt_lag' + str(lag_p)] = df_lr[hedge_name]. \
                    rolling(window=window_size).corr(other=df_lr[target_name].shift(-lag_p))
                model_hedge_lagp = RollingOLS(endog=df_lr[target_name].values,
                                              exog=df_lr[['const', hedge_name]].shift(-lag_p),
                                              window=window_size)
                model_tgt_lagp = RollingOLS(endog=df_lr[hedge_name].values,
                                            exog=df_lr[['const', target_name]].shift(-lag_p),
                                            window=window_size)
                rres_hedge_lagp = model_hedge_lagp.fit()
                rres_tgt_lagp = model_tgt_lagp.fit()
                intercept_lagp = rres_hedge_lagp.params['const']
                slope_lagp = rres_hedge_lagp.params[hedge_name]
                r_squared_lagp = rres_hedge_lagp.rsquared
                df_lr['intercept_hedge_lag' + str(lag_p)] = intercept_lagp
                df_lr['interecept_tgt_lap' + str(lag_p)] = rres_tgt_lagp.params['const']
                df_lr['slope_hedge_lag' + str(lag_p)] = slope_lagp
                df_lr['slope_tgt_lag' + str(lag_p)] = rres_tgt_lagp.params[target_name]
                df_lr['r_squared_hedge_lag' + str(lag_p)] = r_squared_lagp
                df_lr['r_squared_tgt_lag' + str(lag_p)] = rres_tgt_lagp.rsquared
        model = RollingOLS(endog=df_lr[target_name].values,
                           exog=df_lr[['const', hedge_name]],
                           window=window_size)
        rres = model.fit()
        intercept = rres.params['const']
        slope = rres.params[hedge_name]
        r_squared = rres.rsquared
        df_lr['SpearmanCorr'] = df_lr[target_name + 'Rank'].rolling(window=window_size).corr(
            df_lr[hedge_name + 'Rank'])
        df_lr['PearsonCorr'] = df_lr[target_name]. \
            rolling(window=window_size).corr(other=df_lr[hedge_name])
        df_lr['r_squared'] = r_squared
        df_lr['intercept'] = intercept
        df_lr['slope'] = slope
        df_lr['linreg_f_stat_p_val'] = rres.f_pvalue
        p_val_colnames = ['intercept_p_val', 'slope_p_val']
        arrOfArr = np.split(rres.pvalues, 2, axis=1)
        for i in range(len(p_val_colnames)):
            b = np.array(arrOfArr[i]).flatten()
            c = pd.Series(b, index=df_lr.index)
            c.dropna(inplace=True)
            df_lr[p_val_colnames[i]] = c
        fig = rres.plot_recursive_coefficient(variables=[hedge_name], figsize=(14, 6))
        plt.show()
        df_lr = df_lr.drop(columns=[target_name + 'Rank', hedge_name + 'Rank', 'const'], axis=1).dropna()
        return df_lr

    def get_average_true_range(self,
                               ticker,
                               freq='D',
                               window_size=14,
                               start_date='2010-01-01',
                               end_date=str(pd.to_datetime('today')).split(' ')[0]):

        stock_data = self.get_pricing(ticker=ticker, freq=freq, start_date=start_date,
                                      end_date=end_date)[ticker]
        high_minus_low = stock_data['adjHigh'] - stock_data['adjLow']
        high_minus_low = high_minus_low / stock_data['adjClose'].shift(periods=1)
        abs_high_prevClose = (stock_data['adjHigh'] - stock_data['adjClose'].shift(periods=1)).abs()
        abs_high_prevClose = abs_high_prevClose / stock_data['adjClose'].shift(periods=1)
        abs_low_prevClose = (stock_data['adjLow'] - stock_data['adjClose'].shift(periods=1)).abs()
        abs_low_prevClose = abs_low_prevClose / stock_data['adjClose'].shift(periods=1)
        df = pd.concat([stock_data['adjClose'], high_minus_low, abs_high_prevClose, abs_low_prevClose], axis=1)
        df.dropna(inplace=True)
        df.columns = ['AdjClose', 'High-Low', 'Abs(High-PrevClose)', 'Abs(Low-PrevClose)']
        df['Max_Of_All'] = df[['High-Low', 'Abs(High-PrevClose)', 'Abs(Low-PrevClose)']].max(axis=1)
        df['ATR'] = df['Max_Of_All'].rolling(window=window_size).mean()
        df['expATR'] = df['Max_Of_All'].ewm(span=window_size, adjust=False).mean()
        return df

    def get_atr_spread(self,
                       stock_atr_df,
                       benchmark_atr_df):
        linear_regress_window = 12
        atr_spread = stock_atr_df.ATR - benchmark_atr_df.ATR
        emaAtr_spread = stock_atr_df.expATR - benchmark_atr_df.expATR
        fairValue_emaAtr_spread = emaAtr_spread.ewm(span=9, adjust=False).mean()
        recent_vol_higher_lower = (emaAtr_spread - fairValue_emaAtr_spread) / emaAtr_spread.rolling(window=9).std()
        return_df = pd.concat([atr_spread, emaAtr_spread, fairValue_emaAtr_spread, recent_vol_higher_lower], axis=1)
        return_df.dropna(inplace=True)
        atr_columns = ['atr_spread', 'emaAtr_spread', 'fairValue_emaAtr_spread', 'recentVol_emaAtr_diff_Atr']
        return_df.columns = atr_columns
        return_df['ZeroLine'] = 0.0
        # return_df[['atr_spread', 'emaAtr_spread']].plot(title="ATR spread Plot")
        # plt.show()
        # return_df[['emaAtr_spread', 'fairValue_emaAtr_spread']].plot(title='Ema Atr Spread and Fair Value')
        # plt.show()
        # return_df[['recentVol_emaAtr_diff_Atr', 'ZeroLine']].plot(
        #    title='EMA-Simple Mean Spread, Recent Vol Higher or Lower?')
        # plt.show()
        stock_log_rets = np.log(1 + stock_atr_df.AdjClose.pct_change(periods=1).dropna())
        benchmark_log_rets = np.log(1 + benchmark_atr_df.AdjClose.pct_change(periods=1).dropna())
        stock_fwd_returns = stock_atr_df.AdjClose.pct_change(periods=1).shift(-1).dropna()
        stock_fwd_log_rets = np.log(1 + stock_atr_df.AdjClose.pct_change(periods=1)).shift(-1).dropna()
        benchmark_fwd_returns = benchmark_atr_df.AdjClose.pct_change(periods=1).shift(-1).dropna()
        benchmark_fwd_log_rets = np.log(1 + benchmark_atr_df.AdjClose.pct_change(periods=1)).shift(-1).dropna()
        excess_fwd_return = stock_fwd_returns - benchmark_fwd_returns
        excess_fwd_log_rets = stock_fwd_log_rets - benchmark_fwd_log_rets
        stock_fwd_returns = stock_fwd_log_rets
        benchmark_fwd_returns = benchmark_fwd_log_rets
        excess_fwd_return = excess_fwd_log_rets
        abs_val_excess_fwd_return = excess_fwd_log_rets.abs()

        df_lr = pd.concat([stock_fwd_log_rets, benchmark_fwd_log_rets], axis=1)
        df_lr.columns = ['StockFwdLogRets', 'BenchmarkFwdLogRets']
        df_fwdFwd = self.get_rolling_linear_regression(df_lr,
                                                       target_name='StockFwdLogRets',
                                                       hedge_name='BenchmarkFwdLogRets',
                                                       window_size=linear_regress_window)
        df_lr = pd.concat([stock_log_rets, benchmark_fwd_log_rets], axis=1)
        df_lr.columns = ['StockLogRets', 'BenchmarkFwdLogRets']
        df_currFwd = self.get_rolling_linear_regression(df_lr,
                                                        target_name='StockLogRets',
                                                        hedge_name='BenchmarkFwdLogRets',
                                                        window_size=linear_regress_window)
        df_lr = pd.concat([stock_fwd_log_rets, benchmark_log_rets], axis=1)
        df_lr.columns = ['StockFwdLogRets', 'BenchmarkLogRets']
        df_fwdCurr = self.get_rolling_linear_regression(df_lr,
                                                        target_name='BenchmarkLogRets',
                                                        hedge_name='StockFwdLogRets',
                                                        window_size=linear_regress_window)
        df_lr = pd.concat([stock_log_rets, benchmark_log_rets], axis=1)
        df_lr.columns = ['StockLogRets', 'BenchmarkLogRets']
        df_currCurr = self.get_rolling_linear_regression(df_lr,
                                                         target_name='StockLogRets',
                                                         hedge_name='BenchmarkLogRets',
                                                         window_size=linear_regress_window,
                                                         autocorr_periods=0)
        new_col_list = list(return_df.columns) + ['StockFwdRets', 'BenchmarkFwdRets', 'ExcessFwdRets', 'StockAdjClose',
                                                  'BenchmarkAdjClose', 'AbsValExcessFwdRets']
        new_df = pd.concat([return_df, stock_fwd_returns, benchmark_fwd_returns, excess_fwd_return,
                            stock_atr_df.AdjClose, benchmark_atr_df.AdjClose, abs_val_excess_fwd_return], axis=1)
        new_df.columns = new_col_list
        new_df.dropna(inplace=True)
        th = 1.0
        df_analyze = pd.concat([new_df.recentVol_emaAtr_diff_Atr, new_df.ExcessFwdRets, df_currCurr.SpearmanCorr,
                                df_currCurr.PearsonCorr, df_currCurr.r_squared, df_currCurr.intercept,
                                df_currCurr.slope, new_df.AbsValExcessFwdRets, new_df.StockFwdRets,
                                new_df.BenchmarkFwdRets, new_df.StockAdjClose, new_df.BenchmarkAdjClose], axis=1)
        # return df_analyze
        # spearman Correl when standard vol - recent vol greater than
        mean_spearmanCorr_recentVolHigher = df_analyze[df_analyze.recentVol_emaAtr_diff_Atr > th].SpearmanCorr.mean()
        std_spearmanCorr_recentVolHigher = df_analyze[df_analyze.recentVol_emaAtr_diff_Atr > th].SpearmanCorr.std()
        count_recentVolHigher = len(df_analyze[df_analyze.recentVol_emaAtr_diff_Atr > th])
        print("--------------RESULTS OF MEAN SPEARMAN CORREL WHEN RECENT VOL VS. HIST VOL-----------------")
        print(mean_spearmanCorr_recentVolHigher, std_spearmanCorr_recentVolHigher, count_recentVolHigher)

        # spearman Correl when standard vol - recent vol less than
        mean_spearmanCorr_recentVolLower = df_analyze[df_analyze.recentVol_emaAtr_diff_Atr < -th].SpearmanCorr.mean()
        std_spearmanCorr_recentVolLower = df_analyze[df_analyze.recentVol_emaAtr_diff_Atr < -th].SpearmanCorr.std()
        count_recentVolLower = len(df_analyze[df_analyze.recentVol_emaAtr_diff_Atr < -th])
        print(mean_spearmanCorr_recentVolLower, std_spearmanCorr_recentVolLower, count_recentVolLower)

        # spearman Correl when ema vol and standard vol are in line - recent vol in line with historical vol
        mean_spearmanCorr_recentHistVolSame = df_analyze[(df_analyze.recentVol_emaAtr_diff_Atr < th) &
                                                         (df_analyze.recentVol_emaAtr_diff_Atr > -th)]. \
            SpearmanCorr.mean()
        std_spearmanCorr_recentHistVolSame = df_analyze[(df_analyze.recentVol_emaAtr_diff_Atr < th) &
                                                        (df_analyze.recentVol_emaAtr_diff_Atr > -th)]. \
            SpearmanCorr.std()
        count_recentHistVolSame = len(new_df[(new_df.recentVol_emaAtr_diff_Atr < th) &
                                             (new_df.recentVol_emaAtr_diff_Atr > -th)])
        print(mean_spearmanCorr_recentHistVolSame, std_spearmanCorr_recentHistVolSame, count_recentHistVolSame)
        print("--------------------------------------------------------------------------------------------")
        print("-----------------RESULTS OF OLD RECENT VOL VS. HIST VOL USING ATR ANALYSIS------------------")
        # ema vol greater than standard vol - recent vol greater
        mean_return_recentVolHigher = new_df[new_df.recentVol_emaAtr_diff_Atr > th].ExcessFwdRets.mean()
        stdDev_return_recentVolHigher = new_df[new_df.recentVol_emaAtr_diff_Atr > th].ExcessFwdRets.std()
        count_return_recentVolHigher = len(new_df[new_df.recentVol_emaAtr_diff_Atr > th])
        # ema vol greater than standard vol - recent vol less than
        mean_return_recentVolLower = new_df[new_df.recentVol_emaAtr_diff_Atr < -th].ExcessFwdRets.mean()
        stdDev_return_recentVolLower = new_df[new_df.recentVol_emaAtr_diff_Atr < -th].ExcessFwdRets.std()
        count_return_recentVolLower = len(new_df[new_df.recentVol_emaAtr_diff_Atr < -th])
        # ema vol and standard vol are in line - recent vol in line with historical vol
        mean_return_recentHistVolSame = new_df[(new_df.recentVol_emaAtr_diff_Atr < th) &
                                               (new_df.recentVol_emaAtr_diff_Atr > -th)].ExcessFwdRets.mean()
        stdDev_return_recentHistVolSame = new_df[(new_df.recentVol_emaAtr_diff_Atr < th) &
                                                 (new_df.recentVol_emaAtr_diff_Atr > -th)].ExcessFwdRets.std()
        count_return_recentHistVolSame = len(new_df[(new_df.recentVol_emaAtr_diff_Atr < th) &
                                                    (new_df.recentVol_emaAtr_diff_Atr > -th)])
        print((mean_return_recentVolHigher * 100.0).round(4), (stdDev_return_recentVolHigher * 100.0).round(4),
              count_return_recentVolHigher)
        print((mean_return_recentVolLower * 100.0).round(4), (stdDev_return_recentVolLower * 100.0).round(4),
              count_return_recentVolLower)
        print((mean_return_recentHistVolSame * 100.0).round(4), (stdDev_return_recentHistVolSame * 100.0).round(4),
              count_return_recentHistVolSame)
        print("------------------------------------------------------------------------------------------------")
        temp_th = 0.02
        mean_recentVol_emaAtr_diff_Atr_posRet = df_analyze[df_analyze.ExcessFwdRets > temp_th]. \
            recentVol_emaAtr_diff_Atr.mean()
        std_recentVol_emaAtr_diff_Atr_posRet = df_analyze[
            df_analyze.ExcessFwdRets > temp_th].recentVol_emaAtr_diff_Atr.std()
        count_recentVol_emaAtr_diff_Atr_posRet = len(df_analyze[df_analyze.ExcessFwdRets > temp_th])
        mean_recentVol_emaAtr_diff_Atr_negRet = df_analyze[df_analyze.ExcessFwdRets < temp_th]. \
            recentVol_emaAtr_diff_Atr.mean()
        std_recentVol_emaAtr_diff_Atr_negRet = df_analyze[
            df_analyze.ExcessFwdRets < temp_th].recentVol_emaAtr_diff_Atr.std()
        count_recentVol_emaAtr_diff_Atr_negRet = len(df_analyze[df_analyze.ExcessFwdRets < temp_th])
        print("----------------RESULTS OF RECENT VOL WHEN POS RETURN VS. NEG RETURN------------------------")
        print(mean_recentVol_emaAtr_diff_Atr_posRet, std_recentVol_emaAtr_diff_Atr_posRet,
              count_recentVol_emaAtr_diff_Atr_posRet, mean_recentVol_emaAtr_diff_Atr_negRet,
              std_recentVol_emaAtr_diff_Atr_negRet, count_recentVol_emaAtr_diff_Atr_negRet)
        print("--------------------------------------------------------------------------------------------")
        another_th_upper = -0.5
        another_th_lower = -1.0
        spearCorr_th_upper = 1.0
        spearCorr_th_lower = 0.7
        long_ticker_mean = df_analyze[(df_analyze.SpearmanCorr < spearCorr_th_upper) &
                                      (df_analyze.SpearmanCorr > spearCorr_th_lower) &
                                      (df_analyze.recentVol_emaAtr_diff_Atr < another_th_upper) &
                                      (df_analyze.recentVol_emaAtr_diff_Atr > another_th_lower)].StockFwdRets.mean()
        long_bmark_mean = df_analyze[(df_analyze.SpearmanCorr < spearCorr_th_upper) &
                                     (df_analyze.SpearmanCorr > spearCorr_th_lower) &
                                     (df_analyze.recentVol_emaAtr_diff_Atr < another_th_upper) &
                                     (df_analyze.recentVol_emaAtr_diff_Atr > another_th_lower)].BenchmarkFwdRets.mean()
        long_ticker_std = df_analyze[(df_analyze.SpearmanCorr < spearCorr_th_upper) &
                                     (df_analyze.SpearmanCorr > spearCorr_th_lower) &
                                     (df_analyze.recentVol_emaAtr_diff_Atr < another_th_upper) &
                                     (df_analyze.recentVol_emaAtr_diff_Atr > another_th_lower)].StockFwdRets.std()
        long_bmark_std = df_analyze[(df_analyze.SpearmanCorr < spearCorr_th_upper) &
                                    (df_analyze.SpearmanCorr > spearCorr_th_lower) &
                                    (df_analyze.recentVol_emaAtr_diff_Atr < another_th_upper) &
                                    (df_analyze.recentVol_emaAtr_diff_Atr > another_th_lower)].BenchmarkFwdRets.std()
        long_ticker_count = len(df_analyze[(df_analyze.SpearmanCorr < spearCorr_th_upper) &
                                           (df_analyze.SpearmanCorr > spearCorr_th_lower) &
                                           (df_analyze.recentVol_emaAtr_diff_Atr < another_th_upper) &
                                           (df_analyze.recentVol_emaAtr_diff_Atr > another_th_lower)])
        avg_slope = df_analyze[(df_analyze.SpearmanCorr < spearCorr_th_upper) &
                               (df_analyze.SpearmanCorr > spearCorr_th_lower) &
                               (df_analyze.recentVol_emaAtr_diff_Atr < another_th_upper) &
                               (df_analyze.recentVol_emaAtr_diff_Atr > another_th_lower)].slope.mean()
        avg_slope_pos_ret = df_analyze[(df_analyze.SpearmanCorr < spearCorr_th_upper) &
                                       (df_analyze.SpearmanCorr > spearCorr_th_lower) &
                                       (df_analyze.recentVol_emaAtr_diff_Atr < another_th_upper) &
                                       (df_analyze.recentVol_emaAtr_diff_Atr > another_th_lower) &
                                       (df_analyze.StockFwdRets > 0.0)].intercept.mean()
        avg_slope_neg_ret = df_analyze[(df_analyze.SpearmanCorr < spearCorr_th_upper) &
                                       (df_analyze.SpearmanCorr > spearCorr_th_lower) &
                                       (df_analyze.recentVol_emaAtr_diff_Atr < another_th_upper) &
                                       (df_analyze.recentVol_emaAtr_diff_Atr > another_th_lower) &
                                       (df_analyze.StockFwdRets < 0.0)].intercept.mean()
        sr_ticker = (long_ticker_mean / long_ticker_std) * np.sqrt(long_ticker_count)
        sr_bmark = (long_bmark_mean / long_bmark_std) * np.sqrt(long_ticker_count)
        print("-------------------------CURRENT MAIN ANALYSIS SECTION-------------------------------------")
        print(long_ticker_mean, long_ticker_std, long_ticker_count, sr_ticker, long_bmark_mean, long_bmark_std,
              long_ticker_count, sr_bmark, avg_slope, avg_slope_pos_ret, avg_slope_neg_ret)
        long_bmark_mean = df_analyze[(df_analyze.SpearmanCorr > -10.0) & (df_analyze.recentVol_emaAtr_diff_Atr > 1.0)]. \
            BenchmarkFwdRets.mean()
        long_ticker_mean = df_analyze[(df_analyze.SpearmanCorr > -10.0) & (df_analyze.recentVol_emaAtr_diff_Atr > 1.0)]. \
            StockFwdRets.mean()
        long_bmark_std = df_analyze[(df_analyze.SpearmanCorr > -10.0) & (df_analyze.recentVol_emaAtr_diff_Atr > 1.0)]. \
            BenchmarkFwdRets.std()
        long_ticker_std = df_analyze[(df_analyze.SpearmanCorr > -10.0) & (df_analyze.recentVol_emaAtr_diff_Atr > 1.0)]. \
            StockFwdRets.std()
        long_bmark_count = len(
            df_analyze[(df_analyze.SpearmanCorr > -10.0) & (df_analyze.recentVol_emaAtr_diff_Atr > 1.0)])
        print("----------------------------------------------------------------------------------------")
        print(long_bmark_mean, long_bmark_std, long_bmark_count, long_ticker_mean, long_ticker_std, long_bmark_count)

        # Linear Reggression : fwd stock returns vs. recentVol_emaAtr_diff_Atr
        # self.get_linear_regression(new_df, "", "ExcessFwdRets")
        self.get_linear_regression(new_df, "recentVol_emaAtr_diff_Atr", "ExcessFwdRets")
        # print("DF_ANALYZE b4 get_linear_regression() ", df_analyze.tail())
        self.get_linear_regression(df_analyze.dropna(), "SpearmanCorr", "AbsValExcessFwdRets")
        self.get_linear_regression(df_analyze.dropna(), "SpearmanCorr", "recentVol_emaAtr_diff_Atr")
        self.get_linear_regression(new_df, "emaAtr_spread", "AbsValExcessFwdRets")

        return return_df[atr_columns]

    def get_stock_benchmark_spread_px(self,
                                      stock_ticker,
                                      benchmark_ticker,
                                      hedge_ratio_calc_window=20,
                                      hedge_ratio_diff_window=9,
                                      freq='D',
                                      start_date='2010-01-01',
                                      end_date=str(pd.to_datetime('today')).split(' ')[0]):

        pd.set_option('display.max_columns', 7)
        sigmas_multiplier = 0.5
        # use $1,000 invested in stock and $1,000 invested in benchmark
        stock_data = self.get_pricing(ticker=stock_ticker, freq=freq, start_date=start_date, end_date=end_date,
                                      fields=['adjClose'])[stock_ticker]
        benchmark_data = self.get_pricing(ticker=benchmark_ticker, freq=freq, start_date=start_date, end_date=end_date,
                                          fields=['adjClose'])[benchmark_ticker]
        stock_rolling_returns = stock_data.pct_change(periods=1).dropna()
        benchmark_rolling_returns = benchmark_data.pct_change(periods=1).dropna()
        stk_bmark_hedge_ratio = abs(stock_rolling_returns) / abs(benchmark_rolling_returns)
        ema_hedge_ratio = stk_bmark_hedge_ratio.ewm(span=20, adjust=False).mean()
        ema_hedge_ratio.columns = ['ema_hedge_ratio']
        ema_hedge_ratio['rolling_std_dev'] = stk_bmark_hedge_ratio.rolling(20).std()
        ema_hedge_ratio['plusOne_stdDev_hr'] = ema_hedge_ratio.ema_hedge_ratio + \
                                               sigmas_multiplier * ema_hedge_ratio.rolling_std_dev
        ema_hedge_ratio['minusOne_stdDev_hr'] = ema_hedge_ratio.ema_hedge_ratio - \
                                                sigmas_multiplier * ema_hedge_ratio.rolling_std_dev
        ema_hedge_ratio['original_hr'] = stk_bmark_hedge_ratio
        ema_hedge_ratio['upper_band_breach'] = \
            ema_hedge_ratio['original_hr'] - ema_hedge_ratio['plusOne_stdDev_hr'] > 0
        ema_hedge_ratio['lower_band_breach'] = \
            ema_hedge_ratio['original_hr'] - ema_hedge_ratio['minusOne_stdDev_hr'] < 0
        change_hedge_ratio_ema = ema_hedge_ratio.ema_hedge_ratio.diff(periods=1). \
            ewm(span=hedge_ratio_diff_window, adjust=False).mean()
        # change_hedge_ratio_ema. \
        #    plot(title=str(hedge_ratio_diff_window) +
        #               " EMA of Change in EMA Hedge Ratio, Can We Forecast Dislocations?")
        # plt.show()
        change_hedge_ratio_ema.columns = ["ema_diff_hr"]
        # ema_hedge_ratio[['ema_hedge_ratio', 'plusOne_stdDev_hr', 'minusOne_stdDev_hr', 'original_hr']]. \
        #    plot(title="EMA 30 Day Rolling Hedge Ratio")
        hr_ema_stock_df = pd.merge(left=change_hedge_ratio_ema, right=stock_data, left_index=True, right_index=True)
        hr_ema_stock_df.columns = ['ema_diff_hr', 'stock_adjClose']
        hr_ema_df = pd.merge(left=hr_ema_stock_df, right=benchmark_data, left_index=True, right_index=True)
        hr_ema_df.columns = ['ema_diff_hr', 'stock_adjClose', 'benchmark_adjClose']
        hr_ema_df = pd.merge(left=hr_ema_df, right=stock_data.pct_change(periods=1).shift(periods=-1),
                             left_index=True, right_index=True)
        hr_ema_df.columns = ['ema_diff_hr', 'stock_adjClose', 'benchmark_adjClose', 'fwd_stock_return']
        hr_ema_df = pd.merge(left=hr_ema_df, right=benchmark_data.pct_change(periods=1).shift(periods=-1),
                             left_index=True, right_index=True)
        hr_ema_df = pd.merge(left=hr_ema_df,
                             right=ema_hedge_ratio[['original_hr', 'upper_band_breach', 'lower_band_breach']],
                             left_index=True, right_index=True)
        hr_ema_df.columns = ['ema_diff_hr', 'stock_adjClose', 'benchmark_adjClose', 'fwd_stock_return',
                             'fwd_benchmark_return', 'original_hr', 'upper_band_breach', 'lower_band_breach']
        size_when_upper_breach = len(hr_ema_df[(hr_ema_df.upper_band_breach == True)])
        size_when_lower_breach = len(hr_ema_df[(hr_ema_df.lower_band_breach == True)])
        mean_when_upper_breach = hr_ema_df[(hr_ema_df.upper_band_breach == True)].fwd_stock_return.mean()
        std_rets_when_upper_breach = hr_ema_df[(hr_ema_df.upper_band_breach == True)].fwd_stock_return.std()
        mean_when_lower_breach = hr_ema_df[(hr_ema_df.lower_band_breach == True)].fwd_stock_return.mean()
        std_rets_when_lower_breach = hr_ema_df[(hr_ema_df.lower_band_breach == True)].fwd_stock_return.std()
        mean_when_within_bands = hr_ema_df[(hr_ema_df.upper_band_breach == False) &
                                           (hr_ema_df.lower_band_breach == False)].fwd_stock_return.mean()
        std_when_within_bands = hr_ema_df[(hr_ema_df.upper_band_breach == False) &
                                          (hr_ema_df.lower_band_breach == False)].fwd_stock_return.std()
        size_when_within_bands = \
            len(hr_ema_df[(hr_ema_df.upper_band_breach == False) & (hr_ema_df.lower_band_breach == False)])
        print(mean_when_upper_breach)
        print(std_rets_when_upper_breach)
        print(size_when_upper_breach)
        print(mean_when_lower_breach)
        print(std_rets_when_lower_breach)
        print(size_when_lower_breach)
        print(mean_when_within_bands)
        print(std_when_within_bands)
        print(size_when_within_bands)

        return

        print("mean 1-day stock fwd return when ema_diff_hr > 0",
              hr_ema_df[(hr_ema_df.ema_diff_hr < ema_diff_hr_upperth) | (hr_ema_df.ema_diff_hr > ema_diff_hr_lowerth)].
              fwd_stock_return.mean())
        print("std dev 1-day stock fwd return when ema_diff_hr > 0",
              hr_ema_df[(hr_ema_df.ema_diff_hr < ema_diff_hr_upperth) | (hr_ema_df.ema_diff_hr > ema_diff_hr_lowerth)].
              fwd_stock_return.std())
        print("number of samples when ema_diff_hr > 0",
              hr_ema_df[(hr_ema_df.ema_diff_hr < ema_diff_hr_upperth) | (hr_ema_df.ema_diff_hr > ema_diff_hr_lowerth)].
              fwd_stock_return.size)
        # print("mean 1-day benchmark fwd return when ema_diff_hr > 0", hr_ema_df[hr_ema_df.ema_diff_hr < ema_diff_hr_th].
        #      fwd_benchmark_return.mean())
        # print("std dev 1-day benchmark fwd return when ema_diff_hr > 0",
        #      hr_ema_df[hr_ema_df.ema_diff_hr < ema_diff_hr_th].fwd_benchmark_return.std())
        # print("number of samples when ema_diff_hr > 0", hr_ema_df[hr_ema_df.ema_diff_hr < ema_diff_hr_th].
        #      fwd_benchmark_return.size)
        # plt.show()

        return
        init_dollar_value = 1000.0
        stock_log_return = np.log(1 + stock_data.pct_change())
        stock_return = (1 + stock_data.pct_change()).cumprod()
        benchmark_log_return = np.log(1 + benchmark_data.pct_change())
        benchmark_return = (1 + benchmark_data.pct_change()).cumprod()
        combined_df = pd.merge(left=stock_log_return, right=benchmark_log_return, left_index=True, right_index=True)
        combined_df.rename(columns={"adjClose_x": "StockLogRet",
                                    "adjClose_y": "BenchmarkLogRet"}, inplace=True)
        temp_df = pd.merge(left=stock_return, right=benchmark_return, left_index=True, right_index=True)
        temp_df.rename(columns={"adjClose_x": "StockRet",
                                "adjClose_y": "BenchmarkRet"}, inplace=True)
        combined_df = pd.merge(left=combined_df, right=temp_df, left_index=True, right_index=True)
        combined_df['Init_Dollar_Value'] = init_dollar_value
        combined_df['StockHoldingDollarValue'] = (combined_df['StockRet'] * combined_df['Init_Dollar_Value']) - \
                                                 init_dollar_value
        combined_df['StockHoldingCummReturn'] = combined_df.StockHoldingDollarValue / init_dollar_value
        combined_df['BenchmarkHoldingDollarValue'] = combined_df['BenchmarkRet'] * combined_df['Init_Dollar_Value'] - \
                                                     init_dollar_value
        combined_df['BenchmarkHoldingCummReturn'] = \
            combined_df.BenchmarkHoldingDollarValue / init_dollar_value
        combined_df['SpreadHoldingDollarValue'] = combined_df['StockHoldingDollarValue'] - \
                                                  combined_df['BenchmarkHoldingDollarValue']
        combined_df['SpreadHoldingCummReturn'] = \
            combined_df.SpreadHoldingDollarValue / (2.0 * init_dollar_value)
        combined_df['Cumm Return Spread, Hedged vs. Unhedged'] = combined_df.StockHoldingCummReturn - \
                                                                 combined_df.SpreadHoldingCummReturn
        combined_df[['SpreadHoldingDollarValue', 'StockHoldingDollarValue', 'BenchmarkHoldingDollarValue']]. \
            plot(title=str(stock_ticker + ' vs. ' + benchmark_ticker))
        # plt.suptitle(t='LONG AAPL and SHORT IWM hedge, profit on both sides of trade since 2018. '
        #               'Domestic economy booming??')
        combined_df[['SpreadHoldingCummReturn', 'StockHoldingCummReturn', 'BenchmarkHoldingCummReturn']]. \
            plot(title=str(stock_ticker + ' vs. ' + benchmark_ticker + ' Cummulative Returns'))
        # plt.suptitle(t='LONG AAPL and SHORT IWM hedge, profit on both sides of trade since 2018. '
        #               'Domestic economy booming??')
        combined_df[['Cumm Return Spread, Hedged vs. Unhedged']]. \
            plot(title=str(stock_ticker + ' vs. ' + benchmark_ticker + ' Cumm Return Spread, Hedged vs. Unhedged'))
        # plt.suptitle(t='What should we say about this?')
        # plt.show()

    def get_pricing_df_ready(self,
                             ticker,
                             freq='D',
                             start_date='2010-01-01',
                             end_date=str(pd.to_datetime('today')).split(' ')[0],
                             fields=['adjOpen', 'adjHigh', 'adjLow', 'adjClose']):

        if (type(ticker) is not list):
            symbols = [ticker]
        else:
            symbols = ticker
        if len(symbols) > StatisticalMoments.MAX_TICKER_PULL_SIZE:
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
        return_df = None
        for ticker in symbols:
            pricing_df = pricing_dict[ticker]
            if freq is not 'D':
                pricing_df = self.down_sample_daily_price_data(pricing=pricing_df, to_freq=freq)
            pricing = pricing_df[fields]
            new_col_nm_list = []
            for col_nm in pricing.columns:
                new_col_nm = ticker + '_' + col_nm
                new_col_nm_list.append(new_col_nm)
            pricing.columns = new_col_nm_list
            if return_df is None:
                return_df = pricing
            else:
                return_df = pd.concat([return_df, pricing], axis=1)
        return return_df

    def down_sample_daily_price_data(self,
                                     pricing,
                                     to_freq='W'):
        output = pricing.resample(to_freq, loffset=pd.offsets.timedelta(days=-6)).agg({'adjOpen': 'first',
                                                                                       'adjHigh': 'max',
                                                                                       'adjLow': 'min',
                                                                                       'adjClose': 'last',
                                                                                       'adjVolume': 'sum',
                                                                                       'close': 'last'})

        # output = pricing.resample(to_freq,  # Weekly resample
        #                           how={'adjOpen': 'first',
        #                                'adjHigh': 'max',
        #                                'adjLow': 'min',
        #                                'adjClose': 'last',
        #                                'adjVolume': 'sum',
        #                                'close': 'last'},
        #                           loffset=pd.offsets.timedelta(days=-6))
        return output

    """ Calculate the kurtosis of the returns for the specificed ticker, with specificed
        start and end dates. 
    """

    def calc_stock_return_kurtosis(self,
                                   ticker,
                                   data=None,
                                   start_date='2010-01-01',
                                   end_date=str(pd.to_datetime('today')).split(' ')[0]):

        if data is None:
            returns = self.get_stock_returns(ticker=ticker, start_date=start_date, end_date=end_date)
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
                               data=None,
                               start_date='2010-01-01',  # 2010-01-01 to present is max data offered by Tiingo
                               end_date=str(pd.to_datetime('today')).split(' ')[0],  # str, e.g. "2011-01-01"
                               px_type='adjClose'):

        if data is None:
            returns = self.get_stock_returns(ticker=ticker, start_date=start_date, end_date=end_date, px_type=px_type)
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
                                data=None):

        if data is None:
            returns = self.get_stock_returns(ticker=ticker, start_date=start_date, end_date=end_date)
        else:
            returns = data
        roll_func = returns.rolling(window=window_size, center=False).skew()
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
                                data=None):

        if data is None:
            returns = self.get_stock_returns(ticker, start_date, end_date)
        else:
            returns = data
        roll_func = returns.rolling(window=window_size, center=False).kurtosis()
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
        top = mu + sigma_multiplier * sigma
        bottom = mu - sigma_multiplier * sigma
        bool_non_outliers = (px_rets_series <= top) | (px_rets_series >= bottom)
        no_outliers = px_returns_series[bool_non_outliers].sort_index()
        return no_outliers

    @staticmethod
    def remove_outliers_from_non_normal_dist(px_returns_series,
                                             k_IQR=3.0):

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
        percent_outliers = int(outliers.describe()['count']) / num_obs * 100.0
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
        top = mu + 3 * sigma
        bottom = mu - 3 * sigma
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
        MAD = np.sum(abs_deviation) / len(data)
        variance = np.var(data)
        std_dev = np.std(data)
        # semi-variance
        semi_var_lows = [less_mu for less_mu in data if less_mu <= mu]
        semi_var_highs = [more_mu for more_mu in data if more_mu > mu]
        semi_var_lows = np.sum((semi_var_lows - mu) ** 2) / len(semi_var_lows)
        semi_var_highs = np.sum((semi_var_highs - mu) ** 2) / len(semi_var_highs)
        print("Range: ", dist_range, '\n',
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
        hist, edges = np.histogram(normal_dist_sample, bins=int(np.round(bins)), density=True)


if __name__ == '__main__':
    sm = StatisticalMoments()
    sm.random_numbers_test()
    ticker = 'T'
    start_date = '2016-01-01'
    end_date = '2017-01-01'
    pricing = sm.get_pricing(ticker=ticker, start_date=start_date, end_date=end_date)
    X = np.random.randint(100, size=100)
    mu = np.mean(X)
    sm.calc_variance_metrics_example(data=X, semivar_cutoff=mu)
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
    print("test in sample normaility of ticker AMC...")
    sm.test_normality(ticker="AMC",
                      start_date='2014-01-01',
                      end_date='2016-01-01')
    print("...and now out of sample. Can we make the same conclusion about normality?")
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


def do_line_regress(the_data, x_col='date', y_col='AdjClose'):
    # the_data is a dataframe, with the index reset, with columns ['date', 'adjClose']
    from statsmodels import regression
    x_data_const = sm.add_constant(the_data[x_col])
    lr_model = regression.linear_model.OLS(the_data[y_col], x_data_const).fit()
    return lr_model.param[0], lr_model.param[1]
