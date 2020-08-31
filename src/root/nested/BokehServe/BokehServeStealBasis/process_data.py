import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.stattools import jarque_bera
from .data_interface import TiingoDataObject
from . import get_logger

pd.set_option('display.max_columns', 20)
LOGGER = get_logger()
MAX_TICKER_PULL_SIZE = 10


class ProcessData:

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

        LOGGER.info("StatisticalMoments.test_normality(): running function...")
        returns = self.get_stock_returns(ticker=ticker, start_date=start_date, end_date=end_date)
        LOGGER.info('StatisticalMoments.test_normality(): skew is %f', stats.skew(returns))
        LOGGER.info('StatisticalMoments.test_normality(): mean is %f', np.mean(returns))
        LOGGER.info('StatisticalMoments.test_normality(): median is %f', np.median(returns))
        plt.hist(returns, 30)
        plt.show()
        _, pvalue, _, _ = jarque_bera(returns)
        if (pvalue > 0.05):
            LOGGER.info('StatisticalMoments.test_normality(): '
                        'jarques-bera test p-value of %f is within 5pct error tolerance, '
                        'and therefore the returns are likely normal.', pvalue)
            print("jarques-bera test p-value of ", pvalue,
                  " is within 5% error tolerance, and therefore the returns are likely normal.")
        else:
            LOGGER.info('StatisticalMoments.test_normality(): '
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

        LOGGER.info('StatisticalMoments.get_stock_returns(): running function on ticker %s', ticker)
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

        LOGGER.info('StatisticalMoments.get_stock_excess_return(): running function...')
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

        LOGGER.info('StatisticalMoments.get_rolling_excess_returns(): running function ticker %s, '
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

        LOGGER.info('StatisticalMoments.get_rolling_returns(): running function for ticker %s...',
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

        if type(ticker) is not list:
            symbols = [ticker]
        else:
            symbols = ticker
        if len(symbols) > MAX_TICKER_PULL_SIZE:
            LOGGER.error("max number of ticker pulls allowed at once is 3, "
                         "%s given!", str(len(symbols)))
            raise ValueError("max number of ticker to pull at once is " + \
                             str(MAX_TICKER_PULL_SIZE) + \
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
                pricing_df = self.down_sample_daily_price_data(pricing=pricing_df, to_freq=freq)
            pricing = pricing_df[fields]
            return_dict[ticker] = pricing
        return return_dict

    def linear_regression_on_price(self,
                                   df,
                                   colnm_x,
                                   colnm_y=None,
                                   short_interval=30,
                                   long_interval=180):
        """
        The key parameter here is interval. It is defaulted to 30 days.
        :param df:
        :param colnm_x:
        :param colnm_y:
        :param interval:
        :return:
        """
        df.reset_index()
        weeks = [g for n, g in df.reset_index().groupby(pd.Grouper(key='Date', freq='W'))]
        months = [g for n, g in df.reset_index().gorupby(pd.Grouper(key='Date', freq='M'))]



    def get_linear_regression(self,
                              df,
                              colnm_x,
                              colnm_y):

        pd.set_option('display.max_columns', 20)
        df = sm.add_constant(df)
        if 'const' not in df.columns:
            df['const'] = 1.0
        # reg1 = sm.OLS(endog=df.StockFwdLogRets, exog=df[['const', 'BenchmarkFwdLogRets']], missing='drop')
        reg1 = sm.OLS(endog=df[colnm_y], exog=df[['const', colnm_x]], missing='drop')
        results = reg1.fit()
        # X = df['BenchmarkFwdLogRets']
        X = df[colnm_x]
        y = df[colnm_y]
        labels = df.index
        # Replace markers with country labels
        #fig, ax = plt.subplots()
        # ax.scatter(X, y, marker='')
        # ax.plot(X, y, 'o')
        # for i, label in enumerate(labels):
        #    ax.annotate(label, (X.iloc[i], y.iloc[i]))
        # Fit a linear trend line
        np_poly_slope, np_poly_intercept = np.polyfit(X, y, 1)
        # print ("from np polyfit for chart", slope, intercept)
        # ax.plot(X, np_poly_slope * X + np_poly_intercept, color='black')
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

        linear_regress_window = 8
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
        cum_stock_fwd_returns = (1+stock_fwd_returns).cumprod() - 1
        benchmark_fwd_returns = benchmark_fwd_log_rets
        cum_benchmark_fwd_returns = (1+benchmark_fwd_returns).cumprod() - 1
        excess_fwd_return = excess_fwd_log_rets
        cum_excess_fwd_returns = (1+excess_fwd_return).cumprod() - 1
        abs_val_excess_fwd_return = excess_fwd_log_rets.abs()
        # print ("abs val excess fwd return", abs_val_excess_fwd_return.head(40))

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
        # print("what we feed to rolling Linear regression", df_lr.head(20))
        df_currCurr = self.get_rolling_linear_regression(df_lr,
                                                         target_name='StockLogRets',
                                                         hedge_name='BenchmarkLogRets',
                                                         window_size=linear_regress_window,
                                                         autocorr_periods=0)
        # print ("what we get back from rolling Linear Regression", df_currCurr.head(20))
        new_col_list = list(return_df.columns) + ['StockFwdRets', 'BenchmarkFwdRets', 'ExcessFwdRets', 'StockAdjClose',
                                                  'BenchmarkAdjClose', 'AbsValExcessFwdRets']
        new_df = pd.concat([return_df, stock_fwd_returns, benchmark_fwd_returns, excess_fwd_return,
                            stock_atr_df.AdjClose, benchmark_atr_df.AdjClose, abs_val_excess_fwd_return], axis=1)
        new_df.columns = new_col_list
        # print("what is in new_df", new_df.head(20))
        # new_df.dropna(inplace=True)
        th = 1.0
        df_analyze = pd.concat([new_df.recentVol_emaAtr_diff_Atr, new_df.ExcessFwdRets, df_currCurr.SpearmanCorr,
                                df_currCurr.PearsonCorr, df_currCurr.r_squared, df_currCurr.intercept,
                                df_currCurr.slope, new_df.AbsValExcessFwdRets, new_df.StockFwdRets,
                                new_df.BenchmarkFwdRets, new_df.StockAdjClose, new_df.BenchmarkAdjClose,
                                df_lr.StockLogRets, df_lr.BenchmarkLogRets], axis=1)
        # print("what we are feeding to total line regress", df_analyze.head())
        df_analyze.dropna(inplace=True)
        series_retVal = self.get_linear_regression(df_analyze, 'StockFwdRets', 'BenchmarkFwdRets')
        np_poly_slope = series_retVal['np.poly.slope']
        np_poly_intercept = series_retVal['np.poly.intercept']
        np_poly_rsquared = series_retVal['r_squared']
        df_analyze['SlopeTotal'] = np_poly_slope
        df_analyze['InterceptTotal'] = np_poly_intercept
        df_analyze['rSquaredTotal'] = np_poly_rsquared
        df_analyze['LineRegressTotal'] = df_analyze.SlopeTotal.mul(df_analyze.StockFwdRets). \
            add(df_analyze.InterceptTotal)
        df_analyze['CumStockFwdRets'] = cum_stock_fwd_returns
        df_analyze['CumBenchmarkFwdRets'] = cum_benchmark_fwd_returns
        df_analyze['CumExcessFwdRets'] = cum_excess_fwd_returns
        return df_analyze
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
        another_th_upper = -1.0
        another_th_lower = -100.0
        spearCorr_th_upper = 0.1
        spearCorr_th_lower = -1000.0
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
        sr_ticker = (long_ticker_mean / long_ticker_std) * np.sqrt(long_ticker_count)
        sr_bmark = (long_bmark_mean / long_bmark_std) * np.sqrt(long_ticker_count)
        print("-------------------------CURRENT MAIN ANALYSIS SECTION-------------------------------------")
        print(long_ticker_mean, long_ticker_std, long_ticker_count, sr_ticker, long_bmark_mean, long_bmark_std,
              long_ticker_count, sr_bmark, avg_slope)
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
        self.get_linear_regression(df_analyze.dropna(), "SpearmanCorr", "AbsValExcessFwdRets")
        self.get_linear_regression(df_analyze.dropna(), "SpearmanCorr", "recentVol_emaAtr_diff_Atr")
        self.get_linear_regression(new_df, "emaAtr_spread", "AbsValExcessFwdRets")

        return return_df[atr_columns]

    def get_pricing_df_ready(self,
                             ticker,
                             freq='D',
                             start_date='2010-01-01',
                             end_date=str(pd.to_datetime('today')).split(' ')[0],
                             fields=['adjOpen', 'adjHigh', 'adjLow', 'adjClose']):

        if type(ticker) is not list:
            symbols = [ticker]
        else:
            symbols = ticker
        if len(symbols) > MAX_TICKER_PULL_SIZE:
            LOGGER.error("max number of ticker pulls allowed at once is 3, "
                         "%s given!", str(len(symbols)))
            raise ValueError("max number of ticker to pull at once is " + \
                             str(MAX_TICKER_PULL_SIZE) + \
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
            LOGGER.info('StatisticalMoments.calc_stock_return_kurtosis(): Excess Kurtosis is %f.'
                        ' Because the excess kurtosis is negative, Y is platykurtic. Platykurtic'
                        ' distributions cluster around the mean, meaning less tail risk.', kurt)
        elif kurt > 0:
            LOGGER.info('StatisticalMoments.calc_stock_return_kurtosis(): Excess Kurtosis is %f.'
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
        LOGGER.info('StatisticalMoments.calc_stock_return_skew(): Skew = %f', skew)
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
        LOGGER.info('StatisticalMoments.remove_outliers_from_non_normal_dist(): '
                    'Identified Outliers (Series): %d' % int(outliers.describe()['count']))
        percent_outliers = int(outliers.describe()['count']) / num_obs * 100.0
        LOGGER.info('StatisticalMoments.remove_outliers_from_non_normal_dist(): '
                    'Percent Outliers: %.3f%s' % (percent_outliers, '%'))
        # remove outliers
        outliers_removed = px_returns_series[~(px_returns_series.isin(outliers))]
        LOGGER.info('StatisticalMoments.remove_outliers_from_non_normal_dist(): '
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


def take_first(px_series):
    return px_series[0]


def take_last(px_series):
    return px_series[-1]
