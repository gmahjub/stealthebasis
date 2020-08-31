'''
Created on Jan 17, 2018

@author: ghazy
'''
from root.nested import get_logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PerformanceAnalyzer(object):

    def __init__(self):

        self.logger = get_logger()

    def plot_excess_returns(self,
                            stock_data,
                            benchmark_data,
                            title="Excess Returns"):

        excess_returns = stock_data.pct_change().sub(benchmark_data.pct_change(), axis=0)
        excess_returns.plot(title=title)
        plt.show()

    def get_beta(self,
                 stock_returns,
                 benchmark_returns,
                 window_size):
        cov_df = stock_returns.rolling(window=window_size).corr(other=benchmark_returns)
        std_benchmark_df = benchmark_returns.rolling(window=window_size).std()
        std_stock_df = stock_returns.rolling(window=window_size).std()
        rolling_beta = cov_df.dropna().mul(std_stock_df.dropna().div(std_benchmark_df.dropna()))
        return rolling_beta

    def get_lpm(self,
                returns,
                window_size,
                threshold=0.0,
                order=1):
        return returns.rolling(window=window_size).apply(self.getLowerPartialMoment, axis=1,
                                                         threshold=threshold, order=order)

    def get_hpm(self,
                returns,
                window_size,
                threshold=0.0,
                order=1):
        return returns.rolling(window=window_size).apply(self.getHigherPartialMoment, axis=1,
                                                         threshold=threshold, order=order)

    def getLowerPartialMoment(self,
                              returns,
                              threshold,
                              order):
        threshold_array = np.empty(returns.size)
        threshold_array.fill(threshold)
        diff = threshold_array - returns
        diff = diff.clip(min=0)
        return np.sum(diff ** order) / len(returns)

    def getHigherPartialMoment(self,
                               returns,
                               threshold,
                               order):
        threshold_array = np.empty(returns.size)
        threshold_array.fill(threshold)
        diff = returns - threshold_array
        diff = diff.clip(min=0)
        return np.sum(diff ** order) / len(returns)

    def get_draw_down(self,
                      prices,
                      returns,
                      tau):

        return 1

    def get_stock_benchmark_spread_px(self,
                                      stock_ticker,
                                      benchmark_ticker,
                                      stock_data,
                                      benchmark_data):
        print("stock ticker is ", stock_ticker, "benchmark ticker is ", benchmark_ticker)
        # use $1,000 invested in stock and $1,000 invested in benchmark
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
        plt.show()

    def get_excess_log_returns(self,
                               stock_data,
                               benchmark_data):
        stock_return = np.log(1 + stock_data.pct_change())
        benchmark_return = np.log(1 + benchmark_data.pct_change())
        excess_returns = stock_return.sub(benchmark_return, axis=0)
        # excess_returns = stock_data.pct_change().sub(benchmark_data.pct_change(), axis=0)
        return excess_returns.dropna(), benchmark_return.dropna(), stock_return.dropna()

    def plot_average_excess_returns(self,
                                    stock_data,
                                    benchmark_data,
                                    title="Mean Excess Returns"):

        self.logger.info("running plot_average_excess_returns() function...")
        excess_returns = np.log(1 + stock_data.pct_change()).sub(np.log(1 + benchmark_data.pct_change()), axis=0)
        mean_excess_returns = excess_returns.mean()
        mean_excess_returns.plot.bar(title=title)

    def plot_std_dev_excess_returns(self,
                                    stock_data,
                                    benchmark_data,
                                    days=-1,
                                    title="Std Dev of the Return Difference"):

        if days == -1 and stock_data.size != benchmark_data.size:
            self.logger.error("PerformanceAnalyzer.plot_std_dev_excess_returns(): can't normalize to number of days"
                              "because the stock dataframe and benchmark dataframe are different lengths.")
        elif days == -1:
            timeNormalizer = stock_data.size
        else:
            timeNormalizer = days
        self.logger.info("running plot_std_dev_excess_returns function...")
        excess_returns = np.log(1 + stock_data.pct_change()).sub(np.log(1 + benchmark_data.pct_change()), axis=0)
        std_dev_excess_returns = excess_returns.std() * np.sqrt(timeNormalizer)
        std_dev_excess_returns.plot(title=title)
        plt.show()

    def plot_volatility_excess_returns(self,
                                       stock_data,
                                       benchmark_data,
                                       days=-1,
                                       title="Volatility of Excess Returns"):

        if days == -1 and stock_data.size != benchmark_data.size:
            self.logger.error("PerformanceAnalyzer.plot_volatility_excess_returns(): can't normalize to number of days"
                              "because the stock dataframe and benchmark dataframe are different lengths.")
        elif days == -1:
            timeNormalizer = stock_data.size
        else:
            timeNormalizer = days
        self.logger.info("running plot_volatility_excess_returns function...")
        excess_returns = np.log(1 + stock_data.pct_change()).sub(np.log(1 + benchmark_data.pct_change()), axis=0)
        vol_excess_returns = excess_returns.std() * np.sqrt(timeNormalizer)
        vol_excess_returns.plot(title=title)
        plt.show()

    def get_excess_return_volatility(self,
                                     stock_data,
                                     benchmark_data,
                                     days=-1):
        if days == -1 and stock_data.size != benchmark_data.size:
            self.logger.error("PerformanceAnalyzer.get_excess_return_volatility(): can't normalize to number of days"
                              "because the stock dataframe and benchmark dataframe are different lengths.")
        elif days == -1:
            timeNormalizer = stock_data.size
        else:
            timeNormalizer = days
        self.logger.info("PerformanceAnalyzer.get_excess_return_volatility(): running get_excess_return_volatility...")
        excess_returns = np.log(1 + stock_data.pct_change()).sub(np.log(1 + benchmark_data.pct_change()), axis=0)
        vol_excess_returns = excess_returns.std() * np.sqrt(timeNormalizer)
        return vol_excess_returns

    def plot_benchmark_data(self,
                            benchmark_data,
                            title):

        self.logger.info("running plot_benchmark_data function...")
        _ = benchmark_data.plot(subplots=True, title=title)
        plt.show()

    def plot_returns_benchmark_data(self,
                                    benchmark_data,
                                    title="Benchmark Returns"):

        benchmark_returns = benchmark_data.pct_change()
        _ = benchmark_returns.plot(title=title)
        plt.show()

    def plot_returns_stock_data(self,
                                stock_data,
                                title="Stock Data Returns"):

        pct_returns = stock_data.pct_change()
        _ = pct_returns.plot(title=title)
        plt.show()

    def plot_raw_stock_data(self,
                            stock_data,
                            title):

        _ = stock_data.plot(subplots=True, title=title)
        plt.show()

    def plot_annualized_sharpe_ratio(self,
                                     annualized_sharpe_ratio_df,
                                     title="Annualized Sharpe Ratio"):

        annualized_sharpe_ratio_df.plot.bar(title=title)

    def calc_sharpe_ratio(self,
                          stock_data,
                          benchmark_data):

        # stock data is a pandas dataframe
        # benchmark data is a pandas dataframe
        stock_returns = stock_data.pct_change()
        benchmark_returns = benchmark_data.pct_change()
        excess_returns = stock_returns.sub(benchmark_returns, axis=0)
        avg_excess_returns = excess_returns.mean()
        std_dev_excess_returns = excess_returns.std()
        # daily data assumed, so this is a daily sharpe
        daily_sharpe_ratio = avg_excess_returns.div(std_dev_excess_returns)
        annualizing_factor = np.sqrt(252)
        annualized_sharpe_ratio = daily_sharpe_ratio.mul(annualizing_factor)
        self.logger.info("Annualized Sharpe Ratio %s ", str(annualized_sharpe_ratio))
        return annualized_sharpe_ratio

    @staticmethod
    def cust_func(row, excess_flag=False):
        sharpe_exc_ret_col_nm = [col_nm for col_nm in row.index if 'sharpe' in col_nm][0]
        benchmark_std_col_nm = [col_nm for col_nm in row.index if 'benchmark_ret_std' in col_nm][0]
        std_exc_ret_col_nm = [col_nm for col_nm in row.index if 'std_exc_ret' in col_nm][0]
        if not excess_flag:
            cov_col_nm = [col_nm for col_nm in row.index if 'cov' in col_nm][0]
        row['beta'] = row[cov_col_nm] / row[benchmark_std_col_nm]
