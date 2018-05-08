'''
Created on Jan 17, 2018

@author: ghazy
'''
from logging import getLogger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PerformanceAnalyzer(object):
    
    def __init__(self):
        
        self.logger = getLogger()
        
    def plot_excess_returns(self,
                            stock_data,
                            benchmark_data,
                            title="Excess Returns"):
        
        excess_returns = stock_data.pct_change().sub(benchmark_data.pct_change(), axis=0)
        excess_returns.plot(title=title)
        plt.show()
        
    def plot_average_excess_returns(self,
                                    stock_data,
                                    benchmark_data,
                                    title="Mean Excess Returns"):

        self.logger.info("running plot_average_excess_returns() function...")
        excess_returns = stock_data.pct_change().sub(benchmark_data.pct_change(), axis=0)
        mean_excess_returns = excess_returns.mean()
        mean_excess_returns.plot.bar(title=title)
    
    def plot_std_dev_excess_returns(self,
                                    stock_data,
                                    benchmark_data,
                                    title="Std Dev of the Return Difference"):

        self.logger.info("running plot_std_dev_excess_returns function...")
        excess_returns = stock_data.pct_change().sub(benchmark_data.pct_change(), axis=0)
        std_dev_excess_returns = excess_returns.std()
        std_dev_excess_returns.plot(title=title)
        plt.show()
    
    def plot_benchmark_data(self,
                            benchmark_data,
                            title):
        
        _ = benchmark_data.plot(subplots = True, title=title)
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
        
        _ = stock_data.plot(subplots = True, title=title)
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
        annualized_sharpe_ratio =  daily_sharpe_ratio.mul(annualizing_factor)
        self.logger.info("Annualized Sharpe Ratio %s ", str(annualized_sharpe_ratio))
        return annualized_sharpe_ratio