'''
Created on Dec 8, 2017

@author: ghazy
'''

import numpy as np
import pandas as pd
import pandas.core as pdc
from root.nested import get_logger

from pandas_datareader import DataReader
from datetime import datetime


class DataObject(object):
    
    def __init__(self,
                 timeseries_series=None):
        
        self.logger = get_logger()
        self.logger.info('DataObject.__init__: Got logger object.')
        self.timeseries_series = timeseries_series
        
    def read_xls(self,
                 xls_file):
        
        xls = pd.ExcelFile(xls_file)
        # this is a list (sheet_names)
        sheet_names = xls.sheet_names
        xls_df_list = [] # list of dataframes for each sheet in the excel file
        for sheet in sheet_names:
            xls_df = pd.read_excel(xls, sheetname=sheet, na_values='n/a')
            xls_df['sheet'] = sheet
            xls_df_list.append(xls_df)
        
        return xls_df_list
        
    def concat_dataframes(self,
                          list_of_dataframes,
                          axis=1):
        
        # if axis is 0, it simply stacks columns on top of each other
        # if axis is 1, pandas aligns the indexes, and creates a new column(s) for each dataframe
        return pd.concat(list_of_dataframes, axis=axis)
    
    def get_web_finance_stock_data(self,
                                   start,
                                   end,
                                   ticker,
                                   data_source):
        
        #start = date object date(year, month, day)
        # end = date object date(year, month, day)
        # ticker = 'GOOG' for example
        # data_source = 'google' for example
        stock_data = DataReader(ticker, data_source, start, end)
        return stock_data
        # returns open high low close volume
        
    def plot_web_finance_stock_data(self,
                                    stock_data,
                                    price_type,
                                    ticker):
        
        import matplotlib.pyplot as plt
        plot_title = ticker + ' ' + price_type
        stock_data[price_type].plot(title=plot_title)
        plt.show()
    
    # use this if you have 2 pieces of data to plot
    def plot_multiple_data_one_plot(self,
                                    stocks_df,
                                    title,
                                    secondary_y):
        #secondary_y is the name of the column to plot on the secondary y axis
        import matplotlib.pyplot as plt
        stocks_df.plot(title=title, secondary_y=secondary_y)
        plt.tight_layout()
        plt.show()
    
    def get_FRED_data(self,
                      series_code,
                      start,
                      data_source='fred'):
        
        data = DataReader(series_code, data_source, start)
        return data
    
    def sort_stocks_by_market_cap(self,
                                    stocks_df):
        
        return stocks_df.sort_values('Market Capitalization', ascending=False)
        # nyse = nyse.set_index('Stock Symbol')
        # nyse['Market Capitalization'].idxmax() # index of max market cap
    
    def get_largest_blank_in_sector(self,
                                    stocks_df,
                                    sector,
                                    largest_what,
                                    how_many=1):
        
        listings = stocks_df.set_index('Symbol')
        result = listings.loc[listings.Sector==sector, largest_what].nlargest(n=how_many)
        symbols_list = result.index.tolist()
        
        return symbols_list
    
    def get_unique_sectors(self,
                           stocks_df):
        
        return stocks_df['Sector'].unique()
    
    def return_stocks_by_sector(self,
                                stocks_df):
        
        list_of_sectors = stocks_df['Sector'].unique()
        list_of_sector_df = []
        for sector in list_of_sectors:
            sector_df = stocks_df.loc[stocks_df.Sector==sector]
            list_of_sector_df.append(sector_df)
        return list_of_sector_df
    
    def get_largest_in_sector_IPO_year(self,
                                       stocks_df,
                                       sector,
                                       ipo_year):
        stocks_df = stocks_df.set_index('Symbol')
        ticker = stocks_df.loc[(stocks_df.Sector == sector) & (stocks_df['IPO Year']==ipo_year),\
                               'Market Capitalization'].idmax()
        return ticker
    
    def get_specified_columns_from_df(self,
                                      stocks_df,
                                      column_list):
        
        return stocks_df.loc[:, column_list]
        
        
    def get_sector_specific_cos(self,
                                stocks_df,
                                sector):
        
        return stocks_df.loc[stocks_df.Sector==sector]
        
    def select_stocks_by_sector_industry(self):
        
        return 1
    
    def select_stocks_by_IPO_year(self):
        
        return 1
    
    def rename_data_columns(self,
                            data,
                            column,
                            name):
        
        data = data.rename(columns={column: name})
        return data
    
    # just use describe()
    def calculate_quantiles(self,
                            stocks_df,
                            column_name,
                            q_val=[0.25,0.5,0.75,1.0]):
        
        #q_val can also be a np.arange list
        #0.5 quqntile is equal to the median
        # to plot .... stocks_df[column_name].quantile(q_val).plot(kind='bar', title='Quantiles')
        return stocks_df[column_name].quantile(q_val)
    
    # HISTOGRAM
    def seaborn_dist_plot(self,
                          stocks_df,
                          column_to_plot,
                          bins=20):
        
        stocks_df.dropna(inplace=True)
        import matplotlib.pyplot as plt
        import seaborn as sns
        ax=sns.distplot(stocks_df)
        ax.axvline(stocks_df[column_to_plot].median(), bins=bins, color='black', ls='--')
        plt.show()
    
    def seaborn_dist_plot_multi_column(self,
                                       stocks_df,
                                       bins=20):
        import seaborn as sns
        import matplotlib.pyplot as plt
        for column in stocks_df.columns:
            sns.distplot(stocks_df[column], label=column, bins=bins)
        plt.show()
    
    def redindex(self,
                 to_index): # to_index is of type DatetimeIndex
        
        return self.timeseries_series.redindex(to_index)
    
    def calc_log_returns(self):
        
        lrets = np.log(self.timeseries_series/self.timeseries_series.shift(1)).dropna()
        return lrets
        
    def rolling_std_dev(self,
                        window_size,
                        func=pdc.frame.DataFrame.std):    
    
        return self.generic_rolling_window_func(func, window_size)
    
    def rolling_mean(self,
                     window_size,
                     func=pdc.frame.DataFrame.mean):
        
        return self.generic_rolling_window_func(func, window_size)
    
    def rolling_variance(self,
                         window_size,
                         func=pdc.frame.DataFrame.var):
        
        return self.generic_rolling_window_func(func, window_size)
    
    def rolling_max(self,
                    window_size,
                    func=pdc.frame.DataFrame.max):
        
        return self.generic_rolling_window_func(func, window_size)
    
    def rolling_min(self,
                    window_size,
                    func=pdc.frame.DataFrame.min):
        
        return self.generic_rolling_window_func(func, window_size)
    
    def generic_rolling_window_func(self,
                                    func,
                                    window_size):
    
        return self.timeseries.rolling(window=window_size).apply(func)
    
    def mean_weekly_resample(self,
                             num_of_weeks_interval):
        
        resample_interval = str(num_of_weeks_interval)+'W'
        return self.generic_resampler(resample_interval,
                                      pdc.frame.DataFrame.mean)
    
    def mean_monthly_resample(self,
                              num_of_months_interval):
        
        resample_interval = str(num_of_months_interval)+'M'
        return self.generic_resampler(resample_interval,
                                      pdc.frame.DataFrame.mean)
    
    def mean_minute_resmaple(self,
                             num_of_minutes_interval):
        
        resample_interval = str(num_of_minutes_interval)+'T'
        return self.generic_resampler(resample_interval,
                                      pdc.frame.DataFrame.mean)
    
    def mean_daily_resample(self,
                            num_of_days_interval):
        
        resample_interval = str(num_of_days_interval)+'D'
        return self.generic_resampler(resample_interval,
                                      pdc.frame.DataFrame.mean)
    
    def mean_busday_resample(self,
                             num_of_busdays_interval):
        
        resample_interval = str(num_of_busdays_interval)+'B'
        return self.generic_resampler(resample_interval,
                                      pdc.frame.DataFrame.mean)
    
    def mean_hourly_resample(self,
                             num_of_hours_interval):
        
        resample_interval = str(num_of_hours_interval) + 'H'
        return self.generic_resampler(resample_interval,
                                      pdc.frame.DataFrame.mean)
    
    def mean_yearly_resample(self,
                             num_of_years_interval):
        
        resample_interval = str(num_of_years_interval) + 'A'
        return self.generic_resampler(resample_interval,
                                      pdc.frame.DataFrame.mean)
    
    def mean_quarterly_resample(self,
                                num_of_quarters_interval):
        
        resample_interval = str(num_of_quarters_interval) + 'Q'
        return self.generic_resampler(resample_interval,
                                      pdc.frame.DataFrame.mean)
    
    # example of func : sum(), mean(), ffill() (interpolation), bfill (backfill), etc...
    def generic_resampler(self,
                          resample_interval,
                          func):
        
        pdf_data = self.get_from_csv()
        return pdf_data.resample(resample_interval).apply(func)
    
#if __name__ == '__main__':
    
    #do = DataObject()
    #do.get_stock_universe_file_as_df()