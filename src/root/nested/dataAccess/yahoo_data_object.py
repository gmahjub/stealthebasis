'''
Created on Dec 13, 2017

@author: ghazy
'''
import os
from datetime import datetime
import pandas as pd
import pandas_datareader.data as web
import numpy as np

from data_object import DataObject
#from root.nested.data_object import DataObject
from os_mux import OSMuxImpl
#from root.nested.os_mux import OSMuxImpl
from root.nested import get_logger

class YahooDataObject(DataObject):
    
    def __init__(self,
                 start_date,
                 end_date,
                 symbols):
        
        super().__init__()
        self.logger = get_logger()
        
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        
        self.get_px_adj_close = lambda x: web.DataReader(x, 'yahoo', start=self.start_date, end=self.end_date)['Adj Close']
        self.get_px_open = lambda x: web.DataReader(x, 'yahoo', start=self.start_date, end=self.end_date)['Open']
        self.get_px_high = lambda x: web.DataReader(x, 'yahoo', start=self.start_date, end=self.end_date)['High']
        self.get_px_low = lambda x: web.DataReader(x, 'yahoo', start=self.start_date, end=self.end_date)['Low']
        self.get_volume = lambda x: web.DataReader(x, 'yahoo', start=self.start_date, end=self.end_date)['Volume']
        
        self.adj_close_px_df = None
        self.open_px_df = None
        self.high_px_df = None
        self.low_px_df = None
        self.volume_df = None
        
        self.local_yahoo_data_path = '/workspace/data/yahoo/'
        self.local_file_type = '.csv'
        self.local_data_file_pwd =  OSMuxImpl.get_proper_path(self.local_yahoo_data_path)
        
        self.local_adj_close_file_name = '_'.join(self.symbols) + '_AdjClose' + self.local_file_type
        self.local_open_file_name = '_'.join(self.symbols) + '_Open' + self.local_file_type
        self.local_high_file_name = '_'.join(self.symbols) + '_High' + self.local_file_type
        self.local_low_file_name = '_'.join(self.symbols) + '_Low' + self.local_file_type
        self.local_volume_file_name = '_'.join(self.symbols) + '_Volume' + self.local_file_type
        
        self.logger.info("YahooDataObject.__init__.local_data_file_pwd: %s", str(self.local_data_file_pwd))
        
        self.yahoo_client_id = "dj0yJmk9Q09ZdnVWMlNEdzdxJmQ9WVdrOWNFTnNRMFV3TkRRbWNHbzlNQS0tJnM9Y29uc3VtZXJzZWNyZXQmeD1mNA--"
        self.yahoo_client_secret = "41f7939217e13b297ce3862be55c5b9e4b77cab8"
    
    # getter methods
    def get_adj_close_px_df(self):
        return self.adj_close_px_df
    
    def get_open_px_df(self):
        return self.open_px_df
    
    def get_high_px_df(self):
        return self.high_px_df
    
    def get_low_px_df(self):
        return self.low_px_df
    
    def get_volume_df(self):
        return self.volume_df
    ## end getter methods
    
    def get_adjust_close_prices(self):
        
        adj_close_px_df= pd.DataFrame({sym:self.get_px_adj_close(sym) for sym in self.symbols})
        self.adj_close_px_df = adj_close_px_df
        self.adj_close_to_csv()
        return adj_close_px_df
    
    def get_open_prices(self):
        
        open_px_df = pd.DataFrame({sym:self.get_px_open(sym) for sym in self.symbols})
        self.open_px_df = open_px_df
        self.open_to_csv()
        return open_px_df
    
    def get_high_prices(self):
        
        high_px_df = pd.DataFrame({sym:self.get_px_high(sym) for sym in self.symbols})
        self.high_px_df = high_px_df
        self.high_to_csv()
        return high_px_df
    
    def get_low_prices(self):
        
        low_px_df = pd.DataFrame({sym:self.get_px_low(sym) for sym in self.symbols})
        self.low_px_df = low_px_df
        self.low_to_csv()
        return low_px_df
    
    def get_eod_volume(self):
        
        volume_df = pd.DataFrame({sym:self.get_volume(sym) for sym in self.symbols})
        self.volume_df = volume_df
        self.volume_to_csv()
        return volume_df
    
    def get_log_return_prices(self,
                              price_type):
        
        price_type_dict = {'AdjClose': self.get_adjust_close_prices,
                           'Open': self.get_open_prices,
                           'High': self.get_high_prices,
                           'Low': self.get_low_prices}
        func = price_type_dict[price_type]
        
        return np.log(func()/func().shift(1)).dropna()
    
    def adj_close_to_csv(self):
        
        self.adj_close_px_df.to_csv(self.local_data_file_pwd + self.local_adj_close_file_name)
    
    def open_to_csv(self):
        
        self.open_px_df.to_csv(self.local_data_file_pwd + self.local_open_file_name)
        
    def low_to_csv(self):
        
        self.low_px_df.to_csv(self.local_data_file_pwd + self.local_low_file_name)
        
    def high_to_csv(self):
        
        self.high_px_df.to_csv(self.local_data_file_pwd + self.local_high_file_name)
    
    def volume_to_csv(self):
        
        self.volume_df.to_csv(self.local_data_file_pwd + self.local_volume_file_name)
        
    def adj_close_from_csv(self):
        
        self.adj_close_px_df = pd.read_csv(self.local_data_file_pwd + self.local_adj_close_file_name, parse_dates=['Date'], index_col='Date')
        self.logger.info("YahooDataObject.adj_close_from_csv(): Dataframe Columns %s ", str(self.adj_close_px_df.columns))
        
    def high_from_csv(self):
        
        self.high_px_df = pd.read_csv(self.local_data_file_pwd + self.local_high_file_name, parse_dates=['Date'], index_col='Date')
        
    def low_from_csv(self):
        
        self.low_px_df = pd.read_csv(self.local_data_file_pwd + self.local_low_file_name, parse_dates=['Date'], index_col='Date')
        
    def open_from_csv(self):
        
        self.open_px_df = pd.read_csv(self.local_data_file_pwd + self.local_open_file_name, parse_dates=['Date'], index_col='Date')
    
    def volume_from_csv(self):
        
        self.volume_df = pd.read_csv(self.local_data_file_pwd + self.local_volume_file_name, parse_dates=['Date'], index_col='Date')
    
    def get_last_time_file_modified(self):
        
        return self.last_time_file_modified    
    
    def is_local_file_old(self):
        
        todays_date = datetime.now().date()
        if (todays_date > self.get_last_time_file_modified().date()):
            return True
        else:
            return False
        
    def does_local_file_exist(self,
                              price_type):
        
        price_type_dict = {'AdjClose': self.local_adj_close_file_name,
                           'Open' : self.local_open_file_name,
                           'High' : self.local_high_file_name,
                           'Low' : self.local_low_file_name,
                           'Volume' : self.local_volume_file_name}
        
        file_to_find = self.local_data_file_pwd + price_type_dict[price_type]
        b_exists = (os.path.exists(file_to_find))
        if (b_exists):
            # set the last modified time
            self.last_time_file_modified = datetime.fromtimestamp(os.path.getmtime(file_to_find))
        return b_exists
    
    def calc_log_returns(self,
                         price_type,
                         return_interval=1):
        
        price_type_dict = {'AdjClose': self.adj_close_px_df,
                           'Open': self.open_px_df,
                           'High': self.high_px_df,
                           'Low': self.low_px_df}
        df_to_use = price_type_dict[price_type]
        
        return np.log(df_to_use/df_to_use.shift(return_interval)).dropna()

if __name__ == '__main__':
    
    symbols_list = ['SPY']
    end_date = str(pd.to_datetime('today')).split(' ')[0]
    ydo = YahooDataObject('2007-01-01',
                          end_date,
                          symbols_list)
    
    ydo.symbols = symbols_list
    price_type_list = ['AdjClose','Open','High','Low','Volume']
    price_type_dict = {'AdjClose': [ydo.get_adjust_close_prices, ydo.adj_close_from_csv, ydo.adj_close_px_df],
                       'Open': [ydo.get_open_prices, ydo.open_from_csv, ydo.open_px_df],
                       'High': [ydo.get_high_prices, ydo.high_from_csv, ydo.high_px_df],
                       'Low': [ydo.get_low_prices, ydo.low_from_csv, ydo.low_px_df],
                       'Volume': [ydo.get_eod_volume, ydo.volume_from_csv, ydo.volume_df] }
    
    for price_type in price_type_list:
        func_0 = price_type_dict[price_type][0]
        func_1 = price_type_dict[price_type][1]
        df_name = price_type_dict[price_type][2]
        if ydo.does_local_file_exist(price_type):
            if ydo.is_local_file_old():
                ydo.logger.info("Pulling new dataframes from yahoo...")
                _ = func_0()
            else:
                ydo.logger.info("Pulling dataframes from local csv files...")
                df_name = func_1()
        else:
            ydo.logger.info("Pulling new dataframes from yahoo...")
            _ = func_0()
         
