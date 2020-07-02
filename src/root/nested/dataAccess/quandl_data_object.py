'''
Created on Dec 1, 2017

@author: traderghazy
'''

from root.nested.dataAccess.quandl_interface import QuandlSymbolInterface
from root.nested.dataAccess.data_object import DataObject
from root.nested.SysOs.os_mux import OSMuxImpl

import os
import numpy as np
import pandas as pd
import quandl
from datetime import datetime


class QuandlDataObject(DataObject):
    
    def __init__(self,
                 class_of_data,
                 local_symbol,
                 local_file_type):

        # class of data
        ################################
        # 'FOREX'
        # 'EURODOLLARS'
        # 'INTEREST_RATES'
        # 'ECONOMIC_INDICATORS_UNADJ'
        # 'ECONOMIC_INDICATORS_SEAS_ADJ'
        # 'FED_FORECASTS'
        # 'MISC'
        # 'STOCKS'

        # local_symbol
        ################################
        # See the static dictionaries in quandl_interface.py.
        # The local symbol is the key in the key,value pairs that make up those non-reversed dictionaries.
        
        # local_file_type
        ################################
        # usually .csv

        super().__init__()
        self.qsi = QuandlSymbolInterface()
        self.class_of_data = class_of_data
        self.local_symbol = local_symbol
        self.local_file_type = local_file_type
        self.quandl_symbol = self.qsi.get_quandl_symbol(class_of_data,
                                                        local_symbol)
        
        self.local_data_file_pwd = OSMuxImpl.get_proper_path(self.qsi.get_local_quandl_data_path(class_of_data))
        self.logger.info("QuandlDataObject.__init__.local_data_file_pwd " + self.local_data_file_pwd)
        self.local_data_file_name = local_symbol + self.local_file_type
        self.newest_date_at_quandl_dt = None
        self.last_time_file_modified = None
        if self.does_local_file_exist() and self.is_local_file_old() == False:
            self.df = self.get_df_from_local_csv()
        else:
            self.df = self.get_from_quandl()
    
    def get_df(self):
        
        return self.df
    
    def set_df(self,
               df):
        
        self.df = df
        
    def get_df_index_freq(self):
        
        return self.df.index.freq
    
    def get_df_from_local_csv(self):
        
        file_to_find = self.local_data_file_pwd + self.local_data_file_name
        return pd.read_csv(file_to_find, parse_dates=['Date'], index_col='Date')
    
    def get_last_date_in_file(self):
        
        quandl_df = self.get_df_from_local_csv()
        last_date_in_file = np.array(quandl_df.tail(1).index)[0]
        last_date_in_file = last_date_in_file.astype('datetime64[D]')
        self.last_date_in_file = last_date_in_file
        
        return self.last_date_in_file
    
    def does_local_file_exist(self):
        
        file_to_find = self.local_data_file_pwd + self.local_data_file_name
        b_exists = (os.path.exists(file_to_find))
        if (b_exists):
            # set the last modified time
            self.last_time_file_modified = datetime.fromtimestamp(os.path.getmtime(file_to_find))
        return b_exists
    
    def df_to_csv(self):
        
        f = self.local_data_file_pwd + self.local_data_file_name
        if (os.path.exists(f)):
            os.remove(f)
        self.get_df().to_csv(f)
        self.logger.info("QuandleDataObject.df_to_csv(): Dataframe written to file %s ", f)
    
    def get_from_quandl(self):
        
        quandl_df = quandl.get(self.quandl_symbol, authtoken=self.qsi.quandl_auth_token, collapse='daily')
        quandl_df.to_csv(self.local_data_file_pwd + self.local_data_file_name)
        self.newest_daily_freq_date_at_quandl_dt = quandl_df.index[-1].to_pydatetime()
        return quandl_df
    
    def get_from_quandl_date_range(self,
                                    start_date,
                                    end_date):
        
        return quandl.get(self.quandl_symbol, authtoken=self.qsi.quandl_auth_token, start_date=start_date, end_date=end_date )
    
    # return data in a numpy.recarray format, (datetime.datetime, value)
    def get_from_quandl_numpy_array(self):
        
        return quandl.get(self.quandl_symbol, authtoken=self.qsi.quandl_auth_token, returns="numpy")
        
    def get_DF_quarterly_sampling_freq(self):
        
        df_quarterly = quandl.get(self.quandl_symbol, authtoken=self.qsi.quandl_auth_token, collapse='quarterly')
        self.newest_quarterly_freq_date_at_quandl_dt = df_quarterly.index[-1].to_pydatetime()
        return df_quarterly
    
    def get_DF_annual_sampling_freq(self):
        
        df_yearly = quandl.get(self.quandl_symbol, authtoken=self.qsi.quandl_auth_token, collapse='annual')
        self.newest_yearly_freq_date_at_quandl_dt = df_yearly.index[-1].to_pydatetime()
        return df_yearly
    
    def get_DF_monthly_sampling_freq(self):
        
        df_monthly = quandl.get(self.quandl_symbol, authtoken=self.qsi.quandl_auth_token, collapse="monthly")
        self.newest_monthly_freq_date_at_quandl_dt = df_monthly.index[-1].to_pydatetime()
        return df_monthly
    
    def get_DF_weekly_sampling_freq(self):
        
        df_weekly = quandl.get(self.quandl_symbol, authtoken=self.qsi.quandl_auth_token, collapse='weekly')
        self.newest_weekly_freq_date_at_quandl_dt = df_weekly.index[-1].to_pydatetime()
        return df_weekly
    
    def get_DF_difference_transform(self):
        
        return quandl.get(self.quandl_symbol, authtoken=self.qsi.quandl_auth_token, transformation='diff')
    
    def get_DF_percent_diff_transform(self):
        
        return quandl.get(self.quandl_symbol, authtoken=self.qsi.quandl_auth_token, transformation='rdiff')
    
    # also known as normalized to 100
    # useful when looking at data over a specified time range
    def get_DF_indexed_transform(self):
        
        return quandl.get(self.quandl_symbol, authtoken=self.qsi.quandl_auth_token, transformation='normalize')
    
    def get_DF_cum_sum_transform(self):
        
        return quandl.get(self.quandl_symbol, authtoken=self.qsi.quandl_auth_token, transformation="cumul")
    
    def get_from_csv(self):
        
        file_to_find = self.local_data_file_pwd + self.local_data_file_name
        if self.does_local_file_exist():
            quandl_df = pd.read_csv(file_to_find, parse_dates=['Date'], index_col='Date')
        else:
            quandl_df = self.get_from_quandl()
        
        return quandl_df
    
    def get_last_time_file_modified(self):
        
        return self.last_time_file_modified
    
    def is_local_file_old(self):
        
        self.get_last_date_in_file()
        todays_date = np.datetime64('now')
        todays_date = todays_date.astype('datetime64[D]')
        self.logger.info('QuandlDataObject.is_local_file_old(): Filename: %s Today: %s Last Modification Date %s', str(self.local_data_file_name), str(todays_date), str(self.last_time_file_modified))
        a=True
        try:
            a = todays_date > self.get_last_time_file_modified()
        except TypeError:
            todays_date = datetime.now()
            a = todays_date > self.get_last_time_file_modified()
        return a

    def update_csv_file(self):
        return self.get_from_quandl()
    
    
# currently the custom resample is just calculating the mean, but we just want to show this is possible.
def custom_resampler(array_like):
    
    # this is the same as saying ".mean()" in the resample method.
    return np.mean(array_like)


if __name__ == '__main__':

    qdo_eurodollar = QuandlDataObject('EURODOLLARS',
                                      'ED4_WHITE',
                                      '.csv')
    
    qdo_euro = QuandlDataObject('FOREX_TO_USD',
                                'EUR_USD_spot',
                                '.csv')
    qdo_yen = QuandlDataObject('FOREX_TO_USD',
                               'JPY_USD_spot',
                               '.csv')
    qdo_gbp = QuandlDataObject('FOREX_TO_USD',
                               'GBP_USD_spot',
                               '.csv')
