'''
Created on Jan 24, 2018

@author: ghazy
'''
from root.nested import get_logger
from os_mux import OSMuxImpl
import os, requests
import pandas as pd
import numpy as np
from datetime import datetime

class DataProviderInterface(object):
    
    def __init__(self):
        
        self.logger = get_logger()
        
        local_misc_data_path = '/workspace/data/'
        local_stock_universe_file_pwd = OSMuxImpl.get_proper_path(local_misc_data_path)
       
        stock_universe_file_name = 'IWB_holdings'
        stock_universe_file_type = '.csv'
        self.total_pwd_stock_universe_file = local_stock_universe_file_pwd + \
            stock_universe_file_name + stock_universe_file_type
        self.last_time_stock_universe_file_modified = os.path.getmtime(self.total_pwd_stock_universe_file)
        
    def download_stock_universe(self):
        """ The stock universe here is the holdings of IWB, the Russell 1000
            The Russell 1000 is the 1000 largest companies (by market cap) in
            the USA.
        """
        ### Read the CSV file, IWB_holdings.csv
        url = "https://www.ishares.com/us/products/239707/ishares-russell-1000 \
            -etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
        response = requests.get(url)
        ### write out the response to file
        with open(self.total_pwd_stock_universe_file, 'wb') as f:
            f.write(response.content)
        self.logger.info("DataObject.get_stock_universe: HTTP Response Status Code %s " + str(response.status_code)) 
    
    def get_stock_universe_file_as_df(self):
        """ This is the main call to get the stock universe file into a pandas DataFrame
            Returns the dataframe of all stocks in the stock univerrse"""
        if self.does_stock_universe_file_exist() == False:
            self.download_stock_universe()
        if self.is_stock_universe_file_old() == True:
            self.download_stock_universe()
        ## therefore, get the file from csv
        suf_df = pd.read_csv(self.total_pwd_stock_universe_file, index_col='Ticker', delimiter=',', \
                             header=10) #skiprows=lambda x: x in range(0,11))
        return suf_df
    
    def get_last_time_stock_universe_file_modified(self):
        
        return self.last_time_stock_universe_file_modified
    
    def does_stock_universe_file_exist(self):
        
        b_exists = (os.path.exists(self.total_pwd_stock_universe_file))
        if (b_exists):
            # set the last modified time
            self.last_time_stock_universe_file_modified = datetime.fromtimestamp(os.path.getmtime(self.total_pwd_stock_universe_file))
        return b_exists
    
    def is_stock_universe_file_old(self):
        
        todays_date = datetime.now().date()
        if (todays_date > self.get_last_time_stock_universe_file_modified().date()):
            return True
        else:
            return False