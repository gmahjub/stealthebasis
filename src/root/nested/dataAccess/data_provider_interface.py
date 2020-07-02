'''
Created on Jan 24, 2018

@author: ghazy
'''

from root.nested import get_logger
from root.nested.SysOs.os_mux import OSMuxImpl
import os
import requests
import pandas as pd
from datetime import datetime

header = {
  "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
  "X-Requested-With": "XMLHttpRequest"
}
sp500_req_url = "https://etfdailynews.com/etf/spy/"
nq100_req_url = "https://etfdailynews.com/etf/qqq/"
dow30_req_url = "https://etfdailynews.com/etf/dia/"
r_sp = requests.get(sp500_req_url, headers=header)
r_nq = requests.get(nq100_req_url, headers=header)
r_dow = requests.get(dow30_req_url, headers=header)
SP500_HOLDINGS_URL = pd.read_html(r_sp.text, attrs={'id': 'etfs-that-own'})
NQ100_HOLDINGS_URL = pd.read_html(r_nq.text, attrs={'id': 'etfs-that-own'})
DOW30_HOLDINGS_URL = pd.read_html(r_dow.text, attrs={'id': 'etfs-that-own'})
RUSSELL1000_HOLDINGS_URL = "https://www.ishares.com/us/products/239707/ishares-russell-1000" \
                           "-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
RUSSELL3000_HOLDINGS_URL = "https://www.ishares.com/us/products/239714/ishares-russell-3000" \
                           "-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"


class DataProviderInterface(object):

    def __init__(self):

        self.logger = get_logger()

        local_misc_data_path = '/workspace/data/'
        self.local_stock_universe_file_pwd = OSMuxImpl.get_proper_path(local_misc_data_path)

        stock_universe_file_name = 'IWB_holdings'
        self.stock_universe_file_type = '.csv'
        self.total_pwd_stock_universe_file = self.local_stock_universe_file_pwd + \
                                             stock_universe_file_name + self.stock_universe_file_type
        try:
            last_time_stock_universe_file_modified = os.path.getmtime(self.total_pwd_stock_universe_file)
        except FileNotFoundError:
            last_time_stock_universe_file_modified = ""

        russell_1000_stock_universe = 'Russ1K_holdings'
        self.total_pwd_russell1000 = self.local_stock_universe_file_pwd + \
                                     russell_1000_stock_universe + self.stock_universe_file_type
        try:
            last_time_russell_1000_stock_universe_file_modified = os.path.getmtime(self.total_pwd_russell1000)
        except FileNotFoundError:
            last_time_russell_1000_stock_universe_file_modified = ""

        russell_3000_stock_universe = 'Russ3K_holdings'
        self.total_pwd_russell3000 = self.local_stock_universe_file_pwd + \
                                     russell_3000_stock_universe + self.stock_universe_file_type
        try:
            last_time_russell_3000_stock_universe_file_modified = os.path.getmtime(self.total_pwd_russell3000)
        except FileNotFoundError:
            last_time_russell_3000_stock_universe_file_modified = ""

        self.last_modified_times = {stock_universe_file_name: last_time_stock_universe_file_modified,
                                    russell_3000_stock_universe: last_time_russell_3000_stock_universe_file_modified,
                                    russell_1000_stock_universe: last_time_russell_1000_stock_universe_file_modified}

        self.stock_universe_download_func = {stock_universe_file_name: self.download_stock_universe,
                                             russell_1000_stock_universe: self.download_russell_1000_stock_universe,
                                             russell_3000_stock_universe: self.download_russell_3000_stock_universe}

    def download_stock_universe(self):
        """ The stock universe here is the holdings of IWB, the Russell 1000
            The Russell 1000 is the 1000 largest companies (by market cap) in
            the USA.
        """
        url = RUSSELL1000_HOLDINGS_URL
        response = requests.get(url)
        with open(self.total_pwd_stock_universe_file, 'wb') as f:
            f.write(response.content)
        self.logger.info("DataObject.get_stock_universe: HTTP Response Status Code %s " + str(response.status_code))

    def download_russell_1000_stock_universe(self):

        url = RUSSELL1000_HOLDINGS_URL
        response = requests.get(url)
        with open(self.total_pwd_russell1000, 'wb') as f:
            f.write(response.content)
        self.logger.info("DataObject.download_russell_1000_stock_universe: HTTP Response Status Code %s ",
                         str(response.status_code))
        error_response = response.raise_for_status()  # HTTP 4XX, 5XX
        if error_response is None:
            self.last_modified_times['Russ1K_holdings'] = os.path.getmtime(self.total_pwd_russell1000)
        else:
            self.logger.error("DataObject.download_russell_1000_stock_universe: faild download %s",
                              str(error_response))

    def download_russell_3000_stock_universe(self):

        url = RUSSELL3000_HOLDINGS_URL
        response = requests.get(url)
        with open(self.total_pwd_russell3000, 'wb') as f:
            f.write(response.content)
        self.logger.info("DataObject.download_russell_3000_stock_universe: HTTP Response Status Code %s ",
                         str(response.status_code))
        error_response = response.raise_for_status()  # Http 4XX, 5XX
        if error_response is None:
            self.last_modified_times['Russ3K_holdings'] = os.path.getmtime(self.total_pwd_russell3000)
        else:
            self.logger.error("DataObject.download_russsell_3000_stock_universe: failed download %s",
                              str(error_response))

    @staticmethod
    def download_sp500_holdings_symbol_list():

        url = SP500_HOLDINGS_URL
        # below code from IEX example @ https://github.com/timkpaine/pyEX
        sp500 = [x for x in url[0].Symbol.values.tolist() if isinstance(x, str)]
        # returns a list of tickers
        return sp500

    @staticmethod
    def download_nq100_holdings_symbol_list():

        url = NQ100_HOLDINGS_URL
        nq100 = [x for x in url[0].Symbol.values.tolist() if isinstance(x, str)]
        # returns a list of tickers
        return nq100

    @staticmethod
    def download_dow30_holdings_symbol_list():

        url = DOW30_HOLDINGS_URL
        dow30 = [x for x in url[0].Symbol.values.tolist() if isinstance(x, str)]
        return dow30

    def get_stock_universe_file_as_df(self,
                                      stock_universe_filename):

        """ This is the main call to get the stock universe file into a pandas DataFrame
            Returns the dataframe of all stocks in the stock univerrse"""

        if self.does_stock_universe_file_exist(stock_universe_filename) is False or \
                self.is_stock_universe_file_old(stock_universe_filename) is True:
            func = self.stock_universe_download_func[stock_universe_filename]
            self.logger.info("DataProviderInterface.get_stock_universe_file_as_df(%s): "
                             "running function %s with stock universe filename %s", stock_universe_filename, func)
            _ = func()
        full_path = self.local_stock_universe_file_pwd + stock_universe_filename + self.stock_universe_file_type
        self.logger.info("DataProviderInterface.get_stock_universe_file_as_df(%s): full path of stock "
                         "universe file is %s", stock_universe_filename, full_path)
        suf_df = pd.read_csv(full_path, index_col='Ticker', delimiter=',', header=10)
        return suf_df

    def get_last_time_stock_universe_file_modified(self,
                                                   stock_universe_filename):

        return self.last_modified_times[stock_universe_filename]

    def does_stock_universe_file_exist(self,
                                       stock_universe_filename):

        full_path = self.local_stock_universe_file_pwd + stock_universe_filename + self.stock_universe_file_type
        b_exists = (os.path.exists(full_path))
        if b_exists:
            self.last_modified_times[stock_universe_filename] = datetime.fromtimestamp(os.path.getmtime(full_path))
        return b_exists

    def is_stock_universe_file_old(self,
                                   stock_universe_filename):

        today_date = datetime.now().date()
        if today_date > self.get_last_time_stock_universe_file_modified(stock_universe_filename).date():
            return True
        else:
            return False
