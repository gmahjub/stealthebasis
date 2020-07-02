import os
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
from root.nested.dataAccess.data_object import DataObject
from root.nested.SysOs.os_mux import OSMuxImpl

from root.nested import get_logger
from root.nested.dataAccess.secure_keys_access import SecureKeysAccess

'''
Created on Jan 17, 2018

@author: ghazy
'''


class TiingoDataObject(DataObject):
    """description of class"""

    def __init__(self, **kwargs):

        self.logger = get_logger()
        self.file_mod_time_dict = {}
        self.symbols = None
        self.start_date = ""
        self.end_date = ""
        self.source = "tiingo"
        self.api_key = SecureKeysAccess.get_vendor_api_key_static(vendor=str.upper(self.source))

        for key, value in kwargs.items():

            if key == 'symbols':
                self.symbols = value
                self.logger.info("TiingoDataObject.__init__.symbols: %s", str(self.symbols))
            elif key == 'start_date':
                self.start_date = value
                self.logger.info("TiingoDataObject.__init__.start_date: %s", str(self.start_date))
            elif key == 'end_date':
                self.end_date = value
                self.logger.info("TiingoDataObject.__init__.end_date: %s", str(self.end_date))
            elif key == 'source':
                self.source = value
                self.logger.info("TiingoDataObject.__init__.source: %s", str(self.source))

        self.get_px_all_tiingo = lambda x: web.get_data_tiingo(x,
                                                               start='2010-01-01',
                                                               end=str(pd.to_datetime('today')).split(' ')[0],
                                                               api_key=self.api_key)
        self.all_px_df = None
        self.local_tiingo_data_path = '/workspace/data/tiingo/stocks/'
        self.local_file_type = '.csv'
        self.local_data_file_pwd = OSMuxImpl.get_proper_path(self.local_tiingo_data_path)

        # return super().__init__(**kwargs)

    def tiingo_search_for_sym(self):

        import requests
        headers = {
            'Content-Type': 'application/json'
        }
        requestResponse = requests.get(
            "https://api.tiingo.com/tiingo/utilities/search?query=spx&token=" + self.api_key,
            headers=headers)
        print(requestResponse.json())

    def get_all_px_single(self,
                          sym,
                          start_date=None,
                          end_date=None):

        print(sym)
        dict_of_dataframes = {sym: self.get_px_all_tiingo(sym)}
        df = pd.DataFrame(dict_of_dataframes[sym])
        df.reset_index(inplace=True)
        df.set_index('date', inplace=True)
        df.to_csv(self.local_data_file_pwd + str(sym) + self.local_file_type, encoding='utf-8')
        if start_date is None:
            return df
        elif end_date is None:
            return df.loc[start_date:]
        else:
            return df.loc[start_date:end_date]

    def get_all_px_batch(self):

        dict_of_dataframes = {sym: [self.get_px_all_tiingo(sym)] for sym in self.symbols}
        return_dict = {}
        for sym in self.symbols:
            df = pd.DataFrame(dict_of_dataframes[sym][0])
            df.reset_index(inplace=True)
            df.set_index('date', inplace=True)
            return_dict[sym] = df
        self.all_px_df = return_dict
        self.all_px_to_csv()
        return return_dict

    def get_all_px_df(self):

        return self.all_px_df

    def all_px_to_csv(self):

        for key, value in self.all_px_df.items():
            value.to_csv(self.local_data_file_pwd + str(key) + self.local_file_type, encoding='utf-8')

    def all_px_from_csv(self,
                        total_file_path,
                        start_date,
                        end_date):

        self.logger.info("TiingoDataObject.all_px_from_csv(): total_file_path %s ", str(total_file_path))
        all_px_df = pd.read_csv(total_file_path, parse_dates=True, index_col='date')
        return_df = all_px_df.loc[start_date: end_date]
        self.logger.info("TiingoDataObject.all_px_from_csv(): Dataframe Columns %s ", str(all_px_df.columns))
        return return_df

    def does_local_file_exist(self,
                              px_type,
                              total_file_path):

        self.logger.info("TiingoDataObject.does_local_file_exist(): Checking for existence of %s ", total_file_path)
        file_to_find = total_file_path
        b_exists = (os.path.exists(file_to_find))
        if b_exists:
            # set the last modified time
            self.file_mod_time_dict[file_to_find] = datetime.fromtimestamp(os.path.getmtime(file_to_find))
        return b_exists

    def is_local_file_old(self,
                          total_file_path):

        todays_date = datetime.now().date()
        if todays_date > self.file_mod_time_dict[total_file_path].date():
            return True
        else:
            return False

    def get_last_time_file_modified(self):

        return self.last_time_file_modified

    def get_px_data_df(self,
                       start_date,
                       end_date,
                       input_px_type_list=['All']):

        px_type_dict = {'All': [self.get_all_px_single, self.all_px_from_csv, self.all_px_df]}
        return_dict = {}

        for px_type in input_px_type_list:
            func_0 = px_type_dict[px_type][0]
            func_1 = px_type_dict[px_type][1]

            for sym in self.symbols:
                local_all_file_name = sym + self.local_file_type
                total_file_path = self.local_data_file_pwd + local_all_file_name
                self.logger.info("TiingoDataObject.get_px_data_df().total_file_path: %s", str(total_file_path))

                if self.does_local_file_exist(px_type, total_file_path):
                    if self.is_local_file_old(total_file_path):
                        self.logger.info("Pulling new dataframes from Tiingo...")
                        return_dict[sym] = func_0(sym, start_date, end_date)
                    else:
                        self.logger.info("Pulling dataframes from local csv files...")
                        return_dict[sym] = func_1(total_file_path, start_date, end_date)
                else:
                    self.logger.info("Pulling new dataframes from Tiingo...")
                    return_dict[sym] = func_0(sym, start_date, end_date)

        return (return_dict)


if __name__ == '__main__':
    input_px_type_list = ['All']
    symbols = ['SPY', 'IWM']
    start_date = "2011-01-01"
    end_date = str(pd.to_datetime('today')).split(' ')[0]
    source = "Tiingo"
    params_dict = {"symbols": symbols, "start_date": start_date, "end_date": end_date, "source": source}
    mdo = TiingoDataObject(start_date=start_date, end_date=end_date, source=source, symbols=symbols)

    px_type_list = ['AdjClose', 'Open', 'High', 'Low', 'Volume', 'All']
    px_type_dict = {  # 'AdjClose': [mdo.get_adjust_close_prices, mdo.adj_close_from_csv, mdo.adj_close_px_df],
        # 'Open': [mdo.get_open_prices, mdo.open_from_csv, mdo.open_px_df],
        # 'High': [mdo.get_high_prices, mdo.high_from_csv, mdo.high_px_df],
        # 'Low': [mdo.get_low_prices, mdo.low_from_csv, mdo.low_px_df],
        # 'Volume': [mdo.get_eod_volume, mdo.volume_from_csv, mdo.volume_df],
        'All': [mdo.get_all_px_single, mdo.all_px_from_csv, mdo.all_px_df]}

    for px_type in input_px_type_list:
        func_0 = px_type_dict[px_type][0]
        func_1 = px_type_dict[px_type][1]
        df_name = px_type_dict[px_type][2]

        for sym in symbols:
            local_all_file_name = sym + mdo.local_file_type
            total_file_path = mdo.local_data_file_pwd + local_all_file_name
            mdo.logger.info("TiingoDataObject.__init__.local_data_file_pwd: %s", str(total_file_path))

            if mdo.does_local_file_exist(px_type, total_file_path):
                if mdo.is_local_file_old(total_file_path):
                    mdo.logger.info("Pulling new dataframes from Tiingo...")
                    _ = func_0(sym)
                else:
                    mdo.logger.info("Pulling dataframes from local csv files...")
                    df_name = func_1(total_file_path, start_date, end_date)
            else:
                mdo.logger.info("Pulling new dataframes from Tiingo...")
                _ = func_0(sym)
