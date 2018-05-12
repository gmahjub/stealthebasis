'''
Created on Jan 24, 2018

@author: ghazy
'''

import pandas as pd
from data_provider_interface import DataProviderInterface
#from root.nested.data_provider_interface import DataProviderInterface
from yahoo_data_object import YahooDataObject
#from root.nested.yahoo_data_object import YahooDataObject

dpi = DataProviderInterface()
suf_df = dpi.get_stock_universe_file_as_df()
index_np_array = suf_df.index
end_date = str(pd.to_datetime('today')).split(' ')[0]
for idx in index_np_array:
    
    symbol_list = [idx]
    print (symbol_list)
    do_it = True
    if do_it == True:
        ydo = YahooDataObject('2007-01-01',
                              end_date,
                              symbol_list)
        ydo.symbols = idx
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