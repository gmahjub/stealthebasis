import numpy as np
import pandas as pd

from root.nested import get_logger
from tiingo_data_object import TiingoDataObject
from quandl_data_object import QuandlDataObject


class MultiDataframeAnalysis(object):
    """description of class"""

    def __init__(self, **kwargs):

        self.logger = get_logger()
        return super().__init__(**kwargs)

    def tiingo_fred_dataframe_merge(self,
                                    tiingo_symbol_list,
                                    tiingo_start_date,
                                    tiingo_end_date,
                                    class_of_data, # from quandl_interface
                                    local_symbol): # from quandl_interface dictionaries

        source = 'Tiingo'
        tdo_df = TiingoDataObject(start_date = tiingo_start_date, end_date = tiingo_end_date, source = source, symbols = tiingo_symbol_list)
        qdo_df = QuandlDataObject(class_of_data, local_symbol, '.csv').get_df()
        return_df_dict = {}

        for tiingo_symbol in tiingo_symbol_list:
            tdo_df = tdo_df.get_px_data_df()[tiingo_symbol]
            tdo_df = tdo_df.reset_index()
            qdo_df = qdo_df.reset_index()
            merged = pd.merge_asof(tdo_df, qdo_df, left_on = 'date', right_on = 'Date')
            # aggregation function: ohlc(), mean(), last(), first() etc...
            merged = merged.resample('Q', on='date')[['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume']].ohlc()
            return_df_dict[tiingo_symbol] = merged
        
        return (return_df_dict)

if __name__ == '__main__':

    mdfa = MultiDataframeAnalysis()
    tiingo_symbol_list = ['SPY']
    merged_df_dict = mdfa.tiingo_fred_dataframe_merge(tiingo_symbol_list=tiingo_symbol_list, 
                                                      tiingo_start_date="2010-01-01", 
                                                      tiingo_end_date=str(pd.to_datetime('today')).split(' ')[0],
                                                      class_of_data = "GDP", 
                                                      local_symbol = "US_QUARTERLY_REAL_GDP_ANNUAL_RATE_SEAS_ADJ")
    for tiingo_symbol in tiingo_symbol_list:
        merged_df = merged_df_dict['SPY']
        print(merged_df.corr())