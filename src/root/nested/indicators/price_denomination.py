import pandas as pd

from root.nested.dataAccess.yahoo_data_object import YahooDataObject
from root.nested.dataAccess.tiingo_data_object import TiingoDataObject
from root.nested.dataAccess.quandl_data_object import QuandlDataObject

from root.nested import get_logger

class PriceDenomination(object):
    """description of class"""

    def __init__(self, **kwargs):
        
        self.logger = get_logger()
        return super().__init__(**kwargs)

    def get_stock_px(self,
                     symbols,
                     start_date,
                     end_date=str(pd.to_datetime('today')).split(' ')[0],
                     source="Tiingo",
                     input_px_type_list = ['All']):

        params_dict = {"symbols": symbols, "start_date": start_date, "end_date": end_date, "source":source}
        tdo = TiingoDataObject(start_date = start_date, end_date = end_date, source = source, symbols = symbols)
        stock_px_df = tdo.get_px_data_df(input_px_type_list=input_px_type_list)

        return (stock_px_df[symbols[0]])
        
    def get_fx_px(self,
                  data_class,
                  local_symbol,
                  file_type):

        fx_obj = QuandlDataObject(class_of_data=data_class,
                                  local_symbol=local_symbol,
                                  local_file_type=file_type)
        fx_px_df = fx_obj.get_df()
        
        return (fx_px_df)

    def get_index_px(self,
                     data_class,
                     local_symbol,
                     file_type):

        index_obj = QuandlDataObject(class_of_data = data_class,
                                     local_symbol=local_symbol,
                                     local_file_type=file_type)
        idx_px_df = index_obj.get_df()


    def get_common_dates_df(self,
                            stock_px_df,
                            fx_px_df):

        common_dates = stock_px_df.index.intersection(fx_px_df.index)
        fx_px_df = fx_px_df.reindex(common_dates)
        fx_px_df = fx_px_df.dropna()
        stock_px_df = stock_px_df.reindex(common_dates)
        stock_px_df = stock_px_df.dropna()

        return_dict = {}
        return_dict['fx_px_df'] = fx_px_df
        return_dict['stock_px_df'] = stock_px_df
        
        return (return_dict)

    def convert_denomination(self,
                             stock_px_df,
                             fx_px_df):

        values_dict = self.get_common_dates_df(stock_px_df, fx_px_df)
        reindexed_stock_px_df = values_dict['stock_px_df'][['adjClose', 'adjHigh', 'adjLow', 'adjOpen']]
        converted_px = reindexed_stock_px_df.multiply(values_dict['fx_px_df']['Value'], axis = 'rows')
        
        return (converted_px)

if __name__ == '__main__':

    pd = PriceDenomination()
    stock_px = pd.get_stock_px(['SPY'],
                               start_date = '2010-01-01')
    fx_px = pd.get_fx_px(data_class = 'FOREX_TO_USD',
                         local_symbol = 'EUR_USD_spot',
                         file_type = '.csv')
    converted_px = pd.convert_denomination(stock_px,
                                           fx_px)
    



