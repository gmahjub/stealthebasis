import requests
import pandas as pd
from root.nested import get_logger


class IEXTradingApi:

    def __init__(self,
                 sec_type_list = ['cs', 'et']):

        self.logger = get_logger()
        self.iex_trading_root_url = "https://api.iextrading.com/1.0"
        self.get_symbols_universe_url = "/ref-data/symbols"
        self.sec_type_list = sec_type_list

    def get_symbols_universe(self):

        sym_univ_url = self.iex_trading_root_url + self.get_symbols_universe_url
        print (sym_univ_url)
        r = requests.get(sym_univ_url)
        self.logger.info("IEXTradingApi.get_symbols_universe: request to %s returned a status code of %s",
                         sym_univ_url, str(r.status_code))
        list_of_jsons = r.json()
        df = pd.DataFrame.from_dict(list_of_jsons, orient="columns")
        return_df = self.get_these_types_from_symbols_universe(df, type_list=self.sec_type_list)
        return return_df

    def get_these_types_from_symbols_universe(self,
                                              df,
                                              type_list=['cs', 'etf']):

        """ The types from IEX are as follows: ad = ADR, re = REIT, ce = closed end fund,
        si = secondary issue, lp = limited partnership, cs = common stock, et = ETF.
        More info @ https://iextrading.com/developer/docs/#symbols. When I pulled unique
        values with the below function, I get ['cs', 'N/A', 'et', 'ps', 'bo', 'su', 'crypto']."""

        boolean_series = df.type.isin(type_list)
        return df[boolean_series]

    def get_unique_sec_types_symbols_universe(self):

        df = self.get_symbols_universe()
        self.logger.info('IEXTradingApi:get_unique_sec_types_symbols_universe(): the unique security types'
                         'at IEX are %s', str(df.type.unique()))
        return df.type.unique() # return type is list


if __name__ == '__main__':

    iextapi = IEXTradingApi()
    df = iextapi.get_symbols_universe()
    iextapi.get_these_types_from_symbols_universe(type_list=['et'])
    unique_types = iextapi.get_unique_sec_types_symbols_universe()
    print ("IEX unique types are ", unique_types)
