import requests
import pandas as pd
from root.nested import get_logger


class IEXTradingApi:

    def __init__(self):

        self.logger = get_logger()
        self.iex_trading_root_url = "https://api.iextrading.com/1.0"
        self.get_symbols_universe_url = "/ref-data/symbols"

    def get_symbols_universe(self):

        sym_univ_url = self.iex_trading_root_url + self.get_symbols_universe_url
        print (sym_univ_url)
        r = requests.get(sym_univ_url)
        self.logger.info("IEXTradingApi.get_symbols_universe: request to %s returned a status code of %s",
                         sym_univ_url, str(r.status_code))
        list_of_jsons = r.json()
        df = pd.DataFrame.from_dict(list_of_jsons, orient="columns")
        return df
