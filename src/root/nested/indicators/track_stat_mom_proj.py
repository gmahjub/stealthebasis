import numpy as np
import pandas as pd
from root.nested.dataAccess.data_provider_interface import DataProviderInterface


class TrackStatMomProj:

    def __init__(self,
                 stock_universe_filename='Russ3K_holdings'):

        self.stock_universe_data = None
        self.stock_universe_filename = stock_universe_filename

    def get_stock_universe(self):

        dpi = DataProviderInterface()
        df = dpi.get_stock_universe_file_as_df(self.stock_universe_filename)
        return (df)


tsmp = TrackStatMomProj()
df = tsmp.get_stock_universe()
print(df.head())
