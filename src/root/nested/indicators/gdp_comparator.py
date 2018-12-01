import pandas as pd

from quandl_data_object import QuandlDataObject

class GdpComparator(object):
    """description of class"""

    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    def get_gdp_data(self):

        qdo_us_gdp = QuandlDataObject('GDP',
                                      'US_QUARTERLY_REAL_GDP_VALUE_SEAS_ADJ',
                                      '.csv')
        qdo_df = qdo_us_gdp.get_df()

        return (qdo_df)

    def get_china_gdp_from_oecd(self):

        # china gdp is year over year or quarter over quarter.
        # us gdp is quarter over quarter, annualized, which is different.
        china_gdp_df = pd.DataFrame()
        return (china_gdp_df)


if __name__ == '__main__':

    gdp_c = GdpComparator()
    gdp_c.get_gdp_data()