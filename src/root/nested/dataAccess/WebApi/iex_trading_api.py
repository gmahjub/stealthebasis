import requests
import pandas as pd
import pyEX as pyex

from root.nested import get_logger
from root.nested.SysOs.os_mux import OSMuxImpl
from root.nested.visualize.extend_bokeh import ExtendBokeh

class IEXTradingApi:

    def __init__(self,
                 sec_type_list = ['cs', 'et']):

        self.logger = get_logger()
        self.iex_trading_root_url = "https://api.iextrading.com/1.0"
        self.get_symbols_universe_url = "/ref-data/symbols"
        self.sec_type_list = sec_type_list
        self.master_sector_indusry_df = pd.DataFrame(columns = ['Sector', 'Industry'])
        self.master_sector_industry_file = OSMuxImpl.get_proper_path('/workspace/data/IEX/') + 'master_sector_industry.csv'
        self.iex_html_path = OSMuxImpl.get_proper_path('/workspace/data/IEX/html/')
        self.co_earnings_path = OSMuxImpl.get_proper_path('/workspace/data/IEX/earnings/')

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
                                              type_list=['cs', 'et']):

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

    def create_sector_industry_universe(self):

        df = iextapi.get_symbols_universe()
        df.reset_index(inplace=True)
        df_small = df.iloc[0:5,]
        self.master_sector_indusry_df.to_csv(self.master_sector_industry_file)

    def get_co_sector_industry_tags(self,
                                    row):

        ticker = row.symbol
        self.logger.info("IEXTradingApi.get_co_sector_industry_tags(): getting sector, industry, tags for ticker %s", ticker)
        df = pyex.companyDF(symbol = ticker)
        co_sector_industry_tags = (df.loc[ticker, ['sector', 'industry', 'tags', 'companyName', 'website']])
        co_sector_industry_tags['symbol'] = ticker
        # co_sector_industry_tags is a pandas Series
        # access the symbol (ticker) by using ".name" of pd.Series object co_sector_industry_tags
        alt_industries = co_sector_industry_tags['tags']
        rows_to_insert = []
        for alt_industry in alt_industries:
            if alt_industry != co_sector_industry_tags['sector'] and alt_industry != co_sector_industry_tags['industry']:
                rows_to_insert.append([co_sector_industry_tags['sector'], alt_industry])
        rows_to_insert.append([co_sector_industry_tags['sector'], co_sector_industry_tags['industry']])
        for sector_industry in rows_to_insert:
            if self.master_sector_indusry_df.index.size == 0:
                next_index = 0
            else:
                next_index = self.master_sector_indusry_df.index[-1] + 1
            if sector_industry[0] != '' and sector_industry[1] != '':
                self.master_sector_indusry_df.loc[next_index] = sector_industry
        print (self.master_sector_indusry_df)

    def get_co_peers(self,
                     ticker):

        print (pyex.peersDF(symbol = ticker))

    def format_market_cap(self,
                          mc):

        if mc >= 1.0:
            return (str(mc) + 'B')
        in_millions = mc*1000.0
        return_val = str(in_millions) + 'M'
        return (return_val)

    def format_announce_time(self,
                             announce_time):

        announce_time_dict = {'BTO': 'Before Open',
                              'AMC': 'After Close'}
        return (announce_time_dict[announce_time])

    def format_headline(self,
                        headline):

        return (headline)

    def co_earnings_today(self):

        from datetime import datetime
        today_ymd = datetime.now().strftime("%Y%m%d")
        df = pyex.earningsTodayDF()

        if df.empty is True:
            # check for an existing flat csv file, maybe we did a pull already
            df = pd.read_csv(self.co_earnings_path + today_ymd + ".csv", sep=',', header=0)
            df.set_index('symbol', inplace=True)
            df.reset_index(inplace=True)
        else:
            df.to_csv(self.co_earnings_path + today_ymd + ".csv", sep=',')
            df.reset_index(inplace=True)

        market_cap_billions = df['quote.marketCap'].astype('float64')/1000000000.0
        market_cap_billions = market_cap_billions.round(3)
        df['Market Cap'] = market_cap_billions.apply(lambda x: self.format_market_cap(x) )
        df['Report Date'] = df['EPSReportDate'].astype('str')
        df['Announce Time'] = df['announceTime'].apply(lambda x: self.format_announce_time(x))
        df['Consensus EPS'] = df['consensusEPS']
        df['Estimated 1-Yr Change, EPS'] = df['estimatedChangePercent'].mul(100).round(2).astype('str') + '%'
        df['EPS Estimate'] = '$' + df['estimatedEPS'].astype('str')
        df['Quarter End'] = df['fiscalEndDate'].astype('str')
        df['Quarter'] = df['fiscalPeriod']
        df['Headline'] = df['headline'].apply(lambda x: self.format_headline(x))
        df['# of Estimates'] = df['numberOfEstimates'].apply

        list_of_columns = [ 'symbol',
                            'Report Date',
                            'Announce Time',
                            'Consensus EPS',
                            'Estimated 1-Yr Change, EPS',
                            'EPS Estimate',
                            'Quarter End',
                            'Quarter',
                            'Headline',
                            'numberOfEstimates',
                            'Market Cap',
                            'quote.open',
                            'quote.peRatio',
                            'quote.sector',
                            'quote.week52High',
                            'quote.week52Low',
                            'quote.ytdChange',
                            'yearAgo']
        df = df[list_of_columns]
        output_html_filename = self.iex_html_path + 'co_earnings_' + today_ymd + '.html'
        data_table = ExtendBokeh.bokeh_data_table(dataframe=df)
        ExtendBokeh.show_data_table(data_table,
                                    html_output_file=output_html_filename,
                                    html_output_file_title='CoEarningsToday')


if __name__ == '__main__':

    iextapi = IEXTradingApi()
    #iextapi.get_co_sector('AAPL')
    #iextapi.get_co_peers('AAPL')
    #iextapi.get_crypto()

    #df = iextapi.get_symbols_universe()
    #print (df.iloc[0:3,])
    #print (df.info())
    #print (df.head())
    #iextapi.get_these_types_from_symbols_universe(type_list=['et'])
    #unique_types = iextapi.get_unique_sec_types_symbols_universe()
    #print ("IEX unique types are ", unique_types)

    #df=iextapi.get_symbols_universe()
    #df=iextapi.get_these_types_from_symbols_universe(df, type_list=['et'])
    #df.set_index('symbol', inplace = True)
    #print(df.loc['SPY'])

    iextapi.co_earnings_today()

    #print (df.info())

    #iextapi.create_sector_industry_universe()
