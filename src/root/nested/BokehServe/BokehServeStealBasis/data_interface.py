from . import get_logger
import os
import requests
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from .os_mux import OSMuxImpl
from datetime import datetime, timedelta
import pyEX as pyex
import sys
import gspread
from oauth2client.service_account import ServiceAccountCredentials

sys.path.extend(
    ['/Users/traderghazy/PycharmProjects/stealthebasis', '/Users/traderghazy/PycharmProjects/stealthebasis/src'])

header = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
}
sp500_req_url = "https://etfdailynews.com/etf/spy/"
nq100_req_url = "https://etfdailynews.com/etf/qqq/"
dow30_req_url = "https://etfdailynews.com/etf/dia/"
russell2000_req_url = "https://etfdailynews.com/etf/iwm/"
r_sp = requests.get(sp500_req_url, headers=header)
r_nq = requests.get(nq100_req_url, headers=header)
r_dow = requests.get(dow30_req_url, headers=header)
r_r2k = requests.get(russell2000_req_url, headers=header)
SP500_HOLDINGS_URL = pd.read_html(r_sp.text, attrs={'id': 'etfs-that-own'})
NQ100_HOLDINGS_URL = pd.read_html(r_nq.text, attrs={'id': 'etfs-that-own'})
DOW30_HOLDINGS_URL = pd.read_html(r_dow.text, attrs={'id': 'etfs-that-own'})
R2K_HOLDINGS_URL = pd.read_html(r_r2k.text, attrs={'id': 'etfs-that-own'})
RUSSELL1000_HOLDINGS_URL = "https://www.ishares.com/us/products/239707/ishares-russell-1000" \
                           "-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
RUSSELL3000_HOLDINGS_URL = "https://www.ishares.com/us/products/239714/ishares-russell-3000" \
                           "-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
MAIN_INDICES_LIST = ['SP500', 'NQ100', 'DOW30', 'R2K']
LOGGER = get_logger()


class DataProviderInterface(object):

    def __init__(self):

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

        nq_100_stock_universe = 'NQ100'
        self.total_pwd_nq100 = self.local_stock_universe_file_pwd + \
                               nq_100_stock_universe + self.stock_universe_file_type
        try:
            last_time_nq_100_stock_universe_file_modified = os.path.getmtime(self.total_pwd_nq100)
        except FileNotFoundError:
            last_time_nq_100_stock_universe_file_modified = ""

        sp_500_stock_universe = 'SP500'
        self.total_pwd_sp500 = self.local_stock_universe_file_pwd + \
                               sp_500_stock_universe + self.stock_universe_file_type
        try:
            last_time_sp_500_stock_universe_file_modified = os.path.getmtime(self.total_pwd_nq100)
        except FileNotFoundError:
            last_time_sp_500_stock_universe_file_modified = ""

        dow_30_stock_universe = 'DOW30'
        self.total_pwd_dow30 = self.local_stock_universe_file_pwd + \
                               dow_30_stock_universe + self.stock_universe_file_type
        try:
            last_time_dow_30_stock_universe_file_modified = os.path.getmtime(self.total_pwd_nq100)
        except FileNotFoundError:
            last_time_dow_30_stock_universe_file_modified = ""

        russ_2k_stock_universe = 'R2K'
        self.total_pwd_r2k = self.local_stock_universe_file_pwd + \
                             russ_2k_stock_universe + self.stock_universe_file_type
        try:
            last_time_r2k_stock_universe_file_modified = os.path.getmtime(self.total_pwd_r2k)
        except FileNotFoundError:
            last_time_r2k_stock_universe_file_modified = ""

        self.last_modified_times = {stock_universe_file_name: last_time_stock_universe_file_modified,
                                    russell_3000_stock_universe: last_time_russell_3000_stock_universe_file_modified,
                                    russell_1000_stock_universe: last_time_russell_1000_stock_universe_file_modified,
                                    russ_2k_stock_universe: last_time_r2k_stock_universe_file_modified,
                                    nq_100_stock_universe: last_time_nq_100_stock_universe_file_modified,
                                    sp_500_stock_universe: last_time_sp_500_stock_universe_file_modified,
                                    dow_30_stock_universe: last_time_dow_30_stock_universe_file_modified}

        self.stock_universe_download_func = {stock_universe_file_name: self.download_stock_universe,
                                             russell_1000_stock_universe: self.download_russell_1000_stock_universe,
                                             russell_3000_stock_universe: self.download_russell_3000_stock_universe,
                                             russ_2k_stock_universe: self.download_r2k_holdings_symbol_list,
                                             nq_100_stock_universe: self.download_nq100_holdings_symbol_list,
                                             sp_500_stock_universe: self.download_sp500_holdings_symbol_list,
                                             dow_30_stock_universe: self.download_dow30_holdings_symbol_list}

    def download_stock_universe(self):
        """ The stock universe here is the holdings of IWB, the Russell 1000
            The Russell 1000 is the 1000 largest companies (by market cap) in
            the USA.
        """
        url = RUSSELL1000_HOLDINGS_URL
        response = requests.get(url)
        with open(self.total_pwd_stock_universe_file, 'wb') as f:
            f.write(response.content)
        LOGGER.info("DataObject.get_stock_universe: HTTP Response Status Code %s " + str(response.status_code))

    def download_russell_1000_stock_universe(self):

        url = RUSSELL1000_HOLDINGS_URL
        response = requests.get(url)
        with open(self.total_pwd_russell1000, 'wb') as f:
            f.write(response.content)
        LOGGER.info("DataObject.download_russell_1000_stock_universe: HTTP Response Status Code %s ",
                    str(response.status_code))
        error_response = response.raise_for_status()  # HTTP 4XX, 5XX
        if error_response is None:
            self.last_modified_times['Russ1K_holdings'] = os.path.getmtime(self.total_pwd_russell1000)
        else:
            LOGGER.error("DataObject.download_russell_1000_stock_universe: faild download %s",
                         str(error_response))

    def download_russell_3000_stock_universe(self):

        url = RUSSELL3000_HOLDINGS_URL
        response = requests.get(url)
        with open(self.total_pwd_russell3000, 'wb') as f:
            f.write(response.content)
        LOGGER.info("DataObject.download_russell_3000_stock_universe: HTTP Response Status Code %s ",
                    str(response.status_code))
        error_response = response.raise_for_status()  # Http 4XX, 5XX
        if error_response is None:
            self.last_modified_times['Russ3K_holdings'] = os.path.getmtime(self.total_pwd_russell3000)
        else:
            LOGGER.error("DataObject.download_russsell_3000_stock_universe: failed download %s",
                         str(error_response))

    def download_sp500_holdings_symbol_list(self):

        url = SP500_HOLDINGS_URL
        # below code from IEX example @ https://github.com/timkpaine/pyEX
        url[0].reset_index(inplace=True)
        sp500 = [x for x in url[0].Symbol.values.tolist() if isinstance(x, str)]
        url[0].set_index("Symbol", inplace=True)
        the_df = url[0].rename(columns={"Holding Name": "name"})
        the_df['BenchmarkETF'] = 'SPY'
        the_df['BenchmarkIndex'] = 'SPX'
        the_df['BenchmarkFuture'] = 'ES'
        # returns a list of tickers
        the_df.to_csv(self.total_pwd_sp500)
        return the_df, sp500

    def download_nq100_holdings_symbol_list(self):

        url = NQ100_HOLDINGS_URL
        url[0].reset_index(inplace=True)
        nq100 = [x for x in url[0].Symbol.values.tolist() if isinstance(x, str)]
        url[0].set_index("Symbol", inplace=True)
        # returns a list of tickers
        the_df = url[0].rename(columns={"Holding Name": "name"})
        the_df['BenchmarkIndex'] = 'QQQ'
        the_df['BenchmarkIndex'] = 'NDX'
        the_df['BenchmarkFuture'] = 'NQ'
        the_df.to_csv(path_or_buf=self.total_pwd_nq100)
        return the_df, nq100

    def download_dow30_holdings_symbol_list(self):

        url = DOW30_HOLDINGS_URL
        url[0].reset_index(inplace=True)
        dow30 = [x for x in url[0].Symbol.values.tolist() if isinstance(x, str)]
        url[0].set_index("Symbol", inplace=True)
        the_df = url[0].rename(columns={"Holding Name": "name"})
        the_df['BenchmarkIndex'] = 'DJI'
        the_df['BenchmarkETF'] = 'DIA'
        the_df['BenchmarkFuture'] = 'YM'
        the_df.to_csv(self.total_pwd_dow30)
        return the_df, dow30

    def download_r2k_holdings_symbol_list(self):
        url = R2K_HOLDINGS_URL
        url[0].reset_index(inplace=True)
        r2k = [x for x in url[0].Symbol.values.tolist() if isinstance(x, str)]
        url[0].set_index("Symbol", inplace=True)
        the_df = url[0].rename(columns={"Holding Name": "name"})
        the_df['BenchmarkIndex'] = 'RUT'
        the_df['BenchmarkETF'] = 'IWM'
        the_df['BenchmarkFuture'] = 'RTY'
        the_df.to_csv(self.total_pwd_r2k)
        return the_df, r2k

    def get_stock_universe_file_as_df(self,
                                      stock_universe_filename):

        """ This is the main call to get the stock universe file into a pandas DataFrame
            Returns the dataframe of all stocks in the stock univerrse"""

        if any(np.isin(MAIN_INDICES_LIST, stock_universe_filename)):
            func = self.stock_universe_download_func[stock_universe_filename]
            suf_df, _ = func()
            return suf_df
        if self.does_stock_universe_file_exist(stock_universe_filename) is False or \
                self.is_stock_universe_file_old(stock_universe_filename) is True:
            func = self.stock_universe_download_func[stock_universe_filename]
            LOGGER.info("DataProviderInterface.get_stock_universe_file_as_df(%s): "
                        "running function %s with stock universe filename %s", stock_universe_filename,
                        stock_universe_filename, func)
            _ = func()
        full_path = self.local_stock_universe_file_pwd + stock_universe_filename + self.stock_universe_file_type
        LOGGER.info("DataProviderInterface.get_stock_universe_file_as_df(%s): full path of stock "
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


class IEXTradingApi:

    def __init__(self,
                 sec_type_list=['cs', 'et']):

        self.iex_trading_root_url = "https://api.iextrading.com/1.0"
        self.get_symbols_universe_url = "/ref-data/symbols"
        self.sec_type_list = sec_type_list
        self.master_sector_indusry_df = pd.DataFrame(columns=['Sector', 'Industry'])
        self.master_sector_industry_file = OSMuxImpl.get_proper_path(
            '/workspace/data/IEX/') + 'master_sector_industry.csv'
        self.iex_html_path = OSMuxImpl.get_proper_path('/workspace/data/IEX/html/')
        self.co_earnings_path = OSMuxImpl.get_proper_path('/workspace/data/IEX/earnings/')

    def get_symbols_universe(self):

        sym_univ_url = self.iex_trading_root_url + self.get_symbols_universe_url
        r = requests.get(sym_univ_url)
        LOGGER.info("IEXTradingApi.get_symbols_universe(): request to %s returned a status code of %s",
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
        """
        Must update this function because IEX no longer puts a "Type" field in the data. All 'type' fields
        have been replaced to N/A. To find ETF's, we are going to need to parse the 'Name' field."""
        return df[df['name'].str.contains("ETF")]

    def get_unique_sec_types_symbols_universe(self):

        df = self.get_symbols_universe()
        LOGGER.info('IEXTradingApi:get_unique_sec_types_symbols_universe(): the unique security types'
                    'at IEX are %s', str(df.type.unique()))
        return df.type.unique()  # return type is list

    def create_sector_industry_universe(self):

        df = self.get_symbols_universe()
        df.reset_index(inplace=True)
        df_small = df.iloc[0:5, ]
        self.master_sector_indusry_df.to_csv(self.master_sector_industry_file)

    def get_co_sector_industry_tags(self,
                                    row):

        ticker = row.symbol
        LOGGER.info("IEXTradingApi.get_co_sector_industry_tags(): getting sector, industry, tags for ticker %s",
                         ticker)
        df = pyex.companyDF(symbol=ticker)
        co_sector_industry_tags = (df.loc[ticker, ['sector', 'industry', 'tags', 'companyName', 'website']])
        co_sector_industry_tags['symbol'] = ticker
        # co_sector_industry_tags is a pandas Series
        # access the symbol (ticker) by using ".name" of pd.Series object co_sector_industry_tags
        alt_industries = co_sector_industry_tags['tags']
        rows_to_insert = []
        for alt_industry in alt_industries:
            if alt_industry != co_sector_industry_tags['sector'] and \
                    alt_industry != co_sector_industry_tags['industry']:
                rows_to_insert.append([co_sector_industry_tags['sector'], alt_industry])
        rows_to_insert.append([co_sector_industry_tags['sector'], co_sector_industry_tags['industry']])
        for sector_industry in rows_to_insert:
            if self.master_sector_indusry_df.index.size == 0:
                next_index = 0
            else:
                next_index = self.master_sector_indusry_df.index[-1] + 1
            if sector_industry[0] != '' and sector_industry[1] != '':
                self.master_sector_indusry_df.loc[next_index] = sector_industry
        print(self.master_sector_indusry_df)

    def get_co_peers(self,
                     ticker):

        print(pyex.peersDF(symbol=ticker))

    def format_market_cap(self,
                          mc):

        if mc >= 1.0:
            return (str(mc) + 'B')
        in_millions = mc * 1000.0
        return_val = str(in_millions) + 'M'
        return (return_val)

    def format_announce_time(self,
                             announce_time):

        announce_time_dict = {'BTO': 'Before Open',
                              'AMC': 'After Close'}
        return (announce_time_dict[announce_time])

    def format_headline(self,
                        headline):
        import math
        cell_color = "white"
        if isinstance(headline, float) is True and math.isnan(headline):
            LOGGER.info("IEXTradingApi.format_headline(): headline is null, %s", headline)
            headline = " "
        # create a meta_dict for color based on the appearance of the words "beats" and "misses"
        beats_occurences = len([i for i in range(len(headline)) if headline.startswith('beats', i)])
        misses_occurrences = len([i for i in range(len(headline)) if headline.startswith('misses', i)])
        if misses_occurrences == 0 and beats_occurences != 0:
            cell_color = 'green'
        elif misses_occurrences != 0 and beats_occurences != 0:
            cell_color = 'orange'
        elif beats_occurences == 0 and misses_occurrences != 0:
            cell_color = 'red'
        return headline, cell_color

    def co_earnings_today(self,
                          today_ymd=None):

        from datetime import datetime
        try_local_pull = False
        if today_ymd is None or today_ymd == datetime.now().strftime("%Y%m%d"):
            today_ymd = datetime.now().strftime("%Y%m%d")
            df = pyex.earningsTodayDF()
            if df.empty is True and os.path.isfile(self.co_earnings_path + today_ymd + ".csv") is True:
                try_local_pull = True
            elif df.empty is True:
                LOGGER.info("IEXTradingApi:co_earnings_today().: no earnings for date %s!", today_ymd)
                return
        elif os.path.isfile(self.co_earnings_path + today_ymd + ".csv") is False:
            LOGGER.info("IEXTradingApi:co_earnings_today(): no historical file available for date %s", today_ymd)
            return
        else:
            LOGGER.warning("IEXTradingApi:co_earnings_today(): try_local_pull is True, nothing at IEX!!!")
            try_local_pull = True
        if try_local_pull is True:
            # check for an existing flat csv file, maybe we did a pull already
            df = pd.read_csv(self.co_earnings_path + today_ymd + ".csv", sep=',', header=0)
            df.set_index('symbol', inplace=True)
            df.reset_index(inplace=True)
        else:
            df.to_csv(self.co_earnings_path + today_ymd + ".csv", sep=',')
            df.reset_index(inplace=True)
        market_cap_billions = df['quote.marketCap'].astype('float64') / 1000000000.0
        market_cap_billions = market_cap_billions.round(3)
        df['quote.marketCap'] = market_cap_billions.apply(lambda x: self.format_market_cap(x))
        df.EPSReportDate = df.EPSReportDate.astype('str')
        df.announceTime = df.announceTime.apply(lambda x: self.format_announce_time(x))
        df.estimatedChangePercent = df.estimatedChangePercent.mul(100).round(2).astype('str') + '%'
        df.estimatedEPS = '$' + df.estimatedEPS.astype('str')
        df.fiscalEndDate = df.fiscalEndDate.astype('str')
        df.headline = df.headline.apply(lambda x: self.format_headline(x))
        df[['headline', 'headline_color']] = df.headline.apply(pd.Series)
        df["quote.week52High"] = '$' + df['quote.week52High'].astype('str')
        df["quote.week52Low"] = '$' + df['quote.week52Low'].astype('str')
        df['quote.ytdChange'] = df['quote.ytdChange'].mul(100).round(2).astype('str') + '%'
        df.rename(index=str, columns={"symbol": "Symbol",
                                      "quote.peRatio": "P/E Ratio",
                                      "fiscalPeriod": "Quarter",
                                      "numberOfEstimates": "# of Estimates",
                                      "consensusEPS": "Consensus EPS",
                                      "headline": "Headline",
                                      "fiscalEndDate": "Quarter End",
                                      "estimatedEPS": "EPS Estimate",
                                      "estimatedChangePercent": "Estimated 1-Yr Change, EPS",
                                      "announceTime": "Announce Time",
                                      "EPSReportDate": "Report Date",
                                      "quote.marketCap": "Market Cap",
                                      "quote.peRatio": "P/E Ratio",
                                      "quote.sector": "Sector",
                                      "quote.week52High": "52-Week High",
                                      "quote.week52Low": "52-Week Low",
                                      "quote.ytdChange": "YTD Change",
                                      "yearAgo": "Year Ago EPS"}, inplace=True)
        list_of_columns = ['Symbol',
                           'Report Date',
                           'Announce Time',
                           'Consensus EPS',
                           'Estimated 1-Yr Change, EPS',
                           'EPS Estimate',
                           'Quarter End',
                           'Quarter',
                           'Headline',
                           'headline_color',
                           '# of Estimates',
                           'Market Cap',
                           'P/E Ratio',
                           'Sector',
                           '52-Week High',
                           '52-Week Low',
                           'YTD Change',
                           'Year Ago EPS']
        df = df[list_of_columns]
        output_html_filename = self.iex_html_path + 'co_earnings_' + today_ymd + '.html'
        # data_table = ExtendBokeh.bokeh_co_earnings_today_datatable(dataframe=df)
        # ExtendBokeh.save_co_earnings_today_data_table(data_table,
        #                                              html_output_file=output_html_filename,
        #                                              html_output_file_title='CoEarningsToday')

class SecureKeysAccess:

    SCOPE = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive',
             'https://www.googleapis.com/auth/drive.file',
             'https://www.googleapis.com/auth/drive.appdata',
             'https://www.googleapis.com/auth/drive.apps.readonly']
    CACHED_INFO_DIR = "/workspace/data/cachedinfo/"
    MYSQL_CACHED_INFO_FILE = "mysql_server.csv"
    VENDOR_KEYS_FILE = "vendor_api_keys.csv"
    CACHED_INFO_SWITCHER = {
        "mysql_server_ip": OSMuxImpl.get_proper_path(CACHED_INFO_DIR) + MYSQL_CACHED_INFO_FILE,
        "mysql_server_user": OSMuxImpl.get_proper_path(CACHED_INFO_DIR) + MYSQL_CACHED_INFO_FILE,
        "mysql_server_secret": OSMuxImpl.get_proper_path(CACHED_INFO_DIR) + MYSQL_CACHED_INFO_FILE,
        "vendor_api_keys": OSMuxImpl.get_proper_path(CACHED_INFO_DIR) + VENDOR_KEYS_FILE
    }

    @staticmethod
    def check_cached_info(info_type, info_field):
        info_file = SecureKeysAccess.CACHED_INFO_SWITCHER[info_type]
        last_modified_time = datetime.fromtimestamp(os.path.getmtime(info_file))
        return_info = None
        if last_modified_time - datetime.now() < timedelta(days=30):
            df = pd.read_csv(filepath_or_buffer=info_file, index_col=0, header=None)
            df.index.name = 'Name'
            df.columns = [['Value']]
            return_info = df.loc[info_field]['Value'][0]
        return return_info

    @staticmethod
    def get_vendor_api_key_static(vendor):
        cached_vendor_api_key = SecureKeysAccess.check_cached_info(info_type="vendor_api_keys", info_field=vendor)
        if cached_vendor_api_key is None:
            google_api_filesdir = '/workspace/data/googleapi/'
            local_data_file_pwd = OSMuxImpl.get_proper_path(google_api_filesdir)
            google_api_creds_file = local_data_file_pwd + "google_driveAccess_client_secrets.json"
            google_api_creds = ServiceAccountCredentials.from_json_keyfile_name(google_api_creds_file,
                                                                                SecureKeysAccess.SCOPE)
            client = gspread.authorize(google_api_creds)
            sheet_filename = "ghazy_mahjub_api_keys.csv"
            api_key_sheet = client.open(sheet_filename).sheet1
            list_of_hashes = api_key_sheet.get_all_records()
            for hasher in list_of_hashes:
                if hasher['Vendor'] == vendor:
                    return hasher['API_KEY']
            return ""
        return cached_vendor_api_key

    @staticmethod
    def get_vendor_api_key(sheet,
                           vendor_name):

        list_of_hashes = sheet.get_all_records()
        for hasher in list_of_hashes:
            if hasher['Vendor'] == vendor_name:
                return hasher['API_KEY']
        LOGGER.error("SecureKeysAccess.get_vendor_api_key(): input vendor_name %s not found in api_keys!",
                          vendor_name)

class TiingoDataObject:
    """description of class"""

    def __init__(self, **kwargs):

        self.file_mod_time_dict = {}
        self.symbols = None
        self.start_date = ""
        self.end_date = ""
        self.source = "tiingo"
        self.api_key = SecureKeysAccess.get_vendor_api_key_static(vendor=str.upper(self.source))

        for key, value in kwargs.items():

            if key == 'symbols':
                self.symbols = value
                LOGGER.info("TiingoDataObject.__init__.symbols: %s", str(self.symbols))
            elif key == 'start_date':
                self.start_date = value
                LOGGER.info("TiingoDataObject.__init__.start_date: %s", str(self.start_date))
            elif key == 'end_date':
                self.end_date = value
                LOGGER.info("TiingoDataObject.__init__.end_date: %s", str(self.end_date))
            elif key == 'source':
                self.source = value
                LOGGER.info("TiingoDataObject.__init__.source: %s", str(self.source))

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

        LOGGER.info("TiingoDataObject.all_px_from_csv(): total_file_path %s ", str(total_file_path))
        all_px_df = pd.read_csv(total_file_path, parse_dates=True, index_col='date')
        return_df = all_px_df.loc[start_date: end_date]
        LOGGER.info("TiingoDataObject.all_px_from_csv(): Dataframe Columns %s ", str(all_px_df.columns))
        return return_df

    def does_local_file_exist(self,
                              px_type,
                              total_file_path):

        LOGGER.info("TiingoDataObject.does_local_file_exist(): Checking for existence of %s ", total_file_path)
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
                print ("fucking sym", sym)
                print ("fucking local file type", self.local_file_type)
                local_all_file_name = sym + self.local_file_type
                total_file_path = self.local_data_file_pwd + local_all_file_name
                LOGGER.info("TiingoDataObject.get_px_data_df().total_file_path: %s", str(total_file_path))

                if self.does_local_file_exist(px_type, total_file_path):
                    if self.is_local_file_old(total_file_path):
                        LOGGER.info("Pulling new dataframes from Tiingo...")
                        return_dict[sym] = func_0(sym, start_date, end_date)
                    else:
                        LOGGER.info("Pulling dataframes from local csv files...")
                        return_dict[sym] = func_1(total_file_path, start_date, end_date)
                else:
                    LOGGER.info("Pulling new dataframes from Tiingo...")
                    return_dict[sym] = func_0(sym, start_date, end_date)

        return return_dict
