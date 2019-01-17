from root.nested import get_logger
from dateutil import parser
import pandas as pd
from urllib.error import HTTPError
import re

BASE_URL = 'https://www.sharesoutstandinghistory.com/'
LOGGER = get_logger()

class SharesOutstandingHistory:

    token_spec = [
        ('NUMBER', r'\d+.\d{2}'),
        ('MAGNITUDE', r'[A-Z]{1}')
    ]

    so_multiplier_dict = {'M': 1000000.0,
                          'B': 1000000000.0,
                          'T': 1000000000000.0}

    @staticmethod
    def get_simfin_indicator_ids():

        simfin_ind_id_list_url = 'https://simfin.com/data/help/main?topic=api-indicators'
        try:
            html_obj = pd.read_html(simfin_ind_id_list_url)
        except HTTPError as httperror:
            error_code = httperror.getcode()
            if error_code == 404:
                LOGGER.error("invalid URL %s", simfin_ind_id_list_url)
                return error_code
            else:
                LOGGER.error("SharesOutstandingHistory.get_simfin_indicator_ids(): HTTPError code %s", error_code)
                return error_code
        df = html_obj[0].rename(columns = html_obj[0].iloc[0]).drop(html_obj[0].index[0]).reset_index()
        df.drop('index', axis = 1, inplace=True)
        df.set_index('ID', inplace=True)
        return df

    @staticmethod
    def get_shares_outstanding_history(symbol):

        complete_url = BASE_URL + str.lower(symbol) + '/'
        html_obj = SharesOutstandingHistory.get_html(symbol)
        if html_obj == 404:
            LOGGER.error("invalid URL %s, no shares outstanding history for ticker %s", complete_url, symbol)
            return
        df = html_obj[1].rename(columns = html_obj[1].iloc[0]).drop(html_obj[1].index[0])
        df.reset_index(inplace=True)
        df=df.drop(['index'], axis=1)
        convert_date_col = df.Date.apply(lambda x: parser.parse(x))
        df[symbol+'_SharesOutstanding'] = df[str.upper(symbol) + ' Shares Outstanding'].apply(
            lambda x: pd.Series(SharesOutstandingHistory.tokenize_shares_outstanding(x)))
        df.Date = convert_date_col
        df.rename(columns={'Date': 'date'}, inplace=True)
        df.set_index('date', inplace=True)
        return df

    @staticmethod
    def tokenize_shares_outstanding(shares_outs):

        return_dict = {}
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in SharesOutstandingHistory.token_spec)
        for mo in re.finditer(tok_regex, shares_outs):
            kind = mo.lastgroup
            value = mo.group()
            return_dict[kind] = value
        return float(return_dict['NUMBER'])* SharesOutstandingHistory.so_multiplier_dict[return_dict['MAGNITUDE']]

    @staticmethod
    def get_html(symbol):

        complete_url = BASE_URL + str.lower(symbol) + '/'
        try:
            html_obj = pd.read_html(BASE_URL + str.lower(symbol) + '/')
        except HTTPError as httperror:
            error_code = httperror.getcode()
            if error_code == 404:
                LOGGER.error("invalid URL %s, no shares outstanding history for ticker %s", complete_url, symbol)
                return error_code

        return html_obj

if __name__ == '__main__':

    df = SharesOutstandingHistory.get_shares_outstanding_history("AMZN")
    print (df)