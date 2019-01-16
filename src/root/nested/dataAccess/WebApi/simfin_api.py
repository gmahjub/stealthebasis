import requests
import pandas as pd

from root.nested.SysOs.os_mux import OSMuxImpl
from root.nested.dataAccess.secure_keys_access import SecureKeysAccess
from root.nested import get_logger

class SimFinApi:

    def __init__(self,
                 tickers,
                 writer_filename=None):

        self.tickers = tickers
        self.writer_filename = writer_filename
        self.logger = get_logger()
        self.source = 'simfin'
        self.api_key = SecureKeysAccess.get_vendor_api_key_static(vendor=str.upper(self.source))
        self.simfin_pwd = OSMuxImpl.get_proper_path('/workspace/data/simfin/')
        if writer_filename is not None:
            self.writer = self.get_writer(self.simfin_pwd + writer_filename)
        else:
            self.writer = None
        self.sim_ids = self.get_sim_ids(tickers)

    def set_writer_filename(self,
                            writer_filename):

        self.writer_filename = writer_filename

    def get_sim_ids(self,
                    tickers):

        # check if we already have the sim_id locally so that we don't have to pull from remote
        sim_ids = []
        for ticker in tickers:
            simid = SecureKeysAccess.get_ticker_simid_static(ticker)
            if simid is not "":
                self.logger.info("SimFinApi.get_sim_ids(): found simid %s for ticker %s locally!", simid, ticker)
                sim_ids.append(simid)
                continue
            request_url = f'https://simfin.com/api/v1/info/find-id/ticker/{ticker}?api-key={self.api_key}'
            self.logger.info("SimFinApi.get_sim_ids(): request_url is %s", request_url)
            content = requests.get(request_url)
            data = content.json()
            if 'error' in data or len(data) < 1:
                sim_ids.append(None)
            else:
                self.logger.info("SimFinApi.get_sim_ids(): sim_id for ticker %s is %s", ticker, data[0]['simId'])
                sim_ids.append(data[0]['simId'])
                SecureKeysAccess.insert_simid(ticker, data[0]['simId'])
                self.logger.info("SimFinApi.get_sim_ids(): inserted sim_id %s for ticker %s in local file!",
                                 data[0]['simId'],
                                 ticker)

        return (sim_ids)

    def get_writer(self,
                   filename):

        return pd.ExcelWriter(filename, engine = 'xlsxwriter')

    def writer_save(self,
                    writer):

        writer.save()

    def writer_close(self,
                     writer):

        writer.close()

    def get_quarterly_eps(self):

        # get quarter by quarter eps, instead of TTM
        # TTM = trailing 12 month <insert stat i.e. eps>
        data = { "search": [ {'indicatorId': "4-12",
                              'meta': [ { 'id': 6,
                                          'value': 'TTM',
                                          'operator': 'eq'},]}],
                 "simIdList": [
                     111052
                 ]
                 }
        request_url = f'https://simfin.com/api/v1/'
        request_url = f'https://simfin.com/api/v1/finder?api-key={self.api_key}'
        r = requests.post(request_url, json=data)
        print (r.content)

    def get_eps(self):

        data = { "search": [ { "indicatorId": "4-12",
                                "meta": [ { "id": 6,
                                            "value": "TTM",
                                            "operator": "eq"
                                            },
                                          ],
                            },
                {
                    "indicatorId": "4-12",
                    "meta": [
                        {
                            "id": 6,
                            "value": "TTM-1",
                            "operator": "eq"
                        },
                    ],
                },
                {
                    "indicatorId": "4-12",
                    "meta": [
                        {
                            "id": 6,
                            "value": "TTM-2",
                            "operator": "eq"
                        },
                    ],
                }
            ],
            "simIdList": [
                111052
            ]
        }
        request_url = f'https://simfin.com/api/v1/finder?api-key={self.api_key}'
        r = requests.post(request_url, json=data)
        print(r.content)

    def get_data(self,
                 sim_ids,
                 statement_type,
                 time_periods,
                 year_start,
                 year_end,
                 output_file):

        data = {}
        for idx, sim_id in enumerate(sim_ids):
            d = data[tickers[idx]] = {"Line Item": []}
            if sim_id is not None:
                for year in range(year_start, year_end + 1):
                    for time_period in time_periods:
                        period_identifier = time_period + '-' + str(year)
                        if period_identifier not in d:
                            d[period_identifier] = []
                        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/statements/standardised?stype=' \
                            f'{statement_type}&fyear={year}&ptype={time_period}&api-key={self.api_key}'
                        self.logger.info("SimFinApi.get_data(): request_url is %s", request_url)
                        content = requests.get(request_url)
                        statement_data = content.json()
                        if len(d['Line Item']) == 0 and 'values' in statement_data:
                            d['Line Item'] = [x['standardisedName'] for x in statement_data['values']]

                        if 'values' in statement_data:
                            for item in statement_data['values']:
                                d[period_identifier].append(item['valueChosen'])
                        else:
                            # no data found for time period
                            d[period_identifier] = [None for _ in d['Line Item']]
                # fix the periods where no values were available
                len_target = len(d['Line Item'])
                if len_target > 0:
                    for k, v in d.items():
                        if len(v) != len_target:
                            d[k] = [None for _ in d['Line Item']]

            # convert to pandas dataframe
            df = pd.DataFrame(data=d)
            self.logger.info("SimFinApi.get_data(): writing out data for ticker %s", tickers[idx])
            df.to_excel(self.writer, sheet_name=tickers[idx])
        self.writer_save(self.writer)
        self.writer_close(self.writer)


tickers = ['AAPL', 'NVDA', 'WMT']
#tickers = ['HD', 'JPM', 'BA']
sf = SimFinApi(tickers)
statement_type = "pl"
time_periods = ['Q1', 'Q2', 'Q3', 'Q4']
year_start = 2010
year_end = 2018
output_file = 'simfin_data.xlsx'

#sf.get_quarterly_eps()

#sim_ids = sf.get_sim_ids(tickers)
#sf.get_data(sim_ids=sim_ids,
#            statement_type=statement_type,
#            time_periods=time_periods,
#            year_start=year_start,
#            year_end=year_end,
#            output_file=output_file)

