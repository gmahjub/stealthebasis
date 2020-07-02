import requests
import pandas as pd
from selenium import webdriver
from urllib.error import HTTPError

from root.nested.SysOs.os_mux import OSMuxImpl
from root.nested.dataAccess.secure_keys_access import SecureKeysAccess
from root.nested import get_logger

""" Selenium Resources: 
https://selenium-python.readthedocs.io/getting-started.html
https://sites.google.com/a/chromium.org/chromedriver/getting-started
http://stanford.edu/~mgorkove/cgi-bin/rpython_tutorials/Scraping_a_Webpage_Rendered_by_Javascript_Using_Python.php
"""


class SimFinApi:
    INDICATOR_ID_URL = 'https://simfin.com/data/help/main?topic=api-indicators'
    SECTOR_INDUSTRY_URL = 'https://simfin.com/data/help/main?topic=industry-classification'

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

    def get_simfin_indicator_ids(self):

        simfin_ind_id_list_url = SimFinApi.INDICATOR_ID_URL
        simfin_ind_id_html_content = self.simfin_selenium_connect(request_url=simfin_ind_id_list_url)
        try:
            html_obj = pd.read_html(simfin_ind_id_html_content)
        except HTTPError as httperror:
            error_code = httperror.getcode()
            if error_code == 404:
                self.logger.error("SimFinApi.get_simfin_indicator_ids(): invalid URL %s", simfin_ind_id_list_url)
                return error_code
            else:
                self.logger.error("SimFinApi.get_simfin_indicator_ids(): HTTPError code %s", error_code)
                return error_code
        df = html_obj[0].rename(columns=html_obj[0].iloc[0]).drop(html_obj[0].index[0]).reset_index()
        df.drop('index', axis=1, inplace=True)
        df.set_index('ID', inplace=True)
        return df

    def simfin_selenium_connect(self,
                                request_url=None):

        if request_url is None:
            request_url = SimFinApi.INDICATOR_ID_URL
        browser = webdriver.Chrome('/Users/traderghazy/chromedriver/chromedriver')
        browser.get(request_url)
        html_content = browser.execute_script("return document.body.innerHTML")
        browser.quit()
        return html_content

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

        return pd.ExcelWriter(filename, engine='xlsxwriter')

    def writer_save(self,
                    writer):

        writer.save()

    def writer_close(self,
                     writer):

        writer.close()

    def get_all_ratios(self,
                       sim_id):

        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/ratios?api-key={self.api_key}'
        content = requests.get(request_url)
        all_ratios = content.json()
        return (all_ratios)  # list of dictionaries (i.e json)

    def get_single_ratio(self,
                         sim_id,
                         ind_id):

        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/ratios?indicators={ind_id}&api-key={self.api_key}'
        content = requests.get(request_url)
        ratio = content.json()
        return (ratio)

    def get_co_fiscal_year_end(self,
                               sim_id):

        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}?api-key={self.api_key}'
        content = requests.get(request_url)

        try:
            data_dict = content.json()
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_fiscal_year_end(): error message is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_fiscal_year_end(): unknown error!")
            return
        # returns the month of the fiscal year end. eg. 9 -> September
        return data_dict['fyearEnd']

    def get_co_sector_code(self,
                           sim_id):

        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}?api-key={self.api_key}'
        content = requests.get(request_url)

        try:
            data_dict = content.json()
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error('SimFinApi.get_co_sector_code(): error message is %s', error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_sector_code(): unknown error!")
            return
        except Exception as excp:
            self.logger.error("SimFinApi.get_co_sector_code(): %s", str(excp))
            self.logger.error("SimFinApi.get_co_sector_code(): content from request_url get(): %s", content.text)
            self.logger.error("SimFinApi.get_co_sector_code(): %s %s", type(excp).__name__, str(excp.args))
            return

        return data_dict['sectorCode']

    def get_co_num_emps(self,
                        sim_id):

        ind_id = '0-3'
        indicatorName, co_num_emps, as_of_quarter, fiscal_year, quarter_end_date = ("", "", "", "", "")

        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_num_emps(): request url is %s", request_url)
        content = requests.get(request_url)
        try:
            num_emps_dict = content.json()[0]
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_num_emps(): error message is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_num_emps(): unknown error!")
            except Exception as excp:
                self.logger.error('SimFinApi.get_co_num_emps(): exception thrown is %s', str(excp))
            return
        if num_emps_dict['indicatorName'] is not None:
            indicatorName = num_emps_dict['indicatorName']
        if num_emps_dict['value'] is not None:
            co_num_emps = num_emps_dict['value']
        if num_emps_dict['period'] is not None:
            as_of_quarter = num_emps_dict['period']
        if num_emps_dict['fyear'] is not None:
            fiscal_year = num_emps_dict['fyear']
        if num_emps_dict['period-end-date'] is not None:
            quarter_end_date = num_emps_dict['period-end_date']

        return indicatorName, co_num_emps, as_of_quarter, fiscal_year, quarter_end_date

    def get_co_founding_year(self,
                             sim_id):

        ind_id = '0-5'
        indicatorName, founding_year = ("", "")

        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_founding_year(): request url is %s", request_url)
        content = requests.get(request_url)
        try:
            co_founding_year_dict = content.json()[0]
        except IndexError:
            print("content.json() is ", content.json())
            try:
                error_msg = content.json()['error']
            except KeyError:
                self.logger.error("SimFinApi.get_co_founding_year(): unkown error occurred!")
                return
            except Exception as excp:
                self.logger.error('SimFinApi.get_co_founding_year(): exception thrown is %s', str(excp))
                return
            self.logger.error("SimFinApi.get_co_founding_year(): error message is %s", error_msg)
            return
        except Exception as excp:
            self.logger.error("SimFinApi.get_co_founding_year(): %s", str(excp))
            return

        if co_founding_year_dict['indicatorName'] is not None:
            indicatorName = co_founding_year_dict['indicatorName']
        if co_founding_year_dict['value'] is not None:
            founding_year = co_founding_year_dict['value']

        return indicatorName, founding_year

    def get_co_hq_loc(self,
                      sim_id):

        ind_id = '0-6'
        indicatorName, hq_loc = ("", "")

        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_hq_loc(): request url is %s", request_url)
        content = requests.get(request_url)
        try:
            co_hq_loc_dict = content.json()[0]
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_hq_loc(): error message is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_hq_loc(): unknown error occurred!")
            except Exception as excp:
                self.logger.error("SimFinApi.get_co_hq_loc(): %s", str(excp))
            return

        if co_hq_loc_dict['indicatorName'] is not None:
            indicatorName = co_hq_loc_dict['indicatorName']
        if co_hq_loc_dict['value'] is not None:
            hq_loc = co_hq_loc_dict['value']

        return indicatorName, hq_loc

    def get_co_sector_class(self,
                            sim_id):

        ind_id = '0-73'
        indicatorName, sector_class = ("", "")

        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_sector_class(): request url is %s", request_url)
        content = requests.get(request_url)
        try:
            data_dict = content.json()[0]
            if data_dict['indicatorName'] is not None:
                indicatorName = data_dict['indicatorName']
            if data_dict['value'] is not None:
                sector_class = data_dict['value']
            return indicatorName, sector_class
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_sector_class(): error message is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_sector_class(): unknown error occurred!")
            except Exception as excp:
                self.logger.error("SimFinApi.get_co_sector_class(): %s", str(excp))
            return

    def get_co_mkt_cap(self,
                       sim_id):

        ind_id = '4-11'
        indicatorName, co_mkt_cap = ("", "")

        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/' \
                      f'ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_mkt_cap(): request url is %s", request_url)
        content = requests.get(request_url)
        try:
            data_dict = content.json()[0]
            if data_dict['indicatorName'] is not None:
                indicatorName = data_dict['indicatorName']
            if data_dict['value'] is not None:
                co_mkt_cap = data_dict['value']
            return indicatorName, co_mkt_cap
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_mkt_cap(): error message is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_mkt_cap(): unknown error occurredd!")
            except Exception as excp:
                self.logger.error("SimFinApi.get_co_mkt_cap(): %s", str(excp))
            return

    def get_co_common_shares_outstanding(self,
                                         sim_id):

        ind_id = '0-64'
        indicatorName, co_common_shares_outstanding, value_as_of_date = ("", "", "")

        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/' \
                      f'ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_common_shares_outstanding(): request url is %s", request_url)
        content = requests.get(request_url)
        try:
            data_dict = content.json()[0]
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_common_shares_outstanding(): error message is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_common_shares_outstanding(): unknown error occurred!")
            except Exception as excp:
                self.logger.error("SimFinApi.get_co_common_shares_outstanding(): %s", str(excp))
            return
        if data_dict['indicatorName'] is not None:
            indicatorName = data_dict['indicatorName']
        if data_dict['value'] is not None:
            co_common_shares_outstanding = data_dict['value']
        if data_dict['period-end-date'] is not None:
            value_as_of_date = data_dict['period-end-date']
        return indicatorName, co_common_shares_outstanding, value_as_of_date

    def get_co_preferred_shares_outstanding(self,
                                            sim_id):

        ind_id = '0-65'
        indicatorName, co_preferred_shares_outstanding, value_as_of_date = ("", "", "")

        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/' \
                      f'ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_preferred_shares_outstanding(): request url is %s", request_url)
        content = requests.get(request_url)
        try:
            data_dict = content.json()[0]
            if data_dict['indicatorName'] is not None:
                indicatorName = data_dict['indicatorName']
            if data_dict['value'] is not None:
                co_preferred_shares_outstanding = data_dict['value']
            if data_dict['period-end-date'] is not None:
                value_as_of_date = data_dict['period-end_date']
            return indicatorName, co_preferred_shares_outstanding, value_as_of_date
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_preferred_shares_outstanding(): error msg is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_preferred_shares_outstanding(): unknown error!")
            except Exception as excp:
                self.logger.error("SimFinApi.get_co_preferred_shares_outstanding(): %s", str(excp))
            return

    def get_co_avg_shares_outstanding_basic(self,
                                            sim_id):

        ind_id = '0-66'
        indicatorName, co_avg_shares_outstanding_basic, period, fiscal_year, value_as_of_date = \
            ("", "", "", "", "")
        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/' \
                      f'ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_avg_shares_outstanding_basic(): request url is %s", request_url)
        content = requests.get(request_url)
        data_dict = content.json()[0]

        try:
            data_dict = content.json()[0]
            if data_dict['indicatorName'] is not None:
                indicatorName = data_dict['indicatorName']
            if data_dict['value'] is not None:
                co_avg_shares_outstanding_basic = data_dict['value']
            if data_dict['period'] is not None:
                period = data_dict['period']
            if data_dict['fyear'] is not None:
                fiscal_year = data_dict['fyear']
            if data_dict['period-end-date'] is not None:
                value_as_of_date = data_dict['period-end_date']
            return indicatorName, co_avg_shares_outstanding_basic, period, fiscal_year, value_as_of_date
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_avg_shares_outstanding_basic(): error msg is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_avg_shares_outstanding_basic(): unknown error!")
            except Exception as excp:
                self.logger.error("SimFinApi.get_co_avg_shares_outstanding_basic(): %s", str(excp))
            return

    def get_co_avg_shares_outstanding_diluted(self,
                                              sim_id):

        ind_id = '0-67'
        indicatorName, co_avg_shares_outstanding_diluted, period, fiscal_year, value_as_of_date = \
            ("", "", "", "", "")
        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/' \
                      f'ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_avg_shares_outstanding_diluted(): request url is %s", request_url)
        content = requests.get(request_url)
        try:
            data_dict = content.json()[0]
            if data_dict['indicatorName'] is not None:
                indicatorName = data_dict['indicatorName']
            if data_dict['value'] is not None:
                co_avg_shares_outstanding_diluted = data_dict['value']
            if data_dict['period'] is not None:
                period = data_dict['period']
            if data_dict['fyear'] is not None:
                fiscal_year = data_dict['fyear']
            if data_dict['period-end-date'] is not None:
                value_as_of_date = data_dict['period-end-date']
            return indicatorName, co_avg_shares_outstanding_diluted, period, fiscal_year, value_as_of_date
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_avg_shares_outstanding_diluted(): error msg is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_avg_shares_outstanding_diluted(): unknown error!")
            except Exception as excp:
                self.logger.error("SimFinApi.get_co_avg_shares_outstanding_diluted(): %s", str(excp))
            return

    def get_co_revenues(self,
                        sim_id):

        ind_id = '1-1'
        indicatorName, revenues, period, fiscal_year, currency, value_as_of_date = \
            ("", "", "", "", "", "")
        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/' \
                      f'ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_avg_revenues(): request url is %s", request_url)
        content = requests.get(request_url)

        try:
            data_dict = content.json()[0]
            if data_dict['indicatorName'] is not None:
                indicatorName = data_dict['indicatorName']
            if data_dict['value'] is not None:
                revenues = data_dict['value']
            if data_dict['period'] is not None:
                period = data_dict['period']
            if data_dict['fyear'] is not None:
                fiscal_year = data_dict['fyear']
            if data_dict['currency'] is not None:
                currency = data_dict['currency']
            if data_dict['period-end-date'] is not None:
                value_as_of_date = data_dict['period-end-date']
            return indicatorName, revenues, period, fiscal_year, currency, value_as_of_date
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_revenues(): error msg is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_revenues(): unknown error!")
            except Exception as excp:
                self.logger.error("SimFinApi.get_co_revenues(): %s", str(excp))
            return

    def get_co_cost_of_goods_sold(self,
                                  sim_id):

        ind_id = '1-2'
        indicatorName, cogs, period, fiscal_year, currency, value_as_of_date = \
            ("", "", "", "", "", "")
        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/' \
                      f'ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_cost_of_goods_sold(): request url is %s", request_url)
        content = requests.get(request_url)
        try:
            data_dict = content.json()[0]
            if data_dict['indicatorName'] is not None:
                indicatorName = data_dict['indicatorName']
            if data_dict['value'] is not None:
                cogs = data_dict['value']
            if data_dict['period'] is not None:
                period = data_dict['period']
            if data_dict['fyear'] is not None:
                fiscal_year = data_dict['fyear']
            if data_dict['currency'] is not None:
                currency = data_dict['currency']
            if data_dict['period-end-date'] is not None:
                value_as_of_date = data_dict['period-end-date']
            return indicatorName, cogs, period, fiscal_year, currency, value_as_of_date
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_cost_of_goods_sold(): error msg is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_cost_of_goods_sold(): unknown error!")
            except Exception as excp:
                self.logger.error("SimFinApi.get_co_cost_of_goods_sold(): %s", str(excp))
            return

    def get_co_gross_profit(self,
                            sim_id):

        ind_id = '1-4'
        indicatorName, gross_profit, period, fiscal_year, currency, value_as_of_date = \
            ("", "", "", "", "", "")
        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/' \
                      f'ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_gross_profit(): request url is %s", request_url)
        content = requests.get(request_url)

        try:
            data_dict = content.json()[0]
            if data_dict['indicatorName'] is not None:
                indicatorName = data_dict['indicatorName']
            if data_dict['value'] is not None:
                gross_profit = data_dict['value']
            if data_dict['period'] is not None:
                period = data_dict['period']
            if data_dict['fyear'] is not None:
                fiscal_year = data_dict['fyear']
            if data_dict['currency'] is not None:
                currency = data_dict['currency']
            if data_dict['period-end-date'] is not None:
                value_as_of_date = data_dict['period-end-date']
            return indicatorName, gross_profit, period, fiscal_year, currency, value_as_of_date
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_gross_profit(): error msg is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_gross_profit(): unknown error!")
            except Exception as excp:
                self.logger.error("SimFinApi.get_co_gross_profit(): %s", str(excp))
            return

    def get_co_operating_expenses(self,
                                  sim_id):

        ind_id = '1-11'
        indicatorName, operating_expenses, period, fiscal_year, currency, value_as_of_date = \
            ("", "", "", "", "", "")
        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/' \
                      f'ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_operating_expenses(): request url is %s", request_url)
        content = requests.get(request_url)

        try:
            data_dict = content.json()[0]
            if data_dict['indicatorName'] is not None:
                indicatorName = data_dict['indicatorName']
            if data_dict['value'] is not None:
                operating_expenses = data_dict['value']
            if data_dict['period'] is not None:
                period = data_dict['period']
            if data_dict['fyear'] is not None:
                fiscal_year = data_dict['fyear']
            if data_dict['currency'] is not None:
                currency = data_dict['currency']
            if data_dict['period-end-date'] is not None:
                value_as_of_date = data_dict['period-end-date']
            return indicatorName, operating_expenses, period, fiscal_year, currency, value_as_of_date
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_operating_expenses(): error msg is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_operating_expenses(): unknown error!")
            except Exception as excp:
                self.logger.error("SimFinApi.get_co_operating_expenses(): %s", str(excp))
            return

    def get_co_indicator(self,
                         sim_id,
                         ind_id=None):

        if ind_id is None:
            # default indicator is P/E
            ind_id = '4-14'
        indicatorId, indicatorName, value, period, fyear, currency, period_end_date = \
            ("", "", "", "", "", "", "")
        request_url = f'https://simfin.com/api/v1/companies/id/{sim_id}/' \
                      f'ratios?indicators={ind_id}&api-key={self.api_key}'
        self.logger.info("SimFinApi.get_co_indicator(): request url is %s", request_url)
        content = requests.get(request_url)

        try:
            data_dict = content.json()[0]
            if data_dict['indicatorId'] is not None:
                indicatorId = data_dict['indicatorId']
            if data_dict['indicatorName'] is not None:
                indicatorName = data_dict['indicatorName']
            if data_dict['value'] is not None:
                value = data_dict['value']
            if data_dict['period'] is not None:
                period = data_dict['period']
            if data_dict['fyear'] is not None:
                fyear = data_dict['fyear']
            if data_dict['currency'] is not None:
                currency = data_dict['currency']
            if data_dict['period-end-date'] is not None:
                period_end_date = data_dict['period-end-date']
            return indicatorName, value, period, fyear, currency, period_end_date
        except IndexError:
            ret_val = content.json()
            try:
                error_msg = ret_val['error']
                self.logger.error("SimFinApi.get_co_indicator(): error msg is %s", error_msg)
            except KeyError:
                self.logger.error("SimFinApi.get_co_indicator(): unknown error!")
            except Exception as excp:
                self.logger.error("SimFinApi.get_co_indicator(): %s", str(excp))
            return

    def get_quarterly_eps(self):

        # get quarter by quarter eps, instead of TTM
        # TTM = trailing 12 month <insert stat i.e. eps>
        data = {"search": [{'indicatorId': "4-12",
                            'meta': [{'id': 6,
                                      'value': 'TTM',
                                      'operator': 'eq'}, ]}],
                "simIdList": [
                    111052
                ]
                }
        request_url = f'https://simfin.com/api/v1/'
        request_url = f'https://simfin.com/api/v1/finder?api-key={self.api_key}'
        r = requests.post(request_url, json=data)
        print(r.content)

    def get_eps(self):

        data = {"search": [{"indicatorId": "4-12",
                            "meta": [{"id": 6,
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
# tickers = ['HD', 'JPM', 'BA']
sf = SimFinApi(tickers)
statement_type = "pl"
time_periods = ['Q1', 'Q2', 'Q3', 'Q4']
year_start = 2010
year_end = 2018
output_file = 'simfin_data.xlsx'

# sf.get_quarterly_eps()

# sim_ids = sf.get_sim_ids(tickers)
# sf.get_data(sim_ids=sim_ids,
#            statement_type=statement_type,
#            time_periods=time_periods,
#            year_start=year_start,
#            year_end=year_end,
#            output_file=output_file)

# df = sf.get_simfin_indicator_ids()
# sf.get_all_ratios(111052)

# request_url = f'https://simfin.com/api/v1/companies/id/111052/statements/standardised?stype=' \
#    f'{statement_type}&fyear=2010&ptype=Q1&api-key={sf.api_key}'
# content = requests.get(request_url)
# print (content.json())
# print (content.content)
# request_url = f'https://simfin.com/api/v1/companies/id/111052/ratios?api-key={sf.api_key}'
# content = requests.get(request_url)
# print (content.json())
# print (content.content)
# co_founding_year_tuple = sf.get_co_founding_year(111052)
# print (co_founding_year_tuple)
# num_emps = sf.get_co_num_emps(111052)
# print (num_emps)
# co_mkt_cap = sf.get_co_mkt_cap(111052)
# sf.get_co_preferred_shares_outstanding(239962)
print(sf.get_co_avg_shares_outstanding_diluted(111052))
print(sf.get_co_revenues(111052))
print(sf.get_co_cost_of_goods_sold(111052))
print(sf.get_co_gross_profit(111052))
print(sf.get_co_operating_expenses(111052))
print(sf.get_co_indicator(111052))
print(sf.get_co_fiscal_year_end(111052))
print(sf.get_co_sector_code(111052))
