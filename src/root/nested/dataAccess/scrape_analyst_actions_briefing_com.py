import urllib.request
from root.nested import get_logger
from bs4 import BeautifulSoup
import pandas as pd
from pandas.tseries.offsets import BDay

import datetime
from datetime import datetime
import os
import platform
import csv

class scrape_analyst_actions_briefing_com(object):

    """description of class"""

    UPGRADES_DOWNGRADES = "Upgrades-Downgrades"
    UPGRADES = "Upgrades"
    DOWNGRADES = "Downgrades"
    BRIEFING_DOT_COM = "https://www.briefing.com/Investor/Calendars/"
    ANALYST_ACTION_UPGRADE_TYPE = "Upgrades"
    ANALYST_ACTION_DOWNGRADE_TYPE = "Downgrades"
    ANALYST_ACTION_INITIATE_TYPE = "Initiated"
    ANALYST_ACTION_RESUME_TYPE = "Resumed"
    ANALYST_ACTION_REITERATE_TYPE = "Reiterated"

    def __init__(self,
                 analyst_action_day = datetime.now().day,
                 analyst_action_month = datetime.now().month,
                 analyst_action_year = datetime.now().year,
                 briefing_uprades_url=None,
                 briefing_downgrades_url=None,
                 briefing_coverage_init_url=None,
                 briefing_coverage_resumed_url=None,
                 briefing_price_tgt_changed_url=None):

        super().__init__()
        self.logger = get_logger()
        self.briefing_upgrades_url = briefing_uprades_url
        self.briefing_downgrades_url = briefing_downgrades_url
        self.briefing_coverage_init_url = briefing_coverage_init_url
        self.briefing_coverage_resumed_url = briefing_coverage_resumed_url
        self.briefing_price_tgt_changed_url = briefing_price_tgt_changed_url

        if (platform.system() == "Windows"):
            self.local_analyst_action_dir = "C:\\Users\\ghazy\\workspace\\data\\Briefing.com\\AnalystActions\\"
        elif (platform.system() == "Darwin"):
            self.local_analyst_action_dir = "/Users/traderghazy/workspace/data/Briefing.com/AnalystActions/"
        self.current_month = analyst_action_month
        self.current_year = analyst_action_year
        self.current_day = analyst_action_day
        
    def set_briefing_upgrades_url(self,
                                   briefing_upgrades_url):

        self.briefing_upgrades_url = briefing_upgrades_url
        
    def set_briefing_downgrades_url(self,
                                    briefing_downgrades_url):

        self.briefing_downgrades_url = briefing_downgrades_url

    def set_briefing_coverage_init_url(self,
                                       briefing_coverage_init_url):

        self.briefing_coverage_init_url = briefing_coverage_init_url

    def set_briefing_coverage_resumed_url(self,
                                          briefing_coverage_resumed_url):

        self.briefing_coverage_resumed_url = briefing_coverage_resumed_url

    def set_briefing_price_tgt_changed_url(self,
                                           briefing_price_tgt_changed_url):

        self.briefing_price_tgt_changed_url = briefing_price_tgt_changed_url

    def create_url(self,
                   analyst_action_type):
        
        ## https://www.briefing.com/Investor/Calendars/upgrades-downgrades/Upgrades/2018/10/8
        return_url = scrape_analyst_actions_briefing_com.BRIEFING_DOT_COM
        return_url += scrape_analyst_actions_briefing_com.UPGRADES_DOWNGRADES + '/' + analyst_action_type + \
                     '/' + str(self.current_year) + '/' + str(self.current_month) + '/' + str(self.current_day)
        self.logger.info("scrape_analyst_actions_briefing_com.create_url:return_url: %s", str(return_url))
        return (return_url)

    def open_url(self,
                 analyst_action_type,
                 url_to_open = None):

        if url_to_open == None:
            self.logger.error("scrape_analyst_actions_briefing_com.open_url:invalid parameter url_to_open: %s", str(url_to_open))
        else:
            write_out_fileName = str(self.current_year) + str(self.current_month).zfill(2) + \
                    str(self.current_day).zfill(2) + analyst_action_type + ".csv"
            write_out_fileName = self.local_analyst_action_dir + write_out_fileName
            self.logger.info("scrape_analyst_actions_briefing_com.open_url:opening url: %s", str(url_to_open))
            response_html=urllib.request.urlopen(url_to_open)
            soup = BeautifulSoup(response_html, 'html.parser')
            data_table = soup.find('table', attrs={'class':'calendar-table'})
            if data_table is None:
                self.logger.info("scrape_analyst_actions_briefing_com.open_url:data_table is empty: %s", data_table)
                return
            table_header_row = data_table.find_all_next('tr', attrs={'class': 'row-header'})
            #table_header_row = data_table.find_all('tr')
            # the below should only have one row in it, the header row, one iteration.
            tr=table_header_row[0]
            td = tr.find_all('td')
            header_row = [i.text for i in td]
            try:
                with open(write_out_fileName, 'w') as csvFile:
                    writer = csv.DictWriter(csvFile, fieldnames=header_row)
                    writer.writeheader()    
                    # the below are the actual rows with the ratings data
                    table_data_rows = data_table.find_all_next('tr', attrs={'class': ['wh-row','hl-row']} )
                    for tr in table_data_rows:
                        td = tr.find_all('td')
                        row = [i.text for i in td]
                        create_dict = {}
                        i = 0
                        while i < len(row):
                            if (str(row[i]).find('»') != -1):
                                row[i] = str(row[i]).replace('»','->')
                            create_dict[header_row[i]] = row[i]
                            i+=1
                        writer.writerow(create_dict)
            except IOError as ioe:
                print("I/O error", ioe)

def get_archived_analyst_actions(from_archived_datetime,
                                 to_archived_datetime = pd.datetime.today()):

        this_day = to_archived_datetime
        while (this_day > from_archived_datetime):
            print (this_day)
            saabc = scrape_analyst_actions_briefing_com(analyst_action_day = this_day.day,
                                                        analyst_action_month = this_day.month,
                                                        analyst_action_year = this_day.year)
            saabc_url = saabc.create_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_DOWNGRADE_TYPE)
            saabc.open_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_DOWNGRADE_TYPE, saabc_url)

            saabc_url = saabc.create_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_RESUME_TYPE)
            saabc.open_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_RESUME_TYPE, saabc_url)

            saabc_url = saabc.create_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_UPGRADE_TYPE)
            saabc.open_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_UPGRADE_TYPE, saabc_url)

            saabc_url = saabc.create_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_INITIATE_TYPE)
            saabc.open_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_INITIATE_TYPE, saabc_url)

            saabc_url = saabc.create_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_REITERATE_TYPE)
            saabc.open_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_REITERATE_TYPE, saabc_url)

            this_day -= BDay(1)

if __name__ == '__main__':

    from_archived_datetime = datetime(2018, 9, 7)
    get_archived_analyst_actions(from_archived_datetime)
            
    #saabc = scrape_analyst_actions_briefing_com()
    #saabc_url = saabc.create_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_DOWNGRADE_TYPE)
    #saabc.open_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_DOWNGRADE_TYPE, saabc_url)
    #saabc_url = saabc.create_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_RESUME_TYPE)
    #saabc.open_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_RESUME_TYPE, saabc_url)
    #saabc_url = saabc.create_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_UPGRADE_TYPE)
    #saabc.open_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_UPGRADE_TYPE, saabc_url)
    #saabc_url = saabc.create_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_INITIATE_TYPE)
    #saabc.open_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_INITIATE_TYPE, saabc_url)
    #saabc_url = saabc.create_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_REITERATE_TYPE)
    #saabc.open_url(scrape_analyst_actions_briefing_com.ANALYST_ACTION_REITERATE_TYPE, saabc_url)

