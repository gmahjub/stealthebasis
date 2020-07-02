from root.nested import get_logger
from datetime import datetime, timedelta, date
from pytz import timezone
import shutil
from selenium import webdriver
import pandas as pd
from root.nested.SysOs.os_mux import OSMuxImpl
import urllib3
import certifi
import requests
from bs4 import BeautifulSoup
from root.nested.dataAccess.secure_keys_access import SecureKeysAccess
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, DECIMAL, DateTime, String
from sqlalchemy.orm.query import Query
import re

http = urllib3.PoolManager(ca_certs=certifi.where())
LOGGER = get_logger()
Base = declarative_base()
url_dict = {"VIX9D": ("https://www.cboe.com/chart/GetDownloadData/?RequestSymbol=VIX9D", "VIX9D_Data.csv"),
            "VIX": ("https://www.cboe.com/chart/GetDownloadData/?RequestSymbol=VIX", "VIX_Data.csv"),
            "VIX3M": ("https://www.cboe.com/chart/GetDownloadData/?RequestSymbol=VIX3M", "VIX3M_Data.csv"),
            "VIX6M": ("https://www.cboe.com/chart/GetDownloadData/?RequestSymbol=VIX6M", "VIX6M_Data.csv"),
            "VIX1Y": ("https://www.cboe.com/chart/GetDownloadData/?RequestSymbol=VIX1Y", "VIX1Y_Data.csv"),
            "VVIX": ("https://www.cboe.com/chart/GetDownloadData/?RequestSymbol=VVIX", "VVIX_Data.csv"),
            "SPX": ("https://www.cboe.com/chart/GetDownloadData/?RequestSymbol=SPX", "SPX_Data.csv"),
            "SKEW": ("https://www.cboe.com/chart/GetDownloadData/?RequestSymbol=SKEW", "SKEW_Data.csv")
            }
cboe_dashboard_urls_dict = {'VVIX': "http://www.cboe.com/index/dashboard/VVIX",
                            'VIX': "http://www.cboe.com/index/dashboard/VIX#vix-performance",
                            'VIX1Y': "http://www.cboe.com/index/dashboard/VIX1Y#vix1y-performance",
                            'VIX9D': "http://www.cboe.com/index/dashboard/VIX9D",
                            'VIX3M': "http://www.cboe.com/index/dashboard/VIX3M",
                            'VIX6M': "http://www.cboe.com/index/dashboard/VIX6M",
                            'SPX': "http://www.cboe.com/index/dashboard/SPX",
                            'SKEW': "http://www.cboe.com/index/dashboard/SKEW#skew-performance"}
CBOE_LOGIN_URL = "https://www.cboe.com/useradmin/formslogin?ReturnURL=/default.aspx"
LOCAL_CBOE_DATA_DIR = "/workspace/data/cboe/historicaldata/vix/"
DOWNLOAD_DATA_DIR = "/Downloads/"
CBOE_FUTURES_BASE_URL = "https://markets.cboe.com/us/futures/market_statistics/historical_data/"
VX_CSV_REMOTE_DIR = "products/csv/VX/"
VIX_TERM_STRUCTURE_DB_TABLE = "vix_term_structure"
db_column_to_df_column_mapping = {'dt_Id': "Date",
                                  'c_Symbol': 'Symbol',
                                  'c_Change': 'Change',
                                  'd_Open': 'Open',
                                  'd_High': 'High',
                                  'd_Low': 'Low',
                                  'd_Close': 'Close',
                                  'd_LastSale': 'LastSale',
                                  'd_LastTime': 'LastTime'}
MONTHLY_EXPIRY_MONTH_CODE_MAPPING = {'F': 1,
                                     'G': 2,
                                     'H': 3,
                                     'J': 4,
                                     'K': 5,
                                     'M': 6,
                                     'N': 7,
                                     'Q': 8,
                                     'U': 9,
                                     'V': 10,
                                     'X': 11,
                                     'Z': 12}


def get_cboe_sym_ts_from_csv(symbol,
                             query_start_date,
                             query_end_date):
    """
    Same function as get_cboe_sym_ts_from_db() but this retrieves data from csv.
    :param symbol:
    :param query_start_date:
    :param query_end_date:
    :return:
    """
    return 1


def get_cboe_sym_ts_from_db(query_start_date,
                            query_end_date,
                            symbol_list=['SPX'],
                            px_types=['ALL']):
    """
    Get timeseries data from mysql
    :param symbol_list:
    :param query_start_date:
    :param query_end_date:
    :param px_types: Open, High, Low, Close, LastSale
    :return:
    """
    list_poc_db_col_nm = px_types
    session = get_db_session()
    if px_types[0] == 'ALL':
        list_poc_db_col_nm = ['Id', 'Symbol', 'Open', 'High', 'Low', 'Close', 'LastSale', 'LastTime']
        the_columns = [getattr(VixTermStructure, poc_db_col_nm) for poc_db_col_nm in list_poc_db_col_nm]
        q = Query(the_columns, session=session)
    else:
        list_poc_db_col_nm = ['Id', 'Symbol'] + px_types + ['LastSale', 'LastTime']
        the_columns = [getattr(VixTermStructure, poc_db_col_nm) for poc_db_col_nm in list_poc_db_col_nm]
        q = Query(the_columns, session=session)
    from_db = q.filter(VixTermStructure.Id >= str(query_start_date),
                       VixTermStructure.Id <= str(query_end_date),
                       VixTermStructure.Symbol.in_(symbol_list)).all()
    df = pd.DataFrame().from_records(from_db)
    # we are going to need to create a MULTI-INDEX for this returned Dataframe
    # first level is the DATE (Id) and second level is the Symbol (symbol)
    # TODO: LEFT OFF HERE ON JUNE 13th, before heading out to pick up Bane with Nada.
    df.columns = list_poc_db_col_nm
    df.set_index(['Id', 'Symbol'], inplace=True)
    return df


def get_db_session():
    mysql_server_ip = SecureKeysAccess.get_mysql_server_ip()
    mysql_username = SecureKeysAccess.get_mysql_server_user()
    mysql_secret = SecureKeysAccess.get_mysql_server_secret()
    LOGGER.info("nominal_yield_curve.get_db_session(): "
                "creating db engine with pymysql, username %s, secret stays secret, and server_ip %s",
                mysql_username, mysql_server_ip)
    engine = create_engine('mysql+pymysql://' + mysql_username + ':' + mysql_secret + '@' +
                           mysql_server_ip + '/stealbasistrader')
    Session = sessionmaker()
    Session.configure(bind=engine)
    session = Session()
    return session


def read_historical_voltermstruct_csv(eod_run=True, which_product=None):
    """
    Read the historical files into a dataframe and then write out to csv and insert into db.
    the eod_run parameter asks is equal to True as a Default. This means you are not running
    a historical run but an actual EOD daily insert. That means we can insert a subset of the
    data, like the past 7 days for example, as opposed to the entire historical file.
    This will save us a lot of time when doing EOD runs.
    :param which_product:
    :param eod_run:
    :return:
    """
    if which_product is not None:
        prod_list = [which_product]
    else:
        prod_list = url_dict.keys()
    for vol_prod in prod_list:
        move_to_file_path = OSMuxImpl.get_proper_path(LOCAL_CBOE_DATA_DIR) + url_dict[vol_prod][1]
        LOGGER.info("cboe.read_historical_voltermstructure_csv(): loading csv file %s into dataframe",
                    move_to_file_path)
        df = pd.read_csv(move_to_file_path, skiprows=3, parse_dates=True, index_col=0, header=None)
        df.columns = ['Volume', 'd_Open', 'd_High', 'd_Low', 'd_Close']
        df.index.name = 'Date'
        df.drop(['Volume'], axis=1, inplace=True)
        df.to_csv(move_to_file_path + ".clean")
        # store into database table
        if eod_run:
            # reindex using the last 7 days
            last_day_in_df = df.index[-1].to_pydatetime()
            first_day_reindex = last_day_in_df - timedelta(days=7)
            df_ins = df.loc[str(first_day_reindex):]
        else:
            df_ins = df
        vix_term_structure_to_db(df_ins, symbol_to_insert=vol_prod)


def vts_to_db(vts_entry, db_session):
    entry_db = db_session.query(VixTermStructure).filter_by(Id=vts_entry.Id). \
        filter_by(Symbol=vts_entry.Symbol).first()
    if entry_db is None:
        db_session.add(vts_entry)
    else:
        if not entry_db.equals(vts_entry):
            LOGGER.error("cboe.vix_term_strucutre_to_db(): possible issue, older entry in MySQL has a "
                         "different value than overlap entries for daily EquityFutures insert, continuing insert!")
            entry_db.set_all(vts_entry)
        else:
            entry_db.set_all(vts_entry)
    db_session.commit()


def vix_term_structure_to_db(df, symbol_to_insert):
    # when the open price is not available, we need to plug in the previous last price for the open price
    # we need to do something, we can't leave it as NONE.
    entries = df.apply(vix_termstructure_to_db_vectorized, symbol_to_insert=symbol_to_insert, axis=1)
    session = get_db_session()
    session.flush()
    for entry in entries.iteritems():
        vts_to_db(entry[1], session)
        """
        entry_db = session.query(VixTermStructure).filter_by(Id=entry[1].Id). \
            filter_by(Symbol=entry[1].Symbol).first()
        if entry_db is None:
            session.add(entry[1])
        else:
            if not entry_db.equals(entry[1]):
                LOGGER.error("cboe.vix_term_strucutre_to_db(): possible issue, older entry in MySQL has a "
                             "different value than overlap entries for daily EquityFutures insert, continuing insert!")
                entry_db.set_all(entry[1])
            else:
                entry_db.set_all(entry[1])
        """
    session.commit()


def vix_termstructure_to_db_vectorized(row, symbol_to_insert):
    identifier = row.name.to_pydatetime()
    open_val = float(row.loc["d_Open"])
    high_val = float(row.loc["d_High"])
    low_val = float(row.loc["d_Low"])
    close_val = float(row.loc["d_Close"])
    try:
        change_val = row.loc["c_Change"]
    except KeyError as ke:
        change_val = "0.0"
    try:
        lastSale_val = float(row.loc["d_LastSale"])
    except KeyError as ke:
        lastSale_val = close_val
    try:
        lastTime_val = row.loc["dt_LastTime"].to_pydatetime()
    except KeyError as ke:
        lastTime_val = identifier
    entry = VixTermStructure(Id=identifier,
                             Symbol=symbol_to_insert,
                             Change=change_val,
                             Open=open_val,
                             High=high_val,
                             Low=low_val,
                             Close=close_val,
                             LastSale=lastSale_val,
                             LastTime=lastTime_val)
    return entry


def screenscrape_daily_volvals():
    return_dict = {}
    as_of_date = None
    session = get_db_session()
    for vol_prod in cboe_dashboard_urls_dict.keys():
        html_text = requests.get(cboe_dashboard_urls_dict[vol_prod]).text
        b_soup = BeautifulSoup(html_text, 'html.parser')
        attrs = {
            'id': 'div-summary'
        }
        keys_list = []
        values_list = []
        for div_elem in b_soup.find_all('div', attrs=attrs):
            for sub_elem in div_elem.find_all('h5'):
                keys_list.append(sub_elem.text)
            for sub_elem in div_elem.find_all('span'):
                values_list.append(sub_elem.text)
        as_of_date = values_list[-1]
        as_of_date = as_of_date.split("As of ")[1]
        vol_prod_dict = dict(zip(keys_list, values_list))
        vol_prod_dict['AsOfDate'] = re.sub('\s+',' ',as_of_date).strip()
        as_of_date = vol_prod_dict['AsOfDate']
        as_of_date_dt = datetime.strptime(as_of_date, '%Y-%m-%d %H:%M:%S (ET)')
        as_of_date_date = as_of_date_dt.date()
        vts = VixTermStructure(Id=as_of_date_date,
                               Symbol=vol_prod,
                               Change=vol_prod_dict['Change'],
                               Open=float(vol_prod_dict['Open']),
                               High=float(vol_prod_dict['High']),
                               Low=float(vol_prod_dict['Low']),
                               Close=float(vol_prod_dict['Prev Close']),
                               LastSale=float(vol_prod_dict['Last Sale']),
                               LastTime=as_of_date_dt)
        vts_to_db(vts, session)
        return_dict[vol_prod] = vol_prod_dict
    df = pd.DataFrame(data=return_dict)
    daily_vals_csv_file = OSMuxImpl.get_proper_path(LOCAL_CBOE_DATA_DIR) + "daily_vol_vals_" + str(as_of_date) + ".csv"
    as_of_date_dt = datetime.strptime(as_of_date, '%Y-%m-%d %H:%M:%S (ET)')
    as_of_date_dt = as_of_date_dt.replace(tzinfo=timezone("EST"))
    df.to_csv(daily_vals_csv_file)
    return df


def cboe_selenium_connect():
    driver = webdriver.Chrome('/Users/ghazymahjub/chromedriver/chromedriver')
    driver.get(CBOE_LOGIN_URL)
    driver.find_element_by_id("ContentTop_C022_emailOrUserId").send_keys("gmahjub@yahoo.com")
    driver.find_element_by_id("ContentTop_C022_Password").send_keys("8adxQBeFF$d!$qp")
    driver.find_element_by_id("ContentTop_C022_btnLogin").click()
    driver.implicitly_wait(15)
    import time
    for url_key in sorted(set(url_dict.keys())):
        LOGGER.info("cboe.cboe_selenium_connect(): sleep for 15 seconds, zzzzzzzzz....")
        time.sleep(15)
        LOGGER.info("cboe.cboe_selenium_connect(): woke up!")
        LOGGER.info("cboe.cboe_selenium_connect(): getting %s", url_dict[url_key][0])
        driver.get(url_dict[url_key][0])
        LOGGER.info("cboe.cboe_selenium_connect(): sleep for 15 seconds, zzzzzzzzz....")
        time.sleep(15)
        LOGGER.info("cboe.cboe_selenium_connect(): woke up!")
        downloaded_file_path = OSMuxImpl.get_proper_path(DOWNLOAD_DATA_DIR) + url_dict[url_key][1]
        move_to_file_path = OSMuxImpl.get_proper_path(LOCAL_CBOE_DATA_DIR) + url_dict[url_key][1]
        try:
            shutil.move(downloaded_file_path, move_to_file_path)
        except FileNotFoundError as fnfe:
            LOGGER.error("cboe.cboe_selenium_connect(): shutil move failed with... %s from file: "
                         "%s, to file: %s", fnfe.__str__(), downloaded_file_path, move_to_file_path)
    driver.quit()
    return


def get_historical_csv(key=None, url=None):
    if url is None:
        url = "https://www.cboe.com/chart/GetDownloadData/"
        payload = {'RequestSymbol': 'VIX1Y'}
    else:
        payload = None
    url_response = requests.get(url=url, params=payload)
    if url_response.status_code != 200:
        LOGGER.error("cboe.get_historical_csv(): HTTP RESPONSE STATUS CODE %s", str(url_response.status_code))
        LOGGER.error("cboe.get_historical_csv(): url request failed %s", url)
        return url_response.status_code
    attachmentFilename = url_response.headers['Content-Disposition']
    csv_filename = attachmentFilename.split('=')[1].replace('"','')
    LOGGER.info("cboe.get_historical_csv(): HTTP RESPONSE STATUS CODE " +
                str(url_response.status_code))
    historical_csv_file = OSMuxImpl.get_proper_path(LOCAL_CBOE_DATA_DIR) + csv_filename
    with open(historical_csv_file, 'wb') as f:
        for chunk in url_response.iter_content(chunk_size=128):
            f.write(chunk)
    f.close()
    return url_response.status_code


def get_historical_vx_data_from_cboe(contract_expiry_date):
    from os import path
    total_url = CBOE_FUTURES_BASE_URL + VX_CSV_REMOTE_DIR + str(contract_expiry_date)
    file_prefix = "CFE_"
    file_suffix = "_VX.csv"
    keys_list = list(MONTHLY_EXPIRY_MONTH_CODE_MAPPING.keys())
    values_list = list(MONTHLY_EXPIRY_MONTH_CODE_MAPPING.values())
    year = datetime.strftime(contract_expiry_date, "%y")
    month_code = keys_list[values_list.index(contract_expiry_date.month)]
    full_front_filename = file_prefix + month_code + str(year) + file_suffix
    path_to_front_month_file = OSMuxImpl.get_proper_path(LOCAL_CBOE_DATA_DIR) + full_front_filename
    if contract_expiry_date<datetime.now().date() and path.exists(path_to_front_month_file):
        # this contract is expired already. We may have it in the flat file system.
        return 200
    LOGGER.info("cboe.get_historical_vx_data_from_cboe(%s): getting historical vx data from url %s...",
                contract_expiry_date, total_url)
    return_value = get_historical_csv(url=total_url)
    if return_value == 404 and contract_expiry_date > \
            (datetime.now().date() + timedelta(days=180)):
        LOGGER.info("cboe.get_historical_csv(): 404 Page not found, most likely this is ok...")
    elif return_value == 404:
        LOGGER.error("cboe.get_historical_csv(): 404 Page not found, there is something WRONG! %s", total_url)
    return return_value


class VixTermStructure(Base):
    __tablename__ = VIX_TERM_STRUCTURE_DB_TABLE
    Id = Column("dt_Id", DateTime, primary_key=True)
    Symbol = Column("c_Symbol", String(40), primary_key=True)
    Change = Column("c_Change", String(40))
    Open = Column("d_Open", DECIMAL)
    High = Column("d_High", DECIMAL)
    Low = Column("d_Low", DECIMAL)
    Close = Column("d_Close", DECIMAL)
    LastSale = Column("d_LastSale", DECIMAL)
    LastTime = Column("dt_LastTime", DateTime)

    def __repr__(self):
        return "<vix_term_structure(Id='%s', Symbol='%s', Change='%s', Open='%s'," \
               "High='%s', Low='%s', Close='%s', LastSale='%s', LastTime='%s')>" % (
                   self.Id, self.Symbol, self.Change, self.Open, self.High, self.Low, self.Close,
                   self.LastSale, self.LastTime)

    def equals(self, vxts_entry_obj):
        id_check = self.Id == vxts_entry_obj.Id
        symbol_check = self.Symbol == vxts_entry_obj.Symbol
        change_check = self.Change == vxts_entry_obj.Change
        open_check = round(self.Open, 2) == round(vxts_entry_obj.Open, 2)
        high_check = round(self.High, 2) == round(vxts_entry_obj.High, 2)
        low_check = round(self.Low, 2) == round(vxts_entry_obj.Low, 2)
        close_check = round(self.Close, 2) == round(vxts_entry_obj.Close, 2)
        lastsale_check = round(self.LastSale, 2) == round(vxts_entry_obj.LastSale, 2)
        lasttime_check = self.LastTime == vxts_entry_obj.LastTime
        total_check = id_check & symbol_check & change_check & open_check & high_check & low_check & \
                      close_check & lastsale_check & lasttime_check
        return total_check

    def set_all(self, vxts_entry_obj):
        self.Id = vxts_entry_obj.Id
        self.Symbol = vxts_entry_obj.Symbol
        self.Change = vxts_entry_obj.Change
        self.Open = vxts_entry_obj.Open
        self.High = vxts_entry_obj.High
        self.Low = vxts_entry_obj.Low
        self.Close = vxts_entry_obj.Close
        self.LastTime = vxts_entry_obj.LastTime
        self.LastSale = vxts_entry_obj.LastSale


def run_cboe_eod():
    screenscrape_daily_volvals()
    cboe_selenium_connect()
    read_historical_voltermstruct_csv()


def run_cboe_daily():
    screenscrape_daily_volvals()


#read_historical_voltermstruct_csv(eod_run=False, which_product="SKEW")
#run_cboe_eod()
#df = get_cboe_sym_ts_from_db(query_start_date="2020-06-01",
#                             query_end_date="2020-06-12",
#                             symbol_list=['SPX','VIX'],
#                             px_types=['ALL'])
#print(df.head())