import datetime
import bisect
import os
import random
from datetime import datetime, timedelta, date
from itertools import islice
import numpy as np
import pandas as pd
import requests
import quandl
from quandl.errors.quandl_error import NotFoundError
from ftplib import FTP
from zipfile import ZipFile
from root.nested import get_logger
from root.nested.dataAccess.secure_keys_access import SecureKeysAccess
from root.nested.SysOs.os_mux import OSMuxImpl
from root.nested.dataAccess.WebApi import cboe

from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.palettes import Spectral11, colorblind, Inferno, BuGn, brewer
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource, LinearColorMapper, BasicTicker, \
    PrintfTickFormatter, ColorBar, LinearAxis, Range1d

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, DECIMAL, DateTime, String
from sqlalchemy.schema import Table

import warnings

warnings.filterwarnings('ignore')

QUANDL_API_KEY = SecureKeysAccess.get_vendor_api_key_static(vendor="QUANDL")
quandl.ApiConfig.api_key = QUANDL_API_KEY
LOGGER = get_logger()
Base = declarative_base()
CONT_FUT_WRITE_CSV_FILE_PATH = '/workspace/data/quandl/futures/'
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
QUARTERLY_EXPIRY_MONTH_CODE_MAPPING = {'H': 3,
                                       'M': 6,
                                       'U': 9,
                                       'Z': 12}
PREV_CONT_CONTRACT = "FRONT"
EURODOLLAR_YIELD_CURVE_TABLE = "eurodollar_yc"
WTI_CRUDE_OIL_CURVE_TABLE = "wti_crude_oil_curve"
EQUITY_FUTURES_TABLE = "equity_futures_px"
VIX_FUTURES_CURVE_TABLE = "vx_futures_curve"
CURRENCY_TABLE = "currency_fut_spot_px"

CME_SETTLEMENT_REPORT_LOCAL_DIR = "/workspace/data/cme/reports/settlement/"

fut_expiry_code_mapping = {'CL': MONTHLY_EXPIRY_MONTH_CODE_MAPPING,
                           'ES': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'JY': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'J6': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'EC': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'E6': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'A6': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'B6': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'BP': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'C6': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'CD': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'SF': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'S6': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'AD': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'ED': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'NQ': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'YM': QUARTERLY_EXPIRY_MONTH_CODE_MAPPING,
                           'VX': MONTHLY_EXPIRY_MONTH_CODE_MAPPING}


def wti_crude_oil_futs_settle_date(month=0,
                                   year=0,
                                   ib_liquidate=False):
    """
    CL: trading terminates 3 Business Days prior to the 25th day of the month.
    If the 25th is not a business day, then it is 4 business days prior to the 25th.
    :return:
    """
    if month is 0 and year is 0:
        month = datetime.now().date().month
        year = datetime.now().date().year
        twenty_fifth = datetime(year, month, 25).date()
    else:
        twenty_fifth = datetime(year, month, 25).date()
    is_bday = bool(len(pd.bdate_range(twenty_fifth, twenty_fifth)))
    if is_bday is True:
        last_trading_day = timedelta_minus_business_days(twenty_fifth, days=3)
    else:
        last_trading_day = timedelta_minus_business_days(twenty_fifth, days=4)
    # Interactive Brokers will then liquidate 2 business days prior to NYMEX cut-off.
    # so the last day we should be in this contract is 3 business days prior to NYMEX cut-off.
    if ib_liquidate is True:
        last_trading_day = timedelta_minus_business_days(last_trading_day, 3)
        LOGGER.info("cont_fut:wti_crude_oil_futs_settle_date(): Interactive Brokers liquidate flag is TRUE!")
        LOGGER.info("cont_fut_wti_crude_oil_futs_settle_date(): IB liquidate flag means this fuction returns"
                    "a last trading day that is 3 business days before the NYMEX terminates trading in this "
                    "expiration")
    LOGGER.info("cont_fut:wti_crude_oil_futs_settle_date(): last trading day for CL fut for month %s and year %s is %s",
                month, year, last_trading_day)
    return last_trading_day


def nq_futures_settle_date(month=0,
                           year=0,
                           ib_liquidate=False):
    return es_futures_settle_date(month=month,
                                  year=year,
                                  ib_liquidate=ib_liquidate)


def ym_futures_settle_date(month=0,
                           year=0,
                           ib_liquidate=False):
    return es_futures_settle_date(month=month,
                                  year=year,
                                  ib_liquidate=ib_liquidate)


def es_futures_settle_date(month=0,
                           year=0,
                           ib_liquidate=False):
    # trading terminates the 3rd Friday of the contract month.
    # We will roll on the 2nd Friday of the contract month, 1 week before termination.
    if month is 0 and year is 0:
        month = datetime.now().date().month
        year = datetime.now().date().year
    day_one_name = datetime(year, month, 1).weekday()
    # Monday is 0, Tuesday is 1, Wednesday is 2, etc... thru 6
    if day_one_name == 4:
        if ib_liquidate is not False:
            # do last day minus 1 because the contracts STOPS trading in the MORNING.
            return datetime(year, month, 15).date() - timedelta(days=1)
        # the first day of month is Friday, therefore the 3rd Friday is in two weeks, the 15th
        # we want to roll 1 week before the termination of trading, which is the 2nd Friday
        return datetime(year, month, 15).date() - timedelta(days=7)
    elif day_one_name < 4:
        if ib_liquidate is not False:
            # do last day minus 1 because the contracts STOPS trading in the MORNING.
            return datetime(year, month, 1).date() + timedelta(4 - day_one_name) + timedelta(days=14) \
                   - timedelta(days=1)
        # the first day of month is before Wednesday
        return datetime(year, month, 1).date() + timedelta(4 - day_one_name) + timedelta(days=14) - timedelta(days=7)
    elif day_one_name > 4:
        if ib_liquidate is not False:
            # do last day minus 1 because the contracts STOPS trading in the MORNING.
            return datetime(year, month, 1).date() - timedelta(day_one_name - 4) + timedelta(days=21) \
                   - timedelta(days=1)
        # the first day of the month is after Wednesday
        return datetime(year, month, 1).date() - timedelta(day_one_name - 4) + timedelta(days=21) - timedelta(days=7)


def currency_futures_settle_date(month=0,
                                 year=0,
                                 ib_liquidate=False):
    # second business day prior to the 3rd wednesday of the contract month
    # Trading terminates at 9:16 a.m. CT on the second business day prior to the third Wednesday of the contract month.
    # let's first get the 3rd Wednesday...
    day_one_name = datetime(year, month, 1)
    if day_one_name.weekday() == 2:
        # it is wednesday
        third_wed = day_one_name + timedelta(days=14)
        stop_trading_date = timedelta_minus_business_days(third_wed, 2)
    elif day_one_name.weekday() < 2:
        third_wed = day_one_name + timedelta(days=2 - day_one_name.weekday()) + timedelta(days=14)
        stop_trading_date = timedelta_minus_business_days(third_wed, 2)
    else:
        third_wed = day_one_name - timedelta(days=day_one_name.weekday() - 2) + timedelta(days=21)
        stop_trading_date = timedelta_minus_business_days(third_wed, 2)
    if ib_liquidate is False:
        return stop_trading_date.date()
    else:
        return stop_trading_date.date() - timedelta(days=7)


def euro_futures_settle_date(month=0,
                             year=0):
    return 1


def eurodollar_dynamic_roll(merged_df_row):
    if merged_df_row['PREVIOUS_DAY_OPEN_INTEREST_SPREAD'] < 0:
        return -1


def eurodollar_futures_settle_date(month=0,
                                   year=0,
                                   ib_liquidate=False):
    # the quandl data rolls when the contract terminates trading
    # eurodollar contract front month terminates trading on:
    # 2 business days (London banking days) before the 3rd Wednesday of the month
    if ib_liquidate is False:
        if datetime(year, month, 1).weekday() == 6:
            return datetime(year, month, 1).date() + timedelta(days=1)
        elif datetime(year, month, 1).weekday() == 5:
            return datetime(year, month, 1).date() + timedelta(days=2)
        else:
            return datetime(year, month, 1).date()
    if month is 0 and year is 0:
        month = datetime.now().date().month
        year = datetime.now().date().year
    day_one_name = datetime(year, month, 1).weekday()
    # Monday is 0, Tuesday is 1, Wednesday is 2, etc... thru 6
    if day_one_name == 2:
        # the first day of month is Wednesday
        return timedelta_minus_business_days(datetime(year, month, 15).date(), 2)
    elif day_one_name < 2:
        # the first day of month is before Wednesday
        return timedelta_minus_business_days(datetime(year, month, 1).date() +
                                             timedelta(2 - day_one_name) + timedelta(days=14), 2)
    elif day_one_name > 2:
        # the first day of the month is after Wednesday
        return timedelta_minus_business_days(datetime(year, month, 1).date() -
                                             timedelta(day_one_name - 2) + timedelta(days=21), 2)


def vx_futures_settle_date(month=0,
                           year=0,
                           ib_liquidate=False):
    """
    Final Settlement Date: The final settlement date for a contract with the "VX" ticker symbol is on the Wednesday
    that is 30 days prior to the third Friday of the calendar month immediately following the month in which the
    contract expires. The final settlement date for a futures contract with the "VX" ticker symbol followed by a
    number denoting the specific week of a calendar year is on the Wednesday of the week specifically denoted in the
    ticker symbol. If that Wednesday or the Friday that is 30 days following that Wednesday is a Cboe Options
    holiday, the final settlement date for the contract shall be on the business day immediately preceding that
    Wednesday. :param month: :param year: :param ib_liquidate: :return:
    """
    # expires 30 days before the third friday of the month after the month of the contract.
    use_month = (month + 1) % 12
    if use_month == 0:
        use_month = 12
    elif use_month == 1:
        year = year + 1
    first_day_of_next_month = datetime(year, use_month, 1)
    if first_day_of_next_month.weekday() == 4:
        # is it a friday? if yes, then 30 days before 14 days from now
        third_friday_of_next_month = first_day_of_next_month + timedelta(days=14)
        th = TradingHolidates(year=third_friday_of_next_month.year)
        if th.skip(third_friday_of_next_month.date()):
            # if TRUE, then we need to do minus 1 more business day
            expiry_day = third_friday_of_next_month - timedelta(days=30)
            expiry_day = timedelta_minus_business_days(expiry_day, days=1)
        else:
            expiry_day = third_friday_of_next_month - timedelta(days=30)
    elif first_day_of_next_month.weekday() < 4:
        third_friday_of_next_month = first_day_of_next_month + \
                                     timedelta(days=(4 - first_day_of_next_month.weekday())) + timedelta(days=14)
        th = TradingHolidates(year=third_friday_of_next_month.year)
        if th.skip(third_friday_of_next_month.date()):
            # if TRUE, then we need to do minus 1 more business day
            expiry_day = third_friday_of_next_month - timedelta(days=30)
            expiry_day = timedelta_minus_business_days(expiry_day, days=1)
        else:
            expiry_day = third_friday_of_next_month - timedelta(days=30)
    elif first_day_of_next_month.weekday() > 4:
        third_friday_of_next_month = first_day_of_next_month + \
                                     timedelta(days=(7 % first_day_of_next_month.weekday()) + 4) + timedelta(days=14)
        th = TradingHolidates(year=third_friday_of_next_month.year)
        if th.skip(third_friday_of_next_month.date()):
            # if TRUE, then we need to do minus 1 more business day
            expiry_day = third_friday_of_next_month - timedelta(days=30)
            expiry_day = timedelta_minus_business_days(expiry_day, days=1)
        else:
            expiry_day = third_friday_of_next_month - timedelta(days=30)
    if ib_liquidate:
        expiry_day -= timedelta(days=7)
    # last step added is to subtract 1 more business day because otherwise, we return the day the contract stops
    # trading, in the morning. Which means we don't have open, high, low data because it settled and stops in the
    # morning.
    return expiry_day.date()
    return timedelta_minus_business_days(expiry_day.date(), days=1)


fut_con_settle_func_mapping = {'CL': wti_crude_oil_futs_settle_date,
                               'VX': vx_futures_settle_date,
                               'ES': es_futures_settle_date,
                               'NQ': nq_futures_settle_date,
                               'YM': ym_futures_settle_date,
                               'J6': currency_futures_settle_date,
                               'JY': currency_futures_settle_date,
                               'E6': currency_futures_settle_date,
                               'EC': currency_futures_settle_date,
                               'A6': currency_futures_settle_date,
                               'AD': currency_futures_settle_date,
                               'BP': currency_futures_settle_date,
                               'B6': currency_futures_settle_date,
                               'C6': currency_futures_settle_date,
                               'CD': currency_futures_settle_date,
                               'SF': currency_futures_settle_date,
                               'S6': currency_futures_settle_date,
                               'ED': eurodollar_futures_settle_date}


def timedelta_plus_business_days(start_date, days):
    if not days:
        return start_date
    start_date += timedelta(days=1)
    if start_date.weekday() < 5 and not TradingHolidates(year=start_date.year).skip(start_date):
        days -= 1
    return timedelta_plus_business_days(start_date, days)


def timedelta_minus_business_days(start_date, days):
    if not days:
        return start_date
    start_date -= timedelta(days=1)
    if start_date.weekday() < 5 and not TradingHolidates(year=start_date.year).skip(start_date):
        days -= 1
    return timedelta_minus_business_days(start_date, days)


class CmeFutHolidays(object):
    """
    Using the cme holiday calendar, build a holiday calendar object for futures,
    to make sure our continuous price data is correct.
    """

    def __init__(self,
                 year):
        self.CME_HOLIDAY_CAL_BASE_URL = "https://www.cmegroup.com/tools-information/holiday-calendar/files/"
        self.LOCAL_CALENDAR_STORE_PWD = "/workspace/data/cme/calendars/"
        self.year = year

    def get_cme_holiday_calendar(self):
        hol_cal_url = self.CME_HOLIDAY_CAL_BASE_URL + str(self.year) + "-holiday-calendars.zip"
        response = requests.get(hol_cal_url)
        LOGGER.info("cont_fut.get_cme_holiday_calendar(): HTTP RESPONSE STATUS CODE " +
                    str(response.status_code))
        zip_content = response.content
        local_calendar_zip_full_path = OSMuxImpl.get_proper_path(self.LOCAL_CALENDAR_STORE_PWD)
        with open(local_calendar_zip_full_path + str(self.year) + "Holiday_Calendar" + ".zip", 'wb') as f:
            f.write(zip_content)
        cal_zip_file = ZipFile(local_calendar_zip_full_path + str(self.year) + "Holiday_Calendar" +
                               ".zip")
        LOGGER.info("nominal_yield_curve.get_xsd(): extracting zip file into directory %s",
                    local_calendar_zip_full_path)
        cal_zip_file.extractall(path=local_calendar_zip_full_path)
        cal_zip_file.close()
        # https://www.cmegroup.com/tools-information/holiday-calendar/files/2009-holiday-calendars.zip


class ContFut(object):
    """
    Build a continuous futures price series from multiple contract expirations.
    """

    def __init__(self,
                 symbol,
                 expiry_month_code=0,
                 expiry_year=0):

        self.symbol = symbol
        self.expiry_month_code = expiry_month_code
        self.expiry_year = expiry_year

    def futures_rollover_weights(self,
                                 start_date,
                                 expiry_dates,
                                 contracts,
                                 rollover_days=5):
        """ Construct a pandas dataframe that contains weights (between 0.0 and 1.0)
        of contract positions to hold in order to carry out a rollover of rollover_days
        prior to the expiration of the earliest contract. The matrix can then be "multiplied"
        with another DataFrame containing the settle prices of each contract in order to produce
        a continuous time series futures contract."""
        # construct a sequence of dates beginning from the earliest contract start
        # date to the end of the final contract
        dates = pd.date_range(start_date, expiry_dates[-1], freq='B')
        # create the 'roll weights' Dataframe that will store the multipliers for
        # each contract (between 0.0 and 1.0)
        roll_weights = pd.DataFrame(np.zeros((len(dates), len(contracts.columns))), index=dates,
                                    columns=contracts.columns)
        prev_date = roll_weights.index[0]
        last_date = contracts.index[-1]
        # Loop through each contract and create the specific weightings for
        # each contract depending upon the settlement date and rollover_days
        for i, (item, ex_date) in enumerate(expiry_dates.iteritems()):
            if i < len(expiry_dates) - 1:
                roll_weights.ix[prev_date:ex_date - pd.offsets.BDay(), item] = 1
                roll_rng = pd.date_range(end=ex_date - pd.offsets.BDay(),
                                         periods=rollover_days + 1, freq='B')
                # Create a sequence of roll weights (i.e. [0.0,0.2,...,0.8,1.0]
                # and use these to adjust the weightings of each future
                decay_weights = np.linspace(0, 1, rollover_days + 1)
                roll_weights.ix[roll_rng, item] = 1 - decay_weights
                roll_weights.ix[roll_rng, expiry_dates.index[i + 1]] = decay_weights
            else:
                roll_weights.ix[prev_date:, item] = 1
            prev_date = ex_date
        since_last_roll = pd.date_range(start=prev_date, end=last_date)
        roll_weights = roll_weights.reindex(since_last_roll, fill_value=1.0)
        roll_weights.drop(columns=contracts.columns, axis=1, inplace=True)
        roll_weights_consolidated = roll_weights.apply(vectorized_cont_fut_weights, axis=1)
        return roll_weights_consolidated

    def merge_expirations(self,
                          start_date,
                          expiry_dates,
                          contracts):

        dates = pd.date_range(start_date, contracts.index[-1], freq='B')
        merged_px = pd.DataFrame(np.zeros((len(dates), len(contracts.columns))), index=dates,
                                 columns=contracts.columns)
        prev_date = merged_px.index[0]
        last_date = contracts.index[-1]
        for i, (item, ex_date) in enumerate(expiry_dates.iteritems()):
            LOGGER.info("cont_fut.merge_expirations(): expiry date range is %s (in tuple form)", ex_date)
            if i < len(expiry_dates) - 1:
                if ex_date[1] > last_date.date():
                    LOGGER.info("cont_fut.merge_expirations(): broke out of for loop because %s "
                                "beyond the last date in contract price data %s", ex_date[1], last_date.date())
                    break
                roll_rng = pd.date_range(start=timedelta_plus_business_days(ex_date[1], 1),
                                         end=ex_date[0], freq='B')
                LOGGER.info("cont_fut:merge_expirations(): range of roll dates before intersection "
                            "with price data %s", roll_rng)
                all_days = contracts.index  # Store all the days
                new_roll_rng = roll_rng.intersection(all_days)
                LOGGER.info("cont_fut:merge_expirations(): range of roll dates %s", new_roll_rng)
                merged_px.ix[new_roll_rng, "MERGED_" + self.symbol + "_SETTLE"] = \
                    (contracts.ix[new_roll_rng, contracts.columns[1]])
                merged_px.ix[new_roll_rng, "MERGED_" + self.symbol + "_OPEN"] = \
                    (contracts.ix[new_roll_rng, contracts.columns[3]])
                merged_px.ix[new_roll_rng, "MERGED_" + self.symbol + "_HIGH"] = \
                    (contracts.ix[new_roll_rng, contracts.columns[5]])
                merged_px.ix[new_roll_rng, "MERGED_" + self.symbol + "_LOW"] = \
                    (contracts.ix[new_roll_rng, contracts.columns[7]])
                merged_px.ix[new_roll_rng, "MERGED_" + self.symbol + "_LAST"] = \
                    (contracts.ix[new_roll_rng, contracts.columns[9]])
                merged_px.ix[new_roll_rng, "MERGED_" + self.symbol + "_VOLUME"] = \
                    (contracts.ix[new_roll_rng, contracts.columns[11]])
                merged_px.ix[new_roll_rng, "MERGED_" + self.symbol + "_PREVIOUS_DAY_OPEN_INTEREST"] = \
                    (contracts.ix[new_roll_rng, contracts.columns[13]])
            prev_date = ex_date[1]
        merged_px.drop(columns="NEAR_MONTH_" + self.symbol + "_SETTLE", inplace=True)
        merged_px.drop(columns="BACK_MONTH_" + self.symbol + "_SETTLE", inplace=True)
        merged_px.drop(columns="NEAR_MONTH_" + self.symbol + "_OPEN", inplace=True)
        merged_px.drop(columns="BACK_MONTH_" + self.symbol + "_OPEN", inplace=True)
        merged_px.drop(columns="NEAR_MONTH_" + self.symbol + "_HIGH", inplace=True)
        merged_px.drop(columns="BACK_MONTH_" + self.symbol + "_HIGH", inplace=True)
        merged_px.drop(columns="NEAR_MONTH_" + self.symbol + "_LOW", inplace=True)
        merged_px.drop(columns="BACK_MONTH_" + self.symbol + "_LOW", inplace=True)
        merged_px.drop(columns="NEAR_MONTH_" + self.symbol + "_LAST", inplace=True)
        merged_px.drop(columns="BACK_MONTH_" + self.symbol + "_LAST", inplace=True)
        merged_px.drop(columns="NEAR_MONTH_" + self.symbol + "_VOLUME", inplace=True)
        merged_px.drop(columns="BACK_MONTH_" + self.symbol + "_VOLUME", inplace=True)
        merged_px.drop(columns="NEAR_MONTH_" + self.symbol + "_PREVIOUS_DAY_OPEN_INTEREST", inplace=True)
        merged_px.drop(columns="BACK_MONTH_" + self.symbol + "_PREVIOUS_DAY_OPEN_INTEREST", inplace=True)
        merged_px = pd.merge(merged_px, contracts, left_index=True, right_index=True, how='outer')
        merged_px['MERGED_' + self.symbol + "_SETTLE"]. \
            fillna(merged_px['NEAR_MONTH_' + self.symbol + "_SETTLE"], inplace=True)
        merged_px['MERGED_' + self.symbol + "_OPEN"]. \
            fillna(merged_px['NEAR_MONTH_' + self.symbol + "_OPEN"], inplace=True)
        merged_px['MERGED_' + self.symbol + "_HIGH"]. \
            fillna(merged_px['NEAR_MONTH_' + self.symbol + "_HIGH"], inplace=True)
        merged_px['MERGED_' + self.symbol + "_LOW"]. \
            fillna(merged_px['NEAR_MONTH_' + self.symbol + "_LOW"], inplace=True)
        merged_px['MERGED_' + self.symbol + "_LAST"]. \
            fillna(merged_px['NEAR_MONTH_' + self.symbol + "_LAST"], inplace=True)
        merged_px['MERGED_' + self.symbol + "_VOLUME"]. \
            fillna(merged_px['NEAR_MONTH_' + self.symbol + "_VOLUME"], inplace=True)
        merged_px['MERGED_' + self.symbol + "_PREVIOUS_DAY_OPEN_INTEREST"]. \
            fillna(merged_px['NEAR_MONTH_' + self.symbol + "_PREVIOUS_DAY_OPEN_INTEREST"], inplace=True)
        merged_px['PREVIOUS_DAY_OPEN_INTEREST_SPREAD'] = \
            merged_px['NEAR_MONTH_' + self.symbol + '_PREVIOUS_DAY_OPEN_INTEREST'] - \
            merged_px['BACK_MONTH_' + self.symbol + '_PREVIOUS_DAY_OPEN_INTEREST']
        return merged_px

    def eurodollar_price_roll(self,
                              start_date,
                              expiry_dates,
                              contracts):
        rollover_days = 0
        dates = pd.date_range(start_date, contracts.index[-1], freq='B')
        # dates = pd.date_range(start_date, expiry_dates[-1][0], freq='B')
        # create the 'roll weights' Dataframe that will store the multipliers for
        # each contract (between 0.0 and 1.0)
        roll_weights = pd.DataFrame(np.zeros((len(dates), len(contracts.columns))), index=dates,
                                    columns=contracts.columns)
        dynamic_roll = pd.DataFrame(np.zeros((len(dates), 2)), index=dates, columns=['CONT_CONTRACT', 'oi_spread'])
        prev_date = roll_weights.index[0]
        last_date = contracts.index[-1]
        # Loop through each contract and create the specific weightings for
        # each contract depending upon the settlement date and rollover_days
        for i, (item, ex_date) in enumerate(expiry_dates.iteritems()):
            LOGGER.info("cont_fut.eurodollar_price_roll(): expiry date range is %s (in tuple form)", ex_date)
            if i < len(expiry_dates) - 1:
                if ex_date[1] > last_date.date():
                    LOGGER.info("cont_fut.eurodollar_price_roll(): broke out of for loop because %s "
                                "beyond the last date in contract price data %s", ex_date[1], last_date.date())
                    break
                # roll_rng = pd.date_range(end=ex_date[0], start=ex_date[1], freq='B')
                roll_rng = pd.date_range(end=ex_date[1], periods=rollover_days + 1, freq='B')
                LOGGER.info("cont_fut:eurodollar_price_roll(): range of roll dates before intersection "
                            "with price data %s", roll_rng)
                all_days = contracts.index  # Store all the days
                roll_rng = roll_rng.intersection(all_days)
                LOGGER.info("cont_fut:eurodollar_price_roll(): range of roll dates %s", roll_rng)
                roll_weights.ix[roll_rng, item + "_SETTLE"] = \
                    contracts.ix[roll_rng, contracts.columns[1]] - contracts.ix[roll_rng, contracts.columns[0]]
                roll_weights.ix[roll_rng, item + "_OPEN"] = \
                    contracts.ix[roll_rng, contracts.columns[3]] - contracts.ix[roll_rng, contracts.columns[2]]
                roll_weights.ix[roll_rng, item + "_HIGH"] = \
                    contracts.ix[roll_rng, contracts.columns[5]] - contracts.ix[roll_rng, contracts.columns[4]]
                roll_weights.ix[roll_rng, item + "_LOW"] = \
                    contracts.ix[roll_rng, contracts.columns[7]] - contracts.ix[roll_rng, contracts.columns[6]]
                roll_weights.ix[roll_rng, item + "_LAST"] = \
                    contracts.ix[roll_rng, contracts.columns[9]] - contracts.ix[roll_rng, contracts.columns[8]]
                roll_weights.ix[roll_rng, item + "_VOLUME"] = \
                    contracts.ix[roll_rng, contracts.columns[11]] - contracts.ix[roll_rng, contracts.columns[10]]
                roll_weights.ix[roll_rng, item + "_VOLUME"] = 0.0
                roll_weights.ix[roll_rng, item + "_PREVIOUS_DAY_OPEN_INTEREST"] = \
                    contracts.ix[roll_rng, contracts.columns[13]] - contracts.ix[roll_rng, contracts.columns[12]]
                roll_weights.ix[roll_rng, item + "_PREVIOUS_DAY_OPEN_INTEREST"] = 0.0
                roll_weights.bfill(inplace=True)
                roll_weights.fillna(value=0.0, inplace=True)
            prev_date = ex_date[0]
        roll_weights.drop(columns=contracts.columns, axis=1, inplace=True)
        # PREV_CONT_CONTRACT = "FRONT"
        # dynamic_roll = contracts.apply(vectorized_cont_fut_eurodollar_weights, roll_rng=roll_rng, axis=1)
        # merged_df = pd.merge(left=roll_weights, right=dynamic_roll,
        #                     left_index=True, right_index=True)
        # write_to_csv(merged_df, "ED", "Start", "end")
        final_df = roll_weights.apply(vectorized_cont_fut_eurodollar_weights, axis=1)
        return final_df

    def proportional_price_roll(self,
                                start_date,
                                expiry_dates,
                                contracts):
        rollover_days = 0
        dates = pd.date_range(start_date, contracts.index[-1], freq='B')
        # dates = pd.date_range(start_date, expiry_dates[-1][0], freq='B')
        # create the 'roll weights' Dataframe that will store the multipliers for
        # each contract (between 0.0 and 1.0)
        roll_weights = pd.DataFrame(np.zeros((len(dates), len(contracts.columns))), index=dates,
                                    columns=contracts.columns)
        prev_date = roll_weights.index[0]
        last_date = contracts.index[-1]
        # Loop through each contract and create the specific weightings for
        # each contract depending upon the settlement date and rollover_days
        for i, (item, ex_date) in enumerate(expiry_dates.iteritems()):
            LOGGER.info("cont_fut.proportional_price_roll(): expiry date range is %s (in tuple form)", ex_date)
            if i < len(expiry_dates) - 1:
                if ex_date[1] > last_date.date():
                    LOGGER.info("cont_fut.proportional_price_roll(): broke out of for loop because %s "
                                "beyond the last date in contract price data %s", ex_date[1], last_date.date())
                    # break
                roll_rng = pd.date_range(end=ex_date[1], periods=rollover_days + 1, freq='B')
                # roll_rng = pd.date_range(start=ex_date[0], end=ex_date[1], freq='B')
                LOGGER.info("cont_fut:proportional_price_roll(): range of roll dates before intersection "
                            "with price data %s", roll_rng)
                all_days = contracts.index  # Store all the days
                roll_rng = roll_rng.intersection(all_days)
                LOGGER.info("cont_fut:proportional_price_roll(): range of roll dates %s", roll_rng)
                roll_weights.ix[roll_rng, item + "_SETTLE"] = \
                    (contracts.ix[roll_rng, contracts.columns[1]] / contracts.ix[
                        roll_rng, contracts.columns[0]] - 1) + 1
                roll_weights.ix[roll_rng, item + "_OPEN"] = \
                    (contracts.ix[roll_rng, contracts.columns[3]] / contracts.ix[
                        roll_rng, contracts.columns[2]] - 1) + 1
                roll_weights.ix[roll_rng, item + "_HIGH"] = \
                    (contracts.ix[roll_rng, contracts.columns[5]] / contracts.ix[
                        roll_rng, contracts.columns[4]] - 1) + 1
                roll_weights.ix[roll_rng, item + "_LOW"] = \
                    (contracts.ix[roll_rng, contracts.columns[7]] / contracts.ix[
                        roll_rng, contracts.columns[6]] - 1) + 1
                roll_weights.ix[roll_rng, item + "_LAST"] = \
                    (contracts.ix[roll_rng, contracts.columns[9]] / contracts.ix[
                        roll_rng, contracts.columns[8]] - 1) + 1
                roll_weights.ix[roll_rng, item + "_VOLUME"] = 1.0
                roll_weights.ix[roll_rng, item + "_PREVIOUS_DAY_OPEN_INTEREST"] = 1.0
                roll_weights.bfill(inplace=True)
                roll_weights.fillna(value=1.0, inplace=True)
            prev_date = ex_date[0]
        roll_weights.drop(columns=contracts.columns, axis=1, inplace=True)
        prop_roll_weights_cons = roll_weights.apply(vectorized_cont_fut_prop_weights, axis=1)
        return prop_roll_weights_cons


def vectorized_cont_fut_eurodollar_weights(row):
    settle_adj_factor = 0.0
    open_adj_factor = 0.0
    high_adj_factor = 0.0
    low_adj_factor = 0.0
    last_adj_factor = 0.0
    volume_adj_factor = 0.0
    previous_day_open_interest_adj_factor = 0.0
    for val_col in row.index:
        if "SETTLE" in val_col:
            settle_adj_factor = row[val_col] + settle_adj_factor
        elif "OPEN" in val_col:
            open_adj_factor = row[val_col] + open_adj_factor
        elif "HIGH" in val_col:
            high_adj_factor = row[val_col] + high_adj_factor
        elif "LOW" in val_col:
            low_adj_factor = row[val_col] + low_adj_factor
        elif "LAST" in val_col:
            last_adj_factor = row[val_col] + last_adj_factor
        elif "VOLUME" in val_col:
            volume_adj_factor = row[val_col] + volume_adj_factor
        elif "PREVIOUS_DAY_OPEN_INTEREST" in val_col:
            previous_day_open_interest_adj_factor = row[val_col] + previous_day_open_interest_adj_factor
    row["settle_adj_factor"] = settle_adj_factor
    row["open_adj_factor"] = open_adj_factor
    row['high_adj_factor'] = high_adj_factor
    row['low_adj_factor'] = low_adj_factor
    row['last_adj_factor'] = last_adj_factor
    row['volume_adj_factor'] = volume_adj_factor
    row['previous_day_open_interest_adj_factor'] = previous_day_open_interest_adj_factor
    return row


def vectorized_cont_fut_prop_weights(row):
    settle_adj_factor = 1.0
    open_adj_factor = 1.0
    high_adj_factor = 1.0
    low_adj_factor = 1.0
    last_adj_factor = 1.0
    volume_adj_factor = 1.0
    previous_day_open_interest_adj_factor = 1.0
    for val_col in row.index:
        if "SETTLE" in val_col:
            settle_adj_factor = row[val_col] * settle_adj_factor
        elif "OPEN" in val_col:
            open_adj_factor = row[val_col] * open_adj_factor
        elif "HIGH" in val_col:
            high_adj_factor = row[val_col] * high_adj_factor
        elif "LOW" in val_col:
            low_adj_factor = row[val_col] * low_adj_factor
        elif "LAST" in val_col:
            last_adj_factor = row[val_col] * last_adj_factor
        elif "VOLUME" in val_col:
            volume_adj_factor = row[val_col] * volume_adj_factor
        elif "PREVIOUS_DAY_OPEN_INTEREST" in val_col:
            previous_day_open_interest_adj_factor = row[val_col] * previous_day_open_interest_adj_factor
    row["settle_adj_factor"] = settle_adj_factor
    row["open_adj_factor"] = open_adj_factor
    row['high_adj_factor'] = high_adj_factor
    row['low_adj_factor'] = low_adj_factor
    row['last_adj_factor'] = last_adj_factor
    row['volume_adj_factor'] = volume_adj_factor
    row['previous_day_open_interest_adj_factor'] = previous_day_open_interest_adj_factor
    return row


def vectorized_cont_fut_weights(row):
    index_cnt = 0
    for val_col in row.index:
        if row[val_col] != 0.0 and not pd.isna(row[val_col]) and index_cnt == 0:
            row['Front_Month_Weight'] = row[val_col]
            index_cnt += 1
        elif row[val_col] != 0.0 and not pd.isna(row[val_col]) and index_cnt == 1:
            row['Back_Month_Weight'] = row[val_col]
            index_cnt += 1
    if index_cnt == 1:
        row['Back_Month_Weight'] = 0.0
    elif index_cnt == 0:
        LOGGER.error("cont_fut.vectorized_cont_fut_weights(): failed to set weights for both front and back contract!")
    return row


def multiply_df_columns(multiplier, multiplicand):
    return multiplicand * multiplier


def run_spot_currency_roll(spot_quandl_sym,
                           spot_con_desc,
                           start_date,
                           end_date):
    try:
        con = quandl.get(spot_quandl_sym, start_date=start_date, end_date=end_date)
    except NotFoundError as nfe:
        LOGGER.error("cont_fut.run_spot_currency_roll_daily(): error in quandl.get(%s) %s",
                     spot_quandl_sym, str(nfe.quandl_message))
    con.rename(columns={"Date": "SPOT_MKT_OBS_DATE",
                        "Value": "SPOT_MKT_OBS_VALUE"}, inplace=True)
    con = con.where((pd.notnull(con)), None)
    con.fillna(method='ffill', inplace=True)
    con['SPOT_MKT_VALUE_DESC'] = spot_con_desc
    return con


def run_currency_futures_roll_daily(symbol,
                                    quandl_symbol_near,
                                    quandl_symbol_far,
                                    start_date,
                                    end_date,
                                    year_range,
                                    num_expirations,
                                    roll_type="flatprice"):
    return run_equity_futures_roll_daily(symbol=symbol,
                                         quandl_symbol_near=quandl_symbol_near,
                                         quandl_symbol_far=quandl_symbol_far,
                                         start_date=start_date,
                                         end_date=end_date,
                                         year_range=year_range,
                                         num_expirations=num_expirations,
                                         roll_type=roll_type)


def run_equity_futures_roll_daily(symbol,
                                  quandl_symbol_near,
                                  quandl_symbol_far,
                                  start_date,
                                  end_date,
                                  year_range,
                                  num_expirations,
                                  roll_type="flatprice"):
    if roll_type is "flatprice":
        return run_eurodollar_roll_method_daily(symbol,
                                                quandl_symbol_near,
                                                quandl_symbol_far,
                                                start_date,
                                                end_date,
                                                year_range,
                                                num_expirations)
    elif roll_type is "proportional":
        return run_proportional_roll_method_daily(symbol,
                                                  quandl_symbol_near,
                                                  quandl_symbol_far,
                                                  start_date,
                                                  end_date,
                                                  year_range,
                                                  num_expirations)


def run_currency_futures_roll(symbol,
                              quandl_symbol_near,
                              quandl_symbol_far,
                              start_date,
                              end_date,
                              year_range,
                              num_expirations,
                              roll_type="flatprice"):
    return run_equity_futures_roll(symbol=symbol,
                                   quandl_symbol_near=quandl_symbol_near,
                                   quandl_symbol_far=quandl_symbol_far,
                                   start_date=start_date,
                                   end_date=end_date,
                                   year_range=year_range,
                                   num_expirations=num_expirations,
                                   roll_type=roll_type)


def run_equity_futures_roll(symbol,
                            quandl_symbol_near,
                            quandl_symbol_far,
                            start_date,
                            end_date,
                            year_range,
                            num_expirations,
                            roll_type="flatprice"):
    if roll_type is "flatprice":
        return run_eurodollar_roll_method(symbol,
                                          quandl_symbol_near,
                                          quandl_symbol_far,
                                          start_date,
                                          end_date,
                                          year_range,
                                          num_expirations)
    elif roll_type is "proportional":
        return run_proportional_roll_method(symbol,
                                            quandl_symbol_near,
                                            quandl_symbol_far,
                                            start_date,
                                            end_date,
                                            year_range,
                                            num_expirations)


def run_proportional_roll_method_daily(symbol,
                                       quandl_symbol_near,
                                       quandl_symbol_far,
                                       start_date,
                                       end_date,
                                       year_range,
                                       num_expirations):
    """
    Run the proportional daily insert. This means this function must check if we are ready
    to roll the contract. If we are not ready to roll the contract, then we simply insert
    the price data for this date as is. If we are ready to roll the contract, then we must
    update the roll factors for all the price data going backwards. This means running the
    run_proportional_roll_method() for all the years, with standard parameters.
    :param symbol:
    :param quandl_symbol_near:
    :param quandl_symbol_far:
    :param start_date:
    :param end_date:
    :param year_range:
    :param num_expirations:
    :return:
    """
    # first check if we are ready to roll the Eurodollar contracts.
    FUT_MONTH_CODE_MAPPING = fut_expiry_code_mapping[symbol]
    func = fut_con_settle_func_mapping[symbol]
    curr_month = datetime.now().date().month % 12
    pos = bisect.bisect_right(sorted(set(QUARTERLY_EXPIRY_MONTH_CODE_MAPPING.values())), curr_month)
    month_codes_list = sorted(set(QUARTERLY_EXPIRY_MONTH_CODE_MAPPING.keys()))
    month_code = month_codes_list[pos]
    ltd = func(FUT_MONTH_CODE_MAPPING[month_code], year=datetime.now().date().year)
    if datetime.now().date() >= ltd:
        LOGGER.info("cont_fut.run_proportional_roll_method_daily(): today's date is %s, "
                    "which is greater than last trading date %s, doing full roll!",
                    str(datetime.now().date()), str(ltd))
        res = run_proportional_roll_method(symbol,
                                           quandl_symbol_near,
                                           quandl_symbol_far,
                                           start_date,
                                           end_date,
                                           year_range,
                                           num_expirations)
    else:
        modified_start_date = datetime.now().date() - timedelta(days=7)
        modified_end_date = datetime.now().date()
        LOGGER.info("cont_fut.run_proportional_roll_method_daily(): params are symbol %s,"
                    "quandl_symbol_near %s, quandl_symbol_far %s, start_date %s, end_date %s,"
                    " year_range %s, num_expirations %s", symbol, quandl_symbol_near, quandl_symbol_far,
                    modified_start_date, modified_end_date, str(datetime.now().date().year), str(1))
        res = run_proportional_roll_method(symbol=symbol,
                                           quandl_symbol_near=quandl_symbol_near,
                                           quandl_symbol_far=quandl_symbol_far,
                                           start_date=modified_start_date,
                                           end_date=modified_end_date,
                                           year_range=[datetime.now().date().year],
                                           num_expirations=1)
        return res, modified_start_date, modified_end_date
    return res, start_date, end_date


def run_eurodollar_roll_method_daily(symbol,
                                     quandl_symbol_near,
                                     quandl_symbol_far,
                                     start_date,
                                     end_date,
                                     year_range,
                                     num_expirations):
    """
    Run the eurodollar daily insert. This means this function must check if we are ready
    to roll the contract. If we are not ready to roll the contract, then we simply insert
    the price data for this date as is. If we are ready to roll the contract, then we must
    update the roll factors for all the price data going backwards. This means running the
    run_eurodollar_roll_method() for all the years, with standard parameters.
    :param symbol:
    :param quandl_symbol_near:
    :param quandl_symbol_far:
    :param start_date:
    :return:
    """
    # first check if we are ready to roll the Eurodollar contracts.
    FUT_MONTH_CODE_MAPPING = fut_expiry_code_mapping[symbol]
    func = fut_con_settle_func_mapping[symbol]
    curr_month = datetime.now().date().month % 12
    pos = bisect.bisect_right(sorted(set(QUARTERLY_EXPIRY_MONTH_CODE_MAPPING.values())), curr_month)
    month_codes_list = sorted(set(QUARTERLY_EXPIRY_MONTH_CODE_MAPPING.keys()))
    month_code = month_codes_list[pos]
    ltd = func(FUT_MONTH_CODE_MAPPING[month_code], year=datetime.now().date().year)
    if datetime.now().date() >= ltd:
        LOGGER.info("cont_fut.run_eurodollar_roll_method_daily(): today's date is %s, "
                    "which is greater than last trading date, doing full roll!",
                    str(datetime.now().date()), str(ltd))
        res = run_eurodollar_roll_method(symbol,
                                         quandl_symbol_near,
                                         quandl_symbol_far,
                                         start_date,
                                         end_date,
                                         year_range,
                                         num_expirations)
    else:
        modified_start_date = datetime.now().date() - timedelta(days=7)
        modified_end_date = datetime.now().date()
        LOGGER.info("cont_fut.run_eurodollar_roll_method_daily(): params are symbol %s,"
                    "quandl_symbol_near %s, quandl_symbol_far %s, start_date %s, end_date %s,"
                    " year_range %s, num_expirations %s", symbol, quandl_symbol_near, quandl_symbol_far,
                    modified_start_date, modified_end_date, str(datetime.now().date().year), str(1))
        res = run_eurodollar_roll_method(symbol=symbol,
                                         quandl_symbol_near=quandl_symbol_near,
                                         quandl_symbol_far=quandl_symbol_far,
                                         start_date=modified_start_date,
                                         end_date=modified_end_date,
                                         year_range=[datetime.now().date().year],
                                         num_expirations=1)
        return res, modified_start_date, modified_end_date
    return res, start_date, end_date


def high_low_range(row):
    if row.High is not None and row.Low is not None:
        # both the high and low are not Null, so we can calculate range
        row['DailyRange'] = row.High - row.Low
    elif row.High is None and row.Low is not None:
        # High is null and low is not null
        row['DailyRange'] = row.High
    else:
        row['DailyRange'] = row.Low
    return row


def handle_quandl_missing_data(symbol, df, rolling_window=5):
    df.dropna(subset=['Settle'], inplace=True)
    LOGGER.info("cont_fut.handle_quandl_missing_data(): number of NA values in %s Open column: %s",
                symbol, df.Open.isna().sum())
    LOGGER.info("cont_fut.handle_quandl_missing_data(): number of NA values in %s High column: %s",
                symbol, df.High.isna().sum())
    LOGGER.info("cont_fut.handle_quandl_missing_data(): number of NA values in %s Low column: %s",
                symbol, df.Low.isna().sum())
    LOGGER.info("cont_fut.handle_quandl_missing_data(): number of NA values in %s Settle column: %s",
                symbol, df.Settle.isna().sum())
    LOGGER.info("cont_fut.handle_quandl_missing_data(): number of NA values in %s Last column: %s",
                symbol, df.Last.isna().sum())
    LOGGER.info("cont_fut.handle_quandl_missing_data(): number of NA values in %s Volume column: %s",
                symbol, df.Volume.isna().sum())
    LOGGER.info(
        "cont_fut.handle_quandl_missing_data(): number of NA values in %s Previous Day Open Interest column: %s",
        symbol, df['Previous Day Open Interest'].isna().sum())
    df = df.apply(high_low_range, axis=1)
    LOGGER.info("cont_fut.handle_quandl_missing_data(): number of NA values in %s Daily range: %s",
                symbol, df['DailyRange'].isna().sum())
    df['DailyRange'].fillna(0.0, inplace=True)
    LOGGER.info("cont_fut.handle_quandl_missing_data(): after DailyRange fillna(), number of NA values in %s "
                "Daily range: %s", symbol, df['DailyRange'].isna().sum())
    df['PctRange'] = df.DailyRange.div(df.Settle.shift(periods=1))
    how_many_zeros = df[df.PctRange.eq(0.0)].shape[0]
    LOGGER.info("cont_fut.handle_quandl_missing_data(): replace 0.0 with NAN, how many to replace? %s %s",
                symbol, how_many_zeros)
    df.PctRange = df.PctRange.replace({0.0: np.nan})
    LOGGER.info("cont_fut.handle_quandl_missing_data(): after PctRange 0->Nan replace, number of NA values in %s "
                "Daily range: %s", symbol, df['PctRange'].isna().sum())
    df.PctRange.fillna(method='ffill', inplace=True)
    df.High.fillna(value=df.PctRange.mul(df.Settle.shift(periods=1)) + df.Low, inplace=True)
    df.Low.fillna(value=df.High.sub(df.PctRange.mul(df.Settle.shift(periods=1))), inplace=True)
    df.Open.fillna(value=df.Last.shift(periods=1), inplace=True)
    df.Last.fillna(value=df.Settle, inplace=True)
    df.Volume.fillna(value=df.Volume.rolling(window=rolling_window).mean(), inplace=True)
    df['Previous Day Open Interest'].fillna(method='ffill', inplace=True)
    LOGGER.info("cont_fut.handle_quandl_missing_data(): POST HANDLE, None values in %s Open column: %s",
                symbol, df.Open.isna().sum())
    LOGGER.info("cont_fut.handle_quandl_missing_data(): POST HANDLE, None values in %s High column: %s",
                symbol, df.High.isna().sum())
    LOGGER.info("cont_fut.handle_quandl_missing_data(): POST HANDLE, None values in %s Low column: %s",
                symbol, df.Low.isna().sum())
    LOGGER.info("cont_fut.handle_quandl_missing_data(): POST HANDLE, None values in %s Settle column: %s",
                symbol, df.Settle.isna().sum())
    LOGGER.info("cont_fut.handle_quandl_missing_data(): POST HANDLE, None values in %s Last column: %s",
                symbol, df.Last.isna().sum())
    LOGGER.info("cont_fut.handle_quandl_missing_data(): POST HANDLE, None values in %s Volume column: %s",
                symbol, df.Volume.isna().sum())
    LOGGER.info(
        "cont_fut.handle_quandl_missing_data(): POST HANDLE, None values in %s Previous Day Open Interest column: %s",
        symbol, df['Previous Day Open Interest'].isna().sum())
    return df


def column_name_check(df):
    """
    Use this function to create a standard list of column headers to operate on.
    For example, quandl uses Close and Last interchangeably, depending on the timeseries.
    :param df:
    :return:
    """
    df.rename(columns={"Close": "Last"}, inplace=True)
    df.rename(columns={"Total Volume": "Volume"}, inplace=True)
    df.rename(columns={"Prev. Day Open Interest": "Previous Day Open Interest"}, inplace=True)
    return df


def run_eurodollar_roll_method(symbol,
                               quandl_symbol_near,
                               quandl_symbol_far,
                               start_date,
                               end_date,
                               year_range,
                               num_expirations):
    cf = ContFut(symbol=symbol)
    try:
        con_near = quandl.get(quandl_symbol_near, start_date=start_date, end_date=end_date)
    except NotFoundError as nfe:
        LOGGER.error("cont_fut.run_proportional_roll_method(): error in quandl.get(%s) %s",
                     quandl_symbol_near, str(nfe.quandl_message))
    try:
        con_far = quandl.get(quandl_symbol_far, start_date=start_date, end_date=end_date)
    except NotFoundError as nfe:
        LOGGER.error("cont_fut.run_proportional_roll_method(): error in quandl.get(%s) %s",
                     quandl_symbol_far, str(nfe.quandl_message))
    con_near = column_name_check(df=con_near)
    con_far = column_name_check(df=con_far)
    con_near = con_near.where((pd.notnull(con_near)), None)
    con_near = handle_quandl_missing_data(symbol, con_near)
    con_far = con_far.where((pd.notnull(con_far)), None)
    con_far = handle_quandl_missing_data(symbol, con_far)
    cons = pd.DataFrame({"NEAR_MONTH_" + symbol + '_SETTLE': con_near['Settle'],
                         "BACK_MONTH_" + symbol + '_SETTLE': con_far['Settle'],
                         "NEAR_MONTH_" + symbol + '_OPEN': con_near['Open'],
                         "BACK_MONTH_" + symbol + '_OPEN': con_far['Open'],
                         "NEAR_MONTH_" + symbol + '_HIGH': con_near['High'],
                         "BACK_MONTH_" + symbol + '_HIGH': con_far['High'],
                         "NEAR_MONTH_" + symbol + '_LOW': con_near['Low'],
                         "BACK_MONTH_" + symbol + '_LOW': con_far['Low'],
                         "NEAR_MONTH_" + symbol + '_LAST': con_near['Last'],
                         "BACK_MONTH_" + symbol + '_LAST': con_far['Last'],
                         "NEAR_MONTH_" + symbol + '_VOLUME': con_near['Volume'],
                         "BACK_MONTH_" + symbol + '_VOLUME': con_far['Volume'],
                         "NEAR_MONTH_" + symbol + '_PREVIOUS_DAY_OPEN_INTEREST':
                             con_near['Previous Day Open Interest'],
                         "BACK_MONTH_" + symbol + '_PREVIOUS_DAY_OPEN_INTEREST':
                             con_far['Previous Day Open Interest']}, index=con_far.index)
    expiry_dates_dict = dict()
    FUT_MONTH_CODE_MAPPING = fut_expiry_code_mapping[symbol]
    for splice_year in year_range:
        loop_counter = 0
        for month_code in sorted(set(QUARTERLY_EXPIRY_MONTH_CODE_MAPPING.keys())):
            if loop_counter > num_expirations:
                break
            contract_str = cf.symbol + month_code + str(splice_year)
            func = fut_con_settle_func_mapping[symbol]
            ltd = func(FUT_MONTH_CODE_MAPPING[month_code], splice_year)
            liquidate_date = func(FUT_MONTH_CODE_MAPPING[month_code], splice_year, ib_liquidate=True)
            expiry_dates_dict[contract_str] = (liquidate_date, ltd)
            loop_counter += 1
    pd_series_expiry_dates = pd.Series(expiry_dates_dict).sort_values()
    merged_px_df = cf.merge_expirations(con_near.index[0],
                                        pd_series_expiry_dates,
                                        cons)
    weights = cf.eurodollar_price_roll(con_near.index[0],
                                       pd_series_expiry_dates,
                                       cons)
    weights.fillna(0, inplace=True)
    temp_df = weights.rename(columns={'settle_adj_factor': "MERGED_" + symbol + '_SETTLE',
                                      'open_adj_factor': "MERGED_" + symbol + '_OPEN',
                                      'high_adj_factor': "MERGED_" + symbol + '_HIGH',
                                      'low_adj_factor': "MERGED_" + symbol + '_LOW',
                                      'last_adj_factor': "MERGED_" + symbol + '_LAST',
                                      'volume_adj_factor': "MERGED_" + symbol + '_VOLUME',
                                      'previous_day_open_interest_adj_factor': "MERGED_" + symbol +
                                                                               '_PREVIOUS_DAY_OPEN_INTEREST'})
    con_cont_fut = temp_df + merged_px_df
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_SETTLE", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_SETTLE", axis=1, inplace=True)
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_OPEN", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_OPEN", axis=1, inplace=True)
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_HIGH", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_HIGH", axis=1, inplace=True)
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_LOW", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_LOW", axis=1, inplace=True)
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_LAST", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_LAST", axis=1, inplace=True)
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_VOLUME", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_VOLUME", axis=1, inplace=True)
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_PREVIOUS_DAY_OPEN_INTEREST", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_PREVIOUS_DAY_OPEN_INTEREST", axis=1, inplace=True)
    merged_df = pd.merge(left=cons, right=con_cont_fut[['MERGED_' + symbol + '_SETTLE',
                                                        'MERGED_' + symbol + '_OPEN',
                                                        'MERGED_' + symbol + '_HIGH',
                                                        'MERGED_' + symbol + '_LOW',
                                                        'MERGED_' + symbol + '_LAST',
                                                        'MERGED_' + symbol + '_VOLUME',
                                                        'MERGED_' + symbol + '_PREVIOUS_DAY_OPEN_INTEREST']],
                         left_index=True, right_index=True)
    merged_df = pd.merge(left=merged_df, right=temp_df[['MERGED_' + symbol + '_SETTLE',
                                                        'MERGED_' + symbol + '_OPEN',
                                                        'MERGED_' + symbol + '_HIGH',
                                                        'MERGED_' + symbol + '_LOW',
                                                        'MERGED_' + symbol + '_LAST',
                                                        'MERGED_' + symbol + '_VOLUME',
                                                        'MERGED_' + symbol + '_PREVIOUS_DAY_OPEN_INTEREST']],
                         left_index=True, right_index=True)
    merged_df = merged_df.rename(columns={'MERGED_' + symbol + '_SETTLE_x': "MERGED_" + symbol + '_SETTLE',
                                          'MERGED_' + symbol + '_OPEN_x': "MERGED_" + symbol + '_OPEN',
                                          'MERGED_' + symbol + '_HIGH_x': "MERGED_" + symbol + '_HIGH',
                                          'MERGED_' + symbol + '_LOW_x': "MERGED_" + symbol + '_LOW',
                                          'MERGED_' + symbol + '_LAST_x': "MERGED_" + symbol + '_LAST',
                                          'MERGED_' + symbol + '_VOLUME_x': "MERGED_" + symbol + '_VOLUME',
                                          'MERGED_' + symbol + '_PREVIOUS_DAY_OPEN_INTEREST_x':
                                              "MERGED_" + symbol + '_PREVIOUS_DAY_OPEN_INTEREST',
                                          'MERGED_' + symbol + '_SETTLE_y': "adj_factor_" + symbol + '_SETTLE',
                                          'MERGED_' + symbol + '_OPEN_y': "adj_factor_" + symbol + '_OPEN',
                                          'MERGED_' + symbol + '_HIGH_y': "adj_factor_" + symbol + '_HIGH',
                                          'MERGED_' + symbol + '_LOW_y': "adj_factor_" + symbol + '_LOW',
                                          'MERGED_' + symbol + '_LAST_y': "adj_factor_" + symbol + '_LAST',
                                          'MERGED_' + symbol + '_VOLUME_y': "adj_factor_" + symbol + '_VOLUME',
                                          'MERGED_' + symbol + '_PREVIOUS_DAY_OPEN_INTEREST_y':
                                              "adj_factor_" + symbol + '_PREVIOUS_DAY_OPEN_INTEREST'})
    return merged_df


def run_proportional_roll_method(symbol,
                                 quandl_symbol_near,
                                 quandl_symbol_far,
                                 start_date,
                                 end_date,
                                 year_range,
                                 num_expirations):
    # example: symbol for WTI Crude oil futures is CL
    cf = ContFut(symbol=symbol)
    # CHRIS/CME_CL1 = first expiration continuous for CL (crude oil wti)
    # CHRIS/CME_CL2 = second expiration continuous for CL (crude oil wti)
    # start_date and end_date are string format dates ie. 2019-12-31 & 2020-12-31
    LOGGER.info("cont_fut.run_proportional_roll_method(): running quandl_get on con_near %s between date %s "
                "and end date %s", quandl_symbol_near, start_date, end_date)
    try:
        con_near = quandl.get(quandl_symbol_near, start_date=start_date, end_date=end_date)
    except NotFoundError as nfe:
        LOGGER.error("cont_fut.run_proportional_roll_method(): error in quandl.get(%s) %s",
                     quandl_symbol_near, str(nfe.quandl_message))
    LOGGER.info("cont_fut.run_proportional_roll_method(): running quandl_get on con_far %s between date %s "
                "and end date %s", quandl_symbol_far, start_date, end_date)
    try:
        con_far = quandl.get(quandl_symbol_far, start_date=start_date, end_date=end_date)
    except NotFoundError as nfe:
        LOGGER.error("cont_fut.run_proportional_roll_method(): error in quandl.get(%s) %s",
                     quandl_symbol_far, str(nfe.quandl_message))
    con_near = column_name_check(con_near)
    con_far = column_name_check(con_far)
    con_near = con_near.where((pd.notnull(con_near)), None)
    con_near = handle_quandl_missing_data(symbol, con_near)
    con_far = con_far.where((pd.notnull(con_far)), None)
    con_far = handle_quandl_missing_data(symbol, con_far)
    cons = pd.DataFrame({"NEAR_MONTH_" + symbol + '_SETTLE': con_near['Settle'],
                         "BACK_MONTH_" + symbol + '_SETTLE': con_far['Settle'],
                         "NEAR_MONTH_" + symbol + '_OPEN': con_near['Open'],
                         "BACK_MONTH_" + symbol + '_OPEN': con_far['Open'],
                         "NEAR_MONTH_" + symbol + '_HIGH': con_near['High'],
                         "BACK_MONTH_" + symbol + '_HIGH': con_far['High'],
                         "NEAR_MONTH_" + symbol + '_LOW': con_near['Low'],
                         "BACK_MONTH_" + symbol + '_LOW': con_far['Low'],
                         "NEAR_MONTH_" + symbol + '_LAST': con_near['Last'],
                         "BACK_MONTH_" + symbol + '_LAST': con_far['Last'],
                         "NEAR_MONTH_" + symbol + '_VOLUME': con_near['Volume'],
                         "BACK_MONTH_" + symbol + '_VOLUME': con_far['Volume'],
                         "NEAR_MONTH_" + symbol + '_PREVIOUS_DAY_OPEN_INTEREST':
                             con_near['Previous Day Open Interest'],
                         "BACK_MONTH_" + symbol + '_PREVIOUS_DAY_OPEN_INTEREST':
                             con_far['Previous Day Open Interest']}, index=con_far.index)
    expiry_dates_dict = dict()
    loop_counter = 0
    FUT_MONTH_CODE_MAPPING = fut_expiry_code_mapping[symbol]
    for splice_year in year_range:
        for month_code in sorted(set(FUT_MONTH_CODE_MAPPING.keys())):
            if loop_counter > num_expirations:
                break
            contract_str = cf.symbol + month_code + str(splice_year)
            func = fut_con_settle_func_mapping[symbol]
            ltd = func(FUT_MONTH_CODE_MAPPING[month_code], splice_year, ib_liquidate=False)
            ib_liquidate_date = func(FUT_MONTH_CODE_MAPPING[month_code], splice_year, ib_liquidate=True)
            expiry_dates_dict[contract_str] = (ib_liquidate_date, ltd)
            loop_counter += 1
    pd_series_expiry_dates = pd.Series(expiry_dates_dict).sort_values()
    merged_px_df = cf.merge_expirations(con_near.index[0],
                                        pd_series_expiry_dates,
                                        cons)
    weights = cf.proportional_price_roll(con_near.index[0],
                                         pd_series_expiry_dates,
                                         cons)
    weights.fillna(0, inplace=True)
    temp_df = weights.rename(columns={'settle_adj_factor': "MERGED_" + symbol + '_SETTLE',
                                      'open_adj_factor': "MERGED_" + symbol + '_OPEN',
                                      'high_adj_factor': "MERGED_" + symbol + '_HIGH',
                                      'low_adj_factor': "MERGED_" + symbol + '_LOW',
                                      'last_adj_factor': "MERGED_" + symbol + '_LAST',
                                      'volume_adj_factor': "MERGED_" + symbol + '_VOLUME',
                                      'previous_day_open_interest_adj_factor': "MERGED_" + symbol +
                                                                               '_PREVIOUS_DAY_OPEN_INTEREST'})
    con_cont_fut = temp_df * merged_px_df
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_SETTLE", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_SETTLE", axis=1, inplace=True)
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_OPEN", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_OPEN", axis=1, inplace=True)
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_HIGH", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_HIGH", axis=1, inplace=True)
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_LOW", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_LOW", axis=1, inplace=True)
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_LAST", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_LAST", axis=1, inplace=True)
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_VOLUME", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_VOLUME", axis=1, inplace=True)
    con_cont_fut.drop(columns="BACK_MONTH_" + symbol + "_PREVIOUS_DAY_OPEN_INTEREST", axis=1, inplace=True)
    con_cont_fut.drop(columns="NEAR_MONTH_" + symbol + "_PREVIOUS_DAY_OPEN_INTEREST", axis=1, inplace=True)
    merged_df = pd.merge(left=cons, right=con_cont_fut[['MERGED_' + symbol + '_SETTLE',
                                                        'MERGED_' + symbol + '_OPEN',
                                                        'MERGED_' + symbol + '_HIGH',
                                                        'MERGED_' + symbol + '_LOW',
                                                        'MERGED_' + symbol + '_LAST',
                                                        'MERGED_' + symbol + '_VOLUME',
                                                        'MERGED_' + symbol + '_PREVIOUS_DAY_OPEN_INTEREST']],
                         left_index=True, right_index=True)
    merged_df = pd.merge(left=merged_df, right=temp_df[['MERGED_' + symbol + '_SETTLE',
                                                        'MERGED_' + symbol + '_OPEN',
                                                        'MERGED_' + symbol + '_HIGH',
                                                        'MERGED_' + symbol + '_LOW',
                                                        'MERGED_' + symbol + '_LAST',
                                                        'MERGED_' + symbol + '_VOLUME',
                                                        'MERGED_' + symbol + '_PREVIOUS_DAY_OPEN_INTEREST']],
                         left_index=True, right_index=True)
    merged_df = merged_df.rename(columns={'MERGED_' + symbol + '_SETTLE_x': "MERGED_" + symbol + '_SETTLE',
                                          'MERGED_' + symbol + '_OPEN_x': "MERGED_" + symbol + '_OPEN',
                                          'MERGED_' + symbol + '_HIGH_x': "MERGED_" + symbol + '_HIGH',
                                          'MERGED_' + symbol + '_LOW_x': "MERGED_" + symbol + '_LOW',
                                          'MERGED_' + symbol + '_LAST_x': "MERGED_" + symbol + '_LAST',
                                          'MERGED_' + symbol + '_VOLUME_x': "MERGED_" + symbol + '_VOLUME',
                                          'MERGED_' + symbol + '_PREVIOUS_DAY_OPEN_INTEREST_x':
                                              "MERGED_" + symbol + '_PREVIOUS_DAY_OPEN_INTEREST',
                                          'MERGED_' + symbol + '_SETTLE_y': "adj_factor_" + symbol + '_SETTLE',
                                          'MERGED_' + symbol + '_OPEN_y': "adj_factor_" + symbol + '_OPEN',
                                          'MERGED_' + symbol + '_HIGH_y': "adj_factor_" + symbol + '_HIGH',
                                          'MERGED_' + symbol + '_LOW_y': "adj_factor_" + symbol + '_LOW',
                                          'MERGED_' + symbol + '_LAST_y': "adj_factor_" + symbol + '_LAST',
                                          'MERGED_' + symbol + '_VOLUME_y': "adj_factor_" + symbol + '_VOLUME',
                                          'MERGED_' + symbol + '_PREVIOUS_DAY_OPEN_INTEREST_y':
                                              "adj_factor_" + symbol + '_PREVIOUS_DAY_OPEN_INTEREST'})
    return merged_df


def run_weighted_roll_method(symbol,
                             quandl_symbol_near,
                             quandl_symbol_far,
                             start_date,
                             end_date,
                             splice_year,
                             num_expirations):
    # example: symbol for WTI Crude oil futures is CL
    cf = ContFut(symbol=symbol)
    # CHRIS/CME_CL1 = first expiration continuous for CL (crude oil wti)
    # CHRIS/CME_CL2 = second expiration continuous for CL (crude oil wti)
    # start_date and end_date are string format dates ie. 2019-12-31 & 2020-12-31
    con_near = quandl.get(quandl_symbol_near, start_date=start_date, end_date=end_date)
    con_far = quandl.get(quandl_symbol_far, start_date="2019-12-31", end_date="2020-12-31")
    con_near = column_name_check(df=con_near)
    con_far = column_name_check(df=con_far)
    cons = pd.DataFrame({"NEAR_MONTH_" + symbol: con_near['Settle'],
                         "BACK_MONTH_" + symbol: con_far['Settle']}, index=con_far.index)
    expiry_dates_dict = dict()
    FUT_MONTH_CODE_MAPPING = fut_expiry_code_mapping[symbol]
    loop_counter = 0
    for month_code in sorted(set(FUT_MONTH_CODE_MAPPING.keys())):
        if loop_counter > num_expirations:
            break
        contract_str = cf.symbol + month_code + str(splice_year)
        func = fut_con_settle_func_mapping[symbol]
        ltd = func(FUT_MONTH_CODE_MAPPING[month_code], splice_year)
        expiry_dates_dict[contract_str] = ltd
        loop_counter += 1
    pd_series_expiry_dates = pd.Series(expiry_dates_dict).sort_values()
    weights = cf.futures_rollover_weights(con_near.index[0], pd_series_expiry_dates, cons)
    # weights = cf.proportional_price_roll(con_near.index[0], pd_series_expiry_dates, cons)
    # weights.drop(cons.columns, axis=1, inplace=True)
    weights.fillna(0, inplace=True)
    new_col_names = list(expiry_dates_dict.keys())
    new_col_names.append("NEAR_MONTH_" + symbol)
    new_col_names.append("BACK_MONTH_" + symbol)
    weights.rename(columns=dict(zip(list(weights.columns), new_col_names)), inplace=True)
    # construct the continuous futures of the WTI CL contracts
    con_cont_fut = (cons * weights).sum(1).dropna()
    return con_cont_fut


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


def get_engine():
    mysql_server_ip = SecureKeysAccess.get_mysql_server_ip()
    mysql_username = SecureKeysAccess.get_mysql_server_user()
    mysql_secret = SecureKeysAccess.get_mysql_server_secret()
    LOGGER.info("nominal_yield_curve.get_db_session(): "
                "creating db engine with pymysql, username %s, secret stays secret, and server_ip %s",
                mysql_username, mysql_server_ip)
    engine = create_engine('mysql+pymysql://' + mysql_username + ':' + mysql_secret + '@' +
                           mysql_server_ip + '/stealbasistrader')
    return engine


def currency_futs_to_db(df):
    return 1


def merge_vx_futs_dataframes(list_of_dfs):
    total_df = pd.DataFrame(index=list_of_dfs[0].index)
    for df in list_of_dfs:
        if total_df.empty:
            total_df = df
        else:
            total_df = pd.merge(left=total_df, right=df, how='inner', left_index=True, right_index=True)
    total_filename = write_to_csv(df=total_df, symbol='VX_CONT_ALL', start_date='2013-01-01',
                                  end_date=str(datetime.now().date()), force_new_file=True)
    return total_df, total_filename


def vx_curve_to_db(df):
    curve_entries = df.apply(vx_curve_to_db_vectorized, axis=1)
    session = get_db_session()
    session.flush()
    for c_entry in curve_entries.iteritems():
        LOGGER.info("cont_fut.vx_curve_to_db():inserting yield curve entry %s %s", c_entry[0], c_entry[1])
        c_entry_db = session.query(VixCurve).filter_by(Id=c_entry[1].Id).first()
        if c_entry_db is None:
            LOGGER.info("cont_fut.vx_curve_to_db(): entries for date %s do not exist, inserting...",
                        c_entry[1].NEW_DATE)
            session.add(c_entry[1])
        else:
            LOGGER.info("cont_fut.vx_curve_to_db(): entries for date %s do exist, updating...",
                        c_entry[1].NEW_DATE)
            if not c_entry_db.equals(c_entry[1]):
                LOGGER.error("cont_fut.vx_curve_to_db(): possible issue, "
                             "older entry in MySQL has a different value than overlap entries for daily "
                             "VX insert, but continuing with update!")
                c_entry_db.set_all(c_entry[1])
            else:
                c_entry_db.set_all(c_entry[1])
    session.commit()


def vx_curve_to_db_vectorized(row):
    default_vx_px = 1000.0
    identifier = row.name.to_pydatetime()
    # VX1 - settle price
    try:
        if np.isnan(row.loc['Settle.0.Cont']):
            LOGGER.error("cont_fut.vx_curve_to_db_vectorized():Settle.0.cont is NAN! %s", str(row.loc['Settle.0.Cont']))
            LOGGER.info("cont_fut.vx_curve_to_db_vectorized(): Settle.0.cont is NAN, setting value to "
                        "default_vx_price %s", str(default_vx_px))
            merged_vx1_settle = default_vx_px
        else:
            merged_vx1_settle = float(row.loc['Settle.0.Cont'])
    except KeyError:
        merged_vx1_settle = default_vx_px
    # VX2 - settle price
    try:
        if np.isnan(row.loc['Settle.1.Cont']):
            LOGGER.error("cont_fut.vx_curve_to_db_vectorized():Settle.1.cont is NAN! %s", str(row.loc['Settle.1.Cont']))
            LOGGER.info("cont_fut.vx_curve_to_db_vectorized(): Settle.1.cont is NAN, setting value to "
                        "default_vx_price %s", str(default_vx_px))
            merged_vx2_settle = default_vx_px
        else:
            merged_vx2_settle = float(row.loc['Settle.1.Cont'])
    except KeyError:
        merged_vx2_settle = default_vx_px
    # VX3 - settle price
    try:
        if np.isnan(row.loc['Settle.2.Cont']):
            LOGGER.error("cont_fut.vx_curve_to_db_vectorized():Settle.2.cont is NAN! %s", str(row.loc['Settle.2.Cont']))
            LOGGER.info("cont_fut.vx_curve_to_db_vectorized(): Settle.2.cont is NAN, setting value to "
                        "default_vx_price %s", str(default_vx_px))
            merged_vx3_settle = default_vx_px
        else:
            merged_vx3_settle = float(row.loc['Settle.2.Cont'])
    except KeyError:
        merged_vx3_settle = default_vx_px
    # VX4 - settle price
    try:
        if np.isnan(row.loc['Settle.3.Cont']):
            LOGGER.error("cont_fut.vx_curve_to_db_vectorized():Settle.3.cont is NAN! %s", str(row.loc['Settle.3.Cont']))
            LOGGER.info("cont_fut.vx_curve_to_db_vectorized(): Settle.3.cont is NAN, setting value to "
                        "default_vx_price %s", str(default_vx_px))
            merged_vx4_settle = default_vx_px
        else:
            merged_vx4_settle = float(row.loc['Settle.3.Cont'])
    except KeyError:
        merged_vx4_settle = default_vx_px
    # VX5 - settle price
    try:
        if np.isnan(row.loc['Settle.4.Cont']):
            LOGGER.error("cont_fut.vx_curve_to_db_vectorized():Settle.4.cont is NAN! %s", str(row.loc['Settle.4.Cont']))
            LOGGER.info("cont_fut.vx_curve_to_db_vectorized(): Settle.4.cont is NAN, setting value to "
                        "default_vx_price %s", str(default_vx_px))
            merged_vx5_settle = default_vx_px
        else:
            merged_vx5_settle = float(row.loc['Settle.4.Cont'])
    except KeyError:
        merged_vx5_settle = default_vx_px
    # VX6 - settle price
    try:
        if np.isnan(row.loc['Settle.5.Cont']):
            LOGGER.error("cont_fut.vx_curve_to_db_vectorized():Settle.5.cont is NAN! %s", str(row.loc['Settle.5.Cont']))
            LOGGER.info("cont_fut.vx_curve_to_db_vectorized(): Settle.5.cont is NAN, setting value to "
                        "default_vx_price %s", str(default_vx_px))
            merged_vx6_settle = default_vx_px
        else:
            merged_vx6_settle = float(row.loc['Settle.5.Cont'])
    except KeyError:
        merged_vx6_settle = default_vx_px
    # VX7 - settle price
    try:
        if np.isnan(row.loc['Settle.6.Cont']):
            LOGGER.error("cont_fut.vx_curve_to_db_vectorized():Settle.6.cont is NAN! %s", str(row.loc['Settle.6.Cont']))
            LOGGER.info("cont_fut.vx_curve_to_db_vectorized(): Settle.6.cont is NAN, setting value to "
                        "default_vx_price %s", str(default_vx_px))
            merged_vx7_settle = default_vx_px
        else:
            merged_vx7_settle = float(row.loc['Settle.6.Cont'])
    except KeyError:
        merged_vx7_settle = default_vx_px
    # NEW_DATE - date of yield curve
    new_date = row.name.to_pydatetime()
    vxc_entry = VixCurve(Id=identifier,
                         VX1=merged_vx1_settle,
                         VX2=merged_vx2_settle,
                         VX3=merged_vx3_settle,
                         VX4=merged_vx4_settle,
                         VX5=merged_vx5_settle,
                         VX6=merged_vx6_settle,
                         VX7=merged_vx7_settle,
                         NEW_DATE=new_date)
    return vxc_entry


def cl_curve_to_db(df):
    curve_entries = df.apply(cl_curve_to_db_vectorized, axis=1)
    session = get_db_session()
    session.flush()
    for c_entry in curve_entries.iteritems():
        LOGGER.info("cont_fut.cl_curve_to_db():inserting yield curve entry %s %s", c_entry[0], c_entry[1])
        c_entry_db = session.query(WTICrudeOilCurve).filter_by(Id=c_entry[1].Id).first()
        if c_entry_db is None:
            LOGGER.info("cont_fut.cl_curve_to_db(): entries for date %s do not exist, inserting...",
                        c_entry[1].NEW_DATE)
            session.add(c_entry[1])
        else:
            LOGGER.info("cont_fut.cl_curve_to_db(): entries for date %s do exist, updating...",
                        c_entry[1].NEW_DATE)
            if not c_entry_db.equals(c_entry[1]):
                LOGGER.error("cont_fut.cl_curve_to_db(): possible issue, "
                             "older entry in MySQL has a different value than overlap entries for daily "
                             "WTI CL insert, but continuing with update!")
                c_entry_db.set_all(c_entry[1])
            else:
                c_entry_db.set_all(c_entry[1])
    session.commit()


def cl_curve_to_db_vectorized(row):
    """
        cl_curve_to_db(df) takes a dataframe that contains yield curve entry for
        one date. This is a VECTORIZED function. The input is a row of a dataframe.
        """
    default_cl_px = 1000.0
    identifier = row.name.to_pydatetime()
    # Id
    # identifier = df.loc['d:Id']['$']
    # ED1 - settle price
    merged_cl1_settle = float(row.loc['MERGED_CL1_SETTLE'])
    # ED2 - settle price
    merged_cl2_settle = float(row.loc['MERGED_CL2_SETTLE'])
    # ED3 - settle price
    try:
        merged_cl3_settle = float(row.loc['MERGED_CL3_SETTLE'])
    except KeyError:
        merged_cl3_settle = default_cl_px
    # ED4 - settle price
    try:
        merged_cl4_settle = float(row.loc['MERGED_CL4_SETTLE'])
    except KeyError:
        merged_cl4_settle = default_cl_px
    # ED5 - settle price
    try:
        merged_cl5_settle = float(row.loc['MERGED_CL5_SETTLE'])
    except KeyError:
        merged_cl5_settle = default_cl_px
    # ED6 - settle price
    try:
        merged_cl6_settle = float(row.loc['MERGED_CL6_SETTLE'])
    except KeyError:
        merged_cl6_settle = default_cl_px
    # ED7 - settle price
    try:
        merged_cl7_settle = float(row.loc['MERGED_CL7_SETTLE'])
    except KeyError:
        merged_cl7_settle = default_cl_px
    # ED8 - settle price
    try:
        merged_cl8_settle = float(row.loc['MERGED_CL8_SETTLE'])
    except KeyError:
        merged_cl8_settle = default_cl_px
    # ED9 - settle price
    try:
        merged_cl9_settle = float(row.loc['MERGED_CL9_SETTLE'])
    except KeyError:
        merged_cl9_settle = default_cl_px
    # ED10 - settle price
    try:
        merged_cl10_settle = float(row.loc['MERGED_CL10_SETTLE'])
    except KeyError:
        merged_cl10_settle = default_cl_px
    # ED11 - settle price
    try:
        merged_cl11_settle = float(row.loc['MERGED_CL11_SETTLE'])
    except KeyError:
        merged_cl11_settle = default_cl_px
    # ED12 - settle price
    try:
        merged_cl12_settle = float(row.loc['MERGED_CL12_SETTLE'])
    except KeyError:
        merged_cl12_settle = default_cl_px
    # ED13 - settle price
    try:
        merged_cl13_settle = float(row.loc['MERGED_CL13_SETTLE'])
    except KeyError:
        merged_cl13_settle = default_cl_px
    # ED14 - settle price
    try:
        merged_cl14_settle = float(row.loc['MERGED_CL14_SETTLE'])
    except KeyError:
        merged_cl14_settle = default_cl_px
    # ED15 - settle price
    try:
        merged_cl15_settle = float(row.loc['MERGED_CL15_SETTLE'])
    except KeyError:
        merged_cl15_settle = default_cl_px
    # ED16 - settle price
    try:
        merged_cl16_settle = float(row.loc['MERGED_CL16_SETTLE'])
    except KeyError:
        merged_cl16_settle = default_cl_px
    # ED17 - settle price
    try:
        merged_cl17_settle = float(row.loc['MERGED_CL17_SETTLE'])
    except KeyError:
        merged_cl17_settle = default_cl_px
    # ED18 - settle price
    try:
        merged_cl18_settle = float(row.loc['MERGED_CL18_SETTLE'])
    except KeyError:
        merged_cl18_settle = default_cl_px
    # ED19 - settle price
    try:
        merged_cl19_settle = float(row.loc['MERGED_CL19_SETTLE'])
    except KeyError:
        merged_cl19_settle = default_cl_px
    # ED20 - settle price
    try:
        merged_cl20_settle = float(row.loc['MERGED_CL20_SETTLE'])
    except KeyError:
        merged_cl20_settle = default_cl_px
    # ED21 - settle price
    try:
        merged_cl21_settle = float(row.loc['MERGED_CL21_SETTLE'])
    except KeyError:
        merged_cl21_settle = default_cl_px
    # ED22 - settle price
    try:
        merged_cl22_settle = float(row.loc['MERGED_CL22_SETTLE'])
    except KeyError:
        merged_cl22_settle = default_cl_px
    # ED23 - settle price
    try:
        merged_cl23_settle = float(row.loc['MERGED_CL23_SETTLE'])
    except KeyError:
        merged_cl23_settle = default_cl_px
    # ED24 - settle price
    try:
        merged_cl24_settle = float(row.loc['MERGED_CL24_SETTLE'])
    except KeyError:
        merged_cl24_settle = default_cl_px
    try:
        merged_cl25_settle = float(row.loc['MERGED_CL25_SETTLE'])
    except KeyError:
        merged_cl25_settle = default_cl_px
    try:
        merged_cl26_settle = float(row.loc['MERGED_CL26_SETTLE'])
    except KeyError:
        merged_cl26_settle = default_cl_px
    try:
        merged_cl27_settle = float(row.loc['MERGED_CL27_SETTLE'])
    except KeyError:
        merged_cl27_settle = default_cl_px
    try:
        merged_cl28_settle = float(row.loc['MERGED_CL28_SETTLE'])
    except KeyError:
        merged_cl28_settle = default_cl_px
    try:
        merged_cl29_settle = float(row.loc['MERGED_CL29_SETTLE'])
    except KeyError:
        merged_cl29_settle = default_cl_px
    try:
        merged_cl30_settle = float(row.loc['MERGED_CL30_SETTLE'])
    except KeyError:
        merged_cl30_settle = default_cl_px
    try:
        merged_cl31_settle = float(row.loc['MERGED_CL31_SETTLE'])
    except KeyError:
        merged_cl31_settle = default_cl_px
    try:
        merged_cl32_settle = float(row.loc['MERGED_CL32_SETTLE'])
    except KeyError:
        merged_cl32_settle = default_cl_px
    try:
        merged_cl33_settle = float(row.loc['MERGED_CL33_SETTLE'])
    except KeyError:
        merged_cl33_settle = default_cl_px
    try:
        merged_cl34_settle = float(row.loc['MERGED_CL34_SETTLE'])
    except KeyError:
        merged_cl34_settle = default_cl_px
    try:
        merged_cl35_settle = float(row.loc['MERGED_CL35_SETTLE'])
    except KeyError:
        merged_cl35_settle = default_cl_px
    try:
        merged_cl36_settle = float(row.loc['MERGED_CL36_SETTLE'])
    except KeyError:
        merged_cl36_settle = default_cl_px
    try:
        merged_cl37_settle = float(row.loc['MERGED_CL37_SETTLE'])
    except KeyError:
        merged_cl37_settle = default_cl_px
    try:
        merged_cl38_settle = float(row.loc['MERGED_CL38_SETTLE'])
    except KeyError:
        merged_cl38_settle = default_cl_px
    try:
        merged_cl39_settle = float(row.loc['MERGED_CL39_SETTLE'])
    except KeyError:
        merged_cl39_settle = default_cl_px
    # NEW_DATE - date of yield curve
    new_date = row.name.to_pydatetime()

    yc_entry = WTICrudeOilCurve(Id=identifier,
                                CL1=merged_cl1_settle,
                                CL2=merged_cl2_settle,
                                CL3=merged_cl3_settle,
                                CL4=merged_cl4_settle,
                                CL5=merged_cl5_settle,
                                CL6=merged_cl6_settle,
                                CL7=merged_cl7_settle,
                                CL8=merged_cl8_settle,
                                CL9=merged_cl9_settle,
                                CL10=merged_cl10_settle,
                                CL11=merged_cl11_settle,
                                CL12=merged_cl12_settle,
                                CL13=merged_cl13_settle,
                                CL14=merged_cl14_settle,
                                CL15=merged_cl15_settle,
                                CL16=merged_cl16_settle,
                                CL17=merged_cl17_settle,
                                CL18=merged_cl18_settle,
                                CL19=merged_cl19_settle,
                                CL20=merged_cl20_settle,
                                CL21=merged_cl21_settle,
                                CL22=merged_cl22_settle,
                                CL23=merged_cl23_settle,
                                CL24=merged_cl24_settle,
                                CL25=merged_cl25_settle,
                                CL26=merged_cl26_settle,
                                CL27=merged_cl27_settle,
                                CL28=merged_cl28_settle,
                                CL29=merged_cl29_settle,
                                CL30=merged_cl30_settle,
                                CL31=merged_cl31_settle,
                                CL32=merged_cl32_settle,
                                CL33=merged_cl33_settle,
                                CL34=merged_cl34_settle,
                                CL35=merged_cl35_settle,
                                CL36=merged_cl36_settle,
                                CL37=merged_cl37_settle,
                                CL38=merged_cl38_settle,
                                CL39=merged_cl39_settle,
                                NEW_DATE=new_date)
    return yc_entry


def currency_futures_px_to_db(df,
                              roll_type,
                              isDaily=False):
    entries = df.apply(equity_futures_to_db_vectorized, roll_type=roll_type, asset_type="CURRENCY", axis=1)
    session = get_db_session()
    session.flush()
    for entry in entries.iteritems():
        entry_db = session.query(CurrencyFutures).filter_by(Id=entry[1].Id). \
            filter_by(symbol=entry[1].symbol).filter_by(roll_type=entry[1].roll_type).first()
        if entry_db is None:
            LOGGER.info("cont_fut.currency_futures_px_to_db(): adding entry date to DB %s", str(entry[1].Id))
            session.add(entry[1])
        else:
            if isDaily:
                continue
            if not entry_db.equals(entry[1]):
                LOGGER.error("cont_fut.currency_futures_px_to_db(): possible issue, older entry in MySQL has a "
                             "different value than overlap entries for daily CurrencyFutures insert, "
                             "but continuing with update!")
                entry_db.set_all(entry[1])
            else:
                LOGGER.info("cont_fut.currency_futures_px_to_db(): entry in db is equal to entry from source...")
                entry_db.set_all(entry[1])
    session.commit()


def equity_futures_px_to_db(df,
                            roll_type):
    # when the open price is not available, we need to plug in the previous last price for the open price
    # we need to do something, we can't leave it as NONE.
    entries = df.apply(equity_futures_to_db_vectorized, roll_type=roll_type, axis=1)
    session = get_db_session()
    session.flush()
    for entry in entries.iteritems():
        entry_db = session.query(EquityFutures).filter_by(Id=entry[1].Id). \
            filter_by(symbol=entry[1].symbol).filter_by(roll_type=entry[1].roll_type).first()
        if entry_db is None:
            session.add(entry[1])
        else:
            if not entry_db.equals(entry[1]):
                # LOGGER.error("cont_fut.equity_futures_px_to_db(): possible issue, older entry in MySQL has a "
                #             "different value than overlap entries for daily EquityFutures insert, "
                #             "but continuing with insert!")
                # LOGGER.error("cont_fut.equity_futures_px_to_db(): db entry is %s", entry_db.__repr__())
                # LOGGER.error("cont_fut.equity_futures_px_to_db(): replace db entry with %s", entry[1].__repr__())
                entry_db.set_all(entry[1])
            else:
                entry_db.set_all(entry[1])
    session.commit()


def col_header_filter(search_pattern, index_list):
    return [col_header for col_header in index_list if (set(search_pattern.split("_")).issubset(col_header.split("_")))]


def equity_futures_to_db_vectorized(row, roll_type, asset_type="EQUITIES"):
    """
    equity_futures_curve_to_db_vectorized(df) takes a row series that contains curve entry for
    one date. This is a VECTORIZED function. The input is a row of a dataframe.
    """
    identifier = row.name.to_pydatetime()
    near_month_settle_extract = col_header_filter("NEAR_MONTH_SETTLE", row.index)[0]
    symbol = near_month_settle_extract.split("_")[2]
    near_month_settle = float(row.loc[near_month_settle_extract])
    back_month_settle = float(row.loc[col_header_filter('BACK_MONTH_SETTLE', row.index)[0]])
    near_month_open = float(row.loc[col_header_filter('NEAR_MONTH_OPEN', row.index)[0]])
    back_month_open = float(row.loc[col_header_filter('BACK_MONTH_OPEN', row.index)[0]])
    near_month_high = float(row.loc[col_header_filter('NEAR_MONTH_HIGH', row.index)[0]])
    back_month_high = float(row.loc[col_header_filter('BACK_MONTH_HIGH', row.index)[0]])
    near_month_low = float(row.loc[col_header_filter('NEAR_MONTH_LOW', row.index)[0]])
    back_month_low = float(row.loc[col_header_filter('BACK_MONTH_LOW', row.index)[0]])
    near_month_last = float(row.loc[col_header_filter('NEAR_MONTH_LAST', row.index)[0]])
    back_month_last = float(row.loc[col_header_filter('BACK_MONTH_LAST', row.index)[0]])
    near_month_volume = float(row.loc[col_header_filter('NEAR_MONTH_VOLUME', row.index)[0]])
    back_month_volume = float(row.loc[col_header_filter('BACK_MONTH_VOLUME', row.index)[0]])
    near_month_open_interest = float(row.loc[col_header_filter('NEAR_MONTH_PREVIOUS_DAY_OPEN_INTEREST', row.index)[0]])
    back_month_open_interest = float(row.loc[col_header_filter('BACK_MONTH_PREVIOUS_DAY_OPEN_INTEREST', row.index)[0]])
    merged_settle = float(row.loc[col_header_filter('MERGED_SETTLE', row.index)[0]])
    merged_open = float(row.loc[col_header_filter('MERGED_OPEN', row.index)[0]])
    merged_high = float(row.loc[col_header_filter('MERGED_HIGH', row.index)[0]])
    merged_low = float(row.loc[col_header_filter('MERGED_LOW', row.index)[0]])
    merged_last = float(row.loc[col_header_filter('MERGED_LAST', row.index)[0]])
    merged_volume = float(row.loc[col_header_filter('MERGED_VOLUME', row.index)[0]])
    merged_previous_day_open_interest = float(row.loc[col_header_filter('MERGED_PREVIOUS_DAY_OPEN_INTEREST',
                                                                        row.index)[0]])
    adj_factor_settle = float(row.loc[col_header_filter('adj_factor_SETTLE', row.index)[0]])
    adj_factor_open = float(row.loc[col_header_filter('adj_factor_OPEN', row.index)[0]])
    adj_factor_high = float(row.loc[col_header_filter('adj_factor_HIGH', row.index)[0]])
    adj_factor_low = float(row.loc[col_header_filter('adj_factor_LOW', row.index)[0]])
    adj_factor_last = float(row.loc[col_header_filter('adj_factor_LAST', row.index)[0]])
    adj_factor_volume = float(row.loc[col_header_filter('adj_factor_VOLUME', row.index)[0]])
    adj_factor_previous_day_open_interest = float(row.loc[col_header_filter('adj_factor_PREVIOUS_DAY_OPEN_INTEREST',
                                                                            row.index)[0]])
    new_date = row.name.to_pydatetime()
    if asset_type == "EQUITIES":
        entry = EquityFutures(Id=identifier,
                              symbol=symbol,
                              roll_type=roll_type,
                              near_month_settle=near_month_settle,
                              back_month_settle=back_month_settle,
                              near_month_open=near_month_open,
                              back_month_open=back_month_open,
                              near_month_high=near_month_high,
                              back_month_high=back_month_high,
                              near_month_low=near_month_low,
                              back_month_low=back_month_low,
                              near_month_last=near_month_last,
                              back_month_last=back_month_last,
                              near_month_volume=near_month_volume,
                              back_month_volume=back_month_volume,
                              near_month_open_interest=near_month_open_interest,
                              back_month_open_interest=back_month_open_interest,
                              merged_settle=merged_settle,
                              merged_open=merged_open,
                              merged_high=merged_high,
                              merged_low=merged_low,
                              merged_last=merged_last,
                              merged_volume=merged_volume,
                              merged_previous_day_open_interest=merged_previous_day_open_interest,
                              adj_factor_settle=adj_factor_settle,
                              adj_factor_open=adj_factor_open,
                              adj_factor_high=adj_factor_high,
                              adj_factor_low=adj_factor_low,
                              adj_factor_last=adj_factor_last,
                              adj_factor_volume=adj_factor_volume,
                              adj_factor_previous_day_open_interest=adj_factor_previous_day_open_interest,
                              NEW_DATE=new_date)
    elif asset_type == "CURRENCY":
        try:
            spot_mkt_obs_value = float(row.loc[col_header_filter('SPOT_MKT_OBS_VALUE', row.index)[0]])
            spot_mkt_value_desc = row.loc[col_header_filter('SPOT_MKT_VALUE_DESC', row.index)[0]]
        except KeyError as ke:
            LOGGER.error("cont_fut.equity_futures_to_db_vectorized(): "
                         "attempting to retrieve spot mkt currency values from a dataframe that does not have them %s!",
                         ke.message)
        entry = CurrencyFutures(Id=identifier,
                                symbol=symbol,
                                roll_type=roll_type,
                                near_month_settle=near_month_settle,
                                back_month_settle=back_month_settle,
                                near_month_open=near_month_open,
                                back_month_open=back_month_open,
                                near_month_high=near_month_high,
                                back_month_high=back_month_high,
                                near_month_low=near_month_low,
                                back_month_low=back_month_low,
                                near_month_last=near_month_last,
                                back_month_last=back_month_last,
                                near_month_volume=near_month_volume,
                                back_month_volume=back_month_volume,
                                near_month_open_interest=near_month_open_interest,
                                back_month_open_interest=back_month_open_interest,
                                merged_settle=merged_settle,
                                merged_open=merged_open,
                                merged_high=merged_high,
                                merged_low=merged_low,
                                merged_last=merged_last,
                                merged_volume=merged_volume,
                                merged_previous_day_open_interest=merged_previous_day_open_interest,
                                adj_factor_settle=adj_factor_settle,
                                adj_factor_open=adj_factor_open,
                                adj_factor_high=adj_factor_high,
                                adj_factor_low=adj_factor_low,
                                adj_factor_last=adj_factor_last,
                                adj_factor_volume=adj_factor_volume,
                                adj_factor_previous_day_open_interest=adj_factor_previous_day_open_interest,
                                NEW_DATE=new_date,
                                SPOT_MKT_OBS_VALUE=spot_mkt_obs_value,
                                SPOT_MKT_VALUE_DESC=spot_mkt_value_desc)
    else:
        LOGGER.error("cont_fut.equity_futures_to_db_vectorized(): "
                     "invalid asset_type %s, must be one of CURRENCIES/EQUITIES", asset_type)
        return None
    return entry


def ed_curve_to_db(df):
    from sqlalchemy.exc import OperationalError
    yc_entries = df.apply(ed_curve_to_db_vectorized, axis=1)
    session = get_db_session()
    session.flush()
    for yc_entry in yc_entries.iteritems():
        LOGGER.info("cont_fut.ed_curve_to_db():inserting yield curve entry %s %s", yc_entry[0], yc_entry[1])
        try:
            yc_entry_db = session.query(EurodollarYieldCurve).filter_by(Id=yc_entry[1].Id).first()
        except OperationalError as op_error:
            LOGGER.error("cont_fut.ed_curve_to_db(): error in query, message is %s", op_error.__str__())
            LOGGER.error("cont_fut.ed_curve_to_db(): trying query again...")
            yc_entry_db = session.query(EurodollarYieldCurve).filter_by(Id=yc_entry[1].Id).first()
        if yc_entry_db is None:
            LOGGER.info("cont_fut.ed_curve_to_db(): entries for date %s do not exist, inserting...",
                        yc_entry[1].NEW_DATE)
            session.add(yc_entry[1])
        else:
            LOGGER.info("cont_fut.ed_curve_to_db(): entries for date %s do exist, updating...",
                        yc_entry[1].NEW_DATE)
            if not yc_entry_db.equals(yc_entry[1]):
                LOGGER.error("cont_fut.ed_curve_to_db(): possible issue, "
                             "older entry in MySQL has a different value than overlap entries for daily "
                             "EuroDollar insert, but continuing with update!")
                LOGGER.error("cont_fut.ed_curve_to_db(): db entry is %s", yc_entry_db.__repr__())
                LOGGER.error("cont_fut.ed_curve_to_db(): replace db entry with %s", yc_entry[1].__repr__())
                yc_entry_db.set_all(yc_entry[1])
            else:
                yc_entry_db.set_all(yc_entry[1])
    session.commit()


def ed_curve_to_db_vectorized(row):
    """
        ed_curve_to_db(df) takes a dataframe that contains yield curve entry for
        one date. This is a VECTORIZED function. The input is a row of a dataframe.
        """
    default_ed_px = 50.0
    identifier = row.name.to_pydatetime()
    # Id
    # identifier = df.loc['d:Id']['$']
    # ED1 - settle price
    merged_ed1_settle = float(row.loc['MERGED_ED1_SETTLE'])
    # ED2 - settle price
    merged_ed2_settle = float(row.loc['MERGED_ED2_SETTLE'])
    # ED3 - settle price
    try:
        merged_ed3_settle = float(row.loc['MERGED_ED3_SETTLE'])
    except KeyError:
        merged_ed3_settle = default_ed_px
    # ED4 - settle price
    try:
        merged_ed4_settle = float(row.loc['MERGED_ED4_SETTLE'])
    except KeyError:
        merged_ed4_settle = default_ed_px
    # ED5 - settle price
    try:
        merged_ed5_settle = float(row.loc['MERGED_ED5_SETTLE'])
    except KeyError:
        merged_ed5_settle = default_ed_px
    # ED6 - settle price
    try:
        merged_ed6_settle = float(row.loc['MERGED_ED6_SETTLE'])
    except KeyError:
        merged_ed6_settle = default_ed_px
    # ED7 - settle price
    try:
        merged_ed7_settle = float(row.loc['MERGED_ED7_SETTLE'])
    except KeyError:
        merged_ed7_settle = default_ed_px
    # ED8 - settle price
    try:
        merged_ed8_settle = float(row.loc['MERGED_ED8_SETTLE'])
    except KeyError:
        merged_ed8_settle = default_ed_px
    # ED9 - settle price
    try:
        merged_ed9_settle = float(row.loc['MERGED_ED9_SETTLE'])
    except KeyError:
        merged_ed9_settle = default_ed_px
    # ED10 - settle price
    try:
        merged_ed10_settle = float(row.loc['MERGED_ED10_SETTLE'])
    except KeyError:
        merged_ed10_settle = default_ed_px
    # ED11 - settle price
    try:
        merged_ed11_settle = float(row.loc['MERGED_ED11_SETTLE'])
    except KeyError:
        merged_ed11_settle = default_ed_px
    # ED12 - settle price
    try:
        merged_ed12_settle = float(row.loc['MERGED_ED12_SETTLE'])
    except KeyError:
        merged_ed12_settle = default_ed_px
    # ED13 - settle price
    try:
        merged_ed13_settle = float(row.loc['MERGED_ED13_SETTLE'])
    except KeyError:
        merged_ed13_settle = default_ed_px
    # ED14 - settle price
    try:
        merged_ed14_settle = float(row.loc['MERGED_ED14_SETTLE'])
    except KeyError:
        merged_ed14_settle = default_ed_px
    # ED15 - settle price
    try:
        merged_ed15_settle = float(row.loc['MERGED_ED15_SETTLE'])
    except KeyError:
        merged_ed15_settle = default_ed_px
    # ED16 - settle price
    try:
        merged_ed16_settle = float(row.loc['MERGED_ED16_SETTLE'])
    except KeyError:
        merged_ed16_settle = default_ed_px
    # ED17 - settle price
    try:
        merged_ed17_settle = float(row.loc['MERGED_ED17_SETTLE'])
    except KeyError:
        merged_ed17_settle = default_ed_px
    # ED18 - settle price
    try:
        merged_ed18_settle = float(row.loc['MERGED_ED18_SETTLE'])
    except KeyError:
        merged_ed18_settle = default_ed_px
    # ED19 - settle price
    try:
        merged_ed19_settle = float(row.loc['MERGED_ED19_SETTLE'])
    except KeyError:
        merged_ed19_settle = default_ed_px
    # ED20 - settle price
    try:
        merged_ed20_settle = float(row.loc['MERGED_ED20_SETTLE'])
    except KeyError:
        merged_ed20_settle = default_ed_px
    # ED21 - settle price
    try:
        merged_ed21_settle = float(row.loc['MERGED_ED21_SETTLE'])
    except KeyError:
        merged_ed21_settle = default_ed_px
    # ED22 - settle price
    try:
        merged_ed22_settle = float(row.loc['MERGED_ED22_SETTLE'])
    except KeyError:
        merged_ed22_settle = default_ed_px
    # ED23 - settle price
    try:
        merged_ed23_settle = float(row.loc['MERGED_ED23_SETTLE'])
    except KeyError:
        merged_ed23_settle = default_ed_px
    # ED24 - settle price
    try:
        merged_ed24_settle = float(row.loc['MERGED_ED24_SETTLE'])
    except KeyError:
        merged_ed24_settle = default_ed_px
    # NEW_DATE - date of yield curve
    new_date = row.name.to_pydatetime()

    yc_entry = EurodollarYieldCurve(Id=identifier,
                                    ED1=merged_ed1_settle,
                                    ED2=merged_ed2_settle,
                                    ED3=merged_ed3_settle,
                                    ED4=merged_ed4_settle,
                                    ED5=merged_ed5_settle,
                                    ED6=merged_ed6_settle,
                                    ED7=merged_ed7_settle,
                                    ED8=merged_ed8_settle,
                                    ED9=merged_ed9_settle,
                                    ED10=merged_ed10_settle,
                                    ED11=merged_ed11_settle,
                                    ED12=merged_ed12_settle,
                                    ED13=merged_ed13_settle,
                                    ED14=merged_ed14_settle,
                                    ED15=merged_ed15_settle,
                                    ED16=merged_ed16_settle,
                                    ED17=merged_ed17_settle,
                                    ED18=merged_ed18_settle,
                                    ED19=merged_ed19_settle,
                                    ED20=merged_ed20_settle,
                                    ED21=merged_ed21_settle,
                                    ED22=merged_ed22_settle,
                                    ED23=merged_ed23_settle,
                                    ED24=merged_ed24_settle,
                                    NEW_DATE=new_date)
    return yc_entry


def reversed_blocks(file, blocksize=4096):
    "Generate blocks of file's contents in reverse order."
    file.seek(0, os.SEEK_END)
    here = file.tell()
    while 0 < here:
        delta = min(blocksize, here)
        here -= delta
        file.seek(here, os.SEEK_SET)
        yield file.read(delta)


def reversed_lines(file):
    "Generate the lines of file in reverse order."
    part = ''
    for block in reversed_blocks(file):
        for c in reversed(block):
            if c == '\n' and part:
                yield part[::-1]
                part = ''
            part += c
    if part:
        yield part[::-1]


def write_to_csv(df,
                 symbol,
                 start_date,
                 end_date,
                 force_new_file=False):
    date_to_append = df.tail(1).index[0].to_pydatetime().date()
    total_file_path = OSMuxImpl.get_proper_path(CONT_FUT_WRITE_CSV_FILE_PATH)
    total_file_path += symbol + '_' + start_date + '_to_' + end_date + '.csv'
    LOGGER.info("cont_fut.write_to_csv(): writing dataframe to %s", total_file_path)
    # read the last line in the csv file (peek?)
    if os.path.exists(total_file_path) and force_new_file is False:
        with open(total_file_path, 'r') as f:
            last_line = islice(reversed_lines(f), 1)
            line = next(last_line)
            last_date = str(line).split(',')[0]
            format_str = '%Y-%m-%d'  # The format
            last_date_obj = datetime.strptime(last_date, format_str).date()
            df = df.loc[str(last_date_obj + timedelta(days=1)):]
            LOGGER.info("cont_fut.write_to_csv(): last date in file %s is %s", total_file_path, last_date)
            missing_range = pd.date_range(start=str(last_date_obj + timedelta(days=1)), end=str(date_to_append),
                                          freq='B')
            LOGGER.info("cont_fut.write_to_csv(): missing date range before running Class Holidates %s",
                        str(missing_range))
            ex_holidates_range = []
            for missing_date in missing_range:
                th = TradingHolidates(year=missing_date.to_pydatetime().date().year)
                if not th.skip(missing_date.to_pydatetime().date()):
                    # its a holiday, skip it
                    ex_holidates_range.append(missing_date.to_pydatetime().date())
            ex_holidates = pd.DatetimeIndex(ex_holidates_range)
            LOGGER.info("cont_fut.write_to_csv(): missing date range after running Class Holidates %s",
                        str(ex_holidates))
            if len(ex_holidates) > 2 and force_new_file is False:
                LOGGER.error(
                    "cont_fut.write_to_csv(): there are dates before this date that need to be "
                    "inserted first %s!", str(ex_holidates))
                return
    elif force_new_file is True:
        df.to_csv(total_file_path)
        return total_file_path
    else:
        # the file doesn't exit so we are creating a new file.
        force_new_file = True
    LOGGER.info("cont_fut.write_to_csv(): force_new_file flag is set to %s, writing to csv next...",
                str(force_new_file))
    if not force_new_file:
        LOGGER.info("cont_fut.write_to_csv(): append mode on file %s", total_file_path)
        df.to_csv(total_file_path, mode='a', header=False)
        return total_file_path
    elif force_new_file:
        LOGGER.info("cont_fut.write_to_csv(): write mode (existing file will be "
                    "written over if it exists %s", total_file_path)
        df.to_csv(total_file_path)
        return total_file_path


class DailyCMESettlementReport:

    def __init__(self,
                 report_date_dt):
        self.report_date_dt = report_date_dt
        report_str_date = str(report_date_dt.year) + str(report_date_dt.month).zfill(2) + \
                          str(report_date_dt.day).zfill(2)
        # the "dot s" files are uploaded to CME FTP at 4:40 pm CST. They are final settles.
        # which is 11:40 am Hawiian time (they are timestamped Hawaian time)
        # the "dot e" files are the prelim settles uploaded to CME FTP at 3:10 pm CST.
        # which is 10:00 am Hawian time (they are timestamped Hawaiian time)
        self.prelim_csv_settlement_file = "settle/cme.settle." + report_str_date + ".e.csv.zip"
        self.prelim_xml_settlement_file = "settle/cme.settle." + report_str_date + ".e.xml.zip"
        self.final_csv_settlement_file = "settle/cme.settle." + report_str_date + ".s.csv.zip"
        self.final_xml_settlement_file = "settle/cme.settle." + report_str_date + ".s.xml.zip"
        self.settlement_files_list = [self.final_csv_settlement_file,
                                      self.final_xml_settlement_file,
                                      self.prelim_xml_settlement_file,
                                      self.prelim_csv_settlement_file]
        self.cme_ftp_loc = "ftp.cmegroup.com"

    def get_settle_from_csv(self,
                            symbol,
                            security_type="FUT"):

        # use from_csv to load the csv file into dataframe. It is a very large file!
        settle_df = pd.read_csv(filepath_or_buffer="ftp://" + self.cme_ftp_loc + "/" + self.final_csv_settlement_file,
                                memory_map=True, compression="zip")
        filtered_sym_secTyp = settle_df[(settle_df.Sym == symbol) & (settle_df.SecTyp == security_type)]
        LOGGER.info("cont_fut.DailyCMESettlementReport.get_settle_from_csv(): "
                    "%s contracts matched filter where symbol is %s and security type is %s",
                    str(filtered_sym_secTyp.shape[0]), symbol, security_type)
        return filtered_sym_secTyp

    def get_all_reports(self):
        from ftplib import error_perm
        ftp = FTP(self.cme_ftp_loc)
        ftp.login()
        local_cme_settlement_report_dir = OSMuxImpl.get_proper_path(CME_SETTLEMENT_REPORT_LOCAL_DIR)
        # zip_timestamp = datetime.now().isoformat()
        for settlement_file in self.settlement_files_list:
            file_data = open(local_cme_settlement_report_dir + settlement_file, 'wb')
            LOGGER.info("cont_fut.DailyCMESettlementReport.get_all_reports(): getting settlement file %s",
                        settlement_file)
            try:
                ftp.retrbinary('RETR ' + settlement_file, file_data.write)
            except error_perm as ep:
                LOGGER.error("cont_fut.DailyCMESettlementReport.get_all_reports(): failed on %s with msg %s",
                             settlement_file, ep.__str__())
                file_data.close()
                continue
            file_data.close()
            settlement_zip_file = ZipFile(local_cme_settlement_report_dir + settlement_file)
            LOGGER.info("DailyCMESettlementReport.get_csv_report(): extracting zip file %s into directory %s",
                        settlement_zip_file.filename, local_cme_settlement_report_dir)
            settlement_zip_file.extractall(path=local_cme_settlement_report_dir)
            settlement_zip_file.close()
        ftp.quit()


class TradingHolidates:
    def __init__(self,
                 year):
        self.year = year
        # https: // www.timeanddate.com / holidays / us / good - friday
        self.good_friday_dict = {
            2010: datetime(2010, 4, 2).date(),
            2011: datetime(2011, 4, 22).date(),
            2012: datetime(2012, 4, 6).date(),
            2013: datetime(2013, 3, 29).date(),
            2014: datetime(2014, 4, 18).date(),
            2015: datetime(2015, 4, 3).date(),
            2016: datetime(2016, 3, 25).date(),
            2017: datetime(2017, 4, 14).date(),
            2018: datetime(2018, 3, 30).date(),
            2019: datetime(2019, 4, 19).date(),
            2020: datetime(2020, 4, 10).date(),
            2021: datetime(2021, 4, 2).date(),
            2022: datetime(2022, 4, 15).date(),
            2023: datetime(2023, 4, 7).date(),
            2024: datetime(2024, 3, 29).date(),
            2025: datetime(2025, 4, 18).date(),
            2026: datetime(2026, 4, 3).date(),
            2027: datetime(2027, 3, 26).date(),
            2028: datetime(2028, 4, 14).date(),
            2029: datetime(2029, 3, 30).date(),
            2030: datetime(2030, 4, 19).date()
        }

    def consolidate_years_holidates(self):
        return_list = [self.return_xmas(), self.return_good_friday(), self.return_labor_day(),
                       self.return_memorial_day(), self.return_new_years(), self.return_mlk_day(),
                       self.return_presidents_day(), self.return_thanksgiving()]
        return return_list

    def return_mlk_day(self):
        # MLK Day is the third monday of every January
        first_day_jan = datetime(self.year, 1, 1).date()
        day_of_week = first_day_jan.weekday()  # remmeber, 0 is Monday
        mlk_day = first_day_jan + timedelta(days=(7 - day_of_week) % 7 + 14)
        return mlk_day

    def return_presidents_day(self):
        # Presidents Day is the third monday of every February
        first_day_feb = datetime(self.year, 2, 1).date()
        day_of_week = first_day_feb.weekday()
        pres_day = first_day_feb + timedelta(days=(7 - day_of_week) % 7 + 14)
        return pres_day

    def return_good_friday(self):
        return self.good_friday_dict[self.year]

    def return_memorial_day(self):
        # Memorial Day is on the last Monday of May
        last_day_may = datetime(self.year, 5, 31).date()
        day_of_week = last_day_may.weekday()
        memorial_day = last_day_may - timedelta(days=day_of_week)
        return memorial_day

    def return_independence_day(self):
        the_fourth = datetime(self.year, 7, 4).date()
        day_of_week = the_fourth.weekday()
        if day_of_week == 6:
            # sunday, means it will be observed on Monday
            return the_fourth + timedelta(day=1)
        elif day_of_week == 5:
            # saturday, means no trades settlement until Monday, Friday no trade settlent
            return the_fourth - timedelta(days=1)
        else:
            return the_fourth

    def return_labor_day(self):
        # Labor day is the first monday in september
        first_day_sept = datetime(self.year, 9, 1).date()
        day_of_week = first_day_sept.weekday()
        pres_day = first_day_sept + timedelta(days=(7 - day_of_week) % 7)
        return pres_day

    def return_thanksgiving(self):
        # Thanksgiving is the 4th Thursday in November
        # Thursday Weekday() = 3
        first_day_november = datetime(self.year, 11, 1).date()
        day_of_week = first_day_november.weekday()
        if day_of_week == 3:
            xgiving_day = first_day_november + timedelta(days=21)
        elif day_of_week < 3:
            xgiving_day = first_day_november + timedelta(days=3 - day_of_week) + timedelta(days=21)
        elif day_of_week > 3:
            xgiving_day = first_day_november - timedelta(days=day_of_week - 3) + timedelta(days=28)
        return xgiving_day

    def return_xmas(self):
        xmas_day = datetime(self.year, 12, 25).date()
        # if xmas day on Saturday
        day_of_week = xmas_day.weekday()
        if day_of_week == 5:
            # no settle on the previous friday
            return xmas_day - timedelta(days=1)
        elif day_of_week == 6:
            return xmas_day + timedelta(days=1)
        else:
            return xmas_day

    def return_new_years(self):
        ny_day = datetime(self.year, 1, 1).date()
        # if new years day on saturday...
        day_of_week = ny_day.weekday()
        if day_of_week == 5:
            # no settle on the previous friday
            return ny_day - timedelta(days=1)
        elif day_of_week == 6:
            return ny_day + timedelta(days=1)
        else:
            return ny_day

    def skip(self, d):
        holidates_list = self.consolidate_years_holidates()
        bool_list = [hd == d for hd in holidates_list]
        if True in bool_list:
            return True
        else:
            return False


@event.listens_for(Table, "column_reflect")
def column_reflect(inspector, table, column_info):
    # set column.key = "attr_<lower_case_name>"
    column_info['key'] = "attr_%s" % column_info['name'].lower()


class VixCurve(Base):
    __tablename__ = VIX_FUTURES_CURVE_TABLE
    Id = Column("dt_Id", DateTime, primary_key=True)
    VX1 = Column("d_VX1", DECIMAL)
    VX2 = Column("d_VX2", DECIMAL)
    VX3 = Column("d_VX3", DECIMAL)
    VX4 = Column("d_VX4", DECIMAL)
    VX5 = Column("d_VX5", DECIMAL)
    VX6 = Column("d_VX6", DECIMAL)
    VX7 = Column("d_VX7", DECIMAL)
    NEW_DATE = Column("d_NEW_DATE", DateTime)

    def __repr__(self):
        return "<vix_curve(Id='%s', VX1='%s', VX2='%s', VX3='%s'," \
               "VX4='%s', VX5='%s', VX6='%s', VX7='%s', NEW_DATE='%s')>" % (
                   self.Id, self.VX1, self.VX2, self.VX3, self.VX4, self.VX5, self.VX6, self.VX7, self.NEW_DATE)

    def equals(self, vxc_entry_obj):
        id_check = self.Id == vxc_entry_obj.Id
        VX1_check = round(self.VX1, 2) == round(vxc_entry_obj.VX1, 2)
        VX2_check = round(self.VX2, 2) == round(vxc_entry_obj.VX2, 2)
        VX3_check = round(self.VX3, 2) == round(vxc_entry_obj.VX3, 2)
        VX4_check = round(self.VX4, 2) == round(vxc_entry_obj.VX4, 2)
        VX5_check = round(self.VX5, 2) == round(vxc_entry_obj.VX5, 2)
        VX6_check = round(self.VX6, 2) == round(vxc_entry_obj.VX6, 2)
        VX7_check = round(self.VX7, 2) == round(vxc_entry_obj.VX7, 2)
        NEW_DATE_check = self.NEW_DATE == vxc_entry_obj.NEW_DATE
        total_check = id_check & VX1_check & VX2_check & VX3_check & VX4_check & VX5_check & \
                      VX6_check & VX7_check & NEW_DATE_check
        return total_check

    def set_all(self, vxc_entry_obj):
        self.Id = vxc_entry_obj.Id
        self.VX1 = vxc_entry_obj.VX1
        self.VX2 = vxc_entry_obj.VX2
        self.VX3 = vxc_entry_obj.VX3
        self.VX4 = vxc_entry_obj.VX4
        self.VX5 = vxc_entry_obj.VX5
        self.VX6 = vxc_entry_obj.VX6
        self.VX7 = vxc_entry_obj.VX7
        self.NEW_DATE = vxc_entry_obj.NEW_DATE


def get_vx_data_from_cboe_csv():
    """
        Purpose of this function is to have an alternative source of data when QUANDL does not have it.
        This means we parse the CSV files from CBOE directly, which is a pain in the ass but something
        that we have to do since QUANDL data source is free and can't be 100% relied upon.
        :return:
        """
    CBOE_VX_FUTS_HISTORICAL_YEAR_RANGE = (date(2013, 1, 1), date(datetime.now().year + 1, 12, 31))
    months_to_pull = pd.date_range(CBOE_VX_FUTS_HISTORICAL_YEAR_RANGE[0], CBOE_VX_FUTS_HISTORICAL_YEAR_RANGE[1],
                                   freq='M')
    return_value_list = set()
    for m_y in months_to_pull:
        return_value = vx_futures_settle_date(m_y.month, m_y.year)
        url_response_code = cboe.get_historical_vx_data_from_cboe(return_value)
        if url_response_code == 200:
            return_value_list.add(timedelta_minus_business_days(return_value, days=1))
    return sorted(return_value_list)


def vx_continuous(expiry_dates, expiry=0):
    file_prefix = "CFE_"
    file_suffix = "_VX.csv"
    keys_list = list(MONTHLY_EXPIRY_MONTH_CODE_MAPPING.keys())
    values_list = list(MONTHLY_EXPIRY_MONTH_CODE_MAPPING.values())
    total_df = pd.DataFrame()
    for m_y in sorted(expiry_dates):
        prev_m_y = list(sorted(expiry_dates))[list(sorted(expiry_dates)).index(m_y) - 1]
        try:
            keys_list_indexer = values_list.index(m_y.month) + expiry
            month_code = keys_list[keys_list_indexer]
            year = datetime.strftime(m_y, "%y")
        except IndexError:
            keys_list_indexer = (values_list.index(m_y.month) + expiry) % 12
            month_code = keys_list[keys_list_indexer]
            year = datetime.strftime(m_y + pd.DateOffset(years=1), "%y")
        full_front_filename = file_prefix + month_code + str(year) + file_suffix
        path_to_front_month_file = OSMuxImpl.get_proper_path(cboe.LOCAL_CBOE_DATA_DIR) + full_front_filename
        if not os.path.exists(path_to_front_month_file):
            continue
        df_1 = pd.DataFrame().from_csv(path_to_front_month_file, header=0, sep=',', parse_dates=True, index_col=0)
        if not total_df.empty:
            total_df = total_df.loc[total_df.index <= pd.to_datetime(prev_m_y)]
            total_df = total_df.append(df_1.loc[df_1.index > pd.to_datetime(prev_m_y)])
        else:
            total_df = df_1
    return total_df


def vx_roll_adjust_from_csv(unadj_csv_file, rollover_days=5, number_of_cont_contracts=7):
    usecols_list = ['Futures.', 'Open.', 'High.', 'Low.', 'Close.',
                    'Settle.', 'Change.', 'Total Volume.', 'EFP.', 'Open Interest.']
    px_type_list = ['Open.', 'High.', 'Low.', 'Close.', 'Settle.']
    cont_ts_list = []
    for contract_cnt in range(number_of_cont_contracts + 1):
        px_type_list_front_month = [col_nm + str(contract_cnt) for col_nm in px_type_list]
        px_type_list_back_month = [col_nm + str(contract_cnt + 1) for col_nm in px_type_list]
        new_col_nms = new_col_nms_front_month + new_col_nms_back_month
        new_col_nms.insert(0, 'Trade Date')
        df = pd.read_csv(unadj_csv_file, header=0, sep=',', parse_dates=True, index_col=0, usecols=new_col_nms)
        for col in new_col_nms:
            if col.find('Settle.') != -1:
                zero_settle_index = df[df[col] == 0.0].index
                df.drop(zero_settle_index, inplace=True)
            if col != 'Trade Date':
                df[col + '.Shifted'] = df[[col]].shift(periods=-rollover_days)
        roll_range_df = df[df['Futures.' + str(contract_cnt) + '.Shifted'] != df['Futures.' + str(contract_cnt)]]
        roll_range_df.dropna(inplace=True)
        the_years = pd.unique(roll_range_df.index.year)
        decay_weights_front_df = pd.Series(1.0, index=df.index)
        decay_weights_back_df = pd.Series(0.0, index=df.index)
        for unique_year in the_years:
            for mnth in range(1, 13):
                try:
                    this_month_year_df = roll_range_df.loc[str(unique_year) + '-' + str(mnth)]
                except KeyError:
                    continue
                decay_weights_front = 1 - np.linspace(0, 1, min(len(this_month_year_df) + 1, rollover_days + 1))[1:]
                decay_weights_back = np.linspace(0, 1, min(len(this_month_year_df) + 1, rollover_days + 1))[1:]
                decay_weights_front_df[decay_weights_front_df.index.isin(this_month_year_df.index)] = \
                    pd.Series(decay_weights_front, index=this_month_year_df.index)
                decay_weights_back_df[decay_weights_back_df.index.isin(this_month_year_df.index)] = \
                    pd.Series(decay_weights_back, index=this_month_year_df.index)
        res_front = df[px_type_list_front_month].mul(decay_weights_front_df, axis=0)
        res_front.columns = px_type_list
        res_back = df[px_type_list_back_month].mul(decay_weights_back_df, axis=0)
        res_back.columns = px_type_list
        cont_ts = res_front.add(res_back)
        cont_con_col_names = [px_type + str(contract_cnt) + '.Cont' for px_type in px_type_list]
        cont_ts.columns = cont_con_col_names
        cont_ts.insert(0, 'Futures.ContCon.' + str(contract_cnt), contract_cnt)
        cont_ts['DecayWeightsFront.' + str(contract_cnt)] = decay_weights_front_df
        cont_ts['DecayWeightsBack.' + str(contract_cnt)] = decay_weights_back_df
        file_name = write_to_csv(cont_ts, symbol='VX' + str(contract_cnt), start_date='2013-01-01',
                                 end_date=str(datetime.now().date()),
                                 force_new_file=True)
        cont_ts_list.append(cont_ts)
    return cont_ts_list


def vx_continuous_contract_creation():
    expiry_dates = get_vx_data_from_cboe_csv()
    all_vx_contracts_df = pd.DataFrame()
    for con_expiry_num in range(9):
        this_expiry_df = vx_continuous(expiry_dates=expiry_dates, expiry=con_expiry_num)
        new_column_headers = [col + '.' + str(con_expiry_num) for col in this_expiry_df.columns]
        this_expiry_df.rename(columns=dict(zip(this_expiry_df.columns, new_column_headers)), inplace=True)
        if all_vx_contracts_df.empty:
            all_vx_contracts_df = this_expiry_df
        else:
            all_vx_contracts_df = pd.concat([all_vx_contracts_df, this_expiry_df], axis=1, sort=False)
    file_name = write_to_csv(all_vx_contracts_df, 'VX', start_date="2013-01-01",
                             end_date=str(datetime.now().date()), force_new_file=True)
    return file_name


class WTICrudeOilCurve(Base):
    # __table__ = Table(WTI_CRUDE_OIL_CURVE_TABLE, Base.metadata,
    #                  autoload=True, autoload_with=get_engine())
    __tablename__ = WTI_CRUDE_OIL_CURVE_TABLE
    Id = Column("dt_Id", DateTime, primary_key=True)
    CL1 = Column("d_CL1", DECIMAL)
    CL2 = Column("d_CL2", DECIMAL)
    CL3 = Column("d_CL3", DECIMAL)
    CL4 = Column("d_CL4", DECIMAL)
    CL5 = Column("d_CL5", DECIMAL)
    CL6 = Column("d_CL6", DECIMAL)
    CL7 = Column("d_CL7", DECIMAL)
    CL8 = Column("d_CL8", DECIMAL)
    CL9 = Column("d_CL9", DECIMAL)
    CL10 = Column("d_CL10", DECIMAL)
    CL11 = Column("d_CL11", DECIMAL)
    CL12 = Column("d_CL12", DECIMAL)
    CL13 = Column("d_CL13", DECIMAL)
    CL14 = Column("d_CL14", DECIMAL)
    CL15 = Column("d_CL15", DECIMAL)
    CL16 = Column("d_CL16", DECIMAL)
    CL17 = Column("d_CL17", DECIMAL)
    CL18 = Column("d_CL18", DECIMAL)
    CL19 = Column("d_CL19", DECIMAL)
    CL20 = Column("d_CL20", DECIMAL)
    CL21 = Column("d_CL21", DECIMAL)
    CL22 = Column("d_CL22", DECIMAL)
    CL23 = Column("d_CL23", DECIMAL)
    CL24 = Column("d_CL24", DECIMAL)
    CL25 = Column("d_CL25", DECIMAL)
    CL26 = Column("d_CL26", DECIMAL)
    CL27 = Column("d_CL27", DECIMAL)
    CL28 = Column("d_CL28", DECIMAL)
    CL29 = Column("d_CL29", DECIMAL)
    CL30 = Column("d_CL30", DECIMAL)
    CL31 = Column("d_CL31", DECIMAL)
    CL32 = Column("d_CL32", DECIMAL)
    CL33 = Column("d_CL33", DECIMAL)
    CL34 = Column("d_CL34", DECIMAL)
    CL35 = Column("d_CL35", DECIMAL)
    CL36 = Column("d_CL36", DECIMAL)
    CL37 = Column("d_CL37", DECIMAL)
    CL38 = Column("d_CL38", DECIMAL)
    CL39 = Column("d_CL39", DECIMAL)
    NEW_DATE = Column("d_NEW_DATE", DateTime)

    def __repr__(self):
        return "<wti_crude_oil_curve(Id='%s', CL1='%s', CL2='%s', CL3='%s'," \
               "CL4='%s', CL5='%s', CL6='%s', CL7='%s'," \
               "CL8='%s', CL9='%s', CL10='%s', CL11='%s'," \
               "CL12='%s', CL13='%s', CL14='%s', CL15='%s', " \
               "CL16='%s', CL17='%s', CL18='%s', CL19='%s', CL20='%s', " \
               "CL21='%s', CL22='%s', CL23='%s', CL24='%s', CL25='%s', CL26='%s', CL27='%s', CL28='%s'," \
               "CL29='%s', CL30='%s', CL31='%s', CL32='%s', CL33='%s', CL34='%s', CL35='%s', CL36='%s'," \
               "CL37='%s', CL38='%s', CL39='%s', NEW_DATE='%s')>" % (
                   self.Id, self.CL1, self.CL2, self.CL3, self.CL4, self.CL5, self.CL6, self.CL7,
                   self.CL8, self.CL9, self.CL10, self.CL11, self.CL12, self.CL13, self.CL14,
                   self.CL15, self.CL16, self.CL17, self.CL18, self.CL19, self.CL20, self.CL21,
                   self.CL22, self.CL23, self.CL24, self.CL25, self.CL26, self.CL27, self.CL28, self.CL29,
                   self.CL30, self.CL31, self.CL32, self.CL33, self.CL34, self.CL35, self.CL36, self.CL37,
                   self.CL38, self.CL39, self.NEW_DATE)

    def equals(self, yc_entry_obj):
        id_check = self.Id == yc_entry_obj.Id
        CL1_check = round(self.CL1, 2) == round(yc_entry_obj.CL1, 2)
        CL2_check = round(self.CL2, 2) == round(yc_entry_obj.CL2, 2)
        CL3_check = round(self.CL3, 2) == round(yc_entry_obj.CL3, 2)
        CL4_check = round(self.CL4, 2) == round(yc_entry_obj.CL4, 2)
        CL5_check = round(self.CL5, 2) == round(yc_entry_obj.CL5, 2)
        CL6_check = round(self.CL6, 2) == round(yc_entry_obj.CL6, 2)
        CL7_check = round(self.CL7, 2) == round(yc_entry_obj.CL7, 2)
        CL8_check = round(self.CL8, 2) == round(yc_entry_obj.CL8, 2)
        CL9_check = round(self.CL9, 2) == round(yc_entry_obj.CL9, 2)
        CL10_check = round(self.CL10, 2) == round(yc_entry_obj.CL10, 2)
        CL11_check = round(self.CL11, 2) == round(yc_entry_obj.CL11, 2)
        CL12_check = round(self.CL12, 2) == round(yc_entry_obj.CL12, 2)
        CL13_check = round(self.CL13, 2) == round(yc_entry_obj.CL13, 2)
        CL14_check = round(self.CL14, 2) == round(yc_entry_obj.CL14, 2)
        CL15_check = round(self.CL15, 2) == round(yc_entry_obj.CL15, 2)
        CL16_check = round(self.CL16, 2) == round(yc_entry_obj.CL16, 2)
        CL17_check = round(self.CL17, 2) == round(yc_entry_obj.CL17, 2)
        CL18_check = round(self.CL18, 2) == round(yc_entry_obj.CL18, 2)
        CL19_check = round(self.CL19, 2) == round(yc_entry_obj.CL19, 2)
        CL20_check = round(self.CL20, 2) == round(yc_entry_obj.CL20, 2)
        CL21_check = round(self.CL21, 2) == round(yc_entry_obj.CL21, 2)
        CL22_check = round(self.CL22, 2) == round(yc_entry_obj.CL22, 2)
        CL23_check = round(self.CL23, 2) == round(yc_entry_obj.CL23, 2)
        CL24_check = round(self.CL24, 2) == round(yc_entry_obj.CL24, 2)
        CL25_check = round(self.CL25, 2) == round(yc_entry_obj.CL25, 2)
        CL26_check = round(self.CL26, 2) == round(yc_entry_obj.CL26, 2)
        CL27_check = round(self.CL27, 2) == round(yc_entry_obj.CL27, 2)
        CL28_check = round(self.CL28, 2) == round(yc_entry_obj.CL28, 2)
        CL29_check = self.CL29 == yc_entry_obj.CL29
        CL30_check = self.CL30 == yc_entry_obj.CL30
        CL31_check = self.CL31 == yc_entry_obj.CL31
        CL32_check = self.CL32 == yc_entry_obj.CL32
        CL33_check = self.CL33 == yc_entry_obj.CL33
        CL34_check = self.CL34 == yc_entry_obj.CL34
        CL35_check = self.CL35 == yc_entry_obj.CL35
        CL36_check = self.CL36 == yc_entry_obj.CL36
        CL37_check = self.CL37 == yc_entry_obj.CL37
        CL38_check = self.CL38 == yc_entry_obj.CL38
        CL39_check = self.CL39 == yc_entry_obj.CL39
        NEW_DATE_check = self.NEW_DATE == yc_entry_obj.NEW_DATE
        total_bool = id_check & CL1_check & CL2_check & CL3_check & CL4_check & CL5_check & CL6_check & CL7_check \
                     & CL8_check & CL9_check & CL10_check & CL11_check & CL12_check & CL13_check & CL14_check \
                     & CL15_check & CL16_check & CL17_check & CL18_check & CL19_check & CL20_check & CL21_check \
                     & CL22_check & CL23_check & CL24_check & CL25_check & CL26_check & CL27_check & CL28_check \
                     & CL29_check & CL30_check & CL31_check & CL32_check & CL33_check & CL34_check & CL35_check \
                     & CL36_check & CL37_check & CL38_check & CL39_check & NEW_DATE_check
        return total_bool

    def set_all(self, yc_entry_obj):
        self.Id = yc_entry_obj.Id
        self.CL1 = yc_entry_obj.CL1
        self.CL2 = yc_entry_obj.CL2
        self.CL3 = yc_entry_obj.CL3
        self.CL4 = yc_entry_obj.CL4
        self.CL5 = yc_entry_obj.CL5
        self.CL6 = yc_entry_obj.CL6
        self.CL7 = yc_entry_obj.CL7
        self.CL8 = yc_entry_obj.CL8
        self.CL9 = yc_entry_obj.CL9
        self.CL10 = yc_entry_obj.CL10
        self.CL11 = yc_entry_obj.CL11
        self.CL12 = yc_entry_obj.CL12
        self.CL13 = yc_entry_obj.CL13
        self.CL14 = yc_entry_obj.CL14
        self.CL15 = yc_entry_obj.CL15
        self.CL16 = yc_entry_obj.CL16
        self.CL17 = yc_entry_obj.CL17
        self.CL18 = yc_entry_obj.CL18
        self.CL19 = yc_entry_obj.CL19
        self.CL20 = yc_entry_obj.CL20
        self.CL21 = yc_entry_obj.CL21
        self.CL22 = yc_entry_obj.CL22
        self.CL23 = yc_entry_obj.CL23
        self.CL24 = yc_entry_obj.CL24
        self.CL25 = yc_entry_obj.CL25
        self.CL26 = yc_entry_obj.CL26
        self.CL27 = yc_entry_obj.CL27
        self.CL28 = yc_entry_obj.CL28
        self.CL29 = yc_entry_obj.CL29
        self.CL30 = yc_entry_obj.CL30
        self.CL31 = yc_entry_obj.CL31
        self.CL32 = yc_entry_obj.CL32
        self.CL33 = yc_entry_obj.CL33
        self.CL34 = yc_entry_obj.CL34
        self.CL35 = yc_entry_obj.CL35
        self.CL36 = yc_entry_obj.CL36
        self.CL37 = yc_entry_obj.CL37
        self.CL38 = yc_entry_obj.CL38
        self.CL39 = yc_entry_obj.CL39
        self.NEW_DATE = yc_entry_obj.NEW_DATE


class CurrencyFutures(Base):
    __tablename__ = CURRENCY_TABLE
    symbol = Column("c_SYMBOL", String(40), primary_key=True)
    Id = Column("dt_Id", DateTime, primary_key=True)
    roll_type = Column("c_ROLL_TYPE", String(40), primary_key=True)
    near_month_settle = Column("d_NEAR_MONTH_SETTLE", DECIMAL)
    back_month_settle = Column("d_BACK_MONTH_SETTLE", DECIMAL)
    near_month_open = Column("d_NEAR_MONTH_OPEN", DECIMAL)
    back_month_open = Column("d_BACK_MONTH_OPEN", DECIMAL)
    near_month_high = Column("d_NEAR_MONTH_HIGH", DECIMAL)
    back_month_high = Column("d_BACK_MONTH_HIGH", DECIMAL)
    near_month_low = Column("d_NEAR_MONTH_LOW", DECIMAL)
    back_month_low = Column("d_BACK_MONTH_LOW", DECIMAL)
    near_month_last = Column("d_NEAR_MONTH_LAST", DECIMAL)
    back_month_last = Column("d_BACK_MONTH_LAST", DECIMAL)
    near_month_volume = Column("d_NEAR_MONTH_VOLUME", DECIMAL)
    back_month_volume = Column("d_BACK_MONTH_VOLUME", DECIMAL)
    near_month_open_interest = Column("d_NEAR_MONTH_PREVIOUS_DAY_OPEN_INTEREST", DECIMAL)
    back_month_open_interest = Column("d_BACK_MONTH_PREVIOUS_DAY_OPEN_INTEREST", DECIMAL)
    merged_settle = Column("d_MERGED_SETTLE", DECIMAL)
    merged_open = Column("d_MERGED_OPEN", DECIMAL)
    merged_high = Column("d_MERGED_HIGH", DECIMAL)
    merged_low = Column("d_MERGED_LOW", DECIMAL)
    merged_last = Column("d_MERGED_LAST", DECIMAL)
    merged_volume = Column("d_MERGED_VOLUME", DECIMAL)
    merged_previous_day_open_interest = Column("d_MERGED_PREVIOUS_DAY_OPEN_INTEREST", DECIMAL)
    adj_factor_settle = Column("d_ADJUST_FACTOR_SETTLE", DECIMAL)
    adj_factor_open = Column("d_ADJUST_FACTOR_OPEN", DECIMAL)
    adj_factor_high = Column("d_ADJUST_FACTOR_HIGH", DECIMAL)
    adj_factor_low = Column("d_ADJUST_FACTOR_LOW", DECIMAL)
    adj_factor_last = Column("d_ADJUST_FACTOR_LAST", DECIMAL)
    adj_factor_volume = Column("d_ADJUST_FACTOR_VOLUME", DECIMAL)
    adj_factor_previous_day_open_interest = Column("d_ADJUST_FACTOR_PREVIOUS_DAY_OPEN_INTEREST", DECIMAL)
    NEW_DATE = Column("dt_NEW_DATE", DateTime)

    # CASH CURRENCY COLUMNS
    SPOT_MKT_OBS_VALUE = Column("d_SPOT_MKT_OBS_VALUE", DECIMAL)
    SPOT_MKT_VALUE_DESC = Column("c_SPOT_MKT_VALUE_DESC", String(40))  # Like "yen INTO usd"

    def __repr__(self):
        return "<currencyfutures(Id='%s', symbol='%s', roll_type='%s', near_month_settle='%s', back_month_settle='%s'," \
               "near_month_open='%s', back_month_open='%s', near_month_high='%s', back_month_high='%s'," \
               "near_month_low='%s', back_month_low='%s', near_month_last='%s', back_month_last='%s'," \
               "near_month_volume='%s', back_month_volume='%s', near_month_open_interest='%s', " \
               "back_month_open_interest='%s', merged_settle='%s', merged_open='%s', merged_high='%s', " \
               "merged_low='%s', merged_last='%s', merged_volume='%s', merged_previous_day_open_interest='%s', " \
               "adj_factor_settle='%s', adj_factor_open='%s', adj_factor_high='%s', adj_factor_low='%s', " \
               "adj_factor_last='%s', adj_factor_volume='%s', adj_factor_previous_day_open_interest='%s', " \
               "NEW_DATE='%s', spot_mkt_obs_value='%s', spot_mkt_value_desc='%s')>" % (
                   self.Id, self.symbol, self.roll_type, self.near_month_settle, self.back_month_settle,
                   self.near_month_open, self.back_month_open, self.near_month_high,
                   self.back_month_high, self.near_month_low, self.back_month_low,
                   self.near_month_last, self.back_month_last, self.near_month_volume,
                   self.back_month_volume,
                   self.near_month_open_interest, self.back_month_open_interest, self.merged_settle,
                   self.merged_open, self.merged_high, self.merged_low, self.merged_last,
                   self.merged_volume,
                   self.merged_previous_day_open_interest, self.adj_factor_settle,
                   self.adj_factor_open, self.adj_factor_high, self.adj_factor_low,
                   self.adj_factor_last,
                   self.adj_factor_volume, self.adj_factor_previous_day_open_interest, self.NEW_DATE,
                   self.SPOT_MKT_OBS_VALUE, self.SPOT_MKT_VALUE_DESC)

    def equals(self, entry_obj):
        id_check = self.Id == entry_obj.Id
        symbol_check = self.symbol == entry_obj.symbol
        roll_type_check = self.roll_type == entry_obj.roll_type
        near_month_settle_check = self.near_month_settle == entry_obj.near_month_settle
        back_month_settle_check = self.back_month_settle == entry_obj.back_month_settle
        near_month_open_check = self.near_month_open == entry_obj.near_month_open
        back_month_open_check = self.back_month_open == entry_obj.back_month_open
        near_month_high_check = self.near_month_high == entry_obj.near_month_high
        back_month_high_check = self.back_month_high == entry_obj.back_month_high
        near_month_low_check = self.near_month_low == entry_obj.near_month_low
        back_month_low_check = self.back_month_low == entry_obj.back_month_low
        near_month_last_check = self.near_month_last == entry_obj.near_month_last
        back_month_last_check = self.back_month_last == entry_obj.back_month_last
        near_month_volume_check = self.near_month_volume == entry_obj.near_month_volume
        back_month_volume_check = self.back_month_volume == entry_obj.back_month_volume
        near_month_open_interest_check = self.near_month_open_interest == entry_obj.near_month_open_interest
        back_month_open_interest_check = self.back_month_open_interest == entry_obj.back_month_open_interest
        merged_settle_check = self.merged_settle == entry_obj.merged_settle
        merged_open_check = self.merged_open == entry_obj.merged_open
        merged_high_check = self.merged_high == entry_obj.merged_high
        merged_low_check = self.merged_low == entry_obj.merged_low
        merged_last_check = self.merged_last == entry_obj.merged_last
        merged_volume_check = self.merged_volume == entry_obj.merged_volume
        merged_previous_day_open_interest_check = self.merged_previous_day_open_interest == \
                                                  entry_obj.merged_previous_day_open_interest
        adj_factor_settle_check = self.adj_factor_settle == entry_obj.adj_factor_settle
        adj_factor_open_check = self.adj_factor_open == entry_obj.adj_factor_open
        adj_factor_high_check = self.adj_factor_high == entry_obj.adj_factor_high
        adj_factor_low_check = self.adj_factor_low == entry_obj.adj_factor_low
        adj_factor_last_check = self.adj_factor_last == entry_obj.adj_factor_last
        adj_factor_volume_check = self.adj_factor_volume == entry_obj.adj_factor_volume
        adj_factor_previous_day_open_interest = self.adj_factor_previous_day_open_interest \
                                                == entry_obj.adj_factor_previous_day_open_interest
        new_date_check = self.NEW_DATE == entry_obj.NEW_DATE
        spot_mkt_obs_value = self.SPOT_MKT_OBS_VALUE == entry_obj.SPOT_MKT_OBS_VALUE
        spot_mkt_value_desc = self.SPOT_MKT_VALUE_DESC == entry_obj.SPOT_MKT_VALUE_DESC
        total_bool = id_check & symbol_check & roll_type_check & near_month_settle_check & back_month_settle_check \
                     & near_month_open_check \
                     & back_month_open_check & near_month_high_check & back_month_high_check & near_month_low_check \
                     & back_month_low_check & near_month_last_check & back_month_last_check & near_month_volume_check \
                     & back_month_volume_check & near_month_open_interest_check & back_month_open_interest_check \
                     & merged_settle_check & merged_open_check & merged_high_check & merged_low_check & merged_last_check \
                     & merged_volume_check & merged_previous_day_open_interest_check & adj_factor_settle_check \
                     & adj_factor_open_check & adj_factor_high_check & adj_factor_low_check & adj_factor_last_check \
                     & adj_factor_volume_check & adj_factor_previous_day_open_interest & new_date_check \
                     & spot_mkt_obs_value & spot_mkt_value_desc
        return total_bool

    def set_all(self, entry_obj):
        self.Id = entry_obj.Id
        self.symbol = entry_obj.symbol
        self.roll_type = entry_obj.roll_type
        self.near_month_settle = entry_obj.near_month_settle
        self.back_month_settle = entry_obj.back_month_settle
        self.near_month_open = entry_obj.near_month_open
        self.back_month_open = entry_obj.back_month_open
        self.near_month_high = entry_obj.near_month_high
        self.back_month_high = entry_obj.back_month_high
        self.near_month_low = entry_obj.near_month_low
        self.back_month_low = entry_obj.back_month_low
        self.near_month_last = entry_obj.near_month_last
        self.back_month_last = entry_obj.back_month_last
        self.near_month_volume = entry_obj.near_month_volume
        self.back_month_volume = entry_obj.back_month_volume
        self.near_month_open_interest = entry_obj.near_month_open_interest
        self.back_month_open_interest = entry_obj.back_month_open_interest
        self.merged_settle = entry_obj.merged_settle
        self.merged_open = entry_obj.merged_open
        self.merged_high = entry_obj.merged_high
        self.merged_low = entry_obj.merged_low
        self.merged_last = entry_obj.merged_last
        self.merged_volume = entry_obj.merged_volume
        self.merged_previous_day_open_interest = entry_obj.merged_previous_day_open_interest
        self.adj_factor_settle = entry_obj.adj_factor_settle
        self.adj_factor_open = entry_obj.adj_factor_open
        self.adj_factor_high = entry_obj.adj_factor_high
        self.adj_factor_low = entry_obj.adj_factor_low
        self.adj_factor_last = entry_obj.adj_factor_last
        self.adj_factor_volume = entry_obj.adj_factor_volume
        self.adj_factor_previous_day_open_interest = entry_obj.adj_factor_previous_day_open_interest
        self.NEW_DATE = entry_obj.NEW_DATE
        self.SPOT_MKT_VALUE_DESC = entry_obj.SPOT_MKT_VALUE_DESC
        self.SPOT_MKT_OBS_VALUE = entry_obj.SPOT_MKT_OBS_VALUE


class EquityFutures(Base):
    __tablename__ = EQUITY_FUTURES_TABLE
    symbol = Column("c_SYMBOL", String(40), primary_key=True)
    Id = Column("dt_Id", DateTime, primary_key=True)
    roll_type = Column("c_ROLL_TYPE", String(40), primary_key=True)
    near_month_settle = Column("d_NEAR_MONTH_SETTLE", DECIMAL)
    back_month_settle = Column("d_BACK_MONTH_SETTLE", DECIMAL)
    near_month_open = Column("d_NEAR_MONTH_OPEN", DECIMAL)
    back_month_open = Column("d_BACK_MONTH_OPEN", DECIMAL)
    near_month_high = Column("d_NEAR_MONTH_HIGH", DECIMAL)
    back_month_high = Column("d_BACK_MONTH_HIGH", DECIMAL)
    near_month_low = Column("d_NEAR_MONTH_LOW", DECIMAL)
    back_month_low = Column("d_BACK_MONTH_LOW", DECIMAL)
    near_month_last = Column("d_NEAR_MONTH_LAST", DECIMAL)
    back_month_last = Column("d_BACK_MONTH_LAST", DECIMAL)
    near_month_volume = Column("d_NEAR_MONTH_VOLUME", DECIMAL)
    back_month_volume = Column("d_BACK_MONTH_VOLUME", DECIMAL)
    near_month_open_interest = Column("d_NEAR_MONTH_PREVIOUS_DAY_OPEN_INTEREST", DECIMAL)
    back_month_open_interest = Column("d_BACK_MONTH_PREVIOUS_DAY_OPEN_INTEREST", DECIMAL)
    merged_settle = Column("d_MERGED_SETTLE", DECIMAL)
    merged_open = Column("d_MERGED_OPEN", DECIMAL)
    merged_high = Column("d_MERGED_HIGH", DECIMAL)
    merged_low = Column("d_MERGED_LOW", DECIMAL)
    merged_last = Column("d_MERGED_LAST", DECIMAL)
    merged_volume = Column("d_MERGED_VOLUME", DECIMAL)
    merged_previous_day_open_interest = Column("d_MERGED_PREVIOUS_DAY_OPEN_INTEREST", DECIMAL)
    adj_factor_settle = Column("d_ADJUST_FACTOR_SETTLE", DECIMAL)
    adj_factor_open = Column("d_ADJUST_FACTOR_OPEN", DECIMAL)
    adj_factor_high = Column("d_ADJUST_FACTOR_HIGH", DECIMAL)
    adj_factor_low = Column("d_ADJUST_FACTOR_LOW", DECIMAL)
    adj_factor_last = Column("d_ADJUST_FACTOR_LAST", DECIMAL)
    adj_factor_volume = Column("d_ADJUST_FACTOR_VOLUME", DECIMAL)
    adj_factor_previous_day_open_interest = Column("d_ADJUST_FACTOR_PREVIOUS_DAY_OPEN_INTEREST", DECIMAL)
    NEW_DATE = Column("dt_NEW_DATE", DateTime)

    def __repr__(self):
        return "<equityfutures(Id='%s', symbol='%s', roll_type='%s', near_month_settle='%s', back_month_settle='%s'," \
               "near_month_open='%s', back_month_open='%s', near_month_high='%s', back_month_high='%s'," \
               "near_month_low='%s', back_month_low='%s', near_month_last='%s', back_month_last='%s'," \
               "near_month_volume='%s', back_month_volume='%s', near_month_open_interest='%s', " \
               "back_month_open_interest='%s', merged_settle='%s', merged_open='%s', merged_high='%s', " \
               "merged_low='%s', merged_last='%s', merged_volume='%s', merged_previous_day_open_interest='%s', " \
               "adj_factor_settle='%s', adj_factor_open='%s', adj_factor_high='%s', adj_factor_low='%s', " \
               "adj_factor_last='%s', adj_factor_volume='%s', adj_factor_previous_day_open_interest='%s', " \
               "NEW_DATE='%s')>" % (
                   self.Id, self.symbol, self.roll_type, self.near_month_settle, self.back_month_settle,
                   self.near_month_open, self.back_month_open, self.near_month_high,
                   self.back_month_high, self.near_month_low, self.back_month_low,
                   self.near_month_last, self.back_month_last, self.near_month_volume,
                   self.back_month_volume,
                   self.near_month_open_interest, self.back_month_open_interest, self.merged_settle,
                   self.merged_open, self.merged_high, self.merged_low, self.merged_last,
                   self.merged_volume,
                   self.merged_previous_day_open_interest, self.adj_factor_settle,
                   self.adj_factor_open, self.adj_factor_high, self.adj_factor_low,
                   self.adj_factor_last,
                   self.adj_factor_volume, self.adj_factor_previous_day_open_interest, self.NEW_DATE)

    def equals(self, entry_obj):
        id_check = self.Id == entry_obj.Id
        symbol_check = self.symbol == entry_obj.symbol
        roll_type_check = self.roll_type == entry_obj.roll_type
        near_month_settle_check = self.near_month_settle == entry_obj.near_month_settle
        back_month_settle_check = self.back_month_settle == entry_obj.back_month_settle
        near_month_open_check = self.near_month_open == entry_obj.near_month_open
        back_month_open_check = self.back_month_open == entry_obj.back_month_open
        near_month_high_check = self.near_month_high == entry_obj.near_month_high
        back_month_high_check = self.back_month_high == entry_obj.back_month_high
        near_month_low_check = self.near_month_low == entry_obj.near_month_low
        back_month_low_check = self.back_month_low == entry_obj.back_month_low
        near_month_last_check = self.near_month_last == entry_obj.near_month_last
        back_month_last_check = self.back_month_last == entry_obj.back_month_last
        near_month_volume_check = self.near_month_volume == entry_obj.near_month_volume
        back_month_volume_check = self.back_month_volume == entry_obj.back_month_volume
        near_month_open_interest_check = self.near_month_open_interest == entry_obj.near_month_open_interest
        back_month_open_interest_check = self.back_month_open_interest == entry_obj.back_month_open_interest
        merged_settle_check = self.merged_settle == entry_obj.merged_settle
        merged_open_check = self.merged_open == entry_obj.merged_open
        merged_high_check = self.merged_high == entry_obj.merged_high
        merged_low_check = self.merged_low == entry_obj.merged_low
        merged_last_check = self.merged_last == entry_obj.merged_last
        merged_volume_check = self.merged_volume == entry_obj.merged_volume
        merged_previous_day_open_interest_check = self.merged_previous_day_open_interest == \
                                                  entry_obj.merged_previous_day_open_interest
        adj_factor_settle_check = self.adj_factor_settle == entry_obj.adj_factor_settle
        adj_factor_open_check = self.adj_factor_open == entry_obj.adj_factor_open
        adj_factor_high_check = self.adj_factor_high == entry_obj.adj_factor_high
        adj_factor_low_check = self.adj_factor_low == entry_obj.adj_factor_low
        adj_factor_last_check = self.adj_factor_last == entry_obj.adj_factor_last
        adj_factor_volume_check = self.adj_factor_volume == entry_obj.adj_factor_volume
        adj_factor_previous_day_open_interest = self.adj_factor_previous_day_open_interest \
                                                == entry_obj.adj_factor_previous_day_open_interest
        new_date_check = self.NEW_DATE == entry_obj.NEW_DATE
        total_bool = id_check & symbol_check & roll_type_check & near_month_settle_check & back_month_settle_check \
                     & near_month_open_check \
                     & back_month_open_check & near_month_high_check & back_month_high_check & near_month_low_check \
                     & back_month_low_check & near_month_last_check & back_month_last_check & near_month_volume_check \
                     & back_month_volume_check & near_month_open_interest_check & back_month_open_interest_check \
                     & merged_settle_check & merged_open_check & merged_high_check & merged_low_check & merged_last_check \
                     & merged_volume_check & merged_previous_day_open_interest_check & adj_factor_settle_check \
                     & adj_factor_open_check & adj_factor_high_check & adj_factor_low_check & adj_factor_last_check \
                     & adj_factor_volume_check & adj_factor_previous_day_open_interest & new_date_check
        return total_bool

    def set_all(self, entry_obj):
        self.Id = entry_obj.Id
        self.symbol = entry_obj.symbol
        self.roll_type = entry_obj.roll_type
        self.near_month_settle = entry_obj.near_month_settle
        self.back_month_settle = entry_obj.back_month_settle
        self.near_month_open = entry_obj.near_month_open
        self.back_month_open = entry_obj.back_month_open
        self.near_month_high = entry_obj.near_month_high
        self.back_month_high = entry_obj.back_month_high
        self.near_month_low = entry_obj.near_month_low
        self.back_month_low = entry_obj.back_month_low
        self.near_month_last = entry_obj.near_month_last
        self.back_month_last = entry_obj.back_month_last
        self.near_month_volume = entry_obj.near_month_volume
        self.back_month_volume = entry_obj.back_month_volume
        self.near_month_open_interest = entry_obj.near_month_open_interest
        self.back_month_open_interest = entry_obj.back_month_open_interest
        self.merged_settle = entry_obj.merged_settle
        self.merged_open = entry_obj.merged_open
        self.merged_high = entry_obj.merged_high
        self.merged_low = entry_obj.merged_low
        self.merged_last = entry_obj.merged_last
        self.merged_volume = entry_obj.merged_volume
        self.merged_previous_day_open_interest = entry_obj.merged_previous_day_open_interest
        self.adj_factor_settle = entry_obj.adj_factor_settle
        self.adj_factor_open = entry_obj.adj_factor_open
        self.adj_factor_high = entry_obj.adj_factor_high
        self.adj_factor_low = entry_obj.adj_factor_low
        self.adj_factor_last = entry_obj.adj_factor_last
        self.adj_factor_volume = entry_obj.adj_factor_volume
        self.adj_factor_previous_day_open_interest = entry_obj.adj_factor_previous_day_open_interest
        self.NEW_DATE = entry_obj.NEW_DATE


class EurodollarYieldCurve(Base):
    __tablename__ = EURODOLLAR_YIELD_CURVE_TABLE
    Id = Column("dt_Id", DateTime, primary_key=True)
    ED1 = Column("d_ED1", DECIMAL)
    ED2 = Column("d_ED2", DECIMAL)
    ED3 = Column("d_ED3", DECIMAL)
    ED4 = Column("d_ED4", DECIMAL)
    ED5 = Column("d_ED5", DECIMAL)
    ED6 = Column("d_ED6", DECIMAL)
    ED7 = Column("d_ED7", DECIMAL)
    ED8 = Column("d_ED8", DECIMAL)
    ED9 = Column("d_ED9", DECIMAL)
    ED10 = Column("d_ED10", DECIMAL)
    ED11 = Column("d_ED11", DECIMAL)
    ED12 = Column("d_ED12", DECIMAL)
    ED13 = Column("d_ED13", DECIMAL)
    ED14 = Column("d_ED14", DECIMAL)
    ED15 = Column("d_ED15", DECIMAL)
    ED16 = Column("d_ED16", DECIMAL)
    ED17 = Column("d_ED17", DECIMAL)
    ED18 = Column("d_ED18", DECIMAL)
    ED19 = Column("d_ED19", DECIMAL)
    ED20 = Column("d_ED20", DECIMAL)
    ED21 = Column("d_ED21", DECIMAL)
    ED22 = Column("d_ED22", DECIMAL)
    ED23 = Column("d_ED23", DECIMAL)
    ED24 = Column("d_ED24", DECIMAL)
    NEW_DATE = Column("d_NEW_DATE", DateTime)

    def __repr__(self):
        return "<eurodollar_yc(Id='%s', ED1='%s', ED2='%s', ED3='%s'," \
               "ED4='%s', ED5='%s', ED6='%s', ED7='%s'," \
               "ED8='%s', ED9='%s', ED10='%s', ED11='%s'," \
               "ED12='%s', ED13='%s', ED14='%s', ED15='%s', " \
               "ED16='%s', ED17='%s', ED18='%s', ED19='%s', ED20='%s', " \
               "ED21='%s', ED22='%s', ED23='%s', ED24='%s', NEW_DATE='%s')>" % (
                   self.Id, self.ED1, self.ED2, self.ED3, self.ED4, self.ED5, self.ED6, self.ED7,
                   self.ED8, self.ED9, self.ED10, self.ED11, self.ED12, self.ED13, self.ED14,
                   self.ED15, self.ED16, self.ED17, self.ED18, self.ED19, self.ED20, self.ED21,
                   self.ED22, self.ED23, self.ED24, self.NEW_DATE)

    def equals(self, yc_entry_obj):
        id_check = self.Id == yc_entry_obj.Id
        Ed1_check = self.ED1 == yc_entry_obj.ED1
        Ed2_check = self.ED2 == yc_entry_obj.ED2
        Ed3_check = self.ED3 == yc_entry_obj.ED3
        Ed4_check = self.ED4 == yc_entry_obj.ED4
        Ed5_check = self.ED5 == yc_entry_obj.ED5
        Ed6_check = self.ED6 == yc_entry_obj.ED6
        Ed7_check = self.ED7 == yc_entry_obj.ED7
        Ed8_check = self.ED8 == yc_entry_obj.ED8
        Ed9_check = self.ED9 == yc_entry_obj.ED9
        Ed10_check = self.ED10 == yc_entry_obj.ED10
        Ed11_check = self.ED11 == yc_entry_obj.ED11
        Ed12_check = self.ED12 == yc_entry_obj.ED12
        Ed13_check = self.ED13 == yc_entry_obj.ED13
        Ed14_check = self.ED14 == yc_entry_obj.ED14
        Ed15_check = self.ED15 == yc_entry_obj.ED15
        Ed16_check = self.ED16 == yc_entry_obj.ED16
        Ed17_check = self.ED17 == yc_entry_obj.ED17
        Ed18_check = self.ED18 == yc_entry_obj.ED18
        Ed19_check = self.ED19 == yc_entry_obj.ED19
        Ed20_check = self.ED20 == yc_entry_obj.ED20
        Ed21_check = self.ED21 == yc_entry_obj.ED21
        Ed22_check = self.ED22 == yc_entry_obj.ED22
        Ed23_check = self.ED23 == yc_entry_obj.ED23
        Ed24_check = self.ED24 == yc_entry_obj.ED24
        NEW_DATE_check = self.NEW_DATE == yc_entry_obj.NEW_DATE
        total_bool = id_check & Ed1_check & Ed2_check & Ed3_check & Ed4_check & Ed5_check & Ed6_check & Ed7_check \
                     & Ed8_check & Ed9_check & Ed10_check & Ed11_check & Ed12_check & Ed13_check & Ed14_check \
                     & Ed15_check & Ed16_check & Ed17_check & Ed18_check & Ed19_check & Ed20_check & Ed21_check \
                     & Ed22_check & Ed23_check & Ed24_check & NEW_DATE_check
        return total_bool

    def set_all(self, yc_entry_obj):
        self.Id = yc_entry_obj.Id
        self.ED1 = yc_entry_obj.ED1
        self.ED2 = yc_entry_obj.ED2
        self.ED3 = yc_entry_obj.ED3
        self.ED4 = yc_entry_obj.ED4
        self.ED5 = yc_entry_obj.ED5
        self.ED6 = yc_entry_obj.ED6
        self.ED7 = yc_entry_obj.ED7
        self.ED8 = yc_entry_obj.ED8
        self.ED9 = yc_entry_obj.ED9
        self.ED10 = yc_entry_obj.ED10
        self.ED11 = yc_entry_obj.ED11
        self.ED12 = yc_entry_obj.ED12
        self.ED13 = yc_entry_obj.ED13
        self.ED14 = yc_entry_obj.ED14
        self.ED15 = yc_entry_obj.ED15
        self.ED16 = yc_entry_obj.ED16
        self.ED17 = yc_entry_obj.ED17
        self.ED18 = yc_entry_obj.ED18
        self.ED19 = yc_entry_obj.ED19
        self.ED20 = yc_entry_obj.ED20
        self.ED21 = yc_entry_obj.ED21
        self.ED22 = yc_entry_obj.ED22
        self.ED23 = yc_entry_obj.ED23
        self.ED24 = yc_entry_obj.ED24
        self.NEW_DATE = yc_entry_obj.NEW_DATE


class GraphicalTimeSeries:
    POINT_VALUE_DICT = {'ES': 1.0,
                        'NQ': 1.0,
                        'YM': 1.0,
                        'EC': 0.0001,
                        'JY': 1.0,
                        'AD': 0.0001,
                        'CD': 0.0001,
                        'BP': 0.0001,
                        'SF': 0.0001,
                        'CL': 1.0,
                        'ED': 0.01,
                        'VX': 0.01,
                        'SPX': 1.0,
                        'VIX': 0.01}
    SIG_DIGITS_DICT = {'ES': 2,
                       'NQ': 2,
                       'YM': 2,
                       'EC': 5,
                       'JY': 1,
                       'AD': 4,
                       'CD': 5,
                       'BP': 4,
                       'SF': 4,
                       'CL': 2,
                       'ED': 3,
                       'VX': 2,
                       'SPX': 2,
                       'VIX': 2}

    def __init__(self,
                 plot_data=None,
                 ts_px_type="Settle"):
        # plot_data is a dictionary of Dataframes with symbol as key
        if plot_data is not None:
            self.plot_data = plot_data
        else:
            self.plot_data = dict()
        self.ts_px_type = ts_px_type
        self.line_color_choices = {0: "pink",
                                   1: "blue",
                                   2: 'green',
                                   3: 'black',
                                   4: 'red',
                                   5: 'orange',
                                   6: 'yellow',
                                   7: 'olive',
                                   8: 'purple',
                                   9: 'coral',
                                   10: 'gold',
                                   11: 'magenta'}

    def set_curve_futures_plot_data_mysql(self,
                                          contract_sym,
                                          contract_exp,
                                          assetType='WTICrudeOilCurve'):
        assetType_Dict = {"WTICrudeOilCurve": WTICrudeOilCurve,
                          "EuroDollarYieldCurve": EurodollarYieldCurve,
                          "VixCurve": VixCurve}
        cexp_list = [getattr(assetType_Dict[assetType], contract_sym + str(cexp)) for cexp in contract_exp]
        session = get_db_session()
        session.flush()
        cexp_list.append(assetType_Dict[assetType].Id)
        merged_settle_query = session.query(assetType_Dict[assetType]).with_entities(*cexp_list)
        df_px = pd.read_sql(merged_settle_query.statement, session.bind, index_col='dt_Id')
        df_px.index.rename('Date', inplace=True)
        new_col_nms = [col_nm + '_Settle' for col_nm in df_px.columns]
        df_px.columns = new_col_nms
        self.plot_data[contract_sym] = df_px
        return self.plot_data

    def set_vix_term_struct_plot_data_mysql(self,
                                            contract_sym_dict,
                                            assetType="VixTermStructure",
                                            start_date_query=datetime(2017, 1, 1).date(),
                                            end_date_query=datetime.now().date(),
                                            px_type='LastSale'):
        from root.nested.dataAccess.WebApi.cboe import VixTermStructure
        LOGGER.info("GraphicalTimeSeries.set_vix_term_struct_plot_data_mysql(%s, %s): running...",
                    str(contract_sym_dict.keys()), assetType)
        assetType_Dict = {"VixTermStructure": VixTermStructure
                          }
        session = get_db_session()
        session.flush()
        total_df = pd.DataFrame()
        for contract_sym in contract_sym_dict.keys():
            the_query = session.query(assetType_Dict[assetType]).filter_by(Symbol=contract_sym). \
                filter(getattr(assetType_Dict[assetType], "LastTime") > start_date_query). \
                filter(getattr(assetType_Dict[assetType], "LastTime") < end_date_query). \
                with_entities(getattr(assetType_Dict[assetType], px_type), assetType_Dict[assetType].Symbol,
                              assetType_Dict[assetType].Id)
            df_px = pd.read_sql(the_query.statement, session.bind, index_col='dt_Id')
            df_px.columns = [contract_sym + '_' + px_type, 'Symbol']
            df_px.drop(['Symbol'], axis=1, inplace=True)
            df_px.index.rename('Date', inplace=True)
            df_px[contract_sym + '_Plot_Axis'] = contract_sym_dict[contract_sym]
            self.plot_data[contract_sym] = df_px
            if total_df.empty:
                total_df = df_px
            else:
                total_df = pd.merge(left=total_df, right=df_px, left_index=True, right_index=True)
        total_df.index.rename('Date', inplace=True)
        self.plot_data['Multi'] = total_df
        return self.plot_data

    def set_equity_futures_plot_data_mysql(self,
                                           contract_sym,
                                           roll_type='All',
                                           include_front_month=True,
                                           assetType="EquityFutures"):
        LOGGER.info("GraphicalTimeSeries.set_equity_futures_plot_data_mysql(%s, %s, %s): running...",
                    str(contract_sym), str(roll_type), str(include_front_month))
        session = get_db_session()
        session.flush()
        df_flat_px = pd.DataFrame()
        df_prop_px = pd.DataFrame()
        assetType_Dict = {"EquityFutures": EquityFutures,
                          "CurrencyFutures": CurrencyFutures,
                          }
        if roll_type == 'flatprice' or roll_type == 'All':
            if not include_front_month:
                flat_price_merged_settle_query = session.query(assetType_Dict[assetType]). \
                    filter_by(symbol=contract_sym).filter_by(roll_type="flatprice"). \
                    with_entities(assetType_Dict[assetType].merged_settle, assetType_Dict[assetType].Id)
                df_flat_px = pd.read_sql(flat_price_merged_settle_query.statement, session.bind, index_col='dt_Id')
                df_flat_px.columns = [str.upper(contract_sym) + '_' + str.capitalize(self.ts_px_type) +
                                      '_Flat Price Roll']
            else:
                flat_price_merged_settle_query = session.query(assetType_Dict[assetType]). \
                    filter_by(symbol=contract_sym).filter_by(roll_type="flatprice"). \
                    with_entities(assetType_Dict[assetType].merged_settle, assetType_Dict[assetType].Id,
                                  assetType_Dict[assetType].near_month_settle)
                df_flat_px = pd.read_sql(flat_price_merged_settle_query.statement, session.bind, index_col='dt_Id')
                df_flat_px['Price_Gap_Rolled_vs_Front_Month'] = \
                    df_flat_px['d_MERGED_' + str.upper(self.ts_px_type)] - \
                    df_flat_px['d_NEAR_MONTH_' + str.upper(self.ts_px_type)]
                rename_dict = {"d_MERGED_" + str.upper(self.ts_px_type):
                                   str.upper(contract_sym) + '_' + str.capitalize(self.ts_px_type) +
                                   '_Flat_Price_Roll',
                               "d_NEAR_MONTH_" + str.upper(self.ts_px_type):
                                   str.upper(contract_sym) + '_' + str.capitalize(self.ts_px_type) +
                                   '_Front_Month'}
                df_flat_px.rename(columns=rename_dict, inplace=True)
        if roll_type == 'proportional' or roll_type == 'All':
            if not include_front_month:
                prop_price_merged_settle_query = session.query(assetType_Dict[assetType]). \
                    filter_by(symbol=contract_sym).filter_by(roll_type="proportional"). \
                    with_entities(assetType_Dict[assetType].merged_settle, assetType_Dict[assetType].Id)
                df_prop_px = pd.read_sql(prop_price_merged_settle_query.statement, session.bind, index_col='dt_Id')
                df_prop_px.columns = [str.upper(contract_sym) + '_' + str.capitalize(self.ts_px_type) +
                                      '_Proportional_Price_Roll']
            else:
                prop_price_merged_settle_query = session.query(assetType_Dict[assetType]). \
                    filter_by(symbol=contract_sym).filter_by(roll_type="proportional"). \
                    with_entities(assetType_Dict[assetType].merged_settle, assetType_Dict[assetType].Id,
                                  assetType_Dict[assetType].near_month_settle)
                df_prop_px = pd.read_sql(prop_price_merged_settle_query.statement, session.bind, index_col='dt_Id')
                df_prop_px['Price_Gap_Rolled_vs_Front_Month'] = \
                    df_prop_px['d_MERGED_' + str.upper(self.ts_px_type)] - \
                    df_prop_px['d_NEAR_MONTH_' + str.upper(self.ts_px_type)]
                rename_dict = {"d_MERGED_" + str.upper(self.ts_px_type):
                                   str.upper(contract_sym) + '_' + str.capitalize(self.ts_px_type) +
                                   '_Proportional_Price_Roll',
                               "d_NEAR_MONTH_" + str.upper(self.ts_px_type):
                                   str.upper(contract_sym) + '_' + str.capitalize(self.ts_px_type) +
                                   '_Front_Month'}
                df_prop_px.rename(columns=rename_dict, inplace=True)
        if not df_flat_px.empty and not df_prop_px.empty:
            total_df = pd.merge(left=df_flat_px, right=df_prop_px, left_index=True, right_index=True)
            self.plot_data[contract_sym] = total_df
        elif not df_flat_px.empty:
            df_flat_px.index.rename('Date', inplace=True)
            self.plot_data[contract_sym] = df_flat_px
        elif not df_prop_px.empty:
            df_prop_px.index.rename('Date', inplace=True)
            self.plot_data[contract_sym] = df_prop_px
        return self.plot_data

    @staticmethod
    def line_plot_bokeh(self):
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        p = figure(title="Year-wise total number of crimes", y_axis_type="linear", plot_height=400,
                   tools=TOOLS, plot_width=800)
        p.xaxis.axis_label = 'Year'
        p.yaxis.axis_label = 'Total Crimes'
        p.circle(2010, self.plot_data.IncidntNum.min(), size=10, color='red')

        p.line(self.plot_data.Year, self.plot_data.IncidntNum, line_color="purple", line_width=3)
        p.select_one(HoverTool).tooltips = [
            ('year', '@x'),
            ('Number of crimes', '@y'),
        ]
        output_file("line_chart.html", title="Line Chart")
        show(p)

    def multiple_line_plot_bokeh(self, combined_plot=False):
        LOGGER.info("GraphicalTimeSeries.multiple_line_plot_bokeh(): running...")
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        for contract_sym in sorted(set(self.plot_data.keys())):
            if combined_plot is True:
                if contract_sym != "Multi":
                    continue
            try:
                sig_digits = str(GraphicalTimeSeries.SIG_DIGITS_DICT[contract_sym])
            except KeyError:
                sig_digits = 2
            plot_title = "Continuous Futures Price: " + contract_sym + " Contract"
            p = figure(title=plot_title, y_axis_type="linear", x_axis_type='datetime',
                       tools=TOOLS)
            the_data = self.plot_data[contract_sym]
            source = ColumnDataSource(the_data)
            px_type_plot_list = []
            secondary_data_plot_list = []
            color_choice_idx = None
            for col_header in the_data.columns:
                if col_header.find('_Plot_Axis') != -1:
                    which_axis = the_data[col_header].iloc[0]
                    if which_axis == "primary_axis":
                        the_sym = col_header.split('_')[0]
                        for ch in the_data.columns:
                            if ch.find('_Plot_Axis') == -1 and ch.find(the_sym) != -1:
                                px_type_plot_list.append(ch)
                    elif which_axis == 'secondary_axis':
                        the_sym = col_header.split('_')[0]
                        for ch in the_data.columns:
                            if ch.find('_Plot_Axis') == -1 and ch.find(the_sym) != -1:
                                secondary_data_plot_list.append(ch)
            if len(px_type_plot_list) == 0 and len(secondary_data_plot_list) == 0:
                for col_header in the_data.columns:
                    if self.ts_px_type in col_header:
                        px_type_plot_list.append(col_header)
                    else:
                        # secondary data
                        secondary_data_plot_list.append(col_header)
            if max(len(px_type_plot_list), len(secondary_data_plot_list)) <= len(self.line_color_choices):
                color_choice_idx = random.sample(range(len(self.line_color_choices)),
                                                 max(len(px_type_plot_list), len(secondary_data_plot_list)))
            else:
                the_max_len_list = max(len(px_type_plot_list), len(secondary_data_plot_list))
                quo = int(the_max_len_list / len(self.line_color_choices))
                mod = the_max_len_list % len(self.line_color_choices)
                for i in range(quo):
                    if color_choice_idx is None:
                        color_choice_idx = random.sample(
                            range(len(self.line_color_choices)), len(self.line_color_choices))
                    else:
                        color_choice_idx.extend(
                            random.sample(len(self.line_color_choices), len(self.line_color_choices)))
                color_choice_idx.extend(random.sample(len(self.line_color_choices, mod)))
            idx_cnt = 0
            p.select_one(HoverTool).tooltips = [
                ('Date', '@Date{%F}'),
            ]
            p.select_one(HoverTool).formatters = {'@Date': 'datetime'}
            name_cnt = 0
            line_data_min = 10000000.0
            line_data_max = -10000000.0
            for line_name in px_type_plot_list:
                line_data = the_data[line_name]
                LOGGER.info("cont_fut.multiple_line_plot_bokeh(): plotting line data %s in color %s", line_name,
                            self.line_color_choices[color_choice_idx[idx_cnt]])
                if line_data.min() < line_data_min:
                    line_data_min = line_data.min()
                if line_data.max() > line_data_max:
                    line_data_max = line_data.max()
                p.line(x='Date', y=line_name, legend_label=str(px_type_plot_list[name_cnt]), source=source,
                       line_color=self.line_color_choices[color_choice_idx[idx_cnt]], line_width=3)
                idx_cnt += 1
                p.select_one(HoverTool).tooltips.append(
                    (line_name, '@' + line_name + '{%0.' + str(sig_digits) + 'f}'),
                )
                p.select_one(HoverTool).formatters['@' + line_name] = 'printf'
                name_cnt += 1
            try:
                point_value = GraphicalTimeSeries.POINT_VALUE_DICT[contract_sym]
            except KeyError:
                point_value = 1.0
            p.y_range = Range1d(start=line_data_min - point_value * 10.0,
                                end=line_data_max + point_value * 10.0)
            name_cnt = 0
            idx_cnt = 0
            circle_data_min = 10000000.0
            circle_data_max = -10000000.0
            for circle_name in secondary_data_plot_list:
                circle_data = the_data[circle_name]
                if circle_data.min() < circle_data_min:
                    circle_data_min = circle_data.min()
                if circle_data.max() > circle_data_max:
                    circle_data_max = circle_data.max()
                p.line(x='Date', y=circle_name, legend_label=str(secondary_data_plot_list[name_cnt]), source=source,
                       color=self.line_color_choices[color_choice_idx[idx_cnt]], y_range_name="foo")
                idx_cnt += 1
                p.select_one(HoverTool).tooltips.append(
                    (circle_name, '@' + circle_name + '{%0.' + str(sig_digits) + 'f}'))
                p.select_one(HoverTool).formatters['@' + circle_name] = 'printf'
                name_cnt += 1
            try:
                point_value = GraphicalTimeSeries.POINT_VALUE_DICT[contract_sym]
            except KeyError:
                point_value = 1.0
            p.add_layout(LinearAxis(y_range_name="foo"), 'right')
            p.extra_y_ranges = {"foo": Range1d(start=circle_data_min -
                                                     point_value * 10.0,
                                               end=circle_data_max +
                                                   point_value * 10.0)}
            p.select_one(HoverTool).mode = 'vline'
            p.legend.location = "bottom_left"
            p.xaxis.axis_label = 'Trade Date'
            p.yaxis.axis_label = 'Price ($)'
            output_file(contract_sym + "_ContinuousFuturesPrice.html", title="Multi Line Plot")
            show(p)


if __name__ == "__main__":
    #gts = GraphicalTimeSeries()
    # gts.set_equity_futures_plot_data_mysql(contract_sym='ES', roll_type='flatprice')
    # gts.multiple_line_plot_bokeh()
    # gts.set_equity_futures_plot_data_mysql(contract_sym='NQ', roll_type='proportional')
    # gts.multiple_line_plot_bokeh()
    # gts.set_curve_futures_plot_data_mysql(contract_sym='CL', contract_exp=[2], assetType="WTICrudeOilCurve")
    # gts.set_curve_futures_plot_data_mysql(contract_sym='ED', contract_exp=[5,9], assetType="EuroDollarYieldCurve")
    # gts.multiple_line_plot_bokeh()
    # gts.set_curve_futures_plot_data_mysql(contract_sym='VX', contract_exp=[3,5], assetType="VixCurve")
    # gts.multiple_line_plot_bokeh()
    # gts.set_equity_futures_plot_data_mysql(contract_sym='CL1', roll_type='proportional', assetType="WTICrudeOilCurve")
    # gts.multiple_line_plot_bokeh()
    #gts.set_vix_term_struct_plot_data_mysql(contract_sym_dict={'SPX': 'primary_axis',
    #                                                           'VIX': 'secondary_axis',
    #                                                           'VVIX': 'secondary_axis'})
    #gts.multiple_line_plot_bokeh(combined_plot=True)
    LOGGER.info("cont_fut.__main__: running main()...")
    # get_vx_data_from_cboe_csv(missing_dates_range=None)
    # unadjusted_csv_file = vx_continuous_contract_creation()
    # list_cont_contract_dfs = vx_roll_adjust_from_csv(unadj_csv_file=unadjusted_csv_file)
    # single_df_cont_contracts, csv_file_cont_contracts = merge_vx_futs_dataframes(list_cont_contract_dfs)
    # vx_curve_to_db(single_df_cont_contracts)
    # quandl.get("CHRIS/CME_JY1") JY Futures 1st Expiration Continuous
    # quandl.get("CHRIS/CME_JY2") JY Futures 2nd Expiration Continuous
    # quandl.get("CHRIS/CME_EC1") EC Futures 1st Expiration Continuous
    # quandl.get("CHRIS/CME_EC2") EC Futures 2nd Expiration Continuous
    # quandl.get("CHRIS/CME_AD1") AD Futures 1st Expiration Continuous
    # quandl.get("CHRIS/CME_AD2") AD Futures 2nd Expiration Continuous
    # quandl.get("CHRIS/CME_CD1") CD Futures 1st Expiration Continuous
    # quandl.get("CHRIS/CME_CD2") CD Futures 2nd Expiration Continuous
    # quandl.get("CHRIS/CME_BP1") BP Futures 1st Expiration Continuous
    # quandl.get("CHRIS/CME_BP2") BP Futures 2nd Expiration Continuous
    # quandl.get("CHRIS/CME_SF1") SF Futures 1st Expiration Continuous
    # quandl.get("CHRIS/CME_SF2") SF Futures 2nd Expiration Continuous
