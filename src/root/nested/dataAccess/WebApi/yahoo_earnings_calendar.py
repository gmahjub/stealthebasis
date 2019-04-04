import datetime
import json
import requests
import pandas as pd
from root.nested import get_logger


BASE_URL = 'https://finance.yahoo.com/calendar/earnings'
BASE_STOCK_URL = 'https://finance.yahoo.com/quote'
LOGGER = get_logger()

class YahooEarningsCalendar(object):

    def _get_data_dict(self, url):
        page = requests.get(url)
        page_content = page.content.decode(encoding='utf-8', errors='strict')
        page_data_string = [row for row in page_content.split('\n') if row.startswith('root.App.main = ')][0][:-1]
        page_data_string = page_data_string.split('root.App.main = ', 1)[1]
        return json.loads(page_data_string)

    def get_next_earnings_date(self, symbol):
        """Gets the next earnings date of symbol
        Args:
            symbol: A ticker symbol
        Returns:
            Unix timestamp of the next earnings date
        Raises:
            Exception: When symbol is invalid or earnings date is not available
        """
        url = '{0}/{1}'.format(BASE_STOCK_URL, symbol)
        LOGGER.info("YahooEarningsDate.get_next_earnings_date(): url is %s", url)
        try:
            page_data_dict = self._get_data_dict(url)
            timestamp_date = page_data_dict['context']['dispatcher']['stores']['QuoteSummaryStore']['calendarEvents']['earnings']['earningsDate'][0]['raw']
            datetime_obj = datetime.datetime.utcfromtimestamp(timestamp_date)
            return (datetime_obj)
        except:
            raise Exception('Invalid Symbol or Unavailable Earnings Date')

    def earnings_on(self, date):
        """Gets earnings calendar data from Yahoo! on a specific date.
        Args:
            date: A datetime.date instance representing the date of earnings data to be fetched.
        Returns:
            An array of earnings calendar data on date given. E.g.,
            [
                {
                    "ticker": "AMS.S",
                    "companyshortname": "Ams AG",
                    "startdatetime": "2017-04-23T20:00:00.000-04:00",
                    "startdatetimetype": "TAS",
                    "epsestimate": null,
                    "epsactual": null,
                    "epssurprisepct": null,
                    "gmtOffsetMilliSeconds": 72000000
                },
                ...
            ]
        Raises:
            TypeError: When date is not a datetime.date object.
        """
        if not isinstance(date, datetime.date):
            raise TypeError(
                'Date should be a datetime.date object')
        date_str = date.strftime('%Y-%m-%d')
        LOGGER.debug('YahooEarningsCalendar.earnings_on(): getting earnings data for date %s', date_str)
        dated_url = '{0}?day={1}'.format(BASE_URL, date_str)
        page_data_dict = self._get_data_dict(dated_url)
        return page_data_dict['context']['dispatcher']['stores']['ScreenerResultsStore']['results']['rows']

    def earnings_between(self, from_date, to_date):
        """Gets earnings calendar data from Yahoo! in a date range.
        Args:
            from_date: A datetime.date instance representing the from-date (inclusive).
            to_date: A datetime.date instance representing the to-date (inclusive).
        Returns:
            An array of earnigs calendar data of date range. E.g.,
            [
                {
                    "ticker": "AMS.S",
                    "companyshortname": "Ams AG",
                    "startdatetime": "2017-04-23T20:00:00.000-04:00",
                    "startdatetimetype": "TAS",
                    "epsestimate": null,
                    "epsactual": null,
                    "epssurprisepct": null,
                    "gmtOffsetMilliSeconds": 72000000
                },
                ...
            ]
        Raises:
            ValueError: When from_date is after to_date.
            TypeError: When either from_date or to_date is not a datetime.date object.
        """
        if from_date > to_date:
            raise ValueError(
                'From-date should not be after to-date')
        if not (isinstance(from_date, datetime.date) and
                isinstance(to_date, datetime.date)):
            raise TypeError(
                'From-date and to-date should be datetime.date objects')
        earnings_data = []
        current_date = from_date
        delta = datetime.timedelta(days=1)
        while current_date <= to_date:
            earnings_data += self.earnings_on(current_date)
            current_date += delta
        return earnings_data

    def create_earnings_df(self,
                           earnings_dict):

        df = pd.DataFrame.from_dict(earnings_dict)
        print (df.head())
        #print (df)

if __name__ == '__main__':
    date_from = datetime.datetime.strptime(
        'March 13, 2019 01:00AM', '%B %d, %Y %I:%M%p')
    #date_to = datetime.datetime.strptime(
    #    'May 8, 2017  1:00PM', '%b %d, %Y %I:%M%p')
    yec = YahooEarningsCalendar()
    values_dict = yec.earnings_on(date_from)
    # lets convert this dictionary to dataframe
    earnings_df = yec.create_earnings_df(values_dict)
    print (earnings_df)
    #print(yec.earnings_between(date_from, date_to))
    print(yec.get_next_earnings_date('EXPR'))