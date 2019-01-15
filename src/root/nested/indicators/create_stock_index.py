import pandas as pd
import matplotlib.pyplot as plt
import pyEX as pyex
from sklearn import preprocessing

from root.nested.dataAccess.data_provider_interface import DataProviderInterface
from root.nested.statisticalAnalysis.statistical_moments import StatisticalMoments
from root.nested.dataAccess.WebApi.shares_outstanding_history import SharesOutstandingHistory
from root.nested import get_logger

LOGGER = get_logger()

#sp500_holdings_list = DataProviderInterface.download_sp500_holdings_symbol_list()
dow30_holdings_list = DataProviderInterface.download_dow30_holdings_symbol_list()

returned_df_list = []

def pyex_get_co_info(sym):

    #co_info = pyex.companyDF(sym)
    #key_stats = pyex.stockStatsDF(sym)
    stock_quote = pyex.quoteDF(sym)
    # the pyex.price() function returns a delayed price, must use pyex.last for real time
    # but pyex.last() returns empty when market is closed.
    last_price_df = pd.DataFrame({'symbol':[sym], 'lastprice': [pyex.price(sym)]})
    last_price_df.set_index('symbol', inplace=True)
    #co_info = pd.concat([co_info, key_stats, stock_quote, last_price_df], axis = 1)
    co_info = pd.concat([stock_quote, last_price_df], axis = 1)
    return co_info

coInfo_df_list = list(map(lambda sym: pyex_get_co_info(sym), dow30_holdings_list))
listings = pd.concat(coInfo_df_list)
LOGGER.info("create_stock_index: listings dataframe info: %s", listings.info())

# groupby sector
listings_sector_grouped = listings.groupby(by = ['sector'], sort = True).size().sort_values(ascending=False)
print (listings_sector_grouped)

# select the largest company for each sector
components = listings.groupby('sector')['marketCap'].nlargest(1)
tickers = components.index.get_level_values('symbol')
LOGGER.info("create_stock_index: list of symbols: %s", str(tickers))
# marketcap from key_stats IS NOT real time, marketCap from stock_quote IS
info_cols = ['companyName', 'marketCap', 'lastprice']
largest_co_in_sector_df = listings.loc[tickers, info_cols].sort_values('marketCap', ascending=False)
print (largest_co_in_sector_df)

# get stock returns for each of the above in "tickers" list
sm = StatisticalMoments()
daily_prices_raw = sm.get_pricing_df_ready(ticker=list(tickers), fields=['adjClose'], freq='D')
daily_prices_raw.columns = [col_name.split('_')[0] for col_name in daily_prices_raw.columns]
daily_prices = daily_prices_raw.dropna()
daily_returns = (daily_prices.iloc[-1]/daily_prices.iloc[0]).sub(1).mul(100)
daily_returns.sort_values().plot(kind='barh', title='Stock Price Returns')
plt.show()
print ("------------------")
print (daily_prices.head())
print ("------------------")
# calculate real time shares outstanding (using latest price and latest marketCap)
components = listings.loc[tickers, ['marketCap', 'lastprice']]
print (components.head(5))
no_shares = components['marketCap'].div(components['lastprice'])
print (no_shares.sort_values(ascending=False))

# time series of market value per stock
so_df_list = [(ticker, SharesOutstandingHistory.get_shares_outstanding_history(ticker)) for ticker in tickers]
min_max_scaler = preprocessing.MinMaxScaler()
for sym_key, so_df in so_df_list:
    print (so_df)
    shares_outstanding = so_df[sym_key+'_SharesOutstanding']
    merged_df = pd.concat([shares_outstanding, daily_prices_raw[sym_key]], axis = 1)
    #merged_df.fillna(method='bfill', inplace=True)
    merged_df.fillna(method='ffill', inplace=True) # fill recent with the data we have
    merged_df.dropna(inplace=True)
    merged_df['computedMarketCap'] = merged_df[sym_key + '_SharesOutstanding']*merged_df[sym_key]

    merged_df['computedMarketCap_normed'] = merged_df['computedMarketCap'].div(merged_df.iloc[0].computedMarketCap).mul(100.0)#.apply(lambda x: x/(x.iloc[0]).mul(100))
    merged_df[sym_key + '_normed'] = merged_df[sym_key].div(merged_df[sym_key].iloc[0]).mul(100.0)

    print (merged_df[[sym_key + '_normed', 'computedMarketCap_normed']].head())
    merged_df.plot(y=['computedMarketCap_normed', sym_key + '_normed'],
                   kind='line',
                   title = sym_key + ' Normalized Market Cap Time Series')
                   #secondary_y=[sym_key+'_normed'])

    plt.show()