import numpy as np
import pandas as pd

import statsmodels
from statsmodels.tsa.stattools import coint

# just set the seed for the random number generator
np.random.seed(107)

import matplotlib.pyplot as plt

# Generate the daily returns
Xreturns = np.random.normal(0, 1, 100)
# sum them and shift all the prices up
X = pd.Series(np.cumsum(Xreturns), name='X') + 50
X.plot(figsize=(15, 7))
plt.show()

noise = np.random.normal(0, 1, 100)
Y = X + 5 + noise
Y.name = 'Y'
pd.concat([X, Y], axis=1).plot(figsize=(15, 7))
plt.show()

(Y / X).plot(figsize=(15, 7))
plt.axhline((Y / X).mean(), color='red', linestyle='--')
plt.xlabel('Time')
plt.legend(['Price Ratio', 'Mean'])
plt.show()

# compute the p-value of the cointegration test
# will inform us as to whether the ratio between the 2 timeseries is stationary
# around its mean
score, pvalue, _ = coint(X, Y)


def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


from backtester.dataSource.yahoo_data_source import YahooStockDataSource
from datetime import datetime

startDateStr = '2007/12/01'
endDateStr = '2017/12/01'
cachedFolderName = 'yahooData/'
dataSetId = 'testPairsTrading'
instrumentIds = ['SPY', 'AAPL', 'ADBE', 'T', 'EBAY', 'MSFT', 'QCOM',
                 'HPQ', 'JNPR', 'AMD', 'IBM']
ds = YahooStockDataSource(cachedFolderName=cachedFolderName,
                          dataSetId=dataSetId,
                          instrumentIds=instrumentIds,
                          startDateStr=startDateStr,
                          endDateStr=endDateStr,
                          event='history')
data = ds.getBookDataByFeature()['adjClose']

# Heatmap to show the p-values of the cointegration test
# between each pair of stocks

scores, pvalues, pairs = find_cointegrated_pairs(data)
import seaborn

m = [0, 0.2, 0.4, 0.6, 0.8, 1]
seaborn.heatmap(pvalues, xticklabels=instrumentIds,
                yticklabels=instrumentIds, cmap='RdYlGn_r'
                , mask=(pvalues >= 0.98)
                )
plt.show()
print(pairs)

S1 = data['ADBE']
S2 = data['MSFT']
score, pvalue, _ = coint(S1, S2)
print(pvalue)
ratios = S1 / S2
ratios.plot(figsize=(15, 7))
plt.axhline(ratios.mean())
plt.legend(['Price Ratio'])
plt.show()


def zscore(series):
    return (series - series.mean()) / np.std(series)


zscore(ratios).plot(figsize=(15, 7))
plt.axhline(zscore(ratios).mean(), color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Ratio z-score', 'Mean', '+1', '-1'])
plt.show()

ratios = data['ADBE'] / data['MSFT']
print(len(ratios))
train = ratios[:1762]
test = ratios[1762:]

ratios_mavg5 = train.rolling(window=5,
                             center=False).mean()

ratios_mavg60 = train.rolling(window=60,
                              center=False).mean()

std_60 = train.rolling(window=60,
                       center=False).std()

zscore_60_5 = (ratios_mavg5 - ratios_mavg60) / std_60
plt.figure(figsize=(15, 7))
plt.plot(train.index, train.values)
plt.plot(ratios_mavg5.index, ratios_mavg5.values)
plt.plot(ratios_mavg60.index, ratios_mavg60.values)

plt.legend(['Ratio', '5d Ratio MA', '60d Ratio MA'])

plt.ylabel('Ratio')
plt.show()

# Take a rolling 60 day standard deviation
std_60 = train.rolling(window=60, center=False).std()
std_60.name = 'std 60d'

# Compute the z score for each day
zscore_60_5 = (ratios_mavg5 - ratios_mavg60) / std_60
zscore_60_5.name = 'z-score'

plt.figure(figsize=(15, 7))
zscore_60_5.plot()
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
plt.show()

# Plot the ratios and buy and sell signals from z score
plt.figure(figsize=(15, 7))

train[60:].plot()
buy = train.copy()
sell = train.copy()
buy[zscore_60_5 > -1] = 0
sell[zscore_60_5 < 1] = 0
buy[60:].plot(color='g', linestyle='None', marker='^')
sell[60:].plot(color='r', linestyle='None', marker='^')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, ratios.min(), ratios.max()))
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()

# Plot the prices and buy and sell signals from z score
plt.figure(figsize=(18, 9))
S1 = data['ADBE'].iloc[:1762]
S2 = data['MSFT'].iloc[:1762]

S1[60:].plot(color='b')
S2[60:].plot(color='c')
buyR = 0 * S1.copy()
sellR = 0 * S1.copy()

# When buying the ratio, buy S1 and sell S2
buyR[buy != 0] = S1[buy != 0]
sellR[buy != 0] = S2[buy != 0]
# When selling the ratio, sell S1 and buy S2
buyR[sell != 0] = S2[sell != 0]
sellR[sell != 0] = S1[sell != 0]

buyR[60:].plot(color='g', linestyle='None', marker='^')
sellR[60:].plot(color='r', linestyle='None', marker='^')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, min(S1.min(), S2.min()), max(S1.max(), S2.max())))

plt.legend(['ADBE', 'MSFT', 'Buy Signal', 'Sell Signal'])
plt.show()


# Trade using a simple strategy
def trade(S1, S2, window1, window2):
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0

    # Compute rolling mean and rolling standard deviation
    ratios = S1 / S2
    ma1 = ratios.rolling(window=window1,
                         center=False).mean()
    ma2 = ratios.rolling(window=window2,
                         center=False).mean()
    std = ratios.rolling(window=window2,
                         center=False).std()
    zscore = (ma1 - ma2) / std

    # Simulate trading
    # Start with no money and no positions
    money = 0
    countS1 = 0
    countS2 = 0
    for i in range(len(ratios)):
        # Sell short if the z-score is > 1
        if zscore[i] > 1:
            money += S1[i] - S2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]
        # Buy long if the z-score is < 1
        elif zscore[i] < -1:
            money -= S1[i] - S2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.5:
            money += countS1 * S1[i] - S2[i] * countS2
            count = 0

    return money


trade(data['ADBE'].iloc[:1762], data['MSFT'].iloc[:1762], 60, 5)

trade(data['ADBE'].iloc[1762:], data['MSFT'].iloc[1762:], 60, 5)

# Find the window length 0-254
# that gives the highest returns using this strategy
length_scores = [trade(data['ADBE'].iloc[:1762],
                       data['MSFT'].iloc[:1762], l, 5)
                 for l in range(255)]
best_length = np.argmax(length_scores)
print('Best window length:', best_length)

# Find the returns for test data
# using what we think is the best window length
length_scores2 = [trade(data['ADBE'].iloc[1762:],
                        data['MSFT'].iloc[1762:], l, 5)
                  for l in range(255)]
print(best_length, 'day window:', length_scores2[best_length])

# Find the best window length based on this dataset,
# and the returns using this window length
best_length2 = np.argmax(length_scores2)
print(best_length2, 'day window:', length_scores2[best_length2])

plt.figure(figsize=(15, 7))
plt.plot(length_scores)
plt.plot(length_scores2)
plt.xlabel('Window length')
plt.ylabel('Score')
plt.legend(['Training', 'Test'])
plt.show()
