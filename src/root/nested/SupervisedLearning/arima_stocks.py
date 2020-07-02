import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# the df below should have stock price data from somewhere.
df = pd.read_csv('gs://cloud-training/ai4f/AAPL10Y.csv')

#df.resample()

df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df.set_index('date', inplace=True)

print(df.shape)

df.head()
# resample by average the daily prices per week
df_week = df.resample('W').mean()
df_week = df_week[['close']]
df_week.head()
# then do the weekly log returns
df_week['weekly_ret'] = np.log(df_week['close']).diff()
df_week.head()
df_week.dropna(inplace=True)
df_week.weekly_ret.plot(kind='line', figsize=(12,6))
udiff = df_week.drop(['close'], axis=1)
udiff.head()

rolmean = udiff.rolling(20).mean()
rolstd = udiff.rolling(20).std()
plt.figure(figsize=(12, 6))
orig = plt.plot(udiff, color='blue', label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std Deviation')
plt.title('Rolling Mean & Standard Deviation')
plt.legend(loc='best')
plt.show(block=True)

# Perform Dickey-Fuller test
dftest = sm.tsa.adfuller(udiff.weekly_ret, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value ({0})'.format(key)] = value

# look at the dfoutput here... you can reject the null hypothesis
# if the p-value is less than 0.05. The null hypothesis is that
# the data is not stationary.
# From Wikipedia: In statistics, the Dickey–Fuller test tests
# the null hypothesis that a unit root is present in an
# autoregressive model. The alternative hypothesis is
# different depending on which version of the test is used,
# but is usually stationarity or trend-stationarity.

from statsmodels.graphics.tsaplots import plot_acf

# the autocorrelation chart provides just the correlation
# at increasing lags
# see where we exceed the p-value th by the most.
# this is the q  value for the MA parameter
fig, ax = plt.subplots(figsize=(12,5))
plot_acf(udiff.values, lags=10, ax=ax)
plt.show()

from statsmodels.graphics.tsaplots import plot_pacf

# partial autocorrelation, use this to find the AR param
# see where the the last time we exceed the p-value th
fig, ax = plt.subplots(figsize=(12,5))
plot_pacf(udiff.values, lags=10, ax=ax)
plt.show()

from statsmodels.tsa.arima_model import ARMA

# Notice that you have to use udiff - the differenced data rather than the original data.
# udiff is an np.array. They wrap it in a tuple(), I don't
# think that is neccessary.
ar1 = ARMA(udiff.values, (3,1)).fit()# TODO: Fit an ARMA model to the differenced data
ar1.summary()

# the above creates the model ARMA, p =3, q=1

# next we want to plot the fitted values from the ARMA
# model against the actual values in udiff
# the udiff values are the log return values.
plt.figure(figsize=(12,8))
plt.plot(udiff.values, color='blue')
preds=ar1.fittedvalues
plt.plot(preds, color='red')
plt.show()

# next, let's make a 2 step ahead forecast, and plot it
steps = 2
forecast = ar1.forecast(steps=steps)[0]
plt.figure(figsize=(12,8))
plt.plot(udiff.values, color='blue')
preds = ar1.fittedvalues
plt.plot(preds, color='red')
plt.plot(pd.DataFrame(np.array([preds[-1], forecast[0]]).T, index=range(len(udiff.values)+1, len(udiff.values)+3)), color='green')
plt.plot(pd.DataFrame(forecast, index=range(len(udiff.values)+1, len(udiff.values)+1+steps)), color='green')
plt.title('Display the predictions with the ARMA model')
plt.show()