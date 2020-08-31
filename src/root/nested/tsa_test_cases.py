'''
Created on Dec 13, 2017

@author: ghazy
'''
import sys
import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model

import matplotlib.pyplot as plt

p = print

import platform

p('Machine: {} {}\n'.format(platform.system(), platform.machine()))
p(sys.version)


def tsplot(y, lags=None, figsize=(10, 8), style='bmh', title='Time Series Analysis Plots'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
        plt.tight_layout()
        plt.show()
    return


from root.nested.dataAccess.yahoo_data_object import YahooDataObject

end_date = str(pd.to_datetime('today')).split(' ')[0]
ydo_spy = YahooDataObject('2007-01-01',
                          end_date,
                          ['SPY'])
if ydo_spy.does_local_file_exist('AdjClose'):
    if ydo_spy.is_local_file_old():
        ydo_spy.logger.info("Pulling new dataframes from yahoo...")
        _ = ydo_spy.get_adjust_close_prices()
        _ = ydo_spy.get_high_prices()
        _ = ydo_spy.get_low_prices()
        _ = ydo_spy.get_open_prices()
    else:
        ydo_spy.logger.info("Pulling dataframes from local csv files...")
        _ = ydo_spy.adj_close_from_csv()
        _ = ydo_spy.open_from_csv()
        _ = ydo_spy.high_px_df = ydo_spy.high_from_csv()
        _ = ydo_spy.low_px_df = ydo_spy.low_from_csv()
else:
    _ = ydo_spy.get_adjust_close_prices()
    _ = ydo_spy.get_high_prices()
    _ = ydo_spy.get_low_prices()
    _ = ydo_spy.get_open_prices()

symbol_list = ['MSFT']
ydo_1 = YahooDataObject('2007-01-01',
                        '2017-12-01',
                        symbol_list)
if ydo_1.does_local_file_exist('AdjClose'):
    if ydo_1.is_local_file_old():
        ydo_1.logger.info("Pulling new dataframes from yahoo...")
        _ = ydo_1.get_adjust_close_prices()
        _ = ydo_1.get_high_prices()
        _ = ydo_1.get_low_prices()
        _ = ydo_1.get_open_prices()
    else:
        ydo_1.logger.info("Pulling dataframes from local csv files...")
        _ = ydo_1.adj_close_from_csv()
        _ = ydo_1.open_from_csv()
        _ = ydo_1.high_px_df = ydo_1.high_from_csv()
        _ = ydo_1.low_px_df = ydo_1.low_from_csv()
else:
    _ = ydo_1.get_adjust_close_prices()
    _ = ydo_1.get_high_prices()
    _ = ydo_1.get_low_prices()
    _ = ydo_1.get_open_prices()
# p(type(ydo.adj_close_px_df.SPY))
# p(type(np.diff(ydo.adj_close_px_df.SPY)))
lrets_spy = ydo_spy.calc_log_returns('AdjClose')

"""ARCH"""
## simulate an ARCH series
a0 = 2
a1 = 0.5
y = w = np.random.normal(size=1000)  # random numbers, Gaussian distributed
Y = np.empty_like(y)
for t in range(len(y)):
    Y[t] = w[t] + np.sqrt((a0 + a1 * y[t - 1] ** 2))
_ = tsplot(Y, lags=30)

"""GARCH"""
np.random.seed(2)
a0 = 0.2
a1 = 0.5
b1 = 0.3
n = 10000
w = np.random.normal(size=n)
eps = np.zeros_like(w)
sigsq = np.zeros_like(w)
for i in range(1, n):
    sigsq[i] = a0 + a1 * (eps[i - 1] ** 2) + b1 * sigsq[i - 1]
    eps[i] = w[i] * np.sqrt(sigsq[i])
_ = tsplot(eps, lags=30)

"""lets try and fit a GARCH(1,1) and recover above parameters """
am = arch_model(eps)
res = am.fit(update_freq=5)
p(res.summary)

### lets do ARIMA now, add a D term
best_aic = np.inf
best_model = None
best_order = None
pq_range = range(5)
d_range = range(1)
lrets_spy.columns = ['SPY']
lrets_spy.index = pd.DatetimeIndex(lrets_spy.index).to_period('D')
for i in pq_range:
    for j in pq_range:
        for d in d_range:
            try:
                temp_model = smt.ARIMA(lrets_spy.SPY, order=(i, d, j)).fit(method='mle', trend='nc')
                temp_aic = temp_model.aic
                if temp_aic < best_aic:
                    print('setting best model to ', type(temp_model), str(temp_model), 'aic is ', temp_aic)
                    best_aic = temp_aic
                    best_order = (i, d, j)
                    best_model = temp_model
            except Exception as e:
                print("the exception is ", e.__str__())
                continue
p('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
_ = tsplot(best_model.resid, lags=30, title="SPY Log Returns Fitted to ARIMA" + str(best_order))

"""Forecasting"""
## 21 day forecast
n_steps = 2
f, err95, ci95 = best_model.forecast(steps=n_steps)
_, err99, ci99 = best_model.forecast(steps=n_steps, alpha=0.01)  # alphas specifies the CI
data_df = ydo_spy.get_adj_close_px_df()
idx = pd.date_range(data_df.index[-1], periods=n_steps, freq='D')
fc_95 = pd.DataFrame(np.column_stack([f, ci95]), index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([ci99]), index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)
fc_all.head()
plt.style.use('bmh')
fig = plt.figure(figsize=(9, 7))
ax = plt.gca()
ts = lrets_spy.SPY.iloc[-500:].copy()
ts.plot(ax=ax, label='SPY returns')
# in sample prediction
pred = best_model.predict(ts.index[0], ts.index[-1])
pred.plot(ax=ax, style='r-', label='In-sample prediction')
styles = ['b-', '0.2', '0.75', '0.2', '0.75']
fc_all.plot(ax=ax, style=styles)
plt.fill_between(fc_all.index, fc_all.lower_ci_95, fc_all.upper_ci_95, color='gray', alpha=0.7)
plt.fill_between(fc_all.index, fc_all.lower_ci_99, fc_all.upper_ci_99, color='gray', alpha=0.2)
plt.title('{} Day SPY Return Forecast\nARIMA{}'.format(n_steps, best_order))
plt.legend(loc='best', fontsize=10)

## simulate ARMA(3,2) alphas = [0.5,-0.25,0.4], betas = [0.5,-0.3]
""" Alphas are the coefficients for the AR factors
    Betas are the coefficients for the MA factors """
max_lag = 30
n_samples = 5000
burn = int(n_samples / 10) * 4  # burnin parameter means just throw away the first "burn" samples
alphas = np.array([0.5, -0.25, 0.4])
betas = np.array([0.5, -0.3])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]
arma32 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n_samples, burnin=burn)
_ = tsplot(arma32, lags=max_lag)
best_aic = np.inf
best_order = None
best_model = None
rng = range(5)

for i in rng:
    for j in rng:
        try:
            tmp_model = smt.ARMA(arma32, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_model.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_model = tmp_model
        except:
            continue
p('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

### Lets fit this ARMA(3,2) model to SPY
best_aic = np.inf
best_order = None
best_model = None
rng = range(5)

for i in rng:
    for j in rng:
        try:
            p("testing order ", i, j)
            temp_model = smt.ARMA(lrets_spy.SPY, order=(i, j)).fit(method='mle', trend='nc')
            p(temp_model.summary())
            temp_aic = temp_model.aic
            if temp_aic < best_aic:
                best_aic = temp_aic
                best_order = (i, j)
                best_model = temp_model
        except:
            continue

_ = tsplot(best_model.resid, lags=30, title="SPY Log Returns Fitted to ARMA" + str(best_order))

### Original start of this sample code.
np.random.seed(1)
n_samples = 1000
# plot of descrete white noise
randser = np.random.normal(size=n_samples)
p("Random Series\n----------------------\nmean: {:.3f}\nvariance: {:.3f}\nstandard deviation: {:.3f}".format(
    randser.mean(), randser.var(), randser.std()))
_ = tsplot(randser, lags=30, title='Random Series')

# random walk without a drift
np.random.seed(56543678)
n_samples = 1000
x = w = np.random.normal(size=n_samples)
for t in range(n_samples):
    x[t] = x[t - 1] + w[t]
# we are making the random, normal, stationary distribution w into a non-staionary, time-dependant, random walk
p("Random Walk w/o Drift\n----------------------\nmean: {:.3f}\nvariance: {:.3f}\nstandard deviation: {:.3f}".format(
    x.mean(), x.var(), x.std()))
_ = tsplot(x, lags=30, title='Random Walk, No Drift')

# by way of algebra, we can see that w[t] = x[t] - x[t-1] --- the first difference therefore should be stationary
p(
    "1st Difference of random walk w/o drift series\n----------------------\nmean: {:.3f}\nvariance: {:.3f}\nstandard deviation: {:.3f}".format(
        np.diff(x).mean(), np.diff(x).var(), np.diff(x).std()))
p("\n")
_ = tsplot(np.diff(x), lags=30, title='First Diff of Random Walk, No Drift')

p("1st Difference of SPY\n----------------------\nmean: {:.3f}\nvariance: {:.3f}\nstandard deviation: {:.3f}".format(
    np.diff(ydo_spy.adj_close_px_df.SPY).mean(), np.diff(ydo_spy.adj_close_px_df.SPY).var(),
    np.diff(ydo_spy.adj_close_px_df.SPY).std()))
p("\n")
_ = tsplot(np.diff(ydo_spy.adj_close_px_df.SPY), lags=30, title='First Diff SPY')

p(
    "Log Returns " + "SPY" + "\n----------------------\nmean: {:.7f}\nvariance: {:.7f}\nstandard deviation: {:.7f}".format(
        lrets_spy.SPY.mean(), lrets_spy.SPY.var(), lrets_spy.SPY.std()))
p("\n")
_ = tsplot(lrets_spy.SPY, lags=30, title='Log Returns ' + 'SPY')

# linear model y = mx+b, where b is beta0 and m is Beta1
w = np.random.randn(100)  # generate 100 random normally distributed numbers
y = np.empty_like(w)  # just a new array with the same shape and type as w
b0 = -50.
b1 = 25.
for t in range(len(w)):
    y[t] = b0 + b1 * t + w[t]
p("Linear Model\n----------------------\nmean: {:.3f}\nvariance: {:.3f}\nstandard deviation: {:.3f}".format(y.mean(),
                                                                                                            y.var(),
                                                                                                            y.std()))
p("\n")
_ = tsplot(y, lags=30, title='Linear Model Simulation')

# AR(p) model
# AR(1) with alpha1=1 is simply a random walk, therefore not stationary (althought 1st order diff would be stationary)
# AR(2) => x(t) = alpha1*x(t-1) + alpha2*x(t-2) + w(t), where w(t) is white noise

# simulate an AR(1) process with alpha =0.6
np.random.seed(1)
a = 0.6
x = w = np.random.normal(size=n_samples)

for t in range(n_samples):
    x[t] = a * x[t - 1] + w[t]
p("AR(1), Alpha=0.6\n----------------------\nmean: {:.3f}\nvariance: {:.3f}\nstandard deviation: {:.3f}".format(
    x.mean(), x.var(), x.std()))
p("\n")
_ = tsplot(x, lags=30, title='AR(1) Process Simulation, Alpha=0.6')

# Lets fit a AR(p) model to the above... we should get back from the fit that alpha ~ 0.6 and p = 1
mdl = smt.AR(x).fit(maxlag=30, ic='aic', trend='nc')
est_order = smt.AR(x).select_order(maxlag=30, ic='aic', trend='nc', method='mle')
true_order = 1
p('\nalpha estimate: {:3.5f} | best lag order = {}'.format(mdl.params[0], est_order))
p('\ntrue alpha = {} | true order = {}'.format(a, true_order))
p("\n\n")

# simulate AR(2) process with alpha0=0.666, alpha1=-0.333
alphas = np.array([0.666, -0.333])
betas = np.array([0.])  # betas are for later, the MA component when we add MA(q)
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]
ar2 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n_samples)
p(
    "AR(2), Alphas=[0.6666,-0.333]\n----------------------\nmean: {:.3f}\nvariance: {:.3f}\nstandard deviation: {:.3f}".format(
        ar2.mean(), ar2.var(), ar2.std()))
p("\n")
_ = tsplot(ar2, lags=30, title='Simulated AR(2) Process, alphas=[0.666, -0.333]')

# Lets fit a AR(p) model to the simualted AR(2) process
mdl = smt.AR(ar2).fit(maxlag=10, ic='aic', trend='nc')
est_order = smt.AR(ar2).select_order(maxlag=10, ic='aic', trend='nc', method='mle')
true_order = 2
p('\ncoef estimate: {:3.5f} {:3.5f} | best lag order = {}'.format(mdl.params[0], mdl.params[1], est_order))
p('\ntrue coefs = {} | true order = {}'.format([0.666, -.333], true_order))
p("\n\n")

lrets = ydo_1.calc_log_returns('AdjClose')
p("Log Returns " + symbol_list[
    0] + "\n----------------------\nmean: {:.7f}\nvariance: {:.7f}\nstandard deviation: {:.7f}".format(
    lrets[symbol_list[0]].mean(), lrets[symbol_list[0]].var(), lrets[symbol_list[0]].std()))
p("\n")
_ = tsplot(lrets[symbol_list[0]], lags=30, title='Log Returns ' + symbol_list[0])

max_lag = 30
mdl = smt.AR(lrets[symbol_list[0]]).fit(maxlag=max_lag, ic='aic', trend='nc')
est_order = smt.AR(lrets[symbol_list[0]]).select_order(maxlag=max_lag, ic='aic', trend='nc', method='mle')
p('\ncoef estimate: {:3.5f} {:3.5f} | best lag order = {}'.format(mdl.params[0], mdl.params[1], est_order))
p("\n\n")

# simulate a MA(q) model where q=1
# x(t) = w(t) + beta1*w(t-1) + ... + betaQ*w(t-q)
alphas = np.array([0.])  # alphas are 0 here since we are not considering a AR(p) model
betas = np.array([0.6])  # betas are the coefficients of the past white noise
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma1 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n_samples)
_ = tsplot(ma1, lags=30, title="Simulated MA(1) Process, Beta1=0.6")

# fit the simulated data to a MA(1) model and lets see what the Beta1 yielded
mdl = smt.ARMA(ma1, order=(0, 1)).fit(maxlag=30, method='mle', trend='nc', ic='aic')
p(mdl.summary())

# simulate a MA(q) model where q=3
# x(t) = w(t) + beta1*w(t-1) + beta2*w(t-2) + beta3*w(t-3)
alphas = np.array([0.])
betas = np.array([0.6, 0.4, 0.2])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma3 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n_samples)
_ = tsplot(ma3, lags=30, title='Simulated MA(3) Process, Betas=[0.6,0.4,0.2]')

# fit the simulated data to a MA(3) model, check the yielded Betas
mdl = smt.ARMA(ma3, order=(0, 3)).fit(maxlag=30, method='mle', trend='nc', ic='aic')
p(mdl.summary())

mdl = smt.ARMA(lrets_spy.SPY, order=(0, 3)).fit(maxlag=30, method='mle', trend='nc')
p(mdl.summary())
_ = tsplot(mdl.resid, lags=30, title="SPY Log Returns Fitted to MA ")

## ARMA model, combining AR and MA
n_samples = 5000
burn = int(n_samples / 10)
alphas = np.array([0.5, -0.25])
betas = np.array([0.5, -0.3])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]
arma22 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n_samples, burnin=burn)
_ = tsplot(arma22, lags=30, title='ARMA(2,2 Simulation, Sample Generated from arma_generate_sample()')

mdl = smt.ARMA(arma22, order=(2, 2)).fit(maxlag=max_lag, method='mle', trend='nc', burnin=burn)
p(mdl.summary())
