'''
Created on Dec 19, 2017

@author: ghazy
'''

from root.nested.quandl_data_object import QuandlDataObject
from root.nested.interest_rates import HeathJarrowMortonModel

import pandas as pd
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

import matplotlib.pyplot as plt

p=print

def ts_plot(y, lags=None, figsize=(10, 8), style='bmh', title='Time Series Analysis Plots'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (3,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1,0))
        pacf_ax = plt.subplot2grid(layout, (1,1))
        qq_ax = plt.subplot2grid(layout, (2,0))
        pp_ax = plt.subplot2grid(layout, (2,1))
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

qdo = QuandlDataObject('INTEREST_RATES',
                       'US_TREASURY_YIELD_CURVE',
                       '.csv')

yield_curve_df = qdo.get_df()
yield_curve_df.columns = [c.replace(' ', '_') for c in yield_curve_df.columns]
print(yield_curve_df.info())
#yield_curve_df.dropna(subset=[yield_curve_df.columns[0]], inplace=True) # drop any rows where first column is nan
#yield_curve_df.dropna(inplace=True)
two_yr_y = yield_curve_df['10_YR']
#_ = ts_plot(two_yr_y, lags=30)

hjm = HeathJarrowMortonModel(yield_curve_df)
hjm.get_df_info()
yield_curve_df = hjm.vectorized_create_yield_curve() # the returned Dataframe here contains the yield curve object for each date

# interpolate first with Piecewise Cubic hermite spline funciton, just as the treasury.gov data does.
# this is the first step toward getting the spot curve from the par curve in treasury.gov data
hjm.do_interpolation(interp_method='PCS') # PCS is the default, but just for clarity sake, we pass in the arguement
# once we have the interpolated par yield curve, now we can bootstrap to get spot rates.
hjm.do_bootstrap_spot()
#hjm.do_interpolation(interp_method='NSS')
hjm.do_NSS_params_analysis()
# at this point, we've got the spot rates of the interoplated maturities
# now we can continue with NSS and the forecasting process.
#hjm.calc_fmin_stats()
