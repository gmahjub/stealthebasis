'''
Created on Dec 6, 2017

@author: ghazy
'''

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
#from stldecompose import decompose
from root.nested import get_logger


class StationaryObj(object):

    def __init__(self,
                 timeseries):

        self.logger = get_logger()

        self.is_stationary = False
        self.timeseries = timeseries
        self.df_output = None

        self.classical_decomp_result_obj = None
        self.seasonal_classical_decomp = None
        self.trend_classical_decomp = None
        self.residuals_classical_decomp = None

        self.stl_result_obj = None
        self.trend_stl_decomp = None
        self.seasonal_stl_decomp = None
        self.residuals_stl_decomp = None

        self.df_test_exception = None

    def dickey_fuller_test(self,
                           autolag_type,
                           critical_value='1%'):  # options are 1%, 5%, 10%
        # autolag_type: 'AIC' (default), 'BIC', 't-stat', None
        try:
            df_test = adfuller(self.timeseries.iloc[:, 0], autolag=autolag_type)
        except Exception as excp:
            self.df_test_exception = excp
            return False
        df_output = pd.Series(df_test[0:4],
                              index=['Test Statistic', 'p_-value', '#Lags Used', 'Number of Observations Used'])
        is_stationary = False
        for key, value in df_test[4].items():
            df_output['Critical Value (%s)' % key] = value
            if key == critical_value:
                print('key and critcal value equal', value, df_test[1])
                if int(value * 1000000.0) > int(float(df_test[0]) * 1000000.0):
                    is_stationary = True

        self.df_output = df_output
        self.is_stationary = is_stationary

        return is_stationary

    def calc_RSS(self,
                 predicted,
                 actual):

        check_type = (type(predicted) == type(actual) == np.ndarray)
        # check to make sure they are of the same length
        check_len = len(predicted) == len(actual)
        if check_len and check_type:
            # predicted values and actual_values must be numpy array
            return sum((predicted - actual) ** 2)
        else:
            print("exiting program, len(predicted) != len(actual), must be same length arrays and of type numpy.array")
            return None

    def calc_and_verify_MSE(self,
                            predicted,
                            actual):

        sklearn_mse = mean_squared_error(actual, predicted)
        calculated_mse = self.calc_RSS(predicted, actual) / len(predicted)
        if calculated_mse is not None and int(sklearn_mse * 1000000.0) == int(calculated_mse * 1000000.0):
            return calculated_mse
        else:
            return None

    def calc_RMSE(self,
                  predicted,
                  actual):

        return np.sqrt(self.calc_and_verify_MSE(predicted, actual))

    # seasonal decompose method uses as default a additive model
    # but if we are logging the time series (log transform) 
    # before we run this, we are essentially using multiplicative model
    def classical_seasonal_decomposition(self,
                                         model='additive',
                                         # model='multiplicative' is the other option, additive is default
                                         filter_type=None,  # the filter used is the convolution filter as default
                                         two_sided_bool=True):  # true = centered moving average, False = historical values only MA 

        # for filters check out: 1) statsmodels.tsa.filters.bk_filter.bkfilter
        #                        2) statsmodels.tsa.filters.cf_filter.xffilter
        #                        3) statsmodels.tsa.filters.hp_filter.hpfilter

        # we are essentially doing multiplicative model because the data is logged first.
        self.logger.info("stationarity.classical_seasonal_decomposition: DataFrame frequency: %s",
                         str(self.timeseries.index.freq))
        classical_decomp_result_obj = seasonal_decompose(self.timeseries, model, filt=filter_type,
                                                         two_sided=two_sided_bool)
        self.classical_decomp_result_obj = classical_decomp_result_obj

        self.seasonal_classical_decomp = classical_decomp_result_obj.seasonal
        self.trend_classical_decomp = classical_decomp_result_obj.trend
        self.residuals_classical_decomp = classical_decomp_result_obj.resid

    def loess_seassonal_decomposition(self,
                                      period_size,
                                      resample_freq='D',
                                      interpolation='linear'):

        # period size is in interms of number of indexes since this is a DataFrame
        # when we detrend, we are left with seasonal and residual (random) component
        # uses statsmodels.tsa.seasonal.seasonal_decompose
        timeseries_df = (self.timeseries.resample(resample_freq).mean().interpolate(interpolation))

        stl = STL(timeseries_df).fit()
        #stl = decompose(timeseries_df, period=period_size)
        self.stl_result_obj = stl

        self.seasonal_stl_decomp = stl.seasonal
        self.trend_stl_decomp = stl.trend
        self.residuals_stl_decomp = stl.resid

    def classical_detrend(self):

        return 1

    def loess_detrend(self):

        return 1

    def mann_kendall_trend_test(self,
                                timeseries_series,
                                ci=0.95):

        np_values_array = timeseries_series.values
        len_values = len(np_values_array)
        s = 0
        for k in range(len_values - 1):
            for j in range(k + 1, len_values):
                s += np.sign(np_values_array[j] - np_values_array[k])

        unique_value_array = np.unique(np_values_array)
        len_unique_vals = len(unique_value_array)

        if len_unique_vals == len_values:
            var_s = (len_values * (len_values - 1) * (2 * len_values + 5)) / 18
        else:
            tp = np.zeros(unique_value_array.shape)
            for i in range(len(unique_value_array)):
                tp[i] = sum(unique_value_array[i] == np_values_array)
            var_s = (len_values * (len_values - 1) * (2 * len_values + 5) + np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s == 0:
            z = 0
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)

        p = 2 * (1 - norm.cdf(abs(z)))
        h = abs(z) > norm.ppf(ci / 2)

        if (z < 0) and h:
            trend = 'decreasing'
        elif (z > 0) and h:
            trend = 'increasing'
        else:
            trend = 'no trend'

        return trend, h, p, z
