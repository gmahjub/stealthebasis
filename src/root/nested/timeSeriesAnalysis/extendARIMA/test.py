'''
Created on Nov 15, 2017

@author: traderghazy
'''
from root.nested import get_logger
logger = get_logger()

import pandas as pd, numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf, arma_order_select_ic
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from scipy import stats

from root.nested.quandl_data_object import QuandlDataObject
from root.nested.arima_module import ArimaObject
from root.nested.arima_module import ArimaOrder

## put this in to ignore warnings in evaluation of ARIMA
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from math import sqrt

from root.nested.stationary import StationaryObj

warnings.filterwarnings("ignore")
########################################################

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6

logger.debug("Created QuandlSymbolInterface object")

def get_quandl_data(my_symbol,
                    file_type):
    
    qdo = QuandlDataObject(my_symbol,
                           file_type)
    
    qdo.set_df(qdo.get_df().asfreq('B').fillna(method='ffill'))
    qdo.df_to_csv()
    logger.info("DataCamp.get_quandl_data(): DataFrame frequency is set to %s ", str(qdo.get_df_index_freq()))
    return qdo.get_df()

# we will use this when calculating confidence intervals for ACF and PACF plots.
def percent_quantile_function(ci):
    
    pqf = stats.norm.ppf(1-(1-ci)/2.)
    return pqf

# plot the autocorrelation funciton
# plot the partial autocorrelation function
# any value above 0.2 is considered significant on these plots.
def plot_acf_pacf_ibm_ds_code(timeseries,
                              ci,
                              b_fft,
                              plot_show=False):
    
    #Plot ACF:
    lag_acf, confint_acf, qstat, pvalues = acf(timeseries, qstat=True, nlags=10, alpha=(1-ci), fft=b_fft) # there are other arguements, document them.\
    pos_interval_acf = lag_acf - confint_acf[:,0]
    neg_interval_acf = -1*(confint_acf[:,1] - lag_acf)
    
    lag_pacf, confint_pacf = pacf(timeseries, nlags=10, method='ols', alpha=(1-ci)) # document any other parameters.
    pos_interval_pacf = lag_pacf - confint_pacf[:,0]
    neg_interval_pacf = -1*(confint_pacf[:,1] - lag_pacf)
    
    pyplot.figure(1)
    #plot the acf
    pyplot.subplot(121)
    pyplot.plot(lag_acf, marker='o')
    pyplot.axhline(y=0, linestyle='--', color='gray')
    pyplot.plot(pos_interval_acf, linestyle='--', color='red')
    pyplot.plot(neg_interval_acf, linestyle='--', color='red')
    pyplot.axhline(y=-percent_quantile_function(ci)/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    pyplot.axhline(y=percent_quantile_function(ci)/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    pyplot.title('Autocorrelation Function')
    
    #Plot PACF:
    pyplot.subplot(122)
    pyplot.plot(lag_pacf, marker='o')
    pyplot.axhline(y=0,linestyle ='--', color='gray')
    pyplot.plot(pos_interval_pacf, linestyle='--', color='red')
    pyplot.plot(neg_interval_pacf, linestyle='--', color='red')
    pyplot.axhline(y=-percent_quantile_function(ci)/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    pyplot.axhline(y=percent_quantile_function(ci)/np.sqrt(len(timeseries)), linestyle='--', color='gray')
    pyplot.title('Partial Autocorrelation Function')
    
    pyplot.tight_layout()
    pyplot.show()
    
    PQ_tuple = find_p_and_q_values_from_plots(pos_interval_acf,
                                              neg_interval_acf,
                                              lag_acf,   
                                              lag_pacf)

    return PQ_tuple

def find_p_and_q_values_from_plots(pos_interval_acf,
                                   neg_interval_acf,
                                   lag_acf,
                                   lag_pacf):
    
    # first, find lag_acf intercept with upper CI band (pos_interval_acf = Q value
    # second, find lac_pacf intercept with 0 band = P value
    
    logger.info('DataCamp.find_p_and_q_values_from_plots.pos_interval_acf: %s', str(pos_interval_acf))
    logger.info('DataCamp.find_p_and_q_values_from_plots.neg_interval_acf: %s', str(neg_interval_acf))
    logger.info('DataCamp.find_p_and_q_values_from_plots.lag_acf: %s', str(lag_acf))
    logger.info('DataCamp.find_p_and_q_values_from_plots.lag_pacf: %s', str(lag_pacf))
    
    Q = 0
    P = 0
    i=0
    while i < len(pos_interval_acf):
        if (int(pos_interval_acf[i]*100.0 - int(lag_acf[i]*100.0) > -2.5)): 
        #if int(pos_interval_acf[i]*100.0 - 2.5) >= int(lag_acf[i]*100.0):
            # this means lag_acf is less than pos_interval_acf, this value is the Q value
            Q = i
            break
        else:
            i+=1
    i=0
    while i < len(lag_pacf):
        if int(lag_pacf[i]*100.0) <= 2.5: # close enough to 0
            # this means lac_pacf is less than 0, this value is the P value
            P = i
            break
        else:
            i+=1   
    return (P,Q)

# it seems like this data 
def plot_acf_pacf(timeseries_series,
                  ci,
                  b_fft,
                  plot_show=False):

    # the plot_acf and plot_pacf functions take pd.Series datatypes as argument
    pyplot.figure()
    pyplot.subplot(211)
    plot_acf(timeseries_series, lags=61, ax=pyplot.gca(), alpha=(1-ci), fft=b_fft)
    pyplot.title('ACF Plot')
    pyplot.subplot(212)
    plot_pacf(timeseries_series, lags=61, method='ols', alpha=(1-ci), ax=pyplot.gca())
    pyplot.title('PACF Plot')
    pyplot.show()

def do_ARIMA(weekly_boe_xudlerd_log_diff,
             arima_P_value,
             arima_D_value,
             arima_Q_value,
             plot_show=False):
    
    # ARIMA tuple is in the following format... ARIMA(P,D,Q)
    model = ARIMA(weekly_boe_xudlerd_log_diff, order=(arima_P_value, arima_D_value, arima_Q_value))
    try:
        results_ARIMA = model.fit(trend='nc', disp=-1) # trend can be 'nc' also, no constant trend added
    except Exception as excep:
        return -1
    pyplot.plot(weekly_boe_xudlerd_log_diff.index.to_pydatetime(), weekly_boe_xudlerd_log_diff.values)
    pyplot.plot(weekly_boe_xudlerd_log_diff[arima_D_value:].index.to_pydatetime(), results_ARIMA.fittedvalues, color='red')
    #print ('do these match???', len(weekly_boe_xudlerd_log_diff), len(results_ARIMA.fittedvalues))
    # they will always differ by the value of arima_D
    
    # valiodate the model with RSS - the smaller the better for RSS (residual sum of squares)
    rss = sum((results_ARIMA.fittedvalues-weekly_boe_xudlerd_log_diff.iloc[arima_D_value:,0])**2) # residual sum of squares
    rmse = np.sqrt(rss/len(weekly_boe_xudlerd_log_diff))
    print ("In sample RMSE...", rmse, " for order ", (arima_P_value, arima_D_value, arima_Q_value))
    pyplot.title('RMSE: %.4f'% rmse)
    pyplot.show()
    
    # or we can validate the model with either RSS or we can look at the residuals within the ARIMAResultsObject.
    residuals = pd.DataFrame(results_ARIMA.resid)
    residuals.plot(kind='kde')
    pyplot.title('ARIMA Results - Residuals')
    pyplot.show()
    print (residuals.describe())
    return results_ARIMA

def do_in_sample_predict(results_ARIMA,
                         weekly_boe_xudlerd_log_diff,
                         wbxl):
    
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(wbxl.iloc[0].values, index=weekly_boe_xudlerd_log_diff.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    
    return predictions_ARIMA

def plot_in_sample_prediction(predictions_arima,
                              weekly_boe_xudlerd_data,
                              arima_object,
                              differencing_interval,
                              plt_show=False):
    
    pyplot.plot(weekly_boe_xudlerd_data.index.to_pydatetime(), weekly_boe_xudlerd_data.values)
    pyplot.plot(weekly_boe_xudlerd_data.index[differencing_interval:].to_pydatetime(), predictions_arima.values, color='red')
    # they will always differ by value of 1
    rmse = np.sqrt(sum((predictions_arima-weekly_boe_xudlerd_data.iloc[differencing_interval:,0])**2)/len(weekly_boe_xudlerd_data))
    pyplot.title('RMSE: %.4f'% rmse )
    pyplot.show()
    
    return rmse

def optimal_ARIMA_params(timeseries_values,
                         p_values,
                         d_values,
                         q_values,
                         weekly_boe_xudlerd_data,
                         differencing_interval):
    # 3 parameters to find:
    # p: number of AR terms, period look-back to predict X(t) where X is the Euro rate for example
    #    if p = 2, for example, then X(t-1) and X(t-2) are predictors of X(t)
    # q: number of MA terms to use, where MA terms are lagged forecast error in the prediction equation
    #    q = 2, for example, means predictor for X(t) will be e(t-1) and e(t-2), where e(i) is the
    #    difference between MA at ith instant and the actual value at ith instant.
    # d: If we are using a differenced series, than d = 0. If we are not using a differenced series,
    #    then explicity tell it here with the d variable what the differencing interval is.            
    
    return 1

# reverse the difference decomposition
def reverse_difference_decomposition(history, yhat, interval=1):
    return yhat + history[-interval]

def difference_composition(dataset, interval=1):
    #diff = list()
    to_return = np.array(dataset) - np.roll(np.array(dataset), interval)
    return to_return[interval:]
    #for i in range(interval, len(dataset)):
    ##    value = dataset[i] - dataset[i - interval]
    #     diff.append(value)
    #return np.array(diff)

# this function simply evaluates one set of hyperparameters
# we will call this function over and over to find the best parameters
# this function will be called from iterate_arima_model
# In addition, realize that what we are doing here is OUT OF SAMPLE FORECASTING
def evaluate_arima_model(timeseries_values, 
                         arima_order,
                         pct_training_data,
                         weekly_boe_xudlerd_data,
                         differencing_interval,
                         plot_show=False):
    
    train_size = int(len(timeseries_values)*pct_training_data)
    train_size = int(len(timeseries_values) - 15)
    train, test = timeseries_values[0:train_size], timeseries_values[train_size:]
    test_values = test.values
    train_values = train.values
    test_indexes = test.index
    history = [x for x in train_values]
    predictions = list()
    
    exception_in_ARIMA = False
    for t in range(len(test_values)):
        diffed_data = difference_composition(history, interval=differencing_interval)
        model = ARIMA(diffed_data, order=arima_order)
        try:
            model_fit = model.fit(disp=0) # trend='c' is the default, other option is trend='nc' 
        except Exception as excp:
            exception_in_ARIMA=True
            break
        yhat, err95, ci95 = model_fit.forecast(steps = 5)[0]
        # we have to invert the difference here, using same interval as when we difference for stationarity
        yhat = reverse_difference_decomposition(history, yhat, interval=differencing_interval)
        predictions.append(yhat[0])
        obs = test_values[t]
        test_index = test_indexes[t]
        history.append(obs)
        print ('predicted=%f, expected=%f, date=%s' % (np.exp(yhat), np.exp(obs), test_index))
    
    if exception_in_ARIMA:
        return -1
    
    error = mean_squared_error(test_values, predictions)
    print ('\n')
    print ('Printing Mean Squared Error of Predictions...')
    print ('Test RMSE: %.6f' % sqrt(error))
    
    predictions_series = pd.Series(predictions, index=test.index)
    
    fig, ax = pyplot.subplots()
    ax.set(title='Spot Exchange Rate, Euro into USD', xlabel='Date', ylabel='Euro into USD')
    
    ax.plot(weekly_boe_xudlerd_data[-50:], 'o', label='observed')
    # predicted value
    ax.plot(np.exp(predictions_series[-50:]), 'g', label='rolling one-step out-of-sample forecast')
    legend = ax.legend(loc='upper left')
    legend.get_frame().set_facecolor('w')
    pyplot.show()
    return sqrt(error)

def iterate_ARIMA_models(timeseries_series, 
                         p_values, 
                         d_values, 
                         q_values,
                         pct_training_data):
    
    # timeseries_series is a pd.Series and we need to pull the values out of it
    # values should be a numpy array
    np_timeseries = timeseries_series.values
    timeseries_values = np_timeseries.astype('float32')
    best_score, best_cfg = float(1000000000), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                the_arima_object = ArimaObject(p,
                                               d,
                                               q,
                                               "grid_search_arima_hyper_parameters",
                                               "grid_search_arima_hyper_parameters")
                
                try:
                    rmse = the_arima_object.evaluate_arima_model(timeseries_values,
                                                                pct_training_data)
                    if int(rmse*1000000.0) < int(best_score*1000000.0):
                        best_score, best_cfg = rmse, order
                    print (order,rmse)
                    print ("ARIMA%s RMSE=%.6f" % (order,rmse))
                except Exception as excp:
                    #print ('unexpected exception thrown', excp)
                    continue
    return (best_cfg, best_score)

def review_residual_errors(timeseries_series,
                           p_value,
                           d_value,
                           q_value,
                           pct_training_data):
    
    np_timeseries = timeseries_series.values
    timeseries_values = np_timeseries.astype('float32')
    the_arima_object = ArimaObject(p_value,
                                   d_value,
                                   q_value,
                                   "review_residual_error",
                                   "review_residual_error")
    try:
        _ = the_arima_object.evaluate_arima_model(timeseries_values,
                                                     pct_training_data)
    except Exception as excp:
        print ("unexpected exception thrown", excp)
    
    residuals = the_arima_object.calculate_residuals(timeseries_values,
                                                     pct_training_data)
    
    # residuals returned above is a DataFrame object
    print (residuals.describe())
    #plot
    pyplot.figure()
    pyplot.subplot(211)
    residuals.hist(ax=pyplot.gca())
    pyplot.subplot(212)
    residuals.plot(kind='kde', ax=pyplot.gca())
    pyplot.show()

# uses statsmodels.tsa.stattools.arma_order_select_ic
# using fit_kw to speed up the ARMA.fit() process
# fit_kw = { method: 'css'}
# timeseries parameter here is a np.array
# max_ar_params is default 4
# max ma params is default 2
# ic_param is 'AIC', but could be other ones too, look at docs
# trend_param = 'nc', could be other ones too, we will use 'nc'
# returns a coordinate, (x,y) wher x is the row number, y is the col number
# x is the AR order and y is the MA order
def find_ARIMA_params_automated(timeseries,
                                max_ar_param,
                                max_ma_param,
                                ic_param,
                                trend_param):
    
    res = arma_order_select_ic(timeseries, 
                               max_ar=max_ar_param, 
                               max_ma=max_ma_param, 
                               ic=ic_param, 
                               trend=trend_param,
                               model_kw={}, 
                               fit_kw={'method':'css'})

    return res

def test_find_ARIMA_params_automated():
    
    from statsmodels.tsa.arima_process import arma_generate_sample
    
    arparams = np.array([.75, -.25])
    maparams = np.array([0.65, 0.35])
    arparams = np.r_[1, -arparams]
    maparam = np.r_[1, maparams]
    
    # number of observations
    nobs = 250
    np.random.seed(2014)
    
    y = arma_generate_sample(arparams, maparams, nobs)
    res = arma_order_select_ic(y, 
                               max_ar=4, 
                               max_ma=4, 
                               ic=['aic', 'bic'], 
                               trend='nc', 
                               model_kw={}, 
                               fit_kw={'method':'css'})
 
    return res

# need to fix this so that it plots months on X-axis, Jan-Dec, with each year being a diff time series.
def plot_seasonality(timeseries_series,
                     plot_show=False):
    
    groups = timeseries_series.groupby(pd.TimeGrouper('A'))
    #for name, group in groups:
    #    pyplot.plot(group)
    #pyplot.title("Examining Seasonality")
    #pyplot.show()  

# visual examination of stationarity
def plot_stationarity_test(timeseries,
                           window_size,
                           plot_show=False,
                           title='Rolling Mean & Standard Deviation'):
    
    ## rolling statistics, basically a moving window to run the calculation on every data sample
    rolling_mean = timeseries.rolling(window=window_size, center=False).mean()
    rolling_std_dev = timeseries.rolling(window=window_size, center=False).std()
    
    _ = pyplot.plot(timeseries.index.to_pydatetime(), timeseries.values, color='blue', label='Original')
    _ = pyplot.plot(rolling_mean.index.to_pydatetime(), rolling_mean.values, color='red', label='Rolling Mean')
    _ = pyplot.plot(rolling_std_dev.index.to_pydatetime(), rolling_std_dev.values, color='black', label='Rolling Std Dev')
    pyplot.legend(loc='best')
    pyplot.title(title)
    pyplot.show()

def dickey_fuller_test(timeseries,
                       autolag_type,
                       critical_value='1%'): # options are 1%, 5%, 10%
    #autolag_type: 'AIC' (default), 'BIC', 't-stat', None
    stationary_obj = StationaryObj(timeseries)
    stationary_obj.dickey_fuller_test(autolag_type, 
                                      critical_value)
    return stationary_obj
    
    
def test_stationarity(timeseries,
                      window_size,
                      autolag_type,
                      title):
    
    plot_stationarity_test(timeseries, window_size, title=title)
    s_obj = dickey_fuller_test(timeseries, autolag_type, "1%")
    return s_obj
    
# the parameter timeseries_series is a pd.Series object
# returns pd.Series() object.
def non_linear_log_transformation(timeseries_series):
    return pd.Series(np.log(np.array(timeseries_series)))

def seasonal_decomposition(timeseries):
    return seasonal_decompose(timeseries)

# returns a np.array, not a pd.Series
def difference_decomposition(timeseries_series, interval=1):
    # when interval = 1, we are taking the first difference
    # seasonal decomposition is a differenct thing.
    timeseries_values = timeseries_series.values
    diff = list()
    for i in range(interval, len(timeseries_values)):
        value = timeseries_values[i] - timeseries_values[i-interval]
        diff.append(value)
    return np.array(diff)


boe_xulerd_data = get_quandl_data('EURO_USD_spot',
                                  '.csv')
pyplot.figure()
# plot the raw data
pyplot.plot(boe_xulerd_data.index.to_pydatetime(), boe_xulerd_data.values)
pyplot.title("Original Raw Data")
pyplot.show()

#===============================================================================
# # resample to weekly data
weekly_boe_xudlerd_data = boe_xulerd_data.resample('W').mean()
weekly_boe_xudlerd_data = boe_xulerd_data
print (type(weekly_boe_xudlerd_data))
#plot the resampled data
pyplot.plot(weekly_boe_xudlerd_data.index.to_pydatetime(), weekly_boe_xudlerd_data.values)
s_obj = test_stationarity(weekly_boe_xudlerd_data, 260, 'AIC', title='Raw Daily Price - Rolling Mean & Std Dev')
if s_obj.df_test_exception != None:
    print("Dickey Fuller DataCamp failed with exception ", s_obj.df_test_exception)
else:
    print ('Dickey Fuller Stats ', s_obj.df_output)
    print ('Is it stationary? ', s_obj.is_stationary)

s_obj.classical_seasonal_decomposition()
seasonal_classical_decomp = s_obj.seasonal_classical_decomp
trend_classical_decomp = s_obj.trend_classical_decomp
residuals_classical_decomp = s_obj.residuals_classical_decomp

selected_data_to_plot = s_obj.timeseries[-90:] 
pyplot.subplot(411)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), selected_data_to_plot.values, label="Original")
pyplot.legend(loc='best')
pyplot.subplot(412)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), trend_classical_decomp[-90:].values, label='Trend')
pyplot.legend(loc='best')
pyplot.subplot(413)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), residuals_classical_decomp[-90:].values, label='Residuals')
pyplot.legend(loc='best')
pyplot.subplot(414)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), seasonal_classical_decomp[-90:].values, label='Seasonality')
pyplot.legend(loc='best')
pyplot.tight_layout()
pyplot.show()

s_obj.loess_seassonal_decomposition(52)
seasonal_stl_decomp = s_obj.seasonal_stl_decomp
trend_stl_decomp = s_obj.trend_stl_decomp
residuals_stl_decomp = s_obj.residuals_stl_decomp

selected_data_to_plot = s_obj.timeseries[-90:]
pyplot.subplot(411)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), selected_data_to_plot.values, label="Original")
pyplot.legend(loc='best')
pyplot.subplot(412)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), trend_stl_decomp[-90:].values, label='Trend')
pyplot.legend(loc='best')
pyplot.subplot(413)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), residuals_stl_decomp[-90:].values, label='Residuals')
pyplot.legend(loc='best')
pyplot.subplot(414)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), seasonal_stl_decomp[-90:].values, label='Seasonality')
pyplot.legend(loc='best')
pyplot.tight_layout()
pyplot.show()


weekly_boe_xudlerd_log = np.log(weekly_boe_xudlerd_data)
s_obj = test_stationarity(weekly_boe_xudlerd_log, 260, 'AIC', title="Log Daily Price - Rolling Mean & Std Dev")
if s_obj.df_test_exception != None:
    print("Dickey Fuller DataCamp failed with exception ", s_obj.df_test_exception)
else:
    print ('Dickey Fuller Stats ', s_obj.df_output )
    print ('Is it stationary? ', s_obj.is_stationary)

s_obj.classical_seasonal_decomposition()
seasonal_classical_decomp = s_obj.seasonal_classical_decomp
trend_classical_decomp = s_obj.trend_classical_decomp
residuals_classical_decomp = s_obj.residuals_classical_decomp

selected_data_to_plot = s_obj.timeseries[-180:] 
pyplot.subplot(411)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), selected_data_to_plot.values, label="Original")
pyplot.legend(loc='best')
pyplot.subplot(412)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), trend_classical_decomp[-180:].values, label='Trend')
pyplot.legend(loc='best')
pyplot.subplot(413)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), residuals_classical_decomp[-180:].values, label='Residuals')
pyplot.legend(loc='best')
pyplot.subplot(414)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), seasonal_classical_decomp[-180:].values, label='Seasonality')
pyplot.legend(loc='best')
pyplot.tight_layout()
pyplot.show()

differencing_interval = 2
#shift method defaults to 1 period, set "periods=n" for non-default values.
weekly_boe_xudlerd_log_diff = weekly_boe_xudlerd_log - weekly_boe_xudlerd_log.shift(periods=differencing_interval)
pyplot.plot(weekly_boe_xudlerd_log_diff.index.to_pydatetime(), weekly_boe_xudlerd_log_diff.values)
pyplot.title("Daily Log Diff Price Data")
pyplot.show()
weekly_boe_xudlerd_log_diff.dropna(inplace=True)
 
# DataCamp stationarity
s_obj = test_stationarity(weekly_boe_xudlerd_log_diff, 260, 'AIC', title='Log First-Order Difference - Rolling Mean & Std Dev')
if s_obj.df_test_exception != None:
    print("Dickey Fuller DataCamp failed with exception ", s_obj.df_test_exception)
else:
    print ("Dickey Fuller Stats ", s_obj.df_output)
    print ('Is it stationary? ', s_obj.is_stationary)

s_obj.classical_seasonal_decomposition()
seasonal_classical_decomp = s_obj.seasonal_classical_decomp
trend_classical_decomp = s_obj.trend_classical_decomp
residuals_classical_decomp = s_obj.residuals_classical_decomp
 
selected_data_to_plot = s_obj.timeseries[-180:] 
pyplot.subplot(411)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), selected_data_to_plot.values, label="Original")
pyplot.legend(loc='best')
pyplot.subplot(412)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), trend_classical_decomp[-180:].values, label='Trend')
pyplot.legend(loc='best')
pyplot.subplot(413)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), residuals_classical_decomp[-180:].values, label='Residuals')
pyplot.legend(loc='best')
pyplot.subplot(414)
pyplot.plot(selected_data_to_plot.index.to_pydatetime(), seasonal_classical_decomp[-180:].values, label='Seasonality')
pyplot.legend(loc='best')
pyplot.tight_layout()
pyplot.show()
  
#######GRID Search ARIMA Hyperparamenters+
#p_values = range(0,3)
#d_values = range(0,1)
#q_values = range(0,10)
#warnings.filterwarnings("ignore")
#best_cfg, best_score = iterate_ARIMA_models(weekly_boe_xudlerd_log, p_values, d_values, q_values, 0.67)
#print ('Best ARIMA%s RMSE=%.6f' % (best_cfg, best_score))
#print ("Hyper parameter search complete...")  
##################################
  
res = find_ARIMA_params_automated(weekly_boe_xudlerd_log_diff.values, 4, 2, ['aic', 'bic'], 'nc')
  
p_arima_value_aic = res['aic_min_order'][0]
q_arima_value_aic = res['aic_min_order'][1]
p_arima_value_bic = res['bic_min_order'][0]
q_arima_value_bic = res['bic_min_order'][1]
  
list_of_ARIMA_objects = []
  
ao_aic_dplus_interval = ArimaObject(p_arima_value_aic,
                                    0,
                                    q_arima_value_aic,
                                    "arma_order_select_ic",
                                    "arma_order_select_ic",
                                    'W',
                                    False,
                                    differencing_interval=differencing_interval)
list_of_ARIMA_objects.append(ao_aic_dplus_interval)

ao_bic_dplus_interval = ArimaObject(1,
                                    0,
                                    2,
                                    'arima_order_select_ic',
                                    'arima_order_select_ic',
                                    'W',
                                    False,
                                    differencing_interval=differencing_interval)
#list_of_ARIMA_objects.append(ao_bic_dplus_interval)

print ('p_arima_value_AIC: ', p_arima_value_aic, " yielded by Automated ARIMA function")
print ('q_arima_value_AIC: ', q_arima_value_aic, " yielded by Automated ARIMA function") 
print ('p_arima_value_BIC: ', p_arima_value_bic, " yielded by Automated ARIMA function")
print ('q_arima_value_BIC: ', q_arima_value_bic, " yielded by Automated ARIMA function")
  
b_fft = False
ci=0.95
PQ_tuple = plot_acf_pacf_ibm_ds_code(weekly_boe_xudlerd_log_diff,
                                     ci,
                                     b_fft)
print ("P value, Q value ", PQ_tuple, " yielded from plot functions of IBM DS code")
ao_acf_pacf_dplus_interval = ArimaObject(PQ_tuple[0],
                                         0,
                                         PQ_tuple[1],
                                         'pacf',
                                         'acf',
                                         'W',
                                         False,
                                         differencing_interval=differencing_interval)
list_of_ARIMA_objects.append(ao_acf_pacf_dplus_interval)

ao_acf_pacf_dplus_interval = ArimaObject(0,
                                         0,
                                         2,
                                         'pacf',
                                         'acf',
                                         'W',
                                         False,
                                         differencing_interval=differencing_interval)
#list_of_ARIMA_objects.append(ao_acf_pacf_dplus_interval)

ao_acf_pacf_dplus_interval = ArimaObject(2,
                                         0,
                                         0,
                                         'pacf',
                                         'acf',
                                         'W',
                                         False,
                                         differencing_interval=differencing_interval)
#list_of_ARIMA_objects.append(ao_acf_pacf_dplus_interval)
  
plot_acf_pacf(weekly_boe_xudlerd_log_diff,
              ci,
              b_fft)

print ("number of ARIMA objects to DataCamp...", len(list_of_ARIMA_objects))
best_in_sample_score, best_in_sample_order = float(1000000000.0), None
best_out_sample_score, best_out_sample_order = float(1000000000.0), None
lowest_in_sample_AIC, lowest_in_sample_AIC_order = float(1000000000.0), None

for arima_object in list_of_ARIMA_objects:
    print ("running arima order ", arima_object.get_arima_order())
    
    results_ARIMA = do_ARIMA(weekly_boe_xudlerd_log_diff,
                             arima_object.get_p_value(),   
                             arima_object.get_d_value(), 
                             arima_object.get_q_value())
    is_arima_order_obj = None
    if results_ARIMA == -1:
        # means we had an exception in the ARIMA process.
        print ("Got exception in order ", arima_object.get_arima_order(), " whilst performing in-sample model fit")
        try:
            
            new_q_value = arima_object.get_q_value() - 1
            new_p_value = arima_object.get_p_value()
            new_d_value = arima_object.get_d_value()
            
            is_arima_order_obj = ArimaOrder(new_p_value,
                                            new_d_value,
                                            new_q_value)
            
            print ("Reducing the MA term (Q value) by 1 to check for convergence...(", is_arima_order_obj.get_p_value(), is_arima_order_obj.get_d_value(), is_arima_order_obj.get_q_value()-1, ")")
            
            results_ARIMA = do_ARIMA(weekly_boe_xudlerd_log_diff,
                                     is_arima_order_obj.get_p_value(),
                                     is_arima_order_obj.get_d_value(),
                                     is_arima_order_obj.get_q_value())
            
        except Exception as excp:
            print (excp)
            continue
    
    print ("results ARIMA AIC() ", results_ARIMA.aic)
    if int(results_ARIMA.aic*1000000.0) < int(lowest_in_sample_AIC*1000000.0):
        lowest_in_sample_AIC = results_ARIMA.aic
        lowest_in_sample_AIC_order = arima_object.get_arima_order()
    
    predictions_arima = do_in_sample_predict(results_ARIMA,
                                             weekly_boe_xudlerd_log_diff,
                                             weekly_boe_xudlerd_log)
    in_sample_rmse = plot_in_sample_prediction(predictions_arima,
                                               weekly_boe_xudlerd_data,
                                               arima_object,
                                               differencing_interval)
    if int(in_sample_rmse*1000000.0) < int(best_in_sample_score*1000000.0):
        best_in_sample_score = in_sample_rmse
        if is_arima_order_obj == None:
            best_in_sample_order = arima_object.get_arima_order()
        else:
            best_in_sample_order = is_arima_order_obj
            is_arima_order_obj = None
        
    out_sample_rmse = evaluate_arima_model(weekly_boe_xudlerd_log,
                                           arima_object.get_arima_order(),
                                           0.67,
                                           weekly_boe_xudlerd_data,
                                           differencing_interval)
    os_arima_order_obj = None
    if out_sample_rmse == -1:
        print ("Got exception in order ", arima_object.get_arima_order(), " whilst performing out-of-smaple forecasting")
        try:
            new_q_value = arima_object.get_q_value() - 1 #subtract 1 in case helps with SVD convergence
            new_p_value = arima_object.get_p_value()
            new_d_value = arima_object.get_d_value()
            
            os_arima_order_obj = ArimaOrder(new_p_value,
                                            new_d_value,
                                            new_q_value)
            print ("Reducing the MA term (Q value) by 1 to check for convergence...(", os_arima_order_obj.get_arima_order(), ")")
            
            out_sample_rmse = evaluate_arima_model(weekly_boe_xudlerd_log,
                                                   os_arima_order_obj.get_arima_order(),
                                                   0.67,
                                                   weekly_boe_xudlerd_data,
                                                   differencing_interval)
            
            
        except Exception as excp:
            print (excp)
            continue
    
    if int(out_sample_rmse*1000000.0) < int(best_out_sample_score*1000000.0):
        if os_arima_order_obj == None:
            best_out_sample_order = arima_object.get_arima_order()
        else:
            best_out_sample_order = os_arima_order_obj.get_arima_order()
            os_arima_order_obj = None
        best_out_sample_score = out_sample_rmse

print ("Best out of sample score ", best_out_sample_score, " achieved with order ", best_out_sample_order)
print ("Best in sample score ", best_in_sample_score, " achieved with order ", best_in_sample_order)
print ("Lowest in sample AIC ", lowest_in_sample_AIC, " achieved with order ", lowest_in_sample_AIC_order)