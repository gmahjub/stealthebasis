'''
Created on Nov 22, 2017

@author: traderghazy
'''

from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from math import sqrt
from pandas import DataFrame
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf, arma_order_select_ic
from statsmodels.tsa.tests.results import results_arima


class ArimaOrder(object):
    
    def __init__(self,
                 p,
                 d,
                 q):
        
        self.p = p
        self.d = d
        self.q = q
        self.arima_order = None
        
    def get_arima_order(self):
        
        return (self.p, self.d, self.q)
    
    def set_arima_order(self,
                        p,
                        d,
                        q):
        
        self.arima_order = (p,d,q)

class ArimaObject(object):
    
    def __init__(self,
                 p_value,
                 d_value,
                 q_value,
                 p_value_method = None,
                 q_value_method = None,
                 sampling_period = None,
                 include_diff_from_stationarity=False,
                 differencing_interval=1):
        
        self.p_value = p_value
        if (include_diff_from_stationarity):
            self.d_value = d_value + differencing_interval
        else:
            self.d_value = d_value
        self.q_value = q_value
        
        self.p_value_method = p_value_method
        self.q_value_method = q_value_method
        self.sampling_period = sampling_period
        self.include_diff_from_stationarity = include_diff_from_stationarity
        self.differencing_interval = differencing_interval
    
        # these are set once the arima is run
        self.predicitons = None
        self.history = None
        self.validation_dataset_predictions = None
        self.validation_dataset_history = None
        
        self.ARIMA_residuals = None
        self.ARIMA_results = None
        self.ARIMA_rss = None
        self.ARIMA_aic = np.inf
        self.in_smaple_rmse = 0.0
        self.out_sample_rmse = 0.0
    
    def get_arima_order(self):
        
        ret_p_val = self.get_p_value()
        ret_d_val = self.get_d_value()
        ret_q_val = self.get_q_value()
        
        return (ret_p_val, ret_d_val, ret_q_val)
    
    def set_arima_order(self,
                        p_value,
                        d_value,
                        q_value):
        
        self.p_value = p_value
        self.d_value = d_value
        self.q_value = q_value
    
    def set_p_value(self,
                    p_value):
        
        self.p_value = p_value
        
    def get_p_value(self):
    
        return self.p_value
    
    def set_d_value(self,
                    d_value):
        
        self.d_value = d_value
    
    def get_d_value(self):
        
        return self.d_value
    
    def set_q_value(self,
                    q_value):
        
        self.q_value = q_value
        
    def get_q_value(self):
        
        return self.q_value
        
    def set_p_value_method(self,
                           p_value_method):
        
        self.p_value_method = p_value_method
        
    def set_q_value_method(self,
                           q_value_mehtod):
        
        self.q_value_method = q_value_mehtod
    
    def set_seasonality_method(self,
                               sampling_period,
                               differencing_interval,
                               include_diff_from_stationarity=False):
        
        self.sampling_period = sampling_period
        self.include_diff_from_stationarity = include_diff_from_stationarity
        self.differencing_interval = differencing_interval
    
    # here we are doing out of sample forecasting
    def evaluate_arima_model(self,
                             timeseries_values, 
                             pct_training_data,
                             num_steps=1,
                             differencing_interval=1):
        
        # timeseries_values is a series.values numpy array
        train_size = int(len(timeseries_values)*pct_training_data)
        train, test = timeseries_values[0:train_size], timeseries_values[train_size:]
        history = [x for x in train]
    
        predictions = list()
    
        for t in range(len(test)):
            diffed_data = self.difference_composition(history, interval=1)#timeseries_values - timeseries_values.shift(periods=differencing_interval)
            model = ARIMA(diffed_data, order=self.get_arima_order())
            model_fit = model.fit(disp=0) # trend='c' is the default, other option is trend='nc' 
            yhat, err95, ci95 = model_fit.forecast()[0]
            # we have to invert the difference here, using same interval as when we difference for stationarity
            yhat = self.reverse_difference_decomposition(history, yhat, differencing_interval)
            predictions.append(yhat[0])
            obs = test[t]
            history.append(obs)
            #print ('predicted=%f, expected=%f, date=%s' % (np.exp(yhat), np.exp(obs)))
        error = mean_squared_error(test, predictions)
        rmse = sqrt(error)
        
        self.predictions = predictions
        self.history = history
        self.out_smaple_rmse = rmse
        
        return rmse
    
    def calculate_residuals(self,
                            timeseries_values,
                            pct_training_data):
        
        train_size = int(len(timeseries_values)*pct_training_data)
        _, test = timeseries_values[0:train_size], timeseries_values[train_size:]
        
        residuals = [test[i]-self.predictions[i] for i in range(len(test))]
        residuals = DataFrame(residuals)
        
        return residuals
    
    def finalize_model(self,
                       model_fit,
                       fitted_model_name,
                       bias_value=0.0):
        
        model_fit.save(fitted_model_name + '.pkl')
        np.save(fitted_model_name + '_bias.npy', [bias_value])
        
    def predict_model(self,
                      timeseries_series,
                      fitted_model_name,
                      num_steps = 1,
                      differencing_interval=1):
        
        model_fit = ARIMAResults.load(fitted_model_name + '.pkl')
        bias = np.load(fitted_model_name + '_bias.npy')
        yhat, err95, ci95 = model_fit.forecast(steps=num_steps)
        yhat = bias + self.reverse_difference_decomposition(timeseries_series.values, yhat, differencing_interval)
        return yhat
    
    def do_in_sample_predict(self,
                             results_ARIMA,
                             stationary_timeseries,
                             original_timeseries):
    
        predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
        predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
        predictions_ARIMA_log = pd.Series(original_timeseries.iloc[0].values, index=stationary_timeseries.index)
        predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
        predictions_ARIMA = np.exp(predictions_ARIMA_log)
        
        rmse = np.sqrt(sum((predictions_ARIMA-original_timeseries.iloc[1:,0])**2)/len(original_timeseries))
        self.in_sample_rmse = rmse
    
        return predictions_ARIMA
    
    def validate_model(self,
                       validation_timeseries_series,
                       timeseries_series,
                       fitted_model_name,
                       differencing_interval=1):
        
        timeseries_values = timeseries_series.values.astype('float32')
        history = [x for x in timeseries_values]
        validation_timeseries_values = validation_timeseries_series.values.astype('float32')
        model_fit = ARIMAResults.load(fitted_model_name + '.pkl')
        bias = np.load(fitted_model_name + '_bias.npy')
        predictions = list()
        yhat = float(model_fit.forecast()[0])
        yhat = bias + self.reverse_difference_decomposition(history, yhat, differencing_interval)
        predictions.append(yhat)
        history.append(validation_timeseries_values[0])
        
        print ('>Predicted=%.6f, Expected=%6.f' % (yhat, validation_timeseries_values[0]))
        
        for i in range(1, len(validation_timeseries_values)):
            diff = self.difference_composition(history, differencing_interval)
            # predict
            model = ARIMA(diff, order=self.get_arima_order() )
            model_fit = model.fit(trend='nc', disp=0)
            yhat = model_fit.forecast()[0]
            yhat = bias + self.reverse_difference_decomposition(history, yhat, differencing_interval)
            predictions.append(yhat)
            # observation
            obs = validation_timeseries_values[i]
            history.append(obs)
            print ('>Predicted=%.6f, Expected=%6.f' % (yhat, obs))
            
        self.validation_dataset_predictions = predictions
        self.validation_dataset_history = history
        
        mse = mean_squared_error(validation_timeseries_values, predictions)
        rmse = sqrt(mse)
        return rmse
    
    def do_ARIMA(self,
                 timeseries,
                 fit_method): # mle, ols
    
        # ARIMA tuple is in the following format... ARIMA(P,D,Q)
        model = ARIMA(timeseries, order=self.get_arima_order())
        results_ARIMA = model.fit(method=fit_method, disp=-1) # trend can be 'nc' also, no constant trend added
        rss = sum((results_ARIMA.fittedvalues-timeseries.iloc[self.get_d_value():,0])**2) # residual sum of square
        rmse = np.sqrt(rss/len(timeseries))
        residuals = pd.DataFrame(results_ARIMA.resid)    
        
        self.ARIMA_results = self.set_ARIMA_Results(results_ARIMA)
        self.ARIMA_residuals = residuals
        self.ARIMA_rss = rss
        self.ARIMA_aic = model.aic
        
        return results_ARIMA
    
    def set_ARIMA_results(self,
                          results_ARIMA):
        
        self.ARIMA_results = results_arima
        
    def get_ARIMA_results(self):
        
        return self.ARIMA_results
    
    def set_ARIMA_residuals(self,
                            ARIMA_residuals):
        
        self.ARIMA_residuals = ARIMA_residuals
        
    def get_ARIMA_residuals(self):
        
        return self.ARIMA_residuals
            
    # reverse the difference decomposition
    def reverse_difference_decomposition(self, 
                                         history, 
                                         yhat, 
                                         interval=1):
        
        return yhat + history[-interval]
    
    def difference_composition(self,
                               dataset,
                               interval=1):
        
        to_return = np.array(dataset) - np.roll(np.array(dataset), interval)
        return to_return[interval:]
        
import pandas as pd
    
class ArimaImplObject(object):
    
    def __init__(self):
        
        i = 1
        
    def get_pq_tuple(self,
                     timeseries,
                     ci,
                     b_fft):
        # nlags tells us to DataCamp up to 10 lags, meaning a variable and itself up to 10 steps ago.
        # so acf() DataCamp correlation of a variable with itself up to nlags ago
        # and pacf() computes the correlation at each lag step that is not already explained by lower order lags.
        lag_acf, confint_acf, qstat, pvalues = acf(timeseries, qstat=True, nlags=10, alpha=(1-ci), fft=b_fft) # there are other arguements, document them.\
        pos_interval_acf = lag_acf - confint_acf[:,0]
        neg_interval_acf = -1*(confint_acf[:,1] - lag_acf)
        lag_pacf, confint_pacf = pacf(timeseries, nlags=10, method='ols', alpha=(1-ci)) # document any other parameters.
        pos_interval_pacf = lag_pacf - confint_pacf[:,0]
        neg_interval_pacf = -1*(confint_pacf[:,1] - lag_pacf)
        pq_tuple = self.find_p_and_q_values_from_plots(pos_interval_acf,
                                                       neg_interval_acf,
                                                       lag_acf,   
                                                       lag_pacf)

        return pq_tuple
    
    def find_p_and_q_values_from_plots(self,
                                       pos_interval_acf,
                                       neg_interval_acf,
                                       lag_acf,
                                       lag_pacf):
        
        # first, find lag_acf intercept with upper CI band (pos_interval_acf = Q value
        # second, find lac_pacf intercept with 0 band = P value
        Q = 0
        P = 0
        i=0
        while i < len(pos_interval_acf):
            if int(pos_interval_acf[i]*1000000.0) >= int(lag_acf[i]*1000000.0):
            # this means lag_acf is less than pos_interval_acf, this value is the Q value
                Q = i
                break
            else:
                i+=1
        i=0
        while i < len(lag_pacf):
            if int(lag_pacf[i]*1000000.0) <= 0.0:
            # this means lac_pacf is less than 0, this value is the P value
                P = i
                break
            else:
                i+=1   
        return (P,Q)

    def percent_quantile_function(self,
                                  ci):
        
        return stats.norm.ppf(1-(1-ci)/2.)
    
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
    def find_ARIMA_params_automated(self,
                                    timeseries,
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

    def hyper_parameter_grid_search(self,
                                    list_of_ARIMA_objects,
                                    timeseries_series,
                                    pct_training_data):
        best_score = 1000000.0
        best_cfg = None
        return_ao_list = []
        np_timeseries_values = timeseries_series.values()
        timeseries_values = np_timeseries_values.astype('float32')
        
        for ao in list_of_ARIMA_objects:
            order = ao.get_arima_order()
            try:
                rmse = ao.evaluate_arima_model(timeseries_values,
                                               pct_training_data)
                if int(rmse*1000000.0) < int(best_score*1000000.0):
                    best_score, best_cfg = rmse, order
            except Exception as excp:
                ao.exception = excp
                return_ao_list.append(ao)
                continue
            ao.rmse = rmse
            return_ao_list.append(ao)
        
        return_dict = {}
        return_dict['arima_obj_list'] = return_ao_list
        return_dict['best_cfg'] = best_cfg
        return_dict['best_score'] = best_score
        
        return return_dict
    
    def create_lags(self,
                    array_of_lags,
                    timeseries_series):
        
        return_dict = {}
        for lag in array_of_lags:
            return_dict['Lag ' + str(lag)] = timeseries_series.shift(int(lag))
        
        return return_dict
    
    