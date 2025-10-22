from .prediction_models import *
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import prophet
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.robjects as ro
from ..preprocessing.preprocessing import prophet_data_preperation
import numpy as np
import pdb as pdb
import six
import sys
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from numpy import array
#sys.modules['sklearn.externals.six'] = six
#import mlrose
#import joblib

#sys.modules['sklearn.externals.joblib'] = joblib
#from pmdarima.arima import auto_arima 
#from pyramid.arima import auto_arima
# grid search holt winter's exponential smoothing
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from numpy import array
import ast
import itertools
from tqdm import tqdm
import logging 
logging.getLogger('prophet').setLevel(logging.ERROR)
# from prophet import Prophet
import pandas as pd
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# one-step Holt Winter’s Exponential Smoothing forecast
def exp_smoothing_forecast(history, cfg, seasonality = 'add'):
    ''' Creates Exponential Smoothing model for additive and multiplicative smoothing tuning module.
         Input - time-series, set of parameters, seasonality mode (add or mul)
         Output - Predicted forecast'''
      
    damped = cfg['damped']
    
    residual_bias = cfg['residual_bias']
    
    if(cfg['trend'] == 'add'):
        if (seasonality == 'add'):
            
            model = ExponentialSmoothing(history, trend='add', damped_trend=damped, 
                                 seasonal='add', seasonal_periods = 12)
        else:
            model = ExponentialSmoothing(history, trend='add', damped_trend=damped, 
                                 seasonal='mul', seasonal_periods = 12,initialization_method='estimated')
    else:
        if (seasonality == 'add'):
            
            model = ExponentialSmoothing(history, trend='mul', damped_trend=damped, 
                                 seasonal='add', seasonal_periods = 12,initialization_method='estimated')
        else:
            model = ExponentialSmoothing(history, trend='mul', damped_trend=damped, 
                                 seasonal='mul', seasonal_periods = 12,initialization_method='estimated')
    
    # fit model
    model_fit = model.fit(optimized=True,  remove_bias=residual_bias)
    
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    
    return yhat[0]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
    ''' To calculate RMSE of actual and prediced values to determine the set of optimal parameters'''
    return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    ''' To split the time-series into train and test set to evaluate the performance of each set of hyperparameters '''
    return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg, seasonality):
    ''' To calculate RMSE error metric to validate the performance of actual values from test set with the predicted values for each set
        of parameter values.
        Input - Time-series, number of data points to be kept in test set, set of parameters, seasonality mode
        Output - Error metric values for actual and predicted values'''
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
#         print(cfg)
        yhat = exp_smoothing_forecast(history, cfg, seasonality)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error

# score a model, return None on failure
def score_model(data, n_test, cfg, seasonality):
    ''' To calculate error metric for each set of combination of the hyperparameters
        Input - Time-series, no.of data points in test set, combination of parameters, seasonality mode
        Output - A combination of each parameter set and its respective error metric'''
    debug=False
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg, seasonality)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg, seasonality)
        except:
            error = None
    
    return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, seasonality, parallel=True):
    ''' To train model on time-series for each set of hyperparameters in a grid serach fashion.
        Input - Time-series, no.of data points in test set, combination of parameters, seasonality mode
        Output - Top scores i.e. lowest error metrics for the most optimal set of hyperparameters'''
    
    scores = None
    scores = [score_model(data, n_test, cfg_list[i], seasonality) for i in range(len(cfg_list))]
        
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores



def work(series_df, param_grid, seasonality = 'add',n_test=12):
    ''' To create combination of hyperparameters and pass it to model training for grid search of the best hyperparameters'''
    # define dataset
    
    data = series_df.values
    cfg_list = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    # grid search
    scores = grid_search(data, cfg_list, n_test, seasonality)

    return scores[0]

class Additive_smoothing(Prediction_model):
    ''' Class for initializing and creating the model structure for Additive Exponential Smoothing'''

    def __init__(self, series_df, detail_df, fh, model_gov,tune_gov,params):
        ''' To initialise the class variables taken from config file'''
        Prediction_model.__init__(self, series_df, detail_df, fh, model_gov,tune_gov,params)
        self.freq = self.model_gov["frequency"]
        self.fh = fh
        if self.freq*2 > self.series_df.shape[0]:
            self.freq = int(self.series_df.shape[0] / 3 )
        if self.freq == 0:
            self.freq = 1
        self._name = "Holt-Winters Additive Exponential Smoothing"
        self.trend_tune = self.tune_gov['trend']
        self.damped_tune = self.tune_gov['damped']
        self.residual_bias_tune = self.tune_gov['residual_bias']
        
        if(self.params is None):
            pass
        elif(self.params.empty):
            self.trend = 'add'
            self.damped = True
            self.residual_bias = False
        else:
            self.trend = self.params['trend'].iloc[0]
            self.damped = self.params['damped'].iloc[0]
            self.residual_bias = self.params['residual_bias'].iloc[0]

    def tune(self):
        ''' Tuning the model for best hyperparameters'''
        
        param_grid = {  'trend': self.trend_tune, # 'growth': ["linear"."logistic"]
                'damped': self.damped_tune,
#                 'seasonality': ["mul"],
#                 'use_boxcox': [True, False],
                'residual_bias': self.residual_bias_tune,
                
              }
        best_score_add = work(self.series_df,param_grid,'add')
        best_fit_add = ast.literal_eval(best_score_add[0])
        return best_fit_add

        
    def predict(self):
        ''' Predicting the forecast horizon by training the model as per the best hyperparameters taken fro the tuned model'''
        
        warnings.simplefilter('ignore', ConvergenceWarning)
        model = ExponentialSmoothing(np.asarray(self.series_df), trend=self.trend, seasonal="add", damped_trend = self.damped, seasonal_periods=self.freq,initialization_method='estimated').fit(optimized=True, remove_bias = self.residual_bias)
        
#         model = ExponentialSmoothing(np.asarray(self.series_df), trend='add', seasonal="add", seasonal_periods=self.freq).fit()
        
        pred = model.predict(start=len(self.series_df), end=len(self.series_df) + (self.fh - 1))
        pred[pred < 0] = 0
        #print(self._name," done")
        return pd.DataFrame(pred.reshape(1, -1))


class Multiplicative_smoothing(Prediction_model):
    ''' Class for initializing and creating the model structure for Multiplicative Exponential Smoothing'''

    def __init__(self, series_df, detail_df, fh, model_gov,tune_gov,params):
        ''' To initialise the class variables taken from config file'''
        Prediction_model.__init__(self, series_df, detail_df, fh, model_gov,tune_gov,params)
        self.freq = self.model_gov["frequency"]
        if self.freq*2 > self.series_df.shape[0]:
            self.freq = int(self.series_df.shape[0] / 3 )
        if self.freq == 0:
            self.freq = 1
        self.fh = fh
        self._name = "Holt-Winters Multiplicative Exponential Smoothing"
        self.trend_tune = self.tune_gov['trend']
        self.damped_tune = self.tune_gov['damped']
        self.residual_bias_tune = self.tune_gov['residual_bias']
        
        if(self.params is None):
            pass
        elif(self.params.empty):
            self.trend = 'add'
            self.damped = True
            self.residual_bias = True
        else:
            self.trend = self.params['trend'].iloc[0]
            self.damped = self.params['damped'].iloc[0]
            self.residual_bias = self.params['residual_bias'].iloc[0]
        

    def tune(self):
        ''' Tuning the model for best hyperparameters'''
        
        param_grid = {  'trend': self.trend_tune, # 'growth': ["linear"."logistic"]
                'damped': self.damped_tune,
#                 'seasonality': ["mul"],
#                 'use_boxcox': [True, False],
                'residual_bias': self.residual_bias_tune,
                
              }
        best_score_add = work(self.series_df,param_grid,'mul')
        
        best_fit_add = ast.literal_eval(best_score_add[0])
        return best_fit_add
    
    def predict(self):
        ''' Predicting the forecast horizon by training the model as per the best hyperparameters taken fro the tuned model'''
        # Changed in enhancements
        # print('Before Testing this function on zero values',self.series_df[self.series_df==0].shape)
        # print('Before Testing this function on negative values',self.series_df[self.series_df<0].shape)
        series_df = self.series_df[self.series_df>0]
        # print('After Testing this function on zero values',series_df[series_df==0].shape)
        # print('After Testing this function on negative values',series_df[series_df<0].shape)
        # print(self.residual_bias)
        warnings.simplefilter('ignore', ConvergenceWarning)
        model = ExponentialSmoothing(np.asarray(series_df), trend=self.trend, seasonal="mul", damped_trend = self.damped, seasonal_periods=self.freq,initialization_method='estimated').fit(optimized=True, remove_bias = self.residual_bias)
#         model = ExponentialSmoothing(np.asarray(self.series_df), trend='add', seasonal="mul", seasonal_periods=self.freq).fit()
        pred = model.predict(start=len(self.series_df), end=len(self.series_df) + (self.fh - 1))
        pred[pred < 0] = 0

        return pd.DataFrame(pred.reshape(1, -1))


class Auto_arima(Prediction_model):
    ''' Class for initializing and creating the model structure for Auto-ARIMA model'''

    def __init__(self, series_df, detail_df, fh, model_gov, tune_gov,params):
        ''' To initialise the class variables taken from config file'''
        Prediction_model.__init__(self, series_df, detail_df, fh, model_gov,tune_gov,params)
        self.freq = self.model_gov["frequency"]
        if self.freq*2 > self.series_df.shape[0]:
            self.freq = int(self.series_df.shape[0] / 3 )
        if self.freq == 0:
            self.freq = 1
        self.fh = fh
        self._name = "Auto-ARIMA"
        
# As Auto-ARIMA selects the best hyperparameters as per the time-series, there is no need for tuning an Arima model hence no tune module

    def predict(self):
        ''' Predicting the forecast horizon by training the model as per the best hyperparameters taken fro the tuned model'''
        #from pyramid.arima import auto_arima
        #sys.modules['sklearn.externals.joblib'] = joblib
        from pmdarima.arima import auto_arima
        stepwise_model = auto_arima(self.series_df, start_p=1, start_q=1,max_p=3, max_q=3, m=self.freq, start_P=0,d=1, D=1,error_action='ignore')
        
#         stepwise_model = auto_arima(self.series_df, m=self.freq, error_action='ignore')
        stepwise_model.fit(self.series_df)

        auto_arima_pred = stepwise_model.predict(n_periods=self.fh)
        
        auto_arima_pred[auto_arima_pred < 0] = 0
        return pd.DataFrame(auto_arima_pred.values.reshape(1,-1))
#         return pd.DataFrame(auto_arima_pred).reshape(1,-1)




class Prophet(Prediction_model):
    ''' Class for initializing and creating the model structure for FB Prophet model'''

    def __init__(self, series_df, detail_df, fh, model_gov,tune_gov,params):
        ''' To initialise the class variables taken from config file'''
        Prediction_model.__init__(self, series_df, detail_df, fh, model_gov,tune_gov,params)
        self.freq = self.model_gov["frequency"]
        print(self.freq)
        if self.freq*2 > self.series_df.shape[0]:
            self.freq = int(self.series_df.shape[0] / 3 )
        if self.freq == 0:
            self.freq = 1
        print(self.freq)
        self.fh = fh
        self._name = "Prophet"
        self.cap_power = self.model_gov["cap-power"]
        self.growth_tune = self.tune_gov['growth']
        self.changepoint_prior_scale_tune = self.tune_gov['changepoint_prior_scale']
        self.seasonality_prior_scale_tune = self.tune_gov['seasonality_prior_scale']
        self.seasonality_mode_tune = self.tune_gov['seasonality_mode']
        
        if(self.params is None):
            pass
        elif(self.params.empty):
            self.growth = 'logistic'
            self.changepoint_prior_scale = 0.5
            self.seasonality_prior_scale = 0.5
            self.seasonality_mode = 'additive'
        else:
            self.growth = self.params['growth'].iloc[0]
            self.changepoint_prior_scale = self.params['changepoint_prior_scale'].iloc[0]
            self.seasonality_prior_scale = self.params['seasonality_prior_scale'].iloc[0]
            self.seasonality_mode = self.params['seasonality_mode'].iloc[0]

    def tune(self):
        ''' Tuning the model for best hyperparameters given in the model_tuning config file from where parameters can be changed.
             '''
        cap_power = 2
        
        df_prophet, _, scaler_to_fix_decimal = prophet_data_preperation(self.series_df, self.freq, self.cap_power)
#         if(df_prophet.empty):
#             return pd.DataFrame()
#         if np.isnan(df_prophet[-1]) or ( len(df_prophet)>3 and series_df[-1] == 0 and series_df[-2] == 0 and series_df[-3] == 0 ):
#         print("Info: Last 3 month demand missing for {} {}, returning".format(material_name, market_code))
#         return
            
        
        param_grid = {  'growth': self.growth_tune, # 'growth': ["linear"."logistic"]
                        'changepoint_prior_scale': self.changepoint_prior_scale_tune,
                        'seasonality_prior_scale': self.seasonality_prior_scale_tune,
                        'yearly_seasonality': ['auto'],
                        'weekly_seasonality': ['auto'],
                        'daily_seasonality': [False],
                        'seasonality_mode': self.seasonality_mode_tune #'seasonality_mode': ["additive","multiplicative"]
                      }
        
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here
        # can replace with mape too, automatically calculated in the performance metric package
        
        # Use cross validation to evaluate all parameters
        
        for params in tqdm(all_params):
            
            m = prophet.Prophet(**params).fit(df_prophet)  # Fit model with given params
            # use initial long enough to capture all of the components of the model, may be all but last 18 months for our use case
            # change horizon from "days" to "months" for our data , period can be kept as optional
            # rolling window prop : By default 10% of the predictions will be included in each window,i.e.0.1
            a  = (df_prophet.index[[-1]] - df_prophet.index[[0]])[0]
            
            df_cv = prophet.diagnostics.cross_validation(m, initial='{} days'.format(a.days-365),  horizon='365 days', parallel = 'threads')
            
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])
            
        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
#         print(tuning_results)
        best_params = all_params[np.argmin(rmses)]
    
        return best_params

    def predict(self):
        ''' Predicting the forecast horizon by training the model as per the best hyperparameters taken fro the tuned model.
            Prophet model needs data in a particular format which is done in prophet_data_preperation function.
            For logistic growth parameter, cap variable is needed to set a limit which is defined as 2 in the config file.'''

        df_prophet, _, scaler_to_fix_decimal = prophet_data_preperation(self.series_df, self.freq, self.cap_power)

        m = prophet.Prophet(growth=self.growth, changepoint_prior_scale=self.changepoint_prior_scale,seasonality_prior_scale=self.seasonality_prior_scale,daily_seasonality=False, weekly_seasonality='auto',yearly_seasonality='auto',seasonality_mode=self.seasonality_mode)

#         m = prophet.Prophet(growth='logistic', changepoint_prior_scale=0.5, daily_seasonality=False, weekly_seasonality=False)
        m.fit(df_prophet)

        if self.freq == 12:
            future = m.make_future_dataframe(periods=self.fh, freq='M')
        elif self.freq == 52:
            future = m.make_future_dataframe(periods=self.fh, freq='W')
        #else:
        #    print('Please designate Frequency')
        future['cap'] = int(int(df_prophet['y'].max()) * self.cap_power)
        future['floor'] = 0

        fcst = m.predict(future)
        fcst1 = fcst['yhat']
        df_pred = fcst1.reset_index(drop=True)
        df_pred[df_pred < 0] = 0

        df_pred = df_pred / scaler_to_fix_decimal
        return pd.DataFrame(df_pred[-self.fh:].reset_index(drop=True).values.reshape(1,-1))