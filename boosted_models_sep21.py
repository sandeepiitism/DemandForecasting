from .prediction_models import *
import pandas as pd
from datetime import datetime
from ..preprocessing.preprocessing import *
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta
# import joblib
# import sys
# sys.modules['sklearn.externals.joblib'] = joblib
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import hp
from hyperopt import STATUS_OK
from timeit import default_timer as timer
import pdb as pdb
import six
import sys
#sys.modules['sklearn.externals.six'] = six
#import mlrose
#import joblib


#from pmdarima.arima import auto_arima 
 
class tree_forecast(Prediction_model):
    ''' Its a base class for XGBoost and GBM for creating baseline model and training data '''
    
    def __init__(self, series_df, detail_df, fh, model_gov,tune_gov,params):
        ''' Initialise the variables for creating training data from the config file '''
        Prediction_model.__init__(self, series_df, detail_df, fh, model_gov,tune_gov,params)
        self.lag_start = model_gov['lag-start']
        self.lag_end = model_gov['lag-end']
        self._iterations = model_gov['iterations']
        self.min_training_data = model_gov['min-data-sku']
        self.cv = model_gov['cv']
 
    def create_feature(self):
        ''' Create lags for time series data to be given as input to the model '''
        data = pd.DataFrame(self.series_df.copy())
        data.columns = ["y"]
 
        # lags of series
        for i in range(self.lag_start, self.lag_end+1):
            data["lag_{}".format(i)] = data.y.shift(i)
 
        # datetime features
        # data.index = data.index.map(change_index)
        data.dropna(inplace=True)
        data = data.astype('int64')
        data = np.log(data+1)
        return data
 
    def get_model(self):
        ''' Create the structure of model '''
        pass
 
    def predict(self):
        ''' Training the model and predicting forecast using the trained model.
        Create training data by calling create_feature function from tree class and then calling the created model from get_model 
        to train it using time series data and then using the trained model to predict forecast for the same time-series.'''
        
        data = self.create_feature()
        
        self.y = data.dropna().y
        self.X = data.dropna().drop(['y'], axis=1)
        
        model = self.get_model()
        
        result = model.fit(self.X.values, self.y.values)
        pred = []
        data1 = data.copy()
        
        for i in range(self.fh):
            t = data1.iloc[-1, 0:-1].values
            temp = result.predict(t.reshape(1, -1))[0]
            if temp <= 0:
                temp = 0
            pred.append(
                temp
            )
            data1.loc[data1.index[-1] + relativedelta(months=+1)] = \
                np.concatenate((np.array(pred[-1]).reshape(1, -1), np.array(t).reshape(1, -1)), axis=1)[0]
        pred = np.exp(pred)
 
        return pd.DataFrame(np.array(pred).reshape(1, -1))
 
class xgb_forecast(tree_forecast):
    ''' Class for XGBoost model to create the XGBoost model structure and calling tree class predict function to predict the forecast '''
    
    def __init__(self, series_df, detail_df, fh, model_gov,tune_gov,params):
        ''' Call init function from tree clas to initialise all the variables '''
        tree_forecast.__init__(self, series_df, detail_df, fh, model_gov,tune_gov,params)
 
    def get_model(self):
        ''' Create model structure for XGBoost using BayesSearchCV which takes multiple hyperparameters and select the optimum ones '''
       
        return BayesSearchCV(
            estimator=XGBRegressor(
                objective='reg:linear',
                booster='gbtree',
                n_jobs=1,
                eval_metric='rmse',  # logloss
                silent=True,
                tree_method='approx',
                verbose=1,
            ),
            
            search_spaces={
                'learning_rate': (0.001, 1.0, 'log-uniform'),
                'min_child_weight': (0, 10),
                'max_depth': (0, 50),
                'max_delta_step': (0, 20),
                'subsample': (0.01, 1.0, 'uniform'),
                'colsample_bytree': (0.01, 1.0, 'uniform'),
                'colsample_bylevel': (0.01, 1.0, 'uniform'),
                'reg_lambda': (1e-9, 1000, 'log-uniform'),
                'reg_alpha': (1e-9, 1.0, 'log-uniform'),
                'gamma': (1e-9, 0.5, 'log-uniform'),
                'min_child_weight': (0, 5),
                'n_estimators': (50, 100),
                'scale_pos_weight': (1e-5, 500, 'log-uniform')
            },
            scoring='neg_mean_squared_error',  # neg_mean_squared_log_error
            cv=TimeSeriesSplit(n_splits=self.cv),
            n_jobs=-1,
            n_iter=self._iterations,
            verbose=0,
            refit=True,
            random_state=42
        )

class gbm_forecast(tree_forecast):
    def __init__(self, series_df, detail_df, fh, model_gov,tune_gov,params):
        tree_forecast.__init__(self, series_df, detail_df, fh, model_gov,tune_gov,params)
        self.MAX_EVALS = 10
 
    def objective(self, params, n_folds=5):
 
        self.ITERATION += 1
 
        # Retrieve the subsample if present otherwise set to 1.0
        subsample = params['boosting_type'].get('subsample', 1.0)
 
        # Extract the boosting type
        params['boosting_type'] = params['boosting_type']['boosting_type']
        params['subsample'] = subsample
 
        # Make sure parameters that need to be integers are integers
        for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
            params[parameter_name] = int(params[parameter_name])
 
        start = timer()
 
        # Perform n_folds cross validation
        cv_results = lgb.cv(params, self.train_set, num_boost_round=1000, nfold=n_folds,
                            early_stopping_rounds=100, metrics='rmse', seed=50, stratified=False)
 
        run_time = timer() - start
 
        # Extract the best score
        best_score = np.min(cv_results['rmse-mean'])
 
        # Loss must be minimized
        loss = best_score
 
        # Boosting rounds that returned the minimum cv score
        n_estimators = int(np.argmin(cv_results['rmse-mean']) + 1)
 
        return {'loss': loss, 'params': params, 'iteration': self.ITERATION,
                'estimators': n_estimators,
                'train_time': run_time, 'status': STATUS_OK}
 
    def tune(self):
        space = {
            'class_weight': hp.choice('class_weight', [None, 'balanced']),
            'boosting_type': hp.choice('boosting_type',
                                       [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                        # {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                        {'boosting_type': 'goss', 'subsample': 1.0}]),
            'num_leaves': hp.quniform('num_leaves', 5, 100, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
            'min_child_samples': hp.quniform('min_child_samples', 5, 500, 5),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
            'num_rounds': 50000,
            'metric': 'rmse'
        }
 
        # Keep track of results
        bayes_trials = Trials()
 
        self.ITERATION = 0
 
        # Run optimization
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest,
                    max_evals=self.MAX_EVALS, trials=bayes_trials, rstate=np.random.RandomState(50), verbose=1)
 
        bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])
 
        best_model = bayes_trials_results[0]
 
        return best_model
 
    def get_model(self):
        self.train_set = lgb.Dataset(self.X, self.y, silent=False)
 
        best_model = self.tune()
        best_bayes_estimators = int(best_model['estimators'])
        best_bayes_params = best_model['params']
        # print(best_bayes_params)
 
        best_bayes_model = lgb.LGBMRegressor(n_estimators=best_bayes_estimators, # n_jobs=-1,
                                             objective='regression', random_state=50, **best_bayes_params)
 
        return best_bayes_model