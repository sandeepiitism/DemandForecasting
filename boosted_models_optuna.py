from .prediction_models import *
import pandas as pd
from datetime import datetime
from ..preprocessing.preprocessing import *
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from hyperopt import tpe, hp, fmin, Trials, STATUS_OK
from timeit import default_timer as timer
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score

class tree_forecast(Prediction_model):
    def __init__(self, series_df, detail_df, fh, model_gov, tune_gov, params):
        Prediction_model.__init__(self, series_df, detail_df, fh, model_gov, tune_gov, params)
        self.lag_start = model_gov['lag-start']
        self.lag_end = model_gov['lag-end']
        self._iterations = model_gov['iterations']
        self.min_training_data = model_gov['min-data-sku']
        self.cv = model_gov['cv']
        self.freq = model_gov['frequency']

    def create_feature(self):
        data = pd.DataFrame(self.series_df.copy())
        data.columns = ["y"]

        for i in range(self.lag_start, self.lag_end + 1):
            data["lag_{}".format(i)] = data.y.shift(i)

        data.dropna(inplace=True)
        data = data.astype('int64')
        data = np.log(data + 1)
        return data

    def get_model(self):
        pass

    def predict(self):
        if self.freq == 52:
            dataset = pd.DataFrame()
            dataset['ds'] = 1
            for i in range(len(self.series_df)):
                year = int(self.series_df.index[i][3:])
                week = int(self.series_df.index[i][:2])
                datetime = Week(year, week).monday()
                dataset.at[i, 'ds'] = datetime

            self.series_df.index = dataset['ds']

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
            pred.append(temp)
            data1.loc[data1.index[-1] + relativedelta(weeks=+1)] = \
                np.concatenate((np.array(pred[-1]).reshape(1, -1), np.array(t).reshape(1, -1)), axis=1)[0]
        pred = np.exp(pred)

        return pd.DataFrame(np.array(pred).reshape(1, -1))

class xgb_forecast(tree_forecast):
    def __init__(self, series_df, detail_df, fh, model_gov, tune_gov, params):
        tree_forecast.__init__(self, series_df, detail_df, fh, model_gov, tune_gov, params)

    def get_model(self):
        return XGBRegressor(
            objective='reg:squarederror',
            booster='gbtree',
            n_jobs=1,
            eval_metric='rmse',
            tree_method='approx',
        )

    def objective(self, trial, X, y):
        model = self.get_model()
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 0, 10),
            'max_depth': trial.suggest_int('max_depth', 0, 50),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 20),
            'subsample': trial.suggest_uniform('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.01, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-9, 1000),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-9, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 1e-9, 0.5),
            'min_child_weight': trial.suggest_int('min_child_weight', 0, 5),
            'n_estimators': trial.suggest_int('n_estimators', 10, 100),
            'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 1e-5, 500),
        }

        model.set_params(**params)

        cv_results = -cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=self.cv),
                                     scoring='neg_mean_squared_error', n_jobs=-1)

        return np.mean(cv_results)

    def predict(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self._iterations)

        best_params = study.best_params
        best_model = self.get_model().set_params(**best_params)
        best_model.fit(self.X.values, self.y.values)

        pred = []
        data1 = self.X.copy()

        for i in range(self.fh):
            t = data1.iloc[-1, :].values.reshape(1, -1)
            temp = best_model.predict(t)[0]
            if temp <= 0:
                temp = 0
            pred.append(temp)
            data1.loc[data1.index[-1] + relativedelta(weeks=+1)] = \
                np.concatenate((np.array(pred[-1]).reshape(1, -1), np.array(t).reshape(1, -1)), axis=1)[0]

        pred = np.exp(pred)

        return pd.DataFrame(np.array(pred).reshape(1, -1))