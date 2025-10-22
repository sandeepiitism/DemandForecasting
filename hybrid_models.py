import numpy as np
import pandas as pd
import prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from ..preprocessing.preprocessing import prophet_data_preperation
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.model_selection import GridSearchCV
import six
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout,Dense, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import kerastuner
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import BayesianOptimization
import os
import shutil
import tensorflow as tf
tf.compat.v1.set_random_seed(1)

from autopilot_planning.fetch_data import *
from autopilot_planning.write_data import *
from multiprocessing import Process, Manager, Pool, cpu_count
from autopilot_planning import Autopilot, fetch_data, write_data
from autopilot_planning.preprocessing.preprocessing import *
from autopilot_planning.config.config import *
from autopilot_planning.config.config import get_db_engine, get_forecast_db_name
from sqlalchemy import create_engine
import gc
from autopilot_planning.models.boosted_models import *
from autopilot_planning.models.dl_models import *
from autopilot_planning.models.stat_models import *
# from .models.stacking import *
from autopilot_planning.models.ensembling import *
from autopilot_planning.models.hybrid_models import *
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from tensorflow.keras.models import Sequential
import kerastuner
from tensorflow import keras
#from tensorflow.keras.models import Sequential, load_model
import tensorflow as tf
tf.compat.v1.set_random_seed(1)
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning




#sys.modules['sklearn.externals.six'] = six
#import mlrose
#import joblib

#sys.modules['sklearn.externals.joblib'] = joblib
#from pmdarima.arima import auto_arima 

from .prediction_models import *

class hybrid_lstm_uni_per(Prediction_model):
    ''' Class to initialize variables and define generic modules for Additive and Hybrid model using LSTM model created in this class.'''

    def __init__(self, series_df, detail_df, fh, model_gov,tune_gov,params):
        ''' Initialising variables for LSTM model with values from config files'''
        Prediction_model.__init__(self, series_df, detail_df, fh, model_gov,tune_gov,params)
        self.freq = self.model_gov["frequency"]
        self.in_num = self.model_gov['in-num']
        
        if(self.params is None):
            pass
        elif(self.params.empty):
            self.units1 = 20
            self.units2 = 20
            self.learning_rate = 0.01
            self.epochs = 500
            self.batch = 4
        else:
            self.units1 = self.params['units1'].iloc[0]
            self.units2 = self.params['units2'].iloc[0]
            self.learning_rate = self.params['learning_rate'].iloc[0]
            self.learning_rate = self.learning_rate.astype(int)
            self.epochs = self.params['epochs'].iloc[0]
            self.batch = self.params['batch'].iloc[0]
      

    def lstm_bench(self, train):
        ''' This function creates the training data for lstm benchmark model by creating lags of 12 from the input data'''

        scaler = StandardScaler()
        train = np.array(scaler.fit_transform(pd.DataFrame(train))).reshape(1, -1)[0]

        x_train, y_train = train[:-1], np.roll(train, -self.in_num)[:-self.in_num]
        x_test = train[-self.in_num:]

        # reshape input to be [samples, time steps, features] (N-NF samples, 1 time step, 1 feature)
        x_train = np.reshape(x_train, (-1, 1))
        x_test = np.reshape(x_test, (-1, 1))
        temp_test = np.roll(x_test, -1)
        temp_train = np.roll(x_train, -1)

        for x in range(1, self.in_num):
            x_train = np.concatenate((x_train[:-1], temp_train[:-1]), 1)
            x_test = np.concatenate((x_test[:-1], temp_test[:-1]), 1)
            temp_test = np.roll(temp_test, -1)[:-1]
            temp_train = np.roll(temp_train, -1)[:-1]

        n_batch = len(x_train)

        # reshape to match expected input
        x_train = np.reshape(x_train, (-1, self.in_num, 1)) # changed sri
        x_test = np.reshape(x_test, (-1, self.in_num, 1)) # changed Sri
        x_train = tf.convert_to_tensor(x_train)
        x_test = tf.convert_to_tensor(x_test)
        self.units1 = tf.convert_to_tensor(self.units1)
        self.units1 = tf.convert_to_tensor(self.units2)
        print("Srinivas:::", type(self.units1))
        # print(x_train.shape, x_test.shape)
        # create the model
        model = Sequential()
#         model.add(Masking(mask_value=0,batch_input_shape=(n_batch, x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(self.units1, batch_input_shape=(n_batch, x_train.shape[1], x_train.shape[2]), return_sequences=True))
        model.add(LSTM(self.units2))
        model.add(Dense(1))
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=0)
        early_stopping = EarlyStopping(monitor='loss', patience=14, verbose=0)
        model.compile(loss='mean_squared_error',optimizer =keras.optimizers.Adam(learning_rate=0.01))
        
        # fit the model to the training data
        model.fit(x_train, y_train, callbacks=[reduce_lr, early_stopping],epochs=self.epochs, batch_size=n_batch, verbose=0,
                  shuffle=False)  # , callbacks = [EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1)])
#         


        # make predictions
        y_hat_test = []
        last_prediction = model.predict(x_test,batch_size=1)[0]
        print("Death Final", last_prediction)
        for i in range(0, self.fh):
            # Changed in enhancements
            y_hat_test.append(last_prediction)
            # x_test[0] = np.roll(x_test[0], -1)
            x_test = tf.roll(x_test, shift=-1, axis=1)
            row_index = 0
            last_column_index = len(x_test[row_index]) - 1
            updated_x_test = tf.tensor_scatter_nd_update(x_test, indices=[[row_index, last_column_index]], updates=[last_prediction])

            # x_test[0, (len(x_test[0]) - 1)] = last_prediction
            # x_test = tf.tensor_scatter_nd_update(x_test, tf.constant([[0, -1]]), tf.constant([[last_prediction]]))
            # last_prediction = model.predict(x_test,batch_size=1)[0]
            last_prediction = model.predict(updated_x_test,batch_size=1)[0]
            

        return scaler.inverse_transform(np.asarray(y_hat_test)).reshape(1,-1)

    def prophet_fit(self):
        ''' For the Hybrid model, we need a combination of 2 models the first being Exponential Smoothing/ Prophet model and the
            other being the LSTM model. This function creates the structure for Prophet model and train the model with time-series 
            data and get the prediction which is then fetched to lstm model.'''

        self.df_prophet, dataset, scaler_to_fix_decimal = prophet_data_preperation(self.series_df, self.freq, self.cap_power)

        m = prophet.Prophet(growth='logistic', changepoint_prior_scale=0.5, daily_seasonality=False, weekly_seasonality=False)
        m.fit(self.df_prophet)

        if self.freq == 12:
            future = m.make_future_dataframe(periods=self.fh, freq='M')
        elif self.freq == 52:
            future = m.make_future_dataframe(periods=self.fh, freq='W')
        elif self.freq == 365:
            future = m.make_future_dataframe(periods=self.fh, freq='D')
        else:
            print('Please designate Frequency')

        future['cap'] = int(int(self.df_prophet['y'].max()) * self.cap_power)
        future['floor'] = 0

        self.fcst = m.predict(future)
        self.fcst['trend'] = self.fcst['trend'] / scaler_to_fix_decimal
        dataset['y'] = dataset['y'] / scaler_to_fix_decimal
        dataset = dataset.reset_index(drop=True)
        return dataset

    def exponential_smoothning_fit(self):
        ''' For the Hybrid model, we need a combination of 2 models the first being Exponential Smoothing/ Prophet model and the
            other being the LSTM model. This function creates the structure for Exponential Smoothing  model and train the model with 
            time-series data and get the prediction which is then fetched to lstm model.'''
        
        warnings.simplefilter('ignore', ConvergenceWarning)               
        model = ExponentialSmoothing(np.asarray(self.series_df), trend='add', damped_trend=True,initialization_method='estimated').fit()
        
        self.pred = model.predict(start=0, end=len(self.series_df) + self.fh)
        
        
    def data_prep(self,train): 
            ''' This function prepares the training data for lstm model by taking 12 lags for tuning module'''
            scaler = StandardScaler()
            train = np.array(scaler.fit_transform(pd.DataFrame(train))).reshape(1, -1)[0]

            x_train, y_train = train[:-1], np.roll(train, -self.in_num)[:-self.in_num]


            # reshape input to be [samples, time steps, features] (N-NF samples, 1 time step, 1 feature)
            x_train = np.reshape(x_train, (-1, 1))
            temp_train = np.roll(x_train, -1)

            for x in range(1, self.in_num):
                x_train = np.concatenate((x_train[:-1], temp_train[:-1]), 1)
                temp_train = np.roll(temp_train, -1)[:-1]

            n_batch = len(x_train)

            # reshape to match expected input
            x_train = tf.reshape(x_train, (-1, self.in_num, 1)) # changed Sri

            return x_train, tf.array(y_train) # changed sri

class Additive_hybrid_lstm_uni_per(hybrid_lstm_uni_per):
    ''' Class for initialising variables and defining modules for Aditive Hybrid model, base class being hybrid_lstm_uni_per'''

    def __init__(self, series_df, detail_df, fh, model_gov, tune_gov,params):
        ''' Initialising variables taken from config file'''
        hybrid_lstm_uni_per.__init__(self, series_df, detail_df, fh, model_gov,tune_gov,params)
        self._name = "Additive Hybrid"
        self.cap_power =  self.model_gov["cap-power"]
        self.freq = self.model_gov["frequency"]
        self.in_num = self.model_gov['in-num']
        self.units1_min_value = self.tune_gov['units1_min_value']
        self.units1_max_value = self.tune_gov['units1_max_value']
        self.units1_step = self.tune_gov['units1_step']
        self.units2_min_value = self.tune_gov['units2_min_value']
        self.units2_max_value = self.tune_gov['units2_max_value']
        self.units2_step = self.tune_gov['units2_step']
        self.learning_rate = self.tune_gov['learning_rate']
        self.epochs_min_value = self.tune_gov['epochs_min_value']
        self.epochs_max_value = self.tune_gov['epochs_max_value']
        self.epochs_step = self.tune_gov['epochs_step']
        self.batch_size = self.tune_gov['batch_size']
        
        
    def tune(self):
        '''Tuning module for Additive Hybrid by calling data_prep for training data creation and lstm_bench for tuning lstm'''
        super().exponential_smoothning_fit()
        train = self.series_df.values - self.pred[:-(self.fh + 1)]
        Train = train[:-12]
        Val = train[-12:]
        
        
        X_train, Y_train = super().data_prep(Train)
        X_val, Y_val = super().data_prep(Val)
        
        def build_model(hp):
    
#             a = len(X_train)
            n_batch = hp.Choice('batch',values = self.batch_size)
            model = Sequential()
            model.add(LSTM(units=hp.Int('units1',min_value=self.units1_min_value, max_value=self.units1_max_value
                                           ,step=self.units1_step), input_shape=( X_train.shape[1], X_train.shape[2]), return_sequences=True))
    #         
            model.add(LSTM(units=hp.Int('units2',min_value=self.units2_min_value, max_value=self.units2_max_value
                                           ,step=self.units2_step)))
            model.add(Dense(1))
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
            early_stopping = EarlyStopping(monitor='loss', patience=14, verbose=1)
    #         model.add(Dropout(0.2))
    #         model.add(Activation('softmax'))

            model.compile(loss='mean_squared_error',optimizer = keras.optimizers.Adam(hp.Choice('learning_rate',
                          values=self.learning_rate)), metrics=['accuracy'])


            # fit the model to the training data
            model.fit(X_train, Y_train, callbacks=[reduce_lr, early_stopping],epochs=hp.Int('epochs',
                                            min_value=self.epochs_min_value,
                                            max_value=self.epochs_max_value,
                                            step=self.epochs_step), batch_size=n_batch, verbose=1,
                      shuffle=False,validation_split = 0.1)



            return model
        
        if os.path.isdir('autopilot_planning/models/my_dir/Additive_Hybrid'):
            shutil.rmtree('autopilot_planning/models/my_dir/Additive_Hybrid')

        tuner = BayesianOptimization(
            build_model,
            objective=kerastuner.Objective('loss',direction='min'),
            max_trials=5,

            directory='autopilot_planning/models/my_dir',
            project_name='Additive_Hybrid')
        tuner.search(X_train, Y_train,
             epochs=50,
             validation_data=(X_val, Y_val))
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        return best_hps
        
    def predict(self):
        '''Predict function for Additive Hybrid model '''
        ets = True 
#         try:
            
        super().exponential_smoothning_fit()

        train = self.series_df.values - self.pred[:-(self.fh + 1)]
        #print("201")
        y_hat_test_RNN = tf.reshape(super().lstm_bench(train), (-1))
#         y_hat_test_RNN = tf.reshape(super().lstm_bench(train), (-1))
        #print("301")
        for i in range(0, self.fh):
            # Changed in enhancements
            # y_hat_test_RNN[i] = y_hat_test_RNN[i] + self.pred[len(self.series_df) + i]
            y_hat_test_RNN = tf.tensor_scatter_nd_add(y_hat_test_RNN,indices=[[i]],updates=[self.pred[len(self.series_df) + i]])

            if y_hat_test_RNN[i] <= 0:
                # Changed in enhancements
                # y_hat_test_RNN[i]=0
                y_hat_test_RNN = tf.zeros_like(y_hat_test_RNN)
                
#         except Exception as e:
#             ets = False
#             print("Skipping additive hybrid using exponential smoothening")
            
#             print("exception is:",e)
        
        if ets == False:
            try:
                dataset = self.prophet_fit()
                train = dataset['y'] - self.fcst['trend'][:-(self.fh)]
                y_hat_test_RNN = np.reshape(self.lstm_bench(train), (-1))
                for i in range(0, self.fh):
                    y_hat_test_RNN[i] = y_hat_test_RNN[i] + self.fcst['trend'][len(self.df_prophet) + i]
                    if y_hat_test_RNN[i] <= 0:
                        # Changed in enhancements
                        # y_hat_test_RNN[i]=0
                        y_hat_test_RNN = tf.zeros_like(y_hat_test_RNN)
                        #y_hat_test_RNN[i] = self.fcst['trend'][len(self.df_prophet) + i]
            except Exception as e:
                print("Skipping additive hybrid using prophet")
                print(e)
        # Changed in enhancements
        # return pd.DataFrame(y_hat_test_RNN.reshape(1, -1))
        return pd.DataFrame(tf.reshape(y_hat_test_RNN, (1, -1)).numpy())


class Multiplicative_hybrid_lstm_uni_per(hybrid_lstm_uni_per):
    ''' Class for initialising variables and defining modules for Aditive Hybrid model, base class being hybrid_lstm_uni_per'''

    def __init__(self, series_df, detail_df, fh, model_gov, tune_gov,params):
        hybrid_lstm_uni_per.__init__(self, series_df, detail_df, fh, model_gov,tune_gov,params)
        self._name = "Multiplicative Hybrid"
        self.cap_power = self.model_gov["cap-power"]
        self.units1_min_value = self.tune_gov['units1_min_value']
        self.units1_max_value = self.tune_gov['units1_max_value']
        self.units1_step = self.tune_gov['units1_step']
        self.units2_min_value = self.tune_gov['units2_min_value']
        self.units2_max_value = self.tune_gov['units2_max_value']
        self.units2_step = self.tune_gov['units2_step']
        self.learning_rate = self.tune_gov['learning_rate']
        self.epochs_min_value = self.tune_gov['epochs_min_value']
        self.epochs_max_value = self.tune_gov['epochs_max_value']
        self.epochs_step = self.tune_gov['epochs_step']
        self.batch_size = self.tune_gov['batch_size']
        
    def tune(self):

        super().exponential_smoothning_fit()
        train = self.series_df.values - self.pred[:-(self.fh + 1)]
        scaler = StandardScaler()
        train = np.array(scaler.fit_transform(pd.DataFrame(train))).reshape(1, -1)[0]
        Train = train[:-12]
        Val = train[-12:]
        x_test = train[-self.in_num:]
        x_test = np.reshape(x_test, (-1, 1))
        temp_test = np.roll(x_test, -1)
        for x in range(1, self.in_num):
                    x_test = np.concatenate((x_test[:-1], temp_test[:-1]), 1)
                    temp_test = np.roll(temp_test, -1)[:-1]
        x_test = np.reshape(x_test, (-1, self.in_num, 1))
        X_train, Y_train = super().data_prep(Train)
        X_val, Y_val =super(). data_prep(Val)
        
        def build_model(hp):
    
#             a = len(X_train)
            n_batch = hp.Choice('batch',values = self.batch_size)
            model = Sequential()
            model.add(LSTM(units=hp.Int('units1',min_value=self.units1_min_value, max_value=self.units1_max_value
                                           ,step=self.units1_step), input_shape=( X_train.shape[1], X_train.shape[2]), return_sequences=True))
    #         
            model.add(LSTM(units=hp.Int('units2',min_value=self.units2_min_value, max_value=self.units2_max_value
                                           ,step=self.units2_step)))
            model.add(Dense(1))
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
            early_stopping = EarlyStopping(monitor='loss', patience=14, verbose=1)
            model.compile(loss='mean_squared_error',optimizer = keras.optimizers.Adam(hp.Choice('learning_rate',
                          values=self.learning_rate)), metrics=['accuracy'])


            # fit the model to the training data
            model.fit(X_train, Y_train, callbacks=[reduce_lr, early_stopping],epochs=hp.Int('epochs',
                                            min_value=self.epochs_min_value,
                                            max_value=self.epochs_max_value,
                                            step=self.epochs_step), batch_size=n_batch, verbose=1,
                      shuffle=False,validation_split = 0.1)



            return model
        
        if os.path.isdir('autopilot_planning/models/my_dir/Multiplicative_Hybrid'):
            shutil.rmtree('autopilot_planning/models/my_dir/Multiplicative_Hybrid')
    
        tuner = BayesianOptimization(
            build_model,
            objective=kerastuner.Objective('loss',direction='min'),
            max_trials=5,

            directory='autopilot_planning/models/my_dir',
            project_name='Multiplicative_Hybrid')
        tuner.search(X_train, Y_train,
             epochs=50,
             validation_data=(X_val, Y_val))
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        return best_hps
    
    
    def predict(self):

        ets = True
        # print('entering multiplicative hybrid using exponential smoothening')
        try:
            super().exponential_smoothning_fit()
            # print('exponential_smoothning_fit done')
            
            train = self.series_df.values / self.pred[:-(self.fh + 1)]
            # print('train done')
            y_hat_test_RNN = np.reshape(super().lstm_bench(train), (-1))
            # print('y_hat_test_RNN done')
            for i in range(0, self.fh):
                # print('entering for loop', self.fh)
                # print(y_hat_test_RNN[i], self.pred[len(self.series_df) + i])
                y_hat_test_RNN[i] = y_hat_test_RNN[i] * self.pred[len(self.series_df) + i]
                # print('y_hat_test_RNN[i] done')
          
                if y_hat_test_RNN[i] <= 0:
                    # Changed in enhancements
                    # y_hat_test_RNN[i]=0
                    y_hat_test_RNN = tf.zeros_like(y_hat_test_RNN)
                    # print('last step done')
                   
        except Exception as e:
            ets = False
            print("Skipping multiplicative hybrid using exponential smoothening")
            print(e)
        
        if ets == False:
            try:
                dataset = super().prophet_fit()
                train = dataset['y'] / self.fcst['trend'][:-(self.fh)]
                y_hat_test_RNN = np.reshape(super().lstm_bench(train), (-1))
                for i in range(0, self.fh):
                    y_hat_test_RNN[i] = y_hat_test_RNN[i] * self.fcst['trend'][len(self.df_prophet) + i]
                    if y_hat_test_RNN[i] <= 0:
                        # Changed in enhancements
                        # y_hat_test_RNN[i]=0
                        y_hat_test_RNN = tf.zeros_like(y_hat_test_RNN)
                        #y_hat_test_RNN[i] = self.fcst['trend'][len(self.df_prophet) + i]

            except Exception as e:
                print("Skipping multiplicative hybrid using prophet")
                print(e)


        # Changed in enhancements
        # return pd.DataFrame(y_hat_test_RNN.reshape(1, -1))
        return pd.DataFrame(tf.reshape(y_hat_test_RNN, (1, -1)).numpy())
