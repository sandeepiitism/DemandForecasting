from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .prediction_models import *
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
import pandas as pd
import six
import sys
from tqdm import tqdm

#sys.modules['sklearn.externals.six'] = six
#import mlrose
#import joblib

#sys.modules['sklearn.externals.joblib'] = joblib
#from pmdarima.arima import auto_arima 

from tensorflow.keras.models import Sequential, model_from_yaml
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
#from tensorflow.keras.layers.convolutional import Conv1D
#from tensorflow.keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout,Dense, Activation
#from kerastuner.tuners import RandomSearch
import kerastuner
from tensorflow import keras
from kerastuner.tuners import BayesianOptimization
import os
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from dateutil.relativedelta import relativedelta
from ..config.config import get_db_config
from ..preprocessing.preprocessing import timeseries_preprocessing, create_training_data_cnn, reset_weights, \
    create_training_data_dnn, format_date_series, format_date_series_celgene

from hyperopt import Trials, STATUS_OK, tpe, rand

import copy

import pymysql
import os
import datetime
import psycopg2


def fetch_all_history(db_config):
    ''' Returns all historical demand data from database table Gross_Demand_Final '''
    conn = pymysql.connect(host=db_config['host'], user=db_config['user'],port=db_config['port'],passwd=db_config['password'], db=db_config['dbname'])
    data_df = pd.read_sql("""select Material, Material_Name, 
                Market_Code, Market_Name, DP_Trade_Name, 
                Year, Month_Num as Month, 
                sum(Demand) as Total_Demand 
                from dmndfcst_rdl.Gross_Demand_Final
                where Demand_Type in ('COM','FRG')
                and Date is not null
                group by Material, Material_Name, Year, Month_Num, DP_Trade_Name, Market_Code,Market_Name
                order by Year, Month_Num;""", con=conn)
    detail_cols = ["Material", "Material_Name", "Market_Code", "Market_Name", "DP_Trade_Name"]
    data_df["Date"] = (data_df["Month"].map(str) + "-" + data_df["Year"].map(str)).map(format_date_series)
    data_df = data_df.pivot_table(index=detail_cols, columns="Date", values="Total_Demand").reset_index()
    data_df.columns = detail_cols + [col for col in data_df.columns[len(detail_cols):]]
    data_df.replace(np.nan,0,inplace=True)
    conn.close()
    return data_df[data_df.columns.difference(detail_cols)]

def fetch_all_history_celgene(db_config_celgene):
    ''' Returns all historical demand data from database table Gross_Demand_Final - Celgene'''
    print("inside fetch all history celgene")
    conn = psycopg2.connect(host=db_config_celgene['host'],database=db_config_celgene['database'], user=db_config_celgene['user'], password=db_config_celgene['password'])
    print("after connection")
    data_df = pd.read_sql("""select material as "Material", material_name as "Material_Name", 
                demand_market as "Market_Name", dp_trade_name as "DP_Trade_Name", 
                year as "Year", month as "Month", 
                sum(demand) as "Total_Demand" 
                from fcst.v_esa_shipment_data_with_product 
                where commercial_type in ('COMMERCIAL')
                and Date is not null
                group by material, material_name, year, month, dp_trade_name, demand_market
                order by year, month;""", con=conn)

    detail_cols = ["Material", "Material_Name", "Market_Name", "DP_Trade_Name"]

    data_df["Date"] = (data_df["Month"].map(str).str.strip() + "-" + data_df["Year"].map(str).str.strip()).map(format_date_series_celgene)
    data_df = data_df.pivot_table(index=detail_cols, columns="Date", values="Total_Demand").reset_index()
    data_df.columns = detail_cols + [col for col in data_df.columns[len(detail_cols):]]
    data_df.replace(np.nan,0,inplace=True)
    conn.close()

    return data_df[data_df.columns.difference(detail_cols)]


def fetch_all_history_celgene_new_db(db_config_celgene):
    print("inside fetch all history celgene new db")
    conn = pymysql.connect(host=db_config_celgene['host'], user=db_config_celgene['user'],port=db_config_celgene['port'],passwd=db_config_celgene['password'], db=db_config_celgene['dbname'])
    data_df=pd.read_sql("""select sku as "Material",product_description as "Material_Name",
                    country as "Market_Name",product_family as "DP_Trade_Name",
                    year as "Year",month as "Month",
                    sum(actual) as "Total_Demand"
                    from dmndfcst.celgene_monthly_shipment_filtered_v1
                    where DEMAND_TYPE in ('COM')
                    and period_month is not null 
                    group by sku,product_description,year,month,product_family,country 
                    order by year,month;""",con=conn)
    #print("data_df:",data_df)
    #print("after read sql")
    detail_cols = ["Material", "Material_Name", "Market_Name", "DP_Trade_Name"]
    #print("before map")
    #data_df["Date"] = (data_df["Month"].map(str) + "-" + data_df["Year"].map(str)).map(format_date_series_celgene)
    data_df["Date"] = (data_df["Month"].map(str).str.strip() + "-" + data_df["Year"].map(str).str.strip()).map(format_date_series_celgene)
    
    #print("before pivot table")
    data_df = data_df.pivot_table(index=detail_cols, columns="Date", values="Total_Demand").reset_index()
    #print("after pivot table")
    data_df.columns = detail_cols + [col for col in data_df.columns[len(detail_cols):]]
    #print("before replace")
    data_df.replace(np.nan,0,inplace=True)
    #print("before close")
    conn.close()
    #print("before return")
    #print("fetch all history celgene new db executed")
    return data_df[data_df.columns.difference(detail_cols)]
    
class dnn_univariate(Prediction_model):
    ''' Class for Deep Learning Neural Network Model '''
    
    def __init__(self, series_df, detail_df, fh, model_gov, tune_gov,params):
        ''' Initialise the variables for creating training data and hyperparameters from the config file '''
        Prediction_model.__init__(self, series_df, detail_df, fh, model_gov,tune_gov,params)
        self.fh = fh
        self._name = "DNN Univariate"
        self.n_steps_in = self.model_gov["n_steps_in"]
        self.n_steps_out = self.model_gov["n_steps_out"]
        self.units1_min_value = self.tune_gov['units1_min_value']
        self.units1_max_value = self.tune_gov['units1_max_value']
        self.units1_step = self.tune_gov['units1_step']
        self.units2_min_value = self.tune_gov['units2_min_value']
        self.units2_max_value = self.tune_gov['units2_max_value']
        self.units2_step = self.tune_gov['units2_step']
        self.units3_min_value = self.tune_gov['units3_min_value']
        self.units3_max_value = self.tune_gov['units3_max_value']
        self.units3_step = self.tune_gov['units3_step']
        self.optimizer_tune = self.tune_gov['optimizer']
        self.epochs_min_value = self.tune_gov['epochs_min_value']
        self.epochs_max_value = self.tune_gov['epochs_max_value']
        self.epochs_step = self.tune_gov['epochs_step']
        self.batch_size = self.tune_gov['batch_size']
        
        if(self.params is None):
            pass
        elif(self.params.empty):
            self.units1 = 10
            self.units2 = 10
            self.units3 = 10
            self.optimizer = 'rmsprop'
            self.epochs = 100
            self.batch = 4
        else:
            self.units1 = self.params['units1'].iloc[0]
            self.units2 = self.params['units2'].iloc[0]
            self.units3 = self.params['units3'].iloc[0]
            self.optimizer = self.params['optimizer'].iloc[0]
            self.epochs = self.params['epochs'].iloc[0]
            self.batch = self.params['batch'].iloc[0]

       

    def get_model(self):
        ''' Create structure for DNN model  '''
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        
        model = Sequential()
    #Reading parameters for each sku separately from optimized parameters file created after Tuning
        model.add(Dense(self.units1, activation='relu', input_dim=self.n_steps_in))
        #model.add(Dense(10, activation='relu'))
        model.add(Dense(self.units2, activation='relu'))
        model.add(Dense(self.units3, activation='relu'))

        model.add(Dense(self.n_steps_out))
        model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=['mape'])

        return model

    

    
    
    def tune(self):
        ''' Tune module to tune the DNN model for optimized hyperparameters '''
            
        def build_model(hp):
            
            ''' Creates model for Tuning using Keras Tuner , reading parameter range from config file '''
    
            model = keras.Sequential()

            model.add(Dense(units=hp.Int('units1',min_value=self.units1_min_value, max_value=self.units1_max_value
                                           ,step=self.units1_step), activation='relu', input_dim=n_steps_in))
            #model.add(Dense(10, activation='relu'))
            model.add(Dense(units=hp.Int('units2',min_value=self.units2_min_value, max_value=self.units2_max_value
                                           ,step=self.units2_step), activation='relu'))
            model.add(Dense(units=hp.Int('units3',min_value=self.units3_min_value, max_value=self.units3_max_value
                                           ,step=self.units3_step), activation='relu'))

            model.add(Dense(n_steps_out))
            model.compile(optimizer= hp.Choice('optimizer',values=self.optimizer_tune), loss='mean_squared_error', metrics=['mape'])
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
            early_stopping = EarlyStopping(monitor='loss', patience=14, verbose=1)
    
            # fit model
            model.fit(X,
                                y,
                                callbacks=[reduce_lr, early_stopping],
                                epochs=hp.Int('epochs',
                                            min_value=self.epochs_min_value,
                                            max_value=self.epochs_max_value,
                                            step=self.epochs_step),
                                batch_size=hp.Choice('batch',values = self.batch_size),
                                verbose=1)

            return model
    
    
        fh = 48
        _name = "DNN Univariate"
        n_steps_in = 12
        n_steps_out = 1
       
        
        self.scaler = MinMaxScaler()
        
        self.scaled_series_df = pd.DataFrame(self.scaler.fit_transform(self.series_df.values.reshape(-1, 1)), index=self.series_df.index)
        X, y, has_nan = create_training_data_dnn(self.scaled_series_df, self.n_steps_in, self.n_steps_out)
        X = X.reshape(X.shape[0], X.shape[1])
        y = y.reshape(y.shape[0], y.shape[1])
        
        if os.path.isdir('autopilot_planning/models/my_dir/DNN'):
            shutil.rmtree('autopilot_planning/models/my_dir/DNN')

        tuner = BayesianOptimization(
            build_model,
            objective='loss',
            max_trials=10,

            directory='autopilot_planning/models/my_dir',
            project_name='DNN')
        tuner.search(X, y,
             epochs=50,
             validation_split = 0.1)
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
#         best_hps.to_csv('hps_dnn_test.csv')
        
        return best_hps

    def train(self):
        ''' Training the model by calling the model structure and the identified bet hyperparameter set for each SKU'''
        model = self.get_model()
        self.scaler = MinMaxScaler()
        self.scaled_series_df = pd.DataFrame(self.scaler.fit_transform(self.series_df.values.reshape(-1, 1)), index=self.series_df.index)
        X, y, has_nan = create_training_data_dnn(self.scaled_series_df, self.n_steps_in, self.n_steps_out)
        X = X.reshape(X.shape[0], X.shape[1])
        y = y.reshape(y.shape[0], y.shape[1])
        
        # fit model
        history = model.fit(X,
                            y,
                            epochs=self.epochs,
                            batch_size=self.batch,
                            verbose=0)

        return model

    def predict(self):
        ''' Predicting the forecast horizon on the trained model'''
        model = self.train()

        # PREDICTING FOR FORECASTING HORIZON
        pred_set = list(self.series_df.values[-self.n_steps_in:])
        
        scaled_pred_set = list(self.scaled_series_df.values[-self.n_steps_in:,0])
        
        for i in range(0, self.fh):
            
            p = model.predict(
            np.array(scaled_pred_set[-self.n_steps_in:]).reshape(1,-1)
            )
            scaled_pred_set.append(p[0][0])
    
        p_df = pd.DataFrame(self.scaler.inverse_transform(np.array(scaled_pred_set).reshape(-1,1))[-self.fh:,:].reshape(1,-1))
        m = p_df > 0
        #p_df.where(m, 0)
        
        return p_df.where(m, 0)

class cnn_univariate_conjoint(Prediction_model):
    ''' Class to intialise variables and training modules for CNN univariate model'''

    def __init__(self, series_df, detail_df, fh,model_gov,tune_gov,params):
        ''' Initialise the variables for creating training data and hyperparameters from the config file '''
        Prediction_model.__init__(self, series_df, detail_df, fh,model_gov,tune_gov,params)
        self.fh = fh
        self._name = "CNN Univariate Conjoint"
        self.model_location = self.model_gov['model-location']
        self.model_backup_location = model_gov['model-backup-location']
        self.model_arch = model_gov['model-arch']
        self.n_steps_in = model_gov['n-steps-in']
        self.n_steps_out = model_gov['n-steps-out']
        self.n_features = model_gov['n-features']
        self.database_name=model_gov['database_name']
        self.epochs_min_value = self.tune_gov['epochs_min_value']
        self.epochs_max_value = self.tune_gov['epochs_max_value']
        self.epochs_step = self.tune_gov['epochs_step']
        self.batch_size = self.tune_gov['batch_size']
        
        if(self.params is None):
            pass
        elif(self.params.empty):
            self.epochs = 200
            self.batch = 32
        else:
            self.epochs = self.params['epochs'].iloc[0]
            self.batch = self.params['batch'].iloc[0]
        
    def tune(self, db_config, hold_out):
        '''Tuning the CNN model by using parameter values from tuning config file.
           Only epochs and batch size has been tuned as hyperparameters'''
        
        print("Fetching all history")
        print("check whether data is for bms/celgene:",self.database_name)
        if self.database_name=='bms':
            history_df = fetch_all_history(db_config)
        elif self.database_name=='celgene':
            history_df = fetch_all_history_celgene_new_db(db_config)
        
        a = history_df.T
        b = a.iloc[:-12]
        history_df = b.T
        processed_df = pd.DataFrame(columns = history_df.columns)
        for i in tqdm(range(len(history_df)), desc = "Outlier Detection"):
            
            if((history_df.iloc[i]==0).all()):
                continue
            processed_df.loc[i] = timeseries_preprocessing(history_df.iloc[i], hold_out=0, front_trim = False)
       
        master_X, master_y = create_training_data_cnn(processed_df, self.n_steps_in, self.n_steps_out)
        x_train, x_valid, y_train, y_valid = train_test_split(master_X, master_y, shuffle=True, train_size=0.80,
                                                              random_state=11)

        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], self.n_features))
        x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], self.n_features))
        
        
        def build_model(hp):
            
            print("Starting CNN Modelling")
            model = keras.Sequential()
            yaml_file = open(self.model_arch, 'r')
            loaded_model_yaml = yaml_file.read()
            yaml_file.close()
            model = keras.models.model_from_yaml(loaded_model_yaml)
            from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1)
            early_stopping = EarlyStopping(monitor='val_loss', patience=14, verbose=1)

    #         reset_weights(model)

            model.compile(optimizer='rmsprop', loss='mean_squared_logarithmic_error', metrics=['mape'])

            model.fit(x_train,
                                y_train,
                                callbacks=[reduce_lr, early_stopping],
                                epochs=hp.Int('epochs',
                                            min_value=self.epochs_min_value,
                                            max_value=self.epochs_max_value,
                                            step=self.epochs_step),
                                batch_size=hp.Choice('batch',values = self.batch_size),
                                verbose=1, 
                                validation_data=(x_valid, y_valid))

            return model
    
    
        
        if os.path.isdir('autopilot_planning/models/my_dir/CNN'):
            shutil.rmtree('autopilot_planning/models/my_dir/CNN')

        tuner = BayesianOptimization(
                    build_model,
                    objective='loss',
                    max_trials=5,

                    directory='autopilot_planning/models/my_dir',
                    project_name='CNN')
        tuner.search(x_train, y_train,
             epochs=50,
             validation_split = 0.1)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        return best_hps

    
    def train(self, db_config, hold_out=0):
        '''Training the CNN model taking all the SKUs together as training data.
           In this module, the complete data is being fetched from database and training data is prepared for model training through
           create_training_data_cnn module'''
        
        # DATA EXTRACTION LAYER
        print("Fetching all history")
        print("check whether data is for bms/celgene:",self.database_name)
        if self.database_name=='bms':
            history_df = fetch_all_history(db_config)
        elif self.database_name=='celgene':
            history_df = fetch_all_history_celgene_new_db(db_config)
            
        #print("history_df:",history_df)
        # DATA PRE PROCESSING
        #print("Processing data")
        processed_df = pd.DataFrame(columns = history_df.columns)
        for i in range(len(history_df)):
            
            if((history_df.iloc[i]==0).all()):
                continue
            processed_df.loc[i] = timeseries_preprocessing(history_df.iloc[i], hold_out=0, front_trim = False)
        #print("processed df:",processed_df)
        # TRAINING DATA CREATION
        #print("Training Data Creation")
        master_X, master_y = create_training_data_cnn(processed_df, self.n_steps_in, self.n_steps_out)
#         print("master x: \n") #Kamal
#         print(master_X) #Kamal
#         print("master y: \n") #Kamal
#         print(master_y) #Kamal
        x_train, x_valid, y_train, y_valid = train_test_split(master_X, master_y, shuffle=True, train_size=0.80,
                                                              random_state=11)
        print("x train before reshape:",x_train.shape)
        print("y train before reshape:",y_train.shape)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], self.n_features))
        x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], self.n_features))
        print("x train after reshape:",x_train.shape)
        print("y train after reshape:",y_train.shape)
        # MODEL TRAINING
        print("Starting CNN Modelling")
        yaml_file = open(self.model_arch, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(loaded_model_yaml)
        from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=14, verbose=1)

        reset_weights(model)

        model.compile(optimizer='rmsprop', loss='mean_squared_logarithmic_error', metrics=['mape'])

        history = model.fit(x_train,
                            y_train,
                            callbacks=[reduce_lr, early_stopping],
                            epochs=self.epochs,
                            batch_size=self.batch,
                            verbose=1, 
                            validation_data=(x_valid, y_valid))
        #os.rename(self.model_location,
        #          "{}-{}.h5".format(self.model_backup_location, datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
        model.save(self.model_location)

    def predict(self):
        ''' Predicting the forecast horizon for each SKU on the trained model'''
        
        from keras.models import load_model
        model = load_model(self.model_location)

        # PREDICTING FOR FORECASTING HORIZON
        pred_set = list(self.series_df.values[-self.n_steps_in:])
        for i in range(0, self.fh):
            p = model.predict(
                np.asarray(pred_set[-self.n_steps_in:],dtype='float').reshape((1, self.n_steps_in, self.n_features)).astype('float')).reshape(1)
            if p > 0:
                pred_set.append(p)
            else:
                pred_set.append(0)
        
        return pd.DataFrame(np.array(pred_set[-self.fh:]).reshape(1, -1))

