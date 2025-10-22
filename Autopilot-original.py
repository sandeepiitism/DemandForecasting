import pandas as pd
import tensorflow.keras.backend as K
from .preprocessing.preprocessing import timeseries_preprocessing
from .models.boosted_models import *
from .models.dl_models import *
from .models.stat_models import *
# from .models.stacking import *
from .models.ensembling import *
from .models.hybrid_models import *
from .models.dl_models import *
from .models.boosted_models import *
from .config.config import *
from multiprocessing import Process, Manager, Pool, cpu_count
import time
from .fetch_data import *
from .write_data import *
from .eval.evaluate import *
import gc
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
import traceback
np.seterr(all='ignore')
# import swifter #added Oct-22

class Autopilot:
    ''' This is is the main calling function for all the predict and tune modules for all the model class
        * To skip the forecast for any model, comment the respective model in line 38-46'''

    def __init__(self, series_df, detail_df, prev_date, model_gov, tune_gov,prediction_type="univariate",
                 run_models="all", fh=48):

        self.prediction_type = prediction_type
        self.series_df = series_df
        self.detail_df = detail_df
        self.all_models = {
            # TODO: not tested Auto Arima
            "Auto-ARIMA": Auto_arima,
            "Holt-Winters Additive Exponential Smoothing": Additive_smoothing,
            "Holt-Winters Multiplicative Exponential Smoothing": Multiplicative_smoothing,
            "Prophet": Prophet,
            "Additive Hybrid": Additive_hybrid_lstm_uni_per,
            "Multiplicative Hybrid": Multiplicative_hybrid_lstm_uni_per,
            "CNN Univariate Conjoint": cnn_univariate_conjoint,
            "DNN": dnn_univariate,
            "XGBoost": xgb_forecast,
#             "GBM": gbm_forecast
        }
    
       
        
        self.run_models = {}
        model_name_check = False
        if run_models == "all":
            self.run_models = self.all_models
            model_name_check = True
        
        else:
            for run_model in run_models:
                if self.all_models.has_key(run_model):
                    self.run_models[run_model] = self.all_models[run_model]
                    model_name_check = True
      
        if model_name_check == False:
            raise Exception("Invalid model names".format(run_models))

        self.fh = fh
        self.prev_date = prev_date
        self.model_gov = model_gov
        self.tune_gov = tune_gov

    def predict(self, hold_out=0):
        
        
        processed_series_df = timeseries_preprocessing(self.series_df, hold_out)

        pred_df = pd.DataFrame()
        prediction_category = []

        if self.prediction_type == "univariate":

            for model_name, model_class in self.run_models.items():
                try:
                
                    if self.validate_governance(model_name, self.model_gov) == True:

    #                     Params contain the best hyperparameter values for each sku as saved in the hyperparameters file in config folder
                        if(model_name=='Auto-ARIMA'):
                            params = None
                        elif(model_name=='XGBoost'):
                            params = None
                        elif(model_name=='CNN Univariate Conjoint'):
                            params  = pd.read_csv(self.tune_gov[model_name]['Parameters-location'])
                        else:
                            params  = pd.read_csv(self.tune_gov[model_name]['Parameters-location'])
                            params = params[(params['Material']==self.detail_df['Material']) & (params['Market_Code']==self.detail_df['Market_Code'])]

                        pred_model = model_class(processed_series_df, self.detail_df, self.fh, self.model_gov[self.prediction_type][model_name],self.tune_gov[model_name],params)
                        #print("Starting the prediction for {}:".format(model_name))#Kamal
                        prediction = pred_model.predict()
                        pred_df = pd.concat([pred_df, prediction], axis=0, ignore_index=True)

                        prediction_category += [model_name]

                    else:
                        print("{} Skipped due to not meeting governance policy for this SKU {} and market {} \n".format(model_name, self.detail_df[0], self.detail_df[2]))
            
                except Exception as e:
                    print(traceback.format_exc())
                    print("Error: {} Skipped for material {} \n".format(model_name,self.detail_df))
                    ##----temp code starts----
                    skus_Exception_error = pd.DataFrame(columns = ['material_no', 'material_name', 'market_code','model_name']) 
                    
                    skus_Exception_error = skus_Exception_error.append({'material_no' :self.detail_df[0] , 'material_name': self.detail_df[1], 'market_code' :self.detail_df[2],'model_name':model_name},ignore_index = True) 
                    
                    skus_Exception_error.to_csv(r'/mnt/run_results/BMS_Exception(predict)_SKUs.csv',mode="a",header=False)
                    
                    print(e)

            gc.collect()
            
            if len(prediction_category) == 0:
                return
            
            df_col_list = list()
            last_date = self.prev_date-relativedelta(months=+(hold_out+1))
            last_year = last_date.year
            last_month = last_date.month
            new_list_of_dates = list()
            i = 0
            while i < self.fh:
                if last_month == 12:
                    last_month = 1
                    last_year = last_year + 1
                    new_list_of_dates += [[last_year, last_month, 1]]
                else:
                    last_month = last_month + 1
                    last_year = last_year
                    new_list_of_dates += [[last_year, last_month, 1]]
                i += 1
            df_time = pd.to_datetime(pd.DataFrame(new_list_of_dates, columns=['Year', 'Month', 'Day']))
            for item in df_time:
                df_col_list += [str(item.date())]
            pred_df.columns = df_col_list
            pred_df = pred_df.reset_index(drop=True)

            import tensorflow.keras.backend as K
            K.clear_session()

            pred_df["Forecasting_Technique"] = prediction_category
#             pred_df['Anomaly'] = anomaly_flag
            pred_df = self.attach_details(pred_df)

        return pred_df
    
   

    def eval(self):
        pass

    def predict_eval(self):
        pass

    def attach_details(self, pred_df):
        for index, value in self.detail_df.iteritems():
            pred_df[index] = value

        return pred_df

    def validate_governance(self, model_name, model_gov):
        if (model_name=='Additive Hybrid' or model_name=='Multiplicative Hybrid'):
            print("vishnu-param1:::::::\n\n", model_gov[self.prediction_type][model_name]["min-data-sku"])
            print("vishnu-param2:::::::\n\n", self.series_df.shape[0])
            if model_gov[self.prediction_type][model_name]["min-data-sku"] >= self.series_df.shape[0]:
                return False
        else:
            if model_gov[self.prediction_type][model_name]["min-data-sku"] > self.series_df.shape[0]:
                print("Harsha's-model name:::::::", model_name)
                print("Harsha-param1:::::::\n\n", model_gov[self.prediction_type][model_name]["min-data-sku"])
                print("Harsha-param2:::::::\n\n", self.series_df.shape[0])
                return False

        return True




def setup(prediction_type, model_gov, run_date, db_config, hold_out=0):
    ''' This is to train CNN model on entire data at once and store the model'''
    
    if check_for_run(model_gov[prediction_type]["CNN Univariate Conjoint"]["training"], run_date) == True:
        try:
            
            print("CNN training started")
            #dl_models.py
            cnn_univariate_conjoint(None, None, None, model_gov[prediction_type]["CNN Univariate Conjoint"]).train(db_config, hold_out)
            import tensorflow.keras.backend as K
            K.clear_session()
            print("CNN training completed")
        except Exception as e:
                print("Error: CNN training not completed")
                print(e)
                
def CNN_tuning(prediction_type, model_gov, run_date, db_config, hold_out=0):
    ''' This is to tune CNN model on entire data at once and store the model
        Parameters being tuned - Epochs and batch_size
        '''
    

    if check_for_run(model_gov[prediction_type]["CNN Univariate Conjoint"]["parameter-tuning"], run_date) == True:
        try:
            
            print("CNN tuning started")
            #dl_models.py
            best_hps = cnn_univariate_conjoint(None, None, None, model_gov[prediction_type]["CNN Univariate Conjoint"]).tune(db_config,hold_out)
            import tensorflow.keras.backend as K
            K.clear_session()
            print("CNN tuning completed")
        except Exception as e:
                print("Error: CNN tuning not completed")
                print(e)     
    return best_hps
                                

def hps_data_hybrid(best_hps, material_no, material_name, market_code,market_name):
        ''' To store the tuned hyperparameters in a specific format for Hybrid models'''
        hps = dict()
        
        hps['units1'] = best_hps['units1']
        hps['units2'] = best_hps['units2']
        
        hps['learning_rate'] = best_hps['learning_rate']
        hps['epochs'] = best_hps['epochs']
        hps['batch'] = best_hps['batch']
        hps['Material'] = material_no
        hps['Material_Name'] = material_name
        hps['Market_Code'] = market_code
        hps['Market_Name'] = market_name
        return hps
    
def hps_data_dnn(best_hps, material_no, material_name, market_code,market_name):
        ''' To store the tuned hyperparameters in a specific format for DNN model'''
        hps = dict()
        
        hps['units1'] = best_hps['units1']
        hps['units2'] = best_hps['units2']
        hps['units3'] = best_hps['units3']
        hps['optimizer'] = best_hps['optimizer']
        hps['epochs'] = best_hps['epochs']
        hps['batch'] = best_hps['batch']
        hps['Material'] = material_no
        hps['Material_Name'] = material_name
        hps['Market_Code'] = market_code
        hps['Market_Name'] = market_name
#         hps.reset_index(inplace=True)
        return hps
    
def tune(processed_series_df, detail_df, run_date, model_gov, tune_gov, model_name,model_class):
        ''' This is the main calling function for Tuning module, it calls the tune module for each model class
            The parameters are read for each model from tune_gov config file'''

        fh=48
        params = None
        try:
            print('Tuning Started for {} {} {}'.format(detail_df['Material'],detail_df['Market_Code'],detail_df['Material_Name']))
            best_hps = model_class(processed_series_df, detail_df, fh, model_gov['univariate'][model_name],tune_gov[model_name],params).tune()
#             
            if(model_name=='DNN'):
                best_hps = hps_data_dnn(best_hps, detail_df['Material'], detail_df['Material_Name'], detail_df['Market_Code'],detail_df['Market_Name'])
            elif(model_name=='Additive Hybrid' or model_name=='Multiplicative Hybrid'):
                best_hps = hps_data_hybrid(best_hps, detail_df['Material'], detail_df['Material_Name'], detail_df['Market_Code'],detail_df['Market_Name'])
            else:
                best_hps['Material'] = detail_df['Material']
                best_hps['Material_Name'] = detail_df['Material_Name']
                best_hps['Market_Code'] = detail_df['Market_Code']
                best_hps['Market_Name'] = detail_df['Market_Name']

    
        except Exception as e:
                print("Error: Tuning not completed")
                print(e)
#         return tune_df
        return best_hps

def eval_forecast(active_skus, run_date, db_config, date1, date2, date3, date4, date5):
 
    
    lag = 3 #It was previously 4, I changed it; Kamal 
    #Reason: Run date is already lagged by a month in the actual calling in the calling script. So in October run date is 9 (which is the correct current forecast month), and for lag 3 calculations later forecast_month later is being used a rundate.month -4 (so forecast month is being shown as lag 4) which is wrong. 
    alpha = {'lag3': 0.6,
             'agg': 0.4}
    print('Starting:')
    print('Getting historical demand and forecast data from Database for MAPE calculation')
    actual_history, forecast_history1, forecast_history2, forecasting_methods = get_forecast_history_multiprocessing(date1, date2, date3, date4, date5, db_config)
    
    forecast_history2.to_csv(r'/mnt/run_results/forecast_history2_test.csv',index=False)
    forecast_history1.to_csv(r'/mnt/run_results/forecast_history1_test.csv',index=False)
    forecast_history = forecast_history1
    
    ##### Added for ERP Wave 2 Mapping - May 2023 ################################################################
    def erp_map(df):     
        sku_map = pd.read_csv(r'/mnt/data/ERP_Wave2_Mapping.csv')
        
        df_mrg = pd.merge(df, sku_map, 
                      how='left', 
                      left_on=['Market_Name', 'Material'], 
                      right_on=['Old Market Name', 'Old SKU'])
        # display_attr(df_mrg)
        
        l = len(df_mrg[df_mrg['Old Market'].notnull()])
        print(l)
        
        if l > 0:
            df_mrg['Material'] = np.where(df_mrg['Old Market'].isnull(), df_mrg['Material'], df_mrg['New SKU'])
            df_mrg['Material_Name'] = np.where(df_mrg['Old Market'].isnull(), df_mrg['Material_Name'], df_mrg['New Description'])
            df_mrg['Market_Name'] = np.where(df_mrg['Old Market'].isnull(), df_mrg['Market_Name'], df_mrg['New Market Name'])
        else:
            pass
        
        df_mrg['Material'] = df_mrg['Material'].astype(int)
        
        return df_mrg.iloc[:, :10]
    
    print('Forecast remapping started')
    forecast_history = erp_map(forecast_history)
    print('Forecast remapping complete')
    forecast_history1.to_csv(r'/mnt/run_results/forecast_history1_test.csv',index=False)
    ##############################################################################################################
    
    forecasting_methods = list(forecast_history.Forecasting_Technique.unique())
    
    batch = active_skus[['Market_Name','Material']]
    
    #preprocessing.py
    print('Combining Forecast History')
    actual_forecast  = combine_forecast_history(actual_history, forecast_history)
    actual_forecast.to_csv(r'/mnt/run_results/actual_forecast_test.csv',index=False)
    
    print('MAPEs calculation started------------------->')
    tasks = range(5)
    pbar = tqdm(total=len(tasks))
    #evaluate.py
    
    all_Lag3_MAPE = multiprocess_lag_mape(run_date, lag, actual_forecast, batch, forecasting_methods)
    print('Calculation for Lag3_MAPEs completed')
    pbar.update(1)
    #all_Lag3_MAPE = pd.read_csv(r'/mnt/run_results/all_Lag3_MAPE_test.csv')#second run
    #all_Lag3_MAPE_last = pd.read_csv(r'/mnt/run_results/all_Lag3_MAPE_last_test.csv')#second run
    
    all_Lag3_MAPE_last = multiprocess_lag_mape3(run_date, lag, actual_forecast, batch, forecasting_methods)#second_run
    print('Calculation for Lag3_MAPE_last completed')
    pbar.update(1)
    
    all_Lag3_MAPE.to_csv(r'/mnt/run_results/all_Lag3_MAPE_test.csv',index=False)#second run
    all_Lag3_MAPE_last.to_csv(r'/mnt/run_results/all_Lag3_MAPE_last_test.csv',index=False)#second run
    
    
    #all_Lag3_MAPE_last = pd.read_csv(r'/mnt/run_results/all_Lag3_MAPE_last_test.csv')
    
    all_Lag3_MAPE_bias = multiprocess_lag_mape2(run_date, lag, actual_forecast, batch, forecasting_methods)#second run
    print('Calculation for Lag3_MAPE_bias completed')
    pbar.update(1)
    #all_Lag3_MAPE_bias = pd.read_csv(r'/mnt/run_results/all_Lag3_MAPE_bias_test.csv')#second run
    all_Lag3_MAPE_bias.to_csv(r'/mnt/run_results/all_Lag3_MAPE_bias_test.csv',index=False)#second run
    
    #print("Bias table" , all_Lag3_MAPE_bias.dtypes)
    #print("Last table" , all_Lag3_MAPE_last.dtypes)
    all_Lag3_MAPE_bias = all_Lag3_MAPE_bias.astype({'Material': object,
                                              'Market_Name': object,
                                              'Forecasting_Method': object,
                                              'Lag3_bias_6m': float})
    all_Lag3_MAPE_last = all_Lag3_MAPE_last.astype({'Material': object,
                                                    'Market_Name': object,
                                                    'Forecasting_Method': object,
                                              'Lag3_last_month': float})
    all_MAPEs_Bias = pd.merge(all_Lag3_MAPE_bias,
    all_Lag3_MAPE_last,
                    how = 'outer')
    print("Merged the lag3 bias 6m and lag 3 mape last month\n")
#     print(all_MAPEs_Bias.head())
 
    
#     print('Here Lag_3_Last month SFA is calculated')
    #all_MAPEs_Bias = pd.read_csv(r'/mnt/run_results/all_MAPEs_Bias_test.csv')
    all_MAPEs_Bias['SFA_last_month'] = 100 - all_MAPEs_Bias['Lag3_last_month']
    all_MAPEs_Bias['SFA_last_month'][all_MAPEs_Bias['SFA_last_month'] <= 0] = 0
    all_MAPEs_Bias = all_MAPEs_Bias.replace([np.inf, -np.inf], np.nan)
    all_MAPEs_Bias.to_csv(r'/mnt/run_results/all_MAPEs_Bias_test.csv',index=False)
    print('Calculation for Lag_3_Last month SFA is completed')
    pbar.update(1)
#     print("5: lag3 6 month Bias and lag3 mape last month and SFA last month have been merged")
    
    all_agg12m_MAPE = multiprocess_agg(run_date, actual_forecast, batch, forecasting_methods)#second run 
    all_agg12m_MAPE.to_csv(r'/mnt/run_results/all_agg12m_MAPE_test.csv',index=False)#second run
    print('Calculation for agg12m_MAPE is completed')
    pbar.update(1)
    pbar.close()
    #all_agg12m_MAPE = pd.read_csv(r'/mnt/run_results/all_agg12m_MAPE_test.csv')
    
    #print(all_agg12m_MAPE)
#     print("Here is 6")
        
    all_Lag3_MAPE['avg_lag3_MAPE'] = pd.to_numeric(all_Lag3_MAPE['avg_lag3_MAPE'])
    all_agg12m_MAPE = all_agg12m_MAPE.astype({'Material': object,
                                              'Market_Name': object,
                                              'Forecasting_Method': object,
                                              'Avg_agg_MAPE': float})
 
    all_Lag3_MAPE = all_Lag3_MAPE.astype({'Material': object,
                                          'Market_Name': object,
                                          'Forecasting_Method': object,
                                          'avg_lag3_MAPE': float})
#     print("7-1")
    
    all_MAPEs = pd.merge(all_Lag3_MAPE,
                         all_agg12m_MAPE,
                         how='outer')
    
#     print("7-2")
    all_MAPEs.to_csv(r'/mnt/run_results/all_MAPEs_1.csv',index=False)
    
    
    #all_MAPEs = pd.read_csv(r'/mnt/run_results/all_MAPEs_1.csv')
    all_error_metrics = pd.merge(all_MAPEs_Bias,
                     all_MAPEs,
                    how = 'outer')
    all_error_metrics.to_csv(r'/mnt/run_results/all_error_metrics.csv',index=False)
    #all_error_metrics = pd.read_csv(r'/mnt/run_results/all_error_metrics.csv')
    all_MAPEs = all_error_metrics[['Material','Market_Name','Forecasting_Method','avg_lag3_MAPE','Avg_agg_MAPE']]
#     print("8-1")
    df_max = pd.DataFrame(columns=['Material','Market_Name','Lag3_Max','Agg12_Max'])
    i=0
    for name, group in all_MAPEs[all_MAPEs['Forecasting_Method'] != 'Ensemble'].groupby(['Material','Market_Name']) :
        df_max.loc[i] = [name[0],name[1],group['avg_lag3_MAPE'].max(),group['Avg_agg_MAPE'].max()]
        i+=1
    diff_all_MAPEs = pd.merge(all_MAPEs[all_MAPEs['Forecasting_Method'] != 'Ensemble'], 
                     df_max,
                    how = 'inner')
#     print("8-2")
    diff_all_MAPEs['Diff_Lag3_MAPE'] = (diff_all_MAPEs['Lag3_Max'] - diff_all_MAPEs['avg_lag3_MAPE'])**3
    diff_all_MAPEs['Diff_Agg12_MAPE'] = (diff_all_MAPEs['Agg12_Max'] - diff_all_MAPEs['Avg_agg_MAPE'])**3
    diff_all_MAPEs['Lag3_Weights'] = diff_all_MAPEs['Diff_Lag3_MAPE']/diff_all_MAPEs.groupby(['Material','Market_Name'])['Diff_Lag3_MAPE'].transform('sum') #Here you're summing up across all forecasting methods only, the agg mape functions run for a given material, product ID and method, and the multiprocess gives them the entire list. So effectively we're calculating the MAPE's on a material, product and forecast method level. ; Kamal
    diff_all_MAPEs['Agg12_Weights'] = diff_all_MAPEs['Diff_Agg12_MAPE']/diff_all_MAPEs.groupby(['Material','Market_Name'])['Diff_Agg12_MAPE'].transform('sum')
    alpha = {'lag3':0.6,
         'agg' :0.4}
#     diff_all_MAPEs = pd.read_csv('ensemble_try_diff_all_MAPEs.csv')
    diff_all_MAPEs['Material'] = diff_all_MAPEs['Material'].astype(int)
    diff_MAPE = pd.DataFrame()
    print('Optimization for Weighted Ensemble started--------------------------->')
    
    for i in tqdm(range(len(active_skus))):
        try:
            diff = diff_all_MAPEs[(diff_all_MAPEs['Material'].astype(str)==active_skus['Material'].iloc[i].astype(str)) & (diff_all_MAPEs['Market_Name'].astype(str)==active_skus['Market_Name'].iloc[i])]
            diff = diff[diff['Forecasting_Method']!='Weighted Ensemble']
            if(diff.empty):
                continue
 
            w0 = np.empty(diff.shape[0])
            w0.fill(1/diff.shape[0])
        #         print(w0)
            bounds = [(0,0.3) for w in w0]
            def cons1(x):
                return x.sum()-1
            cons = [{'type':'eq',
                        'fun': cons1}]
            def objective_max(weights, diff):
                    diff.replace([np.inf, -np.inf], 0)
                    diff['Diff_Lag3_MAPE'] = diff['Diff_Lag3_MAPE'].fillna(0)
                    diff['Diff_Agg12_MAPE'] = diff['Diff_Agg12_MAPE'].fillna(0)
 
                    lag_metrics = np.cbrt(diff['Diff_Lag3_MAPE']).to_numpy()
 
                    fn1 = np.average(lag_metrics,weights = weights)
                #     agg_metrics = diff['Agg_wts'].to_numpy()
                    agg_metrics = np.cbrt(diff['Diff_Agg12_MAPE']).to_numpy()
                    fn2 = np.average(agg_metrics,weights = weights)
                #     return mean_squared_error(actuals,y_ens)
                    return -(0.6*fn1+0.4*fn2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                res_max = minimize(objective_max, x0 = w0, method = 'trust-constr',args = (diff,),bounds = bounds, constraints = cons,options = {'maxiter':100})
            diff['weighted_perc_score'] = res_max.x
            diff_MAPE = pd.concat([diff_MAPE,diff])
        except:
            print("Exception detected for weight optimization for {}".format(active_skus['Market_Name'].iloc[i]))
            
    diff_MAPE.reset_index(inplace=True)
    diff_MAPE.drop('index',axis=1,inplace=True)
    diff_all_MAPEs = diff_MAPE
#     diff_all_MAPEs['weighted_perc_score_1']= diff_all_MAPEs['Lag3_Weights']*alpha['lag3']+ diff_all_MAPEs['Agg12_Weights']*alpha['agg']
    diff_all_MAPEs['Year'] = run_date.strftime('%Y')
    diff_all_MAPEs['Month'] = run_date.strftime('%m')
    diff_all_MAPEs['Date'] = run_date.strftime('%Y/%m')
    
    diff_all_MAPEs.to_csv(r'/mnt/run_results/difference_all_MAPEs_2.csv',index=False)
    #diff_all_MAPEs = pd.read_csv(r'/mnt/run_results/all_MAPEs_2.csv')
#     print("9")
    
    #forecast_history2.to_csv(r'/mnt/run_results/forecast_df.csv',index=False)
#     diff_all_MAPEs[diff_all_MAPEs['Forecasting_Method'] != 'Ensemble'].to_csv(r'/mnt/run_results/difference_all_mape_df.csv',index=False)
    
    #ensembling.py
    all_forecast, weighted_ens = weighted_ensemble(forecast_history2, diff_all_MAPEs[diff_all_MAPEs['Forecasting_Method'] != 'Ensemble'], run_date, db_config)
    
    
#     print("11")
    #write_to_db(db_config,weighted_ens,'Autopilot_Weighted_Ensemble_Test','append')
    
    #all_MAPEs.to_csv(r'/mnt/run_results/all_MAPEs_4.csv',index=False)
    #write_to_db(db_config,all_MAPEs,'History_Of_Mapes','append')
#     print("12")
    
    return all_forecast, weighted_ens, all_error_metrics, all_MAPEs
#     
#     return diff_all_MAPEs



def eval_forecast_celgene(active_skus, run_date, db_config, date1, date2, date3, date4, date5):
    # import pdb; pdb.set_trace()
    
    lag = 4
    alpha = {'lag3': 0.6,
             'agg': 0.4}

    # active_skus = get_active_skus()
    
    #actual_history = pd.read_csv(r'/mnt/run_results/actual_history_test.csv')
   
    #forecast_history = pd.read_csv(r'/mnt/run_results/forecast_history_test.csv')
    #print("1-3")
    #forecasting_methods = pd.read_csv(r'/mnt/run_results/forecasting_methods_test.csv')
    #print("1-4")
    #forecasting_methods = forecasting_methods['0'].values.tolist()
    
    
    actual_history, forecast_history1, forecast_history2, forecasting_methods = get_forecast_history_multiprocessing_celgene(date1, date2, date3, date4, date5, db_config)
    forecast_history2.to_csv(r'/mnt/run_results/forecast_history2_test_celgene.csv',index=False)
    forecast_history1.to_csv(r'/mnt/run_results/forecast_history1_test_celgene.csv',index=False)
    forecast_history = forecast_history1
    forecasting_methods = list(forecast_history.Forecasting_Technique.unique())
    print("1")
    
    actual_history.to_csv(r'/mnt/run_results/actual_history_test_celgene.csv',index=False)
    #forecast_history.to_csv(r'/mnt/run_results/forecast_history_test.csv',index=False)
    
    
    #pd.DataFrame(forecasting_methods).to_csv(r'/mnt/run_results/forecasting_methods_test.csv',index=False)
    print("2")
    batch = active_skus[['Market_Name','Material']]
    
    
    actual_forecast  = combine_forecast_history(actual_history, forecast_history)
    
    #actual_forecast = pd.read_csv(r'/mnt/run_results/actual_forecast_test.csv')
    actual_forecast.to_csv(r'/mnt/run_results/actual_forecast_test_celgene.csv',index=False)
    print("3")
    #all_Lag3_MAPE = pd.read_csv(r'/mnt/run_results/all_Lag3_MAPE_test.csv')
    
    all_Lag3_MAPE = multiprocess_lag_mape(run_date, lag, actual_forecast, batch, forecasting_methods)
    print("4-1")
    all_Lag3_MAPE_last = multiprocess_lag_mape3(run_date, lag, actual_forecast, batch, forecasting_methods)
    
    all_Lag3_MAPE.to_csv(r'/mnt/run_results/all_Lag3_MAPE_test_celgene.csv',index=False)
    all_Lag3_MAPE_last.to_csv(r'/mnt/run_results/all_Lag3_MAPE_last_test_celgene.csv',index=False)
    #all_Lag3_MAPE_last = pd.read_csv(r'/mnt/run_results/all_Lag3_MAPE_last_test.csv')
    print("4-2")
    all_Lag3_MAPE_bias = multiprocess_lag_mape2(run_date, lag, actual_forecast, batch, forecasting_methods)
    #all_Lag3_MAPE_bias = pd.read_csv(r'/mnt/run_results/all_Lag3_MAPE_bias_test.csv')
    all_Lag3_MAPE_bias.to_csv(r'/mnt/run_results/all_Lag3_MAPE_bias_test_celgene.csv',index=False)
    #print("Bias table" , all_Lag3_MAPE_bias.dtypes)
    #print("Last table" , all_Lag3_MAPE_last.dtypes)
    all_Lag3_MAPE_bias = all_Lag3_MAPE_bias.astype({'Material': object,
                                              'Market_Name': object,
                                              'Forecasting_Method': object,
                                              'Lag3_bias_6m': float})
    all_Lag3_MAPE_last = all_Lag3_MAPE_last.astype({'Material': object,
                                                    'Market_Name': object,
                                                    'Forecasting_Method': object,
                                              'Lag3_last_month': float})
    all_MAPEs_Bias = pd.merge(all_Lag3_MAPE_bias,
    all_Lag3_MAPE_last,
                    how = 'outer')

    
    all_MAPEs_Bias.to_csv(r'/mnt/run_results/all_MAPEs_Bias_test_celgene.csv',index=False)
    #all_MAPEs_Bias = pd.read_csv(r'/mnt/run_results/all_MAPEs_Bias_test.csv')
    all_MAPEs_Bias['SFA_last_month'] = 100 - all_MAPEs_Bias['Lag3_last_month']
    all_MAPEs_Bias['SFA_last_month'][all_MAPEs_Bias['SFA_last_month'] <= 0] = 0
    all_MAPEs_Bias = all_MAPEs_Bias.replace([np.inf, -np.inf], np.nan)
    print("5")
    all_agg12m_MAPE = multiprocess_agg(run_date, actual_forecast, batch, forecasting_methods) 
    all_agg12m_MAPE.to_csv(r'/mnt/run_results/all_agg12m_MAPE_test_celgene.csv',index=False)
    
    #all_agg12m_MAPE = pd.read_csv(r'/mnt/run_results/all_agg12m_MAPE_test.csv')
    
    #print(all_agg12m_MAPE)
    print("6")
    print("all agg12m mape:",all_agg12m_MAPE)
        
    all_Lag3_MAPE['avg_lag3_MAPE'] = pd.to_numeric(all_Lag3_MAPE['avg_lag3_MAPE'])
    
    #all_agg12m_MAPE['Avg_agg_MAPE'] = pd.to_numeric(all_agg12m_MAPE['Avg_agg_MAPE'])
    
    all_agg12m_MAPE = all_agg12m_MAPE.astype({'Material': object,
                                              'Market_Name': object,
                                              'Forecasting_Method': object,
                                              'Avg_agg_MAPE': float})

    all_Lag3_MAPE = all_Lag3_MAPE.astype({'Material': object,
                                          'Market_Name': object,
                                          'Forecasting_Method': object,
                                          'avg_lag3_MAPE': float})
    print("7-1")
    
    all_MAPEs = pd.merge(all_Lag3_MAPE,
                         all_agg12m_MAPE,
                         how='left')
    
    print("7-2")
    all_MAPEs.to_csv(r'/mnt/run_results/all_MAPEs_1_celgene.csv',index=False)
    
    
    #all_MAPEs = pd.read_csv(r'/mnt/run_results/all_MAPEs_1.csv')
    all_error_metrics = pd.merge(all_MAPEs_Bias,
                     all_MAPEs,
                    how = 'outer')
    all_error_metrics.to_csv(r'/mnt/run_results/all_error_metrics_celgene.csv',index=False)
    #all_error_metrics = pd.read_csv(r'/mnt/run_results/all_error_metrics.csv')
    all_MAPEs = all_error_metrics[['Material','Market_Name','Forecasting_Method','avg_lag3_MAPE','Avg_agg_MAPE']]
    print("8-1")
    df_max = pd.DataFrame(columns=['Material','Market_Name','Lag3_Max','Agg12_Max'])
    i=0
    for name, group in all_MAPEs[all_MAPEs['Forecasting_Method'] != 'Ensemble'].groupby(['Material','Market_Name']) :
        df_max.loc[i] = [name[0],name[1],group['avg_lag3_MAPE'].max(),group['Avg_agg_MAPE'].max()]
        i+=1
    diff_all_MAPEs = pd.merge(all_MAPEs[all_MAPEs['Forecasting_Method'] != 'Ensemble'], 
                     df_max,
                    how = 'inner')
    print("8-2")
    print("diff lag3 mape:",diff_all_MAPEs)
    diff_all_MAPEs['Diff_Lag3_MAPE'] = (diff_all_MAPEs['Lag3_Max'] - diff_all_MAPEs['avg_lag3_MAPE'])**3
    diff_all_MAPEs['Diff_Agg12_MAPE'] = (diff_all_MAPEs['Agg12_Max'] - diff_all_MAPEs['Avg_agg_MAPE'])**3
    diff_all_MAPEs['Lag3_Weights'] = diff_all_MAPEs['Diff_Lag3_MAPE']/diff_all_MAPEs.groupby(['Material','Market_Name'])['Diff_Lag3_MAPE'].transform('sum')
    diff_all_MAPEs['Agg12_Weights'] = diff_all_MAPEs['Diff_Agg12_MAPE']/diff_all_MAPEs.groupby(['Material','Market_Name'])['Diff_Agg12_MAPE'].transform('sum')
    alpha = {'lag3':0.6,
         'agg' :0.4}
    diff_all_MAPEs['weighted_perc_score']= diff_all_MAPEs['Lag3_Weights']*alpha['lag3']+ diff_all_MAPEs['Agg12_Weights']*alpha['agg']

    diff_all_MAPEs['Year'] = run_date.strftime('%Y')
    diff_all_MAPEs['Month'] = run_date.strftime('%m')
    diff_all_MAPEs['Date'] = run_date.strftime('%Y/%m')
    
    diff_all_MAPEs.to_csv(r'/mnt/run_results/all_MAPEs_2_celgene.csv',index=False)
    #diff_all_MAPEs = pd.read_csv(r'/mnt/run_results/all_MAPEs_2.csv')
    print("9")
    
    forecast_history2.to_csv(r'/mnt/run_results/forecast_df_celgene.csv',index=False)
    diff_all_MAPEs[diff_all_MAPEs['Forecasting_Method'] != 'Ensemble'].to_csv(r'/mnt/run_results/mape_df_celgene.csv',index=False)
    
    print("diff all mape:",diff_all_MAPEs)
    
    
    #test=diff_all_MAPEs[diff_all_MAPEs['Forecasting_Method'] != 'Ensemble']
    
    #for celgene
    #if test.empty:
    #    print(" diff all mape empty")
    #else:
    
    all_forecast, weighted_ens = weighted_ensemble_celgene(forecast_history2, diff_all_MAPEs[diff_all_MAPEs['Forecasting_Method'] != 'Ensemble'], run_date, db_config)


    print("11")
    #write_to_db(db_config,weighted_ens,'Autopilot_Weighted_Ensemble_Test','append')

    all_MAPEs.to_csv(r'/mnt/run_results/all_MAPEs_4_celgene.csv',index=False)
    #write_to_db(db_config,all_MAPEs,'History_Of_Mapes','append')
    print("12")
    return all_forecast, weighted_ens, all_error_metrics, all_MAPEs





def check_for_run(strategy,run_date):
    if type(strategy) == str:

        if strategy == "yearly":
            if run_date.month == 12:
                return True
            else:
                return False
        if strategy == "quarterly":
            if run_date.month % 3 == 0:
                return True
            else:
                return False
        if strategy == "monthly":
            return True

        else:
            raise Exception("Invalid Strategy: {}".format(strategy))