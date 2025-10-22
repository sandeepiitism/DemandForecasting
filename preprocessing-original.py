import numpy as np
import pandas as pd
import copy
from datetime import datetime
import tensorflow.keras.backend as K
from sklearn.ensemble import IsolationForest
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
pd.options.mode.chained_assignment = None
from tqdm import tqdm
from datetime import datetime
from isoweek import Week

def format_date_series(s):
    ''' Returns %m-%Y format of datetime for bms.'''
    return datetime.strptime(s, "%m-%Y")

def format_date_series_celgene(s):
    ''' Returns %m-%Y format of datetime for celgene.'''
    return datetime.strptime(s, "%B-%Y")

def prepare_data(data_df):
    ''' [For BMS] Returns the separated dataframes for one with SKU details and other with time-series data.
    Input - Data from database (From function fetch_data.get_demand())
    Output - detail_df with material details about SKU, series_df with time-series data
    '''
    try:
        
        if(data_df['DP_Trade_Name'].nunique()==1):
            pass
        else:
            text_file = open('autopilot_planning/validation/Name_inconsistency'+str(date.today().month)+'-'+str(date.today().year)+'.txt','a')
            text_file.write('DP_Trade_Name not unique for  {}, {}, {} \n'.format(data_df['Material'].iloc[0],data_df['Material_Name'].iloc[0],data_df['Market_Code'].iloc[0]))
            text_file.close()
        if(data_df['Material_Name'].nunique()==1):
            pass
        else:
            text_file = open('autopilot_planning/validation/Name_inconsistency'+str(date.today().month)+'-'+str(date.today().year)+'.txt','a')
            text_file.write('Material_Name not unique for  {}, {}, {} \n'.format(data_df['Material'].iloc[0],data_df['Material_Name'].iloc[0],data_df['Market_Code'].iloc[0]))
            text_file.close()
          
        a = pd.to_datetime(date.today()) - pd.to_datetime(str(data_df['Year'].iloc[-1])+'/'+str(data_df['Month'].iloc[-1]))
        missing = divmod(a,np.timedelta64(1,'M'))[0]
        if((missing-1)>0):
            text_file = open('autopilot_planning/validation/Data_Missing'+str(date.today().month)+'-'+str(date.today().year)+'.txt','a')
            text_file.write('Demand data is missing for last {} months for {}, {}, {}\n'.format((missing-1),data_df['Material'].iloc[0],data_df['Material_Name'].iloc[0],data_df['Market_Code'].iloc[0]))
            text_file.close()
        
        data_df['DP_Trade_Name'] = data_df['DP_Trade_Name'].iloc[-1]
        detail_cols = ["Material", "Material_Name", "Market_Code", "Market_Name", "DP_Trade_Name"]
        data_df["Date"] = (data_df["Month"].map(str) + "-" + data_df["Year"].map(str)).map(format_date_series)
        data_df = data_df.pivot_table(index=detail_cols, columns="Date", values="Total_Demand").reset_index()
        data_df.columns = detail_cols + [col for col in data_df.columns[len(detail_cols):]]
        # print(data_df)
        data_df = data_df.iloc[0]
        series_df = data_df.drop(detail_cols)
        detail_df = data_df[detail_cols]
        return series_df, detail_df
    except:
        print("Exception occured for data_df while indexing ",data_df)
        return pd.DataFrame(), pd.DataFrame()
    
def prepare_data_NPI(data_df):
    ''' [For BMS] Returns the separated dataframes for one with SKU details and other with time-series data.
    Input - Data from database (From function fetch_data.get_demand())
    Output - detail_df with material details about SKU, series_df with time-series data
    '''
    try:
        
        if(data_df['DP_Trade_Name'].nunique()==1):
            pass
        else:
            text_file = open('autopilot_planning/validation/Name_inconsistency'+str(date.today().month)+'-'+str(date.today().year)+'.txt','a')
            text_file.write('DP_Trade_Name not unique for  {}, {}, {} \n'.format(data_df['Material'].iloc[0],data_df['Material_Name'].iloc[0],data_df['Market_Code'].iloc[0]))
            text_file.close()
        if(data_df['Material_Name'].nunique()==1):
            pass
        else:
            text_file = open('autopilot_planning/validation/Name_inconsistency'+str(date.today().month)+'-'+str(date.today().year)+'.txt','a')
            text_file.write('Material_Name not unique for  {}, {}, {} \n'.format(data_df['Material'].iloc[0],data_df['Material_Name'].iloc[0],data_df['Market_Code'].iloc[0]))
            text_file.close()          
#         a = pd.to_datetime(date.today()) - pd.to_datetime(str(data_df['Year'].iloc[-1])+'/'+str(data_df['Month'].iloc[-1]))
#         missing = divmod(a,np.timedelta64(1,'M'))[0]
#         if((missing-1)>0):
#             text_file = open('autopilot_planning/validation/Data_Missing'+str(date.today().month)+'-'+str(date.today().year)+'.txt','a')
#             text_file.write('Demand data is missing for last {} months for {}, {}, {}\n'.format((missing-1),data_df['Material'].iloc[0],data_df['Material_Name'].iloc[0],data_df['Market_Code'].iloc[0]))
#             text_file.close()
        
        data_df['DP_Trade_Name'] = data_df['DP_Trade_Name'].iloc[-1]
        detail_cols = ["Material", "Material_Name", "Market_Code", "Market_Name", "DP_Trade_Name"]
        data_df["Date"] = data_df['Week_Year']
        data_df = data_df.pivot_table(index=detail_cols, columns="Date", values="Total_Demand").reset_index()
        data_df.columns = detail_cols + [col for col in data_df.columns[len(detail_cols):]]
        # print(data_df)
        data_df = data_df.iloc[0]
        series_df = data_df.drop(detail_cols)
        detail_df = data_df[detail_cols]
        return series_df, detail_df
    except:
        print("Exception occured for data_df while indexing ",data_df)
        return pd.DataFrame(), pd.DataFrame()

def prepare_data_celgene(data_df):
    ''' [For Celgene] Returns the separated dataframes for one with SKU details and other with time-series data.
        Input - Data from database (From function fetch_data.get_demand())
        Output - detail_df with material details about SKU, series_df with time-series data '''    
    try:
        detail_cols = ["Material", "Material_Name",  "Market_Name", "DP_Trade_Name"]
        data_df["Date"] = (data_df["Month"].map(str).str.strip() + "-" + data_df["Year"].map(str).str.strip()).map(format_date_series_celgene)
        data_df = data_df.pivot_table(index=detail_cols, columns="Date", values="Total_Demand").reset_index()
        data_df.columns = detail_cols + [col for col in data_df.columns[len(detail_cols):]]
        # print(data_df)
        data_df = data_df.iloc[0]
        series_df = data_df.drop(detail_cols)
        detail_df = data_df[detail_cols]
        return series_df, detail_df
    except:
        print("Exception occured for data_df while indexing ",data_df)
        return pd.DataFrame(), pd.DataFrame()



def front_trim_series(series_df):
    ''' Returns time-series with NaN values removed. '''
    ind_del = []
    for index, value in series_df.iteritems():
        if np.isnan(value):
            ind_del.append(index)
        else:
            break
    return series_df.drop(ind_del)

def bollinger(df_org,window=12):
    ''' Returns 12 month Moving Average for each time-series'''
    df = df_org.copy()
    df['MA'] = df['demand'].rolling(window).mean()
    df['STD'] = df['demand'].rolling(window).std()

    df['Upper'] = df['MA'] + df['STD']*2
    df['Lower'] = df['MA'] - df['STD']*2
    return df

def isolation(df_org, contamination=0.05, k=1.5):
    ''' Outlier detection method Isolation Forest is implemented on each time series to identify the anomaly
        Input - time-series, contamination factor(%age of total data pts to be identified as outlier, k as multiplier to IQR range
        Output - time-series with only anomolous points'''
    df = df_org.copy()
    def iqr_bounds(scores,k=1.5):
        q1 = scores.quantile(0.25)
        q3 = scores.quantile(0.75)
        iqr = q3 - q1
        low_bound=(q1 - k * iqr)
        up_bound=(q3 + k * iqr)
        #print("Lower bound:{} \nUpper bound:{}".format(lower_bound,upper_bound))
        return low_bound,up_bound

    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=contamination)
    df.dropna(inplace=True)
    df = df[(df['demand']!=0)]
    iso.fit(df[['demand']])
    df['scores']=iso.decision_function(df[['demand']])
    df['anomaly'] = iso.predict(df[['demand']])
    df.loc[df['anomaly']==1,'anomaly']=0
    df.loc[df['anomaly']==-1,'anomaly']=1
    a = pd.DataFrame()
    a['Date'] = df[df['anomaly']==1]['Date']
    a['demand'] = df[df['anomaly']==1]['demand']
    a['scores'] = df[df['anomaly']==1]['scores']
    #IQR Range
    lower_bound,upper_bound=iqr_bounds(a['scores'],k)
    a['anomaly']=0
    a['anomaly']=(a['scores'] < lower_bound)
    a['anomaly']=a['anomaly'].astype(int)
    
    an = pd.DataFrame()
    an['Date'] = a[a['anomaly']==1]['Date']
    an['anomaly'] = a[a['anomaly']==1]['demand']
    
    return an

def anomaly_detection(df_org, contamination=0.05, k=1.5):
    ''' Calls the function bollinger and isolation to combine both the ethods for outlier detection
        Input - time-series, contamination factor(%age of total data pts to be identified as outlier, k as multiplier to IQR range
        Output - Identified anomalous data points from both the methods'''
    df = df_org.copy()
    df_bol = bollinger(df)
    
    df_if = isolation(df, contamination, k)
    
    df_com = pd.merge(df_bol, df_if, how='inner')
    df_anomaly = df_com[(df_com['demand']<df_com['Lower'])| (df_com['demand']>df_com['Upper']) ]
    
    
    #import plotly.express as px
    #import plotly.graph_objects as go
    #material = df_org['Material']
    #fig = px.line(df_org, x='Date',y='demand', width=1000, title = 'Forecast Range ')
    #fig.add_trace(go.Scatter(x=df_anomaly['Date'], y=df_anomaly['anomaly'], name = 'Anomalies',mode='markers'))
    return df_anomaly
    #fig.show()
    
def outlier_correction_new(series_df, contamination=0.05, k=1.5):
        ''' The identified outliers from anomaly_detection function are imputed with higher/lower bollinger band points, also
        distributing the difference of demand among subsequent remaining months of the year proportionalyto maintain year total.
        Input - time-series, contamination factor(%age of total data pts to be identified as outlier, k as multiplier to IQR range
        Output - time-series with corrected observations '''
   
        series_df = pd.DataFrame(series_df)
        series_df.reset_index(inplace=True)
        series_df.columns = ['Date','demand']
        anomaly_df = anomaly_detection(series_df,contamination,k)
        corrected = series_df
        for i in range(len(anomaly_df)):
            flag=0
            if (anomaly_df['Date'].iloc[i].month!=12):
                temp = corrected[(corrected['Date'].dt.month >= anomaly_df['Date'].iloc[i].month) & (corrected['Date'].dt.year == anomaly_df['Date'].iloc[i].year)]
            else:
                temp = corrected[(corrected['Date'].dt.month == anomaly_df['Date'].iloc[i].month) & (corrected['Date'].dt.year == anomaly_df['Date'].iloc[i].year)]
            temp = temp.append(corrected[(corrected['Date'].dt.month != anomaly_df['Date'].iloc[i].month) & (corrected['Date'].dt.year == anomaly_df['Date'].iloc[i].year)])
    #temp.columns = ['Date','demand']
            corrected = corrected.merge(temp, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']

            temp.columns = ['Date','demand']
            corrected.drop('_merge', axis=1, inplace=True)
            corrected.columns = ['Date','demand']
            if(temp['demand'].iloc[0] > anomaly_df['Upper'].iloc[i]):
                a =  temp['demand'].iloc[0]  - anomaly_df['Upper'].iloc[i]
    #temp['demand'].iloc[0] = anomaly_df['Upper'].iloc[0]
                temp1 = pd.DataFrame(temp)
    #temp1 = temp1[1:]
                t2 = temp1[1:]
                t1 = temp1[0:1]
                t2['new_demand'] = t2['demand'] + t2['demand']*a/t2['demand'].sum()
                t1['new_demand'] = anomaly_df['Upper'].iloc[i]
                temp1 = t1.append(t2)
                flag=1
            elif(temp['demand'].iloc[0] < anomaly_df['Lower'].iloc[i]):
                a =  anomaly_df['Lower'].iloc[i] - temp['demand'].iloc[0]   
    #temp['demand'].iloc[0] = anomaly_df['Upper'].iloc[0]
                temp1 = pd.DataFrame(temp)

                t2 = temp1[1:]
                t1 = temp1[0:1]
                t2['new_demand'] = t2['demand'] - t2['demand']*a/t2['demand'].sum()
                t1['new_demand'] = anomaly_df['Lower'].iloc[i]
                temp1 = t1.append(t2)
                flag=1

        #actual_df = temp1.copy()
        #actual_df.drop('new_demand',axis=1, inplace=True)
            if(flag==1):
                temp1.drop('demand', axis=1, inplace=True)

            temp1.columns = ['Date','demand']
        #actual = pd.concat([series_df1,actual_df])
            corrected = pd.concat([corrected,temp1])
            corrected.sort_index(inplace=True)
            
        #actual.set_index('Date', inplace=True)
        #corrected.set_index('Date', inplace=True)
        corrected = corrected.drop_duplicates(subset=['Date'],keep='last',ignore_index=True)
        corrected.set_index('Date', inplace=True)
        corrected.sort_index(inplace=True)
    #     if (anomaly_df.empty):
    #         anomaly_flag = 0
    #     else:
    #         anomaly_flag = 1
    
        return corrected.squeeze(axis=1)
    


def timeseries_preprocessing(series_df, hold_out=0, front_trim=True):
    ''' Pre-processing of each time-series (removing NaN, imputing demand 0 and outlier detection/correction)
        Input - time-series, hold_out to exclude the no. of data points for recent data, front_trim parameter to remove/not remove NaNs
        Output - Processed time-series'''
    #print("Entered Timeseries Preprocessing: \n") #Kamal
    if front_trim == True:
        series_df = front_trim_series(series_df)
    imputed_series_df = series_df.replace(0, np.nan).interpolate(method='linear', limit_direction='backward',
                                                                 limit_area='inside').bfill().ffill()

    if hold_out!=0:
        series_df = series_df[:-hold_out]

#     default value for contamination is 0.05 but for identifying more outliers we have given 0.1 to identify 10% outliers 
#     which can be changed as required
    outlier_series_df = outlier_correction_new(imputed_series_df, contamination=0.1) #Run From next month
    
    ### newly added #########
#     timestamp_nm = str(datetime.now())
#     outlier_series_df.to_csv(r'/mnt/notebooks/Outlier Detection Analysis/Corrected_Shipment.csv') 
    ########################
    
    return outlier_series_df # Next month make series_df to outlier_series_df 


def to_long_data(process_result, run_date):
    ''' Convert forecast results from pipeline to the long format data with year, month and date expanded
        Input - Forecast result, run_date
        Output - Combined forecasts '''
    data = pd.melt(process_result, id_vars=["Material", "Material_Name", "Market_Code", "Market_Name", "DP_Trade_Name",
                                         "Forecasting_Technique"],
                value_vars=process_result.drop(
                    ["Material", "Material_Name", "Market_Code", "Market_Name", "DP_Trade_Name",
                     "Forecasting_Technique"], axis=1).columns,
                var_name='Date', value_name="forecast")

    data['Year'] = pd.to_datetime(data['Date'], format='%Y-%m-%d').apply(lambda z: z.year)
    data['Month'] = pd.to_datetime(data['Date'], format='%Y-%m-%d').apply(lambda z: z.month)
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d').apply(lambda z: datetime.strftime(z, "%Y/%m"))
    data['Forecast_Year'] = run_date.year
    data['Forecast_Month'] = run_date.month
    return data

def to_long_data_NPI(process_result, run_date):
    ''' Convert forecast results from pipeline to the long format data with year, month and date expanded
        Input - Forecast result, run_date
        Output - Combined forecasts '''
    data = pd.melt(process_result, id_vars=["Material", "Material_Name", "Market_Code", "Market_Name", "DP_Trade_Name",
                                         "Forecasting_Technique"],
                value_vars=process_result.drop(
                    ["Material", "Material_Name", "Market_Code", "Market_Name", "DP_Trade_Name",
                     "Forecasting_Technique"], axis=1).columns,
                var_name='Date', value_name="forecast")

    data['Year'] = pd.to_datetime(data['Date'], format='%m/%Y').apply(lambda z: z.year)
    data['Month'] = pd.to_datetime(data['Date'], format='%m/%Y').apply(lambda z: z.month)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%Y').apply(lambda z: datetime.strftime(z, "%Y/%m"))
    data['Forecast_Year'] = run_date.year
    data['Forecast_Month'] = run_date.month
    return data



def prophet_data_preperation(series_df, freq, cap_power):
    ''' Data Preparation for FbProphet model '''
    dataset = pd.DataFrame(series_df)
    dataset.columns = ["y"]
    
    if(freq==12):
        dataset['Day'] = 1
        dataset['ds'] = pd.to_datetime(dataset.index)  
    elif(freq==52):
        dataset['ds'] = 1
        for i in range(len(series_df)):
            year = int(series_df.index[i][3:])
            week = int(series_df.index[i][:2])
            datetime = Week(year,week).monday()
            dataset['ds'].iloc[i] = datetime
        
#     dataset['ds'] = pd.to_datetime(dataset.index).dt.strftime('%W/%Y')

    scaler_to_fix_decimal = 1
    if dataset['y'][(dataset['y'] < 1) & (dataset['y'] > 0)].any() == True:
        scaler_to_fix_decimal = 1 / dataset['y'].min()
        dataset['y'] = dataset['y'] * scaler_to_fix_decimal
    
    df_prophet = dataset[['ds', 'y']]
    df_prophet['cap'] = int(int(df_prophet['y'].max()) * cap_power)
    df_prophet['floor'] = 0

    return df_prophet, dataset, scaler_to_fix_decimal


def add_data(master_X, master_y, X, y):
    ''' For creating training data for CNN '''
    a = np.random.randint(100)
    m = np.random.randint(5)
    c = np.random.randint(100)

    noicex = np.random.rand(60) * X.min() * 0.05
    #to run celgene script uncomment below line and comment above line
    #noicex = np.random.rand(30) * X.min() * 0.05 
    noicey = np.random.rand(1) * y.min() * 0.05

    master_X = np.concatenate((master_X, X + a), axis=0)
    master_y = np.concatenate((master_y, y + a), axis=0)

    # master_X = np.concatenate((master_X, X - a), axis=0)
    # master_y = np.concatenate((master_y, y - a), axis=0)

    master_X = np.concatenate((master_X, X * m + c), axis=0)
    master_y = np.concatenate((master_y, y * m + c), axis=0)

    master_X = np.concatenate((master_X, X + noicex), axis=0)
    master_y = np.concatenate((master_y, y + noicey), axis=0)
    return master_X, master_y


def nan_check(data):
    ''' To check if any data is NaN return True else False '''
    for x in data:
        if np.isnan(x).any():
            return True
    return False


def split_sequence(sequence, n_steps_in, n_steps_out):
    ''' For preparing training data for CNN (To split data in certain sequence)
        Input - data, shape of input data, shape of output data
        Output - Input data, Output data'''
    X, y = list(), list()
    for i in range(len(sequence)):

        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break

        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

        if nan_check(seq_x) or nan_check(seq_y):
            continue

        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_training_data_cnn(data, n_steps_in, n_steps_out):
    ''' For preparing training data for CNN 
        Input - data, shape of input data, shape of output data
        Output - Input data, Output data'''
    master_X, master_y = [], []
    flag = True
    #print('In create_training_data_cnn')
    for index, row in tqdm(data.iterrows(), desc = "Index"):
        
#         print("index:",index)
        X, y = split_sequence(row.values, n_steps_in, n_steps_out)
        if nan_check(X) or nan_check(y) or len(X) == 0:
            continue
        if flag == True:
            master_X = copy.copy(X)
            master_y = copy.copy(y)
            master_X, master_y = add_data(master_X, master_y, X, y)
            flag = False
        else:
            # print(master_X,X)
            master_X = np.concatenate((master_X, X), axis=0)
            master_y = np.concatenate((master_y, y), axis=0)
            master_X, master_y = add_data(master_X, master_y, X, y)
    return (master_X, master_y)


def reset_weights(model):
    
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


def split_sequence_skuwise(sequence, n_steps_in, n_steps_out):
    ''' For preparing training data for DNN (To split data of each sku in certain sequence)
        Input - data, shape of input data, shape of output data
        Output - Input data, Output data'''
        
    X, y = list(), list()
    for i in range(len(sequence)):

        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break

        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

        if nan_check(seq_x) or nan_check(seq_y):
            continue

        assert len(seq_x) == n_steps_in
        assert len(seq_y) == n_steps_out

        X.append(seq_x)
        y.append(seq_y)
    #print("split_sequence_skuwise -X ", X) #Kamal
    
    #print("split_sequence_skuwise- y ", Y) #Kamal
    return np.array(X), np.array(y)


def create_training_data_dnn(data, n_steps_in, n_steps_out):
    ''' For preparing training data for DNN 
        Input - data, shape of input data, shape of output data
        Output - Input data, Output data'''
    #print("In create_training_data_dnn function ")
    #print("data ",data)
    #print("n_in ", n_steps_in, "n_out ", n_steps_out)
    X, y = split_sequence_skuwise(
        data.values, n_steps_in,
        n_steps_out)
    if len(X) == 0 or len(y) == 0:
        #print(X)
        return [], [], True
    else:
        return X, y, False
    
    
# def change_index(x):
#     return datetime.strptime(x, '%b-%Y')

def combine_forecast_history(actual_history, forecast_history):
    ''' Combining forecast history with actual history from database '''
        
    
    categorical_cols = ['Material_Name', 'Material']
    for col in categorical_cols:
        actual_history[col] = actual_history[col].astype('category')
        forecast_history[col] = forecast_history[col].astype('category')

    actual_history['Month'] = actual_history['Month'].map({'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                                           'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11,
                                                           'Dec': 12})
    # Converting Columns to Int Data Types
    forecast_history['Forecast Month'] = forecast_history['Forecast Month'].astype(int)
    forecast_history['month'] = forecast_history['month'].astype(int)
    forecast_history['year'] = forecast_history['year'].astype(int)
    # forecast_history['Material'] = forecast_history['Material'].astype(int)
    
    # # Before joining make sure the right table has unique entries of primary key(on which join is happening)

    # Remove duplicate columns
    forecast_history = forecast_history[list(forecast_history.columns[~forecast_history.columns.duplicated()])]

    # MERGE DATAFRAME TOGETHER SO FORECAST VS ACTUAL VALUES ARE SIDE BY SIDE AND MATCHED CORRECTLY BY SKU
    actual_forecast = pd.merge(forecast_history,
                               actual_history,
                               left_on=['Material', 'Market_Name', 'month', 'year'],
                               right_on=['Material', 'Market_Name', 'Month', 'Year'],
                               how='inner')

    actual_forecast['Material'] = actual_forecast['Material'].astype(int)
    actual_forecast['Year'] = actual_forecast['Year'].astype(int)
    actual_forecast['year'] = actual_forecast['year'].astype(int)
    actual_forecast['Forecast Year'] = actual_forecast['Forecast Year'].astype(int)

    actual_forecast = actual_forecast.sort_values(by=['Forecasting_Technique', 'year', 'month',
                                                      'Forecast Year', 'Forecast Month'])

   
    actual_forecast['sku_id'] = actual_forecast.Market_Name.astype(str) + '_' + actual_forecast.Material.astype(str)
    

    return actual_forecast