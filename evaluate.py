from multiprocessing import Manager, Process, Pool, cpu_count
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import math
from IPython.core.display import HTML
import warnings
from tqdm import tqdm


def multiprocess_lag_mape(run_date, lag, actual_forecast, batch, forecasting_methods):
   
    run_date = datetime.date(run_date.year, run_date.month, 1)
    with Manager() as manager:

        LAG3 = manager.list()  # <-- can be shared between processes.
        
        pool = Pool(cpu_count())
        output = pool.map(get_lagged_MAPE, [(LAG3, actual_forecast, row[0], row[1], forecasting_methods, run_date, lag) for i, row in batch.iterrows()])
        
        LAG3 = list(LAG3)
        all_Lag3_MAPE = pd.concat(LAG3)
        all_Lag3_MAPE = all_Lag3_MAPE.reset_index(drop=True)
        
        return all_Lag3_MAPE


def get_lagged_MAPE(args):
    """
    :param actual_forecast : Data containing both actuals and forecasts
    :param sku :market_name : Name of the market, material : SKU name
    :param forecasting_methods : list of all the forecasting methods
    :param current_date : date for which the foreacasts are calculated
    :param lag : lag against which MAPE is calculated
    """
    LAG3, actual_forecast, market, material, forecasting_methods, current_date, lag = args
    #print("ARguments in get lagged mape:LAG3",LAG3)
    #print("ARguments in get lagged mape:actual",actual_forecast)
    #print("ARguments in get lagged mape:mkt",market)
    #print("ARguments in get lagged mape mat",material)
    #print("ARguments in get lagged mape:method",forecasting_methods)
    #print("ARguments in get lagged mape:date",current_date)
    #print("ARguments in get lagged mape:lag",lag)
    
    
    market_name = market
    #print("for market:",market)
    #print("for material:",material)
    # Creating an empty dataframe with forecasting methods and Average of MAPE
    MAPE_df = pd.DataFrame(columns=['Material', 'Market_Name',
                                    'Forecasting_Method',
                                    'avg_lag3_MAPE'])

    for fc in range(len(forecasting_methods)):
        # Basic initialisation
        cur_date = current_date
        cur_month = current_date.month
        cur_year = current_date.year
        fc_month = cur_month - (lag)
        fc_year = cur_year

        #print(cur_date, cur_month, cur_year, fc_month, fc_year)
         #update made here
        actual_forecast_tmp = actual_forecast[(actual_forecast['Forecasting_Technique'] == forecasting_methods[fc]) &
                                              (actual_forecast['Material'] == int(material)) &
                                              (actual_forecast['Market_Name'] == market_name)].reset_index(
            drop=True).copy()
       
        
        #print("actual_forecast_tmp:",actual_forecast_tmp)
        
        
        i = 1
        result = []
        while (cur_date != (current_date - relativedelta(months=+6))):
            # print("i",i)
            actual_forecast_sub = actual_forecast_tmp[(actual_forecast_tmp['month'] == cur_month) &
                                                      (actual_forecast_tmp['year'] == cur_year)]

            #       print(cur_date,i month and year considering the cases of year shift
            if i == 1:
                if cur_month - (lag) > 0:
                    fc_month = cur_month - (lag)
                    fc_year = cur_year
                else:
                    fc_month = 12 - abs(cur_month - (lag))
                    fc_year = cur_year - 1

            # ------------------------------------#
            # Updating the forecast month and year
            # ------------------------------------#

            elif fc_month == 1:
                fc_month = 12
                fc_year = cur_year - 1
            else:
                fc_month = fc_month - 1
                fc_year = fc_year
            # print("Forecast month",fc_month,"Forecast year",fc_year)
            # Subsetting for every forecast month and year
            #print("actual forecast sub:",actual_forecast_sub)
            actual_forecast_sub_tmp = (actual_forecast_sub[(actual_forecast_sub['Forecast Month'] == fc_month) &
                                                          (actual_forecast_sub['Forecast Year'] == fc_year)]).drop_duplicates()
            
            #print("actual_forecast_sub_tmp:",actual_forecast_sub_tmp)
            
            
            # Calculating the MAPE and appending it for every month
            # print("actual_forecast_sub_temp", actual_forecast_sub_tmp)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                lag3_MAPE = (abs(actual_forecast_sub_tmp[['demand']].values - actual_forecast_sub_tmp[['Demand']].values) / 
                        actual_forecast_sub_tmp[['Demand']].values) * 100
                
            #print("for market:",market)
            #print("for material:",material)
            #print("lag3_MAPE in get lagged mape:",lag3_MAPE)
            #print("actual forecast demand value:",actual_forecast_sub_tmp[['Demand']].values)
            
            
            try:
                if len(lag3_MAPE) == 0:
                    pass
                elif (math.isnan(lag3_MAPE)):
                    pass
                elif (math.isinf(lag3_MAPE)):
                    pass
                else:
                    result.append(lag3_MAPE)
            except:
                print("Exception occurred in lag3_MAPE calculation \nLag 3 Mape is:",lag3_MAPE)
            # -------------------------------------------------------------#
            # Updating the current month and year and updating the cur_date
            # -------------------------------------------------------------#
            # print("result",result)
            if cur_month == 1:
                cur_month = 12
                cur_year = cur_year - 1
            else:
                cur_month = cur_month - 1
                cur_year = cur_year

            cur_date = datetime.date(cur_year, cur_month, 1)
            # print("cur_date: {}, cureent:{}".format(cur_date,current_date))
            i = i + 1
        # ------------------------------------#
        # Updating the MAPE table with mean
        # ------------------------------------#
        #print("result",result)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            MAPE_df.loc[fc] = [material,
                           market_name,
                           forecasting_methods[fc],
                           np.nanmean(result)]
        # print("mape_df",MAPE_df)
        result = []
        # print(actual_forecast_sub_tmp)
    #print("MAPE_df", MAPE_df)
    LAG3.append(MAPE_df)
    return (MAPE_df)

def multiprocess_agg(run_date, actual_forecast, batch, forecasting_methods):

    run_date = datetime.date(run_date.year, run_date.month, 1)
   
    with Manager() as manager:

        AGG12 = manager.list()
        
        pool = Pool(cpu_count())
        output = pool.map(get_agg_MAPE, [(AGG12, actual_forecast, row[0], row[1], forecasting_methods, run_date) for i, row in batch.iterrows()])

        AGG12 = list(AGG12)
        all_agg12m_MAPE = pd.concat(AGG12)
        all_agg12m_MAPE = all_agg12m_MAPE.reset_index(drop=True)
        return all_agg12m_MAPE


def get_agg_MAPE(args):
    """
    :param actual_forecast : Data containing both actuals and forecasts
    :param sku :market_name : Name of the market, material : SKU name
    :param forecasting_methods : list of all the forecasting methods
    :param current_date : date for which the foreacasts are calculated
    """
    
    AGG12, actual_forecast, market, material, forecasting_methods, current_date = args
    material = material
    market_name = market

    # Creating empty result data frame
    MAPE_df = pd.DataFrame(columns=['Material', 'Market_Name',
                                    'Forecasting_Method',
                                    'Avg_agg_MAPE'])

    # Creating empty temp dataframe inside every forecasting technique that'd store the MAPEs for each month in the six month average period with respective Forecast Dates
    MAPE_with_forecast_date_df = pd.DataFrame(columns=['Material', 'Market_Name',
                                                       'Forecasting_Method', 'Forecast_Date',
                                                       'Agg_12month_MAPE'])
    for fc in range(len(forecasting_methods)):

        # Basic initialisation
        cur_date = current_date

        i = 1
        result = []
        cur_date_MAPE = 0

        # Subsetting for a forcasting method
        actual_forecast_sub = actual_forecast[(actual_forecast['Forecasting_Technique'] ==
                                               forecasting_methods[fc])].reset_index(drop=True)
        
        #if(material == '1163512' and market_name == 'AUSTRALIA'):
            #print("actual_forecast_sub", actual_forecast_sub)
            #actual_forecast_sub.to_csv(r'/mnt/run_results/actual_forecast_sub_test.csv',index=False)
        #display(HTML(actual_forecast_sub.to_html()))
        
        # pdb.set_trace()

        # print(forecasting_methods[fc])

        # Exception Handling
        if actual_forecast_sub.empty:
            #print("Actual_forecast_sub was empty so this {} agg_12_mape data is empty".format(material))
            MAPE_df.loc[fc] = [material, market_name, forecasting_methods[fc], ""]
            continue
        else:
            # Loop to calculate MAPE for most recent 6 months from current date
            j = 0
            while (cur_date != (current_date - relativedelta(months=+6))):

                # Blank lists for storing 12 month actuals and forecasts
                actuals_sum = []
                forecasts_sum = []

                for_date = cur_date - relativedelta(months=+12)
                #print("for_date",for_date)
                try:
                    
                    # Subsetting on Material, Market and the Forecast Date
                    actual_forecast_sub_date = actual_forecast_sub[(actual_forecast_sub['Material'] == int(material)) & (
                                actual_forecast_sub['Market_Name'] == market_name) & (actual_forecast_sub[
                                                                                          'Forecast Year'] == for_date.year) & (
                                                                               actual_forecast_sub[
                                                                                   'Forecast Month'] == for_date.month)].reset_index(
                        drop=True)
                    #if(material == '1163512' and market_name == 'AUSTRALIA'):
                        #print("actual_forecast_sub_date",actual_forecast_sub_date)
                        #actual_forecast_sub_date.to_csv(r'/mnt/run_results/actual_forecast_sub_date.csv',index=False)
                    #display(HTML(actual_forecast_sub_date.to_html()))

                    while (cur_date > for_date):

                        actual_forecast_sub_for_date = actual_forecast_sub_date[
                            (actual_forecast_sub_date['month'] == cur_date.month) & (
                                        actual_forecast_sub_date['year'] == cur_date.year)].reset_index(drop=True)
                        if actual_forecast_sub_for_date.empty:
                            cur_date = cur_date - relativedelta(months=+1)
                            break

                        actuals_sum.append(actual_forecast_sub_for_date['Demand'][0])
                        forecasts_sum.append(actual_forecast_sub_for_date['demand'][0])

                        cur_date = cur_date - relativedelta(months=+1)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        if (len(actuals_sum) > 0 and len(forecasts_sum) > 0):
                            MAPE_with_forecast_date_df.loc[j] = [material, market_name, forecasting_methods[fc], for_date, (
                                    (abs(sum(forecasts_sum) - sum(actuals_sum)) / sum(actuals_sum) )* 100)]
                            result.append((abs(sum(forecasts_sum) - sum(actuals_sum)) / sum(actuals_sum)) * 100)
                    

                    j = j + 1;
                    cur_date = current_date - relativedelta(months=+j)
                except:
                    print("\n Exception has been detected in Agg 12 Mape for  this {}".format(material))
#             To ignore the warning message which might be annoying        
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                MAPE_df.loc[fc] = [material, market_name, forecasting_methods[fc], np.nanmean(result)]#nanmean to ignore nan values
                

    AGG12.append(MAPE_df)
    return MAPE_df

def get_lagged_MAPE2(args):
    """
    :param actual_forecast : Data containing both actuals and forecasts
    :param sku :market_name : Name of the market, material : SKU name
    :param forecasting_methods : list of all the forecasting methods
    :param current_date : date for which the foreacasts are calculated 
    :param lag : lag against which MAPE is calculated
    """
    LAG3_2, actual_forecast, market_name, material, forecasting_methods, current_date, lag = args
    #Creating an empty dataframe with forecasting methods and Average of MAPE
    MAPE_df = pd.DataFrame(columns=['Material', 'Market_Name',
                                    'Forecasting_Method',
                                    'Lag3_bias_6m', 'Date_Actuals'])
    
    for fc in range(len(forecasting_methods)):
        # Basic initialisation 
        cur_date = current_date
        cur_month = current_date.month
        cur_year = current_date.year
        fc_month = cur_month-(lag)
        fc_year = cur_year
#         print("Entered Lag3 bias 6 months", cur_date,cur_month,cur_year,fc_month,fc_year)
        #print(cur_date, cur_month, cur_year, fc_month, fc_year)
        try:
            actual_forecast_tmp = actual_forecast[(actual_forecast['Forecasting_Technique']==forecasting_methods[fc]) &
                                                  (actual_forecast['Material'] == int(material)) &
                                                  (actual_forecast['Market_Name'] == market_name)].reset_index(drop=True).copy()
        
        
            i = 1
            result = []
            actuals_sum = []
            forecasts_sum = []
            while(cur_date != (current_date - relativedelta(months=+6))):
                #print("i",i)
                actual_forecast_sub = actual_forecast_tmp[(actual_forecast_tmp['month']==cur_month) & 
                                                      (actual_forecast_tmp['year']==cur_year)]

                #print(cur_date,i month and year considering the cases of year shift  
                if i == 1:
                    if cur_month-(lag) > 0:
                        fc_month = cur_month-(lag)
                        fc_year = cur_year
                    else:
                        fc_month = 12 - abs(cur_month-(lag))
                        fc_year = cur_year-1
                #------------------------------------#
                # Updating the forecast month and year
                #------------------------------------#
                elif fc_month == 1:
                    fc_month = 12
                    fc_year = cur_year-1
                else:
                    fc_month = fc_month-1
                    fc_year = fc_year
                #print("Forecast month",fc_month,"Forecast year",fc_year)
                #Subsetting for every forecast month and year
                actual_forecast_sub_tmp = (actual_forecast_sub[(actual_forecast_sub['Forecast Month']== fc_month)&
                                                          (actual_forecast_sub['Forecast Year']== fc_year)]).drop_duplicates()

                #Calculating the MAPE and appending it for every month 
                #print("actual_forecast_sub_temp['Demand'].values", actual_forecast_sub_tmp['Demand'].values)
                #print("actual_forecast_sub_temp['demand'].values", actual_forecast_sub_tmp['demand'].values)
                if(any(actual_forecast_sub_tmp['Demand'].values)):
                    actuals_sum.append(actual_forecast_sub_tmp['Demand'].values[0])
                if(any(actual_forecast_sub_tmp['demand'].values)):
                    forecasts_sum.append(actual_forecast_sub_tmp['demand'].values[0])
                #-------------------------------------------------------------#
                # Updating the current month and year and updating the cur_date  
                #-------------------------------------------------------------#
                #print("result",result)
                if cur_month == 1:
                    cur_month = 12
                    cur_year = cur_year-1
                else:
                    cur_month = cur_month-1
                    cur_year = cur_year

                cur_date = datetime.date(cur_year, cur_month,1)
                #print("cur_date", cur_date)
                i = i+1
            #------------------------------------#
            # Updating the MAPE table with mean 
            #------------------------------------#
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if (len(actuals_sum)>0 and len(forecasts_sum)>0):

                    MAPE_df.loc[fc] = [material,
                                   market_name,
                                   forecasting_methods[fc],
                                   ((sum(forecasts_sum) - sum(actuals_sum))/ sum(actuals_sum) *100),
                                   str(current_date.year) +'/' + current_date.strftime('%m')]

#             print("mape_df",MAPE_df)
            result = []
        except:
            print("Exception detected for Lag3 bias 6Months for {}".format(material))
            #print(actual_forecast_sub_tmp)
    LAG3_2.append(MAPE_df)        
    return (MAPE_df)

def multiprocess_lag_mape2(run_date, lag, actual_forecast, batch, forecasting_methods):
   
    run_date = datetime.date(run_date.year, run_date.month, 1)
    with Manager() as manager:

        LAG3_2 = manager.list()  # <-- can be shared between processes.
        
        pool = Pool(cpu_count())
        output = pool.map(get_lagged_MAPE2, [(LAG3_2, actual_forecast, row[0], row[1], forecasting_methods, run_date, lag) for i, row in batch.iterrows()])
        
        LAG3_2 = list(LAG3_2)
        all_Lag3_MAPE_bias = pd.concat(LAG3_2)
        all_Lag3_MAPE_bias = all_Lag3_MAPE_bias.reset_index(drop=True)
        return all_Lag3_MAPE_bias

    
    
def get_lagged_MAPE3(args):
    """
    :param actual_forecast : Data containing both actuals and forecasts
    :param sku :market_name : Name of the market, material : SKU name
    :param forecasting_methods : list of all the forecasting methods
    :param current_date : date for which the foreacasts are calculated 
    :param lag : lag against which MAPE is calculated
    """

    LAG3_3, actual_forecast, market_name, material, forecasting_methods, current_date, lag = args
    
    #Creating an empty dataframe with forecasting methods and Average of MAPE
    MAPE_df = pd.DataFrame(columns=['Material', 'Market_Name',
                                    'Forecasting_Method',
                                    'Lag3_last_month', 'Date_Actuals'])
#     print("Entered Lag 3 MAPE Last month")
    for fc in range(len(forecasting_methods)):
        # Basic initialisation 
        cur_date = current_date
        cur_month = current_date.month
        cur_year = current_date.year
        fc_month = cur_month-(lag)
        fc_year = cur_year
        
#         print("Entered Lag3_last_month",cur_date, cur_month, cur_year, fc_month, fc_year)
        
        actual_forecast_tmp = actual_forecast[(actual_forecast['Forecasting_Technique']==forecasting_methods[fc]) &
                                              (actual_forecast['Material'] == int(material)) &
                                              (actual_forecast['Market_Name'] == market_name)].reset_index(drop=True).copy()
        
        i = 1
        result = []
        while(cur_date != (current_date - relativedelta(months=+1))):
            #print("i",i)
            actual_forecast_sub = actual_forecast_tmp[(actual_forecast_tmp['month']==cur_month) & 
                                                  (actual_forecast_tmp['year']==cur_year)]
            
            #print(cur_date,i month and year considering the cases of year shift  
            if i == 1:
                if cur_month-(lag) > 0:
                    fc_month = cur_month-(lag)
                    fc_year = cur_year
                else:
                    fc_month = 12 - abs(cur_month-(lag))
                    fc_year = cur_year-1
            #------------------------------------#
            # Updating the forecast month and year
            #------------------------------------#
            elif fc_month == 1:
                fc_month = 12
                fc_year = cur_year-1
            else:
                fc_month = fc_month-1
                fc_year = fc_year
            #print("Forecast month",fc_month,"Forecast year",fc_year)
            #Subsetting for every forecast month and year
            actual_forecast_sub_tmp = (actual_forecast_sub[(actual_forecast_sub['Forecast Month']== fc_month)&
                                                      (actual_forecast_sub['Forecast Year']== fc_year)]).drop_duplicates()
            
            #Calculating the MAPE and appending it for every month 
            #print("actual_forecast_sub_temp", actual_forecast_sub_tmp)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                lag3_MAPE = abs(actual_forecast_sub_tmp[['demand']].values-actual_forecast_sub_tmp[['Demand']].values)/actual_forecast_sub_tmp[['Demand']].values*100
            try:
                if len(lag3_MAPE) == 0:
                    pass
                elif (math.isnan(lag3_MAPE)):
                    pass
                elif (math.isinf(lag3_MAPE)):
                    pass
                else:
                    result.append(lag3_MAPE)
            except:
                print("lag3_MAPE",lag3_MAPE)
            #-------------------------------------------------------------#
            # Updating the current month and year and updating the cur_date  
            #-------------------------------------------------------------#
            #print("result",result)
            if cur_month == 1:
                cur_month = 12
                cur_year = cur_year-1
            else:
                cur_month = cur_month-1
                cur_year = cur_year

            cur_date = datetime.date(cur_year, cur_month,1)
            #print("cur_date", cur_date)
            i = i+1
        #------------------------------------#
        # Updating the MAPE table with mean 
        #------------------------------------#
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            MAPE_df.loc[fc] = [material,
                           market_name,
                           forecasting_methods[fc],
                           np.nanmean(result),#nanmean to ignore nan values
                           str(current_date.year) +'/' + current_date.strftime('%m')]
        #print("mape_df",MAPE_df)
        #print(result)
        result = []
        
            #print(actual_forecast_sub_tmp)
    LAG3_3.append(MAPE_df)    
#     print('last line of lag 3 last month MAPE') 
    return (MAPE_df)
    

def multiprocess_lag_mape3(run_date, lag, actual_forecast, batch, forecasting_methods):
   
    run_date = datetime.date(run_date.year, run_date.month, 1)
    with Manager() as manager:

        LAG3_3 = manager.list()  # <-- can be shared between processes.
        
        pool = Pool(cpu_count())
        output = pool.map(get_lagged_MAPE3, [(LAG3_3, actual_forecast, row[0], row[1], forecasting_methods, run_date, lag) for i, row in batch.iterrows()])
        
        LAG3_3 = list(LAG3_3)
        all_Lag3_MAPE_last = pd.concat(LAG3_3)
        all_Lag3_MAPE_last = all_Lag3_MAPE_last.reset_index(drop=True)
        return all_Lag3_MAPE_last

