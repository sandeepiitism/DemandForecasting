import pandas as pd
#from .autopilot_planning.write_data import *


class ensemble:

    def __init__(self, pred_df):
        self.pred_df = pred_df


    def mean(self):
        ensemb_index = len(self.pred_df)
        ensemble_forecast = list()
        for col in self.pred_df.columns:
            ensemble_forecast += [self.pred_df[col].mean()]
            self.pred_df.loc[ensemb_index, col] = self.pred_df[col].mean()

        return self.pred_df

# PULLS IN THE CURRENT FORECASTS AND USING THE MAPE TABLE CREATES A WEIGHTED ENSEMBLE THROUGH GROUPING DATA AND MULTIPLYING
# FORECAST DEMAND BY TEH CORRECT COORESPONDING WEIGHTED SCORE AND AGGREGATING
def weighted_ensemble(forecast_df, mape_df, run_date, db_config):
    # FIND THE CURRENT DATE AND PULL FORECAST DATA FOR PREVIOUS MONTH TO APPLY WEIGHTED ENSEMBLE TO
    # ONE MONTH BACK SINCE ACTUALS ARENT OBTAINED UNTIL THE NEXT MONTH
    current_date = run_date

    
    #forecast_df.to_csv(r'/mnt/run_results/forecast_df_test.csv',index=False)
    # Re-cast Data types to allow datasets to be merged
    forecast_df['Material'] = forecast_df['Material'].astype(float)
    mape_df['Material'] = mape_df['Material'].astype(float)
    forecast_df1 = forecast_df[forecast_df['Forecasting_Technique']!='Ensemble']
    # Perform merge and calcualte a scale column that needs to be aggregated to obtain single weighted ensemble forecast value
    temp = pd.merge(forecast_df1,
                    mape_df,
                    left_on=['Material', 'Market_Name', 'Forecasting_Technique'],
                    right_on=['Material', 'Market_Name', 'Forecasting_Method'],
                    how='outer')
    #temp.to_csv(r'/mnt/run_results/temp_test_1.csv',index=False)
    temp['scale'] = (temp['weighted_perc_score'] * temp['demand'])
    #temp.to_csv(r'/mnt/run_results/temp_test_2_.csv',index=False)
    # COMMENT TO REMOVE WEIGHTED ENSEMBLE VALUES OF ZERO AS NULL VALUES GET DROPPED
    temp.dropna(subset=['weighted_perc_score', 'scale'], inplace=True)
    #temp.to_csv(r'/mnt/run_results/temp_test_3.csv',index=False)
    # Aggregate Weighted Ensemble total for each date by performing a group by
    temp = temp.groupby(['Market_Name', 'Material', 'Material_Name', 'Forecast Year',
                         'Forecast Month', 'year', 'month', 'Date'])['scale'].sum().reset_index()
    #temp.to_csv(r'/mnt/run_results/temp_test_4.csv',index=False)
    # Creating identifier column to match usual forecast format
    temp['Forecasting_Technique'] = 'Weighted Ensemble'

    # Rename columns to match usual forecast format
    temp = temp.rename(columns={'scale': 'demand', 'Date': 'Date'})
    #print(temp.columns)
    #print("temp")
    #print(temp)
    # Create a list to order the DF in a way that matches usual forecast format
    order = ['Forecasting_Technique', 'Market_Name', 'Material', 'Material_Name', 'demand', 'Forecast Year',
             'Forecast Month', 'year',
             'month', 'Date']

    ensemble_df = temp[order]
    
    
    
    #ensemble_df.to_csv(r'/mnt/run_results/weighted_ensemble_df_test.csv',index=False)
    # Append the Weighted Ensemble dataframe ouput back to most current forecasts sicne theyve already been created
    final_df = ensemble_df.append(forecast_df, sort=False)

    # Fix column names
    final_df = final_df.rename(columns={'Forecast Year': 'Forecast_Year', 'Forecast Month': 'Forecast_Month',
                                        'year': 'Year', 'month': 'Month'})

    final_df = final_df.sort_values(by=['Market_Name', 'Material_Name', 'Forecasting_Technique', 'Year', 'Month'])
    #final_df.to_csv(r'/mnt/run_results/final_df_test.csv',index=False)
    return (final_df.reset_index(drop=True), ensemble_df.reset_index(drop=True))


def weighted_ensemble_celgene(forecast_df, mape_df, run_date, db_config):
    print("forecast df in ensemble:",forecast_df)
    # FIND THE CURRENT DATE AND PULL FORECAST DATA FOR PREVIOUS MONTH TO APPLY WEIGHTED ENSEMBLE TO
    # ONE MONTH BACK SINCE ACTUALS ARENT OBTAINED UNTIL THE NEXT MONTH
    print("inside function weighted ensemble celgene")
    current_date = run_date
    #print("forecast df in ensemble:",forecast_df)
    print("mape df in ensemble:",mape_df)
    
    forecast_df.to_csv(r'/mnt/run_results/forecast_df_test_celgene.csv',index=False)
    # Re-cast Data types to allow datasets to be merged
    forecast_df['Material'] = forecast_df['Material'].astype(float)
    mape_df['Material'] = mape_df['Material'].astype(float)
    forecast_df1 = forecast_df[forecast_df['Forecasting_Technique']!='Ensemble']
    print("forecast df1 in ensemble:",forecast_df1)
    
    # Perform merge and calcualte a scale column that needs to be aggregated to obtain single weighted ensemble forecast value
    temp = pd.merge(forecast_df1,
                    mape_df,
                    left_on=['Material', 'Market_Name', 'Forecasting_Technique'],
                    right_on=['Material', 'Market_Name', 'Forecasting_Method'],
                    how='inner')
    #print("temp after merging with forecast df1:",temp)
    #print("temp columns after merging with forecast df1:",temp.columns)
    temp.to_csv(r'/mnt/run_results/temp_test_1_celgene.csv',index=False)
    temp['scale'] = (temp['weighted_perc_score'] * temp['demand'])
    temp.to_csv(r'/mnt/run_results/temp_test_2_celgene.csv',index=False)
    # COMMENT TO REMOVE WEIGHTED ENSEMBLE VALUES OF ZERO AS NULL VALUES GET DROPPED
    temp.dropna(subset=['weighted_perc_score', 'scale'], inplace=True)
    temp.to_csv(r'/mnt/run_results/temp_test_3_celgene.csv',index=False)
    #print("temp columns before groupby:",temp.columns)
    # Aggregate Weighted Ensemble total for each date by performing a group by
    
    #if temp.empty:
    #    print("empty temp")
    #else:
        #print("wadhwa")
    temp = temp.groupby(['Market_Name', 'Material', 'Material_Name', 'Forecast Year',
                    'Forecast Month', 'year', 'month', 'Date_x'])['scale'].sum().reset_index()

# for celgene
#temp = temp.groupby(['Market_Name', 'Material', 'Material_Name', 'Forecast Year',
#                     'Forecast Month', 'year', 'month', 'Date_x'],as_index=False,group_keys=False)['scale'].sum()


    #print("temp columns after groupby:",temp.columns)
    temp.to_csv(r'/mnt/run_results/temp_test_4_celgene.csv',index=False)
    # Creating identifier column to match usual forecast format
    temp['Forecasting_Technique'] = 'Weighted Ensemble'

    # Rename columns to match usual forecast format
    temp = temp.rename(columns={'scale': 'demand', 'Date_x': 'Date'})
    print(temp.columns)
    print("temp")
    print(temp)
    # Create a list to order the DF in a way that matches usual forecast format
    order = ['Forecasting_Technique', 'Market_Name', 'Material', 'Material_Name', 'demand', 'Forecast Year',
             'Forecast Month', 'year',
             'month', 'Date']

    ensemble_df = temp[order]



    ensemble_df.to_csv(r'/mnt/run_results/weighted_ensemble_df_test_celgene.csv',index=False)
    # Append the Weighted Ensemble dataframe ouput back to most current forecasts sicne theyve already been created
    final_df = ensemble_df.append(forecast_df, sort=False)

    # Fix column names
    final_df = final_df.rename(columns={'Forecast Year': 'Forecast_Year', 'Forecast Month': 'Forecast_Month',
                                        'year': 'Year', 'month': 'Month'})

    final_df = final_df.sort_values(by=['Market_Name', 'Material_Name', 'Forecasting_Technique', 'Year', 'Month'])
    final_df.to_csv(r'/mnt/run_results/final_df_test_celgene.csv',index=False)
    return (final_df.reset_index(drop=True), ensemble_df.reset_index(drop=True))
