#ML model for IBM HAck challenge 2020

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def getWeekYear(data):
    data.date = data.date.apply(lambda x: str(str(x)[0:4])+"-"+str(dt.date(int(str(x)[0:4]),int(str(x)[5:7]),int(str(x)[8:10])).isocalendar()[1]).zfill(2))
    data = data.groupby(['item','date'])['sales'].sum().reset_index()        
    return data

df = pd.read_csv("D:\IBM_Hack_2020\Project_IBM\model\Warehouse_train.csv")
weekly_data = getWeekYear(df)
weekly_s_collection = {}
model_df = {}
weekly_collection = {}
    
def generate_model_df():
    for i in range(1,11):
        weekly_collection[i] = weekly_data[weekly_data.item == i]    
        
    def get_diff(data):
        data['sales_diff'] = data.sales.diff()    
        data = data.dropna()      
        return data

    for i in range(1,11):
        weekly_s_collection[i] = get_diff(weekly_collection[i])


    def generate_supervised(data,item_no):
        supervised_df = data.copy()
        
        #create column for each lag
        for i in range(1,53):
            col = 'lag_' + str(i)
            supervised_df[col] = supervised_df['sales_diff'].shift(i)
        
        #drop null values. 2013 data will be dropped as they have a lag of null (nan)
        supervised_df = supervised_df.dropna().reset_index(drop=True)
        supervised_df.to_csv('D:\IBM_Hack_2020\Project_IBM\model\item_'+str(item_no)+'.csv', index=False)
        return supervised_df

    for i in range(1,11):
        model_df[i] = generate_supervised(weekly_s_collection[i],i)
        
        
#Below function is only for appending week
def appendWeek(item_no, date):
    weekly_data.loc[len(weekly_data)] = [item_no, date, 0]
    generate_model_df()
    
    
def makeModel():
    #train test split    
    def tts(data):
        data = data.drop(['item','sales','date'],axis=1)
        train, test = data[:-1].values, data[-1:].values
        
        return train, test
    
    #scale data
    def scale_data(train_set, test_set):
        #apply Min Max Scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train_set)
        
        # reshape training set
        train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
        train_set_scaled = scaler.transform(train_set)
        
        # reshape test set
        test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
        test_set_scaled = scaler.transform(test_set)
        
        X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
        X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()
        
        return X_train, y_train, X_test, y_test, scaler
    
    #unscaling predictions, X_test, scaler_object
    def undo_scaling(y_pred, x_test, scaler_obj, lstm=False):  
        #reshape y_pred
        y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)
        
        if not lstm:
            x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
        
        #rebuild test set for inverse transform
        pred_test_set = []
        for index in range(0,len(y_pred)):
            pred_test_set.append(np.concatenate([y_pred[index],x_test[index]],axis=1))
            
        #reshape pred_test_set
        pred_test_set = np.array(pred_test_set)
        pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
        
        #inverse transform
        pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)
        
        return pred_test_set_inverted
    
    #prediction function
    def predict_df(unscaled_predictions, original_df):
        #create dataframe that shows the predicted sales
        result_list = []
        sales_dates = list(original_df[-2:].date)
        act_sales = list(original_df[-2:].sales)
        
        for index in range(0,len(unscaled_predictions)):
            result_dict = {}
            result_dict['pred_value'] = int(unscaled_predictions[index][0] + act_sales[index])
            result_dict['date'] = sales_dates[index+1]
            result_list.append(result_dict)
            
        df_result = pd.DataFrame(result_list) 
        return df_result
    
    #Plotting the results in a graph
    def plot_results(results, original_df, model_name):
        #original_df = original_df[0:-2]
        fig, ax = plt.subplots(figsize=(15,5))
        sns.lineplot(original_df.date, original_df.sales, data=original_df, ax=ax, 
                     label='Original', color='mediumblue')
        sns.lineplot(results.date, results.pred_value, data=results, ax=ax, 
                     label='Predicted', color='Red')
        
        ax.set(xlabel = "Date",
               ylabel = "Sales",
               title = f"{model_name} Sales Forecasting Prediction")
        
        ax.legend() 
        sns.despine()
    
    
    model_scores = {}
    
    def get_scores(unscaled_df, original_df, model_name):
        """Prints the root mean squared error, mean absolute error, and r2 scores
        for each model. Saves all results in a model_scores dictionary for
        comparison.
        Keyword arguments:
        -- unscaled_predictions: the model predictions that do not have min-max or
                                 other scaling applied
        -- original_df: the original monthly sales dataframe
        -- model_name: the name that will be used to store model scores
        """
        rmse = np.sqrt(mean_squared_error(original_df.sales[-2:], unscaled_df.pred_value))
        mae = mean_absolute_error(original_df.sales[-2:], unscaled_df.pred_value)
        r2 = r2_score(original_df.sales[-2:], unscaled_df.pred_value)
        model_scores[model_name] = [rmse, mae, r2]
    
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2 Score: {r2}")    
        
    def regressive_model(train_data, test_data, model, model_name):
        
        X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)
        mod = model
        mod.fit(X_train, y_train)
        predictions = mod.predict(X_test)
        
        # Undo scaling to compare predictions against original data
        original_df = weekly_s_collection[1]
        unscaled = undo_scaling(predictions, X_test, scaler_object)
        unscaled_df = predict_df(unscaled, original_df)
        
        #get_scores(unscaled_df, original_df, model_name)
        plot_results(unscaled_df, original_df, model_name)
        return unscaled_df, original_df, mod
    
    train, test = tts(model_df[1])    
    unscaled_df, original_df, model = regressive_model(train, test, xgb.XGBRegressor(n_estimators=100, learning_rate=0.2, objective='reg:squarederror'),'XGBoost')
    

generate_model_df()
appendWeek(1, '2018-01')
makeModel()
