#import os

import xgboost as xgb
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from csv import writer
from datetime import datetime

def callingFunction(item_no, date1, sales1, date2):
    def getWeekYear(data):
        data.date = data.date.apply(lambda x: str(str(x)[0:4])+"-"+str(dt.date(
            int(str(x)[0:4]), int(str(x)[5:7]), int(str(x)[8:10])).isocalendar()[1]).zfill(2))
        data = data.groupby(['item', 'date'])['sales'].sum().reset_index()
        return data

    weekly_s_collection = {}
    model_df = {}
    weekly_collection = {}

    def generate_model_df():
        for i in range(1, 11):
            weekly_collection[i] = weekly_data[weekly_data.item == i]

        def get_diff(data):
            data['sales_diff'] = data.sales.diff()
            data = data.dropna()
            return data

        for i in range(1, 11):
            weekly_s_collection[i] = get_diff(weekly_collection[i])

        def generate_supervised(data, item_no):
            supervised_df = data.copy()

            # create column for each lag
            for i in range(1, 53):
                col = 'lag_' + str(i)
                supervised_df[col] = supervised_df['sales_diff'].shift(i)

            # drop null values. 2013 data will be dropped as they have a lag of null (nan)
            supervised_df = supervised_df.dropna().reset_index(drop=True)
            #supervised_df.to_csv('D:\IBM_Hack_2020\Project_IBM\model\item_'+str(item_no)+'.csv', index=False)
            return supervised_df

        for i in range(1, 11):
            model_df[i] = generate_supervised(weekly_s_collection[i], i)

    # Below function is only for appending week

    def appendWeek(item_no, date, sales):
        weekly_data.loc[len(weekly_data)] = [item_no, date, sales]
        generate_model_df()

    def makeModel():
        # train test split
        def tts(data):
            data = data.drop(['item', 'sales', 'date'], axis=1)
            train, test = data[:-1].values, data[-1:].values

            return train, test

        # scale data
        def scale_data(train_set, test_set):
            # apply Min Max Scaler
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler = scaler.fit(train_set)

            # reshape training set
            train_set = train_set.reshape(
                train_set.shape[0], train_set.shape[1])
            train_set_scaled = scaler.transform(train_set)

            # reshape test set
            test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
            test_set_scaled = scaler.transform(test_set)

            X_train, y_train = train_set_scaled[:,
                                                1:], train_set_scaled[:, 0:1].ravel()
            X_test, y_test = test_set_scaled[:,
                                             1:], test_set_scaled[:, 0:1].ravel()

            return X_train, y_train, X_test, y_test, scaler

        # unscaling predictions, X_test, scaler_object
        def undo_scaling(y_pred, x_test, scaler_obj, lstm=False):
            # reshape y_pred
            y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)

            if not lstm:
                x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

            # rebuild test set for inverse transform
            pred_test_set = []
            for index in range(0, len(y_pred)):
                pred_test_set.append(np.concatenate(
                    [y_pred[index], x_test[index]], axis=1))

            # reshape pred_test_set
            pred_test_set = np.array(pred_test_set)
            pred_test_set = pred_test_set.reshape(
                pred_test_set.shape[0], pred_test_set.shape[2])

            # inverse transform
            pred_test_set_inverted = scaler_obj.inverse_transform(
                pred_test_set)

            return pred_test_set_inverted

        # prediction function
        def predict_df(unscaled_predictions, original_df):
            # create dataframe that shows the predicted sales
            result_list = []
            sales_dates = list(original_df[-2:].date)
            act_sales = list(original_df[-2:].sales)

            for index in range(0, len(unscaled_predictions)):
                result_dict = {}
                result_dict['pred_value'] = int(
                    unscaled_predictions[index][0] + act_sales[index])
                result_dict['date'] = sales_dates[index+1]
                result_list.append(result_dict)

            df_result = pd.DataFrame(result_list)
            return df_result

        def regressive_model(train_data, test_data, model, model_name):

            X_train, y_train, X_test, y_test, scaler_object = scale_data(
                train_data, test_data)
            mod = model
            mod.fit(X_train, y_train)
            predictions = mod.predict(X_test)

            # Undo scaling to compare predictions against original data
            original_df = weekly_s_collection[item_no]
            unscaled = undo_scaling(predictions, X_test, scaler_object)
            unscaled_df = predict_df(unscaled, original_df)
            return unscaled_df, original_df, mod

        train, test = tts(model_df[item_no])
        unscaled_df, original_df, model = regressive_model(train, test, xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.2, objective='reg:squarederror'), 'XGBoost')
        return unscaled_df

    #write the new actual sales to csv
    def update_csv():
        with open("D:\IBM_Hack_2020/Warehouse_train_copy.csv", 'a+', newline='') as write_obj :
            csv_writer = writer(write_obj)
            csv_writer.writerow([datetime.strptime(date1, '%Y-%m-%d').date(), 0, item_no, sales1])
    
    
    update_csv()
    ans_date = date2
    df = pd.read_csv("D:\IBM_Hack_2020/Warehouse_train_copy.csv")
    date1 = str(str(date1)[0:4])+"-"+str(dt.date(int(str(date1)[0:4]), int(str(date1)[5:7]), int(str(date1)[8:10])).isocalendar()[1]).zfill(2)
    date2 = str(str(date2)[0:4])+"-"+str(dt.date(int(str(date2)[0:4]), int(str(date2)[5:7]), int(str(date2)[8:10])).isocalendar()[1]).zfill(2)

    weekly_data = getWeekYear(df)
    generate_model_df()
    #appendWeek(item_no, date1, sales1)
    appendWeek(item_no, date2, 0)
    
    unscaled_df = makeModel()

    dict1 = {
        "item_no": item_no,
        "date": ans_date,
        "sales": int(unscaled_df.iloc[0]['pred_value'])
    }

    return dict1
