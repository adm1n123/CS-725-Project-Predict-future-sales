import datetime
from time import localtime

import pandas as pd
import numpy as np

# #######################################################  ARIMA Model ####################################################################
# Kaggle Score: 1.17767
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings("ignore")


def get_next_sale_prediction(data):
    arima_model = ARIMA(data, order=(0, 0, 0))
    try:
        arima_fitted_model = arima_model.fit(disp=0)
        # print(arima_fitted_model.summary())
        # print(data)
        series = arima_fitted_model.predict(start=34, end=34, dynamic=True, typ='linear')
        return series.get(34)
    except:
        print("error")
        return 0


def get_block_average(block, train_input, test):
    prediction = train_input[train_input.date_block_num == block].groupby(['shop_id', 'item_id']).item_cnt_day.sum()
    prediction = prediction.clip(0, 20)
    prediction = pd.merge(test, prediction, how='left', on=['shop_id', 'item_id']).fillna(0.)
    prediction = prediction.rename(index=str, columns={'item_cnt_day': 'item_cnt_month'})
    # print(prediction)
    return prediction['item_cnt_month']


def predict_arima_model(train_input, test):
    max_block = train_input['date_block_num'].max() + 1
    print("max number of blocks: {max_no}".format(max_no=max_block))

    df = pd.DataFrame()
    for block in range(max_block):
        df[block] = get_block_average(block, train_input, test)

    # count = 0
    # df[max_block] = 0
    prediction = []
    for index, row in df.iterrows():
        prediction.append(get_next_sale_prediction(row[:-1]))
        # print(row[max_block])

    # print(prediction)
    sales_df = pd.DataFrame(data=prediction, columns=['item_cnt_month'])

    prediction_df = pd.DataFrame([i for i in range(len(test.index))], columns=['ID'])
    prediction_df['item_cnt_month'] = sales_df['item_cnt_month']

    prediction_df.clip(0, 20)

    prediction_df.to_csv("data/arima_model.csv", index=False)
    return None


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


@singleton
class Data:
    def __init__(self):
        self.data = None
        self.test = None

    def load_data(self):
        self.data = pd.read_csv('data/sales_train.csv')
        self.data.drop(labels=['date', 'item_price'], axis=1, inplace=True)

        self.test = pd.read_csv('data/test.csv')

    def get_train_df(self):
        return self.data.copy()

    def get_data(self):
        df = self.get_train_df()
        # monthly_sales_df.to_csv('data/grouped_sales_data.csv')
        # print(self.test)
        return df, self.test.copy()


def main():
    data = Data()
    data.load_data()
    train_input, test = data.get_data()

    predict_arima_model(train_input, test)


if __name__ == '__main__':
    main()