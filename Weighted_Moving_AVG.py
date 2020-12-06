import datetime

import pandas as pd
import numpy as np


# #######################################################  MODEL Weighted avg. ####################################################################
# Kaggle Score: 1.05798     Rank: 5283
def get_block_average(block, train_input, test):
    prediction = train_input[train_input.date_block_num == block].groupby(['shop_id', 'item_id']).item_cnt_day.sum()
    prediction = prediction.clip(0, 20)
    prediction = pd.merge(test, prediction, how='left', on=['shop_id', 'item_id']).fillna(0.)
    prediction = prediction.rename(index=str, columns={'item_cnt_day': 'item_cnt_month'})
    # print(prediction)
    return prediction['item_cnt_month']


def get_weighted_avg(df, max_block):
    remaining_blocks = max_block - 2 + 1

    remaining_blocks_avg = df[0]  # take first col and iterate for rest.
    for block in range(1, remaining_blocks, 1):
        remaining_blocks_avg += df[block]
    remaining_blocks_avg /= remaining_blocks  # take avg of remaining cols

    return df[max_block] * .6 + df[max_block - 1] * .2 + remaining_blocks_avg * .2  # return weighted avg.


def predict_weighted_avg(train_input, test):
    max_block = train_input['date_block_num'].max()
    print("max number of blocks: {max_no}".format(max_no=max_block))

    df = pd.DataFrame()
    for block in range(max_block+1):
        df[block] = get_block_average(block, train_input, test)

    df[max_block+1] = get_weighted_avg(df, max_block)

    df['ID'] = [i for i in range(len(test.index))]
    df = df[['ID', 34]]
    df = df.rename(index=str, columns={max_block+1: 'item_cnt_month'})
    df.clip(0, 20)

    print(df)

    df.to_csv("data/weighted_avg.csv", index=False)
    return None


# #######################################################  MODEL Predict Last Month #############################################################
# Kaggle Score: 1.16777      Rank: 6146
def predict_last_month(train_input, test):
    prediction = train_input[train_input.date_block_num == 33].groupby(['shop_id', 'item_id']).item_cnt_day.sum()
    prediction = prediction.clip(0, 20)
    prediction = pd.merge(test, prediction, how='left', on=['shop_id', 'item_id']).fillna(0.)
    prediction = prediction.rename(index=str, columns={'item_cnt_day': 'item_cnt_month'})
    prediction.to_csv('data/submit.csv', columns=['ID', 'item_cnt_month'], index=False)
    return None


# ############################################################# Data Input #####################################################################


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

    # predict_last_month(train_input, test)

    predict_weighted_avg(train_input, test)


if __name__ == '__main__':
    main()
