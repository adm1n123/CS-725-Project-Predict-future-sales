#!/usr/bin/env python
# coding: utf-8
# # Random Forest using sklearn




from sklearn.ensemble import RandomForestRegressor


from pandas import read_csv
from matplotlib import pyplot
import pandas as pd
import numpy as np
series = read_csv('sales_train.csv')
item_cat=read_csv('items.csv') 

combi_list=[]
for i in series['date_block_num'].unique():
    shop_ids=series[series['date_block_num']==i]['shop_id'].unique()
    item_ids=series[series['date_block_num']==i]['item_id'].unique()
    combi_array=np.array(np.meshgrid(shop_ids, item_ids,[i])).T.reshape(-1,3)
    combi_list.append(combi_array)
cols=['shop_id','item_id','date_block_num']
data=pd.DataFrame(np.vstack(combi_list),columns=cols)
series_agg=series.groupby(['shop_id','item_id','date_block_num']).agg({'item_cnt_day': 'sum','item_price':'mean'})
series_agg=pd.merge(data,series_agg,how='left',on=cols)
series_agg['item_cnt_day']=series_agg['item_cnt_day'].fillna(0)
series_agg['item_price']=series_agg['item_price'].fillna(0)
series_agg=pd.merge(series_agg,item_cat[['item_id','item_category_id']],how='left')
series_agg=series_agg.reset_index()

# Above idea to create combination of all shop_id,item_id for a particular date_block_num is taken from one of kaggle threads with modifications (https://www.kaggle.com/szhou42/predict-future-sales-top-11-solution)



for i in range(1,5):
    new_series=series_agg.copy()
    new_series['date_block_num']=new_series['date_block_num']+i
    new_series['item_cnt_prev'+str(i)]=new_series['item_cnt_day']
    series_agg=pd.merge(series_agg,new_series[['shop_id','item_id','date_block_num','item_cnt_prev'+str(i)]],how='left')
    series_agg['item_cnt_prev'+str(i)]=series_agg['item_cnt_prev'+str(i)].fillna(0)


data_train=series_agg.copy()
train_target=data_train['item_cnt_day']
del data_train['item_cnt_day']
print(data_train)
train_features=data_train.to_numpy()
train_targets=train_target.to_numpy()


price_agg=series.groupby(['shop_id','item_id']).agg({'item_price':'mean'})
price_agg=price_agg.reset_index()
test_data=read_csv("test.csv")
test_data['date_block_num']=34
test_data=pd.merge(test_data,price_agg[['shop_id','item_id','item_price']],how='left')
test_data['item_price']=test_data['item_price'].fillna(0)
test_data=pd.merge(test_data,item_cat[['item_id','item_category_id']],how='left')
for i in range(1,5):
    new_series=series_agg.copy()
    new_series['date_block_num']=new_series['date_block_num']+i
    new_series['item_cnt_prev'+str(i)]=new_series['item_cnt_day']
    test_data=pd.merge(test_data,new_series[['shop_id','item_id','date_block_num','item_cnt_prev'+str(i)]],how='left')
    test_data['item_cnt_prev'+str(i)]=series_agg['item_cnt_prev'+str(i)].fillna(0)





def rmse(y, y_hat):
    return np.sqrt(np.mean((y_hat-y)**2))





#without any parameter
model = RandomForestRegressor()
model.fit(train_features, train_targets)
res = model.predict(test_data)
res_train=model.predict(train_features)

res_train=model.predict(train_features)
train_error=rmse(res_train,train_targets)
print(train_error)
#Train error= 0.754
#Test error= 4.825





#with maxdepth = 15
model_2 = RandomForestRegressor(max_depth=15)
model_2.fit(train_features, train_targets)
res_2 = model_2.predict(test_data)
res_2_train=model_2.predict(train_features)
train_error_2=rmse(res_2_train,train_targets)
print(train_error_2)
#Train error = 1.525
#test error= 2.836





#with maxdepth = 10
model_3 = RandomForestRegressor(max_depth=10)
model_3.fit(train_features, train_targets)
res_3 = model_3.predict(test_data)
res_3_train=model_3.predict(train_features)
train_error_3=rmse(res_3_train,train_targets)
print(train_error_3)
#Train error = 1.789
#test error = 1.725





Id=[]
ans_new=res_3
for i in range(0,len(ans_new),1):
    Id.append(i)
    #print(Id)
res_new=pd.DataFrame(np.column_stack([Id, ans_new]), columns=['Id', 'item_cnt_month'])
    #print(res)
    #df = pd.DataFrame(ans ,columns = ['ID', 'Predicted']) 
res_new["Id"]=pd.to_numeric(res_new["Id"],downcast='integer')
res_new.to_csv("predict_4.csv",index=False)

