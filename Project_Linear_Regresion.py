#!/usr/bin/env python
# coding: utf-8

# # Linear Regression




import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)


class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        pass
    
        
    def __call__(self,features, is_train=False):
        minmax_data=features.describe().loc[['min','max']]
        for i in features.columns:
             features[i]=(features[i]-minmax_data[i]['min'])/(minmax_data[i]['max']-minmax_data[i]['min'])
        return features


def get_features(csv_path,is_train=False,scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''
    data=pd.read_csv(csv_path)
    try:
        data=data.drop(columns=[' shares'],axis=1)
    except:
        data=data
        
    data=scaler(data)
    
    #minmax_data=data.describe().loc[['min','max']]
    #for i in data.columns:
        #data[i]=(data[i]-minmax_data[i]['min'])/(minmax_data[i]['max']-minmax_data[i]['min'])
    #stdmean=data.describe().loc[['mean','std']]
    #for i in data.columns:
        #data[i]=(data[i]-stdmean[i]['mean'])/stdmean[i]['std']
    
    data['constant'] = 1
    feature_array=data.to_numpy()
    return feature_array
    

    
    
   
    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''

    '''
    help:
    useful links: 
        * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        * https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
    '''

    raise NotImplementedError

def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    data=pd.read_csv(csv_path)
    temp=data[' shares']
    target=temp.to_numpy()
    return target
    
    raise NotImplementedError
     

def analytical_solution(feature_matrix, targets, C=0.0):
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 4b
    return value: numpy array
    '''
     
    feature_matrix_transpose= feature_matrix.transpose()
    result_product=np.matmul(feature_matrix_transpose,feature_matrix)
    I=np.identity(feature_matrix.shape[1],dtype=float)
    m=feature_matrix.shape[0]
    result_addition=np.add(result_product,C*I*m)
    result_inverse=np.linalg.inv(result_addition)
    result_product_2=np.matmul(result_inverse,feature_matrix_transpose)
    final_parameters=np.matmul(result_product_2,targets)
    return final_parameters



    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape m x 1
    '''

    raise NotImplementedError 

def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''
    predictions=np.dot(feature_matrix,weights)
    return predictions



    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''

    raise NotImplementedError

def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''
    predictions=get_predictions(feature_matrix,weights)
    loss=np.subtract(predictions,targets)
    total_square_loss=np.square(loss)
    mse=total_square_loss.mean()
    return mse

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''
    raise NotImplementedError

def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''
    sum_w=np.sum(weights*weights, dtype = np.float32)
    return sum_w
    '''
    Arguments
    weights: numpy array of shape n x 1
    '''
    raise NotImplementedError

def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''
    return mse_loss(feature_matrix, weights, targets)+ C*l2_regularizer(weights)
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''

    raise NotImplementedError

def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''
    temp_predictions=np.dot(feature_matrix,weights)
    diff= np.subtract(temp_predictions,targets)
    m=feature_matrix.shape[0]
    feature_matrix_transpose=feature_matrix.transpose()
    #diff_transpose=diff.transpose()
    #gradient_transpose=np.dot(diff_transpose,feature_matrix)
    gradient=np.dot(feature_matrix_transpose,diff)
    gradient_2=gradient/m
    gradient_final=np.add(C*weights,gradient_2)
    return 2*gradient_final

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    raise NotImplementedError

def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''
    total_rows=feature_matrix.shape[0]
    random_indices = np.random.choice(total_rows, size=batch_size, replace=False)
    sampled_feature_matrix = feature_matrix[random_indices, :]
    sampled_targets= targets[random_indices]
    return (sampled_feature_matrix, sampled_targets)

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''    
    raise NotImplementedError
    
def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''
    initial_weights=np.random.rand(n)
    return initial_weights

    '''
    Arguments
    n: int
    '''
    raise NotImplementedError

def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''
    new_weights=weights-lr*gradients
    return new_weights
    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''    

    raise NotImplementedError

def early_stopping(val_error,weights, val_error_lowest, arg_n=None):
    # allowed to modify argument list as per your need
    # return True or False
    if(val_error<val_error_lowest):
        return True
    else:
        return False
    raise NotImplementedError
    

def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        #dev_feature_matrix,
                        #dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights
    a sample code is as follows -- 
    '''
    weights = initialize_weights(train_feature_matrix.shape[1])
    count_no_update=0
    patients_parameter=10000
    val_error_lowest=float('inf')
    #dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)

    print("step {} \t train loss: {}".format(0,train_loss))
    for step in range(1,max_steps+1):

        #sample a batch of features and gradients
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        
        #compute gradients
        gradients = compute_gradients(features, weights, targets, C)
        
        #update weights
        weights = update_weights(weights, gradients, lr)

       

    return weights

def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss





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






data_train=series_agg.copy()
train_target=data_train['item_cnt_day']
del data_train['item_cnt_day']
print(data_train)
train_features=data_train.to_numpy()
train_targets=train_target.to_numpy()
a_solution = analytical_solution(train_features,train_targets,C=1e-8)
print('evaluating analytical_solution...')
train_loss=do_evaluation(train_features, train_targets, a_solution)
print(train_loss)



ans=get_predictions(test_data, a_solution)





Id=[]
for i in range(0,len(ans),1):
    Id.append(i)
    #print(Id)
res=pd.DataFrame(np.column_stack([Id, ans]), columns=['Id', 'item_cnt_month'])
res["Id"]=pd.to_numeric(res["Id"],downcast='integer')
res.to_csv("predict_1_r.csv",index=False)

