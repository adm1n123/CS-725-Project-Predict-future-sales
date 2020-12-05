import sys
import os
import numpy as np
import pandas as pd
import datetime

np.random.seed(42)

NUM_FEATS = 3


class Net(object):

    def __init__(self, num_layers, num_units):

        self.num_layers = num_layers  # number of hidden layers except output & input layer.
        self.num_units = num_units  # number of neurons in each hidden layer.

        self.layer_inp = [None] * (self.num_layers + 1)
        self.layer_out = [None] * (self.num_layers + 1)
        self.layer_weight = [None] * (self.num_layers + 1)
        self.layer_bias = [None] * (self.num_layers + 1)

        # 0th layer is output layer with single neuron for each layer input to neuron is (batch_size x number of neurons)
        # for 0th layer neuron first column will be input and output since there is only one neuron.
        # top layer has only one neuron
        self.layer_weight[0] = np.random.uniform(-1, 1, self.num_units)
        self.layer_bias[0] = np.random.uniform(-1, 1, 1)
        for layer in range(1, self.num_layers):
            self.layer_weight[layer] = np.random.uniform(-1, 1, (self.num_units, self.num_units))
            self.layer_bias[layer] = np.random.uniform(-1, 1, self.num_units)

        # lowest layer near input
        self.layer_weight[self.num_layers] = np.random.uniform(-1, 1, (NUM_FEATS, self.num_units))
        self.layer_bias[self.num_layers] = np.random.uniform(-1, 1, self.num_units)

        # gradient
        self.delta_layer_inp = [None] * (self.num_layers + 1)
        self.delta_layer_out = [None] * (self.num_layers + 1)
        self.delta_layer_weight = [None] * (self.num_layers + 1)
        self.delta_layer_bias = [None] * (self.num_layers + 1)

    def __call__(self, X):

        #  process last layer i.e. bottom layer near input
        self.layer_inp[self.num_layers] = np.dot(X, self.layer_weight[self.num_layers]) + self.layer_bias[
            self.num_layers]
        self.layer_out[self.num_layers] = self.relu(self.layer_inp[self.num_layers])

        for layer in range(self.num_layers - 1, 0, -1):
            self.layer_inp[layer] = np.dot(self.layer_out[layer + 1], self.layer_weight[layer]) + self.layer_bias[layer]
            self.layer_out[layer] = self.relu(self.layer_inp[layer])

        # solve for 0th (output layer) it has single neuron
        self.layer_inp[0] = np.dot(self.layer_out[1], self.layer_weight[0]) + self.layer_bias[0]
        self.layer_out[0] = np.copy(self.layer_inp[0])

        return np.copy(self.layer_out[0])

    def backward(self, X, y, lamda):

        # forward pass is run already just take y_hat
        y_hat = np.copy(self.layer_out[0])
        m = y.shape[0]
        # finding delta for output layer with single neuron f(x) = x.
        self.delta_layer_out[0] = (2 / m) * (y_hat - y)  # for single example loss = (y_hat-y)^2 / 1
        self.delta_layer_inp[0] = np.copy(self.delta_layer_out[0])
        self.delta_layer_weight[0] = np.dot(self.delta_layer_inp[0], self.layer_out[1]) / m
        self.delta_layer_weight[0] += lamda * (2 / m) * self.layer_weight[0]
        self.delta_layer_bias[0] = np.sum(self.delta_layer_inp[0]) / m
        self.delta_layer_bias[0] += lamda * (2 / m) * self.layer_bias[0]

        if self.num_layers == 1:
            # solve for single layer
            # outer multiplication because col_vector x row_vector = matrix (it should be delta matrix input_X_units)
            # instead or using outer we can simply convert vector into col and row matrix using reshape and use np.dot()
            self.delta_layer_out[1] = np.outer(self.delta_layer_inp[0], self.layer_weight[0])
            self.delta_layer_inp[1] = self.delta_layer_out[1] * self.delta_relu(self.layer_inp[1])
            self.delta_layer_weight[1] = np.dot(X.transpose(), self.delta_layer_inp[1]) / m
            self.delta_layer_weight[1] += lamda * (2 / m) * self.layer_weight[1]
            self.delta_layer_bias[1] = np.dot(np.ones(m), self.delta_layer_inp[1]) / m
            self.delta_layer_bias[1] += lamda * (2 / m) * self.layer_bias[1]

        else:
            # solve for multi-layer
            # solve for 1st hidden layer
            self.delta_layer_out[1] = np.outer(self.delta_layer_inp[0], self.layer_weight[0])
            self.delta_layer_inp[1] = self.delta_layer_out[1] * self.delta_relu(self.layer_inp[1])
            self.delta_layer_weight[1] = np.dot(self.layer_out[2].transpose(), self.delta_layer_inp[1]) / m
            self.delta_layer_weight[1] += lamda * (2 / m) * self.layer_weight[1]
            self.delta_layer_bias[1] = np.dot(np.ones(m), self.delta_layer_inp[1]) / m
            self.delta_layer_bias[1] += lamda * (2 / m) * self.layer_bias[1]

            # loop for middle hidden layers
            for layer in range(2, self.num_layers):
                self.delta_layer_out[layer] = np.dot(self.delta_layer_inp[layer - 1], self.layer_weight[layer - 1].transpose())
                self.delta_layer_inp[layer] = self.delta_layer_out[layer] * self.delta_relu(self.layer_inp[layer])
                self.delta_layer_weight[layer] = np.dot(self.layer_out[layer + 1].transpose(), self.delta_layer_inp[layer]) / m
                self.delta_layer_weight[layer] += lamda * (2 / m) * self.layer_weight[layer]
                self.delta_layer_bias[layer] = np.dot(np.ones(m), self.delta_layer_inp[layer]) / m
                self.delta_layer_bias[layer] += lamda * (2 / m) * self.layer_bias[layer]

            # solve for last hidden layer
            layer = self.num_layers
            self.delta_layer_out[layer] = np.dot(self.delta_layer_inp[layer - 1], self.layer_weight[layer - 1].transpose())
            self.delta_layer_inp[layer] = self.delta_layer_out[layer] * self.delta_relu(self.layer_inp[layer])
            self.delta_layer_weight[layer] = np.dot(X.transpose(), self.delta_layer_inp[layer]) / m
            self.delta_layer_weight[layer] += lamda * (2 / m) * self.layer_weight[layer]
            self.delta_layer_bias[layer] = np.dot(np.ones(m), self.delta_layer_inp[layer]) / m
            self.delta_layer_bias[layer] += lamda * (2 / m) * self.layer_bias[layer]

        return self.copy_list_of_np(self.delta_layer_weight), self.copy_list_of_np(self.delta_layer_bias)

    def relu(self, matrix):
        return np.maximum(matrix, 0)

    def delta_relu(self, matrix):
        temp = np.copy(matrix)
        temp[temp <= 0] = 0
        temp[temp > 0] = 1
        return temp

    def copy_list_of_np(self, lst):
        py_list = [None] * len(lst)
        for idx in range(len(lst)):
            py_list[idx] = np.copy(lst[idx])

        return py_list


class Optimizer(object):

    def __init__(self, learning_rate):

        #   add different criteria to object and use those during training.
        self.learning_rate = learning_rate

    def step(self, weights, biases, delta_weights, delta_biases):

        updated_weights = [None] * len(weights)
        updated_biases = [None] * len(biases)

        for layer in range(len(weights)):
            updated_weights[layer] = np.subtract(weights[layer], self.learning_rate * delta_weights[layer])
            updated_biases[layer] = np.subtract(biases[layer], self.learning_rate * delta_biases[layer])

        return updated_weights, updated_biases


def loss_mse(y, y_hat):

    diff = np.subtract(y_hat, y)
    sqr_error = np.square(diff)
    mse = sqr_error.mean()
    return mse


def loss_regularization(weights, biases):

    l2_norm = 0
    for layer in range(len(weights)):
        l2_norm += np.sum(np.square(weights[layer])) + np.sum(np.square(biases[layer]))

    return l2_norm


def loss_fn(y, y_hat, weights, biases, lamda):

    return loss_mse(y, y_hat) + lamda * loss_regularization(weights, biases)


def rmse(y, y_hat):

    return np.sqrt(loss_mse(y, y_hat))


def train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        test_input, test_target
):
    print("starting training")
    for epoch in range(max_epochs):
        for batch in range(train_input.shape[0] // batch_size):
            train_sample_input = train_input[batch * batch_size: batch * batch_size + batch_size, :]
            train_sample_target = train_target[batch * batch_size: batch * batch_size + batch_size]

            # forward pass
            train_sample_y_hat = net(train_sample_input)

            # backward pass
            delta_weights, delta_biases = net.backward(train_sample_input, train_sample_target, lamda)

            net.layer_weight, net.layer_bias = optimizer.step(net.layer_weight, net.layer_bias, delta_weights, delta_biases)
            # rms_error = rmse(train_sample_y_hat, train_sample_target)
            # print(rms_error)
        print("Epoch "+str(epoch+1)+" Completed")
        test_rmse = rmse(net(test_input), test_target)
        print(test_rmse)


def print_rmse(name, net, input_data, target_data):
    y_hat = net(input_data)
    #print(y_hat.tolist())
    print(name, " rmse: ", rmse(target_data, y_hat))


def neural_network(train_input, train_target, test_input, test_target):
    max_epochs = 2
    batch_size = 25             
    learning_rate = 0.01
    num_layers = 1
    num_units = 100
    lamda = 5  # Regularization Parameter

    net = Net(num_layers, num_units)
    optimizer = Optimizer(learning_rate)
    
    train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        test_input, test_target
    )

    test_input = np.array(pd.read_csv("data/test.csv"))
    test_input[:, 0] = 34
    get_test_data_predictions(net, test_input)


def get_test_data_predictions(net, inputs):

    ans=net(inputs)
    #ans=np.ceil(ans)
    ans=np.round(ans,1)
    
    Id=[]
    for i in range(0,len(ans),1):
        Id.append(i)
        #print(Id)
    res=pd.DataFrame(np.column_stack([Id, ans]), columns=['Id', 'item_cnt_month'])
        #print(res)
        #df = pd.DataFrame(ans ,columns = ['ID', 'Predicted']) 
    res.to_csv("data/nn_predict.csv",index=False)

    res["Id"]=pd.to_numeric(res["Id"],downcast='integer')
    res.to_csv("data/nn_predict.csv",index=False)


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
        self.train_input_df = None
        self.train_input_np = None
        self.train_target_df = None
        self.train_target_np = None

    def load_data(self):
        self.data = pd.read_csv('data/sales_train.csv')

        self.train_input_df = self.data.copy()
        self.train_input_df.drop(columns=['item_price'])
        self.train_input_np = self.train_input_df.to_numpy()

        self.train_target_df = self.data['item_price'].copy()
        self.train_target_np = self.train_target_df.to_numpy()

    def get_train_df(self):
        return self.train_input_df.copy()

    def get_train_np(self):
        return np.copy(self.train_input_np)

    def get_data(self, train_prct, dev_prct):
        data = self.get_train_df()
        data.drop(columns='date', inplace=True)
        data.drop(columns='item_price',inplace=True)
        # data = data.to_numpy()
        monthly_sales = data.groupby(["date_block_num", "shop_id", "item_id"])[
            "date_block_num", "shop_id", "item_id", "item_cnt_day"].agg(
            {"date_block_num": "first", "shop_id": "first", "item_id": "first", "item_cnt_day": "sum"})

        data = monthly_sales.to_numpy()

        # splitting into train, dev and dev
        total = train_prct + dev_prct
        train, dev = train_prct / total, dev_prct / total
        rows, cols = data.shape
        train_cnt = int(rows * train)
        dev_cnt = int(rows * dev)

        train_data = data[0:train_cnt, :]
        dev_data = data[train_cnt: train_cnt+dev_cnt, :]

        train_target = train_data[:, -1]
        train_input = np.delete(train_data, -1, 1)
  
        #train_input[:, 0] = train_input[:, 0] % 12

        dev_target = dev_data[:, -1]
        dev_input = np.delete(dev_data, -1, 1)
      
        # dev_input[:, 0] = dev_input[:, 0] % 12
        # dev_input[:, -1] = 0

        return train_input, train_target, dev_input, dev_target


def main():
    data = Data()
    data.load_data()
    train_input, train_target, dev_input, dev_target = data.get_data(train_prct=99.9999, dev_prct=0.0001)
    # a = train_target[:].argsort()

    neural_network(train_input, train_target, dev_input, dev_target)


if __name__ == '__main__':
    main()

