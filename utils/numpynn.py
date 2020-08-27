# python 3.6
# Author: Scc_hy
# Create date: 2020-08-25
# Function: 用numpy实现神经网络

import numpy as np

class npLayer():
    def __init__(self, n_input, n_out, activation=None, weights=None
                ,bias=None):
        self.weights = weights if weights is not None else np.random.randn(n_input, n_out) * np.sqrt(1 / n_out)
        self.bias = bias if bias is not None else np.random.randn(n_out) * 0.1
        self.activation = activation 
        self.last_activation = None 
        self.error = None 
        self.delta = None 

    def activate(self, x):
        # 前向传播
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self.apply_activation(r)
        return self.last_activation 
    
    def apply_activation(self, r):
        # 计算激活函数的输出
        if self.activation is None:
            return r
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1/(1 + np.exp(-r))
        return r

    def apply_activation_derivative(self, act_r):
        # 计算激活函数的导数
        # 无激活函数，导数为1
        if self.activation is None:
            return np.ones_like(act_r)
        elif self.activation == 'relu':
            return (act_r > 0) * 1
        elif self.activation == 'tanh':
            return 1 - act_r ** 2
        elif self.activation == 'sigmoid':
            return act_r * (1 - act_r)
        return act_r
    
    def __call__(self, x):
        return self.activate(x)


class NeuralNetwork():
    def __init__(self):
        self._layers = []
    
    def add_layer(self, layer):
        self._layers.append(layer)
    
    def feed_forward(self, x):
        # 前向传播
        for layer in self._layers:
            x = layer(x)
        return x 

    def backpropagation(self, x, y, learning_rate):
        # 反向传播算法实现
        ## 从后向前计算梯度 
        output = self.feed_forward(x) # 最后层输出
        layer_len = len(self._layers)
        for i in reversed(range(layer_len)):
            layer = self._layers[i] 
            # 如果是输出层
            if layer  == self._layers[-1]:
                delta_i = layer.apply_activation_derivative(output)
                layer.error = output - y
                layer.delta = layer.error * delta_i
            else:
                next_layer = self._layers[i + 1]
                delta_i = layer.apply_activation_derivative(layer.last_activation)
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * delta_i
                
        # 梯度下降
        for i in range(layer_len):
            layer = self._layers[i]
            o_i = np.atleast_2d(x if i == 0 else self._layers[i - 1].last_activation)
            layer.weights -= layer.delta * o_i.T * learning_rate


    def train(self, x_train, x_test, y_train, y_test, learning_rate, max_epochs, verbose_eval=5):
        # 网络训练
        # one-hot
        depth_ = len(np.unique(y_train))
        y_onehot = np.zeros((y_train.shape[0], depth_))
        # 索引赋值
        y_onehot[np.arange(y_train.shape[0]), y_train.astype(int)] = 1
        # 计算mse并更新参数
        mses, accs = [], []
        print('start training ..... ')
        for i in range(max_epochs + 1):
            for j in range(len(x_train)): # batch=1
                self.backpropagation(x_train[j], y_onehot[j], learning_rate)
            if i % verbose_eval == 0:
                # 打印mse
                mse = np.mean(np.square(y_onehot - self.predict(x_train)))
                mses.append(mse)
                print('\n','=='*40)
                print(f"Epoch: # {i}, MSE: {mse:.5f}")
                # 打印准确率
                acc = self.accuracy(y_test.flatten() , self.predict(x_test)) * 100
                accs.append(acc/100)
                print(f'Accuracy: {acc:.2f} %','\n')
        return mses, accs

    def predict(self, x):
        return self.feed_forward(x)

    def accuracy(self, y_true, y_pred):
        y_pred_max = np.argmax(y_pred, axis=1)
        corrects = sum(y_true == y_pred_max)
        return corrects/y_pred_max.shape[0]
