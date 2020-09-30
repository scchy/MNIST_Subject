# python 3.6
# Author: Scc_hy
# Create date: 2020-09-30
# Function: tf2.keras 训练预测mnist数据集


import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers, Model
from utils.utils_tools import clock, get_ministdata
import warnings
warnings.filterwarnings(action='ignore')
import time

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc0 = layers.Dense(256, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.fc1 = layers.Dense(32, activation='relu', kernel_regularizer = regularizers.l2(0.06))
        self.bn2 = layers.BatchNormalization()
        self.fc2 = layers.Dense(10, activation='sigmoid')

    def call(self, inputs_, training=None):
        x = self.fc0(inputs_)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.bn2(x)
        return self.fc2(x)


def build_nnmodel(input_shape=(None, 784)):
    nnmodel = MyModel()
    nnmodel.build(input_shape=input_shape)
    nnmodel.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )
    return nnmodel

if __name__ == '__main__':
    mnistdf = get_ministdata()
    te_index = mnistdf.sample(frac=0.8).index.tolist()
    mnist_te = mnistdf.loc[te_index, :]
    mnist_tr = mnistdf.loc[~mnistdf.index.isin(te_index), :]
    mnist_tr_x, mnist_tr_y = mnist_tr.iloc[:, :-1].values, tf.keras.utils.to_categorical(mnist_tr.iloc[:, -1].values)
    mnist_te_x, mnist_te_y = mnist_te.iloc[:, :-1].values, tf.keras.utils.to_categorical(mnist_te.iloc[:, -1].values)

    nnmodel = build_nnmodel()
    stop = tf.keras.callbacks.EarlyStopping(monitor = 'accuracy', min_delta=0.0001)
    file_path = r'D:\Python_data\My_python\Projects\MNIST_Subject\mnist_model\mnist_keras.hdf5'
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath = file_path,
        monitor = 'accuracy', 
        save_best_only = True,
        verbose = 1
    )
    st = time.perf_counter()
    history = nnmodel.fit(mnist_tr_x, mnist_tr_y
                        ,validation_data = (mnist_te_x, mnist_te_y), verbose = 1
                        ,epochs=10
                        ,callbacks = [stop, checkpointer])
    acc_final = round(max(history.history['accuracy']), 2)

    print(f"acc:{acc_final:.3f}")
    predict_ = np.argmax(nnmodel.predict(mnist_te_x), axis=1)
    te_y = np.argmax(mnist_te_y, axis=1)
    print(predict_)
    print(te_y)
    cost_seconds = time.perf_counter()  - st
    print(f'auc: {sum(predict_ == te_y)/te_y.shape[0]:.3f}, cost: {cost_seconds:.3f}')
