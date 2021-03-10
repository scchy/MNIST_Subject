# python 3.6
# Author: Scc_hy
# Create date: 2021-03-10
# Function: 用torch接口的神经网络，并训练预测mnist数据集
# Reference: https://www.bilibili.com/video/BV1jE41177A4?p=11
__doc__ = """
Train : 0.98
Test : 0.954
"""

import pandas as pd 
import numpy as np
from utils.utils_tools import clock, get_ministdata
import warnings
warnings.filterwarnings(action='ignore')
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from tqdm import tqdm
BATCH_SIZE=128

@clock
def get_TrainTest_df():
    mnistdf = get_ministdata()
    te_index = mnistdf.sample(frac=0.8).index.tolist()
    mnist_te = mnistdf.loc[te_index, :]
    mnist_tr = mnistdf.loc[~mnistdf.index.isin(te_index), :]
    x_tr, y_tr = mnist_tr.iloc[:, :-1].values, mnist_tr.iloc[:, -1].values
    x_te, y_te = mnist_te.iloc[:, :-1].values, mnist_te.iloc[:, -1].values
    return x_tr, y_tr, x_te, y_te


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 【batch_size, in_channels, height_1, width_1】
        # 28 * 28 * 1 => 26 * 26 * 32
        self.l1 = nn.Linear(784, 256)
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size = 3) 
        self.active_1 = F.relu
        # self.pool = nn.MaxPool2d(2, 2)
        self.d1 = nn.Linear(256, 32)
        self.active_2 = F.relu
        self.d2 = nn.Linear(32, 10)
        self.softmax = F.softmax
        self._compile_opt()
        

    def forward(self, x):
        x = self.active_1(self.l1(x))
        x = x.flatten(start_dim=1)
        x = self.active_2(self.d1(x))
        logits = self.d2(x)
        out = self.softmax(logits, dim=1)
        return out

    def model_test(self, lossfunc, x_te: 'np.array', y_te: 'np.array'):
        x_te, y_te = torch.tensor(x_te, dtype=float), torch.tensor(y_te, dtype=torch.long)
        out = self(x_te)
        loss_ = lossfunc(out, y_te)
        loss_out = loss_.detach().item()
        acc_out = (torch.argmax(out, 1).flatten() == y_te).type(torch.float).mean().item()
        return loss_out, acc_out

    def _compile_opt(self):
        self.optm = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss = nn.CrossEntropyLoss()
 
    def batch_backward(self, x_t, y_t):
        # forward + backward + loss
        out = self(x_t)
        loss_ = self.loss(out, y_t)
        self.optm.zero_grad()
        loss_.backward() 
        # update model parameters
        self.optm.step()
        return out, loss_
    
    def predict(self, x, batch_size, loss_flag = False, y_te = None):
        loss_total = 0.0
        idx_list = list(range(0, x.shape[0], batch_size))
        out_list = []
        loss_ = 0
        for i in tqdm(idx_list):
            x_t = torch.tensor( x[i:i+batch_size], dtype=float)
            out = self(x_t)
            if loss_flag:
                y_t = torch.tensor(y_te[i:i+batch_size], dtype=torch.long)
                loss_ = self.loss(out, y_t)
            tmp_predict = torch.argmax(out, 1).flatten()
            out_list.append(tmp_predict)
            loss_total += loss_.detach().item()
        # print(out_list[0])?
        return np.concatenate(out_list), loss_total




def bacth_data(x, y, batch_size=BATCH_SIZE):
    idx_list = list(range(0, x.shape[0], batch_size))
    for i in idx_list:
        x_t, y_t = x[i:i+batch_size], y[i:i+batch_size]
        # x_t = x_t.reshape((x_t.shape[0], 1, 28, 28))
        yield torch.tensor(x_t, dtype=float), torch.tensor(y_t, dtype=torch.long)


@clock
def mian():
    torch.set_default_tensor_type(torch.DoubleTensor)
    nn_model = MyModel()
    nn_model.to('cpu')
    print('Loading Data ...')
    x_tr, y_tr, x_te, y_te = get_TrainTest_df()
    print(x_tr.shape)
    print(np.unique(y_tr))

    for epoch in range(80):
        train_runing_loss = 0.0
        total_count = 0.0 
        true_count = 0.0
        bd = bacth_data(x_tr, y_tr)
        loop = True
        while loop:
            try:
                x_t, y_t = next(bd)
                x_t.to('cpu')
                y_t.to('cpu')
                # x_t = Variable(x_t)
                # y_t = Variable(y_t)
                out, loss_ = nn_model.batch_backward(x_t, y_t)
                train_runing_loss += loss_.detach().item()
                pred_out = torch.argmax(out, 1).flatten()
                true_count += (pred_out == y_t).type(torch.float).sum().item()
                total_count += x_t.shape[0]
            except Exception as e:
                print(e)
                loop = False 
        # print(pred_out)
        train_acc = true_count / total_count
        print(f'Epoch: [ {epoch} ] | Train Loss {train_runing_loss:.4f} | Train Acc: {train_acc:.2f}')

        if epoch % 10 == 0:
            test_pred, loss_total = nn_model.predict(x_te, BATCH_SIZE, loss_flag=True, y_te=y_te)
            acc = (y_te == test_pred).mean()
            print(f'>>> | Test Loss {loss_total:.4f} | Train Acc: {acc:.3f}')

    test_pred, loss_total = nn_model.predict(x_te, BATCH_SIZE, loss_flag=True, y_te=y_te)
    acc = (y_te == test_pred).mean()
    print(f'>>> | Test Loss {loss_total:.4f} | Train Acc: {acc:.3f}')
    return nn_model


if __name__ == '__main__':
    mian()
