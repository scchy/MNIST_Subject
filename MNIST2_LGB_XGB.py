# python 3.6
# Author: Scc_hy
# Create date: 2020-08-20
# Function: lgb xgb 包对 mnist识别

import pandas as pd 
import numpy as np
from sklearn.datasets import fetch_mldata
import xgboost as xgb
import lightgbm as lgb
import time
from functools import wraps
import warnings
warnings.filterwarnings(action='ignore')

def clock(func):
    @wraps(func)
    def clocked(*args, **kwargs):
        st = time.perf_counter()
        res = func(*args, **kwargs)
        take_time = time.perf_counter() - st
        fmt = '{func_name}, take_time:{take_time:.5f}s >> {res}'
        print(fmt.format(func_name=func.__name__, take_time=take_time, res=res))
        return res
    return clocked

def get_ministdata():
    data_home = r'D:\Python_data\My_python\Projects\MNIST_Subject\mnist_data'
    mnist = fetch_mldata('MNIST original', data_home=data_home)
    return pd.DataFrame(np.c_[mnist['data']/255, mnist['target']])


lgb_param = {
    'boosting': 'gbdt',
    'num_iterations': 150,
    'num_threads' : 8, 
    'verbosity': 0,
    'learning_rate': 0.1,
    'max_depth' : 8,
    'num_leaves' : 10,
    'subsample' : 0.8,
    'subsample_freq': 5,
    'colsample_bytree' : 1,
    'reg_alpha': 0.8,
    'reg_lambda': 0.3,
    'objective': 'multiclass',
    'num_class': 10,
    'metric': 'multi_logloss',
    'early_stopping': 25
}

xgb_param = {
    'booster': 'gbtree',
    'tree_method':'gpu_hist',
    'num_rounds': 150,
    'nthread' : 8, 
    'silent' : 1,
    'learning_rate': 0.1,
    'max_depth' : 8,
    'num_leaves' : 10,
    'subsample' : 0.8,
    'colsample_bytree' : 1,
    'reg_alpha': 0.8,
    'reg_lambda': 0.3,
    'objective': 'multi:softprob',
    'num_class': 10,
    'metric': 'mlogloss',
    'early_stopping': 25
}


@clock
def lgb_xgb_train(model, param, tr, te ):
    if model.__name__ == 'lightgbm':
        trdt = model.Dataset(data=tr.iloc[:, :-1].values, label=tr.iloc[:, -1].values)
        tedt = model.Dataset(data=te.iloc[:, :-1].values, label=te.iloc[:, -1].values)
        clf_model = model.train(param, trdt, valid_sets=[trdt, tedt] ,verbose_eval = 20)
        pred = np.argmax(clf_model.predict(te.iloc[:, :-1].values, num_iteration=clf_model.best_iteration ), axis=1)

    else:
        trdt = model.DMatrix(data=tr.iloc[:, :-1].values, label=tr.iloc[:, -1].values)
        tedt = model.DMatrix(data=te.iloc[:, :-1].values, label=te.iloc[:, -1].values)
        clf_model = model.train(param, trdt, evals=[(trdt, 'train'), (tedt, 'test')], verbose_eval = 20)
        pred = np.argmax(clf_model.predict(tedt, ntree_limit=-1), axis=1)
    
    y_te =  te.iloc[:, -1].values
    acc_ = sum(pred == y_te)/len(y_te) * 100
    return f'model: {model.__name__}, acc: {acc_:.2f}'

# help(xgb.train)

if __name__ == '__main__':
    mnistdf = get_ministdata()
    te_index = mnistdf.sample(frac=0.8).index.tolist()
    mnist_te = mnistdf.loc[te_index, :]
    mnist_tr = mnistdf.loc[~mnistdf.index.isin(te_index), :]
    print('train xgb ...')
    resxgb = lgb_xgb_train(xgb, xgb_param, mnist_tr, mnist_te)
    print('train lgb ...')
    reslgb = lgb_xgb_train(lgb, lgb_param, mnist_tr, mnist_te)

"""

"""
