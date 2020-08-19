# python 3.6
# Author: Scc_hy
# Create date: 2020-08-20
# Function: lgb xgb 包对 mnist识别

import pandas as pd 
import numpy as np
from sklearn.datasets import fetch_mldata
import xgboost as xgb
import lightgbm as lgb
from utils.utils_tools import clock, get_ministdata
import warnings
warnings.filterwarnings(action='ignore')

lgb_param = {
    'boosting': 'gbdt',
    'num_iterations': 145,
    'num_threads' : 8, 
    'verbosity': 0,
    'learning_rate': 0.2,
    'max_depth' : 10,
    'num_leaves' : 8,
    'subsample' : 0.75,
    'subsample_freq': 5,
    'colsample_bytree' : 1,
    'reg_alpha': 1.5,
    'reg_lambda': 0.75,
    'objective': 'multiclass',
    'num_class': 10,
    'metric': 'multi_logloss',
    'early_stopping': 25
    # 'device': 'gpu',
    # 'gpu_platform_id': 0,
    # 'gpu_device_id': 0
}

xgb_param = {
    'booster': 'gbtree',
    'tree_method':'gpu_hist',
    'num_rounds': 160,
    'nthread' : 8, 
    'silent' : 1,
    'learning_rate': 0.2,
    'max_depth' : 10,
    'num_leaves' : 8,
    'subsample' : 0.75,
    'colsample_bytree' : 1,
    'reg_alpha': 1.5,
    'reg_lambda': 0.75,
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
train xgb ...
[0]     train-merror:0.085857   test-merror:0.149357
[9]     train-merror:0.014714   test-merror:0.067696
lgb_xgb_train, take_time:34.88644s >> model: xgboost, acc: 93.23
train lgb ...
Training until validation scores don't improve for 25 rounds.
[20]    training's multi_logloss: 0.375235      valid_1's multi_logloss: 0.420369
[40]    training's multi_logloss: 0.188535      valid_1's multi_logloss: 0.256125
[60]    training's multi_logloss: 0.116575      valid_1's multi_logloss: 0.202916
[80]    training's multi_logloss: 0.0781876     valid_1's multi_logloss: 0.177008
[100]   training's multi_logloss: 0.0550978     valid_1's multi_logloss: 0.161271
[120]   training's multi_logloss: 0.0408921     valid_1's multi_logloss: 0.151961
[140]   training's multi_logloss: 0.0318648     valid_1's multi_logloss: 0.146336
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
Did not meet early stopping. Best iteration is:
[145]   training's multi_logloss: 0.030263      valid_1's multi_logloss: 0.145314
lgb_xgb_train, take_time:24.91101s >> model: lightgbm, acc: 95.65

"""
