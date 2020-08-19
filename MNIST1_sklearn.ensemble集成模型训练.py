# python 3.6
# Author: Scc_hy
# Create date: 2020-08-19
# Function: 用sklearn 包对 mnist识别

import pandas as pd 
import numpy as np
from sklearn.datasets import fetch_mldata
import warnings
warnings.filterwarnings(action='ignore')
from utils.utils_tools import clock, get_ministdata


@clock
def sklearn_clf(clf_model_func, tr, te):
    clf_model = clf_model_func()
    clf_model.fit(tr.iloc[:, :-1].values, tr.iloc[:, -1].values)
    pred = clf_model.predict(te.iloc[:, :-1].values)
    y_te =  te.iloc[:, -1].values
    acc_ = sum(pred == y_te)/len(y_te) * 100
    return f'model: {clf_model_func.__name__}, acc: {acc_:.2f}'


from sklearn.ensemble import RandomForestClassifier
import sklearn as skl
from multiprocessing import Pool

# 一个文件产生错误，把函数定义放在另一个py文件中再引入 

if __name__ == '__main__':
    mnistdf = get_ministdata()
    te_index = mnistdf.sample(frac=0.8).index.tolist()
    mnist_te = mnistdf.loc[te_index, :]
    mnist_tr = mnistdf.loc[~mnistdf.index.isin(te_index), :]
    # pool = Pool(processes=4) # 进程池
    # 用集成模型训练 & 预测
    ensemble_func_lst = [i for i in dir(skl.ensemble) if 'Classifier' in i and 'Voting' not in i]
    # pool.starmap_async(sklearn_clf, [ [eval(f'skl.ensemble.{clf_}') ,mnist_tr, mnist_te] for clf_ in ensemble_func_lst ])
    res_list = []
    for clf_ in ensemble_func_lst:
        res_list.append(sklearn_clf(eval(f'skl.ensemble.{clf_}'), mnist_tr, mnist_te))
    print('sucesses')
