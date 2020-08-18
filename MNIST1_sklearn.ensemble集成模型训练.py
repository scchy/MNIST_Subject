# python 3.6
# Author: Scc_hy
# Create date: 2020-08-19
# Function: 用sklearn 包对 mnist识别

import pandas as pd 
import numpy as np
from sklearn.datasets import fetch_mldata
import warnings
warnings.filterwarnings(action='ignore')
import time
from functools import wraps


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
if __name__ == '__main__':
    mnistdf = get_ministdata()
    te_index = mnistdf.sample(frac=0.8).index.tolist()
    mnist_te = mnistdf.loc[te_index, :]
    mnist_tr = mnistdf.loc[~mnistdf.index.isin(te_index), :]
    # 用集成模型训练 & 预测
    ensemble_func_lst = [i for i in dir(skl.ensemble) if 'Classifier' in i and 'Voting' not in i]
    res_list = []
    for clf_ in ensemble_func_lst:
        res_list.append(sklearn_clf(eval(f'skl.ensemble.{clf_}'), mnist_tr, mnist_te))


"""
sklearn_clf, take_time:43.97123s >> model: AdaBoostClassifier, acc: 70.88
sklearn_clf, take_time:62.52457s >> model: BaggingClassifier, acc: 91.86
sklearn_clf, take_time:3.11310s >> model: ExtraTreesClassifier, acc: 92.34
sklearn_clf, take_time:1510.23123s >> model: GradientBoostingClassifier, acc: 93.48
sklearn_clf, take_time:3.57081s >> model: RandomForestClassifier, acc: 91.63

"""
