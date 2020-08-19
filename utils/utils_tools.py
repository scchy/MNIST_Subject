# python 3.6
# Author: Scc_hy
# Create date: 2020-08-20
# Function: mnist数据加载，和装饰器

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
