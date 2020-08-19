# python 3.6
# Author: Scc_hy
# Create date: 2020-08-20
# Function: 用sklearn 包对 mnist识别 多进程

from sklearn.ensemble import RandomForestClassifier
import sklearn as skl
from multiprocessing import Pool
from mnist_sklearn import get_ministdata, sklearn_clf # MNIST1_sklearn.py

if __name__ == '__main__':
    mnistdf = get_ministdata()
    te_index = mnistdf.sample(frac=0.8).index.tolist()
    mnist_te = mnistdf.loc[te_index, :]
    mnist_tr = mnistdf.loc[~mnistdf.index.isin(te_index), :]
    pool = Pool(processes=4) # 进程池
    # 用集成模型训练 & 预测
    ensemble_func_lst = [i for i in dir(skl.ensemble) if 'Classifier' in i and 'Voting' not in i]
    print('start train async model')
    res_list = pool.starmap_async(sklearn_clf, [ [eval(f'skl.ensemble.{clf_}') ,mnist_tr, mnist_te] for clf_ in ensemble_func_lst ]).get()
    print(res_list)

"""
start train async model
sklearn_clf, take_time:3.36981s >> model: ExtraTreesClassifier, acc: 92.32
sklearn_clf, take_time:2.18172s >> model: RandomForestClassifier, acc: 91.88
sklearn_clf, take_time:47.99032s >> model: AdaBoostClassifier, acc: 71.85
sklearn_clf, take_time:46.66177s >> model: BaggingClassifier, acc: 91.33
sklearn_clf, take_time:1510.23123s >> model: GradientBoostingClassifier, acc: 93.48
"""
