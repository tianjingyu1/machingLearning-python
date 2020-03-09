# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:44:00 2020

@author: 82041
"""

from sklearn import svm               # svm函数需要的
import numpy as np                    # numpy科学计算库
from sklearn import model_selection   
import matplotlib.pyplot as plt       # 画图的库

def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

path ='D:\iris\iris_data.txt'  # 之前保存的文件路径
data = np.loadtxt(path,                          # 路径
                  dtype=float,                   # 数据类型
                  delimiter=',',                 # 数据以什么分割符号分割数据
                  converters={4: iris_type})     # 对某一列数据（第四列）进行某种类型的转换（）

X, y = np.split(data, (4,), axis=1)
x = X[:, 0:2]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3)

clf = svm.SVC(kernel='rbf',                      # 核函数
               gamma=0.1,
             decision_function_shape='ovo',     # one vs one 分类问题
              C=0.8)
clf.fit(x_train, y_train)                        # 训练
print(clf.score(x_train, y_train))               # 输出训练集的准确率 

y_train_hat=clf.predict(x_train)
y_train_1d=y_train.reshape((-1))
comp=zip(y_train_1d,y_train_hat)
print(list(comp))

print(clf.score(x_test,y_test))
y_test_hat=clf.predict(x_test)
y_test_1d=y_test.reshape((-1))
comp=zip(y_test_1d,y_test_hat)
print(list(comp))

plt.figure()
plt.subplot(121)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.reshape((-1)), edgecolors='k',s=50)
plt.subplot(122)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train_hat.reshape((-1)), edgecolors='k',s=50)
plt.show()

