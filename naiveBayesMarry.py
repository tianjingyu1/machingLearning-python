import numpy as np
from sklearn.naive_bayes import  GaussianNB
"""
X数据集数据，将对应数据转换为了数字
["帅","不好","矮","不上进"],["不帅","好","矮","上进"],["帅","好","矮","上进"],
["不帅","爆好","高","上进"],["帅","不好","矮","上进"],["帅","不好","矮","上进"],
["帅","好","高","不上进"],["不帅","好","中","上进"],["帅","爆好","中","上进"],
["不帅","不好","高","上进"],["帅","好","矮","不上进"],["帅","好","矮","不上进"]
Y数据集数据["不嫁","不嫁","嫁","嫁","不嫁","不嫁","嫁","嫁","嫁","嫁","不嫁",
"不嫁"]
"""
X = np.array([[1,0,0,0],[0,1,0,1],[1,1,0,1],[0,2,2,1],
              [1,0,0,1],[1,0,0,1],[1,1,2,0],[0,1,1,1],
              [1,2,1,1],[0,0,2,1],[1,1,0,0],[1,1,0,0]])
Y = np.array([0,0,1,1,0,0,1,1,1,1,0,0])

mnm = GaussianNB(priors=None)
mnm.fit(X,Y)
#["帅","不好","中","不上进"]的预测
rel = mnm.predict([[1,0,1,0]])
if rel==1:
    print("嫁")
else:
    print("不嫁")