import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris #导入数据集iris
#%matplotlib inline

iris = load_iris() #载入数据集
#print(iris.data)  #打印输出显示

#print(iris.target)

iris.data.shape  # iris数据集150行4列的二维数组

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
dataset.hist() #数据直方图histograms
plt.show()
#print(dataset.describe())

dataset.plot(x='sepal-length', y='sepal-width', kind='scatter') #散点图，x轴表示sepal-length花萼长度，y轴表示sepal-width花萼宽度
plt.show()

dataset.plot(kind='kde') #KDE图，KDE图也被称作密度图(Kernel Density Estimate,核密度估计)
plt.show()

#kind='box'绘制箱图,包含子图且子图的行列布局layout为2*2，子图共用x轴、y轴刻度，标签为False
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

from pandas.plotting import radviz
radviz(dataset,'class')
plt.show()

from pandas.plotting import andrews_curves
andrews_curves(dataset,'class')
plt.show()

from pandas.plotting import parallel_coordinates
parallel_coordinates(dataset,'class')
plt.show()

from pandas.plotting import scatter_matrix
scatter_matrix(dataset, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()

#因素分析(FactorAnalysis, FA)
from sklearn import decomposition
pca = decomposition.FactorAnalysis(n_components=2)
X = pca.fit_transform(dataset.iloc[:,:-1].values)
pos = pd.DataFrame()
pos['X'] = X[:,0]
pos['Y'] = X[:,1]
pos['class'] = dataset['class']
ax = pos.loc[pos['class']=='Iris-virginica'].plot(kind='scatter', x='X', y='Y', color='blue', label='Iris-virginica')
ax = pos.loc[pos['class']=='Iris-setosa'].plot(kind='scatter', x='X', y='Y', color='green', label='Iris-setosa', ax=ax)
ax = pos.loc[pos['class']=='Iris-versicolor'].plot(kind='scatter', x='X', y='Y', color='red', label='Iris-versicolor', ax=ax)
plt.show()

#主成分分析（PCA）
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
X = pca.fit_transform(dataset.iloc[:,:-1].values)
pos = pd.DataFrame()
pos['X'] = X[:,0]
pos['Y'] = X[:,1]
pos['class'] = dataset['class']
ax = pos.loc[pos['class']=='Iris-virginica'].plot(kind='scatter', x='X', y='Y', color='blue', label='Iris-virginica')
ax = pos.loc[pos['class']=='Iris-setosa'].plot(kind='scatter', x='X', y='Y', color='green', label='Iris-setosa', ax=ax)
ax = pos.loc[pos['class']=='Iris-versicolor'].plot(kind='scatter', x='X', y='Y', color='red', label='Iris-versicolor', ax=ax)
plt.show()

#独立成分分析(ICA)
from sklearn import decomposition
pca = decomposition.FastICA(n_components=2)
X = pca.fit_transform(dataset.iloc[:,:-1].values)
pos = pd.DataFrame()
pos['X'] = X[:,0]
pos['Y'] = X[:,1]
pos['class'] = dataset['class']
ax = pos.loc[pos['class']=='Iris-virginica'].plot(kind='scatter', x='X', y='Y', color='blue', label='Iris-virginica')
ax = pos.loc[pos['class']=='Iris-setosa'].plot(kind='scatter', x='X', y='Y', color='green', label='Iris-setosa', ax=ax)
ax = pos.loc[pos['class']=='Iris-versicolor'].plot(kind='scatter', x='X', y='Y', color='red', label='Iris-versicolor', ax=ax)
plt.show()

#多维度量尺（Multi-dimensional scaling, MDS）
from sklearn import manifold
from sklearn.metrics import euclidean_distances
similarities = euclidean_distances(dataset.iloc[:,:-1].values)
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity='precomputed', n_jobs=1)
X = mds.fit(similarities).embedding_
pos = pd.DataFrame(X, columns=['X','Y'])
pos['class'] = dataset['class']
ax = pos.loc[pos['class']=='Iris-virginica'].plot(kind='scatter', x='X', y='Y', color='blue', label='Iris-virginica')
ax = pos.loc[pos['class']=='Iris-setosa'].plot(kind='scatter', x='X', y='Y', color='green', label='Iris-setosa', ax=ax)
ax = pos.loc[pos['class']=='Iris-versicolor'].plot(kind='scatter', x='X', y='Y', color='red', label='Iris-versicolor', ax=ax)
plt.show()
