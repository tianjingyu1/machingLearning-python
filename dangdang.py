import json
import csv
import numpy as npy
import pandas as pda
#读取.json文件
dic=[]
f = open("D:/machineLearning-python/getdata.json", 'r',encoding='utf-8')#这里为.json文件路径
for line in f.readlines():
    dic.append(json.loads(line))
#对爬取到的数据作处理，将价格和评论数由字符串处理为数字
tmp=''
name,price,comnum,link=[],[],[],[]
for i in range(0,1260):
    dic[i]['price']=tmp + dic[i]['price'][1:]
    dic[i]['comnum']=dic[i]['comnum'][:-3]+tmp
    price.append(float(dic[i]['price']))
    comnum.append(int(dic[i]['comnum']))
    name.append(dic[i]['name'])
    link.append(dic[i]['link'])
data = npy.array([name,price,comnum,link]).T
print (data)

#要存储.csv文件的路径
csvFile = open('D:/machineLearning-python/dangdang.csv','w')
writer = csv.writer(csvFile)
writer.writerow(['name', 'price', 'comnum','link'])
for i in range(0,1260):
    writer.writerow(data[i])
csvFile.close()

#读取.csv文件数据
#coding: unicode_escape
data = pda.read_csv("D:/machineLearning-python/dangdang.csv",encoding='gbk')
#发现缺失值，将评论数为0的值转为None
data["comnum"][(data["comnum"]==0)]=None
#均值填充处理
#data.fillna(value=data["comnum"].mean(),inplace=True)
#删除处理,data1为缺失值处理后的数据
data1=data.dropna(axis=0,subset=["comnum"])

import matplotlib.pyplot as plt
#画散点图（横轴：价格，纵轴：评论数）
#设置图框大小
fig = plt.figure(figsize=(10,6))
plt.plot(data1['price'],data1['comnum'],"o")
#展示x，y轴标签
plt.xlabel('price')
plt.ylabel('comnum')
plt.show()

fig1 = plt.figure(figsize=(10,6))
#初始化两个子图，分布为一行两列
ax1 = fig1.add_subplot(1,2,1)
ax2 = fig1.add_subplot(1,2,2)
#绘制箱型图
ax1.boxplot(data1['price'].values)
ax1.set_xlabel('price')
ax2.boxplot(data1['comnum'].values)
ax2.set_xlabel('comnum')
#设置x，y轴取值范围
ax1.set_ylim(0,150)
ax2.set_ylim(0,1000)
plt.show()

#删除价格￥120以上，评论数700以上的数据
data2=data[data['price']<120]
data3=data2[data2['comnum']<700]
#data3为异常值处理后的数据
fig2 = plt.figure(figsize=(10,6))
plt.plot(data3['price'],data3['comnum'],"o")
plt.xlabel('price')
plt.ylabel('comnum')
plt.show()

#转换数据格式
tmp=npy.array([data3['price'],data3['comnum']]).T
#调用python关于机器学习sklearn库中的KMeans
from sklearn.cluster import KMeans
#设置分为3类，并训练数据
kms=KMeans(n_clusters=3)
y=kms.fit_predict(tmp)
#将分类结果以散点图形式展示
fig3 = plt.figure(figsize=(10,6))
plt.xlabel('price')
plt.ylabel('comnum')
for i in range(0,len(y)):
    if(y[i]==0):
        plt.plot(tmp[i,0],tmp[i,1],"*r")
    elif(y[i]==1):
        plt.plot(tmp[i,0],tmp[i,1],"sy")
    elif(y[i]==2):
        plt.plot(tmp[i,0],tmp[i,1],"pb")
plt.show()

