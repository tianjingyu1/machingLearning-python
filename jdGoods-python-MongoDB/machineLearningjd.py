import pandas as pd
import matplotlib.pyplot as plt
import pymongo
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
from pylab import mpl
import pylab as pl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
from sklearn.cluster import KMeans


# 连接mongodb数据库
client = pymongo.MongoClient("localhost")
# 连接数据库
db = client["jd"]
# 数据表
jianpan = db["键盘"]
# 将mongodb中的数据读出
data = pd.DataFrame(list(jianpan.find()))
data = data.drop(['_id','img'],axis=1) # 去掉_id列
data.head()
# 保存为txt格式
data.to_csv('jd-jianpan.txt',encoding='utf-8')
# 读取txt数据
df = pd.read_csv('jd-jianpan.txt',low_memory=False,index_col=0)
#print(df)

# 商品信息词云
fr = open('jd-jianpan.txt', 'r+',encoding='UTF-8')
lines = fr.readlines()
goodName = []
price = []
for line in lines:
    items = line.strip().split(",")
    goodName.append(items[1])
goodName.pop(0)
str1 = ','.join(str(i) for i in goodName)
str2=str1.replace("'","")
str3=str2.replace("[","")
str4=str3.replace("\\t\\n","")
str5=str4.replace("\"","")
str6=str5.replace("[","")
#print(str6)
wordList_jieba1 = jieba.cut(str6, cut_all=False); # 使用jieba分词
data1 = ','.join(wordList_jieba1);
font = r'C:\Windows\Fonts\STXINWEI.TTF'; # 设置字体为华文新魏
wc1 = WordCloud(font_path=font).generate(data1); # 商品信息词云
plt.imshow(wc1, interpolation='bilinear')
plt.axis("off")
plt.show()

# 价格商店散点图
fig = plt.figure(figsize=(10,10))
price = [float(y) for x in data['price'] for y in x]
#print(price)
shop = [str(y) for x in data['shop'] for y in x ]
#print(shop)
plt.plot(price,shop,"o")
#展示x，y轴标签
plt.xlabel('price')
plt.ylabel('shop')
plt.show()

# 价格直方图
plt.hist(price, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
# 显示横轴标签
plt.xlabel("价格")
# 显示纵轴标签
plt.ylabel("频数/频率")
# 显示图标题
plt.title("价格频数分布直方图")
plt.show()

# 价格聚类分析
km = KMeans(n_clusters=4)
price1 = np.array(price)
price1 = price1.reshape(-1,1)
label = km.fit_predict(price1)
expenses = np.sum(km.cluster_centers_, axis=1)
# print(expenses)
ShopCluster = [[], [], [], []]
for i in range(len(shop)):
    ShopCluster[label[i]].append(shop[i])
for i in range(len(ShopCluster)):
    print("price:%.2f" % expenses[i])
    print(ShopCluster[i])

# 对价格进行非线性回归
