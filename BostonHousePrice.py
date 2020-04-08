import tensorflow as tf  # 导入Tensorflow
import numpy as np  # 导入numpy
import matplotlib.pyplot as plt  # 导入matplotlib
import pandas as pd  # 导入pandas
from sklearn.utils import shuffle  # 导入sklearn的shuffle
from sklearn.preprocessing import scale  # 导入sklearn的scale

df = pd.read_csv("data/boston.csv",header=0)
print(df.describe())

df.head(3)  # 显示前3条数据
df.tail(3)  # 显示后3条数据

ds = df.values  # df.values以np.array形式返回数据集的值
print(ds.shape)  # 查看数据的形状
print(ds)  # 查看数据集的值

# x_data 为归一化后的前12列特征数据
x_data = ds[:,:12]

# y_data 为最后1列标签数据
y_data = ds[:,12]

for i in range(12):
    x_data[:,i]=(x_data[:,i]-x_data[:,i].min())/(x_data[:,i].max()-x_data[:,i].min())

print('x_data shape=', x_data.shape)
print('y_data shape=', y_data.shape)

train_num = 300  # 训练集的数目
valid_num = 100  # 验证集的数目
test_num = len(x_data) - train_num - valid_num  # 测试集的数目 = 506 - 训练集的数目 - 验证集的数目

# 训练集划分
x_train = x_data[:train_num]
y_train = y_data[:train_num]

# 验证集划分
x_valid = x_data[train_num:train_num+valid_num]
y_valid = y_data[train_num:train_num+valid_num]

# 测试集划分
x_test = x_data[train_num+valid_num:train_num+valid_num+test_num]
y_test = y_data[train_num+valid_num:train_num+valid_num+test_num]

x_train = tf.cast(scale(x_train), dtype=tf.float32)
x_valid = tf.cast(scale(x_valid), dtype=tf.float32)
x_test = tf.cast(scale(x_test), dtype=tf.float32)

def model(x, w, b):
    return tf.matmul(x, w) + b

W = tf.Variable(tf.random.normal([12, 1],mean=0.0, stddev=1.0, dtype=tf.float32))
B = tf.Variable(tf.zeros(1), dtype = tf.float32)
print(W)
print(B)

training_epochs = 50 # 迭代次数
learning_rate = 0.001 # 学习率
batch_size = 10 # 批量训练一次的样本数

# 采用均方差作为损失函数

def loss(x, y, w, b):
    err = model(x, w, b) - y # 计算模型预测值和标签值的差异
    squared_err = tf.square(err) # 求平方，得出方差
    return tf.reduce_mean(squared_err) # 求均值，得出均方差

# 计算样本数据[x,y]在参数[w,b]点上的梯度

def grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
    return tape.gradient(loss_, [w,b]) # 返回梯度向量

optimizer = tf.keras.optimizers.SGD(learning_rate) # 创建优化器，指定学习率

loss_list_train = [] # 用于保存训练集loss值的列表
loss_list_valid = [] # 用于保存验证集loss值的列表
total_step = int(train_num/batch_size)

for epoch in range(training_epochs):
    for step in range(total_step):
        xs = x_train[step*batch_size:(step+1)*batch_size,:]
        ys = y_train[step*batch_size:(step+1)*batch_size]

        grads = grad(xs, ys, W, B) # 计算梯度
        optimizer.apply_gradients(zip(grads, [W, B])) # 优化器根据梯度自动调整变量w和b

    loss_train = loss(x_train, y_train, W, B).numpy()  # 计算当前轮训练损失
    loss_valid = loss(x_valid, y_valid, W, B).numpy()  # 计算当前轮验证损失
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    print("epoch={:3d} ,train_loss={:.4f},valid_loss={:.4f}".format(epoch+1, loss_train, loss_valid))

plt.xlabel("Epochs")
plt.ylabel("loss")
plt.plot(loss_list_train,'blue',label="Train Loss")
plt.plot(loss_list_valid,'red',label="Valid Loss")
plt.legend(loc=1) # 通过参数loc指定图例位置
plt.show()

print("Test_loss: {:.4f}".format(loss(x_test, y_test, W, B).numpy()))

test_house_id = np.random.randint(0,test_num)
y = y_test[test_house_id]
y_pred = model(x_test,W,B)[test_house_id]
y_predit=tf.reshape(y_pred,()).numpy()
print("House id",test_house_id,"Actual value",y,"Predicted value ",y_predit)
