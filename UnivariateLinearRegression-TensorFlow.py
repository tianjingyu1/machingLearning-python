# 在Jupyter中，使用matplotlib显示图像需要设置为inline模式，否则不会显示图像
# %matplotlib inline

import matplotlib.pyplot as plt  # 载入matplotlib
import numpy as np # 载入numpy
import tensorflow as tf # 载入Tensorflow


# 设置随机数种子
np.random.seed(5)

# 直接采用np生成等差数列的方法，生成100个点，每个店的取值在-1~1之间
x_data = np.linspace(-1, 1, 100)

# y = 2x + 1 + 噪声，其中，噪声的维度与x_data一致
y_data = 2 * x_data + 1.0 + np.random.randn(*x_data.shape) * 0.4

# 画出随机生成数据的散点图
plt.scatter(x_data, y_data)

# 画出我们想要学习到的线性函数 y = 2x + 1
plt.plot(x_data, 2 * x_data + 1.0, color = 'red', linewidth = 3)

# 定义模型函数
def model(x, w, b):
    return tf.multiply(x,w) + b

# 创建变量
# 构建线性函数的斜率，变量w
w = tf.Variable(np.random.randn(), tf.float32)

# 构建线性函数的截距，变量b
b = tf.Variable(0.0, tf.float32)

# 设置训练参数
training_epochs = 10

# 学习率
learning_rate = 0.01

# 定义损失函数L2
def loss(x,y,w,b):
    err = model(x,w,b) - y  # 计算模型预测值和标签值的差异
    squared_err = tf.square(err)  # 求平方，得出方差
    return tf.reduce_mean(squared_err)  # 求均值，得出均方差

# 梯度下降优化器
def grad(x,y,w,d):
    with tf.GradientTape() as tape:
        loss_ = loss(x,y,w,b)
    return tape.gradient(loss_, [w,b])    # 返回梯度向量

step = 0  # 记录训练步数
loss_list = []  # 用于保存loss值的列表
display_step = 10 # 控制训练过程数据显示的频率，不是超参数

for epoch in range(training_epochs):
    for xs,ys in zip(x_data, y_data):
        loss_=loss(xs,ys,w,b)  # 计算损失
        loss_list.append(loss_)  # 保存本次损失计算结果

        delta_w,delta_b = grad(xs,ys,w,b)  # 计算该当前[w,b]点的梯度
        change_w = delta_w * learning_rate  # 计算变量w需要调整的量
        change_b = delta_b * learning_rate  # 计算变量b需要调整的量
        w.assign_sub(change_w)  # 变量w值变更为减去change_w后的值
        b.assign_sub(change_b)  # 变量w值变更为减去change_b后的值

        step=step+1  # 训练步数+1
        if step % display_step == 0:   # 显示训练过程信息
            print("Training Epoch:", '%02d' % (epoch+1),"Step: %03d" % (step),"loss=%.6f" % (loss_))
    plt.plot(x_data,w.numpy() * x_data + b.numpy())  #完成一轮训练后，画出回归的线条

# 打印结果
print("w: ", w.numpy())  # w的值应该在2附近
print("b: ", b.numpy())  # b的值应该在1附近

# 可视化
plt.scatter(x_data,y_data,label='Original data')
plt.plot(x_data, x_data * 2.0+1.0,label='Object line',color='g',linewidth=3)
plt.plot(x_data, x_data * w.numpy()+b.numpy(),label='Fitted line',color='r',linewidth=3)
plt.legend(loc=2) # 通过参数loc指定图例位置
plt.show()

# 查看损失变化情况
plt.plot(loss_list)
plt.show()

# 图形化显示损失值
plt.plot(loss_list)
plt.show()
plt.plot(loss_list,'r+')
plt.show()

# 进行预测
x_test = 3.21

predict = model(x_test,w.numpy(),b.numpy())
print("预测值：%f" % predict)

target = 2* x_test + 1.0
print("目标值：%f" % target)

# 批量梯度下降优化
training_epochs=100  # 迭代次数
learning_rate = 0.05  # 学习率

loss_list = []  # 用于保存loss值的列表

for epoch in range(training_epochs):
    loss_ = loss(x_data,y_data,w,b)  # 计算损失，所有样本作为一个整体参与计算
    loss_list.append(loss_)  # 保存本次损失计算结果

    delta_w,delta_b = grad(x_data,y_data,w,b)  # 计算该当前[w,b]点的梯度
    change_w = delta_w * learning_rate  # 计算变量w需要调整的量
    change_b = delta_b * learning_rate  # 计算变量b需要调整的量
    w.assign_sub(change_w)  # 变量w值变更为减去change_w后的值
    b.assign_sub(change_b)  # 变量b值变更为减去change_b后的值

    print("Training Epoch:",'%02d' % (epoch+1),"loss=%.6f" % (loss_))
    plt.plot(x_data,w.numpy() * x_data + b.numpy())  #完成一轮训练后，画出回归的线条
    plt.show()
    plt.plot(loss_list)
    plt.show()