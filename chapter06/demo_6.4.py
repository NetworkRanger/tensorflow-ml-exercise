#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/8 下午3:59

# 6.4 用TensorFlow实现单层神经网络

# 1. 创建计算图会话，导入必要的编程库
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# 2. 加载Iris数据集，存储花萼长度作为目标值，然后开始一个计算图会话
iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])
sess = tf.Session()

# 3. 因为数据集比较小，我们设置一个种子使得返回结果可复现
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

# 4. 为了准备数据集，我们创建一个80-20分的训练集和测试集。通过min-max缩放法正则化特征值为0到1之间
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
def normailize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)

x_vals_train = np.nan_to_num(normailize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normailize_cols(x_vals_test))

# 5. 现在为数据集和目标值声明批量大小和占位符
batch_size = 50
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 6. 这一步相当重要，声明有合适形状的模型变量。我们能声明隐藏层为任意大小，本例中设置为有五个隐藏节点
hidden_layer_nodes = 5
A1 = tf.Variable(tf.random_normal(shape=[3,hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1]))
b2 = tf.Variable(tf.random_normal(shape=[1]))

# 7. 分两步声明训练模型: 第一步，创建一个隐藏层输出；第二步，创建训练模型的最后输出
"""
注意，本例中的模型有三个特征、五个隐藏节点和一个输出结果值。
"""
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

# 8. 这里定义均方误差为损失函数
loss = tf.reduce_mean(tf.square(y_target - final_output))

# 9. 声明优化算法，初始化模型变量
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

# 10. 遍历迭代训练模型。我们也初始化两个列表(list)存储训练损失和测试损失。在每次迭代训练时，随机选择批量训练数据来拟合模型
# First we initialize the loss vectors for storage.
loss_vec = []
test_loss = []
for i in range(500):
    # First we select a random set of indices for the batch.
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    # We then select the training values
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    # Now we run the training loss
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    # We save the training loss
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(np.sqrt(temp_loss))

    # Finaly, we run the test-set loss and save it.
    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(np.sqrt(test_temp_loss))
    if (i+1)%50 == 0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))

# 11. 使用matplotlib绘制损失函数
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

