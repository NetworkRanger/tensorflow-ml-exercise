#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/4 下午3:36

# 3.8 用TensorFlow 实现弹性网络回归算法

# 1. 导入必要的编程库并初始化一个计算图
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
sess = tf.Session()

# 2. 加载数据集
iris = datasets.load_iris()
x_vals = np.array([[x[1], x[2], x[3]] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# 3. 声明批量大小、占位符、变量和模型输出
batch_size = 50
learning_rate = 0.001
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[3,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
model_output = tf.add(tf.matmul(x_data, A), b)

# 4. 对于弹性网络回归算法，损失函数包含斜率L1正则和L2正则。创建L1和L2正则项，然后加入到损失函数中
elastic_param1 = tf.constant(1.)
elastic_param2 = tf.constant(1.)
l1_a_loss = tf.reduce_mean(tf.abs(A))
l2_a_loss = tf.reduce_mean(tf.square(A))
e1_term = tf.multiply(elastic_param1, l1_a_loss)
e2_term = tf.multiply(elastic_param2, l2_a_loss)
loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)

# 5. 现在初始化变量，声明优化器，然后遍历迭代支行，训练拟合得到系数
init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)
loss_vec = []
for i in range(1000):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss[0])
    if (i+1)%250 == 0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))


# 6. 代码支行输出的结果
"""
Step #250 A = [[ 1.0351679]
 [ 1.4237307]
 [-1.5953726]] b = [[-1.003884]]
Loss = [3.5570922]
Step #500 A = [[ 1.1186575]
 [ 1.2246706]
 [-1.2564372]] b = [[-0.89256376]]
Loss = [2.967632]
Step #750 A = [[ 1.1500645 ]
 [ 1.0798686 ]
 [-0.96323276]] b = [[-0.79036397]]
Loss = [2.4493415]
Step #1000 A = [[ 1.1597048]
 [ 0.9703532]
 [-0.7137709]] b = [[-0.6926596]]
Loss = [2.2044964]
"""

# 7. 现在能观察到，随着训练迭代后损失函数已收敛
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()