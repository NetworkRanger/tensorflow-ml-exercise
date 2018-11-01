#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/1 下午11:02

# 2.6 TensorFlow 实现反向传播

# 1.导入Python的数值计算模块，numpy和tensorflow
import numpy as np
import tensorflow as tf

# 2.创建计算图会话
sess = tf.Session()

# 3.生成数据，创建占位符和变量A
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1]))

# 4.增加乘法操作
my_output = tf.mul(x_data, A)

# 5.增加L2正则损失函数
loss = tf.square(my_output - y_target)

# 6.在运行之前，需要初始化变量
init = tf.initialize_all_variables()
sess.run(init)

# 7.现在声明变量的优化器。
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)

"""
选取最优的学习率的理论很多，但真正解决机器学习算法的问题很难。
"""

# 8.最后一步是训练算法。
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%25 == 0:
        print('Setp #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

"""
Setp #25 A = [ 6.55572081]
Loss = [ 18.17212105]
Setp #50 A = [ 8.83527851]
Loss = [ 0.0003357]
Setp #75 A = [ 9.65467072]
Loss = [  6.54210817e-06]
Setp #100 A = [ 9.98147297]
Loss = [ 0.34434497]
"""

# 9.现在将介绍简单的分类算法例子

# 10.首先，重置计算图，并且重新初始化变量
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.Session()

# 11. 从正态分布(N(-1,1), N(3,1))生成数据
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

"""
初始化变量A为10附近的值，远离理论值-1。这样可以清楚地显示算法是如何从10收敛为-1的。
"""

# 12. 增加转换操作
my_output = tf.add(x_data, A)

# 13.由于指定的损失函数期望批量数的维度，这里使用expand_dims()函数增加维度
my_output_expanded = tf.expand_dims(my_output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)

# 14.初始化变量A
init = tf.initialize_all_variables()
sess.run(init)

# 15.声明损失函数，这里使用一个带非归一化logits的交叉熵的损失函数，同时会用sigmod函数转换
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(my_output_expanded, y_target_expanded)

# 16. 如前面回归算法的例子，增加一个优化器函数让TensorFlow知道如何更新和偏差变量
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

# 17.最后，通过随机选择的数据迭代几百次，相应地更新变量A
for i in range(1400):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%200 == 0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))

"""
Step #200 A = [ 4.99277401]
Loss = [[ 4.54716539]]
Step #400 A = [ 1.1278429]
Loss = [[ 0.60332268]]
Step #600 A = [-0.43532506]
Loss = [[ 0.3442266]]
Step #800 A = [-0.69333619]
Loss = [[ 0.06842079]]
Step #1000 A = [-0.92347127]
Loss = [[ 0.66925561]]
Step #1200 A = [-0.87109089]
Loss = [[ 0.13961451]]
Step #1400 A = [-0.95246935]
Loss = [[ 0.12325969]]
"""