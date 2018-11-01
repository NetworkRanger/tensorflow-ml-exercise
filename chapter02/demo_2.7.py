#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/1 下午11:25

# 2.7 TensorFlow 实现随机训练和批量训练

import matplotlib as plt
import numpy as np
import tensorflow as tf
sess = tf.Session()

# 1.开始声明批量大小
batch_size = 10

# 2.接下来，声明模型的数据、占位符和变量
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))

# 3.现在在计算图中增加矩阵乘法操作，切记矩阵乘法不满足交换律，所以在matmul()函数中的矩阵参数顺序要正确
my_output = tf.matmul(x_data, A)

# 4.改变损失函数
loss = tf.reduce_mean(tf.square(my_output - y_target))

# 5.声明优化器
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()
sess.run(init)

# 6.在训练中通过循环迭代优化模型算法
loss_batch = []
for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%5 == 0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_batch.append(temp_loss)

# 7.迭代100次输出最终返回值
"""
Step #5 A = [[ 1.62740648]]
Loss = 71.4052
Step #10 A = [[ 3.15520906]]
Loss = 45.0727
Step #15 A = [[ 4.41177416]]
Loss = 33.1787
Step #20 A = [[ 5.43022346]]
Loss = 20.9183
Step #25 A = [[ 6.27166891]]
Loss = 14.6422
Step #30 A = [[ 6.95059967]]
Loss = 8.77935
Step #35 A = [[ 7.51708412]]
Loss = 7.13663
Step #40 A = [[ 7.98048306]]
Loss = 4.8445
Step #45 A = [[ 8.34100628]]
Loss = 3.47932
Step #50 A = [[ 8.65559578]]
Loss = 3.65749
Step #55 A = [[ 8.91608334]]
Loss = 2.38124
Step #60 A = [[ 9.10496807]]
Loss = 2.3072
Step #65 A = [[ 9.26584053]]
Loss = 0.708979
Step #70 A = [[ 9.41215897]]
Loss = 0.859235
Step #75 A = [[ 9.51047802]]
Loss = 1.53797
Step #80 A = [[ 9.62157536]]
Loss = 1.14755
Step #85 A = [[ 9.71101952]]
Loss = 1.3226
Step #90 A = [[ 9.75013924]]
Loss = 0.588486
Step #95 A = [[ 9.80838585]]
Loss = 1.37821
Step #100 A = [[ 9.85362339]]
Loss = 0.95122
"""