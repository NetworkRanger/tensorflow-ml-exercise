#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/2 下午9:48

# 1.4 使用占位符和变量

import tensorflow as tf
import numpy as np

my_var = tf.Variable(tf.zeros([2,3]))
sess = tf.Session()
# 声明变量后需要初始化变量
initialize_op = tf.global_variabels_initializer()
sess.run(initialize_op)

# 占位符仅仅声明数据位置，用于传入数据到计算图。占位符通过会话的feed_dict参数获取数据。
sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[2,2])
y = tf.identity(x)
x_vals = np.random.rand(2, 2)
sess.run(y, feed_dict={x: x_vals})
# Note that sess.run(x, feed_dict={x: x_vals}) will result in a self-referencing error.

