#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/1 下午9:44

# 2.2 计算图中的操作

import numpy as np
import tensorflow as tf

# 1.首先，声明张量和占位符。这里，创建一个numpy数组，传入计算图操作

x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.)
my_product = tf.mul(x_data, m_const)

with tf.Session() as sess:
    for x_val in x_vals:
        print(sess.run(my_product, feed_dict={x_data: x_val}))

"""
3.0
9.0
15.0
21.0
27.0
"""