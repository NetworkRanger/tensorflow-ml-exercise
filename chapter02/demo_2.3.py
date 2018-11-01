#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/1 下午10:06

# 2.3 TensorFlow 的嵌入Layer

import numpy as np
import tensorflow as tf

# 1.首先，创建数据和占位符

my_array = np.array([[1., 3., 5., 7., 9.],
                [-2., 0., 2., 4., 6.],
                [-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array+1])
x_data = tf.placeholder(tf.float32, shape=(3, 5))

# 2.接着，创建矩阵乘法和加法中要用到的常量矩阵
m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# 3.现在声明操作，表示成计算图
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)

# 4.最后，通过计算图赋值
with tf.Session() as sess:
    for x_val in x_vals:
        print(sess.run(add1, feed_dict={x_data: x_val}))

"""
[[ 102.]
 [  66.]
 [  58.]]
[[ 114.]
 [  78.]
 [  70.]]
"""