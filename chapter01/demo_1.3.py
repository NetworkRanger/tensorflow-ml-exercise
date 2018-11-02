#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/2 下午9:28

# 1.3 声明张量

import tensorflow as tf

# 1.固定张量
# 创建指定维度的零张量
# zero_tsr = tf.zeros([row_dim, col_dim])
# 创建指定维度的单位张量
# ones_str = tf.ones([row_dim, col_dim])
# 创建指定维度的常数填充的张量
# filled_tsr = tf.fill([row_dim, col_dim], 42)
# 用已知常数张量创建一个张量
constant_tsr = tf.constant([1, 2, 3])

"""
tf.constant()函数也能广播一个值为数组，然后模拟tf.fill()函数的功能，具体的写法为: tf.constant(42, [row_dim, col_dim])
"""

# 2. 相似形状的张量
# 新建一个与给定的tensor类型大小一致的tensor
zeros_similar = tf.zeros_like(constant_tsr)
ones_siilar = tf.ones_like(constant_tsr)

"""
因为这些张量依赖给定的张量，所以初始化时需要按序进行。如果打算一次性初始化所有张量，那么程序将会报错。
"""

# 3. 序列张量
# TensorFlow可以创建指定间隔的张量
linear_tsr = tf.linspace(start=0, stop=1, num=3) # [0.0, 0.5, 1.0]

interger_seq_tsr = tf.range(start=6, limit=15, delta=3) # [6, 9, 12] 不包括limit值

# 4. 随机张量
# tf.random_uniform()函数生成均匀分布的随机数
# randunif_tsr = tf.random_normal([row_dim, col_dim], minval=0, maxval=1)  # (minval <= x < maxval)
# tf.random_normal()函数生成正态分布的随机数
# randorm_tsr = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)
# tf.truncated_normal()函数生成带有指定边界的正态分布的随机数，其正态分布的随机数位于指定均值(期望)到两个标准差之间的区间
# runcnorm_tsr = tf.truncated_normal([row_dim, col_dim], minval=0.0, stddev=1.0)
# 张量/数组的随机化。
# shuffled_output = tf.random_shuffle(input_tensor)
# cropped_output = tf.random_crop(input_tensor, crop_size)
# 张量的随机剪裁。tf.random_crop()可以实现对张量指定大小的随机剪裁。
# cropped_image = tf.random_crop(my_image, [height/2, width/2, 3])

