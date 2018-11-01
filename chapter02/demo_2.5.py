#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/1 下午10:33

# 2.5 TensorFlow实现损失函数

import matplotlib.pyplot as plt
import tensorflow as tf

x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

# 1.L2正则损失函数（即欧拉损失函数）
l2_y_vals = tf.square(target - x_vals)
with tf.Session() as sess:
    sess.run(l2_y_vals)

"""
TensorFlow有内建的L2正则形式，称为nn.l2_loss()。这个函数其实是实际L2正则的一半，换句话说，它是上面l2_y_vals的1/2。
"""

# 2.L1正则损失函数（即绝对值损失函数）
l1_y_vals = tf.abs(target - x_vals)
with tf.Session() as sess:
    sess.run(l1_y_vals)

# 3. Pseudo-Huber损失函数是Huber损失函数的连续、平滑估计，试图利用L1和L2正则消减极值处的陡峭，使得目标值附近连续
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.mul(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)
with tf.Session() as sess:
    phuber1_y_out = sess.run(phuber1_y_vals)

delta2 = tf.constant(5.)
phuber2_y_vals = tf.mul(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals)/delta2)) - 1.)
with tf.Session() as sess:
    phuber2_y_out = sess.run(phuber2_y_vals)

# 4.分类损失函数是用来评估预测分类结果的

# 5.重新给x_vals和target赋值,保存返回值并在下节绘制出来
x_vals = tf.linspace(-3., 5., 500)
target = tf.constant(1.)
targets = tf.fill([500,], 1.)

# 6.Hinge损失函数主要用来评估支持向量机算法，但有时也用来评估神经网络算法
hinge_y_vals = tf.maximum(0., 1., -tf.mul(target, x_vals))
with tf.Session() as sess:
    hinge_y_out = sess.run(hinge_y_vals)

# 7.两类交叉熵损失函数(Cross-entropy loss)有时也作为逻辑损失函数
xentropy_y_vals = -tf.mul(target, tf.log(x_vals)) - tf.mul((1. - target), tf.log(1. - x_vals))
with tf.Session() as sess:
    xentropy_y_out = sess.run(xentropy_y_vals)

# 8.Sigmoid交叉熵损失函数(Sigmoid cross entropy loss)与上一个损失函数非常类似，有一点不同的是，它先把x_vals值通过sigmoid函数转换，再计算交叉熵损失
xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(x_vals, targets)
with tf.Session() as sess:
    xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)

# 9.加权交叉熵损失函数(Weighted cross entropy loss)是Sigmoid交叉熵损失函数的加权，对正目标加权
weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, targets, weight)
with tf.Session() as sess:
    xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

# 10.Softmax交叉熵损失函数(Softmax cross-entropy loss)是作用于非归一化的输出结果,只针对单个目标分类的计算损失
unscaled_logits = tf.constant([[1., -3., 10.]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(unscaled_logits)
with tf.Session() as sess:
    print(sess.run(softmax_xentropy))

# [ 1.16012561]

# 11. 稀疏Softmax交叉损失函数（Spare softmax cross-entropy loss)和上一个损失函数类似，它是把目标分类为true的转换成index，而Softmax交叉熵损失函数将目标转换成概率分布
unscaled_logits = tf.constant([[1., -3., 10.]])
spare_target_dist = tf.constant([2])
spare_xentroy = tf.nn.sparse_softmax_cross_entropy_with_logits(unscaled_logits)
with tf.Session() as sess:
    print(sess.run(spare_xentroy))

# [ 0.00012564]