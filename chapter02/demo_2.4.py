#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/1 下午10:18

# 2.4 TensorFlow 的多层Layer

import numpy as np
import tensorflow as tf

# 1.首先，通过numpy创建2D图像，4x4像素图片。
x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)

# 2.下面在计算图中创建占位符
x_data = tf.placeholder(tf.float32, shape=x_shape)

# 3.为了创建过滤4x4像素图片的滑动窗口，我们将用TensorFlow内建函数conv2d()(常用来做图像处理) 卷积2x2形状的常量窗口
my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
my_strides = [1, 2, 2, 1]
mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_Avg_Window')

"""
可用使用公式: Output = (W-F+2P)/S+1 计算卷积层的返回值形状。这里，W是输入形状，F是过滤器形状，P是padding的大小，S是步长形状。
"""

# 4.注意，我们通过conv2d()函数的name参数，把这层Layer命名为"Moving_Avg_Window"

# 5.现在定义一个自定义Layer,操作滑动窗口平均的2x2的返回值。
def customer_layer(input_matrix):
    """
    :param input_matrix: 输入张量
    :return: 乘以一个2x2的矩阵张量，然后每个元素加1
    """
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    temp1 = tf.matmul(A, input_matrix_sqeezed)
    temp = tf.add(temp1, b) # Ax+b
    return (tf.sigmoid(temp))

# 6.现在把刚刚新定义的Layer加入到计算图中，并且用tf.name_scope()命名唯一的Layer名字，后续在计算图中可折叠/扩展Custom_Layer层
with tf.name_scope('Custom_Layer') as scope:
    customer_layer1 = customer_layer(mov_avg_layer)

# 7.为占位符传入4x4像素图片，然后执行计算图
with tf.Session() as sess:
    print(sess.run(customer_layer1, feed_dict={x_data: x_val}))

"""
[[ 0.89407796  0.90433455]
 [ 0.78013289  0.87182242]]
"""