#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/2 下午9:23

# 1.2 TensorFlow 如何工作

import tensorflow as tf

# 1. 导入/生成样本数据集。

# 2. 转换和归一化数据。
# data = tf.nn.batch_norm_with_global_normalization(...)

# 3. 划分样本数据集为训练样本集、测试样本集和验证样本集。

# 4. 设置机器学习参数（超参数）。
learning_rate = 0.01
batch_size = 100
iterations = 1000

# 5. 初始化变量和占位符。
a_var = tf.constant(42)
# x_input = tf.placeholder(tf.float32, [None, input_size])
# y_input = tf.placeholder(tf.float32, [None, num_classses])

# 6. 定义模型结构。
# y_pred = tf.add(tf.mul(x_input, weight_matrix), b_matrix)

# 7. 声明损失函数。
# loss = tf.reduce_mean(tf.square(y_actual - y_pred))

# 8. 初始化模型和训练模型。
# with tf.Session(graph=graph) as session:
#     ...
#     session.run(...)
#     ...

# 9. 评估机器学习模型。

# 10. 调优超参数。

# 11. 发布/预测结果。
