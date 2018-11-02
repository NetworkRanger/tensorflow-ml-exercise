#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/2 下午10:28

# 1.7 实现激励函数

import tensorflow as tf
sess = tf.Session()

# 1. 整流线性单元(Rectifier linear unit, ReLU)是神经网络最常用的非线性函数
print(sess.run(tf.nn.relu([-3., 3., 10.]))) # [ 0.  3. 10.]

# 2. 有时为了抵消ReLU激励函数的线性增长部分，会在min()函数中嵌入max(0, x), 其在TensorFlow中的实现称作ReLU6，表示为min(max(0,x),6)
print(sess.run(tf.nn.relu6([-3., 3., 10.]))) # [0. 3. 6.]

# 3. sigmoid函数是最常用的连续、平滑的激励函数。它也被称作逻辑函数(Logistic 函数),表示为1/(1+exp(-x))。范围-1到1
print(sess.run(tf.nn.sigmoid([-1., 0., 1.]))) # [0.26894143 0.5        0.7310586 ]

"""
注意：有些激励函数不以0为中心，比如，sigmoid函数。在大部分计算图算法中要求优先使用均值为0的样本数据。
"""

# 4.另外一种激励函数是双曲正切函数(hyper tangent, tanh)。范围0到1。双曲正弦与双曲余弦的比值，表达式(exp(x)-exp(-x))/((exp(x)+exp(-x))
print(sess.run(tf.nn.tanh([-1., 0., 1.]))) # [-0.7615942  0.         0.7615942]

# 5. softsign 函数也是一种激励函数，表达式为: x/(abs(x)+1)。softsign函数是符号函数的连续估计
print(sess.run(tf.nn.softsign([-1., 0., -1.]))) # [-0.5  0.  -0.5]

# 6. softplus 激励函数是ReLU激励函数的平滑版，表达式为: log(exp(x)+1)
print(sess.run(tf.nn.softplus([-1., 0., -1.]))) # [0.31326166 0.6931472  0.31326166]

"""
注意，当输入增加时，softplus激励函数趋近于无限大，softsign函数趋近于1；当输入减小时，softplus激励函数趋近于0，softsign函数趋近于-1。
"""

# 7. ELU 激励函数（Exponential Linear Unit， ELU）与softplus激励函数相似，不同点在于：当输入无限小时，ELU激励函数趋近于-1，而softplus函数趋近于0。表达式为(exp(x)+1) if x < 0 else x
print(sess.run(tf.nn.elu([-1., 0., -1.]))) # [-0.63212055  0.         -0.63212055]
