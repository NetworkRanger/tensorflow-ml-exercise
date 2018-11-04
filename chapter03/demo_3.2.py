#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/4 下午1:49

# 3.2 用TensorFlow 求逆矩阵

# 1. 导入必要的编程库，初始化计算图，并生成数据
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
sess = tf.Session()
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)

# 2. 创建后续求逆方法所需的矩阵
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))
b = np.transpose(np.matrix(y_vals))

# 3. 将A和矩阵转换成张量
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

# 4. 现在，使用TensorFlow的tf.matrix_inverse()方法
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
tA_A_inv = tf.matrix_inverse(tA_A)
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
solution = tf.matmul(product, b_tensor)
solution_eval = sess.run(solution)

# 5. 从解中抽取系数、斜率和y截距y-intercept
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
print('slope: ' + str(slope))
print('y_intercept： ' + str(y_intercept))

"""
slope: 0.9644710280808397
y_intercept： 0.21858593646035482
"""

best_fit = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)

plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.show()