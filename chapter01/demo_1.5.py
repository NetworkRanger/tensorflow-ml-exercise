#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/2 下午9:53

# 1.5 操作（计算）矩阵
import tensorflow as tf
import numpy as np

# 1. 创建矩阵
identity_matrix = tf.diag([1.0, 1.0, 1.0])
A = tf.truncated_normal([2, 3])
B = tf.fill([2, 3], 5.0)
C = tf.random_uniform([3, 2])
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
with tf.Session() as sess:
    print(sess.run(identity_matrix))
    """
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    """

    print(sess.run(A))
    """
    [[ 1.66514695  0.21781044  1.36229742]
     [ 0.81741899 -1.43445051 -1.40141094]]
    """

    print(sess.run(B))
    """
    [[ 5.  5.  5.]
     [ 5.  5.  5.]]
    """

    print(sess.run(C))
    """
    [[ 0.53440428  0.14045691]
     [ 0.45097244  0.75755489]
     [ 0.99427879  0.47933364]]
    """

    print(sess.run(D))
    """
    [[ 1.  2.  3.]
     [-3. -7. -1.]
     [ 0.  5. -2.]]
    """

"""
注意，如果再次运行sess.run(C), TensorFlow 会重新初始化随机变量，并得得不同的随机数。
"""

# 2. 矩阵的加法和减法
with tf.Session() as sess:
    print(sess.run(A+B))
    """
    [[ 5.62702417  4.91175747  5.57057142]
     [ 4.77277422  5.41800356  5.61466312]]
    """

    print(sess.run(B-B))
    """
    [[ 0.  0.  0.]
     [ 0.  0.  0.]]
    """

    print(sess.run(tf.matmul(B, identity_matrix)))
    """
    [[ 5.  5.  5.]
     [ 5.  5.  5.]]
    """

# 3. 矩阵乘法函数matmul()可以通过参数指定在矩阵乘法操作前是否进行矩阵转置

# 4. 矩阵转置
with tf.Session() as sess:
    print(sess.run(tf.transpose(C)))
    """
    [[ 0.99314523  0.26559901  0.4991864 ]
     [ 0.19599497  0.75304008  0.37292743]]
    """

# 5. 再次强调，重新初始化将会得到不同的值

# 6. 对于矩阵行列式，使用方式如下
with tf.Session() as sess:
    print(sess.run(tf.matrix_determinant(D))) # -38.0

    # 矩阵的转置
    print(sess.run(tf.matrix_inverse(D)))
    """
    [[-0.5        -0.5        -0.5       ]
     [ 0.15789474  0.05263158  0.21052632]
     [ 0.39473684  0.13157895  0.02631579]]
    """

"""
TensorFlow中的矩阵求逆方法是Cholesky 矩阵分解法（又称为平方根法），矩阵需要为对称正定矩阵或者可以进行LU分解。
"""

# 7. 矩阵分解
# Cholesky 矩阵分解法
with tf.Session() as sess:
    print(sess.run(tf.cholesky(identity_matrix)))
    """
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    """

# 8. 矩阵的特征值和特征向量
with tf.Session() as sess:
    print(sess.run(tf.self_adjoint_eig(D))) # 第一行为特征值，剩下的向量是对应的向量
"""
[[-10.65907521   -0.22750691    2.88658212]
 [  0.21749542    0.63250104   -0.74339638]
 [  0.84526515    0.2587998     0.46749277]  
 [ -0.4880805     0.73004459    0.47834331]]
"""