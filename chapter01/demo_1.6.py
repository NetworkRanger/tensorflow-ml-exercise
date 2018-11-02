#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/2 下午10:16

# 1.6 声明操作

import tensorflow as tf
sess = tf.Session()

# 1. TensorFlow 提供div()函数的多种变种形式和相关的函数

# 2. 值得注意的, div()函数返回值的数据类型与输入数据类型一致
print(sess.run(tf.div(3, 4))) # 0
print(sess.run(tf.truediv(3,4))) # 0.75

# 3. 如果要对浮点数进行整数除法，可以使用floordiv()函数
print(sess.run(tf.floordiv(3.0, 4.0))) # 0.0

# 4. 另外一个重要的函数是mod()（取模)。此函数返回除法的余数。
print(sess.run(tf.mod(22.0, 5.0))) # 2.0

# 5. 通过cross()函数云计算两个张量间的点积
print(sess.run(tf.cross([1., 0., 0.], [0., 1., 0.]))) # [ 0.   0.   1.0]

# 6. 数学函数的列表

# 7. 特殊数学函数

