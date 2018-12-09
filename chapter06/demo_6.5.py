#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/8 下午4:06

# 6.5 用TensorFlow实现神经网络常见层

# 1. 导入需要的编程库，创建计算图会话
import tensorflow as tf
from tensorflow.python import ops
import numpy as np
sess = tf.Session()

# 2. 初始化数据，该数据为NumPy数组，长度为25。创建传入数据的占位符
data_size = 25
data_id = np.random.normal(size=data_size)
x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])

# 3. 定义一个卷积层的函数。接着声明一个随机过滤层，创建一个卷积层
"""
注意，许多TensorFlow的层函数是为四维数据设计的(4D = [batch_size, width, height, channels])。我们需要调整输入数据和输出数据，包括扩展维度和降维。在本例中，批量大小为1，宽度为1，高度为25，颜色通道为1。为了扩展维度，使用expand_dims()函数；子降维使用squeeze()函数。卷积层的输出结果的维度公式为output_size=(W-F+2P)/s+1，其中W为输入数据维度，F为过滤层大小，P是padding大小，S是步长大小。
"""

def conv_layer_1d(input_1d, my_filter):
    # Make 1d input into 4d
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform convolution
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1,1,1,1], padding='VALID')
    # Now drop extra dimensions
    conv_output_1d = tf.squeeze(convolution_output)
    return (conv_output_1d)

my_filter = tf.Variable(tf.random_normal(shape=[1,5,1,1]))
my_convolution_output = conv_layer_1d(x_input_1d, my_filter)

# 4. TensorFlow的激励函数默认是逐个元素进行操作。这意味着，在部分层中使用激励函数。下面创建一个激励函数并初始化
def activation(input_1d):
    return (tf.nn.relu(input_1d))
my_activation_output = activation(my_convolution_output)

# 5. 声明一个池化层函数，该函数在一维向量的移动窗口上创建池化层函数。对于本例，其宽度为5
"""
TensorFlow的池化层函数的参数与卷积层函数参数非常相似。但是它没有过滤层，只有形状、步长和padding选项。因为我们的窗口宽度为5，并且具有valid padding(即非零padding)，所以输出数组将有4或者2*floor(5/2)项。
"""

def max_pool(input_1d, width):
    # First we make the 1d input into 4d.
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform the max pool operation
    pool_ouput = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1], strides=[1, 1, 1, 1], padding='VALID')
    pool_output_1d = tf.squeeze(pool_ouput)
    return (pool_output_1d)
my_maxpool_output = max_pool(my_activation_output, width=5)


# 6. 最后一层连接的是全连接层。创建一个函数，该函数输入一维数据，输出值的索引。记住一维数组做矩阵乘法需要提前扩展为二维
def fully_connected(input_layer, num_outputs):
    # Create weights
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs]]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])
    # Make input into 2d
    input_layer_2d = tf.expand_dims(input_layer, 0)
    # Perform fully connected opeartions
    full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)
    # Drop extra dimensions
    full_output_1d = tf.squeeze(full_output)
    return(full_output_1d)

my_full_output = fully_connected(my_maxpool_output, 5)

# 7. 初始化所有的变量，运行计算图打印出每层的输出结果
init = tf.global_variables_initializer()
sess.run(init)
feed_dict = {x_input_1d: data_id}
# Convolution Output
print('Input = array of length 25')
print('Convolution w/filter, length = 5, stride size = 1, results in array of length 21:')
print(sess.run(my_convolution_output, feed_dict=feed_dict))
# Activation Output
print('\nInput = the above array of length 21')
print('ReLU element wise returns the array of length 21:')
print(sess.run(my_activation_output, feed_dict=feed_dict))
# Maxpool Output
print('\nInput = the above array of length 21')
print('MaxPool, widnow length = 5, stride size = 1, results in the array of length 17:')
print(sess.run(my_maxpool_output, feed_dict=feed_dict))
# Fully Connected Output
print('\nInput = the above array of length 17')
print('Fully connected layer on all four rows width five outputs')
print(sess.run(my_full_output, feed_dict=feed_dict))

# 8. 输出结果如下

"""
Input = array of length 25
Convolution w/filter, length = 5, stride size = 1, results in array of length 21:
[-0.04226831  2.1766417   1.479511    3.9582567   2.9083648   0.08136459
 -1.4482962  -1.6872277  -4.463433   -2.8461943   3.3791964   7.6700335
  3.8769639  -5.8582215  -4.809179    4.828122    1.1037129  -6.4981747
  1.2000874   5.6548576   2.8061943 ]

Input = the above array of length 21
ReLU element wise returns the array of length 21:
[0.         2.1766417  1.479511   3.9582567  2.9083648  0.08136459
 0.         0.         0.         0.         3.3791964  7.6700335
 3.8769639  0.         0.         4.828122   1.1037129  0.
 1.2000874  5.6548576  2.8061943 ]

Input = the above array of length 21
MaxPool, widnow length = 5, stride size = 1, results in the array of length 17:
[3.9582567  3.9582567  3.9582567  3.9582567  2.9083648  0.08136459
 3.3791964  7.6700335  7.6700335  7.6700335  7.6700335  7.6700335
 4.828122   4.828122   4.828122   5.6548576  5.6548576 ]

Input = the above array of length 17
Fully connected layer on all four rows width five outputs
[-2.1958802   2.4280741  -0.4436941   0.64456964 -4.056323  ]
"""

"""
神经网络对于一维数据非常重要。时序数据集、信号处理数据集和一些文本嵌入数据集都是一维数据，会频繁使用到神经网络算法
"""

# 二维数据集上进行层函数操作

# 1. 重置计算图会话
ops.reset_default_graph()
sess = tf.Session()

# 2. 初始化输入数组为10x10的矩阵，然后初始化计算图的占位符
data_size = [10,10]
data_2d = np.random.normal(size=data_size)
x_input_2d = tf.placeholder(dtype=tf.float32, shape=data_size)

# 3. 声明一个卷积层函数。因为数据集已经具有高度和宽度了，这里仅需要再扩展两维(指大小为1，颜色通道为1)即可使用卷积conv2d()函数。本例将使用一个随机的2x2过滤层，两个方向的步长和valid padding (非零padding）。由于输入数据是10x10，因此卷积输出为5x5
def conv_layer_2d(input_2d, my_filter):
    # First, change 2d input to 4d
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform convolution
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1,2,2,1], padding='VALID')
    # Drop extra dimensions
    conv_output_2d = tf.squeeze(convolution_output)
    return (conv_output_2d)

my_filter = tf.Variable(tf.random_normal(shape=[2,2,1,1]))
my_convolution_output = conv_layer_2d(x_input_2d, my_filter)

# 4. 激励函数是针对逐个元素的，现创建激励函数并初始化
def activation(input_2d):
    return (tf.nn.relu(input_2d))
my_activation_output = activation(my_convolution_output)

# 5. 本例的池化层与一维数据例子中的相似，有一点不同的是，我们需要声明池化层移动窗口的宽度和高度。这里将与二维卷积层一样，将扩展池化层为二维
def max_pool(input_2d, width, height):
    # Make 2d input into 4d
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    # Perform max pool
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, height, width, 1], strides=[1, 1, 1, 1], padding='VALID')
    # Drop extra dimensions
    pool_output_2d = tf.squeeze(pool_output)
    return (pool_output_2d)

my_maxpool_output = max_pool(my_activation_output, width=2, height=2)

# 6. 本例中的全连接层也与一维数据的输出相似。注意，全连接层的二维输入看作一个对象，为了实现每项连接到每个输出，我们打平二维矩阵，然后在做矩阵乘法时再扩展维度
def fully_connected(input_layer, num_outputs):
    # Flatten into id
    flat_input = tf.reshape(input_layer, [-1])
    # Create weights
    weight_shape = tf.squeeze(tf.stack([tf.shape(flat_input), [num_outputs]]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])
    # Change into 2d
    input_2d = tf.expand_dims(flat_input, 0)
    # Perform fully connected operations
    full_output = tf.add(tf.matmul(input_2d, weight), bias)
    # Drop extra dimensions
    full_output_2d = tf.squeeze(full_output)
    return (full_output_2d)

my_full_output = fully_connected(my_maxpool_output, 5)

# 7. 初始化变量，创建一个赋值字典
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_2d: data_2d}

# 8. 打印每层的输出结果
# Convolution Output
print('Input = [10 x 10] array')
print('2x2 Convolution, stride size = [2x2], results in the [5x5] array: ')
print(sess.run(my_convolution_output, feed_dict=feed_dict))
# Activation Output
print('\nInput = the above [5x5] array')
print('ReLU element wise returns the [5x5] array: ')
print(sess.run(my_activation_output, feed_dict=feed_dict))
# Max Pool Output
print('\nInput = the above [5x5] array')
print('MaxPool, stride size = [1x1], results in the [4x4] array: ')
print(sess.run(my_maxpool_output, feed_dict=feed_dict))
# Fully Connected Output
print('\nInput = the above [4x4] array')
print('Fully connected layer on all four rows with five outputs: ')
print(sess.run(my_full_output, feed_dict=feed_dict))

# 9. 输出结果如下
"""
Input = [10 x 10] array
2x2 Convolution, stride size = [2x2], results in the [5x5] array: 
[[ 0.39217567 -0.06355717 -0.6060767  -0.9238882   1.0734402 ]
 [ 0.6006787  -0.46232924 -0.99625874 -0.14551526 -0.67814124]
 [ 0.8102118   1.4385259  -0.17739128 -0.6986068   0.35558495]
 [-0.7145037  -1.1163896   1.4005777  -0.21027148  0.09515984]
 [ 1.077063   -0.5544648   0.05885498 -0.34765473 -0.4881786 ]]

Input = the above [5x5] array
ReLU element wise returns the [5x5] array: 
[[0.39217567 0.         0.         0.         1.0734402 ]
 [0.6006787  0.         0.         0.         0.        ]
 [0.8102118  1.4385259  0.         0.         0.35558495]
 [0.         0.         1.4005777  0.         0.09515984]
 [1.077063   0.         0.05885498 0.         0.        ]]

Input = the above [5x5] array
MaxPool, stride size = [1x1], results in the [4x4] array: 
[[0.6006787  0.         0.         1.0734402 ]
 [1.4385259  1.4385259  0.         0.35558495]
 [1.4385259  1.4385259  1.4005777  0.35558495]
 [1.077063   1.4005777  1.4005777  0.09515984]]

Input = the above [4x4] array
Fully connected layer on all four rows with five outputs: 
[-0.1304383   0.10000962  0.2066786  -0.00190854 -0.09034774]
"""