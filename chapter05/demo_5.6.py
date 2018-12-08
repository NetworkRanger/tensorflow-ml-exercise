#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/8 下午1:52

# 5.6 用TensorFlow实现图像识别

# 1. 导入必要的编程库
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

# 2. 创建一个计算图会话，加载MNIST手写数字数据集，并指定one-hot编码
sess = tf.Session()
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

"""
one-hot编码是分类类别的数值化，这样更有利于后续的数值计算。本例包含10个类别(数字0到9)，采用长度为10的0-1向量表示。例如，类别"0"表示为向量：1,0,0,0,0,0,0,0,0,类别"1"表示向量：0,1,0,0,0,0,0,0,0，等等。
"""

# 3. 由于MINIST手写数字数据集较大，直接计算成千上万个输入的784个特征之间的距离是比较困难的，所以本例会抽样成小数据集进行训练
train_size = 1000
test_size = 102
rand_train_indices = np.random.choice(len(mnist.train.images), train_size, replace=False)
rand_test_indices = np.random.choice(len(mnist.test.images), test_size, replace=False)
x_vals_train = mnist.train.images[rand_train_indices]
x_vals_test = mnist.test.images[rand_test_indices]
y_vals_train = mnist.train.labels[rand_train_indices]
y_vals_test = mnist.test.labels[rand_test_indices]

# 4. 声明k值和批量的大小
k = 4
batch_size = 6

# 5. 现在在计算图中开始初始化占位符，并赋值
x_data_train = tf.placeholder(shape=[None, 784], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y_target_train = tf.placeholder(shape=[None, 10], dtype=tf.float32)
y_target_test = tf.placeholder(shape=[None, 10], dtype=tf.float32)

# 6. 声明距离度量函数。本例使用L1范数(即绝对值)作为距离函数
distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), reduction_indices=2)

"""
注意，我们可以把距离函数定义为L2范数。对应的代码为：
distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), reduction_indices = 1))。
"""

# 7. 找到最接近的top k图片和预测模型。在数据集的one-hot编码索引上进行预测模型计算，然后统计发生的数量
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
prediction_indices = tf.gather(y_target_train, top_k_indices)
count_of_predictions = tf.reduce_sum(prediction_indices, dimension=1)
prediction = tf.argmax(count_of_predictions, dimension=1)

# 8. 在测试集上遍历迭代运行，计算预测值，并将结果存储
num_loops = int(np.ceil(len(x_vals_test)/batch_size))
test_output = []
actual_vals = []
for i in range(num_loops):
    min_index = i*batch_size
    max_index = min((i+1)*batch_size, len(x_vals_train))
    x_batch = x_vals_test[min_index:max_index]
    y_batch = y_vals_test[min_index:max_index]
    predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})

# 9. 现在已经保存了实际值和预测返回值，下面计算模型训练准确度。不过该结果会因为测试数据集和训练数据集的随机抽样而变化，但是其准确度约为80%~90%
accuracy = sum([1./test_size for i in range(test_size) if test_output[i] == actual_vals[i]])
print('Accuracy on test set:' + str(accuracy))
# Accyract ib test set: 0.8333333333333325

# 10. 绘制最后批次的计算结果
actuals = np.argmax(y_batch, axis=1)
Nrows = 2
Ncols = 3
for i in range(len(actuals)):
    plt.subplot(Nrows, Ncols, i+1)
    plt.imshow(np.reshape(x_batch[i], [28,28]), cmap='Greys_r')
    plt.title('Actual: ' + str(actuals[i]) + ' Pred: ' + str(predictions[i]), fontsize=10)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

