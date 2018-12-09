#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/9 下午5:55

# 6.7 线性预测模型的优化

# 1. 导入必要的编程库，初始化计算图会话
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
sess = tf.Session()

# 2. 加载低出生体重数据集，并对其进行抽取和归一化。有一点不同的是，本例中将使用低出生体重指示变量作为目标值，而不是实际出生体重
birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')[5:]
birth_header = [x for x in birth_data[0].split(' ') if len(x) >= 1]
birth_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in birth_data[1:] if len(y)>=1]
y_vals = np.array(x[1] for x in birth_data)
x_vals = np.array(x[2:9] for x in birth_data)
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

def normailze_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)

# 3. 声明批量大小和占位符
batch_size = 90
x_data = tf.placeholder(shape=[None, 8], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 4. 我们声明函数来初始化算法模型中的变量和层。为了创建一个更好的逻辑层，我们需要创建一个返回输入层的逻辑层的函数。换句话说，我们需要使用全连接 层，返回每层的值。注意，损失函数包括最终的sigmoid函数，所以我们指定最后一层不必返回输出的sigmoid值
def init_variable(shape):
    return (tf.Variable(tf.random_normal(shape=shape)))

# Create a logistic layer definition
def logistic(input_layer, multiplication_weight, bias_weight, activation = True):
    linear_layer = tf.add(tf.matmul(input_layer, multiplication_weight), bias_weight)

    if activation:
        return (tf.nn.sigmoid(linear_layer))
    else:
        return (linear_layer)

# 5. 声明神经网络的三层(两个隐藏层和一个输出层)。我们为每层初始化一个权重矩阵和偏置矩阵，并定义每层的操作
# First logistic layer (7 input to 14 hiddden nodes)
A1 = init_variable(shape=[7, 14])
b1 = init_variable(shape=[14])
logistic_layer1 = logistic(x_data, A1, b1)
# Second logistic layer( 14 hidden inputs to 5 hidden nodes)
A2 = init_variable(shape=[14, 5])
b2 = init_variable(shape=[5])
logistic_layer2 = logistic(logistic_layer1, A2, b2)
A3 = init_variable(shape=[5,1])
b3 = init_variable(shape=[1])
final_output = logistic(logistic_layer2, A3, b3, activation=False)

# 6. 声明损失函数(本例中使用的是交叉熵损失函数)和优化算法，并初始化变量
# Create loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(final_output, y_target))
# Declare optimizer
my_opt = tf.train.AdamOptimizer(learning_rate= 0.002)
train_step = my_opt.minimize(loss)
# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

"""
交叉熵是度量概率之间的距离。这里度量确定值（0或者1）和模型概率值(0<x<1)之间的差值。在TensorFlow中实现的交叉熵是用函数sigmoid()内建的。采用超参数调优对于寻找最好的损失函数、学习率和优化算法是相当重要的，但是为了本节示例的乘法性，这里不佛如超参数调优。
"""

# 7. 为了评估和比较算法模型，创建计算图预测操作和准确操作。这使得我们可以传入测试集并计算准确度
prediction = tf.round(tf.nn.sigmoid(final_output))
predictions_corrent = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_corrent)

# 8. 准备开始遍历迭代训练模型。本例训练 1500次，并为后续绘图保存模型的损失函数和训练集/测试集准确度
# Initialize loss and accuracy vectors
loss_vec = []
train_acc = []
test_acc = []

for i in range(15000):
    # Select random indices for batch selection
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    # Select batch
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    # Run training step
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    # Get traning loss
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    # Get training accuracy
    temp_acc_train = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_acc.append(temp_acc_train)
    # Get test accuracy
    temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_acc.append(temp_acc_test)
    if (i+1)%150==0:
        print('Loss = ' + str(temp_loss))

# 9. 输出结果如下
"""

"""

# 10. 下面的代码块展示如何用matplotlib模块绘制交叉熵损失函数和测试集/训练集准确度
# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()
# Plot train and etst accuracy
plt.plot(train_acc, 'k-', label='Train Set Accuracy')
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
