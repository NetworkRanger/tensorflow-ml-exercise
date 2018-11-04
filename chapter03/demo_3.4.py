#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/4 下午2:10

# 3.4 用TensorFlow 实现线性回归算法

# 1. 导入必要的编程库，创建计算图，加载数据集
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess = tf.Session()
iris = datasets.load_iris()

x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# 2. 声明学习率，批量大小，占位符和模型变量
learning_rate = 0.25
batch_size = 25
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# 3. 增加线性模型，y=Ax+b
model_output = tf.add(tf.matmul(x_data, A), b)

# 4. 下一步，声明L2损失函数，其为批量损失的平均值
loss = tf.reduce_mean(tf.square(y_target - model_output))
init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# 5. 现在遍历迭代，并在随机选择的批量数据上进行模型训练
loss_vec = []
for i in range(100):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    if (i+1)%25 == 0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

    """
    Step #25 A = [[1.1186129]] b = [[4.624386]]
    Loss = 0.2815529
    Step #50 A = [[0.99625796]] b = [[4.8526797]]
    Loss = 0.19473928
    Step #75 A = [[0.8756156]] b = [[4.7133036]]
    Loss = 0.27699533
    Step #100 A = [[0.7275057]] b = [[4.6071167]]
    Loss = 0.19546796
    """

# 6. 抽取系数，创建最佳拟合直线
[slope] = sess.run(A)
[y_intercept] = sess.run(b)
best_fit = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)

# 7. 这里将线绘制两幅图
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pdeal Width')
plt.ylabel('Sepal Length')
plt.show()
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.show()

"""
这里很容易看出算法模型是过拟合还是欠拟合。将数据集分割成测试数据集和训练数据集，如果训练数据集的准确度更大，而测试数据集的准确度更低，那么该拟合为过拟合；如果在测试数据集和训练数据集上的准度度都一直在增加，那么该拟合是欠拟合，需要继续训练。
"""