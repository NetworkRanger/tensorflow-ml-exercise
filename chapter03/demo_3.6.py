#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/4 下午2:50

# 3.6 用TensorFlow 实现戴明回归算法

# 1. 导入必要的编程库，创建计算图，加载数据集
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
sess = tf.Session()
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])
batch_size = 50
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
model_output = tf.add(tf.matmul(x_data, A), b)

# 2. 损失函数是由分子和分母组成的几何公式。给你直线y = mx + b,点(x0,y0)到直线的距离公式为d=abs(y0-(mx0+b))/sqrt(m**2+1)
demming_numerator = tf.abs(tf.subtract(y_target, tf.add(tf.matmul(x_data, A), b)))
demming_denominator = tf.sqrt(tf.add(tf.square(A), 1))
loss = tf.reduce_mean(tf.divide(demming_numerator, demming_denominator))

# 3. 现在初始化变量，声明优化器，遍历迭代训练集以得到参数
init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(0.1)
train_step = my_opt.minimize(loss)
loss_vec = []
for i in range(250):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    if (i+1)%50 == 0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

    """
    Step #50 A = [[3.5363586]] b = [[0.7682902]]
    Loss = 0.45738918
    Step #100 A = [[3.6291955]] b = [[0.9980916]]
    Loss = 0.45340508
    Step #150 A = [[3.5347779]] b = [[1.0987846]]
    Loss = 0.5320002
    Step #200 A = [[3.4473722]] b = [[1.213543]]
    Loss = 0.51078707
    Step #250 A = [[3.3606324]] b = [[1.3318269]]
    Loss = 0.40380427
    """

# 4. 绘制输出结果
[slope] = sess.run(A)
[y_intercept] = sess.run(b)
best_fit = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Peal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Speal Length')
plt.show()

"""
注意，这里戴明回归算法的实现类型是总体回归（总的最小二乘法误差）。总体回归算法是假设x值和y值的误差是相似的。我们也可以根据不同的理念使用不同的误差来扩展x轴和y轴的距离计算。
"""