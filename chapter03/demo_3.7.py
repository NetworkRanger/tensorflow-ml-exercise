#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/4 下午3:11

# 3.7 用TensorFlow 实现lasso 回归和岭回归算法

# 1. 这次还是使用iris数据集，导入必要的编程库，创建一个计算图会话，加载数据集，声明批量大小，创建占位符，变量和模型输出
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
sess = tf.Session()
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])
batch_size = 50
learning_rate = 0.001
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
model_output = tf.add(tf.matmul(x_data, A), b)

# 2. 增加损失函数，其为改良过的连续阶跃函数，lasso回归的截止点设为0.9。这里意味着斜率系统不超过0.9
lasso_param = tf.constant(0.9)
heavyside_step = tf.divide(1., tf.add(1., tf.exp(tf.multiply(-100., tf.subtract(A, lasso_param)))))
regulariztion_param = tf.multiply(heavyside_step, 99.)
loss = tf.add(tf.reduce_mean(tf.square(y_target - model_output)), regulariztion_param)

# 3. 初始化变量和声明优化器
init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# 4. 遍历迭代支行一段时间，因为需要一会才会收敛。最后结果显示斜率系数小于0.9
loss_vec = []
for i in range(1500):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss[0])
    if (i+1)%300 == 0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

    """
    Step #300 A = [[0.8268108]] b = [[2.4417148]]
    Loss = [[5.914397]]
    Step #600 A = [[0.8190999]] b = [[3.5311697]]
    Loss = [[2.1670737]]
    Step #900 A = [[0.81321365]] b = [[4.12967]]
    Loss = [[0.7221513]]
    Step #1200 A = [[0.8087082]] b = [[4.465177]]
    Loss = [[0.4463676]]
    Step #1500 A = [[0.80388826]] b = [[4.648514]]
    Loss = [[0.33279842]]
    """

# 5. 绘制输出结果
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