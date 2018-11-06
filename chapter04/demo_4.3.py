#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/5 下午10:13

# 4.3 弱化为线性回归

# 1. 导入必要的编程库，创建一个计算图会话，加载iris数据集。然后分割数据集为训练集和测试集，并且可视化相应的损失函数
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
sess = tf.Session()
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

"""
对于这个例子，我们分割数据集为训练集和测试集。有时也经常分割为三个数据集，还包括验证集。我们用验证集验证训练过的模型是否过拟合。
"""

# 2. 声明批量大小、占位符和变量，创建线性模型
batch_size = 50

x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

model_output = tf.add(tf.matmul(x_data, A), b)

# 3. 声明损失函数。该损失函数如前所述，实现时ε = 0.5。注意，ε 是损失函数的一部分，其允许soft margin代替为hard margin
epsilon = tf.constant([0.5])
loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_output, y_target)), epsilon)))

# 4. 创建一个优化器，初始化变量
my_opt = tf.train.GradientDescentOptimizer(0.075)
train_step = my_opt.minimize(loss)

init = tf.initialize_all_variables()
sess.run(init)

# 5. 现在开始200次迭代训练，保存训练集和测试集损失函数，后续用来绘图
train_loss = []
test_loss = []
for i in range(200):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_train_loss = sess.run(loss,
        feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
    train_loss.append(temp_train_loss)

    temp_test_loss = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
    test_loss.append(temp_test_loss)

    if (i+1)%50 == 0:
        print('------------')
        print('Generation: ' + str(i))
        print('A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Train Loss = ' + str(temp_train_loss))
        print('Test Loss = ' + str(temp_test_loss))

# 6. 下面是迭代训练输出结果
"""
------------
Generation: 49
A = [[2.057484]] b = [[2.754403]]
Train Loss = 0.5301603
Test Loss = 0.6597586
------------
Generation: 99
A = [[1.6298344]] b = [[3.718903]]
Train Loss = 0.22433636
Test Loss = 0.29391733
------------
Generation: 149
A = [[1.2507848]] b = [[4.282903]]
Train Loss = 0.09597528
Test Loss = 0.15989089
------------
Generation: 199
A = [[1.1154844]] b = [[4.4914026]]
Train Loss = 0.0787632
Test Loss = 0.13063176
"""

# 7. 现在抽取系数，获取最佳拟合直线的截距。为了后续画图，这里也获取间隔宽度值
[[slope]] = sess.run(A)
[[y_intercept]] = sess.run(b)
[width] = sess.run(epsilon)

best_fit = []
best_fit_upper = []
best_fit_lower = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)
    best_fit_upper.append(slope*i+y_intercept+width)
    best_fit_lower.append(slope*i+y_intercept-width)

# 8. 最后，绘制数据点和拟合直线，以及训练集和测试集损失
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='SVM Regression Line', linewidth=3)
plt.plot(x_vals, best_fit_upper, 'r--', linewidth=2)
plt.plot(x_vals, best_fit_lower, 'r--', linewidth=2)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Speal Length')
plt.show()
plt.plot(train_loss, 'k--', label='Train Set Loss')
plt.plot(test_loss, 'r--', label='Test Set Loss')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.legend(loc='upper right')
plt.show()





