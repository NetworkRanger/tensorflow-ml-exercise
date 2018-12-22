#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/22 下午8:30

# 11.6 用TensorFlow求解常微分方程问题


# 1. 导入必要的编程库，创建一个计算图会话
import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.Session()

# 2. 声明计算图中常量和变量
x_initial = tf.constant(1.0)
y_initial = tf.constant(1.0)
X_t1 = tf.Variable(x_initial)
Y_t1 = tf.Variable(y_initial)
# Make the placeholders
t_delta = tf.placeholder(tf.float32, shape=())
a = tf.placeholder(tf.float32, shape=())
b = tf.placeholder(tf.float32, shape=())
c = tf.placeholder(tf.float32, shape=())
d = tf.placeholder(tf.float32, shape=())

# 3. 实现前面介绍的离散系统，然后更新X和Y的数量
X_t2 = X_t1 + (a * X_t1 + b * X_t1 * Y_t1) * t_delta
Y_t2 = Y_t1 + (c * Y_t1 + d * X_t1 * Y_t1) * t_delta
# Update to New Population
step = tf.group(X_t1.assign(X_t2), Y_t1.assign(Y_t2))

# 4. 初始化计算图，运行离散常微分方程系统展示周期性的行为
init = tf.global_variables_initializer()
sess.run(init)
# Run the ODE
prey_values = []
predator_values = []
for i in range(1000):
    # Step simulation (using constants for a known cyclic solution)
    step.run({a: (2./3.), b: {-4./3}, c: -1.0, d: 1.0, t_delta: 0.01}, session=sess)
    # Store each outcome
    temp_prey, temp_pred = sess.run([X_t1, Y_t1])
    prey_values.append(temp_prey)
    predator_values.append(temp_pred)

"""
获得洛特卡-沃尔泰拉方程的稳定求解与指定参数和起始值有较大关系。我们鼓励读者尝试不同的参数和初始值来看会发生什么
"""

# 5. 现在绘制掠食者与猎物的值
plt.plot(prey_values, label='Prey')
plt.plot(predator_values, label='Predator')
plt.legend(loc='upper right')
plt.show()