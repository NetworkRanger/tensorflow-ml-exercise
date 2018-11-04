#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/4 下午12:04

# 2.8 TensorFlow 实现创建张量

# 1. 导入相应的工具库，初始化计算图
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
sess = tf.Session()

# 2. 导入iris数据集，根据目标数据是否为山鸢尾将其转换成1或者0。由于iris数据集将山鸢尾标记为0，我们将其从0置为1，同时把其他物种标记为0
iris = datasets.load_iris()
binary_target = np.array([1. if x==0 else 0. for x in iris.target])
iris_2d = np.array([[x[2], x[3]] for x in iris.data])

# 3. 声明变量训练大小，数据占位符和模型变量
batch_size = 20
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 4. 定义线性模型
my_mult = tf.matmul(x2_data, A)
my_add = tf.add(my_mult, b)
my_output = tf.subtract(x1_data, my_add)

# 5. 增加TensorFlow 的 sigmoid 交叉熵损失函数 sigmoid_cross_entropy_with_logits()
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target)

# 6. 声明优化器方法，最小化交叉熵损失
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

# 7. 创建一个变量初始化操作，然后让TensorFlow执行它
init = tf.initialize_all_variables()
sess.run(init)

# 8. 现在迭代100次训练线性模型
for i in range(1000):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})
    if (i+1)%200 == 0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))

    """
    Step #200 A = [[8.574331]], b = [[-3.5213127]]
    Step #400 A = [[10.120312]], b = [[-4.6367807]]
    Step #600 A = [[11.085849]], b = [[-5.303544]]
    Step #800 A = [[11.831396]], b = [[-5.835288]]
    Step #1000 A = [[12.395876]], b = [[-6.260936]]
    """

# 9. 下面的命令抽取模型变量并绘图
[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)
x = np.linspace(0, 3, num=50)
ablineValues = []
for i in x:
    ablineValues.append(slope+intercept)

setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i] == 1]
setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i] == 1]
non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i] == 0]
non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i] == 0]
plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label='Non-setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 2.7])
plt.suptitle('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()