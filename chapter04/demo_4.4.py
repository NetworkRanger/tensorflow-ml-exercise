#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/5 下午10:44

# 4.4 TensorFlow上核函数的使用

# 1. 导入必要编程库，创建一个计算图会话
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
sess = tf.Session()

# 2. 生成模拟数。生成的数据是一两个同心圆数据，每个不同的环代表不同的类，确保只有类-1或者1。为了让绘图方便，这里将每类数据分成x值和y值
(x_vals, y_vals) = datasets.make_circles(n_samples=500, factor=.5, noise=.1)
y_vals = np.array([1 if y==1 else -1 for y in y_vals])
class1_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i] == 1]
class1_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i] == 1]
class2_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i] == -1]
class2_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i] == -1]

# 3. 声明批量大小、占位符，创建模型变量b。对于SVM算法，为了让每次迭代训练不波动，得到一个稳定的训练模型，这时批量大小得取更大。注意，本例为预测数据点声明有额外的占位符。最后创建彩色的网格来可视化不同的区域代表不同的类别
batch_size = 250
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[1, batch_size]))

# 4. 创建高斯核函数。该核函数用矩阵操作来表示
gamma = tf.constant(-50.0)
dist = tf.reduce_mean(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1, 1])
sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

"""
注意，在sq_dists中应用广播加法和减法操作。线性核函数可以表示为: my_kernel = tf.matmul(x_data, tf.transpose(x_data))。
"""

# 5. 声明对偶问题。为了最大化，这里采用最小化损失函数的负数: tf.negative()
model_output = tf.matmul(b, my_kernel)
first_term = tf.reduce_mean(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
second_term = tf.reduce_mean(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
loss = tf.negative(tf.subtract(first_term, second_term))

# 6. 创建预测函数和准确度函数。先创建一个预测核函数，但用预测数据点的核函数用模拟数据点的核函数。预测值是模型输出的符号函数值
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), b), pred_kernel)
prediction = tf.sign(prediction_output-tf.reduce_mean(prediction_output))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

"""
为了实现线性预测核函数，将预测核函数改为: pred_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))。
"""

# 7. 创建优化器函数，初始化所有的变量
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)

# 8. 开始迭代训练
loss_vec = []
batch_accuracy = []
for i in range(500):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})
    batch_accuracy.append(acc_temp)

    if (i+1) % 100 == 0:
        print('Step #' + str(i+1))
        print('Loss = ' + str(temp_loss))


# 9. 输出结果如下

"""
Step #100
Loss = -0.040872738
Step #200
Loss = -0.04066868
Step #300
Loss = -0.04294016
Step #400
Loss = -0.042239938
Step #500
Loss = -0.043024104
"""

# 10. 为了能够在整个数据空间可视化分类返回结果，我们将创建预测数据点的网格，在其上进行预测
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]
[grid_prediction] = sess.run(prediction, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: grid_points})
grid_prediction = grid_prediction.reshape(xx.shape)

# 11. 下面绘制预测结果、批量准确度和损失函数
plt.contourf(xx, yy, grid_prediction, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='Class 1')
plt.plot(class2_x, class2_y, 'kx', label='Class -1')
plt.legend(loc='lower right')
plt.ylim([-1.5, 1.5])
plt.xlim([-1.5, 1.5])
plt.show()

plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

# 12. 简单扼要，这里只显示训练结果图，不过也可以分开运行绘图代码展示其他效果
