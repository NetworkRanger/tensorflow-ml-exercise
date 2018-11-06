#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/5 下午9:40

# 4.2 线性支持向量机的使用

# 1. 导入必要的编程库，包括导入scikit learn的datasets库来访问iris数据集
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

"""
安装scikit learn可使用: $ pip install -U scikit-learn。注意，也可以使用Anaconda来安装
"""

# 2. 创建一个计算图会话，加载需要的数据集。注意，加载iris数据集的第一列和第四列特征变量，其为花萼长度和花萼宽度。加载目标变量时，山鸢尾花为1，否则为-1
sess = tf.Session()

iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])

# 3. 分割数据集为训练集和测试集
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# 4. 设置批量大小、占位符和模型变量
batch_size = 100

x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[2,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# 5. 声明模型输出
model_output = tf.subtract(tf.matmul(x_data, A), b)

# 6. 声明最大间隔损失函数
l2_norm = tf.reduce_mean(tf.square(A))

alpha = tf.constant([0.1])

classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))

loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

# 7. 声明预测函数和准确函数，用来评估训练集和测试训练的准确度
prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

# 8. 声明优化器函数，并初始化模型变量
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

init = tf.initialize_all_variables()
sess.run(init)

# 9. 开始遍历迭代训练模型，记录训练集和测试集的损失和准确度
loss_vec = []
train_accuracy = []
test_accuracy = []
for i in range(500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_accuracy.append(train_acc_temp)

    test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_accuracy.append(test_acc_temp)

    if (i+1)%100 == 0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

# 10. 训练过程中前面脚本的输出结果如下
"""
Step #100 A = [[-0.14051871]
 [-0.14030194]] b = [[-0.09554245]]
Loss = [0.566379]
Step #200 A = [[-0.09420998]
 [-0.38695714]] b = [[-0.13814245]]
Loss = [0.4527266]
Step #300 A = [[-0.05266156]
 [-0.5964568 ]] b = [[-0.18274252]]
Loss = [0.48020816]
Step #400 A = [[-0.00550152]
 [-0.7935583 ]] b = [[-0.23534255]]
Loss = [0.41857398]
Step #500 A = [[ 0.01591796]
 [-0.96598893]] b = [[-0.2834425]]
Loss = [0.36375013]
"""

# 11. 为了绘制输出结果，需要抽取系数，分割x_vals为山鸢尾花(I. setosa)和非山鸢尾花(non-I. setosa)
[[a1], [a2]] = sess.run(A)
[[b]] = sess.run(b)
slope = -a2/a1
y_intercept = b/a1

x1_vals = [d[1] for d in x_vals]

best_fit = []
for i in x1_vals:
    best_fit.append(slope*i+y_intercept)

setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i] == 1]
not_setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i] == -1]

# 12. 下面是缎绘制数据的线性分类器、准度度和损失图
plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.ylim([0,10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

plt.plot(train_accuracy, 'k-', label='Training Accuracy')
plt.plot(test_accuracy, 'r--', label='Test Accuracy')
plt.title('Train and Test Set Accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

"""
使用TensorFlow 实现SVD算法可能导致每次运行的结果不尽相同。原因包括训练集和测试集的随机分割，每批训练的批量大小不同，在理想情况下每次迭代后学习率缓慢减小。
"""