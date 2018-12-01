#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/1 下午6:41

# 5.4 用TensorFlow实现混合距离计算

# 1. 导入必要的编程库，创建一个计算图会话
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
sess = tf.Session()

# 2. 加载数据集，存储为numpy数组。再次提醒，我们只使用某些列来预测，不使用id变量或者方差非常小的变量
housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
num_features = len(cols_used)
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]
y_vals = np.transpose([np.array([np.array([y[13] for y in housing_data])])])
x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])

# 3. 用min-max缩放法缩放x_vals值到0和1之间
x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

# 4. 创建对角权重矩阵，该矩阵提供归一化的距离度量，其值为特征的标准差
weight_diagonal = x_vals.std(0)
weight_matrix = tf.cast(tf.diag(weight_diagonal), dtype=tf.float32)

# 5. 分割数据集为训练集和测试集。声明k值，该值为最近邻域的数量。设置批量大小为测试集大小
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
k = 4
batch_size = len(x_vals_test)

# 6. 声明所需的占位符。占位符有四个，分别是训练集和测试集的x值输入和y目标输入
x_data_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
y_target_train = tf.placeholder(shape=[None,1], dtype=tf.float32)
y_target_test = tf.placeholder(shape=[None,1], dtype=tf.float32)

# 7. 声明距离函数。为了使可读性更好，我们将距离函数分解。注意：本例需要tf.tile函数为权重矩阵指定batch_size维度扩展。使用matmul()函数进行批量矩阵乘法
subtraction_term = tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))
first_product = tf.matmul(subtraction_term, tf.tile(tf.expand_dims(weight_matrix,0), [batch_size,1,1]))
second_product = tf.matmul(first_product, tf.transpose(subtraction_term, perm=[0,2,1]))
distance = tf.sqrt(tf.matrix_diag_part(second_product))

# 8. 计算完每个测试数据点的距离，需要返回k-NN法的前k个最近邻域（使用tf.nn.top_k()函数)。因为tf.nn.top_k()函数返回最大值，而我们需要的是最小距离，所以转换成返回距离负值的最大值。然后将前k个最近邻域的距离进行加权平均做预测
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1), 1)
x_sums_repeated = tf.matmul(x_sums, tf.ones([1,k], tf.float32))
x_val_weights = tf.expand_dims(tf.div(top_k_xvals, x_sums_repeated), 1)
top_k_yvals = tf.gather(y_target_train, top_k_indices)
prediction = tf.squeeze(tf.matmul(x_val_weights, top_k_yvals), squeeze_dims=[1])

# 9. 计算预测值的MSE，评估训练模型
mse = tf.div(tf.reduce_sum(tf.square(tf.subtract(prediction, y_target_test))), batch_size)

# 10. 遍历迭代训练批量测试数据，每次迭代计算其MSE
num_loops = int(np.ceil(len(x_vals_test)/batch_size))
for i in range(num_loops):
    min_index = i*batch_size
    max_index = min((i+1)*batch_size, len(x_vals_train))
    x_batch = x_vals_test[min_index:max_index]
    y_batch = y_vals_test[min_index:max_index]
    predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_vals_test})
    batch_mse = sess.run(mse, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})
    print('Batch #' + str(i+1) + 'MSE: ' + str(np.round(batch_mse, 3)))

# 11. 输出结果如下
"""
Batch #1 MSe: 21.322
"""

# 12.为了最终对比，我们绘制测试数据集的房价分布和测试集上的预测值分布
bins = np.linspace(5, 50, 45)
plt.hist(predictions, bins, alpha=0.5, label='prediction')
plt.hist(y_batch, bins, alpha=0.5, label='Acutal')
plt.title('Histogram of Predicted and Actual Values')
plt.xlabel('Med Home Value in $1,000s')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()
