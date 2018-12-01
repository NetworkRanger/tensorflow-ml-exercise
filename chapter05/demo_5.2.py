#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/11/6 下午11:27

# 5.2 最近邻域法的使用

# 1. 导入必要的编辑库，创建一个计算图会话
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests

sess = tf.Session()

# 2. 使用requests 模块加载数据集
housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
num_features = len(cols_used)

# Request data
housing_file = requests.get(housing_url)

# Parse data
housing_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in housing_file.text.split('\n') if len(y) >= 1]

# 3. 分离数据集为特征依赖的数据集和特征无关的数据集。
y_vals = np.transpose([np.array([y[13] for y in housing_data])])
x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])

x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

# 4. 分离x_vals值和y_vals值为训练数据集和测试数据集。随机选择80%的行作为训练集，剩下的20%数据行作为测试集
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# 5. 声明k值和批量大小
k = 4
batch_size = len(x_vals_test)

# 6. 声明占位符。
x_data_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 7. 为批量测试集创建距离函数，这里使用L1范数距离
distance = tf.reduce_sum(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1)), reduction_indices=2)

"""
注意，L2范数距离函数也经常使用，代码为:
distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), reduction_indices=1))
"""

# 8. 创建预测函数。使用top_k()函数，其以张量的方式返回最大值的值和索引。因为需要找到最小距离的索引，所以将对最大距离取负。声明预测函数和目标值的均方误差(MSE)
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1), 1)
x_sums_repeated = tf.matmul(x_sums, tf.ones([1, k], tf.float32))
x_val_weights = tf.expand_dims(tf.div(top_k_xvals, x_sums_repeated), 1)

top_k_yvals = tf.gather(y_target_train, top_k_indices)
prediction = tf.squeeze(tf.matmul(x_val_weights, top_k_yvals), squeeze_dims=[1])
mse = tf.div(tf.reduce_sum(tf.square(tf.subtract(prediction, y_target_test))), batch_size)

# 9. 进行测试
num_loops = int(np.ceil(len(x_vals_test)/batch_size))

for i in range(num_loops):
    min_index = i*batch_size
    max_index = min((i+1)*batch_size, len(x_vals_train))
    x_batch = x_vals_test[min_index:max_index]
    y_batch = y_vals_test[min_index:max_index]
    predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})
    batch_mse = sess.run(mse, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})
    print('Batch #' + str(i+1) + 'MSE: ' + str(np.round(batch_mse, 3)))

"""
Batch #1 MSE: 23.153
"""

# 10. 下面通过直方图来比较实际值和预测值。使用的是平均方法，所以在预测目标值最大和最小极值时遇到问题
bins = np.linspace(5, 50, 45)

plt.hist(predictions, bins, alpha=0.5, label='Prediction')
plt.hist(y_batch, bins, alpha=0.5, label='Actual')
plt.title('Histogram of Prediction and Actual Values')
plt.xlabel('Med Home Values in $1,000s')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()


















