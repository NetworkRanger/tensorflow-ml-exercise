#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/8 下午4:44

# 6.6 用TensorFlow实现多层神经网络

# 1. 导入必要的编程库
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
import numpy as np
sess = tf.Session()

# 2. 使用requests模块从网站加载数据集，然后分离出需要的特征数据和目标值
birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')[5:]
birth_header = [x for x in birth_data[0].split(' ') if len(x) >= 1]
birth_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in birth_data[1:] if len(y)>=1]
y_vals = np.array([x[10] for x in birth_data])
cols_of_interest = ['AGE', 'LMT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UT', 'FTV']
x_vals = np.array([[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data])

# 3. 为了后面可以复现，为NumPy和TensorFlow设置随机种子，然后声明批量大小
seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)
batch_size = 100

# 4. 分割数据集为80-20的训练集和测试集，然后使用min-max方法归一化输入特征数据为0到1之间
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[test_indices]
y_vals_test = y_vals[test_indices]

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

"""
归一化输入特征数据是常用的特征转化方法，对神经网络算法特别有帮助。如果样本数据集是以0到1为中心的，它将有利于激励函数操作的收敛。
"""

# 5. 因为有多个层含有相似的变量初始化，因此我们将创建一个初始化函数，该函数可以初始化加权权重和偏置
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return (weight)

def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stdev =st_dev))
    return (bias)

# 6. 初始化占位符。本例中将有八个输入特征数据和一个输出结果(出生体重，单位：克)
x_data = tf.placeholder(shape=[None, 8], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtypet=tf.float32)

# 7. 全连接层将在三个隐藏层中使用三次，为了避免代码上的重复，我们将创建一个层函数来初始化算法模型
def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return (tf.nn.relu(layer))

# 8. 现在创建算法模型。对于每一层(包括输出层），我们将初始化一个权重矩阵、偏置矩阵和全连接层。在本例中，三个隐藏层的大小分别为25、10和3
"""
本例中使用的算法模型需要拟合522个变量。下面来看这个数值是如何计算的？输入数据和第一隐藏层之间有225(8*25+25)个变量，继续用这种方式计算隐藏层并加在一起有522(255+260+33+4)个变量。很明显，这比之前在逻辑回归算法中的9个变量要多得多。
"""
# Create second layer (25 hidden nodes)
weight_1 = init_weight(shape=[8, 25], st_dev=10.0)
bias_1 = init_bias(shape=[25], st_dev=10.0)
layer_1 = fully_connected(x_data, weight_1, bias_1)

# Create second layer (10 hidden nodes)
weight_2 = init_weight(shape=[25, 10], st_dev=10.0)
bias_2 = init_bias(shape=[10], st_dev=10.0)
layer_2 = fully_connected(layer_1, weight_2, bias_2)

# Create third layer (3 hidden nodes)
weight_3 = init_weight(shape=[10,3], st_dev=10.0)
bias_3 = init_bias(shape=[3], st_dev=10.0)
layer_3 = fully_connected(layer_2, weight_3, bias_3)

# Create output layer (1 output values)
weight_4 = init_weight(shape=[3, 1], st_dev=10.0)
bias_4 = init_bias(shape=[1], st_dev=10.0)
final_output = fully_connected(layer_3, weight_4, bias_4)

# 9. 使用L1范数损失函数(绝对值), 声明优化器(Adam优化器）和初始化变量
loss = tf.reduce_mean(tf.abs(y_target - final_output))
my_opt = tf.train.AdamOptimizer(0.05)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)


"""
本例中为Adam优化器选择的学习率为0.05，有研究建议设置更低的学习率可以产生更好的效果。在本节中，我们使用比较大的学习率是为了数据集的一致性和快速收敛。
"""

# 10. 迭代训练模型200次。下面的代码也包括存储训练损失和测试损失，选择随机批量大小和每25次迭代就打印状态
loss_vec = []
test_loss = []
for i in range(200):
    # Choose random indices for batch selection
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    # Get random batch
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    # Run the training step
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    # Get and store the train loss
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    # Get and store the test loss
    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose(([y_vals_test]))})
    test_loss.append(test_temp_loss)
    if (i+1)%25 == 0:
        print('Generation : ' + str(i+1) + '. Loss = ' + str(temp_loss))

# 11. 输出结果如下
"""

"""

# 12. 使用matplotlib模块绘制训练损失和测试损失代码
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# 13. 现在我们想比较预测出体重结果和前面的逻辑结果。逻辑回归算法中，我们在迭代上千次后得到了大约60%的精确度。为了在这里做比较，我们将输出训练集/测试集和回归结果，然后传入一个指示函数(判断是否大于2500克),将回归结果转换成分类结果
actuals = np.array([x[1] for x in birth_data])
test_acctuals = actuals[test_indices]
train_acctuals= actuals[train_indices]
test_preds = [x[0] for x in sess.run(final_output, feed_dict={x_data: x_vals_test})]
train_preds = [x[0] for x in sess.run(final_output, feed_dict={x_data: x_vals_train})]
test_preds = np.array([1.0 if x<2500.0 else 0.0 for x in test_preds])
train_preds = np.array([1.0 if x<2500.0 else 0.0 for x in train_preds])
# Print out accuracies
test_acc = np.mean([x==y for x,y in zip(test_preds, test_acctuals)])
train_acc = np.mean([x==y for x,y in zip(test_preds, train_acctuals)])
print('On predicting the category of low birthwieght from regression output (<2500g):')
print('Test Accuracy: {}'.format(test_acc))
print('Train Accuracy: {}'.format(train_acc))

# 14. 准确度的结果如下
"""
Test Accuracy: 0.5526315789473685
Train Accuracy: 0.6688741721854304
"""



