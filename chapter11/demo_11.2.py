#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/22 下午6:39

# 11.2 TensorFlow可视化: Tensorboard

# 1. 导入必要的编程库
import os
import io
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 2. 初始化计算图会话，创建summary-writer将Tensorboard summary写入Tensorboard文件夹
sess = tf.Session()
# Create a visualizer object
summary_writer = tf.train.SummaryWriter('tensorboard', tf.get_default_graph())

# 3. 确保summary_writer写入的Tensorboardyyywr夹存在
if not os.path.exists('tensorboard'):
    os.makedirs('tensorboard')

# 4. 设置模型参数，为模型生成线性数据集。注意，设置真实斜率true_slope为2(注：迭代训练时，我们将随着时间的变化可视化斜率，起到取到真实斜率值)
batch_size = 50
generations = 100
# Create sample input data
x_data = np.arange(1000)/10.
true_slope = 2.
y_data = x_data * true_slope + np.random.normal(loc=0.0, scale=25, size=1000)

# 5. 分割数据集为测试集和训练集
train_ix = np.random.choice(len(x_data), size=int(len(x_data)*0.9), replace=False)
test_ix = np.setdiff1d(np.arange(1000), train_ix)
x_data_train, y_data_train = x_data[train_ix], y_data[train_ix]
x_data_test, y_data_test = x_data[test_ix], y_data[test_ix]

# 6. 创建占位符、变量、模型操作、损失和优化器操作
x_graph_input = tf.placeholder(tf.float32, [None])
y_graph_input = tf.placeholder(tf.float32, [None])
# Declare model vairables
m = tf.Variable(tf.random_normal([1], dtype=tf.float32), name='Slope')
# Declare model
output = tf.multiply(m, x_graph_input, name='Batch_Multiplication')
# Declare loss function (L1)
residuals = output - y_graph_input
l2_loss = tf.reduce_mean(tf.abs(residuals), name='L2_Loss')
# Declare optimization function
my_optim = tf.train.GradientDescentOptimizer(0.01)
train_step = my_optim.minimize(l2_loss)

# 7. 创建Tensorboard操作汇总一个标量值。该汇总的标量值为模型的斜率估计
with tf.name_scope('Slope_Estimate'):
    tf.scalar_summary('Slope_Estimate', tf.sequeeze(m))

# 8. 添加到Tensorboard的别一个汇总数据是直方图汇总。该直方图汇总。该直方图汇总输入疑是，输出曲线图和直方图
with tf.name_scope('Loss_and_Residuals'):
    tf.histogram_summary('Histogram_Erros', l2_loss)
    tf.histogram_summary('Histogram_Residuals', residuals)

# 9. 创建完这些汇总操作，我们创建汇总合并操作综合所有的汇总数据，然后初始化模型变量
summary_op = tf.merge_all_sumaries()
# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# 10. 现在训练线性回归模型，将每次迭代训练写入汇总数据
for i in range(generations):
    batch_indices = np.random.choice(len(x_data_train), size=batch_size)
    x_batch = x_data_train[batch_indices]
    y_batch = y_data_train[batch_indices]
    _, train_loss, summary = sess.run([train_step, l2_loss, summary_op], feed_dict={x_graph_input: x_batch, y_graph_input: y_batch})

    test_loss, test_resids = sess.run([l2_loss, residuals], feed_dict={x_graph_input: x_data_test, y_graph_input: y_data_test})

    if (i+1)%10 == 0:
        print('Generation {} of {}. Train Loss: {:.3}, Test Loss: {:.3}'.format(i+1, generations, train_loss, test_loss))

    log_writer = tf.train.SummaryWriter('tensorbaord')
    log_writer.add_sumary(summary, i)

# 11. 为了可视化数据点拟合的线性回归模型，我们创建protobuff格式的图形。开始之前，我们创建函数输出protobuff格式的图形
def gen_linear_plot(slope):
    linear_prediction = x_data* slope
    plt.plot(x_data, y_data, 'b.', label='data')
    plt.plot(x_data, linear_prediction, 'r-', linewidth=3, label='predicted line')
    plt.legend(loc='upper left')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return (buf)

# 12. 创建并且将protobuf格式的图形增加到Tensorboard
slope = sess.run(m)
plot_buf = gen_linear_plot(slope[0])
# Convert PNG buffer to TF image
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
# Add the batch dimension
image = tf.expand_dims(image, 0)
# Add image summary
image_summary_op = tf.image_summary('Linear Plot', image)
image_summary = sess.run(image_summary_op)
log_writer.add_sumary(image_summary, i)
log_writer.close()

"""
注意Tensorboard写图形汇总太频繁。例如，如果我们迭代训练10 000次每次都写入汇总数据，那将生成10 000幅图。这会迅速吃掉磁盘空间
"""

