#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/20 下午10:18

# 9. 用TensorFlow实现RNN模型进行垃圾短信预测

# 1. 导入必要的编程库
import os
import re
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile

# 2. 开始计算图会话，并设置RNN模型参数。训练数据20个epoch，批量大小为250。短信最大长度为25个单词，超过的部分会被截取掉，不够的部分用0填充。RNN模型由10个单元组成。我们仅仅处理词频超过10的单词，每个单词会嵌套在长度为50的词向量中。dropout概率为占位符，训练模型时设为0.5，评估模型时设为1.0。
sess = tf.Session()
epochs = 20
batch_size = 10
max_sequence_length = 25
rnn_size = 10
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005
dropout_keep_prob = tf.placeholder(tf.float32)

# 3. 获取SMS文本数据集。首先，在下载文本数据集前检查是否已下载过。如果已经下载过数据集，直接从文件中读取
data_dir = 'temp'
data_file = 'text_data.txt'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.isfile(os.path.join(data_dir, data_file)):
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    # Format Data
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    # Save data to text file
    with open(os.path.join(data_dir, data_file), 'w') as file_conn:
        for text in text_data:
            file_conn.write('{}\n'.format(text))
else:
    # Open data from text file
    text_data = []
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        for row in file_conn:
            text_data.append(row)
    text_data = text_data[:-1]
text_data = [x.split('\t') for x in text_data if len(x) >= 1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]

# 4. 我们将清洗文本数据集，移除特殊字符，将所有文本转为小写，以空格提取单词
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = ' '.join(text_string.split())
    text_string = text_string.lower()
    return (text_string)

"""
注意从文本数据中清洗移除特殊字符的步骤，有时可以用空格替换该特殊字符。在理想情况下，需要根据数据集的格式选择具体的方法处理。
"""

# 5. 使用TensorFlow内建的词汇处理器处理文本。该步骤将文本转换为索引列表
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length, min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transorm(text_data_train)))

# 6. 随机shuffle文本数据集
text_processed = np.array(text_processed)
text_data_target = np.array([1 if x=='ham' else 0 for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

# 7. 分割数据集为80-20的训练-测试数据集
ix_cutoff = int(len(y_shuffled)*0.80)
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)
print('Vocabulary Size: {:d}'.format(vocab_size))
print('80-20 Train Test split: {:d} -- {:d}'.format(len(y_train), len(y_test)))

"""
本次我们不准备做超参数调优。如果读者有这方面的需求，请在预处理前将数据集分割为训练集-测试集-验证集。scikit-learn的model_selection.train_test_split()函数可以随机分割(划分)训练集和测试集。
"""

# 8. 声明计算图的占位符。输入数据x_data是形状为[None, max_sequence_length]的占位符，其以短信最大允许的长度为批量大小。输出结果y_output的占位符为整数0或者1，即正常短信或者垃圾短信
x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])

# 9. 创建输入数据x_data的嵌套矩阵和嵌套查找操作
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)

# 10. 声明算法模型。首先，初始化RNN单元的类型，大小为10.然后通过动态RNN函数tf.nn.dynamic_rnn()创建RNN序列，接着增加dropout操作
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_size)
output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
output = tf.nn.dropout(output, dropout_keep_prob)

"""
注意，动态RNN允许变长序列。即使本例所使用的是固定长度的序列，我们也推荐使用TensorFlow的tf.nn.dynamic_rnn()的函数。主要原因是：实践证明动态RNN实际计算更快，并且允许RNN中运行不同长度的序列。
"""

# 11. 为了进行预测，转置并重新排列RNN的输出结果，剪切最后的输出结果
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0])-1)

# 12. 为了完成RNN预测，我们通过全连接层将rnn_size大小的输出转换为二分类输出
weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))
logits_out = tf.nn.softmax(tf.matmul(last, weight)+bias)

# 13. 声明损失函数。本例使用TensorFlow和spare_softmax函数，目标值是int型索引，logits是float型
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_out, y_output)
loss = tf.reduce_mean(losses)

# 14. 创建准确度函数，比较训练集和测试集的训练结果
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32))

# 15. 创建优化器函数，初始化模型变量
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

# 16. 开始遍历迭代训练模型。遍历数据集多次，最佳实践表明：每个epoch都需要随机shuffle数据，避免过拟合
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
# Start training
for epoch in range(epochs):
    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffled_ix]
    y_train = y_train[shuffled_ix]
    num_batches = int(len(x_train)/batch_size)+1
    for i in range(num_batches):
        # Select train data
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i+1)*batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]

        # Run train step
        train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob: 0.5}
        sess.run(train_step, feed_dict=train_dict)

    # Run train step
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)

    # Run Eval Step
    test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob: 1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))

# 17. 输出结果如下
"""

"""

# 18. 绘制训练集、测试集损失和准确度的代码如下
epoch_seq = np.arange(1, epochs+1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()
# Plot accuracy over time
plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()




