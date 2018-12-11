#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/10 下午9:30

# 7.3 用TensorFlow实现TF-IDF算法

# 1. 导入必要的编程库。本例中会导入scikit-learn的TF-IDF处理模块处理文本数据集
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import string
import requests
import io
import nltk
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer
# 2. 创建一个计算图会话, 声明批量大小和词汇的最大长度
sess = tf.Session()
batch_size = 200
max_features = 1000

# 3. 加载文本数据集。可以从网站下载或者从上次保存的temp文件夹加载
save_file_name = os.path.join('temp', 'temp_spam_data.csv')
if os.path.isfile(save_file_name):
    text_data = []
    with open(save_file_name, 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
            text_data.append(row)
else:
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    # Format Data
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x) >= 1]

    # And write to csv
    with open(save_file_name, 'w') as temp_output_file:
        writer = csv.writer(temp_output_file)
        writer.writerows(text_data)

texts = [x[1] for x in text_data]
target = [x[0] for x in text_data]
# Relabel 'spam' as 1, 'ham' as 0
target = [1 if x == 'spam' else 0 for x in target]

# 4. 声明词汇大小。本例中也会将所有字符转换成小写，剔除标点符号和数字
# Lower case
texts = [x.lower() for x in texts]
# Remove punctuation
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
# Remove numbers
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
# Trim extra whitespace
texts = [' '.join(x.split()) for x in texts]

# 5. 为了使用scikit-learn的TF—IDF处理函数，我们需要输入切分好的语句(即将句子切分为相关的单词）。nltk包可以提供非常棒的分词器来实现分词功能
def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words
# Create TF-IDF of texts
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=max_features)
sparse_tfidf_texts = tfidf.fit_transform(texts)

# 6. 分割数据集为训练集和测试集
train_indices = np.random.choice(sparse_tfidf_texts.shape[0], round(0.8*sparse_tfidf_texts.shape[0]), replace=False)
test_indices = np.array(list(set(range(sparse_tfidf_texts.shape[0])) - set(train_indices)))
texts_train = sparse_tfidf_texts[train_indices]
texts_test = sparse_tfidf_texts[test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# 7. 声明逻辑回归模型的变量和数据集的占位符
A = tf.Variable(tf.random_normal(shape=[max_features, 1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
# Initialize placeholders
x_data = tf.placeholder(shape=[None, max_features], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 8. 声明算法模型操作和损失函数。注意，逻辑回归算法的sigmoid部分是在损失函数中实现的
model_output = tf.add(tf.matmul(x_data, A), b)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=model_output, logits=y_target))

# 9. 为计算图增加预测和准确度函数(可以让我们看到模型训练中训练集和测试集的准确度)
prediction = tf.round(tf.sigmoid(model_output))
predictions_corrent = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_corrent)

# 10. 声明优化器，初始化计算图中的变量
my_opt = tf.train.GradientDescentOptimizer(0.0025)
train_step = my_opt.minimize(loss)
# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# 11. 遍历迭代训练模型10000次，记录测试集和训练集损失，以及每迭代100次的准确度，然后每迭代500次就打印状态信息
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    rand_index = np.random.choice(texts_train.shape[0], size=batch_size)
    rand_x = texts_train[rand_index].todense()
    rand_y = np.transpose([target_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    # Only record loss and accuracy every 100 generations
    if (i+1)%100 == 0:
        i_data.append(i+1)
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)

        test_loss_temp =  sess.run(loss, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_acc.append(test_acc_temp)

    if (i+1)%500 == 0:
        acc_and_loss = [i+1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

# 12. 输出结果如下
"""
Generation # 500. Train Loss (Test Loss): 0.66 (0.63). Train Acc (Test Acc): 0.21 (0.24)
Generation # 1000. Train Loss (Test Loss): 0.59 (0.61). Train Acc (Test Acc): 0.18 (0.22)
Generation # 9500. Train Loss (Test Loss): 0.14 (0.23). Train Acc (Test Acc): 0.17 (0.13)
Generation # 10000. Train Loss (Test Loss): 0.24 (0.20). Train Acc (Test Acc): 0.12 (0.13)
"""

# 13. 绘制训练集和测试的准确度的损失函数


