#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/11 下午9:55

# 1. 导入必要的编辑和计算图会话
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pickle
import string
import requests
import collections
import io
import tarfile
import urllib.request
from . import text_helpers
from nltk.corpus import stopwords
sess = tf.Session()

# 2. 声明算法模型参数。注意，选择与前一节中CBOW方法相同的嵌套大小
embedding_size = 200
vocabulary_size = 2000
batch_size = 100
max_wowrds = 100
stops = stopwords.words('english')

# 3. 用text_helpers.py脚本加载和转换文本数据集
data_folder_name = 'temp'
texts, target = text_helpers.load_movie_data(data_folder_name)
# Normalize text
print('Normalizing Text Data')
texts = text_helpers.normalize_text(texts, stops)
# Texts must contain at least 3 words
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]

train_indices = np.random.choice(len(target), round(0.8*len(target)), replace=False)
test_indices = np.array(list(set(range(len(target))) - set(train_indices)))
texts_train = [x for ix, x in enumerate(texts) if ix in test_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# 4. 现在加载CBOW嵌套中保存的单词字典，这使得我们拥有相同的单词到嵌套索引的映射
dict_file = os.path.join(data_folder_name, 'movie_vocab.pkl')
word_dictionary = pickle.load(open(dict_file, 'rb'))

# 5. 通过单词字典将加载的句子转换为数值型numpy数组
text_data_train = np.array(text_helpers.text_to_numbers(texts_train, word_dictionary))
text_data_test = np.array(text_helpers.text_to_numbers(texts_test, word_dictionary))

# 6. 由于电影影评长度不一，我们用同一长度（设为100个单词长度）将其标准化。如果电影影评长度少于100个单词，我们将用0去填充
text_data_train = np.array([x[0:max_wowrds] for x in [y+[0]*max_wowrds] for y in text_data_train])
text_data_test = np.array([x[0:max_wowrds] for x in [y+[0]*max_wowrds] for y in text_data_test])

# 7. 声明逻辑回归的模型变量和占位符
A = tf.Variable(tf.random_normal(shape=[embedding_size, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
# Initialize placeholers
x_data = tf.placeholder(shape=[None, max_wowrds], dtype=tf.int32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 8. 为了使得TensorFlow可以重用训练过的词向量，首先需要给存储方法设置一个变量。这里创建一个嵌套变量，其形状与将要加载的单词嵌套相同
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# 9. 在计算图中加入嵌套查找操作，计算句子中所有单词的平均嵌套
embed = tf.nn.embedding_lookup(embeddings, x_data)
# Take average of all word embeddings in documents
embed_avg = tf.reduce_mean(embed, 1)

# 10. 声明模型操作和损失函数。记住，损失函数中已经内建了sigmoid操作
model_output = tf.add(tf.matmul(embed_avg, A), b)
# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(model_output, y_target))

# 11. 在计算图中增加预测函数和准确函数，评估训练模型的准确度
prediction = tf.round(tf.sigmoid(model_output))
predictions_corrent = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_corrent)

# 12. 声明优化器函数，并初始化下面的模型变量
my_opt = tf.train.AdamOptimizer(0.005)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

# 13. 随机初始化单词嵌套后，调用Saver方法来加载上一节中保存好的CBOW嵌套到嵌套变量
model_checkpoint_path = os.path.join(data_folder_name, 'cbow_movie_embeddings.ckpt')
saver = tf.train.Saver({'embeddings': embeddings})
saver.restore(sess, model_checkpoint_path)

# 14. 开始迭代训练。注意，每迭代100次就保存训练集和测试集的损失和准确度。每迭代500次就打印一次模型状态
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    rand_index = np.random.choice(text_data_train.shape[0], size=batch_size)
    rand_x = text_data_train[rand_index]
    rand_y = np.transpose([target_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    # Only record loss and accuracy every 100 generations
    if (i+1)%100 == 0:
        i_data.append(i+1)
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(loss, feed_dict={x_data: text_data_test, y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)
        test_acc_temp = sess.run(accuracy, feed_dict={x_data: text_data_test, y_target: np.transpose([target_test])})
    if (i+1)%500 == 0:
        acc_and_loss = [i+1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

# 15. 打印结果如下

"""

"""

# 16. 绘制训练集和测试集损失函数和准确度的代码
# Plot loss over time
plt.plot(i_data, train_loss, 'k-', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()
# Plot train and test accuracy
plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()