#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/9 下午10:44

# 7.2 词袋的使用

# 1. 导入必要的编程库。本例中需要.zip文件库来解压从UCI机器学习数据库中下载的.zip文件
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
import string
import requests
import io
from zipfile import ZipFile
from tensorflow.contrib import learn
sess = tf.Session()

# 2. 为了让脚本运行时不用每次都去下载文件数据，我们将下载文件存储，并检查之间是否保存过。该步骤避免了文本数据的重复下载。下载完文本数据集后，抽取输入数据和目标数据，并调整目标值(垃圾短信(spam)置为1，正常短信(ham)置为0)。
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

# 3. 为了减小词汇量大小，我们对文本进行规则化处理。移除文本中大小写和数字的影响
# Convert to lower case
texts = [x.lower() for x in texts]
# Remove punctuation
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
# Remove numbers
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
# Trim extra whitespace
texts = [' '.join(x.split()) for x in texts]

# 4. 计算最长句子大小。我们使用文本数据集和文本长度直方图,并取最佳截止点
# Plot histogram of text lengths
text_lengths = [len(x.split()) for x in texts]
text_lengths = [x for x in text_lengths if x < 50]
plt.hist(text_lengths, bins=25)
plt.title('Histogram of # of Words in Texts')
sentence_size= 25
min_word_freq = 3

# 5. TensorFlow 自带分词器VocabularyProcessor()，该函数位于learn.preprocessing库
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)
vocab_processor.fit_transform(texts)
embedding_size = len(vocab_processor.vocabulary_)

# 6. 分割数据集为训练集和测试集
train_indices = np.random.choice(len(texts), round(len(texts)*0.8), replace=False)
test_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = [x for ix, x in enumerate(target) if ix in train_indices]
target_test = [x for ix, x in enumerate(target) if ix in test_indices]

# 7. 声明词嵌入矩阵。将句子单词转成索引，再将索引转成one-hot向量，该向量为单位矩阵。我们使用该矩阵为每个单词查找稀疏向量
identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

# 8. 因为最后要进行逻辑回归预测垃圾短信的概率，所以我们需要声明逻辑回归向量。然后声明占位符，注意x_data输入占位符是整数类型，因为它被用来查找单位矩阵和行索引，而TensorFlow要求其为整数类型
A = tf.Variable(tf.random_normal(shape=[embedding_size,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
# Initialize placehoders
x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y_target = tf.placeholder(shape=[1,1], dtype=tf.float32)

# 9. 使用TensorFlow的嵌入查找函数来映射句子中的单词为单位矩阵的one-host向量。然后把前面的词向量求和
x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
x_col_sums = tf.reduce_sum(x_embed, 0)

# 10. 有了每个句子的固定长度的句子向量之后，我们进行逻辑回归训练。声明逻辑回归算法模型。因为一次做一个数据点的随机训练，所有扩展输入数据的维度，并进行线性回归操作。记住，TensorFlow中的损失函数已经包含了sigmoid激励函数，所以我们不需要在输出时加入激励函数
x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
model_output = tf.add(tf.matmul(x_col_sums_2D, A), b)

# 11. 声明训练模型的损失函数、预测函数和优化器
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=model_output, logits=y_target))
# Prediction operation
prediction = tf.sigmoid(model_output)
# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)

# 12. 接下来初始化计算图中的变量
init = tf.global_variables_initializer()
sess.run(init)

# 13. 开始迭代训练。TensorFlow的内建函数vocab_processor.fit()是一个符合本例的生成器。我们将使用该函数来进行随机训练逻辑回归模型。为了得到准确度的趋势，我们保留最近50次迭代的平均值。如果只绘制当前值，我们会依赖预测训练数据点是否正确而得到1或者0的值
loss_vec = []
train_acc_all = []
train_acc_avg = []
for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):
    y_data = [[target_train[ix]]]
    sess.run(train_step, feed_dict={x_data: t, y_target: y_data})
    temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
    loss_vec.append(temp_loss)

    if (ix+1)%10 == 0:
        print('Training Observation #' + str(ix+1) + ': Loss = ' + str(temp_loss))

    # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
    # Get True/False if prediction is accurate
    train_acc_temp = target_train[ix] == np.round(temp_pred)
    train_acc_all.append(train_acc_temp)
    if len(train_acc_all) >= 50:
        train_acc_avg.append(np.mean(train_acc_all[-50:]))


# 14. 训练结果如下
"""
Starting Training Over 4459 Sentences.
Training Observation #10: Loss = -4.8815227
Training Observation #20: Loss = -10.887854
Training Observation #30: Loss = 0.6931472
Training Observation #4430: Loss = 0.6931472
Training Observation #4440: Loss = 0.6931472
Training Observation #4450: Loss = 0.6931472
"""

# 15. 为了得到测试集的准确度，我们重复处理过程，对测试文本只进行预测操作，而不进行训练操作
print('Getting Test Set Accuracy')
test_acc_all = []
for ix, t in enumerate(vocab_processor.fit_transform(texts_test)):
    y_data = [[target_test[ix]]]

    if (ix+1)%50 == 0:
        print('Test Observation #' + str(ix+1))

    # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
    # Get True/False if prediction is accurate
    test_acc_temp = target_test[ix] == np.round(temp_pred)
    test_acc_all.append(test_acc_temp)

print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))
"""
Getting Test Set Accuracy 1115 Sentences.
Test Observation #10
Test Observation #20
Test Observation #30
Test Observation #1000
Test Observation #1050
Test Observation #1100

Overall Test Accuracy: 0.14618834080717488
"""


