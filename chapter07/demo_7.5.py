#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/11 下午8:39

# 7.5 用TensorFlow实现CBOW词嵌入模型

# 1. 导入必要的编程库，包括前面的text_helpers.py脚本，该脚本可以进行文本加载和处理。然后创建一个计算图会话
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
import text_helpers
from nltk.corpus import stopwords
sess = tf.Session()

# 2. 确保临时数据和参数存储在文件夹中
# Make a saving directory if it doesn't exist
data_folder_name = 'temp'
if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)

# 3. 声明算法模型的参数，这些参数和上一节中的skip-gram模型类似
# Declare model parameters
batch_size = 500
embedding_size = 200
vocabulary_size = 2000
generations = 50000
model_learning_rate = 0.001
num_sampled = int(batch_size/2)
window_size = 3
# Add checkpoints to training
save_embeddings_every = 5000
print_valid_every = 5000
print_loss_every = 100
# Declare stop words
stops = stopwords.words('english')
# We pick some test words. We are expecting synonyms to appear
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']

# 4. 调用第一步中封装的辅助函数、数据加载函数和文本归一化函数。在本例中，设置电影影评大于三个单词
texts, target = text_helpers.load_movie_data(data_folder_name)
texts = text_helpers.normalize_text(texts, stops)
# Texts must contain at least 3 words
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]

# 5. 创建单词字典，以便查找单词。同时，我们也需要一个逆序单词字典，可以通过索引查找单词。当我们想打印出验证单词集中每个单词最近的单词时，可使用逆序单词字典
word_dictionary = text_helpers.build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_helpers.text_to_numbers(texts, word_dictionary)
# Get validation word keys
valid_examples = [word_dictionary[x] for x in valid_words]

# 6. 初始化待拟合的单词嵌套并声明算法模型的数据占位符
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
# Create data/target placeholders
x_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2*window_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# 7. 处理单词嵌套。因为CBOW模型将上下文窗口内的单词嵌套叠加在一起，所以创建一个循环将窗口内的所有单词嵌套加在一起
# Lookup the word embeddings and
# Add together window embeddings:
embed = tf.zeros([batch_size, embedding_size])
for element in range(2*window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:element])

# 8. 使用TensorFlow内建的NCE损失函数。因为本例中的输出结果稀疏太强，导致softmax函数收敛存在一定问题
# NCE loss parameters
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
# Declare loss function (NCE)
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, y_target, num_sampled, vocabulary_size))

# 9. 如上一节中的skip—gram模型一样，我们使用余弦相似度量来度量验证单词集中每个单词最接近的单词
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings/norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# 10. 为了保存词向量，我们需要加载TensorFlow的train.Saver()方法。该方法默认会保存整个计算图会话，但是本例中我们会指定参数只保存嵌套变量，并设置名字。这里设置保存的名字与计算图中的变量名相同
saver = tf.train.Saver({'embeddings': embeddings})

# 11. 现在声明优化器函数，初始化模型变量
optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate).minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

# 12. 最后，遍历迭代训练，打印损失函数，保存单词嵌套到指定文件夹
loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = text_helpers.generate_batch_data(text_data, batch_size, window_size, method='cbow')
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}
    # Run the train step
    sess.run(optimizer, feed_dict=feed_dict)
    # Return the loss
    if (i+1)%print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at step {}: {}'.format(i+1, loss_val))

    # Validatiaon: Print some random words and stop 5 related words
    if (i+1)%print_valid_every == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_rev[valid_examples[j]]
            top_k = 5 # Number of nearest neighbors
            nearest = (-sim[j, :]).argsort()[1:top_k+1]
            log_str = "Nearest to {}:".format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                print_str = '{} {}, '.format(log_str, close_word)
            print(print_str)
    # Save dictionary + embeddings
    if (i+1)%save_embeddings_every == 0:
        # Save vocabulary dictionary
        with open(os.path.join(data_folder_name, 'movie_vocab.pkl'), 'wb') as f:
            pickle.dump(word_dictionary, f)

        # Save embeddings
        model_checkpoint_path = os.path.join(os.getcwd(), data_folder_name, 'cbow_movie_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))

# 13. 输出结果如下

"""

"""

