#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/10 下午10:06

# 7.4 用TensorFlow实现skip-gram模型

# 1. 导入必要的编程库，开始一个计算图会话
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import string
import requests
import collections
import io
import tarfile
import urllib.request
from nltk.corpus import stopwords
sess = tf.Session()

# 2. 声明一些模型参数。
batch_size = 50
embedding_size = 200
vocabulary_size = 10000
generations = 50000
print_loss_every = 500
num_sampled = int(batch_size/2)
window_size = 2
stops = stopwords.words('english')
print_valid_every = 2000
valid_words = ['cliche', 'love', 'hate', 'silly', 'sad']

# 3. 声明数据加载函数，该函数会在下载数据前先检测是否已下载过该数据集，如果已经下载过，将直接从磁盘加载数据
def load_movie_data():
    save_foler_name = 'temp'
    pos_file = os.path.join(save_foler_name, 'rt-polarity.pos')
    neg_file = os.path.join(save_foler_name, 'rt-polarity.neg')
    # Check if files are already downloaded
    if False and os.path.exists(save_foler_name):
        pos_data = []
        with open(pos_file, 'r') as temp_pos_file:
            for row in temp_pos_file:
                pos_data.append(row)
        neg_data = []
        with open(neg_file, 'r') as temp_neg_file:
            for row in temp_neg_file:
                neg_data.append(row)
    else: # if not download, download and save
        movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
        stream_data = urllib.request.urlopen(movie_data_url)
        tmp = io.BytesIO()
        while True:
            s = stream_data.read(1638400)
            if not s:
                break
            tmp.write(s)
            stream_data.close()
            tmp.seek(0)
        tar_file = tarfile.open(fileobj=tmp, mode='r:gz')
        pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
        neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
        # Save pos/neg reviews
        pos_data = []
        for line in pos:
            pos_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
        neg_data = []
        for line in neg:
            neg_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
        tar_file.close()
        # Write to file
        if not os.path.exists(save_foler_name):
            os.makedirs(save_foler_name)
        # Save files
        with open(pos_file, 'w') as pos_file_handler:
            pos_file_handler.write(''.join(pos_data))
        with open(neg_file, 'w') as neg_file_hander:
            neg_file_hander.write(''.join(neg_data))
    texts = pos_data+neg_data
    target = [1]*len(pos_data) + [0]*len(neg_data)
    return (texts, target)

texts, target = load_movie_data()

# 4. 创建归一化文本函数。该函数输入一列字符串，转换大小字符，移除标点符号，移除数字，去除多余的空白字符, 并移除"停词"
def normalize_text(texts, stops):
    # Lower case
    texts = [x.lower() for x in texts]
    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]

    return (texts)

texts = normalize_text(texts, stops)

# 5. 为了确保所有电影影评的有效性，我们检查其中的影评长度。可以强制影评长度为三个单词或者更长长度的单词
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 3]

# 6. 构建词汇表，创建函数来建立一个单词字典（该单词词典是单词和单词数对）。词频不够的单词（即标记为unknown的单词）标记为RARE
def build_dictionary(sentences, vocabulary_size):
    # Turn sentences (list of strings) into lists of words
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]
    # Initialize list of [word, word_count], for each word, starting with unknown
    count = [['RARE', -1]]
    # Now add most frequest words, limited to the N-most frequent (N=vocabulary size)
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    # Now create the dictionary
    word_dict = {}
    # For each word, that we want in the dictionary, add it, then make it the value of the prior dictionary length
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    return (word_dict)

# 7. 创建一个函数将一系列的句子转化成单词索引列表，并将单词索引列表传入嵌套寻找函数
def text_to_numbers(sentences, word_dict):
    # Initialize the returned data
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        for word in sentence:
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
        data.append(sentence_data)
    return (data)

# 8. 创建单词字典，转换句子列表为单词索引列表
word_dictionary = build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_to_numbers(texts, word_dictionary)

# 9. 从预处理的单词词典中，查找第二步中选择的验证单词的索引
valid_examples = [word_dictionary[x] for x in valid_words]

# 10. 创建函数返回skip-gram模型的指数据
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random sentence to start
        rand_sentence = np.random.choice(sentences)
        # Generate consecutive windows to look at
        window_sequences = [rand_sentence[max(ix-window_size,0):(ix+window_size+1)] for ix, x in enumerate(rand_sentence)]
        # Denoate which element of each window is the center word of interset
        label_indices = [ix if ix<window_size else window_size for ix,x in enumerate(window_sequences)]

        # Pull out center word of interset for each window and create a tuple for each window
        if method=='skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x,y in zip(window_sequences, label_indices)]
            # Make it in to a big list of tuples (target word, surrounding word)
            tuple_data = [(x, y_) for x,y in batch_and_labels for y_ in y]
        else:
            raise ValueError('Method {} not implmented yet.'.format(method))

        # extract batch and labels
        batch, labels = [list(x) for x in zip(*tuple_data)]
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return (batch_data, label_data)

# 11. 初始化嵌套矩阵，声明占位符和嵌套查找函数
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
# Create data/target placeholders
x_inputs = tf.placeholder(tf.float32, shape=[batch_size])
y_target = tf.placeholder(tf.float32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.float32)

# Lookup the word embedding:
embed = tf.nn.embedding_lookup(embeddings, x_inputs)

# 12. softmax损失函数是用来实现多类分类问题常见的损失函数，上一节中其计算预测错误单词分类的损失。
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, y_target, num_sampled, vocabulary_size))

# 13. 创建函数寻找验证单词周围的单词
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalize_embeddings = embeddings/norm
valid_embeddings = tf.nn.embedding_lookup(normalize_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalize_embeddings, transpose_b=True)

# 14. 声明优化器函数，初始化模型变量
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

# 15. 迭代训练词嵌套，打印出损失函数和验证单词集单词的最接近的单词
loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size)
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}
    # Run the train step
    sess.run(optimizer, feed_dict=feed_dict)
    # Return the loss
    if (i+1)%print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at step {}: {}'.format(i+1, loss_val))
    # Validation: Print some random words and top 5 related words
    if (i+1)%print_valid_every == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_rev[valid_examples[j]]
            top_k = 5 # number of nearest neighbors
            nearest = (-sim[j,:]).argsort()[1:top_k+1]
            log_str = 'NEarest to {}:'.format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

# 16. 输出结果如下
"""

"""