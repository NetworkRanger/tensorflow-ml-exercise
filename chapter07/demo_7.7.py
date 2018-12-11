#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/11 下午10:31

# 7.7 用TensorFlow实现基于Doc2Vec的情感分析

# 1. 导入必要的编程库
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

# 2. 加载影评数据集
data_foler_name = 'temp'
if not os.path.exists(data_foler_name):
    os.makedirs(data_foler_name)
texts, target = text_helpers.load_movie_data(data_foler_name)

# 3. 声明算法模型参数
batch_size = 500
vocabulary_size = 7500
generations = 100000
model_learning_rate = 0.001
embedding_size = 200 # Word embedding size
doc_embedding_size = 100 # Document embedding size
concatenated_size = embedding_size + doc_embedding_size
num_sampled = int(batch_size/2)
window_size = 3 # How many words to consider to the left.
# Add checkpoints to traings
save_embeddings_every = 5000
print_valid_every = 5000
print_loss_every = 100
# Declare stop words
stops = stopwords.words('english')
# We pick a few test words.
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']

# 4. 归一化电影影评，确保每条影评都大于指定的窗口大小
texts = text_helpers.normalize_text(texts, stops)
# Texts must contain at least as much as the prior window size
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > window_size]
texts = [x for x in texts if len(x.split()) > window_size]
assert(len(target) == len(texts))

# 5. 创建单词字典。值得注意的是，我们无须创建文档字典。文档索引仅仅是文档的索引值，每个文档有唯一的索引值
word_dictionary = text_helpers.build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values()), word_dictionary.keys())
text_data = text_helpers.text_to_numbers(texts, word_dictionary)
# Get validation word keys
valid_examples = [word_dictionary[x] for x in valid_words]

# 6. 定义单词嵌套和文档嵌套。然后声明对比噪声损失函数
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
doc_embeddings = tf.Variable(tf.random_uniform([len(texts), doc_embedding_size], -1.0, 1.0))
# NCE loss parameters
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, concatenated_size], stddev=1.0/np.sqrt(concatenated_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# 7. 声明Doc2Vec索引和目标单词索引的占位符。注意，输入索引的大小是窗口大小加1，这是因为每个生成的数据窗口将有一个额外的文档索引
x_inputs = tf.placeholder(tf.int32, shape=[None, window_size+1])
y_target = tf.placeholder(tf.int32, shape=[None, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# 8. 创建嵌套函数将单词嵌套求和，然后连接文档嵌套
embed = tf.zeros([batch_size, embedding_size])
for element in range(window_size):
    embed += tf.nn.embedding_lookup(embeddings, x_inputs[:,element])
doc_indices = tf.slice(x_inputs, [0, window_size], [batch_size, 1])
doc_embed = tf.nn.embedding_lookup(doc_embeddings, doc_indices)
# concatenate embeddings
final_embed = tf.concat(1, [embed, tf.squeeze(doc_embed)])

# 9. 现在声明损失函数并创建优化器
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, final_embed, y_target, num_sampled, vocabulary_size))

# Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate)
train_step = optimizer.minimize(loss)

# 10. 声明验证单词集的余弦距离
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings/norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# 11. 为了保存嵌套，创建模型的saver函数，然后初始化模型变量
saver = tf.train.Saver({'embeddings': embeddings, 'doc_embeddings': doc_embeddings})
init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
loss_x_vec = []
for i in range(generations):
    batch_inputs,batch_labels = text_helpers.generate_batch_data(text_data, batch_size, window_size, method='doc2vec')
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}
    # Run the train stop
    sess.run(train_step, feed_dict=feed_dict)

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
            nearest = (-sim[j, :]).argsort()[1:top_k+1]
            log_str = "Nearest to {}".format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                log_str = '{} {},'.format(log_str, close_word)
            print(log_str)

    # Save dictionary + embeddings
    if (i+1)%save_embeddings_every == 0:
        # Save vocabulary dictionary
        with open(os.path.join(data_foler_name, 'movie_vocab.pkl')) as f:
            pickle.dump(word_dictionary, f)

        # Save embeddings
        model_checkpoint_path = os.path.join(os.getcwd(), data_foler_name, 'doc2vec_movie_embeddings.ckpt')
        save_path = saver.save(sess, model_checkpoint_path)
        print('Model saved in file: {}'.format(save_path))

# 12. 打印结果如下

"""

"""

# 13. 训练完Dod2Vec嵌套，我们能使用这些嵌套训练逻辑回归模型，预测影评情感色彩。首先设置逻辑回归模型的一些参数
max_words = 20 # maximum review word length
logistic_batch_size = 500 # training batch size

# 14. 分割数据集为训练集和测试集
train_indices = np.sort(np.random.choice(len(target), round(0.8*len(target)), replace=False))
test_indices = np.sort(np.array(list(set(range(len(target))) - set(train_indices))))
texts_train = [x for ix,x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# 15. 将电影影评转换成数值型的单词索引，填充或者裁剪每条影评为20个单词
text_data_train = np.array(text_helpers.text_to_numbers(texts_train, word_dictionary))
text_data_test = np.array(text_helpers.text_to_numbers(texts_test, word_dictionary))
# Pad/crop movie reviews to specific length
text_data_train = np.array([x[0:max_words] for x in [y+[0]*max_words] for y in text_data_train])
text_data_test = np.array([x[0:max_words] for x in [y+[0]*max_words] for y in text_data_test])

# 16. 声明逻辑回归模型的数据占位符、模型变量、模型操作和损失函数
# Define Logistic placeholders
log_x_inputs = tf.placeholder(tf.int32, shape=[None, max_words+1])
log_y_target = tf.placeholder(tf.int32, shape=[None, 1])
A = tf.Variable(tf.random_normal(shape=[concatenated_size, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Declare logistic model (sigmoid in los function)
model_output = tf.add(tf.matmul(log_final_embed, A), b)

# Declare loss function (Cross Entropy loss)
logistic_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(model_output, tf.cast(log_y_target, tf.float32)))
tf.cast(log_y_target, tf.float32)

# 17. 创建另外一个嵌套函数。前半部分的一个嵌套函数是训练3个单词窗口和文档索引预测最近的单词。这里也是类似的功能，不同的是训练20个单词的影评
# Add together element embeddings in window:
log_embed = tf.zeros([logistic_batch_size, embedding_size])
for element in range(max_words):
    log_embed += tf.nn.embedding_lookup(embeddings, log_x_inputs[:,element])
log_doc_indices = tf.slice(log_x_inputs, [0,max_words], [logistic_batch_size, 1])
log_doc_embed = tf.nn.embedding_lookup(doc_embeddings, log_doc_indices)
# concatenate embeddings
log_final_embed = tf.concat(1, [log_embed, tf.squeeze(log_doc_embed)])

# 18. 创建预测函数和准确度，评估迭代训练模型。然后声明优化器函数，初始化所有模型变量
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, tf.cast(log_y_target, tf.float32), tf.float32))
accuracy = tf.reduce_mean(predictions_correct)
# Declare optimizer
logistic_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
logistic_train_step = logistic_opt.minimize(logistic_loss, var_list=[A,b])
# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# 19. 开始训练逻辑回归模型
train_loss = []
test_los = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    rand_index = np.random.choice(text_data_train.shape[0], size=logistic_batch_size)
    rand_x = text_data_train[rand_index]
    # Append review index at the end of text data
    rand_x_doc_indices = train_indices[rand_index]
    rand_x = np.hstack((rand_x, np.transpose([rand_x_doc_indices])))
    rand_y = np.transpose([target_train[rand_index]])

    feed_dict = {log_x_inputs: rand_x, log_y_target: rand_y}
    sess.run(logistic_train_step, feed_dict=feed_dict)

    # Only record loss and accuracy every 100 generations
    if (i+1)%100 == 0:
        rand_index_test = np.random.choice(text_data_test.shape[0], size=logistic_batch_size)
        rand_x_test = text_data_test[rand_index_test]
        # Append review index at the end of text data
        rand_x_doc_indices_test = test_indices[rand_index_test]
        rand_x_test = np.hstack((rand_x_test, np.transpose([rand_x_doc_indices_test])))
        rand_y_test = np.transpose([target_test[rand_index_test]])

        i_data.append(i+1)
        train_loss_temp = sess.run(logistic_loss, feed_dict=feed_dict)
        train_loss.append(train_loss_temp)
        test_loss_temp = sess.run(logistic_loss, feed_dict=feed_dict)
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(logistic_loss, feed_dict=feed_dict)
        train_acc.append(train_acc_temp)
        test_acc_temp = sess.run(logistic_loss, feed_dict=feed_dict)
        test_acc.append(test_acc_temp)

    if (i+1)%500 == 0:
        acc_and_loss = [i+1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

# 20. 打印结果如下

"""

"""

# 21. 我们已经创建了独立的批量数据生成的方法----text_helpers.generate_batch_data()函数。本节前面使用该方法训练Doc2Vec嵌套
def generate_bach_data(sentences, batch_size, window_size, method='skip_gram'):
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random sentence to start
        rand_sentence_ix = int(np.random.choice(len(sentences), size=1))
        rand_sentence = sentences[rand_sentence_ix]
        # Generate consecutive windows to look at
        window_sequences = [rand_sentence[max(ix-window_size,0):(ix+window_size+1)] for ix, x in enumerate(rand_sentence)]
        # Denote which element of each window is the center word f interset
        label_indices = [ix if ix<window_size else window_size for ix, x in enumerate(window_sequences)]
        # Pull out center word of interest for each window and create a tuple for each window
        if method=='skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x,y in zip(window_sequences, label_indices)]
            # Make it in to a big list of tuples (target word, surrounding word)
            tuple_data = [(x, y_) for x,y in batch_and_labels for y_ in y]
        elif method=='cbow':
            batch_and_labels = [(x[:y] + x[(y+1):], x[y]) for x, y in zip(window_sequences, label_indices)]

            # Only keep windows with consistant 2*window_size
            batch_and_labels = [(x,y) for x,y in batch_and_labels if len(x) == 2*window_size]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
        elif method=='doc2vec':
            # For doc2vec we keep LHS window only to predict target word
            batch_and_labels = [(rand_sentence[i:i+window_size], rand_sentence[i+window_size]) for i in range(0, len(rand_sentence)-window_size)]
            batch, labels = [list(x) for x in zip(*batch_and_labels)]
            # Add document index to batch!! Remember that we must extaract the last index in batch for the doc-index
            batch = [x + [rand_sentence_ix] for x in batch]

        else:
            raise ValueError('Method {} not implmented yet.'.format(method))

        # extract batch and labels
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return (batch_data, label_data)





