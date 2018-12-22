#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/22 下午1:52

# 9.6 TensorFlow实现孪生RNN预测相似度

# 1. 导入必要的编程库
import os
import random
import string
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
sess = tf.Session()

# 2. 设置模型参数
batch_size = 200
n_batches = 300
max_address_len = 20
margin = 0.25
num_features = 50
dropout_keep_prob = 0.8

# 3. 创建一个孪生RNN相似度模型类
def snn(address1,address2,dropout_keep_prob, vocab_size, num_features, input_length):
    # Define the siamese double RNN with a fully connected layer at the end
    def siamee_nn(input_vector, num_hidden):
        cell_input = tf.nn.rnn_cell.BasicLSTMCell

        # Forward direction cell
        lstm_forward_cell = cell_unit(num_hidden, forget_bias=1.0)
        lstm_forward_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_forward_cell, output_keep_prob=dropout_keep_prob)

        # split title into a character sequence
        input_embed_split = tf.split(1, input_length, input_vector)
        input_embed_split = [tf.squeeze(x, squeeze_dims=[1]) for x in input_embed_split]

        # Create bidirectional layer
        outputs, _, _ = tf.nn.bidirectional_dynamic_rnn(lstm_forward_cell, lstm_forward_cell, input_embed_split, dtype=tf.float32)
        # Average The output over the sequence
        temporal_mean = tf.add_n(outputs)/input_length

        # Fully connected layer
        output_size = 10
        A = tf.get_variable(name='A', shape=[2*num_hidden, output_size], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(name='b', shape=[output_size], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))

        final_output = tf.matmul(temporal_mean, A) + b
        final_output = tf.nn.dropout(final_output, dropout_keep_prob)
        return (final_output)

    with tf.variable_scope('siamese') as scope:
        output1 = siamee_nn(address1, num_features)
        # Declare that we will use the same variables on the second string
        scope.reuse_variables()
        output2 = siamee_nn(address2, num_features)

    # Unit normalize the outputs
    output1 = tf.nn.l2_normalize(output1, 1)
    output2 = tf.nn.l2_normalize(output2, 1)
    # Return cosine distance
    # in this case,  the dot product of the norms is the same.
    dot_prod = tf.reduce_sum(tf.multiply(output1, output2), 1)

    return (dot_prod)

"""
使用tf.variable_scope可在Siamese网络的两个部分共享变量参数。注意，余弦距离是归一化向量的点积。
"""

# 4. 声明预测函数，该函数是余弦距离的符号值
def get_predicitions(scores):
    predictions = tf.sign(scores, name="predictions")
    return (predictions)

# 5. 声明损失函数。我们希望为error预留一个margin（类似于SVM模型）。损失函数项中包括正损失和负损失。
def loss(scores, y_target, margin):
    # Calculate the positive losses
    pos_loss_term = 0.25 * tf.square(tf.subtract(1., scores))
    pos_mult = tf.cast(y_target, tf.float32)

    # Make sure positive losses are on similar strings
    positive_loss = tf.multiply(pos_mult, pos_loss_term)

    # Calculate negative losses, then make sure on dissimilar strings
    neg_mult = tf.subtract(1., tf.cast(y_target, tf.float32))

    negative_loss = neg_mult*tf.square(scores)

    # Combine similar and dissimilar losses
    loss = tf.add(positive_loss, negative_loss)

    # Create the margin term. This is when the targets are 0., and the scores are less than m, return 0.

    # Check if target is zero (dissimilar strings)
    target_zero = tf.equal(tf.cast(y_target, tf.float32), 0.)
    # Check if cosine outputs is smaller than margin
    less_than_margin = tf.less(scores, margin)
    # Check if both are true
    both_logical = tf.logical_and(target_zero, less_than_margin)
    both_logical = tf.cast(both_logical, tf.float32)
    # If both are true, then multiply by (1-1)=0.
    multiplicative_factor = tf.cast(1. - both_logical, tf.float32)
    total_loss= tf.multiply(loss, multiplicative_factor)

    # Average loss over batch
    avg_loss = tf.reduce_mean(total_loss)
    return (avg_loss)

# 6. 声明准确度函数
def accuracy(scores, y_target):
    predictions = get_predicitions(scores)
    correct_predictions = tf.equal(predictions, y_target)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return (accuracy)

# 7. 使用基准地址创建有"打印错误"的相似地址
def create_typo(s):
    rand_ind = random.choice(range(len(s)))
    s_list = list(s)
    s_list[rand_ind] = random.choice(string.ascii_lowercase + '0123456789')
    s = ''.join(s_list)
    return (s)

# 8. 将街道号、街道名和街道后缀随机组合生成数据。街道名和街道后缀的列表如下
street_name = ['abbey', 'baker', 'canal', 'donner', 'elm', 'fifth', 'grandivia', 'hollywood', 'interstate', 'jay', 'kings']
street_types = ['rd', 'st', 'ln', 'pass', 'ave', 'hwy', 'cir', 'dr', 'jct']

# 9. 生成测试查询地址和基准地址
test_queries = ['111 abbey ln', '271 doner cicle', '314 king avenuse', 'tensorflow is fun']
test_referrences = ['123 abbery ln ', '217 donner cir', '314 kings ave', '404 hollywood st', 'tensorflow is so fun']

"""
最后的查询和基准地址对于本例模型来说都未见过，但是我们希望模型能识别出它们的相似性。
"""

# 10. 定义如何生成批量数据。本例的指数据是一半相似的地址（基准地址和"打印错误"地址）和一半不相似的地址。不相似的地址是通过读取地址列表的后半部分，并使用numpy.roll()函数将其向后循环移动1位获取的
def get_batch(n):
    # Generate a list of reference addresses with similar address that have
    # a typo.
    numbers = [random.randint(1, 9999) for i in range(n)]
    streets = [random.choice(street_name) for i in range(n)]
    street_suffs = [random.choice(street_types) for i in range(n)]
    full_streets = [str(w) + ' ' + x + ' ' + y for w,x,y in zip(numbers, streets, street_suffs)]
    typo_streets = [create_typo(x) for x in full_streets]
    reference = [list(x) for x in zip(full_streets, typo_streets)]

    # Shuffle last half of them for training on dissimilar address
    half_ix = int(n/2)
    bottom_half = reference[half_ix:]
    true_address = [x[0] for x in bottom_half]
    typo_streets = [x[1] for x in bottom_half]
    typo_streets = list(np.roll(typo_streets, 1))
    bottom_half = [[x,y] for x,y in zip(full_streets, typo_streets)]
    reference[half_ix:] = bottom_half

    # Get target similarities (1's for similar, -1's for non-similar)
    target = [1] * (n-half_ix) + [-1]*half_ix
    reference = [[x,y] for x,y in zip(reference, target)]
    return (reference)

# 11. 定义地址词汇表，以及如何将地址one-hot编码为索引
vocab_chars = string.ascii_lowercase + '0123456789'
vocab2ix_dict = {char:(ix+1) for ix, char in enumerate(vocab_chars)}
vocab_length = len(vocab_chars)+1

# Define vocab one-hot encoding
def address2onehot(address, vocab2ix_dict = vocab2ix_dict, max_address_len = max_address_len):
    # translate address string into indices
    address_ix = [vocab2ix_dict[x] for x in list(address)]

    # Pad or crop to max_address_len
    address_ix = (address_ix + [0]*max_address_len)[0:max_address_len]
    return (address_ix)

# 12. 处理好词汇表，我们开始声明模型占位符和嵌套查找函数。对于嵌套查的来说，我们将使用one-hot编码嵌套，使用单位矩阵作为查找矩阵
address1_ph = tf.placeholder(tf.int32, [None, max_address_len], name='address1_ph')
address2_ph = tf.placeholder(tf.int32, [None, max_address_len], name='address2_ph')
y_target_ph = tf.placeholder(tf.int32, [None], name='y_target_ph')
dropout_keep_prob_ph = tf.placeholder(tf.float32, name='dropout_keep_prob')

# Create embedding lookup
identity_mat = tf.diag(tf.ones(shape=[vocab_length]))
address1_embed = tf.nn.embedding_lookup(identity_mat, address1_ph)
address2_embed = tf.nn.embedding_lookup(identity_mat, address2_ph)

# 13. 声明算法模型、准确度、损失函数和预测操作
# Define Model
text_snn = model.snn(address1_embed, address2_embed, dropout_keep_prob_ph, vocab_length, num_features, max_address_len)
# Define Accuracy
batch_accuracy = model.accuracy(text_snn, y_target_ph)
# Define Loss
batch_loss = model.get_predictions(text_snn)

# 14. 在开始训练模型之前，增加优化器函数，初始化变量
# Declare optimizer
optimizer = tf.train.AdamOptimizer(0.01)
# Apply gradients
train_op = optimizer.minimize(batch_loss)
# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# 15. 现在遍历迭代训练，记录损失函数和准确度
train_los_vec = []
train_acc_vec = []
for b in range(n_batches):
    # Get a batch of data
    batch_data = get_batch(batch_size)
    # Shuffle data
    np.random.shuffle(batch_data)
    # Parse address and targets
    input_address = [x[0] for x in batch_data]
    target_similarity = np.array([x[1] for x in batch_data])
    address1 = np.array([address2onehot(x[0]) for x in input_address])
    address2 = np.array([address2onehot(x[1]) for x in input_address])

    train_feed_dict = {address1_ph: address1, address2_ph: address2, y_target_ph: target_similarity, dropout_keep_prob_ph: dropout_keep_prob_ph}
    # Save train loss and accuracy
    train_loss_vec.append(train_loss)
    train_acc_vec.append(train_acc)

# 16. 训练模型之后，我们处理测试查询和基准地址来查看模型效果
test_queries_ix = np.array([address2onehot(x) for x in test_queries])
test_referrences_ix = np.array([address2onehot(x) for x in test_referrences])
num_refs = test_referrences_ix.shape(0)
best_fix_refs = []
for query in test_queries_ix:
    test_queries = np.repeat(np.array([query]), num_refs, axis=0)
    test_feed_dict = {address1_ph: address1, address2_ph: address2, y_target_ph: target_similarity, dropout_keep_prob: 1.0}
    test_out = sess.run(text_snn, feed_dict=test_feed_dict)
    best_fit = test_references[np.argmax(test_out)]
    best_fit_refs.append(best_fit)
print('Query Address: {}'.format(test_queries))
print('Model Found Matches: {}'.format(best_fit_refs))

# 17. 输出结果如下

"""

"""
