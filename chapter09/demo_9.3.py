#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/20 下午11:16

# 9.3 用TensorFlow实现LSTM模型

# 1. 导入必要的编程库
import os
import re
import string
import requests
import numpy as np
import collections
import random
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

# 2. 开始计算图会话，并设置RNN参数
sess = tf.Session()
# Set RNN Parameters
min_word_freq = 5
rnn_size = 128
epochs = 10
batch_size = 100
learning_reate = 0.001
training_seq_len = 50
embedding_size = rnn_size
save_every = 500
eval_every = 50
prime_texts = ['thou art more', 'to be or not to', 'wherefore art thou']

# 3. 定义数据和模型的文件夹和文件名。我们将保留连字符和省略符，因为莎士比亚频繁地使用这些字符来组合单词和音节
data_dir = 'temp'
data_file = 'shakespeare.txt'
model_path = 'shakespeare_model'
full_model_dir = os.path.join(data_dir, model_path)
# Declare punctuation to remove, everyting except hyphens and apostrophes
punctuation = string.punctuation
punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])

# 4. 下载文本数据集。如果该数据集存在，我们将直接加载数据；如果不存在，我们将下载该文本数据集，并保存
if not os.path.exists(full_model_dir):
    os.makedirs(full_model_dir)
# Make data directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
print('Loading Shakespeare Data')
# Check if file is downloaded.
if not os.path.isfile(os.path.join(data_dir, data_file)):
    print('Not found, downloading Shakespeare texts from www.gutenberg.org')
    shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
    # Get Shakespeare text
    response = requests.get(shakespeare_url)
    shakespeare_file = response.content
    # Decode binary into string
    s_text = shakespeare_file.decode('utf-8')
    # Drop first few descriptive paragraphs.
    s_text = s_text[7675:]
    # Remove newlines
    s_text = s_text.replace('\r\n', '')
    s_text = s_text.replace('\n', '')

    # Write to file
    with open(os.path.join(data_dir, data_file), 'w') as out_conn:
        out_conn.write(s_text)
else:
    # If file has been saved, load from that file
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        s_text = file_conn.read().replace('\n', '')

# 5. 清洗莎士比亚词汇表。我们创建build_vocab()返回两个单词字典（单词到索引的映射和索引到单词的映射），其中出现的单词要符合频次要求
s_text = re.sub(r'[{}]'.format(punctuation), ' ', s_text)
s_text = re.sub('\s+', ' ', s_text.strip().lower())


# 6. 创建莎士比亚词汇表。我们创建build_vocab()返回两个单词字典（单词到索引的映射和索引到单词的映射），其中出现的单词要符合频次要求
def build_vocab(text, min_word_freq):
    word_counts = collections.Counter(text.split(' '))
    # limit word counts to those more frequent than cutoff
    word_counts = {key: val for key, val in word_counts.items() if val > min_word_freq}
    # Create vocab --> index mapping
    words = word_counts.keys()
    vocab_to_ix_dict = {key: (ix + 1) for ix, key in enumerate(words)}
    vocab_to_ix_dict['unknown'] = 0
    # Create index --> vocab mapping
    ix_to_vocab_dict = {val: key for key, val in vocab_to_ix_dict.items()}

    return (ix_to_vocab_dict, vocab_to_ix_dict)


ix2vocab, vocab2ix = build_vocab(s_text, min_word_freq)
vocab_size = len(ix2vocab) + 1

"""
处理文本时，我们需要注意单词索引为0的值，将其保存并填充。对于未知单词也采取相同方法处理。
"""

# 7. 有了单词词汇表，我们将莎士比亚文本转换成索引数组
s_text_words = s_text.split(' ')
s_text_ix = []
for ix, x in enumerate(s_text_words):
    try:
        s_text_ix.append(vocab2ix[x])
    except:
        s_text_ix.append(0)

s_text_ix = np.array(s_text_ix)


# 8. 本例将展示如何用class对象创建算法模型。我们将使用相同的模型（相同模型的参数）来训练批量数据和抽样生成的文本。如果没有class对象，将很难用抽样方法训练相同的模型。在理想情况下，该class代码单独保存在一个Python文件中，它可以在脚本起始位置导入
class LSTM_Model():
    def __init__(self, rnn_size, batch_size, learning_rate, training_seq_len, vocab_size, infer=False):
        self.rnn_size = rnn_size
        self.vocab_size = vocab_size
        self.infer = infer
        self.learing_rate = learning_rate

        if infer:
            self.batch_size = 1
            self.training_seq_len = 1
        else:
            self.batch_size = batch_size
            self.training_seq_len = training_seq_len

        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

        self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])

        with tf.variable_scope('lstm_vars'):
            # Softmax Output Weights
            W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.constant_initializer(0, 0))

            # Define Embedding
            embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.rnn_size], tf.float32,
                tf.random_normal_initializer())

            embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data)
            rnn_inputs = tf.split(1, self.training_seq_len, embedding_output)
            rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs]

        # If we are inferring (generating text), we add a 'loop' function
        # Define how to get the i+1 th input from the i th output
        def inferred_loop(prev, count):
            prev_transformed = tf.matmul(prev, W) + b
            prev_symbol = tf.stop_gradient(tf.argmax(prev_transformed, 1))
            output = tf.nn.embedding_lookup(embedding_mat, prev_symbol)
            return (output)

        decoder = tf.nn.seq2seq.rnn_decoder
        outputs, last_state = decoder(rnn_inputs_trimmed, self.initial_state, self.lstm_cell,
            loop_function=inferred_loop if infer else None)
        # Non inferred outputs
        output = tf.reshape(tf.concat(1, outputs), [-1, self.rnn_size])
        # Logits and output
        self.cost = tf.reduce_sum(loss) / (self.batch_size * self.training_seq_len)
        self.final_state = last_state
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.learing_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

    def sample(self, sess, words=ix2vocab, vocab=vocab2ix, num=10, prime_text='thou art'):
        state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
        word_list = prime_text.split()
        for word in word_list[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed_dict=feed_dict)

        out_sentence = prime_text
        word = word_list[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
            sample = np.argmax(model_output[0])
            if sample == 0:
                break
            word = words[sample]
            out_sentence = out_sentence + ' ' + word

        return (out_sentence)

# 9. 声明LSTM模型及其测试模型。使用tf.variable_scope管理模型变量，使得测试LSTM模型可以复用LSTM模型相同的参数
with tf.variable_scope('lstm_model') as scope:
    # Define LSTM Model
    lstm_model = LSTM_Model(rnn_size, batch_size, learning_rate, training_seq_len, vocab_size)
    scope.reuse_variables()
    test_lstm_model = LSTM_Model(rnn_size, batch_size, learning_rate, training_seq_len, vocab_size)

# 10. 创建saver操作，并分割输入文本为相同的批量大小的块，然后初始化模型变量
saver = tf.train.Saver()
# Create batches for each epoch
num_batches = int(len(s_text_ix)/(batch_size * training_seq_len)) + 1
# Split up text indices into subarrays, of equal size
batches = np.array_split(s_text_ix, num_batches)
# Reshape each split into [batch_size, training_seq_len]
batches = [np.resize(x, [batch_size, training_seq_len]) for x in batches]
# Initialize all vairables
init = tf.global_variables_initializer()
sess.run(init)

# 11. 现在通过epoch迭代训练，并在每个epoch之前将数据shuffle。虽然文本数据是相同的，但是会用numpy.roll()函数改变顺序
train_loss = []
iteration_count = 1
for epoch in range(epochs):
    # Shuffle word indices
    random.shuffle(batches)
    # Create targets from shuffled batches
    targets = [np.roll(x, -1, axis=1) for x in batches]
    # Run a through one epoch
    print('Starting Epoch #{} of {}.'.format(epoch+1, epochs))
    # Reset initial LSTM state every epoch
    state = sess.run(lstm_model.initial_state)
    for ix, batch in enumerate(batches):
        training_dict = {lstm_model.x_data: batch, lstm_model.y_output: targets[ix]}
        c, h = lstm_model.initial_state
        training_dict[c] = state.c
        training_dict[h] = state.h

        temp_loss, state, _ = sess.run([lstm_model.cost, lstm_model.final_state, lstm_model.train_op], feed_dict=training_dict)
        train_loss.append(temp_loss)

    # Save the model and the vocab
    if iteration_count % save_every == 0:
        # Save model
        model_file_name = os.path.join(full_model_dir, 'model')
        saver.save(sess, model_file_name, globa_step = iteration_count)
        print('Model Saved To: {}'.format(model_file_name))
        # Save vocabulary
        dictionary_file = os.path.join(full_model_dir, 'vocab.pkl')
        with open(dictionary_file, 'wb') as dict_file_conn:
            pickle.dump([vocab2ix, ix2vocab], dict_file_conn)

        if iteration_count % eval_every == 0:
            for sample in prime_texts:
                print(test_lstm_model.sample(sess, ix2vocab, vocab2ix, num=10, prime_text=sample))

        iteration_count += 1

# 12. 输出结果如下

"""
"""

# 13. 绘制训练损失随epoch的趋势图
plt.plot(train_loss, 'k-')
plt.title('Sequence to Sequence Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()


