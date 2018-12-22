#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/21 下午11:33

# 9.5 用TensorFlow实现Seq2Seq翻译模型

# 1. 导入必要的编程库
import os
import string
import requests
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile
from collections import Counter
from tensorflow.models.rnn.translate import seq2seq_model

sess = tf.Session()

# 2. 设置模型参数。学习率设为0.1，本例也会每迭代100次模型衰减1%的学习率，这会在迭代过程中微调算法模型。设置截止最大梯度。RNN大小为500。英语和德语词汇的词频设为10 000。我们将所有词汇转为小写，并移除标点符号。将德语umlaut（元音变音, a、e、i、o、u）和eszett（德文字母之一, β）转为字母数字，归一化德语词汇
learning_rate = 0.1
lr_decay_rate = 0.99
lr_decay_every = 100
max_gradient = 5.0
batch_size = 50
num_layers = 3
rnn_size = 500
layer_size = 512
generations = 10000
vocab_size = 10000
save_every = 1000
eval_every = 500
output_every = 50
punct = string.punctuation
data_dir = 'temp'
data_file = 'eng_ger.txt'
model_path = 'seq2seq_model'
full_model_dir = os.path.join(data_dir, model_path)

# 3. 准备三个英文句子测试翻译模型，看下训练的模型效果
test_english = ['hello where is my computer',
    'the quick brown fox jumped over the lazy dog',
    'is it going to rain tomorrow']

# 4. 创建模型文件夹。检查语料文件是否已下载，如果已经下载过语料文件，则直接读取文件；如果没有下载，则下载并保存到指定文件夹
if not os.path.exists(full_model_dir):
    os.makedirs(full_model_dir)
# Make data directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
print('Loading English-German Data')
# Check for data, if it doesn't exist, download it and save it
if not os.path.isfile(os.path.join(data_dir, data_file)):
    print('Data not found, downloading Eng-Ger sentences from www.manythings.org')
    sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
    r = requests.get(sentence_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('deu.txt')
    # Format Data
    eng_ger_data = file.decode()
    eng_ger_data = eng_ger_data.encode('ascii', errors='ignore')
    eng_ger_data = eng_ger_data.decode().split('\n')
    # Write to file
    with open(os.path.join(data_dir, data_file), 'w') as out_conn:
        for sentence in eng_ger_data:
            out_conn.write(sentence + '\n')
else:
    eng_ger_data = []
    with open(os.path.join(data_dir, data_file), 'r') as in_conn:
        for fow in in_conn:
            eng_ger_data.append(row[:-1])

# 5. 清洗预料数据集，移除标点符号，分割句子中的英语和德语，并全部转为小写
eng_ger_data = [''.join(char for char in sent if char not in punct) for sent in eng_ger_data]
# Split each sentence by tabs
eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x) >= 1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
english_sentence = [x.lower().split() for x in english_sentence]
german_sentence = [x.lower().split() for x in german_sentence]

# 6. 创建英语词汇表和德语词汇表，其中词频都要求至少10 000。不符合词频要求的单词标记为0（未知）。大部分低频词为代词（名字或者地名）。
all_english_words = [word for sentence in english_sentence for word in sentence]
all_english_counts = Counter(all_english_words)
eng_word_keys = [x[0] for x in all_english_counts.most_common(vocab_size - 1)]  # -1 because 0=unknown is also in there
eng_vocab2ix = dict(zip(eng_word_keys, range(1, vocab_size)))
eng_ix2vocab = {val: key for key, val in eng_vocab2ix.items()}
english_processed = []
for sent in english_sentence:
    temp_sentence = []
    for word in sent:
        try:
            temp_sentence.append(eng_vocab2ix[word])
        except:
            temp_sentence.append(0)
    english_processed.append(temp_sentence)
all_german_words = [word for sentence in german_sentence for word in sentence]
all_german_counts = Counter(all_german_words)
ger_word_keys = [x[0] for x in all_german_counts.most_common(vocab_size - 1)]
ger_vocab2ix = dict(zip(ger_word_keys, range(1, vocab_size)))
ger_ix2vocab = {val: key for key, val in ger_vocab2ix.items()}
german_processed = []
for sent in german_sentence:
    temp_sentence = []
    for word in sent:
        try:
            temp_sentence.append(ger_vocab2ix[word])
        except:
            temp_sentence.append(0)
german_processed.append(temp_sentence)

# 7. 预处理测试词汇，将其写入词汇索引中
test_data = []
for sentence in test_english:
    temp_sentence = []
    for word in sentence.split(' '):
        try:
            temp_sentence.append(eng_vocab2ix[word])
        except:
            # Use '0' if the word isn't in our vocabulary
            temp_sentence.append(0)
    test_data.append(temp_sentence)

# 8. 因为某些句子太长或者太短，所以我们为不同长度的句子创建单独的模型。做这些的原因之一是最小化短句子中填充字符的影响。解决该问题的方法之一是将相似长度的句子分桶处理。我们为每个分桶设置长度范围，这样相似长度的句子就会进入同一个分桶
x_maxs = [5, 7, 11, 50]
y_max = [10, 12, 17, 60]
buckets = [x for x in zip(x_maxs, y_maxs)]
bucketed_data = [[] for _ in range(len(x_maxs))]
for eng, ger in zip(english_processed, german_processed):
    for ix, (x_max, y_max) in buckets:
        if (len(eng) <= x_max) and (len(ger) <= y_max):
            bucketed_data[ix].append([eng, ger])
            break


# 9. 将上述参数传入TensorFlow内建的Seq2Seq模型。创建translation_model()函数保证训练模型和测试模型可以共享相同的变量
def translation_model(sess, input_vocab_size, output_vocb_size, buckets, rnn_size, num_layers, max_gradient,
        learning_rate, lr_decay_rate, forward_only):
    model = seq2seq_model.Seq2SeqModel(
        input_vocab_size,
        output_vocb_size,
        buckets,
        rnn_size,
        num_layers,
        max_gradient,
        batch_size,
        learning_rate,
        lr_decay_rate,
        forward_only=forward_only,
        dtype=tf.float32)
    return (model)


# 10. 创建训练模型，使用tf.variable_scope管理模型变量，声明训练模型的变量在scope范围内可重用。创建测试模型，其批量大小为1
input_vocab_size = vocab_size
output_vocab_size = vocab_size
with tf.variable_scope('translate_model') as scope:
    translation_model = translation_model(sess, vocab_size, vocab_size, buckets, rnn_size, num_layers,
        max_gradient, learning_rate, lr_decay_rate, False)
    # Reuse the variable for the test model
    scope.reuse_variables()
    test_model = translation_model(sess, vocab_size, vocab_size, buckets, rnn_size, num_layers,
        max_gradient, learning_rate, lr_decay_rate, True)
    test_model.batch_size = 1

# 11. 初始化模型变量
init = tf.global_variables_initializer()
sess.run(init)

# 12. 调用step()函数迭代训练Seq2Seq模型。TensorFlow的Seq2Seq模型有get_batch()函数，该函数可以从分桶索引迭代批量句子。衰减学习率，保存Seq2Seq训练模型，并利用测试句子进行模型评估。
train_loss = []
for i in range(generations):
    rand_bucket_ix = np.random.choice(len(bucketed_data))

    model_outputs = translation_model.get_batch(bucketed_data, rand_bucket_ix)
    encoder_inputs, decoder_inputs, target_weights = model_outputs

    # Get the (gradient norm, loss, and outputs)
    _, step_loss, _ = translation_model.step(sess, encoder_inputs, decoder_inputs, target_weights, rand_bucket_ix,
        False)

    # Output status
    if (i + 1) % output_every == 0:
        train_loss.append(step_loss)
        print('Gen #{} out of {}. Loss: {:.4}'.format(i + 1, generations, step_loss))

    # Check if we should decay the learning rate
    if (i + 1) % lr_decay_every == 0:
        sess.run(translate_model.learning_rate_decay_op)

    # Save model
    if (i + 1) % save_every == 0:
        print('Saving model to {}.'.format(full_model_dir))
        model_save_paht = os.path.join(full_model_dir, 'eng_ger_translation.ckpt')
        translate_model.saver.save(sess, moel_save_path, global_step=i)

    # Eval on test set
    if (i + 1) % eval_every == 0:
        for ix, sentence in enumerate(test_data):
            # Find which bucket sentence goes in
            bucket_id = next(index for index, val in enumerate(x_maxs) if val >= len(sentence))
            # Get RNN model outputs
            encoder_inputs, decoder_inputs, target_weights = test_model.get_batch({bucket_id: [(sentence, [])]},
                bucket_id)
            # Get logits
            _, test_loss, output_logits = test_model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                bucket_id, True)
            ix_output = [int(np.argmax(logit, axis=1) for logit in output_logits)]
            # If there is a 0 symbol in outputs end the output thers.
            ix_output = ix_output[0:[ix for ix, x in enumerate(ix_output + [0]) if x == 0][0]]
            # Get german words from indices
            test_german = [ger_ix2vocab[x] for x in ix_output]
            print('English: {}'.format(test_english[ix]))
            print('German: {}'.format(test_german))

# 13. 下面是输出结果

"""

"""

# 14. 使用matplotlib模块绘制训练损失图
loss_generations = [i for i in range(generations) if i % output_every == 0]
plt.plot(loss_generations, train_loss, 'k-')
plt.title('Sequence to Sequence Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()