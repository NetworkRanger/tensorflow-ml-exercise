#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/22 下午4:36

# 10.5 TensorFlow产品化开发提示

# 1. 当运行TensorFlow程序时,我们可能希望确保内存中没有其他计算图会话，或者每次调试程序时重置计算图会话
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# 2. 当处理文本或者任意数据管道，我们需要确保保存处理过的数据，以便随后用相同的方式处理评估数据。如果处理文本数据，我们需要确定保存和加载词汇字典。下面的代码是保存JSON格式的词汇字典的例子
import json
word_list = ['to', 'be', 'or', 'not', 'to', 'be']
vocab_list = list(set(word_list))
vocab2ix_dict = dict(zip(vocab_list, range(len(vocab_list))))
ix2vocab_dict = {val: key for key,val in vocab2ix_dict.items()}
# Save vocabulary
import json
with open('vocab2ix_dict.json', 'w') as file_conn:
    json.dump('vocab2ix_dict.json', file_conn)
# Load vocabulary
with open('vocab2ix_dict.json', 'r') as file_conn:
    vocab2ix_dict = json.load(file_conn)

"""
本例使用JSON格式存储词汇字典，但是我们也可以将其存储为txt、CSV或者二进制文件。如果词汇字典太大，则建议使用二进制文件。用pickle库创建pkl二进制文件。但是pkl文件在不同的Python版本间不兼容。
"""

# 3. 为了保存算法模型的计算图和变量，我们在计算图中创建Saver()操作。建议在模型训练过程中按一定规则保存模型。
# After model declaration, add a saving operations
saver = tf.train.Saver()
# Then during training, save every so often, referencing the training generation
for i in range(generations):
    # ...
    if i%save_every == 0:
        saver.save(sess, 'my_model', global_step=step)
# Can also save only specific variables:
saver = tf.train.Saver({'my_var': my_variable})

"""
注意，Saver()操作也可以传参数。它能接收变量和张量的字典来保存指定元素，也可以接收checkpoint_every_n_hour参数来按时间规则保存操作，而不是按迭代次数。默认保存操作只保存最近的五个模型（考虑存储空间），但是也可以通过max_to_keep选项改变（默认值为5)。
"""

# 4. 在保存算法模型之前，确保为模型重要的操作命名。如果不提前命名，TensorFlow很难加载指定占位符、操作或者变量。TensorFlow的大部分操作和函数都接受name参数
conv_weights = tf.Variable(tf.random_normal(), name='conv_weights')
loss = tf.reduce_mean(..., name='loss')

# 5. TensorFlow的tf.apps.flags库使得执行命令行参数解析相当容易。你可以定义string、float、integer或者boolean型的命令行参数。运行tf.app.run()函数即可运行带有flag参数的main()函数
tf.app.flags.DEFINE_string('worker_locations', '', 'List of worker address.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning training generations.')
tf.app.flags.DEFINE_boolean('run_unit_tests', False, 'If true, run tests.')
# Need to define a 'main' function for the app to run
def main(_):
    worker_ips = FALGS.worker_locations.split(',')
    learing_rate = FLAGS.learning_rate
    generations = FLAGS.generations
    run_unit_tests = FLAGS.run_unit_tests
# Run the Tensorflow app
if __name__ == '__main__':
    tf.app.run()

# 6. TensorFlow有内建的loggint设置日志级别。其日志级别可设置为DEBUG、INFO、WARN、ERROR和FATAL，默认级别为WARN
tf.loggint.set_verbosity(tf.loggin.WARN)
# WARN is the default value, but to see more information, you can set it to
# INFO or DEBUG
tf.loggint.set_verbosity(tf.logging.DEBUG)




