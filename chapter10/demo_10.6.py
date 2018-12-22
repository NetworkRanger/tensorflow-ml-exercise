#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/22 下午5:29

# 10.6 TensorFlow产品化的实例

# 1. 导入必要的编程库，声明TensorFlow应用的flag
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
tf.app.flags.DEFINE_string('storage_foler', 'temp', 'Where to store model and data.')
tf.app.flags.DEFINE_string('model_file', False, 'Model file location.')
tf.app.flags.DEFINE_boolean('rnn_unit_tests', False, 'If true, run tests.')
FALGS = tf.app.falgs.FLAGS

# 2. 声明文本清洗函数，在训练脚本中也有相同的清洗函数
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = ''.join(text_string.split())
    text_string = text_string.lower()
    return (text_string)

# 3. 加载词汇处理函数
def load_vocab():
    vocab_path = os.path.join(FLAGS.storage_folder, 'vocab')
    vocab_processor = tf.contrib.learn.preprocessing.vocabularyProcessor.restore(vocab_path)
    return (vocab_path)

# 4. 有了清洗的文本数据和词汇处理函数，即可创建数据处理管道
def process_data(input_data, vocab_processor):
    inpu_data = clean_text(input_data)
    input_data = input_data.split()
    processed_input = np.array(list(vocab_processor.transform(input_data)))
    return (processed_input)

# 5. 我们需要数据评估模型。我们将要求用户在屏幕上输入文本，然后处理输入文本和返回处理文本
def get_input_data():
    input_text = input('Please enter a text message to evaluate:')
    vocab_processor = load_vocab()
    return (process_data(input_text, vocab_processor))

"""
对于本例而言，我们通过要求用户输入来创建评估数据，也有许多应用通过提供文件或者API来获取数据，我们可以根据需要调整输入函数。
"""

# 6. 对于单元测试，应确保文本处理函数的行为符合预期
class clean_test(tf.test.TestCase):
    # Make sure cleaning function behaves correctly
    def clean_string(self):
        with self.test_session():
            test_input = '--Tensorflow\'s so Great! Don\'t you think so?   '
            test_expected = 'tensorflows so great don you think so'
            test_out = clean_text(test_input)
            self.assertEqual(test_expected, test_out)

# 7. 现在，有了算法模型和数据集，我们运行主函数。该主函数获取数据集，建立计算图，加载模型变量，传入处理过的数据，打印输出结果
def main(args):
    # Get flags
    storage_folder = FLAGS.storage_folder
    # Get user input text
    x_data = get_input_data()

    # Load model
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph('{}.meta'.format(os.path.join(storage_folder, 'model.ckpt')))
            saver.restore(sess, os.path.join(storage_folder, 'model.ckpt'))
            # Get the placeholders from the graph by name
            x_data_ph = graph.get_operations_by_name('x_data_ph').outputs[0]
            dropout_keep_prob = graph.get_operation_by_name('probability_outputs').outputs[0]
            # Make the precition
            eval_feed_dict = {x_data_ph: x_data, dropout_keep_prob: 1.0}
            probablity_prediction = sess.run(tf.reduce_mean(probablity_outputs, 0), eval_feed_dict)

            # Print output (Or save to file or DB connection?)
            print('Probability of Spam: {:.4}'.format(probablity_prediction[1]))

# 8. 如下代码展示了main()函数或单元测试如何运行
if __name__ == '__main':
    if FLAGS.run_unit_tests:
        # Perform unit tests
        tf.tes.main()
    else:
        # Run evaluation
        tf.app.run()
