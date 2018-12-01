#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/1 下午6:07

# 5.3 如何度量文本距离

"""
TensorFlow的文本距离度----字符串间的编辑距离(Levenshtein距离)。
Levenshtein距离是指由一个字符串转换成另一个字符串所需的最少编辑操作次数。允许的编辑操作包括插入一个字符，删除一个字符和将一个字符替换成另一个字符。使用TensorFlow的内建函数edit_distance()求解Levenshtein距离。
"""

"""
注意，TensorFlow的内建函数edit_distance()仅仅接受稀疏张量。因此，得把字符串转换成稀疏张量。
"""

# 1. 加载TensorFlow，初始化一个计算图会话
import tensorflow as tf
sess = tf.Session()

# 2. 展示如何计算两个单词'bear'和'beer'间的编辑距离。用Python的list()函数创建字符list,然后将list映射为一个三维稀疏矩阵。TensorFlow的tf.SparseTensor()函数需指定字符索引、矩阵形状和张量中的非零值。编辑距离计算时，指定normalize=False表示计算总的编辑距离: 指定normalize=True表示计算归一化编辑距离，通过编辑距离除以第二个单词的长度进行归一化

"""
TensorFlow文档把两个字符串处理为参数字符串（hypothesis)和真实字符串(ground truth)。本例标记为h张量和t张量。
"""
hypothesis = list('bear')
truth = list('bears')
h1 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3]], hypothesis, [1,1,1])
t1 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4]], truth, [1,1,1])

print(sess.run(tf.edit_distance(h1, t1, normalize=False)))

# 3. 编辑距离计算结果如下
"""
[[ 2.]]
"""

"""
TensorFlow的SparseTensorValue()函数是创建稀疏张量的方法，要传入所需创建的稀疏张量的索引、值和形状大小。
"""

# 4. 下面比较两个单词bear和beer与另一个单词beers。为了做比较，需要重复beers使得比较的单词有相同的数量
hypothesis2 = list('bearbeer')
truth2 = list('beersbeers')
h2 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,1,0], [0,1,1], [0,1,2], [0,1,3]], hypothesis2, [1,2,4])
t2 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,1,0], [0,1,1], [0,1,2], [0,1,3],[0,1,4]], truth2, [1,2,5])

print(sess.run(tf.edit_distance(h2, t2, normalize=True)))

# 5. 结果如下
"""
[[ 0.40000001 0.2       ]]
"""

# 6. 另外一种更有效地比较一个单词集合与单个单词的方法。事先为参考字符串(hypothesis)和真实字符串(ground)创建索引和字符列表
hypothesis_words = ['bear','bar','tensor','flow']
truth_word = ['beers']
num_h_words = len(hypothesis_words)
h_indices = [[xi,0,yi] for xi,x in enumerate(hypothesis_words) for yi,y in enumerate(x)]
h_chars = list(''.join(hypothesis_words))
h3 = tf.SparseTensor(h_indices, h_chars, [num_h_words, 1, 1])
truth_word_vec = truth_word*num_h_words
t_indices = [[xi,0,yi] for xi,x in enumerate(truth_word_vec) for yi,y in enumerate(x)]
t_chars = list(''.join(truth_word_vec))
t3 = tf.SparseTensor(t_indices, t_chars, [num_h_words, 1, 1])

print(sess.run(tf.edit_distance(h3, t3, normalize=True)))

# 7. 结果如下
"""
[[0.4]
 [0.6]
 [1. ]
 [1. ]]
"""

# 8. 展示如何用占位符来计算两个单词列表间的编辑距离。基本思路是一样的，不同的是现在用SparseTensorValue()替代先前稀疏张量。首先，创建一个函数，该函数根据单词列表，输出稀疏张量
def create_sparse_vec(word_list):
    num_words = len(word_list)
    indices = [[xi, 0, yi] for xi,x in enumerate(word_list) for yi,y in enumerate(x)]
    chars = list(''.join(word_list))
    return(tf.SparseTensorValue(indices, chars, [num_words,1,1]))

hyp_string_sparse = create_sparse_vec(hypothesis_words)
truth_string_sparse = create_sparse_vec(truth_word*len(hypothesis_words))

hyp_input = tf.sparse_placeholder(dtype=tf.string)
truth_input = tf.sparse_placeholder(dtype=tf.string)

edit_distance = tf.edit_distance(hyp_input, truth_input, normalize=True)
feed_dict = {hyp_input: hyp_string_sparse, truth_input: truth_string_sparse}

print(sess.run(edit_distance, feed_dict=feed_dict))

# 9. 输出结果如下
"""
[[0.4]
 [0.6]
 [1. ]
 [1. ]]
"""

