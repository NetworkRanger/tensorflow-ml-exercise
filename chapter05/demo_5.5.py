#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/7 下午9:56

# 5.5 用TensorFlow 实现地址匹配

# 1. 先导入必要的编程库
import random
import string
import numpy  as np
import tensorflow as tf

# 2. 创建参考数据集
n = 10
street_names = ['abbey', 'baker', 'canal', 'donner', 'elm']
street_types = ['rd', 'st', 'ln', 'pass', 'ave']
rand_zips = [random.randint(65000, 65999) for i in range(5)]
numbers = [random.randint(1, 9999) for i in range(n)]
streets = [random.choice(street_names) for i in range(n)]
street_suffs = [random.choice(street_types) for i in range(n)]
zips = [random.choice(rand_zips) for i in range(n)]
full_streets = [str(x) + ' ' + y + ' ' + z for x,y,z in zip(numbers, streets, street_suffs)]
reference_data = [list(x) for x in zip(full_streets, zips)]

# 3. 为了创建一个测试数据集，我们需要一个随机创建"打印错误"的字符函数，然后返回结果字符串
def create_typo(s, prob=0.75):
    if random.uniform(0,1) < prob:
        rand_ind = random.choice(range(len(s)))
        s_list = list(s)
        s_list[rand_ind] = random.choice(string.ascii_lowercase)
        s = ''.join(s_list)
    return (s)

typo_streets = [create_typo(x) for x in streets]
typo_full_streets = [str(x) + ' ' + y + ' ' + z for x,y,z in zip(numbers, typo_streets, street_suffs)]
test_data = [list(x) for x in zip(typo_full_streets, zips)]

# 4. 初始化一个计算图会话，声明所需占位符
sess = tf.Session()
test_address = tf.sparse_placeholder(dtype=tf.string)
test_zip = tf.placeholder(shape=[None, 1], dtype=tf.float32)
ref_address = tf.sparse_placeholder(dtype=tf.string)
ref_zip = tf.placeholder(shape=[None, n], dtype=tf.float32)

# 5. 声明数值的邮政编码距离和地址字符串的编辑距离
zip_dist = tf.square(tf.subtract(ref_zip, test_zip))
address_dist = tf.edit_distance(test_address, ref_address, normalize=True)

# 6. 把邮政编码距离和地址距离转换成相似度
zip_max = tf.gather(tf.squeeze(zip_dist), tf.argmax(zip_dist, 1))
zip_min = tf.gather(tf.squeeze(zip_dist), tf.argmin(zip_dist, 1))
zip_sim = tf.divide(tf.subtract(zip_max, zip_dist), tf.subtract(zip_max, zip_min))
address_sim = tf.subtract(1., address_dist)

# 7. 结合上面两个相似度函数，并对其进行加权平均
address_weight = 0.5
zip_weight = 1. - address_weight
weighted_sim = tf.add(tf.transpose(tf.multiply(address_weight, address_sim)), tf.multiply(zip_weight, zip_sim))
top_match_index = tf.argmax(weighted_sim, 1)

# 8. 为了在TensorFlow中使用编辑距离，我们必须把地址字符串转换成稀疏向量
def sparse_from_word_vec(word_vec):
    num_words = len(word_vec)
    indices = [[xi, 0, yi] for xi, x in enumerate(word_vec) for yi,y in enumerate(x)]
    chars = list(''.join((word_vec)))
    # Now we return our sparse vector
    return (tf.SparseTensorValue(indices, chars, [num_words, 1, 1]))

# 9. 分离参考集中的地址和邮政编码，然后在遍历迭代中占位符赋值
reference_address = [x[0] for x in reference_data]
reference_zips = np.array([[x[1] for x in reference_data]])

# 10. 利用步骤8中创建的函数将参考地址转换为稀疏矩阵
sparse_ref_set = sparse_from_word_vec(reference_address)

# 11. 遍历循环测试集和每项，返回参考集中最接近的索引，打印出测试集和参考集的每项
for i in range(n):
    test_address_entry = test_data[i][0]
    test_zip_entry = [[test_data[i][1]]]

    # Create sparse address vectors
    test_address_repeated = [test_address_entry] * n
    sparse_test_set = sparse_from_word_vec(test_address_repeated)

    feeddict = {test_address: sparse_test_set, test_zip: test_zip_entry, ref_address: sparse_ref_set, ref_zip: reference_zips}
    best_match = sess.run(top_match_index, feed_dict=feeddict)
    best_street = np.array(reference_address)[best_match]
    [best_zip] = reference_zips[0][best_match]
    [[test_zip_]] = test_zip_entry
    print('Address: ' + str(test_address_entry) + ', ' + str(test_zip_))
    print('Match: ' + str(best_street) + ', ' + str(best_zip))

# 12. 输出结果如下
"""
Address: 9125 donnem ln, 65746
Match: ['9125 donner ln'], 65746
Address: 5867 baker pass, 65746
Match: ['5867 baker pass'], 65746
Address: 6799 nlm ave, 65913
Match: ['6799 elm ave'], 65913
Address: 6258 baket rd, 65913
Match: ['6258 baker rd'], 65913
Address: 6807 abbyy pass, 65192
Match: ['6807 abbey pass'], 65192
Address: 8979 canql st, 65530
Match: ['8979 canal st'], 65530
Address: 4029 abbey ln, 65530
Match: ['4029 abbey ln'], 65530
Address: 1836 abbed st, 65314
Match: ['1836 abbey st'], 65314
Address: 9946 alm st, 65913
Match: ['9946 elm st'], 65913
Address: 2954 canas rd, 65530
Match: ['2954 canal rd'], 65530
"""