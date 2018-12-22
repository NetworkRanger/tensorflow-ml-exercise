#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/22 下午7:21

# 11.4 用TensorFlow实现遗传算法

# 1. 导入必要的编程库
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 2. 接下来，我们设置遗传算法的参数。在本例中，有100个个体，每个长度为50。选择的百分比是20%，即适应度排序前20的个体。变异定义为特征数的倒数，这意味着下一代种群的一个特征会变异。运行遗传算法迭代200次
pop_size = 100
features = 50
selection = 0.2
mutation = 1./features
generations = 200
num_parents = int(pop_size*selection)
num_children = pop_size - num_parents

# 3. 初始化计算图会话，创建ground truth函数，该函数用来计算适应度
sess = tf.Session()
# Create ground truth
truth = np.sin(2*np.pi*(np.arange(features, dtype=np.float32))/features)

# 4. 使用TensorFlow的变量（随机下态分布输入）初始化种群
population = tf.Variable(np.random.randn(pop_size, features), dtype=tf.float32)

# 5. 现在创建遗传算法的占位符。该占位符是为ground truth和每次迭代改变的数据。因为我们希望父代变化和变异概率变化交叉，这些都是模型的占位符
truth_ph = tf.placeholder(tf.float32, [1, features])
crossover_mat_ph = tf.placeholder(tf.float32, [num_children, features])
mutation_val_ph = tf.placeholder(tf.float32, [num_children, features])

# 6. 计算群体的适应度（均方误差的负值），选择高适应度的个体
fitness = -tf.reduce_mean(tf.square(tf.subtract(population, truth_ph)), 1)
top_vals, top_ind = tf.nn.top_k(fitness, k=pop_size)

# 7. 为了获得最后的结果并绘图，我们希望检索种群中适应度最高的个体
best_val = tf.reduce_min(top_vals)
best_ind = tf.arg_min(top_vals, 0)
best_individual = tf.gather(population, best_ind)

# 8. 排序父种群，截取适应度较高的个体作为下一代
population_sorted = tf.gather(population, top_ind)
parents = tf.slice(population_sorted, [0, 0], [num_parents, features])

# 9. 通过创建两个随机shuffle的父种群矩阵来创建下一代种群。将交叉矩阵分别与1和0相加，然后与父种群矩阵相乘，生成每一代的占位符
# Indices to shuffle-gather parents
rand_parent1_ix = np.random.choice(num_parents, num_children)
rand_parent2_ix = np.random.choice(num_parents, num_children)
# Gather parents by shuffled indices, expand back out to pop_size too
rand_parent1 = tf.gather(parents, rand_parent1_ix)
rand_parent2 = tf.gather(parents, rand_parent2_ix)
rand_parent1_sel = tf.multiply(rand_parent1, crossover_mat_ph)
rand_parent2_sel = tf.multiply(rand_parent2, tf.subtract(1., crossover_mat_ph))
children_after_sel = tf.add(rand_parent1_sel, rand_parent2_sel)

# 10. 最后一个步骤是变异下一代，本例将增加一个随机正常值到下一代种群矩阵的特征分数的倒数，然后将这个矩阵和父种群连接
mutated_children = tf.add(children_after_sel, mutation_val_ph)
# Combine children and parents into new popluation
new_population = tf.concat(0, [parents, mutated_children])

# 11. 模型的最后一步是，使用TensorFlow的group()操作分配下一代种群到父一代种群的变量
step = tf.group(population.assign(new_population))

# 12. 初始化模型变量
init = tf.global_variables_initializer()
sess.run(init)

# 13. 迭代训练模型，再创建随机交叉矩阵和变异矩阵，更新每代的种群
for i in range(generations):
    # Create cross-over matrices for plugging in.
    crossover_mat = np.ones(shape=[num_children, features])
    crossover_point = np.random.choice(np.arange(1, features-1, step=1), num_children)
    for pop_ix in range(num_parents):
        crossover_mat[pop_ix, 0:crossover_point[pop_ix]] = 0.
    # Generate mutation probability matrices
    mutation_prob_mat = np.random.uniform(size=[num_children, features])
    mutation_values = np.random.normal(size=[num_children. features])
    mutation_values[mutation_prob_mat >= mutation] = 0

    # Run GA step
    feed_dict = {truth_ph: truth.reshape([1,features]), crossover_mat_ph: crossover_mat, mutation_val_ph: mutation_values}
    step.run(feed_dict, sess=sess)
    best_individual_val = sess.run(best_individual, feed_dict=feed_dict)

    if i%5 == 0:
        best_fit = sess.run(best_val, feed_dict = feed_dict)
        print('Generation: {}, Best Fitness (lower MSE): {:.2}'.format(i, -best_fit))

# 14. 输出结果如下

"""
"""
