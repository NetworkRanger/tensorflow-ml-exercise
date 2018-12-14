#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/14 下午11:39

# 8.5 用TensorFlow实现模型仿大师绘画

# 1. 下载预训练的网络，存为.mat文件格式。mat文件格式是一种matlab对象，利用Python的scipy模块读取该文件。下面是下载mat对象的链接，该模型保存在Python脚本在同一文件夹下。
# http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat

# 2. 导入必要的编程库
import os
import scipy.misc
import numpy as np
import tensorflow as tf

# 3. 开始创建计算图会话，声明两幅图片（原始图片和风格图片）的位置。我们将使用封面作为原始图片；梵高的大作《Starry Night》作为风格图片。这两幅图片可以在GitHub(https://github.com/nfmcclure/tensorflow_cookbook)上下载
sess = tf.Session()
original_image_file = 'temp/book_cover.jpg'
style_image_file = 'temp/starry_night.jpg'

# 4. 设置模型参数: mat文件位置、网络权重、学习率、迭代次数和输出中间图片的频率。该权重可以增加应用于原始图片中风格图片的权重。这些参数可以根据实际需求稍微做出调整
vgg_path = 'imagenet-vgg-verydeep-19.mat'
original_image_weight = 5.0
style_image_weight = 200.0
regulariztion_weight = 50.0
learning_rate = 0.1
generations = 10000
output_generations = 500

# 5. 使用scipy模块加载两幅图片，并将风格图片的维度调整的和原始图片一致
original_image = scipy.misc.imread(original_image_file)
style_image = scipy.misc.imread(style_image_file)
# Get shape of target and make the style image the same
target_shape = original_image.shape
style_image = scipy.misc.imresize(style_image, target_shape[1]/style_image_file.shape[1])


# 6. 从文章中获知，我们能定义各层出现的顺序，本例中使用文章作者约定的名称
vgg_layers = [
    'conv1_1', 'relu1_1',
    'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1',
    'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1',
    'conv3_2', 'relu3_2',
    'conv3_3', 'relu3_3',
    'conv4_4', 'relu4_4', 'pool3',
    'conv4_1', 'relu4_1',
    'conv4_2', 'relu4_2',
    'conv4_3', 'relu4_3',
    'conv4_4', 'relu5_4', 'pool4',
    'conv5_1', 'relu5_1',
    'conv5_2', 'relu5_2',
    'conv5_3', 'relu5_3',
    'conv5_4', 'relu5_4',
]

# 7. 定义函数抽取mat文件中的参数
def extract_net_info(path_to_params):
    vgg_data = scipy.io.loadmat(path_to_params)
    normalization_matrix = vgg_data['normalization'][0][0][0]
    mat_mean = np.mean(normalization_matrix, axis=(0,1))
    network_weights = vgg_data['layers'][0]
    return (mat_mean, network_weights)

# 8. 基于上述加载的权重和网络层定义，通过TensorFlow的内建函数来创建网络。迭代训练层，并分配合适的权重和偏置
def vgg_network(network_weights, init_merge):
    network = {}
    image = init_merge
    for i, layer in enumerate(vgg_layers):
        if layer[1] == 'c':
            weights, bias = network_weights[i][0][0][0][0]
            weights = np.transpose(weights, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            conv_layer = tf.nn.conv2d(image, tf.constant(weights), (1, 1, 1, 1), 'SAME')
            image = tf.nn.bias_add(conv_layer, bias)
        elif layer[1] == 'r':
            image = tf.nn.relu(image)
        else:
            image = tf.nn.max_pool(image, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
        network[layer] = image
    return (network)

# 9. 参考文章中推荐了为原始图片和风格图片分配中间层的一些策略。在本例中，原始图片采用relu4_2层，风格图片采用reluX_1层组合
original_layer = 'relu4_2'
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

# 10. 运行extract_net_info()函数获取网络权重和平均值。在图片的起始位置增加一个维度，调整图片的形状为四维。TensorFlow的图像操作是针对四维的，所以需要增加维度
normalization_mean, network_weights = extract_net_info(vgg_path)
shape = (1,) + original_image_file.shape
original_features = {}
style_features = {}

# 11. 声明image占位符，并创建该占位符的网络
image = tf.placeholder('float', shape=shape)
vgg_net = vgg_network(network_weights, image)

# 12. 归一化原始图片矩阵，接着运行网络
original_minus_mean = original_image - normalization_mean
original_norm = np.array([original_minus_mean])
original_features[original_layer] = sess.run(vgg_net[original_layer], feed_dict={image: original_norm})

# 13. 为步骤9中选择的每个风格层重复上述过程
image = tf.placeholder('float', shape=style_shape)
vgg_net = vgg_network(network_weights, image)
style_minus_mean = style_image - normalization_mean
style_norm = np.array([style_minus_mean])
for layer in style_layers:
    layer_output = sess.run(vgg_net[layer], feed_dict={image: style_norm})
    layer_output = np.transpose(layer_output, (-1, layer_output.shape[3]))
    style_gram_matrix = np.matmul(layer_output.T, layer_output) / layer_output.size
    style_features[layer] = style_gram_matrix

# 14. 为了创建综合的图片，我们开始加入随机噪声，并运行网络
initial = tf.random_normal(shape)*0.05
image = tf.Variable(initial)
vgg_net = vgg_network(network_weights, images)

# 15. 声明第一个损失函数，该损失函数是原始图片，定义为步骤9中选择的原始图片的relu4_2层输出与步骤12中归一化原始图片的输出的差值的1.2范数
original_loss = original_image_weight * (2*tf.nn.l2_loss(vgg_net[original_layer] - original_features[original_layer])/original_features[original_layer].size)

# 16. 为风格图片的每个层计算损失函数
style_loss = 0
style_losses = []
for style_layer in style_layers:
    layer = vgg_net[style_layer]
    feats, height, width, channels = [x.value for x in layer.get_shape()]
    size = height * width * channels
    features = tf.reshape(layer, (-1, channels))
    style_gram_matrix = tf.matmul(tf.transpose(features).features) / size
    style_expected = style_features[style_layer]
    style_losses.append(2 * tf.nn.l2_loss(style_gram_matrix - style_expected) / style_expected.size)
    style_loss += style_image_weight * tf.reduce_sum(style_losses)

# 17. 第三个损失项成为总变分为损失，该损失函数来自于总变分的计算。其相似于总变分去噪，真实图片有较低的局部变分，噪声图片具有较高的局部变分。下面代码中的关键部分是second_term_numerator, 其减去附近的像素，高噪声的图片有较高的变分。我们最小化损失函数。
total_var_x = sess.run(tf.reduce_prod(image[:,1:,:,:].get_shape()))
total_var_y = sess.run(tf.reduce_prod(image[:,:, 1:,:].get_shape()))
first_term = regulariztion_weight * 2
second_term_numerator = tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:])
second_term = second_term_numerator/total_var_y
third_term = (tf.nn.l2_loss(image[:,:1:,:] - image[:,:,:shape[2]-1,:]) / total_var_x)
total_variation_loss = first_term * (second_term + third_term)

# 18. 最小化总的损失函数。其中，总的损失函数是原始图片损失、风格图片损失和总变分损失的组合
loss = orginal_loss + style_loss + total_variation_loss

# 19. 声明优化器函数，初始化所有模型变量
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(loss)
sess.run(tf.global_variables_initializer())

# 20. 遍历迭代训练模型，频繁地打印更新的状态并保存临时图片文件。因为运行该算法的速度依赖于图片的选择，所以需要保存临时图片。在迭代次数较大的情况下，当临时图片显示训练的结果足够时，我们可以随时停止该训练过程
for i in range(generations):
    sess.run(train_step)
    # Print update and save temporary output
    if (i+1)%output_generations == 0:
        print('Generation {} out of {}'.format(i+1, generations))
        image_eval = sess.run(image)
        best_image_add_mean = image_eval.reshape(shape[1:i]) + normalization_mean
        output_file = 'temp_output_{}.jpg'.format(i)
        scipy.misc.imsave(output_file, best_image_add_mean)

# 21. 算法训练结束，我们将保存最后的输出结果
image_eval = sess.run(image)
best_image_add_mean = image_eval.reshape(shape[1:]) + normalization_mean
output_file = 'final_output.jpg'
scipy.misc.imsave(output_file, best_image_add_mean)
