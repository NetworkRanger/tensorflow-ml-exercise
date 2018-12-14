#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/12 下午11:09

# 1. 导入必要的编程库，创建一个计算图会话
import os
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib
sess = tf.Session()

# 2. 声明一些参数。将训练集和测试集的批量大小设为128。我们总共迭代20000次，并且每迭代50次打印出状态值。每迭代500次，我们将在测试集的批量数据上进行模型评估。设置图片长度和宽度，以及随机裁剪图片的大小。颜色通道设为3通道（红色，绿色和蓝色），目标分类设为10类。最后声明存储数据和批量图片的位置
batch_size = 128
output_every = 50
generations = 20000
eval_every = 500
image_height = 32
image_width = 32
crop_height = 24
crop_width = 24
num_channels = 3
num_targets = 10
data_dir = 'temp'
extract_folder = 'cifar-10-batches-bin'

# 3. 推荐降低学习率来训练更好的模型，所以我们采用指数级减小学习率：学习率初始值设为0.1，每迭代250次指数级减少学习率，因为为10%。公式为：0.1*0.9*(x/250)，其中x是当前迭代的次数。TensorFlow默认是连接减小学习率，但是也接受阶梯式更新学习率
learning_rate = 0.1
ly_deccay = 0.9
num_gens_to_wait = 250.

# 4. 设置读取二进制CIFA-10图片的参数
image_vec_length = image_height*image_width*num_channels
record_length = 1 + image_vec_length

# 5. 设置下载CIFAR-10图像数据集的URL和数据目录
data_dir = 'temp'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
cifar10_url = 'http://www.cs.toronto.edu/-kirz/cifar-10-binary.tar.gz'
data_file = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
if not os.path.isfile(data_file):
    # Download file
    filepath, _ = urllib.request.urlretrieve(cifar10_url, data_file, progress)
    # Extract file
    tarfile.open(filepath, 'r:gz').extractall(data_dir)
# 6. 使用read_cifar_files()函数建立图片读取器，返回一个随机打乱的图片。首先，声明一个读取固定字节长度的读取器；然后从图像队列中读取图片，抽取图片并标记；最后使用TensorFlow内建的图像修改函数随机的打乱图片
def read_cifar_files(filename_queue, distort_images=True):
    reader = tf.FixedLengthRecordReader(record_bytes=record_length)
    key, record_string = reader.read(filename_queue)
    record_bytes = tf.decode_raw(record_string, tf.uint8)
    # Extract label
    image_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
    # Extract image
    image_extracted = tf.reshape(tf.slice(record_bytes, [1], [image_vec_length]), [num_channels, image_height, image_width])
    # Reshape image
    image_uint8image = tf.transpose(image_extracted, [1, 2, 0])
    reshaped_image = tf.cast(image_uint8image, tf.float32)
    # Randomly Crop image
    final_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, crop_width, crop_height)
    if distort_images:
        # Ranomly flip the image horizontally, change the brightness and contrast
        final_image = tf.image.random_flip_left_right(final_image)
        final_image = tf.image.random_brightness(final_image, max_delta=63)
        final_image = tf.image.random_contrast(final_image, lower=0.2, upper=1.8)
    # Normalize whitening
    final_image = tf.image.per_image_standardization(final_image)
    return (final_image, image_label)

# 7. 声明批量处理使用的图像管道填充函数。首先，需要建立读取图片的列表，定义如何用TensorFlow内建函数创建的input producer对象读取这些图片列表。把input producer传入上一步创建的图片读取函数read_cifar_files()中。然后创建图像队列的批量读取器，shuffle_batch()。
def input_pipeline(batch_size, train_logical=True):
    if train_logical:
        files = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1,6)]
    else:
        files = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
    filename_queue = tf.train.string_input_producer(files)
    image, label = read_cifar_files(filename_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size, capacity, min_after_dequeue)
    return (example_batch, label_batch)

"""
设置合适的min_after_dequeue值是相当重要的。该参数是设置抽样图片缓存最小值。TensorFlow官方文档推荐设置为(#threads + erros margin)*batch_size。注意，该参数设置太大会导致更多的shuffle。从图像队列中shuffle大的图像数据集需要更多的内存。
"""

# 8. 声明模型函数。本例的模型使用两个卷积层，接着是三个全连接层。为了便于声明模型变量，我们将定义两个变量函数。两层卷积操作各创建64个特征。第一个全连接层连接第二个卷积层，有384个隐藏节点。第二个连接层连接刚才的384个隐藏节点到192个隐藏节点。最后的隐藏层操作连接192个隐藏节点到10个输出分类。
def cifar_cnn_model(input_images, batch_size, train_logical=True):
    def truncated_normal_var(name, shape, dtype):
        return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))
    def zero_var(name, shape, dtype):
        return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))
    # First Convolutional Layer
    with tf.variable_scope('conv1') as scope:
        # Conv_kernel is 5x5 for all 3 colors and we will create 64 features
        conv1_kernel = truncated_normal_var(name='conv_kernel1', shape=[5, 5, 3, 64], dtype=tf.float32)
        # We convolve across the image with a stride size of 1
        conv1 = tf.nn.conv2d(input_images, conv1_kernel, [1, 1, 1, 1], padding='SAME')
        # Initialize and add the bias term
        conv1_bias = zero_var(name='conv_bias1', shape=[64], dtype=tf.float32)
        conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
        # ReLU element wise
        relu_conv1 = tf.nn.relu(conv1_add_bias)
    # Max Pooling
    pool1 = tf.nn.max_pool(relu_conv1, kisze=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer1')

    # Local Response Normalization
    norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')
    # Second Convolutional Layer
    with tf.variable_scope('conv2') as scope:
        # Conv kernel is 5x5, across all prior 64 features and we create 64 more features
        conv2_kernel = truncated_normal_var(name='conv_kernel2', shape=[5, 5, 64, 64], dtype=tf.float32)
        # Convolve filter across prior output with stride size of 1
        conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
        # Initialize and add the bias
        conv2_bias = zero_var(name='conv_bias2', shape=[64], dtype=tf.float32)
        conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
        # ReLU element wise
        relu_conv2 = tf.nn.relu(conv2_add_bias)
    # Max Pooling
    pool2 = tf.nn.max_pool(relu_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer2')
    # Local Response Normaliztion (parameters from paper)
    norm2 = tf.nn.lrn(pool2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')
    # Reshape ouput into a single matrix for multiplication for the fully connected layers
    reshaped_output = tf.reshape(norm2, [batch_size, -1])
    reshaped_dim = reshaped_output.get_shape()[1].value

    # First Fully Connected Layer
    with tf.variable_scope('full1') as scope:
        # Fully connected layer will have 384 outputs.
        full_weight1 = truncated_normal_var(name='full_mult1', shape=[reshaped_dim, 384], dtype=tf.float32)
        full_bias1 = zero_var(name='full_bias1', shape=[384], dtype=tf.float32)
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full_weight1), full_bias1))
    # Second Fully Connected Layer
    with tf.variable_scope('full2') as scope:
        # Second fully connected layer has 192 outputs.
        full_weight2 = truncated_normal_var(name='full_mult2', shape=[384, 192], dtype=tf.float32)
        full_bias2 = zero_var(name='full_bias2', shape=[192], dtype=tf.float32)
        full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))
    # Final Fully Connected Layer -> 10 categories for output (num_targets)
    with tf.variable_scope('full3') as scope:
        # Finaly fully connected layer has 10 (num_targets) outputs.
        full_weight3 = truncated_normal_var(name='full_mult3', shape=[192, num_targets], dtype=tf.float32)
        full_bias3 = zero_var(name='full_bias3', shape=[num_targets], dtype=tf.float32)
        final_output = tf.add(tf.matmul(full_layer2, full_weight3), full_bias3)
    return (final_output)

# 9. 创建损失函数。本例使用softmax损失函数，因为一张图片应该属于其中一个类别，所以输出结果应该是10类分类的概率分布
def cifar_loss(logits, targets):
    # Get rid of extra demensions and cast targets into integers
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Calculate cross entropy from logits and targets
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
    # Take the average loss across batch size
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return (cross_entropy_mean)

# 10. 定义训练步骤函数。在训练步骤中学习率将指数级减小
def train_step(loss_value, generation_num):
    # Our learning rate is an exponential decay (stepped down)
    model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num, num_gens_to_wait, lr_decay, staircase=True)
    # Create optimizer
    my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
    # Initialize train step
    train_step = my_optimizer.minimize(loss_value)
    return (train_step)

# 11. 创建变量图片的准确度函数。该函数输入logits和目标向量，输出平均准确度。训练批量图片和测试批量图片都可以使用该准确函数
def accuracy_of_batch(logits, targets):
    # Make sure targets are intergers and drop extra dimensions
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Get predicated values by finding which logit is the greatest
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    # Check if they are equal across the batch
    predicted_correctly = tf.equal(batch_predictions, targets)
    # Average the 1's and 0's (True's and False's) across the batch size
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return (accuracy)

# 12. 有了图像管道函数input_pipleline()后，我们开始初始化训练图像管道和测试图像管道
images, targets = input_pipeline(batch_size, train_logical=True)
test_images, test_targets = input_pipeline(batch_size, tarin_logical=False)

# 13. 初始化训练模型。值得注意的是，需要在创建训练模型时声明scope.reuse_variables()，这样可以在创建测试模型时重用训练模型相同的模型参数
with tf.variable_scope('mode_definition') as scope:
    # Declare the training network model
    model_output = cifar_cnn_model(images, batch_size)
    # Use same variables within scope
    scope.reuse_variables()
    # Declare test model output
    test_output = cifar_cnn_model(test_images, batch_size)

# 14. 初始化损失函数和测试准确度函数。然后声明迭代变量。该迭代变量需要声明为非训练型变量，并传入训练函数，用于计算学习率的指数级衰减值
loss = cifar_loss(model_output, targets)
accuracy = accuracy_of_batch(test_output, test_targets)
generation_num = tf.Variable(0, trainable=False)
train_op = train_step(loss, generation_num)

# 15. 初始化所有模型变量，然后运行TensorFlow的start_queue_runners()函数启动图像管道。图像管道通过赋值字典传入批量图片，开始训练模型和测试模型输出
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

# 16. 现在遍历迭代训练，保存训练集损失函数和测试集准确度
train_loss = []
test_accuracy = []
for i in range(generations):
    _, loss_value = sess.run([train_op, loss])
    if (i+1)%output_every == 0:
        train_loss.append(loss_value)
        output = 'Generation {}: Loss = {:.5f}'.format((i+1), loss_value)
        print(output)
    if (i+1)%eval_every == 0:
        [temp_accuracy] = sess.run([accuracy])
        test_accuracy.append(temp_accuracy)
        acc_output = ' --- Test Accuracy={:.2f}%.'.format(100.*temp_accuracy)
        print(acc_output)

# 17. 输出结果如下

"""
"""

# 18. 使用matmplotlib模块绘制损失函数和准确度
eval_indices = range(0, generations, eval_every)
output_indices = range(0, generations, output_every)
# Plot loss over time
plt.plot(output_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()
# Plot accuracy over time
plt.plot(eval_indices, test_accuracy, 'k-')
plt.title('Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.show()








