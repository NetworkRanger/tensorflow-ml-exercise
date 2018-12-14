#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/14 下午11:02

# 8.4 再训练已有的CNN模型

# 1. 导入必要的编程库，包括下载、解压和保存CIFAR-10图片数据的编程库
import os
import tarfile
import _pickle as cPickle
import numpy as np
import urllib.request
import scipy.misc

# 2. 定义CIFAR-10图片数据链接，创建存储数据的临时文件夹，并声明图片的十个分类
cifar_link = 'https://www.cs.toronto.edu/~kirz/cifar-10-python.tar.gz'
data_dir = 'temp'
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
objects = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 3. 下载CIFAR-10.tar数据文件，并解压压缩文件
target_file = os.path.join(data_dir, 'cifar-10-python.tar.gz')
if not os.path.isfile(target_file):
    print('CIFAR-10 file not found. Downloading CIFAR data (Size = 163MB)')
    print('This may take a few minutes, please wait.')
    filename, headers = urllib.request.urlretrieve(cifar_link, target_file)
# Extraact into memory
tar = tarfile.open(target_file)
tar.extractall(path=data_dir)
tar.close()


#  4. 创建训练所需的文件夹结构。临时目录下有两个文件夹train_dir和validation_dir。每个文件夹下有10个子文件夹，分别存储10个目标分类
# Create train images folders
train_folder = 'train_dir'
if not os.path.isdir(os.path.join(data_dir, train_folder)):
    for i in range(10):
        folder = os.path.join(data_dir, train_folder, objects[i])
        os.makedirs(folder)
# Create test image folders
test_folder = 'validation_dir'
if not os.path.isdir(os.path.join(data_dir, test_folder)):
    for i in range(10):
        folder = os.path.join(data_dir, test_folder, objects[i])
        os.makedirs(folder)

# 5. 为了保存图片，我们将创建函数从内存中加载图片并存入文件夹
def load_batch_from_file(file):
    file_conn = open(file, 'rb')
    image_dictionary = cPickle.load(file_conn, encoding='latin1')
    file_conn.close()
    return (image_dictionary)

# 6. 在上一步的文件夹中，为每个目标分类保存一个文件
def save_imaeg_from_dict(image_dict, foler='data_dir'):
    for ix,label in enumerate(image_dict['labels']):
        folder_path = os.path.join(data_dir, folder, objects[label])
        filename = image_dict['filename'][ix]
        # Transform image data
        image_array = image_dict['data'][ix]
        image_array.resize([3, 32, 32])
        # Save image
        output_location = os.path.join(folder_path, filename)
        scipy.misc.imsave(output_location, image_array.transpose())

# 7. 对于上一步的函数，遍历下载数据文件，并把每个图片保存到正确的位置
data_location = os.path.join(data_dir, 'cifar-10-batches-py')
train_names = ['data_batch_'+str(x) for x in range(1,6)]
test_names = ['test_batch']
# Sort train images
for file in train_names:
    print('Saving images from file: {}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dict = load_batch_from_file(file_location)
    save_imaeg_from_dict(image_dict, folder=train_folder)
# Sort test images
for file in test_names:
    print('Saving images from file: {}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dict = load_batch_from_file(file_location)
    save_imaeg_from_dict(image_dict, folder=test_folder)

# 8. Python脚本最后部分是创建标注文件。该文件用标注（而不是数值索引）自解释输出结果
cifar_labels_file = os.path.join(data_dir, 'cifar10_labels.txt')
print('Writing labels file, {}'.format(cifar_labels_file))
with open(cifar_labels_file, 'w') as labels_file:
    for item in objects:
        labels_file.write('{}\n'.format(item))

# 9. 上面的脚本运行之后，下载图片数据集并排序归类。接着按TensorFlow官方示例操作，先复制例子源码
# git clone https://github.com/tensorflow/models/tree/master/inception/inception

# 10. 为了重用已训练好的模型，我们下载神经网络权重并应用于新神经网络模型
"""
me@computer: ~$ curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
me@computer: ~$ tar xzf inception-v3-2016-03-1.tar.gz
"""

# 11. 准备好图片文件，我们将其转为TFRecords对象
"""
me@computer: ~$ python3 data/build_image_data.py
---train_directory="temp/train_dir/"
--validation_directory="temp/validation_dir"
--output_directory="temp" --labels_file="temp/cifar10_labels.txt"
"""

# 12. 使用bazel模块训练算法模型，设置fine_tune参数为true。该脚本每迭代10次输出损失函数。我们可以随机终止进程，因为模型输出结果都保存于temp/training_results文件夹。我们能从该文件夹加载模型数据进行模型评估
"""
me@computer: ~$ bazel-bin/inception/flowers_train
--train_dir="temp/training_results" --data-dir="temp/data_dir"
--pretrained_model_checkpoint_path="model.ckpt-157585"
--fine_tune=True --initial_learning_rate=0.001
--input_queue_memory_factor=1
"""

# 13. 训练输出结果如下
"""
2016-09-18 12:16:32.563577: step 1290, loss = 2.02 (1.2 examples/sec; 26.965 sec/batch)
2016-09-18 12:25:42.316540: step 1300, loss = 2.01 (1.2 exapmles/sec; 26.357 sec/batch)
"""


