#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/10/26 下午11:20

# 1.8.3.4 MNIST手写体字库

from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(len(minst.train.images)) # 55000
print(len(minst.test.images)) # 10000
print(len(minst.validation.images)) # 5000
print(minst.train.labels[1, :]) # The first label is a '''
# [ 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]