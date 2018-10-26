#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/10/26 下午10:58

# 1.8.2.1 鸢尾花卉数据集(Iris data)。

from sklearn import datasets

iris = datasets.load_iris()
print(len(iris.data)) # 150

print(len(iris.target)) # 150

print(iris.target[0]) # Sepal length, Speal width, Petal length, Petal width

# [5.1 3.5 1.4 0.2]

print(set(iris.target)) # I. setosa, I. virginica, I. versicolor

# {0, 1, 2}

