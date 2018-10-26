#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/10/26 下午11:09

# 1.8.2.3 波士顿房价数据(Boston Housing data)。

import requests

housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV0']
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split('') if len(x) >= 1] for y in housing_file.text.split('\n') if len(y) >= 1]
print(len(housing_data)) # 506
print(len(housing_data[0])) # 14
