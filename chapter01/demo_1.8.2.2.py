#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/10/26 下午11:04

# 1.8.2.2 出生体重数据(Birth weight data)。

import requests

birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file = requests.get(birthdata_url)
print(birth_file)
birth_data = birth_file.text.split('\'r\n')[5:]
birth_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
print(len(birth_data)) # 189
print(len(birth_data[0])) # 11