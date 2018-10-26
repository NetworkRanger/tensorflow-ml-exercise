#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/10/26 下午11:50

# 1.8.2.8 莎士比亚著作文本数据集(Shakespeare text data)。

import requests

shakespeare_url = 'https://www.guteberg.org/cache/epub/100/pg100.txt'

# Get Shakespeare text
response = requests.get(shakespeare_url)
shakespeare_file = response.content

# Deocode binary into string
shakespeare_text = shakespeare_file.decode('utf-8')

# Drop first few descriptive pargraphs.
shakespeare_text = shakespeare_text[7675:]

print(len(shakespeare_text)) # Number of characters
