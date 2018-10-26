#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/10/26 下午11:35

# 1.8.2.5 垃圾短信文本数据集(Spam-ham text data)。

import requests
import io
from zipfile import ZipFile

zip_url = 'https://archieve.ics.ici.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
r = requests.get(zip_url)
z = ZipFile(io.BytesIO(r.content))
file = z.read('SMSSpamCollection')
text_data = file.decode()
text_data = text_data.encode('ascii', errors='ignore')
text_data = text_data.decode().split('\n')
text_data = [x.split('\t') for x in text_data if len(x) >= 1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]
print(len(text_data_train)) # 5574
print(len(text_data_train[1])) # Ok lar... Joking wif u oni...