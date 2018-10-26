#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/10/26 下午11:54

# 1.8.2.9 英德句子翻译样本集。

import requests
import io
from zipfile import ZipFile

sentense_url = 'http://www.manythings.org/anki/deu-eng.zip'
r = requests.get(sentense_url)
z = ZipFile(io.BinaryIO(r.content))
file = z.read('edu.txt')

# Format Data
eng_ger_data = file.decode()
eng_ger_data = eng_ger_data.encode('ascii', errors='ignore')
eng_ger_data = eng_ger_data.decode().split('\n')
eng_ger_data = [x.split('\t') for x in range(eng_ger_data) if len(x) >= 1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]

print(len(english_sentence)) # 137673
print(len(german_sentence)) # 137673
print(eng_ger_data[10]) # ['I Won! , 'Ich habe gewonen!']