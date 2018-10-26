#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/10/26 下午11:40

# 1.8.2.6 影评样本数据集。

import requests
import io
import tarfile

movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
r = requests.get(movie_data_url)

# Stream data into temp object
stream_data = io.BytesIO(r.content)
tmp = io.BytesIO

while True:
    s = stream_data.read(16384)
    if not s:
        break
    tmp.write(s)

stream_data.close()
tmp.seek(0)

# Extract tar file
tar_file = tarfile.open(fileobj=tmp, mode="r:gz")
pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')

# Save pos/neg reviews (Also deal with encoding)
pos_data = []
for line in pos:
    pos_data.append(line.decode('ISO-8859-1')).encode('ascii', errors='ignore').decode()

neg_data = []
for line in pos:
    neg_data.append(line.decode('ISO-8859-1')).encode('ascii', errors='ignore').decode()

tar_file.close()

print(len(pos_data)) # 5331
print(len(neg_data)) # 5331

# Print out first negative review
print(neg_data[0]) # simplistic, silly and tedious