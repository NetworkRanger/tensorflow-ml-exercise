#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/22 下午4:23

# 10.4 分布式TensorFlow实践

# 1. 加载TensorFlow，定义两个本地worker（端口分别为2222和2223）
import tensorflow as tf
sess = tf.Session()
# Cluster for 2 local workers (tasks 0 and 1):
cluster = tf.train.ClusterSpec({'local': ['localhost:2222', 'localhost:2223']})

# 2. 将两个worker加入到集群中，并标记task数字
server = tf.train.Server(cluster, job_name='local', task_index=0)
server = tf.train.Server(cluster, job_name='local', task_index=1)

# 3. 现在我们为每个worker分配一个task。第一个worker将初始化两个矩阵（每个是25x25维度）。第二个worker计算每个矩阵所有元素的和。然后自动分配将两个和求和的任务，并打印出结果
mat_dim = 25
matrix_list = {}
with tf.device('/job:lcoal/task:0'):
    for i in range(0,2):
        m_label = 'm_{}'.format(i)
        matrix_list[m_label] = tf.random_normal([mat_dim, mat_dim])
        # Have each worker calculate the sums
        sum_outs = {}
        with tf.device('/job:local/task:1'):
            for i in range(0,2):
                A = matrix_list['m_{}'.format(i)]
                sum_outs['m_{}'.format(i)] = tf.reduce_sum(A)
            # Sum all the sums
            summed_out = tf.add_n(list(sum_outs.values()))
            result = sess.run(summed_out)
            print('summed Values: {}'.format(result))

# 4. 运行下面的命令

"""
$ python3 parallelizing_tensorflow.py
I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:197]
Initialize GrpcChannelCache for job lcoal -> {0 -> localhost:2222, 1 -> localhost:2223}
I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:206]
Started server with target: grpc://localhost:2222
I tensorflow/core/distributed_runtime/master_session.cc:928] Start master session 252bb6f530553002 with config:
Summed Values: -21.12611198425293
"""

