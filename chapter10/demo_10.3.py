#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/22 下午4:06

# 10.3 TensorFlow的并发执行

# 1. 为了能够找到TensorFlow的什么操作正在使用什么设备，我们在计算图会话中传入一个config参数，将log_device_placement设为True。当我们在命令行运行脚本时，会看到指定设备输出
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
a = tf.constant_initializer([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Runs the op.
print(sess.run(c))

# 2. 从控制台运行下面的命令
"""
$ python3 using_multpile_devices.py
Device mapping: no known devices.
I tensorflow/core/common_runtime/direct_session.cc:175] Device mapping:
MatMul: /job:localhost/replica:0/task:0/cpu:0
b: /job:localhost/replica:0/task:0/cpu:0
I tensorflow/core/common_runtime/simple_placer.cc:818] b: /job:localhost/replica:0/task:0/cpu:0
I tensorflow/core/common_runtime/simple_placer.cc:818] a: /job:localhost/replica:0/task:0/cpu:0
[[22. 28.]
 [49. 64.]
"""

# 3. 有时，我们希望搞清楚TensorFlow正在使用的设备。当加载先前保存过的模型，并且该模型在计算图中已分配固定设备时，服务器可提供不同的设备给计算图使用。实现该功能只需在config设置软设备
config = tf.ConfigProto()
config.allow_soft_placement = True
sess_soft = tf.Session(config=config)

# 4. 当使用CPU时，TensorFlow默认占据大部分CPU内存。虽然这也是时常期望的，但是我们能谨慎分配GPU内存。当TensorFlow一直不释放GPU内存时，如有必要，我们可以设置GPU内存增长选项让GPU内存分配缓慢增大到最大限制
config.gpu_options.allow_growth = True
sess_grow = tf.Session(config=config)

# 5. 如果希望限制死TensorFlow使用GPU内存的百分比，可以使用config设置per_process_gpu_memory_fraction
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess_limited = tf.Session(config=config)

# 6. 有时，我们希望代码健壮到可以决定运行多少GPU合适。TensorFlow有内建函数可以探测到。如果我们期望代码在GPU内存合适时利用GPU计算能力，并分配指定操作给GPU，那么该功能是有益的
if tf.test.is_built_with_cuda(): pass

# 7. 我们希望分配指定操作给GPU。下面是一个示例代码，做了一些简单的计算，并将它们分配给主CPU和两个副GPU
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 3.0, 5.0], shape=[1,3])
    b = tf.constant([2.0, 4.0, 6.0], shape=[3, 1])

    with tf.device('/gpu:0'):
        c = tf.matmul(a,b)
        c = tf.reshape(c, [-1])

    with tf.device('/gpu:1'):
        d = tf.matmul(b, a)
        flat_d = tf.reshape(d, [-1])

    combined = tf.multiply(c, flat_d)
print(sess.run(combined))

