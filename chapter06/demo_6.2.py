#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/8 下午2:43

# 6.2 使用TensorFlow实现门函数

# 1. 加载TensorFlow模块，创建一个计算图会话
import tensorflow as tf
sess = tf.Session()

# 2. 声明模型变量、输入数据集和占位符。本例输入数据为5，所以乘法因为为10，可以得到50的预期值(5*10=10)
a = tf.Variable(tf.constant(4.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)

# 3. 增加操作到计算图中
multiplication = tf.multiply(a, x_data)

# 4. 声明损失函数：输出结果与预期目标值(50)之间的L2距离函数
loss = tf.square(tf.subtract(multiplication, 50.))

# 5. 初始化模型变量，声明标准梯度下降优化算法
init = tf.initialize_all_variables()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# 6. 优化模型输出结果。连续输入值5， 反向传播损失函数来更新模型变量以达到值10
print('Optimizing a Multiplication Gate Output to 50.')
for i in range(10):
    sess.run(train_step, feed_dict={x_data: x_val})
    a_val = sess.run(a)
    mult_output = sess.run(multiplication, feed_dict={x_data: x_val})
    print(str(a_val) + ' * ' + str(x_val) + ' = ' + str(mult_output))

# 7. 输出结果如下
"""
Optimizing a Multiplication Gate Output to 50.
7.0 * 5.0 = 35.0
8.5 * 5.0 = 42.5
9.25 * 5.0 = 46.25
9.625 * 5.0 = 48.125
9.8125 * 5.0 = 49.0625
9.90625 * 5.0 = 49.53125
9.953125 * 5.0 = 49.765625
9.9765625 * 5.0 = 49.882812
9.988281 * 5.0 = 49.941406
9.994141 * 5.0 = 49.970703
"""

# 8. 对两个嵌套操作的例子f(x) = a*x+b, 也执行上述相同的步骤
# 9. 开始第二个例子，不同在于本例中包含两个模型变量： a和b
from tensorflow.python import ops
ops.reset_default_graph()
sess = tf.Session()

a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)

two_gate = tf.add(tf.multiply(a, x_data), b)

loss = tf.square(tf.subtract(two_gate, 50.))

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

init = tf.initialize_all_variables()
sess.run(init)

# 10. 优化模型变量，训练输出结果，以达到预期目标值50
print('\nOptimizing Two Gate Output to 50.')
for i in range(10):
    # Run the train step
    sess.run(train_step, feed_dict={x_data: x_val})
    # Get teh a and b values
    a_val, b_val = (sess.run(a), sess.run(b))
    # Run the two-gate graph output
    two_gate_output = sess.run(two_gate, feed_dict={x_data: x_val})
    print(str(a_val) + ' * ' + str(x_val) + ' + ' + str(b_val) + ' = ' + str(two_gate_output))

# 11. 输出结果如下
"""
Optimizing Two Gate Output to 50.
5.4 * 5.0 + 1.88 = 28.88
7.512 * 5.0 + 2.3024 = 39.8624
8.52576 * 5.0 + 2.5051522 = 45.133953
9.012364 * 5.0 + 2.6024733 = 47.664295
9.2459345 * 5.0 + 2.6491873 = 48.87886
9.358048 * 5.0 + 2.67161 = 49.461853
9.411863 * 5.0 + 2.682373 = 49.74169
9.437695 * 5.0 + 2.687539 = 49.87601
9.450093 * 5.0 + 2.690019 = 49.940483
9.456045 * 5.0 + 2.6912093 = 49.971436
"""

"""
这里需要注意的是，第二个例子的解决方法不是唯一的。这在神经网络算法中不太重要，因为所有的参数是根据减小损失函数来调整的。最终的解决方案依赖于a和b的初始值。如果它们是随机初始化的，而不是1，我们将会看到每次迭代的模型变量的输出结果并不相同。
"""