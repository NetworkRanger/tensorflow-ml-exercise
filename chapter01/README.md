# TensorFlow 基础

## 1.1 TensorFlow介绍
Google 的TensorFlow 引擎提供了一种解决机器学习问题和高效方法。机器学习在各行各业应用广泛，特别是计算机视觉，语音识别，语言翻译和健康医疗等邻域。

## [1.2 TensorFlow 如何工作](./demo_1.2.py)

### TensorFlow 算法的一般流程
1. 导入/生成样本数据集。
2. 转换和归一化数据。
3. 划分样本数据集为训练样本集、测试样本集和验证样本集。
4. 设置机器学习参数（超参数）。
5. 初始化变量和占位符。
6. 定义模型结构。
7. 声明损失函数。
8. 初始化模型和训练模型。
9. 评估机器学习模型。
10. 调优超参数。
11. 发布/预测结果。

## [1.3 声明张量](./demo_1.3.py)

1. 固定张量
2. 相似形状的张量
3. 序列张量
4. 随机张量

## [1.4 使用占位符和变量](./demo_1.4.py)
## [1.5 操作（计算）矩阵](./demo_1.5.py)

1. 创建矩阵
2. 矩阵的加法和减法
3. 矩阵乘法函数matmul()可以通过参数指定在矩阵乘法操作前是否进行矩阵转置
4. 矩阵转置
5. 再次强调，重新初始化将会得到不同的值
6. 对于矩阵行列式，使用方式如下
7. 矩阵分解
8. 矩阵的特征值和特征向量

## [1.6 声明操作](./demo_1.6.py)

1. TensorFlow 提供div()函数的多种变种形式和相关的函数
2. 值得注意的, div()函数返回值的数据类型与输入数据类型一致
3. 如果要对浮点数进行整数除法，可以使用floordiv()函数
4. 另外一个重要的函数是mod()（取模)。此函数返回除法的余数。
5. 通过cross()函数云计算两个张量间的点积
6. 数学函数的列表

函数名|功能
-----|---
abs()|返回输入参数张量的绝对值
ceil()|返回输入参数张量的向上取整结果
cos()|返回输入参数张量的余弦值
exp()|返回输入参数疑是的自然常数e的指数
floor()|返回输入参数张量的向下取整结果
inv()|返回输入参数张量的倒数
log()|返回输入参数张量的自然对数
maximum()|返回两个输入参数张量中的最大值
minimum()|返回两个输入参数张量中的最小值
neg()|返回输入参数张量的负值
pow()|返回输入参数第一个张量的第二个张量的次幂
round()|返回输入参数张量的四舍五入结果
rsqrt()|返回输入参数张量的平方根的倒数
sign()|根据输入参数张量的符号，返回-1、0或1
sin()|返回输入参数张量的正弦值
sqrt()|返回输入参数张量的平方根
square()|返回输入参数张量的平方

7. 特殊数学函数

函数名|函数功能
-----|-------
digamma()|普西函数(Psi函数), lgamma()函数的导数
erf()|返回张量的高斯误差函数
erfc()|返回张量的互补误差函数
igamma()|返回下不完全伽马函数
igammac()|返回上不完全伽马函数
lbeta()|返回贝塔函数绝对值的自然对数
lgamma()|返回伽马函数绝对值的自然对数
squared_differenc()|返回两个张量间差值的平方

## [1.7 实现激励函数](./demo_1.7.py)

1. 整流线性单元(Rectifier linear unit, ReLU)是神经网络最常用的非线性函数
2. 有时为了抵消ReLU激励函数的线性增长部分，会在min()函数中嵌入max(0, x), 其在TensorFlow中的实现称作ReLU6，表示为min(max(0,x),6)
3. sigmoid函数是最常用的连续、平滑的激励函数。它也被称作逻辑函数(Logistic 函数),表示为1/(1+exp(-x))。范围-1到1
4.另外一种激励函数是双曲正切函数(hyper tangent, tanh)。范围0到1。双曲正弦与双曲余弦的比值，表达式(exp(x)-exp(-x))/((exp(x)+exp(-x))
5. softsign 函数也是一种激励函数，表达式为: x/(abs(x)+1)。softsign函数是符号函数的连续估计
6. softplus 激励函数是ReLU激励函数的平滑版，表达式为: log(exp(x)+1)
7. ELU 激励函数（Exponential Linear Unit， ELU）与softplus激励函数相似，不同点在于：当输入无限小时，ELU激励函数趋近于-1，而softplus函数趋近于0。表达式为(exp(x)+1) if x < 0 else x

# [1.8 读取数据源]
1. 鸢尾花卉数据集(Iris data)。
2. 出生体重数据(Birth weight data)。
3. 波士顿房价数据(Boston Housing data)。
4. MNIST手写体字库
5. 垃圾短信文本数据集(Spam-ham text data)。
6. 影评样本数据集。
7. CIFAR-10图像数据集。 http://www.cs.toronto.edu/~kriz/cifar.html。
8. 莎士比亚著作文本数据集(Shakespeare text data)。
9. 英德句子翻译样本集。

## 学习资料
* https://github.com/nfmcclure/tensorflow_cookbook
* https://www.tensorflow.org/api_docs/python
* https://www.tensorflow.org/tutorials/index.html
* https://github.com/tensorflow/tensorflow
* https://hub.docker.com/r/tensorflow/tensorflow
* http://stackoverflow.com/questions/tagged/TensorFlow
* https://www.udacity.com/course/deep-learning--ud730
* http://playgroud.tensorflow.org
* https://www.coursera.org/learn/neural-networks
* http://cs231n.stanford.edu/