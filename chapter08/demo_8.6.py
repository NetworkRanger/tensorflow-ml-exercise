#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/20 下午9:25

# 8.6 用TensorFlow实现DeepDream

# 1. 在开始实现DeepDream之前，我们需要下载GoogleNet，其为CIFAR-1000图片数据集上已训练好的CNN模型

"""
me@computer:~$ wget https://storage.googleapis.com/download.tensorflow.org/models/inceptions5h.zip
me@computer:~$ unzip inception5h.zip
"""

# 2. 导入必要的代码库，并创建一个计算图会话
import os
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf
from io import BytesIO
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

# 3. 声明解压的模型参数的位置，并且将这些参数加载进TensorFlow的计算图
# Model location
model_fn = 'tensorflow_inception_graph.pb'
# Load graph parameters
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# 4. 创建输入数据的占位符，设置imagenet_mean为117.0；然后导入计算图定义，并传入归一化的占位符
# Create placeholder for input
t_input = tf.placeholder(np.float32, name='input')
# Imagenet average bias to subtract off images
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

# 5. 导入卷积层进行可视化，并在后续处理DeepDream时使用
# Create a list of layers that we can refer to later
layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
# Count how many outputs for each layer
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

# 6. 现在可以找某一层进行可视化了。我们可以通过层的名字或者特征数字139来查看。对图片进行噪声处理
layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
img_noise = np.random.uniform(size=(224,224,3)) + 100.0

# 7. 声明函数来绘制图片数组
def showarray(a, fmt='jpeg'):
    # First make sure everything is between 0 and 255
    a = np.uint8(np.clip(a, 0, 1)*255)
    # Pick an in-memory format for image display
    f = BytesIO()
    # Create the in memory image
    PIL.Image.fromarray(a).save(f, fmt)
    # Show image
    plt.imshow(a)

# 8. 在计算图中创建层迭代函数来简化重复的代码，其以层的名字来迭代
def T(layer):
    # Helper for getting layer output tensor
    return graph.get_tensor_by_name('import/%s:0'%layer)

# 9. 下面封装一个创建占位符的函数，其可以指定参数返回占位符
# The following function returns a function wapper that will create the placeholder
# inputs of a specified dtype
def tffunc(*argtypes):
    '''
    Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    :param argtypes:
    :return:
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# 10. 创建调整图片大小的函数，其可以指定图片大小。该函数采用TensorFlow的内建图片线性差值函数tf.image.resize.bilinear()
# bilinear():
# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    # Change 'img' size by linear interpolation
    return tf.image.resize_bilinear(img, size)[0,:,:,:]

# 11. 现在需要一种方法更新源图片，让其更像选择的特征。我们通过指定图片的梯度如何计算来实现。我们定义函数计算图片上子区域（方格）的梯度计算，使得梯度计算更快。我们将在图片的x轴和y轴方向上随机移动或者滚动，这将平滑方格的影响
def calc_grad_tiled(img, t_grad, tile_size=512):
    '''
    Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over multiple iterations.
    :param img:
    :param t_grad:
    :param tile_size:
    :return:
    '''
    # Pick a subregion square size
    sz = tile_size
    # Get the image height and width
    h, w = img.shape[:2]
    # Get a random shift amount in the x and y direction
    sx, sy = np.random.randint(sz, size=2)
    # Randomly shift the image (roll image) in the x and y directions
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    # Initialize the while image gradient as zeros
    grad = np.zeros_like(img)
    # Now we loop through all the sub-tiles in the image
    for y in range(0, max(h-sz//2, sz), sz):
        for x in range(0, max(w-sz//2, sz), sz):
            # Select the sub image tile
            sub = img_shift[y:y+sz,x:x+sz]
            # Calculate the gradient for the tile
            g = sess.run(t_grad, {t_input: sub})
            # Apply the gradient of the tile to the whole image graident
            grad[y:y+sz, x:x+sz] = g

    # Return the gradient, undoing the roll operation
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

# 12. 声明DeepDream算法的对象是选择特征的平均值。损失函数是基于梯度的，其依赖于输入图片和选取特征之间的距离。分割图像为高频部分和低频部分，在低频部分上计算梯度。将高频部分的结果再分割为高频部分和低频部分，重复前端的过程。原始图片和低频图片称为octaves。对传入的每个对象，计算其梯度并应用到图片中
def render_deepDream(t_obj, img0=img_noise, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    # defining the optimization objective, the objective is the mean of the feature
    t_score = tf.reduce_mean(t_obj)
    # Our gradients will be defined as changing the t_input to get closer tothe values of t_score. Here, t_score is the mean of the feature we select.
    # t_input will be the image octave (starting with the last)
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    # Store the image
    img = img0
    # Initialize the image octave list
    octaves = []
    # Since we stored the image, we need to only calculate n-1 octaves
    for i in range(octave_n-1):
        # Extract the image shape
        hw = img.shape[:2]
        # Resize the image, scale by the octave_scale (resize by linear interpolation)
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        # Residual is hi. Where residual = image - (Resize lo to be hw-shape)
        hi = img-resize(lo, hw)
        # Save the lo image for re-iterating
        img = lo
        # Save teh extracted hi-image
        octaves.append(hi)

    # generate details octave by octave
    for octave in range(octave_n):
        if octave>0:
            # Start with the last octave
            hi = octaves[-octave]
            #
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            # Calculate gradient of the image.
            g = calc_grad_tiled(img, t_grad)
            # Ideally, we would just add the gradient, g, but
            # we want do a forward step size of it ('step'),
            # and divide it by the avg. norm of the gradient, so
            # we are adding a gradient of a certain size each step.
            # Also, to make sure we aren't dividing by zero, we add 1e-7.
            img += g*(step/np.abs(g).mean()+1e-7)
            print('.', end='')
        showarray(img/255.0)

# 13. 所有函数准备好之后，开始运行DeepDream算法
# Run Deep Dream
if __name__ == '__main__':
    # Create resize function that has a wrapper that creates specified placeholder types
    resize = tffunc(np.float32, np.int32)(resize)

    # open image
    img0 = PIL.Image.open('book_cover.jpg')
    img0 = np.float32(img0)
    # Show Original Image
    showarray(img0/255.0)
    # Create deep dream
    render_deepDream(T(layer)[:,:,:139], img0, iter_n=15)
    sess.close()

