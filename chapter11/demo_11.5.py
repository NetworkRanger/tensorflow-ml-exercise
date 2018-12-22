#!/usr/bin/python2.7
# -*- coding:utf-8 -*-

# Author: NetworkRanger
# Date: 2018/12/22 下午7:50

# 11.5 TensorFlow实现k-means算法

# 1. 开始导入必要的编程库。因为后续需将四维的结果数据转换为二维数据进行可视化，所以也要从sklearn库导入PCA工具
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# 2. 创建一个计算图会话，加载iris数据集
sess = tf.Session()
iris = datasets.load_iris()
num_pts = len(iris.data)
num_feats = len(iris.data[0])

# 3. 设置分类数，迭代次数，创建计算图所需的变量
k = 3
generations = 25
data_points = tf.Variable(iris.data)
cluster_labels = tf.Variable(tf.zeros([num_pts], dtype=tf.int64))

# 4. 声明每个分组所需的几何中心变量。我们通过随机选择iris数据集中的三个数据点来初始化k-means聚类算法的几何中心
rand_starts = np.array([iris.data[np.random.choice(len(iris.data))] for _ in range(k)])
centroids = tf.Variable(rand_starts)

# 5. 计算每个数据点到每个几何中心的距离。本例的计算方法是，将几何中心点的数据点分别放入矩阵中，然后计算两个矩阵的欧式距离
centroids_matrix = tf.reshape(tf.tile(centroids, [num_pts, 1]), [num_pts, k, num_feats])
point_matrix = tf.reshape(tf.tile(data_points, [1, k], [num_pts, k, num_feats]))
distances = tf.reduce_sum(tf.square(point_matrix - centroids_matrix), reduction_indices=2)

# 6. 分配时，是以到每个数据点最小距离为最接近的几何中心点
centroid_group = tf.argmin(distances, 1)

# 7. 计算每组分类的平均距离得到新的几何中心点
def data_group_avg(group_ids, data):
    # Sum each group
    sum_total = tf.unsorted_segment_sum(data, group_ids, 3)
    # Count each group
    num_total = tf.unsorted_segment_sum(tf.ones_like(data), group_ids, 3)
    # Calculate average
    avg_by_group = sum_total/num_total
    return (avg_by_group)

means = data_group_avg(centroid_group, data_points)
update = tf.group(centroids.assign(means), cluster_labels.assign(centroid_group))

# 8. 初始化模型变量
init = tf.global_variables_initializer()
sess.run(init)

# 9. 遍历迭代训练，相应地更新每组分类的几何中心点
for i in range(generations):
    print('Calculating gen {}, out of {}.'.format(i, generations))
    _, centroid_group_count = sess.run([update, centroid_group])
    group_count = []
    for ix in range(k):
        group_count.append(np.sum(centroid_group_count==ix))
    print('Group counts: {}'.format(group_count))

# 10. 输出结果如下

"""
"""

# 11. 为了验证聚类模型，我们使用距离模型预测。看下有多少数据点与实际iris数据集中的鸢尾花物种匹配
[centers, assignments] = sess.run([centroids, cluster_labels])
def most_common(my_list):
    return (max(set(my_list), key=my_list.count))
label0 = most_common(list(assignments[0:50]))
label1 = most_common(list(assignments[50:100]))
label2 = most_common(list(assignments[100:150]))
group0_count = np.sum(assignments[0:50]==label0)
group1_count = np.sum(assignments[50:100]==label1)
group2_count = np.sum(assignments[100:150]==label2)
accuracy = (group0_count + group1_count + group2_count)/150.
print('Accuracy: {:.2}'.format(accuracy))

# 12. 输出结果如下

"""
Accuracy: 0.89
"""

# 13. 为了可视化分组过程，以及是否分离出鸢尾花物种，我们将用PCA工具将四维结果数据转为二维结果数据，并绘制数据点和分组。PCA分解之后，我们创建预测，并在x-y轴网格绘制彩色图形
pca_model = PCA(n_components=2)
reduced_data = pca_model.fit_transform(iris.data)
# Transform centers
reduced_centers = pca_model.transform(centers)
# Step size of mesh for plotting
h = .02
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:,0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Get k-means classifications for the grid points
xx_pt = list(xx.ravel())
yy_pt = list(yy.ravel())
xy_pts = np.array([[x,y] for x,y in zip(xx_pt, yy_pt)])
mytree = cKDTree(reduced_centers)
dist, indexes = mytree.query(xy_pts)
indexes = indexes.reshape(xx.shape)

# 14. 下面是用matplotlib模块在同一幅图形中绘制所有结果的代码。绘图部分的代码来自sklearn官方文档的示例(http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html)
plt.clf()
plt.imshow(indexes, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired, aspect='auto', origin='lower')
# Plot each of the true iris data groups
symbols = ['o', '^', 'D']
label_name = ['Setosa', 'Versicolour', 'Viginice']
for i in range(3):
    temp_group = reduced_data[(i*50):(50)*(i+1)]
    plt.plot(temp_group[:,0], temp_group[:,1], symbols[i], markersize=10, label=label_name[i])
# Plot the centroids as a white X
plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], color='w', zorder=10)
plt.title('K-means clustering on Iris Datasets\nCentroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='lower right')
plt.show()



