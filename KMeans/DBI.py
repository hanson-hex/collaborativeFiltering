import math
import numpy as np

# nc is number of clusters
# to be implemented without the use of any libraries (from the scratch)
 
def vectorDistance(v1, v2):
    """
    this function calculates de euclidean distance between two
    vectors.
    """
    return np.sqrt(np.sum(np.square(np.array(v1)-np.array(v2))))
 
def compute_Si(i, x, cluster,nc):
    norm_c = nc
    s = 0
    for t in x[i]:
        s += vectorDistance(t,cluster)
    return s/norm_c
 
def compute_Rij(i, j, x, clusters, nc):
    Mij = vectorDistance(clusters[i],clusters[j])
    Rij = (compute_Si(i,x,clusters[i],nc) + compute_Si(j,x,clusters[j],nc))/Mij
    return Rij
 
def compute_Di(i, x, clusters, nc):
    list_r = []
    for j in range(nc):
        if i != j:
            temp = compute_Rij(i, j, x, clusters, nc)
            list_r.append(temp)
    return max(list_r)
 
def compute_DB_index(x, clusters, nc):
    sigma_R = 0.0
    for i in range(nc):
        sigma_R = sigma_R + compute_Di(i, x, clusters, nc)
    DB_index = float(sigma_R)/float(nc)
    return DB_index

#  ```
#  1、计算Si

#     DBI计算公式中首先定义了Si变量，Si计算的是类内数据到簇质心的平均距离，代表了簇类i中各时间序列的分散程度，计算公式为：其中Xj代表簇类i中第j个数据点，也就是一个时间序列，Ai是簇类i的质心，Ti是簇类i中数据的个数，p在通常情况下取2，这样就可以计算独立的数据点和质心的欧式距离（euclidean metric），当然在考察流型和高维数据的时候，欧氏距离也许不是最佳的距离计算方式，但也是比较典型的了。

#     2、计算Mij

#     分子之和计算完后，需计算分母Mij，定义为簇类i与簇类j的距离，计算公式为：

# ak,i代表簇类i质心点的第k个值，Mij就是簇类i与簇类j质心的距离。

#     3、计算Rij
#     计算了分子与分母后，DBI定义了一个衡量相似度的值Rij，计算公式为：

#     4、计算DBI

#     有了以上公式的基础，我们做一个基于簇类数n的n^2的嵌套循环，对每一个簇类i计算最大值的Rij，记为Di，即，也即簇类i与其他类的最大相似度值，也就是取出最差结果。然后对所有类的最大相似度取均值就得到了DBI指数，计算公式为：
#  ```