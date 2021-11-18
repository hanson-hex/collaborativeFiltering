import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import davies_bouldin_score as dbs
# from sklearn.metrics import silhouette_score as dbs
# from sklearn.metrics import calinski_harabaz_score as dbs
# from sklearn.metrics import adjusted_rand_score as dbs
from DBI import compute_DB_index
import math 

# dataset = pd.read_csv('./watermelon_4.csv', delimiter=",")
# data = dataset.values

# dataset = pd.read_csv('./Absenteeism_at_work.csv', delimiter=";")
# Z = dataset.values

# iris = load_iris()
# X = iris.data
# print('X', X)

# wine = load_wine()
# X = wine.data
# print("Y", Y)

# dataset = pd.read_csv('./Absenteeism_at_work.csv', delimiter=";")
# X = dataset.values

class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns # array of column names to encode


    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = preprocessing.LabelEncoder().fit(X[col])
        return self


    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output


    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)


    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output

dataset = pd.read_csv('./Frogs_MFCCs.csv', delimiter=",")
X = dataset.values
encoder = preprocessing.LabelEncoder()
X[:, 22] = encoder.fit_transform(X[:, 22])
X[:, 23] = encoder.fit_transform(X[:, 23])
X[:, 24] = encoder.fit_transform(X[:, 24])
K = 4
print('X', X)

# ct = ColumnTransformer([("Name_Of_Your_Step", preprocessing.LabelEncoder(),[21, 22, 23])], remainder="passthrough")
# y = ct.fit_transform(X)

# print('Y', y)

# %%

def kmeans(data,k, maxIter):
    def _distance(p1,p2):
        """
        Return Eclud distance between two points.
        p1 = np.array([0,0]), p2 = np.array([1,1]) => 1.414
        """
        tmp = np.sum((p1-p2)**2)
        return np.sqrt(tmp)
    def _rand_center(data,k):
        """Generate k center within the range of data set."""
        n = data.shape[1] # features
        centroids = np.zeros((k,n)) # init with (0,0)....
        for i in range(n):
            dmin, dmax = np.min(data[:,i]), np.max(data[:,i])
            centroids[:,i] = dmin + (dmax - dmin) * np.random.rand(k)
        return centroids
    
    def _converged(centroids1, centroids2):
        
        # if centroids not changed, we say 'converged'
         set1 = set([tuple(c) for c in centroids1])
         set2 = set([tuple(c) for c in centroids2])
         return (set1 == set2)
        
    
    n = data.shape[0] # number of entries
    centroids = _rand_center(data,k)
    label = np.zeros(n,dtype=np.int) # track the nearest centroid
    assement = np.zeros(n) # for the assement of our model
    converged = False
    curIter = maxIter  # 最大的迭代次数
    while not converged:
        curIter -= 1
        old_centroids = np.copy(centroids)
        for i in range(n):
            # determine the nearest centroid and track it with label
            min_dist, min_index = np.inf, -1
            for j in range(k):
                dist = _distance(data[i],centroids[j])
                if dist < min_dist:
                    min_dist, min_index = dist, j
                    label[i] = j
            assement[i] = _distance(data[i],centroids[label[i]])**2
        
        # update centroid
        for m in range(k):
            centroids[m] = np.mean(data[label==m],axis=0)
        converged = _converged(old_centroids,centroids)    
    return centroids, label, np.sum(assement), maxIter - curIter, '1'


import random
def distance(x1, x2):  # 计算距离
    return np.sqrt(np.sum(np.square(np.array(x1)-np.array(x2))))

def Kmeans(D,K,maxIter):
    m, n = np.shape(D)
    if K >= m:
        return D
    initSet = set()
    curK = K
    while(curK>0):  # 随机选取k个样本
        randomInt = random.randint(0, m-1)
        if randomInt not in initSet:
            curK -= 1
            initSet.add(randomInt)
    
    U = D[list(initSet), :]  # 均值向量,即质心
    C = np.zeros(m)
    curIter = maxIter  # 最大的迭代次数
    dbsList = [float('inf')]
    while curIter > 0:
        curIter -= 1
        # 计算样本到各均值向量的距离
        for i in range(m):
            p = 0
            minDistance = distance(D[i], U[0])
            for j in range(1, K):
                if distance(D[i], U[j]) < minDistance:
                    p = j
                    minDistance = distance(D[i], U[j])
            C[i] = p
        newU = np.zeros((K, n))
        cnt = np.zeros(K)

        for i in range(m):
            newU[int(C[i])] = newU[int(C[i])] + D[i]
            cnt[int(C[i])] += 1
        dbsList.append(dbs(D, C))
        changed = 0
        # 判断质心是否发生变化，如果发生变化则继续迭代，否则结束
        for i in range(K):
            newU[i] /= cnt[i]
            for j in range(n):
                if U[i, j] != newU[i, j]:
                    changed = 1
                    U[i, j] = newU[i, j]
        if changed == 0:
            cluster = [[D[i] for i, j in enumerate(C) if (j == k)] for k in range(K)]
            # indexCluster = [[i + 1 for i, j in enumerate(C) if (j == k)] for k in range(K)]
            lastList = [dbsList[len(dbsList) - 1] for i in range(curIter)]
            dbsList = dbsList + lastList
            return U, C, maxIter-curIter, cluster, dbsList
    cluster = [[D[i]  for i, j in enumerate(C) if (j == k)] for k in range(K)]
    # indexCluster = [[i + 1 for i, j in enumerate(C) if (j == k)] for k in range(K)]

    return U, C, maxIter-curIter, cluster, dbsList


def averFitness(func, X, K, number, maxIter):
    s = []
    for i in range(number):
        U, C, iter, cluster, dbsLists = func(X, K, maxIter)
        s.append(dbs(X, C))
    return max(s), min(s), sum(s) / number

centroids, label, assement, iter = kmeans(X, K, 100)
print('kmeans1')
print(dbs(X, label))
print('iter', iter)

km = KMeans(n_clusters=K)
km.fit(X)
centers = km.cluster_centers_
print('kmeans2')
print(dbs(X, km.labels_))

U, C, iter, cluster, dbsList = Kmeans(X, K, 100)
print('kmeans3')
print('dbsList', dbsList[len(dbsList) - 1])
print('iter', iter)



# 绘制适应度曲线
# plt.figure(1)
# print('dbsList', dbsList[len(dbsList) - 1])
# plt.plot(dbsList, 'r-', linewidth=2)
# plt.xlabel('Iteration', fontsize='medium')
# plt.ylabel("DBI", fontsize='medium')
# plt.legend(["Kmeans"])
# plt.grid()
# plt.title('K-means', fontsize='large')
# plt.show()

# max, min, aver = averFitness(Kmeans, X=X, K = K, number = 30, maxIter = 10)
# print('k-means最大值：', max)
# print('k-means最小值:', min)
# print('k-means平均值：', aver)

max, min, aver = averFitness(kmeans, X=X, K=K, number = 30, maxIter = 10)

print('k-means最大值：', max)
print('k-means最小值:', min)
print('k-means平均值：', aver)

# max, min, aver = averFitness(Kmeans, X=Z, K=28, number = 30, maxIter = 10)

# print('k-means最大值：', max)
# print('k-means最小值:', min)
# print('k-means平均值：', aver)



# print('iter', iter)
# print('C', C)
# print(compute_DB_index(cluster, U, 3))
# print(DaviesBouldin(X, C))


# U, C, iter, cluster = MyKmeans(Y, 13, 99)
# print('iter', iter)
# print('C', C)
# print(dbs(Y, C))
# print(compute_DB_index(cluster, U, 3))
# print(DaviesBouldin(Y, C))


# f1 = plt.figure(1)
# plt.title('watermelon_4')
# plt.xlabel('hua_e')
# plt.ylabel('hua_b')
# plt.scatter(X[:, 0], X[:, 1], marker='o', color='g', s=50)
# plt.scatter(U[:, 0], U[:, 1], marker='o', color='r', s=60)
# plt.scatter(U[:, 0], U[:, 2], marker='o', color='y', s=80)
# plt.xlim(0,10)
# plt.ylim(0,5)
# m, n = np.shape(X)
# # for i in range(m):
# #     plt.plot([X[i, 0] * X[i, 1], U[int(C[i]), 0] * U[int(C[i]), 1]], [X[i, 2] * X[i, 3], U[int(C[i]), 2] * U[int(C[i]), 3]], "c--", linewidth=0.3)
# plt.show()


# estimator = KMeans(n_clusters=3, max_iter=1, random_state=1) # 构造聚类器
# estimator.fit(X) # 聚类
# label_pred = estimator.labels_ # 获取聚类标签
# print(dbs(X, label_pred))

# #绘制k-means结果
# x0 = X[label_pred == 0]
# x1 = X[label_pred == 1]
# x2 = X[label_pred == 2]
# plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
# plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
# plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
# plt.xlabel('petal length')
# plt.ylabel('petal width')
# plt.legend(loc=2)
# plt.show()


# f1 = plt.figure(1)
# plt.title('watermelon_4')
# plt.xlabel('density')
# plt.ylabel('ratio')
# plt.scatter(X[:, 0], X[:, 1], marker='o', color='g', s=50)
# plt.scatter(U[:, 0], U[:, 1], marker='o', color='r', s=100)
# # plt.xlim(0,1)
# # plt.ylim(0,1)
# m, n = np.shape(X)
# for i in range(m):
#     plt.plot([X[i, 0], U[int(C[i]), 0]], [X[i, 1], U[int(C[i]), 1]], "c--", linewidth=0.3)
# plt.show()

# f1 = plt.figure(1)
# plt.title('watermelon_4')
# plt.xlabel('density')
# plt.ylabel('ratio')
# plt.scatter(data[:, 0], data[:, 1], marker='o', color='g', s=50)
# plt.scatter(U[:, 0], U[:, 1], marker='o', color='r', s=100)
# # plt.xlim(0,1)
# # plt.ylim(0,1)
# m, n = np.shape(data)
# for i in range(m):
#     plt.plot([data[i, 0], U[int(C[i]), 0]], [data[i, 1], U[int(C[i]), 1]], "c--", linewidth=0.3)
# plt.show()

# # %%