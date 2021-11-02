import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import davies_bouldin_score as dbs
from DBI import compute_DB_index
import math 

dataset = pd.read_csv('./watermelon_4.csv', delimiter=",")
data = dataset.values

dataset = pd.read_csv('./Absenteeism_at_work.csv', delimiter=";")
Z = dataset.values

iris = load_iris()
X = iris.data
# print('X', X)

wine = load_wine()
Y = wine.data
# print("Y", Y)

# %%



import random
def distance(x1, x2):  # 计算距离
    return np.sqrt(np.sum(np.square(np.array(x1)-np.array(x2))))

def Kmeans(D,K,maxIter):
    m, n = np.shape(D)
    if K >= m:
        return D
    # initSet = set()
    # curK = K
    # while(curK>0):  # 随机选取k个样本
    #     randomInt = random.randint(0, m-1)
    #     if randomInt not in initSet:
    #         curK -= 1
    #         initSet.add(randomInt)
    
    # initSet = (124, 63, 12)
    # initSet = (88, 63, 8)
    # initSet =  (48, 578,24, 221, 395,  654,  489,  367,  250,  706,  302,  162, 364, 245, 333, 525, 185, 477,  438, 68, 120, 442, 65, 640, 460, 598, 128, 652)
    # initSet = (79, 147, 0)
    # initSet = (69, 84, 13)
    # initSet = (125, 328,100, 151, 239, 26, 43, 706,108, 553, 455,194, 409, 13, 497, 337, 158, 557, 119, 659, 598, 82, 192, 182, 391, 351, 464, 339)
    # initSet = (23, 61, 139)
    # initSet = (51, 4, 151)
    initSet = (483, 195, 432, 340, 171, 385, 116, 45, 491, 286, 519, 112, 26, 344, 557, 506, 593, 677, 451, 499, 248, 550, 296, 325, 115, 490, 367, 177)
    U = D[list(initSet), :]  # 均值向量,即质心
    C = np.zeros(m)
    curIter = maxIter  # 最大的迭代次数
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
            newU[int(C[i])] += D[i]
            cnt[int(C[i])] += 1

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
            return U, C, maxIter-curIter, cluster
    cluster = [[D[i]  for i, j in enumerate(C) if (j == k)] for k in range(K)]
    # indexCluster = [[i + 1 for i, j in enumerate(C) if (j == k)] for k in range(K)]

    return U, C, maxIter-curIter, cluster


def averFitness(func, X, K, number, maxIter):
    s = []
    for i in range(number):
        U, C, iter, cluster = func(X, K, maxIter)
        s.append(dbs(X, C))
    return max(s), min(s), sum(s) / number

# U, C, iter, cluster = MyKmeans(X, 4, 10)

# max, min, aver = averFitness(Kmeans, X=X, K=3, number = 30, maxIter = 10)
# print('k-means最大值：', max)
# print('k-means最小值:', min)
# print('k-means平均值：', aver)

# max, min, aver = averFitness(Kmeans, X=Y, K=3, number = 30, maxIter = 10)

# print('k-means最大值：', max)
# print('k-means最小值:', min)
# print('k-means平均值：', aver)

max, min, aver = averFitness(Kmeans, X=Z, K=28, number = 30, maxIter = 10)

print('k-means最大值：', max)
print('k-means最小值:', min)
print('k-means平均值：', aver)



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


# U, C, iter, cluster = Kmeans(data, 3, 1)
# print('iter', iter)
# print('C', C)
# print('data', data)
# print(dbs(data, C))
# print(compute_DB_index(cluster, U, 10))


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