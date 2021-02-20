import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DBI import compute_DB_index
dataset = pd.read_csv('./watermelon_4.csv', delimiter=",")
data = dataset.values

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
            print('cluster', cluster)
            return U, C, maxIter-curIter, cluster
    cluster = [[D[i] for i, j in enumerate(C) if (j == k)] for k in range(K)]
    print('cluster', cluster)
    return U, C, maxIter-curIter, cluster

U, C, iter, cluster = Kmeans(data,3,10)
print('C', C) # [2, 0, 0, 0, 1]
print('U', U)
print(compute_DB_index(data, cluster, 3))

f1 = plt.figure(1)
plt.title('watermelon_4')
plt.xlabel('density')
plt.ylabel('ratio')
plt.scatter(data[:, 0], data[:, 1], marker='o', color='g', s=50)
plt.scatter(U[:, 0], U[:, 1], marker='o', color='r', s=100)
# plt.xlim(0,1)
# plt.ylim(0,1)
m, n = np.shape(data)
for i in range(m):
    plt.plot([data[i, 0], U[int(C[i]), 0]], [data[i, 1], U[int(C[i]), 1]], "c--", linewidth=0.3)
plt.show()

# # %%