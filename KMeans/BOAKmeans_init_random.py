import numpy as np
import random
import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.metrics import davies_bouldin_score as dbs
from sklearn.datasets import load_iris, load_wine

'''优化函数'''


# y = x^2, 用户可以自己定义其他函数
def sphere(X):
    output = sum(np.square(X)/25)
    return output



def kFun(D, X, K):
    m, n = np.shape(D)
    result = 0
    X = [int(i) for i in X]
    U = D[X, :]  # 均值向量,即质心
    C = np.zeros(m)
    # 计算样本到各均值向量的距离
    for i in range(m):
        p = 0;
        minDistance = distance(D[i], U[0]);
        for j in range(1, K):
            if distance(D[i], U[j]) < minDistance:
                p = j
                minDistance = distance(D[i], U[j])
        result += minDistance
    return result


''' 种群初始化函数 '''

def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = int(random.randint(lb[j], ub[j] ))
    return X, lb, ub


'''边界检查函数'''
def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            X[i, j] = int(float(X[i, j]))
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X

def BorderCheckItem(X, ub, lb, dim):
    X = X.astype(int)
    for j in range(dim):
        if X[j] > ub[j]:
            X[j] = ub[j]
        elif X[j] < lb[j]:
            X[j] = lb[j]
    return X


'''计算适应度函数'''
def CaculateFitness(X, fun, D, k):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(D, X[i, :], k)
    return fitness

'''适应度排序'''
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index

'''根据适应度对位置进行排序'''
def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew

def sensory_modality_NEW(x,Ngen):
    b = 0.025
    # b = 1
    return x+(b/(x*Ngen))
   
def distance(x1, x2):  # 计算距离
    return np.sqrt(np.sum(np.square(np.array(x1)-np.array(x2))))
   
def initialBOA(pop, k, ub, lb):
    X = np.zeros([pop, k])
    for i in range(pop):
        curK = k
        while (curK > 0):
                randomInt = random.randint(lb[0], ub[0])
                if randomInt not in X[i]:
                    X[i, (k - curK)] = randomInt
                    curK -= 1
    return X, lb, ub
  
def BOAK(pop, k, MaxIter, D):
    lb = 0 * np.ones([k, 1])  # 下边界
    ub =  (len(D) - 1)* np.ones([k, 1])  # 上边界
    p=0.8 #probabibility switch
    power_exponent=0.1  # a = 0.1
    sensory_modality=0.01 # c = 0.01
    fun=kFun
    X, lb, ub = initialBOA(pop, k, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun, D, k)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序

    GbestScore = fitness[0]
    GbestPositon = np.zeros([1, k])
    GbestPositon[0,:] = X[0, :]
    # return GbestScore, GbestPositon
    X_new = X
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):
        for i in range(pop):
            FP = sensory_modality*(fitness**power_exponent)
            # 全局最优
            if random.random()>p:
                dis = random.random()*random.random()*GbestPositon - X[i,:]
                Temp = np.matrix(dis*FP[0,:])
                X_new[i,:] = X[i,:] + Temp[0,:]
            else:
                # Find random butterflies in the neighbourhood
                #epsilon = random.random()
                Temp = range(pop)
                JK = random.sample(Temp,pop)
                dis=random.random()*random.random()*X[JK[0],:]-X[JK[1],:]
                Temp = np.matrix(dis*FP[0,:])
                X_new[i,:] = X[i,:] + Temp[0,:]
            #如果更优才更新
            X_new[i, :] = BorderCheckItem(X_new[i, :], ub, lb, k)
            if(fun(D, X_new[i,:], k)<fitness[i]):
                X[i,:] = X_new[i,:]
            
        X = X_new    
        X = BorderCheck(X, ub, lb, pop, k)  # 边界检测
        fitness = CaculateFitness(X, fun, D, k)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if fitness[0] <= GbestScore:  # 更新全局最优
            GbestScore = fitness[0]
            GbestPositon[0,:] = X[0, :]
        Curve[t] = GbestScore
        #更新sensory_modality
        sensory_modality = sensory_modality_NEW(sensory_modality, t+1)
    return GbestScore, GbestPositon, Curve

def Kmeans(D,K,maxIter):
    m, n = np.shape(D)
    if K >= m:
        return D
    GbestScore, GbestPositon, Curve = BOAK(pop, 28, MaxIter, D)
    GbestPositon  = GbestPositon.astype(int)
    initSet = GbestPositon[0]
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


'''主函数 '''
# 设置参数
pop = 50  # 种群数量
MaxIter = 5 # 最大迭代次数
dim = 28 # 维度

# X, ub, lb = initialBOA(pop, dim, ub, lb)
# print('X', X)

def averFitness(func, X, K, number, maxIter):
    s = []
    for i in range(number):
        U, C, iter, cluster = func(X, K, maxIter)
        s.append(dbs(X, C))
    return max(s), min(s), sum(s) / number


iris = load_iris()
X = iris.data

wine = load_wine()
Y = wine.data

dataset = pd.read_csv('./Absenteeism_at_work.csv', delimiter=";")
Z = dataset.values


# GbestScore, GbestPositon, Curve = BOAK(pop, 3, MaxIter, X)
# print('GBestScore', GbestScore)
# print('CbestPositon', GbestPositon)
# print('Curve', Curve)

# GbestScore, GbestPositon, Curve = BOAK(pop, 3, MaxIter, Y)
# print('GBestScore', GbestScore)
# print('CbestPositon', GbestPositon)
# print('Curve', Curve)

GbestScore, GbestPositon, Curve = BOAK(pop, 28, MaxIter, Z)


print('GBestScore', GbestScore)
print('CbestPositon', GbestPositon)
print('Curve', Curve)