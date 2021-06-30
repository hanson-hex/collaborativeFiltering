import numpy as np
import random
import math
from matplotlib import pyplot as plt
from sklearn.metrics import davies_bouldin_score as dbs

'''优化函数'''


# y = x^2, 用户可以自己定义其他函数
def sphere(X):
    output = sum(np.square(X)/25)
    return output

def a(D, X):
    m, n = np.shape(D)
    K = 0
    for i in range(len(X)):
        K += math.pow(2, i) * X[len(X) - 1 - i]
    K = int(K)
    initSet = set()
    curK = K
    while(curK>0):  # 随机选取k个样本
        randomInt = random.randint(0, m-1)
        if randomInt not in initSet:
            curK -= 1
            initSet.add(randomInt)
    
    U = D[list(initSet), :]  # 均值向量,即质心
    C = np.zeros(m)
    # 计算样本到各均值向量的距离
    for i in range(m):
        p = 0
        minDistance = distance(D[i], U[0])
        for j in range(1, K):
            if distance(D[i], U[j]) < minDistance:
                p = j
                minDistance = distance(D[i], U[j])
        C[i] = p
    dbs(D, C)


print(a([1, 1, 1, 1, 1, 1, 1, 1]))

''' 种群初始化函数 '''

def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.randint(lb[j], ub[j])
    return X, lb, ub


'''边界检查函数'''
def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''计算适应度函数'''
def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
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
    y=x+(b/(x*Ngen))
   
def distance(x1, x2):  # 计算距离
    return np.sqrt(np.sum(np.square(np.array(x1)-np.array(x2))))
   
def initialBOA(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            a = random.randint(lb[j], ub[j])
            while(1):
                a = random.randint(lb[j], ub[j])
                if a not in X[i]:
                    break
            X[i, j] = a
    return X, lb, ub
  

def merge(X, A):
    pass
   
def BOAK(pop, dim, lb, ub, maxIter, fun, A):
    p=0.8 #probabibility switch
    power_exponent=0.1  # a = 0.1
    sensory_modality=0.01 # c = 0.01
    
    X, lb, ub = initialBOA(pop, dim, ub, lb)  # 初始化种群
    # X,C = merge(X, A)
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = fitness[0]
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = X[0, :]
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
            if(fun(X_new[i,:])<fitness[i]):
                X[i,:] = X_new[i,:]
            
        X = X_new    
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if fitness[0] <= GbestScore:  # 更新全局最优
            GbestScore = fitness[0]
            GbestPositon[0,:] = X[0, :]
        Curve[t] = GbestScore
        #更新sensory_modality
        sensory_modality = sensory_modality_NEW(sensory_modality, t+1)

    return GbestScore, GbestPositon, Curve

def getInitSet(D, K, m):
    pass

def Kmeans(D,K,maxIter):
    m, n = np.shape(D)
    if K >= m:
        return D
    initSet = getInitSet(D, K, m) 
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



def getMutate(pop, dim, X, ub, lb):
    mutant = np.zeros([pop, dim])
    F = 0.2 # 变异因子
    for i in range(pop):
        r0, r1, r2 = 0, 0, 0
        while r0 == r1 or r1 == r2 or r0 == r2 or r0 == i:
            r0 = random.randint(0, pop-1)
            r1 = random.randint(0, pop-1)
            r2 = random.randint(0, pop-1)
        mutant[i,:]= X[r0,:] + (X[r1,:] - X[r2,:]) * F
        for t in range(dim):
            if mutant[i, t] >= ub[t] or mutant[i, t] <= lb[t]:
                mutant[i, t] = random.uniform(lb[t], ub[t])
    return mutant


def csAndSelect(pop, dim, X, mutate, fun, fitness):
   CR = 0.1
   X_new = X
   for i in range(pop):
        Jrand = random.randint(0, dim)
        for j in range(dim):
            if random.random() > CR and j != Jrand:
                mutate[i, j] = X[i, j]
            tmp = fun(mutate[i,:])
            if tmp < fitness[i]:
                X_new[i,:] = mutate[i,:]
   return X_new
   
    

'''主函数 '''
# 设置参数
pop = 50  # 种群数量
MaxIter = 500 # 最大迭代次数
dim = 8 # 维度
lb = 0 * np.ones([dim, 1])  # 下边界
ub = 1 * np.ones([dim, 1])  # 上边界

# X, ub, lb = initialBOA(pop, dim, ub, lb)
# print('X', X)

def averFitness(BOA, function, number):
    s = 0
    for i in range(number):
        GbestScore, GbestPositon, Curve = BOA(pop, dim, lb, ub, MaxIter, function)
        s += GbestScore
    return s / number


# GbestScore, GbestPositon, Curve = BOA(pop, dim, lb, ub, MaxIter, sphere)


print(random.randint(0, 1))
print(random.randint(0, 1))
print(random.randint(0, 1))
print(random.randint(0, 1))