import numpy as np
import random
import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.metrics import cluster, davies_bouldin_score as dbs
from sklearn.datasets import load_iris, load_wine
from sklearn import preprocessing


'''优化函数'''


# y = x^2, 用户可以自己定义其他函数
def sphere(X):
    output = sum(np.square(X)/25)
    return output

def kFun(D, X, K):
    m, dim = np.shape(D)
    result = 0
    U = np.zeros([K, dim])
    for i in range(K):
        U[i] = X[i * dim : (i +1)* dim]
    C = np.zeros(m)
    # 计算样本到各均值向量的距离
    for i in range(m):
        p = 0;
        minDistance = distance(D[i], U[0]);
        for j in range(1, K):
            if distance(D[i], U[j]) < minDistance:
                p = j
                minDistance = distance(D[i], U[j])
        C[i] = p
    #     result += minDistance
    # return result
    if len(set(C)) == 1:
        return float('inf')
    return dbs(D, C)


''' 种群初始化函数 '''

def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
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

'''边界检查函数'''
def BorderCheckItem(X, ub, lb, dim):
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
   
  
def BOAK(pop, k, D):
    m, dim = np.shape(D)
    lb = np.zeros(dim * k)  # 下边界
    ub =  np.zeros(dim * k)  # 上边界
    for i in range(dim * k):
        lb[i] = min([row[i % dim] for row in D])
        ub[i] = max([row[i % dim] for row in D])
    MaxIter = 5
    p=0.8 #probabibility switch
    power_exponent=0.1  # a = 0.1
    sensory_modality=0.01 # c = 0.01
    fun=kFun
    X, lb, ub = initial(pop, dim * k, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun, D, k)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序

    GbestScore = fitness[0]
    GbestPositon = np.zeros([1, dim * k])
    GbestPositon[0,:] = X[0, :]
    X_new = X
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):
        for i in range(pop):
            FP = sensory_modality*(fitness**power_exponent)
            # 全局最优
            if random.random()>p:
                # dis = random.random()*random.random()*GbestPositon - X[i,:]
                Temp = range(pop)
                JK = random.sample(Temp,pop)
                dis = random.random()*(GbestPositon - X[i,:]) + random.random()*(X[JK[0],:] - X[JK[1],:])
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
                X_new[i,:] = BorderCheckItem(X_new[i,:], ub, lb, dim*k)
            #如果更优才更新
            if(fun(D, X_new[i,:], k)<fitness[i]):
                X[i,:] = X_new[i,:]
            
        X = X_new    
        X = BorderCheck(X, ub, lb, pop, dim * k)  # 边界检测
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

def BOAKA(pop, k, D):
    lb = 0 * np.ones([k, 1])  # 下边界
    ub =  (len(D) - 1)* np.ones([k, 1])  # 上边界
    MaxIter = 5
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
        a = 2*math.exp(-t/MaxIter)
        for i in range(pop):
            FP = sensory_modality*(fitness**power_exponent)
            # 全局最优
            if random.random()>p:
                dis = random.random()*random.random()*GbestPositon - X[i,:]
                Temp = np.matrix(dis*FP[0,:])
                X_new[i,:] = a*X[i,:] + Temp[0,:]
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

def kcluster(rows,k,maxIter):  
  m, n = np.shape(rows)
#   # 确定每个点的最大值和最小值，给随机数定个范围  
#   ranges=[(min([row[i] for row in rows]),max([row[i] for row in rows]))   
#   for i in range(len(rows[0]))]  

  m, dim = np.shape(rows)
  GbestScore, GbestPositon, Curve = BOAK(pop, k, rows)
  U = GbestPositon[0]
  # 随机建立k个中心点  
#   clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0]   
#   for i in range(len(rows[0]))] for j in range(k)]  
  clusters = np.zeros([k, dim])
  for i in range(k):
      clusters[i] = U[i *  dim: (i +1)* dim]

  print('clusters', clusters)

  lastmatches=None
  # 设定循环100次，看你的数据大小，次数自定义  

  dbsList = [float('inf')]
  C = np.zeros(m)
  for t in range(100):  
    bestmatches=[[] for i in range(k)]  

    # 在每一行中寻找距离最近的中心点  
    for j in range(len(rows)):
      row=rows[j]
      bestmatch=0
      for i in range(k):
        d=distance(clusters[i],row)
        if d<distance(clusters[bestmatch],row): bestmatch=i  
      C[j] = bestmatch
      bestmatches[bestmatch].append(j)

    # 如果结果与上一次的相同，则整个过程结束  
    if bestmatches==lastmatches: break  
    lastmatches=bestmatches
    dbsList.append(dbs(rows, C))
    # 将中心点移到其所有成员的平均位置处  
    for i in range(k):  
      avgs=[0.0]*len(rows[0]) 
      if len(bestmatches[i])>0:
        for rowid in bestmatches[i]:  
          for m in range(len(rows[rowid])):  
            avgs[m]+=rows[rowid][m]
        for j in range(len(avgs)):
          avgs[j]/=len(bestmatches[i])
        clusters[i]=avgs
  dbsList = dbsList + [dbsList[len(dbsList) - 1] for i in range(100 - len(dbsList))]
  print('dbsList', dbsList)
  return bestmatches, C, dbsList


def kmeans(data, K, maxIter):
    m, dim = np.shape(data)
    k = K
    GbestScore, GbestPositon, Curve = BOAK(pop, k, data)
    # GbestPositon = [[4.3, 2, 3.10221011, 0.1,4.3, 2,1,0.1,4.3,2,1,0.1]]
    U = GbestPositon[0]
    def _distance(p1,p2):
        """
        Return Eclud distance between two points.
        p1 = np.array([0,0]), p2 = np.array([1,1]) => 1.414
        """
        return np.sqrt(np.sum(np.square(np.array(p1)-np.array(p2))))

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
        
    dbsList = [float('inf')]
    n = data.shape[0] # number of entries
    centroids = np.zeros([k, dim])
    for i in range(k):
        centroids[i] = U[i *  dim: (i +1)* dim]
    label = np.zeros(n,dtype=np.int) # track the nearest centroid
    assement = np.zeros(n) # for the assement of our model
    converged = False
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
    dbsList.append(dbs(data, label))
    new_centroids = []
    for m in range(k):
        if len(data[label==m]) == 0:
            k -= 1
        else:
            centroids[m] = np.mean(data[label==m],axis=0)
            new_centroids.append(centroids[m])
    centroids = new_centroids
    converged = _converged(old_centroids,centroids)  
    # while not converged:
    #     old_centroids = np.copy(centroids)
    #     for i in range(n):
    #         # determine the nearest centroid and track it with label
    #         min_dist, min_index = np.inf, -1
    #         for j in range(k):
    #             dist = _distance(data[i],centroids[j])
    #             if dist < min_dist:
    #                 min_dist, min_index = dist, j
    #                 label[i] = j
    #         assement[i] = _distance(data[i],centroids[label[i]])**2
        
    #     # update centroid
    #     dbsList.append(dbs(data, label))
    #     new_centroids = []
    #     for m in range(k):
    #         if len(data[label==m]) == 0:
    #             k -= 1
    #         else:
    #          centroids[m] = np.mean(data[label==m],axis=0)
    #          new_centroids.append(centroids[m])
    #     centroids = new_centroids
    #     converged = _converged(old_centroids,centroids)    
    # dbsList = dbsList + [dbsList[len(dbsList) - 1] for i in range(100 - len(dbsList))]
    print('dbsList', dbsList)
    return centroids, label, dbsList

def Kmeans(data,k,maxIter):
    m, dim = np.shape(data)
    GbestScore, GbestPositon, Curve = BOAK(pop, k, data)
    print('GBestScore', GbestScore[0])
    U = GbestPositon[0]
    def _distance(p1,p2):
        """
        Return Eclud distance between two points.
        p1 = np.array([0,0]), p2 = np.array([1,1]) => 1.414
        """
        return np.sqrt(np.sum(np.square(np.array(p1)-np.array(p2))))
        # tmp = np.sum((p1-p2)**2)
        # return np.sqrt(tmp)
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
    centroids = np.zeros([k, dim])
    for i in range(k):
        centroids[i] = U[i *  dim: (i +1)* dim]
    label = np.zeros(n,dtype=np.int) # track the nearest centroid
    assement = np.zeros(n) # for the assement of our model

    converged = False
    curIter = maxIter  # 最大的迭代次数
    dbsList = [float('inf')]
    while curIter > 0:
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
            if len(data[label==m]) == 0:
                return centroids, label, np.sum(assement)
            centroids[m] = np.mean(data[label==m],axis=0)
        converged = _converged(old_centroids,centroids)    
        if converged:
            return centroids, label, np.sum(assement)
    return centroids, label, np.sum(assement)


'''主函数 '''
# 设置参数
pop = 5  # 种群数量

# X, ub, lb = initialBOA(pop, dim, ub, lb)
# print('X', X)

def averFitness(func, X, K, number, maxIter):
    s = []
    for i in range(number):
        U, C, iter = func(X, K, maxIter)
        s.append(dbs(X, C))
    return max(s), min(s), sum(s) / number


iris = load_iris()
X = iris.data

wine = load_wine()
Y = wine.data

dataset = pd.read_csv('./Absenteeism_at_work.csv', delimiter=";")
Z = dataset.values

dataset = pd.read_csv('./Frogs_MFCCs.csv', delimiter=",")
XX = dataset.values
encoder = preprocessing.LabelEncoder()
XX[:, 22] = encoder.fit_transform(XX[:, 22])
XX[:, 23] = encoder.fit_transform(XX[:, 23])
XX[:, 24] = encoder.fit_transform(XX[:, 24])


# centroids, label, dbsList = kcluster(X, 3, 50)
# print('centroids', centroids)
# print('dbsList', dbsList)

# GbestScore, GbestPositon, Curve = BOAK(pop, 3, Y)
# U = GbestPositon[0]
# print('GbestScore', GbestScore)
# print('GBestPositon', GbestPositon)

maxK, minK, aver = averFitness(kmeans, X=X, K=3, number = 30, maxIter = 10)
print('k-means最大值：', maxK)
print('k-means最小值:', minK)
print('k-means平均值：', aver)

# maxK, minK, aver = averFitness(kmeans, X=Z, K=28, number = 30, maxIter = 10)
# print('k-means最大值：', maxK)
# print('k-means最小值:', minK)
# print('k-means平均值：', aver)

# max, min, aver = averFitness(Kmeans, X=XX, K=4, number = 30, maxIter = 10)
# print('k-means最大值：', max)
# print('k-means最小值:', min)
# print('k-means平均值：', aver)

# max, min, aver = averFitness(Kmeans, X=Z, K=28, number = 30, maxIter = 10)
# print('k-means最大值：', max)
# print('k-means最小值:', min)
# print('k-means平均值：', aver)

# GbestScore, GbestPositon, Curve = BOAK(pop, 3, X)
# print('GBestScore', GbestScore)
# print('CbestPositon', GbestPositon)
# print('Curve', Curve)

# GbestScore, GbestPositon, Curve = BOAK(pop, 4, XX)
# print('GBestScore', GbestScore)
# print('CbestPositon', GbestPositon)
# print('Curve', Curve)

# GbestScore, GbestPositon, Curve = BOAK(pop, 28,5, Z)
# print('GBestScore', GbestScore)
# print('CbestPositon', GbestPositon)
# print('Curve', Curve)