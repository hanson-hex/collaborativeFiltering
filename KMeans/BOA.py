import numpy as np
import random
import math
from matplotlib import pyplot as plt

'''优化函数'''


# y = x^2, 用户可以自己定义其他函数
def sphere(X):
    output = sum(np.square(X)/25)
    return output

def rastringin(X):
    Y = X[0: len(X)//2]
    Z = X[len(X)//2: len(X)-1]
    A = 10
    d = len(X) - 1
    res = 10 * d + np.sum(X**2 - 10*np.cos(2*np.pi*X))
    return res
    output = sum(np.square(Y)) - sum(10*np.cos(2*np.pi*Z)) + 10
    return output

def Step(X):
    output = np.sum(int(X + 0.5)**2)
    return output

def Schwefel(X):
    d = len(X) - 1
    output = 418.9829*d - np.sum(X*np.sin(np.sqrt(np.abs(X))))
    return output

def beale(X):
    Y = X[0: len(X)//2]
    Z = X[len(X)//2: len(X)]
    output = np.power(1.5 - Y + Z*Y, 2) + np.power(2.25 - Y + Y*(Z**2), 2) + np.power(2.625 - Y + Y*(Z**3), 2)
    return output

def Griewank(X):
    d = len(X)
    i = np.arange(1, d+1)
    return 1 + np.sum(X ** 2) / 4000 - np.prod(np.cos(X / np.sqrt(i)))

def ackley(X):
    output = - 20 * np.exp(-0.2 * np.sqrt(np.mean(X**2))) - np.exp(np.mean(np.cos(2*np.pi*X)+np.cos(2 * np.pi * X))) + 20 + np.exp(1)
    return output

def booth(X):
    Y = X[0: len(X)//2]
    Z = X[len(X)//2:len(X)]
    output = np.power(sum(Y + 2*Z) - 7, 2) + np.power(sum(2*Y+Z)-5, 2)
    return output

def alpine(X):
    output = np.sum(np.abs(0.1*X + X*np.sin(X)))
    return output


''' 种群初始化函数 '''

def initial1(pop, dim, ub, lb, fun):
    X = np.zeros([pop, dim])
    XAll = np.zeros([2*pop,dim])
    for i in range(pop):
        for j in range(dim):
            XAll[i, j] = random.random()*(ub[j] - lb[j]) + lb[j]
            XAll[i+pop,j] = (ub[j]+lb[j]) - XAll[i, j] #求反向种群
            if XAll[i,j]>ub[j]:
                XAll[i, j] = ub[j]
            if XAll[i,j]<lb[j]:
                XAll[i, j] = lb[j]
            if XAll[i + pop,j]>ub[j]:
                XAll[i+pop, j] = ub[j]
            if XAll[i+pop,j]<lb[j]:
                XAll[i+pop, j] = lb[j]
        fitness = fun(XAll[i,:])
        fitnessBack = fun(XAll[i+pop,:])
        if(fitnessBack<fitness): #反向解更好的给原始解
            XAll[i,:] = XAll[i+pop,:]
    
    X = XAll[0:pop,:]
    #获取精英边界
    lbT = np.min(X,0)
    ubT = np.max(X,0)
    
    for i in range(X.shape[0]):
        X[i,:] = random.random()*(lbT + ubT) - X[i,:]
        for j in range(dim):
            if X[i,j]>ub[j]:
                X[i, j] = ub[j]
            if X[i,j]<lb[j]:
                X[i, j] = lb[j]  
    return X,lb,ub

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


'''计算适应度函数'''

def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        # fitness[i] = fun(X[i, :][0:dim//2], X[i, :][dim//2:dim])
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
    return x+(b/(x*Ngen))


'''蝴蝶优化算法'''
## 基础蝴蝶算法
def BOA(pop, dim, lb, ub, MaxIter, fun):
    p=0.8 #probabibility switch
    power_exponent=0.1  # a = 0.1
    sensory_modality=0.01 # c = 0.01
    
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
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


def levy_flight(Lambda, dim):
    sigma1 = np.power((math.gamma(1 + Lambda) * np.sin((np.pi * Lambda) / 2)) \
                      / math.gamma((1 + Lambda) / 2) * np.power(2, (Lambda - 1) / 2), 1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, size=dim)
    v = np.random.normal(0, sigma2, size=dim)
    step = u / np.power(np.fabs(v), 1 / Lambda)
    return step

## 基础蝴蝶算法 + levy
def BOAF(pop, dim, lb, ub, MaxIter, fun):
    p=0.8 #probabibility switch
    power_exponent=0.1  # a = 0.1
    sensory_modality=0.01 # c = 0.01
    
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
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

## 融入差分进化和精英算法
def BOA1(pop, dim, lb, ub, MaxIter, fun):
    p=0.8 #probabibility switch
    power_exponent=0.1  # a = 0.1
    sensory_modality=0.01 # c = 0.01
    
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
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

        mutant = getMutate(pop, dim, X, ub, lb)
        X = csAndSelect(pop, dim, X, mutant, fun, fitness)

        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if fitness[0] <= GbestScore:  # 更新全局最优
            GbestScore = fitness[0]
            GbestPositon[0,:] = X[0, :]
        
        V = GbestPositon[0, :] + 0.001*np.random.randn(1, dim)
        X[pop - 1,:] = V[0,:]

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

## 指数收敛因子+ 差分精华算法
def EDEIBOA(pop, dim, lb, ub, MaxIter, fun):
    p=0.8 #probabibility switch
    power_exponent=0.1  # a = 0.1
    sensory_modality=0.01 # c = 0.01
    
    # X, lb, ub = initial1(pop, dim, ub, lb, fun)  # 初始化种群
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = fitness[0]
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = X[0, :]
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

        # mutant = getMutate(pop, dim, X, ub, lb)
        mutant = getMutate1(pop, dim, X, ub, lb, i, MaxIter)
        X = csAndSelect(pop, dim, X, mutant, fun, fitness)

        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if fitness[0] <= GbestScore:  # 更新全局最优
            GbestScore = fitness[0]
            GbestPositon[0,:] = X[0, :]

        V = GbestPositon[0, :] + 0.001*np.random.randn(1, dim);
        X[pop - 1,:] = V[0,:];

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

## 指数收敛因子+ 差分精华算法
def EDEIBOA2(pop, dim, lb, ub, MaxIter, fun):
    p=0.8 #probabibility switch
    power_exponent=0.1  # a = 0.1
    sensory_modality=0.01 # c = 0.01
    
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = fitness[0]
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = X[0, :]
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

        mutant = getMutate(pop, dim, X, ub, lb)
        X = csAndSelect(pop, dim, X, mutant, fun, fitness)

        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if fitness[0] <= GbestScore:  # 更新全局最优
            GbestScore = fitness[0]
            GbestPositon[0,:] = X[0, :]

        # V = GbestPositon[0, :] + 0.001*np.random.randn(1, dim);
        # X[pop - 1,:] = V[0,:];

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

## 指数收敛因子+ 差分精华算法
def EDEIBOA3(pop, dim, lb, ub, MaxIter, fun):
    p=0.8 #probabibility switch
    power_exponent=0.1  # a = 0.1
    sensory_modality=0.01 # c = 0.01
    
    # X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = fitness[0]
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = X[0, :]
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

        mutant = getMutate1(pop, dim, X, ub, lb, i, MaxIter)
        X = csAndSelect(pop, dim, X, mutant, fun, fitness)

        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if fitness[0] <= GbestScore:  # 更新全局最优
            GbestScore = fitness[0]
            GbestPositon[0,:] = X[0, :]

        # V = GbestPositon[0, :] + 0.001*np.random.randn(1, dim);
        # X[pop - 1,:] = V[0,:];

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


def BOA3(pop, dim, lb, ub, MaxIter, fun):
    p=0.8 #probabibility switch
    power_exponent=0.1  # a = 0.1
    sensory_modality=0.01 # c = 0.01
    
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = fitness[0]
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = X[0, :]
    X_new = X
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):
        a = math.exp(-t)
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
                V = GbestPositon[0, :] + 0.001*np.random.randn(1, dim)
                X[pop - 1,:] = V[0,:]
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

## 指数收敛因子
def EBOA(pop, dim, lb, ub, MaxIter, fun):
    p=0.8 #probabibility switch
    power_exponent=0.1  # a = 0.1
    sensory_modality=0.01 # c = 0.01
    
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = fitness[0]
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = X[0, :]
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
                V = GbestPositon[0, :] + 0.001*np.random.randn(1, dim)
                X[pop - 1,:] = V[0,:]
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
  
## 线性收敛因子
def LBOA(pop, dim, lb, ub, MaxIter, fun):
    p=0.8 #probabibility switch
    power_exponent=0.1  # a = 0.1
    sensory_modality=0.01 # c = 0.01
    
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = fitness[0]
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = X[0, :]
    X_new = X
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):
        a = 2 - t / MaxIter
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


def BOA5(pop, dim, lb, ub, MaxIter, fun):
    p=0.8 #probabibility switch
    power_exponent=0.1  # a = 0.1
    sensory_modality=0.01 # c = 0.01
    
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
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


def getMutate1(pop, dim, X, ub, lb, i, maxIter):
    mutant = np.zeros([pop, dim])
    a = 1 - maxIter/(maxIter + 1 - i)
    F = 0.2*(math.e)**a # 变异因子
    # F = 0.2 # 变异因子
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
dim = 30 # 维度
lb = -100 * np.ones([dim, 1])  # 下边界
ub = 100 * np.ones([dim, 1])  # 上边界
num = 30

def averFitness(BOA, function, number):
    s = 0
    for i in range(number):
        GbestScore, GbestPositon, Curve = BOA(pop, dim, lb, ub, MaxIter, function)
        s += GbestScore
    return s / number


# GbestScore, GbestPositon, Curve = BOAF(pop, dim, lb, ub, MaxIter, sphere)


# print('最优适应度值：', GbestScore)
# print('最优解：', GbestPositon)


# print('普通sphere平均最优适应度值：', averFitness(BOA, sphere, num))
# print('普通alpine平均最优适应度值：', averFitness(BOA, alpine, num))
# print('普通rastringin平均最优适应度值：', averFitness(BOA, rastringin, num))
# print('普通Schwefel平均最优适应度值：', averFitness(BOA, Schwefel, num))
# # print('普通step平均最优适应度值：', averFitness(BOA, Step, num))
# print('普通Griewank平均最优适应度值：', averFitness(BOA, Griewank, num))
# print('普通ackley平均最优适应度值：', averFitness(BOA, ackley, num))
# print('普通booth平均最优适应度值：', averFitness(BOA, booth, num))
# # print('普通beale平均最优适应度值：', averFitness(BOA, beale, num))

# print('普通sphere平均最优适应度值：', averFitness(BOA, sphere, num))
# print('普通alpine平均最优适应度值：', averFitness(BOA, alpine, num))
# print('普通rastringin平均最优适应度值：', averFitness(BOA, rastringin, num))
# print('普通Schwefel平均最优适应度值：', averFitness(BOA, Schwefel, num))
# # print('普通step平均最优适应度值：', averFitness(BOA, Step, num))
# print('普通Griewank平均最优适应度值：', averFitness(BOA, Griewank, num))
# print('普通ackley平均最优适应度值：', averFitness(BOA, ackley, num))
# print('普通booth平均最优适应度值：', averFitness(BOA, booth, num))


# print('普通sphere BOA平均最优适应度值：', averFitness(BOA, sphere, 10))
# print('普通sphere LBOA平均最优适应度值：', averFitness(LBOA, sphere, 10))
# print('普通sphere EBOA平均最优适应度值：', averFitness(EBOA, sphere, 10))
print('普通sphere BOAF平均最优适应度值：', averFitness(BOAF, sphere, 1))

# print('普通beale平均最优适应度值：', averFitness(BOA, beale, num))
# GbestScore0, GbestPositon0, Curve0 = EDEIBOA(pop, dim, lb, ub, MaxIter, sphere)
# print('0最优适应度值：', GbestScore0)
# print('0最优解：', GbestPositon0)

# GbestScore1, GbestPositon1, Curve1 = EDEIBOA2(pop, dim, lb, ub, MaxIter, sphere)
# print('1最优适应度值：', GbestScore1)
# print('1最优解：', GbestPositon1)

# print('普通sphere平均最优适应度值：', averFitness(BOAF, sphere, 10))
# print('普通alpine平均最优适应度值：', averFitness(BOAF, alpine, 10))
# GbestScore2, GbestPositon2, Curve2 = EDEIBOA3(pop, dim, lb, ub, MaxIter, sphere)
# print('2最优适应度值：', GbestScore2)
# print('2最优解：', GbestPositon2)

# print('线性sphere最优适应度值：', averFitness(BOA4, sphere, 30))
# print('线性alpine平均最优适应度值：', averFitness(BOA4, alpine, 30))

# print('指数sphere最优适应度值：', averFitness(BOA35, sphere, 30))
# print('指数apline平均最优适应度值：', averFitness(BOA35, alpine, 30))

# print('指数sphere最优适应度值：', averFitness(BOA35, sphere, 30))
# print('指数apline平均最优适应度值：', averFitness(BOA35, alpine, 30))

# print('指数sphere最优适应度值：', averFitness(BOA35, sphere, 30))
# print('指数apline平均最优适应度值：', averFitness(BOA35, alpine, 30))

# print('指数sphere最优适应度值：', averFitness(EDEIBOA, sphere, 30))
# print('指数apline平均最优适应度值：', averFitness(EDEIBOA, alpine, 30))

# GbestScore1, GbestPositon1, Curve1 = BOA1(pop, dim, lb, ub, MaxIter, fun)
# print('1最优适应度值：', GbestScore1)
# print('1最优解：', GbestPositon1)

# GbestScore, GbestPositon, Curve = BOA(pop, dim, lb, ub, MaxIter, sphere)
# print('最优适应度值：', GbestScore)
# print('最优解：', GbestPositon)

# GbestScore2, GbestPositon2, Curve2 = EDEIBOA(pop, dim, lb, ub, MaxIter, sphere)
# print('2最优适应度值：', GbestScore2)
# print('2最优解：', GbestPositon2)

# GbestScore3, GbestPositon3, Curve3 = BOA3(pop, dim, lb, ub, MaxIter, sphere)
# print('3最优适应度值：', GbestScore3)
# print('3最优解：', GbestPositon3)

# GbestScore35, GbestPositon35, Curve35 = BOA35(pop, dim, lb, ub, MaxIter, sphere)
# print('35最优适应度值：', GbestScore35)
# print('35最优解：', GbestPositon35)


# GbestScore4, GbestPositon4, Curve4 = BOA3(pop, dim, lb, ub, MaxIter, sphere)
# print('4最优适应度值：', GbestScore4)
# print('4最优解：', GbestPositon4)

# GbestScore5, GbestPositon5, Curve5 = BOA5(pop, dim, lb, ub, MaxIter, fun)

# print('最优适应度值：', GbestScore5)
# print('最优解：', GbestPositon5)

# 绘制适应度曲线
# plt.figure(1)
# plt.plot(Curve1, 'r-', linewidth=2)
# # plt.plot(Curve1, 'g-', linewidth=2)
# plt.plot(Curve2, 'g-', linewidth=2)
# # plt.plot(Curve3, 'y-', linewidth=2)
# # plt.plot(Curve35, 'c-', linewidth=2)
# # plt.plot(Curve4, 'b-', linewidth=2)
# plt.xlabel('Iteration', fontsize='medium')
# plt.ylabel("Fitness", fontsize='medium')
# plt.legend(["BOA", "IBOA"])
# plt.grid()
# plt.title('BOA', fontsize='large')
# plt.show()

# # 绘制搜索空间
# fig = plt.figure(2)
# ax = Axes3D(fig)
# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# X, Y = np.meshgrid(X, Y)
# Z = X ** 2 + Y ** 2
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
