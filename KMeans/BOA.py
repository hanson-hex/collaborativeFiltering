import numpy as np
import random
from matplotlib import pyplot as plt

'''优化函数'''


# y = x^2, 用户可以自己定义其他函数
def fun(X):
    output = sum(np.square(X))
    return output


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
    # b = 0.025
    y=x+(0.025/(x*Ngen))
    return y



'''蝴蝶优化算法'''
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
           

        print('GbestPositon', GbestPositon[0,:])
        V = GbestPositon[0,:] + 0.001*np.random.randn(0, dim)
        print('V', V)
        print('V[pop - 1,:]', X[pop - 1,:])
        X[pop - 1,:] = V    

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
    F = 0.8 # 变异因子
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
   CR = 0.5
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
MaxIter = 1000 # 最大迭代次数
dim = 3 # 维度
lb = -100 * np.ones([dim, 1])  # 下边界
ub = 100 * np.ones([dim, 1])  # 上边界

GbestScore, GbestPositon, Curve = BOA(pop, dim, lb, ub, MaxIter, fun)
GbestScore1, GbestPositon1, Curve1 = BOA1(pop, dim, lb, ub, MaxIter, fun)
print('最优适应度值：', GbestScore)
print('最优解：', GbestPositon)

print('1最优适应度值：', GbestScore1)
print('1最优解：', GbestPositon1)

# 绘制适应度曲线
plt.figure(1)
plt.plot(Curve, 'r-', linewidth=2)
plt.plot(Curve1, 'g-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("Fitness", fontsize='medium')
plt.legend(["BOA", "IBOA"])
plt.grid()
# plt.title('BOA', fontsize='large')
plt.show()

# # 绘制搜索空间
# fig = plt.figure(2)
# ax = Axes3D(fig)
# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# X, Y = np.meshgrid(X, Y)
# Z = X ** 2 + Y ** 2
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
