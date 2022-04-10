from KMeans.BOAKmeans import BOAK
import math 
import random
import os
import pandas as pd
import datetime
import json
import imp
from KMeans.KMeans import *
# from KMeans.DBI import *

### 1M 的数据集 rating是6040位用户对3952部电影的评分数据共计（1000209条 ）

class UserBasedCF:
    def __init__(self, path, splitToken):
        self.train = {} #用户-物品的评分表 训练集
        self.test = {} #用户-物品的评分表 测试集
        self.pred = {} # // 预测评分
        self.records = []
        self.splitToken = splitToken
        self.pf = None
        self.generate_dataset(path)

    def loadfile(self, path):
        with open(path, 'r', encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                yield line.strip('\r\n')
    
    def getRecord(self):
        # rui:是用户u对物品i的实际评分，pui是算法预测出来的用户u对物品i的评分
        for u, items in self.test.items():
            for i in items.keys():
                self.records.append([u,i, items[i], self.pred[u].get(i, self.average_rating(u))])

    def generate_dataset(self, path, pivot=0.7):
        # 读取文件，并生成用户-物品的评分表和测试集
        i = 0
        for line in self.loadfile(path):
            user, movie, rating, _ = line.split(self.splitToken)
            if i <= 10:
                print('{},{},{},{}'.format(user, movie, rating, _))
            i += 1  
            if random.random() < pivot:
                self.train.setdefault(user, {})
                self.train[user][movie] = int(rating)
            else:
                self.test.setdefault(user, {})
                self.test[user][movie] = int(rating)
        df = pd.DataFrame(self.train).T
        self.df = df.apply(lambda row: row.fillna(row.mean()), axis=1)
        records = {
          "train": self.train
        }
        with open("./train.json","w") as f:
            json.dump(records,f, indent=4)
            print("加载入文件完成...")

    def calUserSim(self):  
        # 建立物品-用户的倒排表
        movie_user = {}
        for ukey in self.train.keys():
            for mkey in self.train[ukey].keys():
                if mkey not in movie_user:
                    movie_user[mkey] = []
                movie_user[mkey].append(ukey)  
    
        # calculated co-rated movies between users
        C = {}
        for movie, users in movie_user.items():
            for u in users:
                C.setdefault(u,{})
                for n in users:
                    if u == n:
                        continue
                    C[u].setdefault(n,[])
                    C[u][n].append(movie)  
    
        # 计算用户-用户共现矩
        userSim = {}
        for u in C.keys():  

            for n in C[u].keys():  

                userSim.setdefault(u,{})
                userSim[u].setdefault(n,0)  
    
                average_u_rate = self.average_rating(u)
                average_n_rate = self.average_rating(n)  
    
                part1 = 0
                part2 = 0
                part3 = 0
                for m in C[u][n]:  

                    part1 += (self.train[u][m]-average_u_rate)*(self.train[n][m]-average_n_rate)*1.0
                    part2 += pow(self.train[u][m]-average_u_rate, 2)*1.0
                    part3 += pow(self.train[n][m]-average_n_rate, 2)*1.0  
    
                part2 = math.sqrt(part2)
                part3 = math.sqrt(part3)
                if part2 == 0:
                    part2 = 0.001
                if part3 == 0:
                    part3 = 0.001
                userSim[u][n] = part1 / (part2 * part3)
        self.W = userSim
        return userSim

    def UserSimilarity(self):
        #建立物品-用户的倒排表
        self.item_users = dict()
        for user,items in self.train.items():
            for i in items.keys():
                if i not in self.item_users:
                    self.item_users[i] = set()
                self.item_users[i].add(user)
        # 计算用户-用户共现矩阵
        C = dict()  #用户-用户共现矩阵
        N = dict()  #用户产生行为的物品个数
        for i,users in self.item_users.items():
            for u in users:
                N.setdefault(u,0)
                N[u] += 1
                C.setdefault(u,{})
                for v in users:
                    if u == v:
                        continue
                    C[u].setdefault(v,0)
                    C[u][v] += 1

        # 计算用户-用户相似度，余弦相似度
        self.W = dict()
        # 相似度矩阵
        for u,related_users in C.items():
            self.W.setdefault(u,{})
            for v,cuv in related_users.items():
                self.W[u][v] = cuv / math.sqrt(N[u] * N[v])
        return self.W, C, N

    def average_rating(self, user):
        average = 0
        for u in self.train[user].keys():
            average += self.train[user][u]
        average = average * 1.0 / len(self.train[user].keys())
        return average 
       
    def calKmeansUserSim(self, cluster, indexCluster):
        def aa (row1, row2):
            avg1 = np.mean(row1)
            avg2 = np.mean(row2)
            a = 0
            b = 0
            c = 0
            for i in range(len(row1)):
                a += (row1[i] - avg1)*(row2[i] - avg2)
                b += pow(row1[i] - avg1, 2)
                c += pow(row2[i] - avg2, 2)
            if b == 0:
                b = 0.001
            if c == 0:
                c = 0.001
            return a / (math.sqrt(b)*math.sqrt(c))
        self.KmeansW = dict()
        for index, clusterItem in enumerate(cluster):
            indexClusterItem = indexCluster[int(index)]
            for i, item in enumerate(clusterItem):
                for j, jItem in enumerate(clusterItem):
                    if (i >= j):
                        if len(clusterItem) == 1:
                            u1 = indexClusterItem[i]
                            u1 = str(u1)
                            self.KmeansW.setdefault(u1, {})
                            self.KmeansW[u1][u1] = 1
                        continue
                    u1 = indexClusterItem[i]
                    u1 = str(u1)
                    self.KmeansW.setdefault(u1, {})
                    u2 = indexClusterItem[j]
                    u2 = str(u2)
                    self.KmeansW[u1][u2] = aa(item, jItem)
                    self.KmeansW.setdefault(u2, {})
                    self.KmeansW[u2][u1] = self.KmeansW[u1][u2]

    def getKmeansPredition(self, K):
        self.pred = {}
        for u, items in self.train.items():
            self.pred.setdefault(u, {})
            average_u_rate = self.average_rating(u)
            sumUserSim = 0
            # # 用户user产生过行为的item
            action_item = self.train[u].keys()
            for v,wuv in sorted(self.KmeansW[u].items(),key=lambda x:x[1],reverse=True)[0:K]:
                average_n_rate = self.average_rating(v)
                # 遍历前K个与user最相关的
                # i：用户v有过行为的物品i
                # rvi：用户v对物品i的打分
                for i,rvi in self.train[v].items():
                    if i in action_item:
                        continue
                    self.pred[u].setdefault(i, 0)
                    self.pred[u][i] += wuv * (rvi - average_n_rate)
                sumUserSim += wuv

            for i, rating in self.pred[u].items():
                if sumUserSim == 0:
                    result = 0.0
                else:
                    result = (self.pred[u][i]*1.0) / sumUserSim
                self.pred[u][i] = average_u_rate + result

    def getAllUserPredition(self, K):
        self.pred = {}
        for u, items in self.train.items():
            self.pred.setdefault(u, {})
            average_u_rate = self.average_rating(u)
            sumUserSim = 0
            # # 用户user产生过行为的item
            action_item = self.train[u].keys()
            for v,wuv in sorted(self.W[u].items(),key=lambda x:x[1],reverse=True)[0:K]:
                average_n_rate = self.average_rating(v)
                # 遍历前K个与user最相关的
                # i：用户v有过行为的物品i
                # rvi：用户v对物品i的打分
                for i,rvi in self.train[v].items():
                    if i in action_item:
                        continue
                    self.pred[u].setdefault(i, 0)
                    self.pred[u][i] += wuv * (rvi - average_n_rate)
                sumUserSim += wuv

            for i, rating in self.pred[u].items():
                self.pred[u][i] = average_u_rate + (self.pred[u][i]*1.0) / sumUserSim

    def kMeans(self, K, itter):
        data = self.df.values
        U, C, itter, cluster, indexCluster = Kmeans(data, K, itter)
        for i, clusterItem in enumerate(cluster):
            print('index: {}, ClusterItem: {}'.format(i, indexCluster[i]))
        self.calKmeansUserSim(cluster, indexCluster)
        records = {
            "KmeansW": self.KmeansW
        }
        with open("./KMeansW.json","w") as f:
            json.dump(records,f, indent=4)
            print("KmeansW 保存成功")

    def calKOfKMeans(self):
        pass

    #给用户user推荐，前K个相关用户
    def Recommend(self,u,K=3,N=10):
        rank = dict()
        action_item = self.train[u].keys()    
        # 用户user产生过行为的item
        # v: 用户v
        # wuv：用户u和用户v的相似度
        for v,wuv in sorted(self.W[u].items(),key=lambda x:x[1],reverse=True)[0:K]:
            # 遍历前K个与user最相关的用户
            # i：用户v有过行为的物品i
            # rvi：用户v对物品i的打分
            for i,rvi in self.train[v].items():
                if i in action_item:
                    continue
                rank.setdefault(i,0)
                # 用户对物品的感兴趣程度：用户u和用户v的相似度*用户v对物品i的打分
                rank[i] += wuv * rvi
        return dict(sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:N])   #推荐结果的取前N个
    
    # 计算召回率和准确率
    # 召回率 = 推荐的物品数 / 所有物品集合
    # 准确率 = 推荐对的数量 / 推荐总数
    def recallAndPrecision(self,k=8,nitem=10):
        hit = 0
        recall = 0
        precision = 0
        for user, items in self.test.items():
            rank = self.Recommend(user, K=k, N=nitem)
            hit += len(set(rank.keys()) & set(items.keys()))
            recall += len(items)
            precision += nitem
        return (hit / (recall * 1.0),hit / (precision * 1.0))

class Evalution:
    def __init__(self, records):
        self.records = records

    def RMSE (self):
        return math.sqrt(\
            sum([(rui - pui) * (rui - pui) for u, i , rui, pui in self.records])\
            / float(len(self.records)))

    def MAE(self):
          return sum([abs(rui - pui) for u, i, rui, pui in self.records]) / len(self.records)

def testKOnKMeans():
    with open("./data.json",'r') as load_f:
     data = json.load(load_f)
     data = data["data"]
     dim = 8
     lb = 0 * np.ones([dim, 1])  # 下边界
     ub = 1 * np.ones([dim, 1])  # 上边界
     GbestScore, GbestPositon, Curve = BOAK(50, dim, lb, ub, 500, np.array(data))
     print('GBestScore, ', GbestScore)
     print('GBestPostioon', GbestPositon)
     print('Curve', Curve)


if __name__ == '__main__':
  print("何")
  start = datetime.datetime.now()

  #path = os.path.join('data', 'ratings.dat')
  path = os.path.join('ml-100k', 'u.data')

#   ucf = UserBasedCF(path, "\t")
#   W = ucf.UserSimilarity()
#   ucf.getAllUserPredition(90)
#   record = ucf.getRecord()

#   testKOnKMeans()
    # ucf.calKOfKMeans()
#    ucf.kMeans(10, 100)
#   ucf = UserBasedCF(path, "\t") 
#   ucf.kMeans(11, 100)
#   ucf.getKmeansPredition(70)
#   record = ucf.getRecord()

#     # testKOnKMeans()
#   records = {
#       "record": ucf.records
#   }

#   with open("./record.json","w") as f:
#       json.dump(records,f, indent=4)
#       print("加载入文件完成...")

#   #   with open("./record.json",'r') as load_f:
#   #   records = json.load(load_f)

#   e = Evalution(records["record"])
#   print(e.MAE())
#   print(e.RMSE())
  