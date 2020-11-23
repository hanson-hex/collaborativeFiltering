 #!/usr/bin/python
 # -*- coding: utf-8 -*-
 
user_data = {"小明": {"张学友": 4, "周杰伦": 3, "刘德华": 4},
           "小海": {"张学友": 5, "周杰伦": 2},
           "李梅": {"周杰伦": 3.5, "刘德华": 4},
           "李磊": {"张学友": 5, "刘德华": 3}}
 
class slopOne:
   def __init__(self,data):
       self.frequency={}
       self.deviation={}
       self.allgood = []
       self.data=data
       self.filledData = {}

   #计算所有item之间评分偏差
   def computeDeviation(self):
       for ratings in self.data.values():
           for item,rating in ratings.items():
               self.frequency.setdefault(item,{})
               self.deviation.setdefault(item,{})
               for item2,rating2 in ratings.items():
                   if item!=item2:
                       self.frequency[item].setdefault(item2,0)
                       self.deviation[item].setdefault(item2,0.0)
                       self.frequency[item][item2]+=1#两个项目的用户数
                       self.deviation[item][item2]+=(rating-rating2)#累加两个评分差值
                   if item2 not in self.allgood:
                       self.allgood.append(item2)

       for item,ratings in self.deviation.items():
           for item2 in ratings:
               ratings[item2]/=self.frequency[item][item2]
   def fillZero(self):
      for user, rattings in self.data.items():
        self.filledData.setdefault(user, {})
        self.filledData[user] = self.predictRating(rattings)
   #评分预测
   def predictRating(self,userRatings):
       recommendations={}
       frequencies={}
       a = {}
       for item,rating in userRatings.items():
           for diffItem,diffRating in self.deviation.items():
               if diffItem in userRatings:
                 a.setdefault(diffItem, userRatings[diffItem])
               if diffItem not in userRatings and item in self.deviation[diffItem]:
                   fre=self.frequency[diffItem][item]
                   recommendations.setdefault(diffItem,0.0)
                   frequencies.setdefault(diffItem,0)
                   # 分子部分
                   recommendations[diffItem]+=(diffRating[item]+rating)*fre
                   # 分母部分
                   frequencies[diffItem]+=fre
                   a[diffItem] = recommendations[diffItem] / frequencies[diffItem]
      #  recommendations=[(k,v/frequencies[k]) for (k,v) in recommendations.items()]

       #排序返回前k个
      #  recommendations.sort(key=lambda a_tuple:a_tuple[1],reverse=True)
       return a

if __name__=='__main__':
    r= slopOne(user_data)
    r.computeDeviation()
    r.fillZero()