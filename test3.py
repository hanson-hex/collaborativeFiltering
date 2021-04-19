# -*- coding: utf-8 -*-
# @Author  : Peidong
# @Site    : 
# @File    : recommendsystem.py
# @Software: PyCharm
# 使用MovieLens数据集，它是在实现和测试推荐引擎时所使用的最常见的数据集之一。它包含来自于943个用户
# 以及精选的1682部电影的100K个电影打分。
import numpy as np
import math
import pandas as pd

def aa(row1, row2):
    avg1 = np.mean(row1)
    avg2 = np.mean(row2)
    a = 0
    b = 0
    c = 0
    for i in range(len(row1)):
        a += (row1[i] - avg1)*(row2[i] - avg2)
        b += pow(row1[i] - avg1, 2)
        c += pow(row2[i] - avg2, 2)
    return a / (math.sqrt(b)*math.sqrt(c))