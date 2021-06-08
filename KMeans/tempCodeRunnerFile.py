import numpy as np
import random
import math
from matplotlib import pyplot as plt

def rastringin(X):

    output = np.sum(np.square(X)) - np.sum(10*np.cos(2*np.pi*X)) + 20
    return output
   
# A = 10
# X = 1
# Y = 2
# Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)
# print('--', rastringin(np.array([1, 2])))
# print('Z', Z)

def alpine(X):
    output = np.sum(np.abs(0.1*X + X*np.sin(X)))
    return output

# print('--', alpine(np.array([1, 2])))
# print('Z', Z1)


