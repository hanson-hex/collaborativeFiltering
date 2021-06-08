import numpy as np
import math
from matplotlib import pyplot as plt


def line (MaxIter):
	Curve = np.zeros([MaxIter, 1])
	for t in range(MaxIter):
		Curve[t] = 2 - 2*t/MaxIter
	return Curve

def exp(MaxIter):
	Curve = np.zeros([MaxIter, 1])
	for t in range(MaxIter):
		Curve[t] = 2 * math.exp(-t/MaxIter)
	return Curve

MaxIter = 5000
Curve1 = line(MaxIter)
Curve2 = exp(MaxIter)


plt.figure(1)
plt.plot(Curve1, 'g-', linewidth=2)
plt.plot(Curve2, 'y-', linewidth=2)
plt.xlabel('Iteration', fontsize='medium')
plt.ylabel("α", fontsize='medium')
plt.legend(["linear", "exponential"])
plt.grid()
# plt.title('BOA', fontsize='large')
plt.show()