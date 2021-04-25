import os
from numpy import *
from matplotlib import pyplot as plt 
from matplotlib import cm
from op import x1, x2

t = linspace(0, 2 * pi, 200)
plt.plot(x1(t), x2(t), '-')

d = load(os.path.join(os.getcwd(), 'data', 'xyw_achiral_chiral_TM.npz'))
x, y, w = d['x'], d['y'], d['w'] 

plt.contourf(x, y, w, cmap=cm.PuBu_r)
plt.show()
