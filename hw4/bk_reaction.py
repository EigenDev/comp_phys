#! /usr/bin/env python

#Newman Exercise 8.18
#Belousov-Zhabotinsky reaction 
#Prof Tinker

import numpy as np 
import matplotlib.pyplot as plt 
from  mymodules import odes 

def f(r, a, b):
    """
    The BZ odes input inside a vector function
    """
    x = r[0]
    y = r[1]
    fx = -(b+1)*x + a*x**2*y + 1
    fy = -a*x**2*y + b*x
    
    return np.array([fx, fy], dtype=float)


t_0 = 0
#t_max = 20
H = 20
t_arr = np.arange(t_0, t_0+H, H)

#initial conditions
x_0 = 0.0
y_0 = 0.0
delta = 1.e-10
a = 1.0
b = 3.0
r = np.array([x_0, y_0], float)
args = (a, b)
xs, ys, ts = odes.adapt_bulirsch_stoer(f, r, t_arr, H, delta, *args)

#print(solution)
print('Done')
plt.plot(ts, xs)
plt.plot(ts, ys)
plt.plot(ts, xs, 'ro')
plt.plot(ts, ys, 'go')

plt.title('Belousov-Zhabotinsky Reaction Function', fontsize = 20)
plt.xlabel('Times (s)', fontsize=20)
plt.ylabel('Concentration', fontsize = 20)

plt.savefig('BZ.png')
plt.show()