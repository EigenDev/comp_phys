#! /usr/bin/env python

#Prof Tinker
#Newman Exercise 9.7

import numpy as np 
import matplotlib.pyplot as plt 

#gravity in m/s^2
g = 9.8 
     
#Constants
M = 100 #number of grid points
t = 10 #time in seconds
x_0 = 0.0 #boundary problem
target = 1.e-6 
h = t/100 #time spacing in seconds

#Create arrays to hold the x values
x = np.zeros(M+1, float)
x[0] = x_0 
x[-1] = x_0
xprime = np.empty(M+1, float)

delta = 1.0
while delta > target:
    for i in range(M+1):
        if i == 0 or i ==M:
            xprime[i] = x[i]
            
        else:
            xprime[i] = (x[i+1] + x[i-1] + h**2*g)/2
            
    #Calculate the max difference from old values
    #print(x)
    delta = np.max(np.abs(x - xprime))

    #print(delta)
    #zzzz = input('')
    #Swap the two arrays around
    x, xprime = xprime, x 

print(x)
t = np.linspace(0, 10, x.size)
plt.plot(t, x)
plt.xlabel('Time [s]')
plt.ylabel('Height [m]')

plt.savefig('traj.pdf')
plt.show()