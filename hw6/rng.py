#! /usr/bin/env python


#Prof Tinker
#HW 6 Prob 3

import numpy as np 
import matplotlib.pyplot as plt 

#A group of modules I created
from modules import *
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

#Newman's constants
a = 1664525
c = 1013904225
m = 2 ** 32

#Test the random sampler
#sample = random_sample(100, [0,1], a, c, m)
#rand = random_sample(1, [0,1], a, c, m)
#print(rand)

N = 10000
xs = []
ys = []
sigma = 1
x, y = gaussian_rng(N, sigma)
    
xs = np.array(x)
ys = np.array(y)

x_hist = np.histogram(xs)
norm_fac = x_hist[0].max()
#print(norm_fac)

#print(r.max())

plt.hist(xs)
plt.yscale('log', nonposy='clip')
#Normalize the unit Gaussian then scale it
#to the max bin height in the plot
gy = gaussian(xs, sigma)
gy = gy/gy.max()

plt.scatter(xs, norm_fac*gy, c='red', zorder=2)
plt.xlabel('Random Variable', fontsize=15)
plt.ylabel('Frequency', fontsize=15)

plt.savefig('gaussian_hist.pdf')
plt.show()

#Part C
#power, kvals = dft(xs)
#plt.loglog(kvals, power)
#plt.ylabel('Log P', fontsize=15)
#plt.xlabel('Log K', fontsize=15)
#plt.savefig('power_spec.pdf')
#plt.show()

#Part D

start = -1 if random() < 0.5 else 1
walk = [start]
#Generate the random walk
for i in range(1, N):
    movement = xs[i]  #i-th Gaussian Number
    value = walk[i-1] + movement
    walk.append(value)
walk = np.array(walk)
plt.plot(walk)
plt.xlabel('Iteration Number', fontsize=15)
plt.ylabel('Posotion', fontsize=15)
plt.grid(True)
plt.savefig('random_walk.pdf')
plt.show()

#Part E
power, kvals = dft(walk)
print(kvals)
#Scale the theoretical relation by the max
#value in the calculate power spectrum
theory = power.max()*kvals**(-2.)

plt.loglog(kvals, power,'r', label='Raw Data')
plt.loglog(kvals, theory,'b--', label='Theoretical')
plt.ylabel('Log P', fontsize=15)
plt.xlabel('Log K', fontsize=15)
plt.grid(True)
plt.legend()
plt.savefig('power_spec_walk.pdf')
plt.show()