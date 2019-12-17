#A place for modules that can be used across
#programs within this directory


import numpy as np 
import matplotlib.pyplot as plt 
from random import random 


#Newman's constants
a = 1664525
c = 1013904225
m = 2 ** 32

def gaussian(x, sigma):
    """
    Used to feed numbers into the Gassian (normal)
    distribution.
    
    INPUTS:
    sigma -- the width of the Gaussian
    x -- in input values from the distribution
    
    
    OUTPUTS:
    p -- the probability distribution from the standard Gaussian
    function
    """
    norm = 1/np.sqrt(2*np.pi*sigma**2)
    exp_arg = -x**2/(2*sigma**2)
    
    return norm*np.exp(exp_arg)

def lcg(a,c,m,seed):
    """ The LCG implementation"""
    xi = seed
    while True:
        xf = (a*xi + c) % m
        xi = xf 
        yield xf
        

def random_sample(n, interval,a,c,m, seed = 20200420162000):
    """
    Generate random sample within given interval
    """
    lower, upper = interval[0], interval[1]
    sample = []
    glibc = lcg(a, c, m, seed)       # parameters as in GNU C Library

    for i in range(n):
        observation = (upper - lower) * (next(glibc) / m) + lower
        sample.append(float(observation))
    if n > 1:
        return np.array(sample)
    else:
        return sample[0]

def gaussian_rng(n, sigma):
    """
    Generate a set of Gaussian variables
    """
    rand = random_sample(n, [0,1],a,c,m)
    r = np.sqrt(-2*sigma**2*np.log(1 - rand))
    #rand = random_sample(1, [0,1],a,c,m)
    thetas = []
    for i in range(n):
        theta = 2*np.pi*random()
        thetas.append(theta)
        
    thetas = np.array(thetas)
    x = r*np.cos(thetas)
    y = r*np.sin(thetas)
    
    return x, y

def dft(samples):
    """
    Performs the Discrete Fourier Transform
    """
    N = len(samples)
    c = np.zeros(N, dtype=complex)

    #Check if N even or odd
    if N/2 % 2 == 0:
        k_range = int(N/2 + 1)
    else:
        k_range = int((N+1)/2)
    print('Performing DFT...')
    mag_c = np.zeros(k_range,dtype=float)
    
    for k in range(k_range):
        for n in range (N):
            c[k] += samples[n]*np.exp(2j*np.pi*k*n/N)
            
        mag_c[k] = np.absolute(c[k])**2

    k_arr = np.array(range(k_range))
    return (mag_c, k_arr)