#! /usr/bin/env python



#Newman question 6.11
#Prof Tinker


import numpy as np 
import matplotlib.pyplot as plt 

#constants
c=2

def central_diff(x, func, h=1.e-3):
    y_prime = (func(x+h/2)-func(x-h/2))/h
    
    return y_prime

def relaxation(x, func, eps=1.e-6):
    i = 1
    x_arr = [0]
    while True:
        x,x_prime = func(x), x
        x_arr.append(x)
        
        error = np.abs((x-x_prime)/(1-central_diff(x, func)**(-1.)))
        
        if error <= eps:
            print("Answer:{0}, Iterations:{1}".format(x_arr[i], i))
            break
        i += 1
        
def over_relaxation(x, func, omega=0.2, eps=1.e-6):
    i = 1
    x_arr = [0]
    while True:
        x,x_prime = (1+omega)*func(x) - omega*x, x
        x_arr.append(x)
        
        error = np.abs((x-x_prime)/(1-( (1+omega)*central_diff(x, func)-omega )**(-1.)))

        
        if error <= eps:
            print("Answer:{0}, Iterations:{1}".format(x_arr[i], i))
            break
        i += 1
    

#Define the function we're integrating
def f(x):
    return 1-np.exp(-c*x)

#Print the rsults
print('Relaxation results...')
relaxation(1, f)
print("")
print('Overrelaxation Results...')
over_relaxation(1, f, omega=0.5)