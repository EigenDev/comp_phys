#! /usr/bin/python

#Prof Tinker
#Integration Stuff
import numpy as np 
import matplotlib.pyplot as plt

#Using numpy's float32 dtype ensures single-point precision

def midpoint(func, a, b, n, true_val):
    """
    Midpoint Rule integration method 

    returns error and integral value
    """
    h = np.float32( (b-a)/ n)
    val = 0
    for i in range(n):
        val += np.float32(func((a + h/2) + i*h))

    val *= h
    rel_err = np.abs(-true_val + val)/np.abs(val)
    
    return np.array([val, rel_err], dtype=np.float32)
    
def trapezoidal(func,a,b,n, true_val):
    """ 
    Trapezoidal integration method defined using the standard
    trapzoudal rule

    Returns value of integral given bounds and an arbitrary function
    """

    h = np.float32((b-a)/(n-1))

    val = np.float32((func(a)+func(b))/2)
    for i in range(1,n-1):
        val += np.float32(func(a+h*i))

    val *= h
    rel_error = np.abs(-true_val + val)/np.abs(val)

    return np.array([val, rel_error], dtype=np.float32)

def simpson(func,a,b,n, true_val):
    """ 
    Simpson's integration method defined using the standard
    Simpson's Rule

    Returns value of integral given bounds and an arbitrary function
    """
    w = np.float32((b-a)/n)

    val = 0
    for i in range(0,n):
        val += np.float32(func(a+i*w) + 4*func(a+(i+0.5)*w) + func(a+(i+1)*w))

    val *= w/6
    rel_err = np.abs(-true_val + val)/np.abs(val)

    return np.array([val, rel_err], dtype=np.float32)

def f(x):
    return np.exp(-x)

N = np.linspace(2, 1e6, 10, dtype=int)
true_val = -np.float32(np.exp(-1)) + 1

simps_arr = []
for num in N:
    simps_arr.append(simpson(f, 0, 1, num, true_val)[1])

simps_arr = np.array(simps_arr, dtype=np.float32)

trap_arr = []
for n in N:
    trap_arr.append(trapezoidal(f, 0, 1, n, true_val)[1])
trap_arr = np.array(trap_arr, dtype=np.float32)


midpoint_arr = []
for n in N:
    midpoint_arr.append(midpoint(f, 0, 1, n, true_val)[1])

midpoint_arr = np.array(midpoint_arr, dtype=np.float32)

#Plot the results of the relative error vs bin size
plt.loglog(N, simps_arr, label='Simpson')
plt.loglog(N, trap_arr, label='Trapezoidal')
plt.loglog(N, midpoint_arr, label='Midpoint')

plt.xlabel('N', fontsize=25)
plt.ylabel('$\epsilon$', fontsize=25)
plt.legend()

plt.savefig('integration.pdf', bbox_inches='tight')
plt.show()