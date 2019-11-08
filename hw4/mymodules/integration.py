"""
Module Integration is a collection of useful functions which are defined, stored locally at $HOME/python/mymodules/integration.py where mymodules directory has been added to the $PYTHONPATH environment.

The Following Functions are:
prod: Returns a pi product of everything in a given iterable
trepezoid: Returns the value of a trapezoidal integral given bounds
simpson: Returns the value of a simpson' rules integral given by=ounds
monte_carlo: Returns the value of a Monte Carlo integral as well as the error

Marcus Dupont
Florida State University
Department of Physics
Feb 2018
"""


from __future__ import print_function, division

import numpy as np 
import operator
import sys

def prod(iterable):
    """Returns a pi product of the values within the array"""
    return reduce(operator.mul,iterable,1)

def trapezoidal(func,a,b,n, *args):
    """ 
    Trapexoidal integration method defined using the standard
    trapzoudal rule

    Returns value of integral given bounds and an arbitrary function
    """

    h = (b-a)/n

    val = (func(a, *args)+func(b, *args))/2
    for i in range(1,n):
        val += func(a+h*i, *args)

    return h*val

def simpson(func,a,b,n, *args):
    """ 
    Simpson's integration method defined using the standard
    Simpson's Rule

    Returns value of integral given bounds and an arbitrary function
    """
    w = (b-a)/n

    val = 0
    for i in range(0,n):
        val += func(a+i*w, *args) + 4*func(a+(i+0.5)*w, *args) + func(a+(i+1)*w, *args)
    return w*val/6

def adapt_trap(func,a,b,N=1,precision=1.e-3, *args):
    """
    Given a function, this adaptive trapezoidal method prints out various value for the 
    integral using Trapezoidal Approximation until it reaches accuracy of eps = 1e-6

    returns -- value
    """

    trap_array = [trapezoidal(func,a,b,N, *args)]
    trap_vals = []
    n_arr = []

    i = 0
    while True:
        w = (b-a)/N

        if N >= 2:
            trap_array.append(trap_array[i-1]/2)
            for k in range(1,N,2):
                trap_array[i]+= w*func(a+k*w, *args)

            trap_err = (np.abs(trap_array[i]-trap_array[i-1]))/3

            I = trap_array[i]+trap_err
            trap_vals += [I]
            n_arr += [N]

            if (trap_err <= precision):
                return (trap_vals, n_arr)
                break
        #else:
            #print("Trapezoidal Value:", trap_array[i], "Slice Number:", N)
        N*=2
        i+=1
    
def adapt_simps(func,a,b,N=1):
    """
    Given a function, this adaptive simpson method prints out various value for the 
    integral using Simpson's Rule until it reaches accuracy of eps = 1e-6

    returns -- None
    """

    simp_array = [simpson(func,a,b,N)]
    i = 0
    while True:
        w = (b-a)/N

        if N >= 2:
            simp_array.append(simp_array[i-1]/2)

            for k in range(1,N,2):
                simp_array[i]+= -w/3*func(a+w*k, *args)
            for j in range(0,N):
                simp_array[i]+= 2*w/3*func(a+w*(j+0.5), *args)

    
            simps_err = (np.abs(simp_array[i]-simp_array[i-1]))/15

            while (simps_err > 1.e-6):
                print("Simpson's Value:", simp_array[i]+simps_err, "Slice Number:",N)
                break
            if simps_err <= 1.e-6:
                break
        else:
            print("Simpson's Value:", simp_array[i]+simp_array[i]/15,"Slice Number:", N)
        N*=2
        i+=1
    

def monte_carlo(func,dim,limit,N=1000):
    """
    The Monte Carlo method for integration for multidimensional integration.
    
    Returns a mean integration value for your given function. Note that one needs
    to evaluate the error in the integration due to the random number generation.
    """
    
    I, val = 1/N,0
    I *= prod([limit[i][1]-limit[i][0] for i in range(dim)])

    for k in range(N):
        x = []
        for n in range(dim):
            x += [limit[n][0] + (limit[n][1] - limit[n][0])*np.random.random()]
        val += func(x)
    return I*val


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        prod(), trapezoidal(),simpson(),monte_carlo()
     
