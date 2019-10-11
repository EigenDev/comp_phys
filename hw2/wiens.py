#! /usr/bin/env python

#Wien's Displacement Law
#Prof Tinker

import numpy as np 
import matplotlib.pyplot as plt 
from astropy import constants, units

class Bisection:
    """
    Root finding method using Bisection Method
    """

    def __init__(self, f, precision=0.1):
        self.f = f
        self.precision = precision

    def getRoot(self, xa,xb, *args):
        """
        Returns the root of a given function along with the iteration count for each loop
        """
        
        count = 0

        while np.abs(xa-xb) > self.precision:
            x = (xa + xb)/2
            count += 1
            if self.f(x, *args)*self.f(xa, *args) > 0:
                xa = x
            else:
                xb = x
        return [x, count]
    
def f(x):
    return -5 + 5*np.exp(-x) + x

#Instantiate and define class variables
bisect = Bisection(f)
bisect.precision = 1.e-6

#Define physical constants
h = constants.h
c = constants.c 
k_B = constants.k_B

#Perform the bisection method and calculations
x = bisect.getRoot(1,5)[0]
b = h*c/(k_B*x)
lam = 502e-9 * units.m

#Print Stuff
print("The value of x is: {}".format(x))
print("The value of the displacement constant is: {}".format(b))
print("At a wavelength of 502nm, the Sun's temperature is roughly: {}".format(b/lam))
