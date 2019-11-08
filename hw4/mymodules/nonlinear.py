"""
Module Nonlinear is a collection of useful functions which are defined, stored locally at $HOME/python/mymodules/integration.py where mymodules directory has been added to the $PYTHONPATH environment.

The Following Functions are:
newtons: Returns the roots of a given function using the Newton Method

Marcus Dupont
Florida State University
Department of Physics
Feb 2018
"""

from __future__ import print_function, division
import numpy as np
import inspect

class Newton:
    """
    Root finding method using Newotn's method
        x = x_old + f(x) / df(x)/dx
    """

    def __init__(self, f, dfdx, precision=0.1):
        """
        """
        self.f, self.dfdx = f, dfdx
        self.precision = precision

    def getRoot(self, x):
        """
        Root finding method using Newton's method
        """

        lastX, count = float("inf"), 0

        while abs(x - lastX) > self.precision:
            lastX = x
            count += 1
            x = lastX - self.f(lastX)/self.dfdx(lastX)

        return [x, count]

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


class False_Position:
    """
    Root finding method using Bisection Method
    """

    def __init__(self, f, precision=0.1):
        self.f = f
        self.precision = precision

    def getRoot(self, xa,xb):
        """
        Returns the Root of a given function along with the iteration count for each root
        """
       
        count = 0
       
        xInt = xa
        xIntOld = float("inf")


        while np.abs(xInt - xIntOld) > self.precision:
            xIntOld = xInt
            count += 1
            m = (self.f(xb)-self.f(xa))/(xb-xa)
            yInt = self.f(xa) - m*xa
            xInt = -yInt/m

            if self.f(xInt)*self.f(xa) > 0:
                xa = xInt
            else:
                xb = xInt
        return [xInt, count]



class Secant:
    """
    Root finding method using Bisection Method
    """

    def __init__(self, f, precision=0.1,*args):
        self.f = f
        self.precision = precision
        self.f_args = args

    def slope(self,y,x1,x2):
        return (y(x2)-y(x1))/(x2-x1)

    def getRoot(self, a,b, shooting=False):
        """
        Returns the Root of a given function along with the iteration count for each root

        If boundary == True, the function must take the independent variable as the first
        argument
        """

     
        count = 0

        if shooting:
            while np.abs(a - b) > self.precision:

                function = self.f(a, *self.f_args)

                bound1 = self.f(a, *self.f_args)[-1]

                function = self.f(b, *self.f_args)

                bound2 = self.f(b, *self.f_args)[-1]

                x = b - bound2*(b-a)/(bound2 - bound1)
                count +=1 

                a, b = b, x
            return [x, count]

        else:


            while np.abs(a-b) > self.precision:
                x = b - self.f(b)/self.slope(self.f,a,b, self.f_args)
                count += 1
                a, b = b, x
            return [x, count]