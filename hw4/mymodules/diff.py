

#Marcus Dupont
#Prof Eugenio
#This program calculates first derivatives

from __future__ import print_function, division

import numpy as np 

def diff(func,x,h=1.e-6):
    "Returns the first derivative using central difference of a function evaluated at a value x"""
    return (func(x+h/2)-func(x-h/2))/h
