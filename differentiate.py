#! usr/bin/env python

#Prof Jeremy Tinker
import numpy as np
import matplotlib.pyplot as plt

def differentiate(func, x, h,true_dfdx, method='forward'):
    """
    Returns the derivatives and errors of a give
    function using either the forward,central
    or extrapolated difference formula respectively
    """
    if method == 'forward':
        y_prime = (func(x+h) - func(x))/h
        rel_err = np.abs((-true_dfdx +y_prime)/y_prime)
        return np.array([y_prime, rel_err], dtype=np.float32)

    elif method == 'central':
        y_prime = (func(x+h/2)-func(x-h/2))/h
        rel_err = 3*np.abs((- true_dfdx + y_prime )/y_prime)
        return np.array([y_prime, rel_err],dtype=np.float32)

    elif method == 'extrapolated':
        y_prime = ( (1/12)*func(x-2*h) - (2/3)*func(x-h) + (2/3)*func(x+h) - (1/12)*func(x+2*h) )/h
        rel_err = np.abs((- true_dfdx + y_prime )/y_prime)
        return np.array([y_prime, rel_err], dtype=np.float32)
    else:
        raise ValueError('Please choose either the central, forward, or extrapolated\
            differentiation choices')


#initialize the test point array
x = np.array([0.1, 10], dtype=np.float32)

#intiialize the step size array
h = np.linspace(1e-6, 1, 300, dtype=np.float32)

 

#Create the figure and subplot axes
fig = plt.figure()

k = 1
for i in [1,2,3,4]:
    #print(i)
    #toggle k index between 0 and 1
    k ^= 1

    if i == 1 or i == 2:
        ax = fig.add_subplot(2, 2, i)
        dydx = np.float32(-np.sin(x[k]))
        ax.loglog(h, differentiate(np.cos, x[k], h,dydx, method='forward')[1], label='Forward')
        ax.loglog(h, differentiate(np.cos, x[k], h, dydx, method='central')[1], label='Central')
        ax.loglog(h, differentiate(np.cos, x[k], h, dydx, method='extrapolated')[1], label='Extrapolated')
        ax.set_title('Diff Cos(x) at x = %s'%(x[k]), fontsize=5)
    else:
        ax = fig.add_subplot(2, 2, i)
        dydx = np.float32(np.exp(x[k]))
        ax.loglog(h, differentiate(np.exp, x[k], h,dydx, method='forward')[1], linestyle='--')
        ax.loglog(h, differentiate(np.exp, x[k], h, dydx, method='central')[1], linestyle='--')
        ax.loglog(h, differentiate(np.exp, x[k], h, dydx, method='extrapolated')[1], linestyle='--')
        ax.set_title('Diff exp(x) at x = %s'%(x[k]), fontsize=5code c)


fig.text(0.5, 0.04, 'h', ha='center', fontsize=20)
fig.text(0.04, 0.5, '$\epsilon$', va='center', rotation='vertical', fontsize=20)

plt.savefig('differentiation_plot.pdf', bbox_inches='tight')
plt.show()