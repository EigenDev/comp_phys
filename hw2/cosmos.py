#! /usr/bin/env python


#Cosmos
#Prof Tinker
import numpy as np 
import matplotlib.pyplot as plt 
from inspect import signature 

def central_diff(x, func, h=1.e-3, *args, **kwargs):
    """
    First derivative function
    
    wrt indicates which argument to differentiate wrt (i.e., 2nd,3rd)
    """
    y_prime = (func(x+h/2, *args, **kwargs)-func(x-h/2, *args, **kwargs))/h
    return y_prime
    

#Create three functions with fitting params in different places
def schechter_mstar(m_star,m_gal=None, alpha=0, phi=0):
    """
    Return the Schedter function with free parameters
    alpha, phi, and m_star
    """
    
    return phi*(m_gal/m_star)**(alpha+1)*np.exp(-m_gal/m_star)*np.log(10)

def schechter_phi(phi, m_star = 0,m_gal = 0, alpha = 0):
    """
    Return the Schedter function with free parameters
    alpha, phi, and m_star
    """
    return phi*(m_gal/m_star)**(alpha+1)*np.exp(-m_gal/m_star)*np.log(10)

def schechter_alpha(alpha,m_star = 0, m_gal=0, phi=0):
    """
    Return the Schedter function with free parameters
    alpha, phi, and m_star
    """
    return phi*(m_gal/m_star)**(alpha+1)*np.exp(-m_gal/m_star)*np.log(10)



def f(x):
    if isinstance(x, (np.ndarray, list)):
        return (x[0]-2)**2 + (x[1]-2)**2
    else:
        return (x-2)**2

def grad_descent(func, gamma, eps=1.e-6, dim=1,x=0.9, **kwargs):
    """
    Gradient Descent Method
    """
    #Define some constants and arrays
    diff = 1
    i = 0
    
    if dim > 1:
        x = np.zeros(dim)
        while diff > eps:
            grad_vec = []
            
            for i in range(dim):
                grad_vec.append(central_diff(x[i], func, **kwargs))
                
            grad_vec = np.array(grad_vec)
            x,x_prime = x - gamma * grad_vec, x
            i += 1
            diff = np.abs(x - x_prime)
            diff = diff[0]
    else:
        while diff > eps:
            grad = central_diff(x, func, **kwargs)
            #print(grad)
            grad = np.array(grad)
            #print('Grad vec: {}'.format(grad))
            x,x_prime = x - gamma * grad, x
            #print(x)
            i += 1
            diff = np.abs(x - x_prime)
            diff = diff
            #print(diff)
    print('The minimum occurs at =:{}'.format(x))
    print('Iteration Count: {}'.format(i))
    return x
        
#Test the gradient descent on the simple function
# f(x, y) = (x-2)**2 + (y-2)**2
grad_descent(f, 0.9, dim=2)  

#Load in the data file
smf_cosmos = np.loadtxt('smf_cosmos.dat')


log_mgal = smf_cosmos[:,0]
n_mgal = smf_cosmos[:,1]
error_n = smf_cosmos[:,2]

#Perform grad descent for different params
mstars = []
alphas = []
phis = []

for mass in log_mgal:
    min_mstar = grad_descent(schechter_mstar, 10,x=10, dim=1, phi=1.e-9,alpha=-1.2,m_gal=mass, eps=1.e-6)
    min_phi = grad_descent(schechter_phi, 0.1,x=1.e-6, dim=1, m_star=11,alpha=1.5,m_gal=mass, eps=1.e-1) 
    min_alpha = grad_descent(schechter_alpha,0.2,x=-1.2 , dim=1, phi=1.e-9,m_star=11,m_gal=mass, eps=1.e-4) 

    mstars.append(min_mstar)
    phis.append(min_phi)
    alphas.append(min_alpha)

mstars = np.array(mstars)
alphas = np.array(alphas)
phis = np.array(alphas)

print(schechter_mstar(mstars,m_gal=log_mgal,
                                        alpha=alphas, phi=phis))
plt.plot(log_mgal, schechter_mstar(mstars,m_gal=log_mgal,
                                        alpha=alphas, phi=phis), 'ro')
plt.semilogy(log_mgal, n_mgal)
plt.show()