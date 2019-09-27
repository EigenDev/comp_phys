#! /usr/bin/python

#Prof Tinker
#Cosmology

import numpy as np 
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt 

def simpson(func,a,b,n, *args):
    """ 
    Simpson's integration method defined using the standard
    Simpson's Rule

    Returns value of integral given bounds and an arbitrary function
    """
    const = args[0]
    w = (b-a)/n

    val = 0
    for i in range(0,n):
        val += func(a+i*w, const) + 4*func(a+(i+0.5)*w, const) + func(a+(i+1)*w, const)

    return w*val/6

def f(x, r):
    """
    integrand
    """
    #print(x)
    return (2*np.pi**2)**(-1.) *x**2 *P(x) * (np.sin(x*r)/(x*r))

r = np.arange(50, 120)

#Load in the data
data = np.loadtxt('lcdm_z0.matter_pk')
#print(data[:,3])

#Grab the relevant data parameter
k = data[:, 0]
p = data[:, 1]

#print(k)
#print(p)

P = interpolate.interp1d(k, p, kind='cubic', fill_value=0.0)




#zzz = input('Press any key to continue...')
#print('Continuing...')
xi_arr = []

n = int(3000)
for i in r:
    args = (i, 0)
    xi_arr.append(simpson(f, min(k), max(k),n, *args))


xi = np.array(xi_arr)
xi_squared = r**2*xi


#PLot The results
fig, axs = plt.subplots(2, 1, figsize=(7, 11))

axs[0].loglog(k ,P(k))
axs[0].set_xlabel('k [Mpc/h]', fontsize=10)
axs[0].set_ylabel('P(k) [MPC/h]$^3$', fontsize=10)

axs[1].loglog(r ,xi_squared)
axs[1].set_xlabel('r [Mpc/h]', fontsize=10)
axs[1].set_ylabel(r'$r^2 \xi(r)$', fontsize=10)

#Clearly indicate the peak

#Shift arrays away from max value at the beginning
r_shift = r[30:]
xi_shift = xi_squared[30:]

#Find the peak in the array
peak_index = np.where(xi_shift == np.amax(xi_shift))
r_scale = r_shift[peak_index]

axs[1].axvline(r_shift[peak_index], ymax=0.8, linestyle='--', color='r')
axs[1].annotate('r scale = {}'.format(r_scale), xy=(r_shift[peak_index], xi_shift[peak_index]),
                xytext=(r_shift[peak_index]+0.5, xi_shift[peak_index]-14), rotation=-90) 
plt.savefig('cosmology.pdf')
plt.show()

