#! /usr/bin/env python

#Prof Tinker    
#Poisson Equation E&M
import numpy as np 
import matplotlib.pyplot as plt 
import operator

from  astropy import constants as const 

from functools import reduce

def prod(iterable):
    return reduce(operator.mul, iterable, 1)


p_dat = np.loadtxt('particles.dat')

#Coordinates of charges
#x = p_dat[:,0]
#y = p_dat[:, 1]

#Constants
M = 100 #Grid Size
V = 0 #Potential at the boundaries
target = 1.e-6 
e = const.e.value #Electron charge
eps0 = const.eps0.value

######################
# Part A
######################
# Assign charges to a grid using
# Cloud in Cell Technique


def cic(pos):
    rho = np.zeros([M+1, M+1], float)
    for pt in pos:
        i, j = np.array(pt, int)
        x_c = i + 0.5 
        y_c = j + 0.5 
        
        dx = abs(i - x_c)  
        dy = abs(j - y_c)
        tx = 1 - dx 
        ty = 1 - dy
        
        #Adjust the density based on
        # neighboring cells
        rho[i][j] = rho[i][j] + tx*ty
        rho[i+1][j] = rho[i+1][j] + tx*ty 
        rho[i][j+1] = rho[i][j+1] + tx*ty 
        rho[i+1][j+1] = rho[i+1][j+1] + tx*ty
        
    return rho

def part_a():
    
    rho = cic(p_dat)
    plt.imshow(rho, origin='lower', cmap = 'plasma')
    plt.xlabel('X')
    plt.ylabel('Y')
    cb = plt.colorbar(extend='both')
    cb.set_label('Charge Denisty')
    plt.savefig('cahrge_density.pdf')
    plt.show()

#######################
#   Part B
#######################
L = 1

a = L/M
def relaxation(V, M, dat, target=1.e-3):
    #Create arrays to hold the x values
    phi = np.zeros([M+1,M+1], float)

    rho = cic(dat)
    #Boundary Conditions
    phi[0,:] = V
    phi[-1, :] = V
    phi[:, 0] = V
    phi[:, -1] = V
    
    phiprime = np.empty([M+1, M+1], float)
    
    #Start things
    delta = 1.0
    count = 0
    while delta > target:
        for i in range(M+1):
            for j in range(M+1):
                if i == 0 or i == M or j == 0 or j == M:
                    phiprime[i][j] = phi[i][j]
                else:
                    #print(phi)
                    phiprime[i][j] = 0.25*(phi[i+1][j] + phi[i-1][j] +
                                    phi[i][j+1] + phi[i][j-1] + 
                                    (a**2*e/(4*np.pi*eps0))*rho[i][j])
                    
                    
        #Max difference from old values
        delta = np.max(np.abs(phi - phiprime))
        
        phi, phiprime = phiprime, phi
        #print(phi.shape, phiprime.shape)
        
        count += 1
        #print(delta, count)
        
        
    return phi, count

def part_b():
    phi, i = relaxation(0.0, 100, p_dat, target=1.e-6)
    print('Iterations:', i)
    plt.imshow(phi, origin='lower', cmap = 'plasma')
    plt.xlabel('X')
    plt.ylabel('Y Coordinates')
    c = plt.colorbar()
    c.set_label('$\Phi(x,y)$')
    plt.savefig('relaxed_potential_field.pdf')

    plt.show()
    
    
############################
#   Part C  
############################

def part_c():
    
    # Golden-Ratio search 
    def golden_minimization(func, xL, xR, tol=1.e-3, *args, **kwargs):
        phi = (1.0 + np.sqrt(5.0))/2.0
        iter = 1
        #print(tol)
        err = 1 # Initial error

        # Print column headers
        print('{0}\t{1}\t{2}\t{3}'.format('Iter.','xopt','f(xopt)','Error %'))

        gs_arr = []
        
        # Iterate until target precision is reached
        while err > tol:
    
            d = (xR - xL)/phi
            a = xR - d
            b = xL + d

            result, yb = func(b, *args, **kwargs)
            result, ya = func(a, *args, **kwargs)
            if ya < yb:
                xopt = a
                xR = b
    
            else:
    
                xopt = b
                xL = a
            
            #sxopt = (xR + xL)/2
            err = (2-phi)*((xR - xL)/xopt)*100

            result, func_iter = func(xopt, *args, **kwargs)
            # Print iteration, x-optimal, f(x-optimal), %Error 
            print('{0}\t{1}\t{2}\t{3}'.format(int(iter),
                                        round(xopt, 4),
                                        int(func_iter),
                                        round(err, 5)))
                  
            iter += 1
            gs_arr.append(xopt)
            
        return np.array(gs_arr), iter
        
        
        
    def gs_orelaxation(omega, M, data , target=1.e-5, a=1):
        phi = np.zeros([M+1, M+1], float)
        
        rho = cic(data)
        
        
        # Main loop
        count = 0
        delta = 1.0
        #phi[0, :] = 1.0
        #print(omega)
        while delta>target:
            delta = 0
            # Calculate new values of the potential
            for i in range(M+1):
                for j in range(M+1):
                    if not i == 0 and not j == 0 and not i == M and not j == M:
                        old_phi = phi[i][j]
                        new_phi = (1+omega)*(phi[i+1,j] + phi[i-1,j] \
                                            + phi[i,j+1] + phi[i,j-1] + (a**2*e/(4*np.pi*eps0))*rho[i][j])/4 - omega*old_phi
                        phi[i][j] = new_phi
                        delta = max([delta, abs(new_phi - old_phi)])
                        
                    
            count += 1
        

        return phi, count 
    def plot_phi_rat():
        ws = np.linspace(0.1,0.99, 20)
        iters = []
        for w in ws:
            phi, i = gs_orelaxation(100, p_dat, w, target=1.e-10)
            iters.append(i)
        
        iters = np.array(iters)
        plt.plot(ws, iters)
        plt.xlabel('$\omega$')
        plt.ylabel('Iterations')
        plt.savefig('omega_i.pdf')
        plt.show()
    def plot_rat():
        f_args = (100,p_dat)
        omegas, counts = golden_minimization(gs_orelaxation, 0.9,0.99,1.e-3, *f_args, a=1,target=1.e-10)
        plt.plot(range(omegas.size), omegas)
        plt.grid(True)
        plt.xlabel('Iteration Step', fontsize=12)
        plt.ylabel('$\omega$', fontsize=12)
        plt.show()
    def plot_ideal_phi():
        phi, i = gs_orelaxation(0.9413, 100, p_dat, target=1.e-10)
        plt.imshow(phi, cmap='plasma', origin='lower')
        plt.xlabel('X')
        plt.ylabel('Y')
        c = plt.colorbar(extend='both')
        c.set_label('$\Phi(x,y)$ with $\omega={}$'.format(0.940))
        plt.savefig('golden_phi.pdf')
        plt.show()
        
    plot_ideal_phi()
choice = input('Which part of the problem do you want to see? (a/b/c):')
if choice == 'a':
    part_a()
elif choice == 'b':
    part_b()
elif choice == 'c':
    part_c()
else:
    raise ValueError('Please choose either a, b, or c.')