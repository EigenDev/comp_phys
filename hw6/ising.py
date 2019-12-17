#! /usr/bin/env python

#Newman Ex. 10.9
#Prof Tinker   

import numpy as np 
import matplotlib.pyplot as plt 

from random import random,randint, seed

seed(1)
#Define our constants
L = 20 #size of our lattice square
J = 1 #positive interaction constant
T =1 #temperature
kB = 1 #Boltzmann constant is normalized to 1
beta = 2/(kB*T) #The inverse temperature

#steps = 250000 #Number of Monte Carlo moves
            

#Let's build a class of the Ising model 
#for some practice with object-oriented 
#programming
class Ising():
    """
    A class used to perform the Ising 
    model given user-defined parameters
    
    INPUTS:
    s -- the initial configuration of the randomized particles
    kB -- k Boltzmann constant if user wants to use it
    T -- temperature of the system
    """
    def __init__(self, L, kB, T, steps):
        self.L = L 
        self.kB = kB 
        self.T = T
        self.beta = 1/(kB*T)
        self.steps = steps
        #Initialize our lattice of random particles
        self.config = np.zeros([self.L,self.L], dtype=int)

        #Fill the lattice with randomly aligned
        #particles
        for x in range(L):
            for y in range(L):
                if random()<0.5:
                    self.config[x,y] = 1
                else:
                    self.config[x,y] = -1
        
    def energy_function(self):
        """Return the energy of a system 
        of magnetized particles given by the Ising 
        model
        
        Inputs:
        None
        
        Outputs:
        energy -- the energy of the lattice after summing
        through the respective pairs
        """
        E = 0
        for i in range(len(self.config)):
            for j in range(len(self.config)):
                s = self.config[i,j]
                #Calculate the impact of neighboring particle pairs
                neighbors = (self.config[(i+1)%L, j] +
                              self.config[i, (j+1)%L] + 
                              self.config[(i-1)%L, j] + 
                              self.config[i, (j-1)%L])
                E += -J*s*neighbors
        #fix for extra neighbors
        return E/4 
    
    def calcMag(self):
        """ 
        Calculate the total magnetization
        of the configuration
        """
        M = np.sum(self.config)
        return M

    def metro_monte_carlo(self, save_conf = False):
        """
        Computes the Monte Carlo simulation making
        use of the Metropolis probability check
        """
        E1 = self.energy_function()
        mplot = []
        M = self.calcMag()
        for k in range(self.steps):
            #Pick a random particle coordinate
            x = randint(0,L-1)
            y = randint(0,L-1)
            self.config[x, y] *= -1
            E2 = self.energy_function()
            
            dE = E2 - E1
            
            if dE > 0:
                if random() < np.exp(-self.beta*dE):
                    E1 = E2
                    M = self.calcMag()
                else:
                    self.config[x, y] *= -1
            else:
                E1 = E2 
                M = self.calcMag()
                
            mplot.append(M)
            
            if save_conf:
                if k == 100: self.cplot(self.fig, k,2)
                if k == self.steps/100: self.cplot(self.fig,k, 3)
                if k == self.steps-1: self.cplot(self.fig, k, 4)
                
        return mplot
    
    def cplot(self, figure, i, n):
        """
        Plot the configuration at some specific MC times
        """
        xx, yy = np.meshgrid(range(self.L), range(self.L))
        ax = figure.add_subplot(2,2,n)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)      
        plt.pcolormesh(xx, yy, self.config, cmap=plt.cm.RdBu);
        plt.title('Time=%d'%i, fontsize=20)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y',fontsize=12) 
        plt.axis('tight') 
        self.ax = ax
        
    def simulate(self):
        """
        Simulates the particle configuration
        """
        self.fig = plt.figure(figsize=(15,15), dpi=80)
        self.cplot(self.fig, 0 , 1)
        
        #Save the configuration at the defined times
        self.metro_monte_carlo(save_conf=True)
        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        #fig.colorbar(self.ax, cax=cbar_ax)
            
        plt.savefig('config.pdf')
    def mplot(self):
        mplot = self.metro_monte_carlo()
        
        ax = plt.subplot(111)
        ax.semilogx(mplot)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_xlabel('Monte Carlo Time Steps', fontsize=20)
        ax.set_ylabel('Magnetization', fontsize=20)
        plt.savefig('mplot_seed1.pdf')
        
        plt.show()

    
if __name__ == '__main__':
    T = input('Input a Temperature: \t')
    T = int(T)
    s = Ising(20, 1, T, 100000)
    s.simulate()
    
