#! /usr/bin/env python

#Newman 10.11
#Prof Tinker
import numpy as np 
import matplotlib.pyplot as plt
from random import random, randint
from matplotlib import collections  as mc


L = 50 #The size of the lattice
cube = np.zeros([L,L], float)

class Dimer():
    """
    Returns some info about the dimers in the 
    lattice
    """
    
    def __init__(self, L,T_max, T_min, t, tau):
        self.L = L
        self.T_max = T_max
        self.T_min = T_min 
        self.t = t
        self.tau = tau
        self.grid = np.zeros([self.L, self.L], dtype=int)
        
    def temp_func(self):
        """
        Temperature Function
        """
        return self.T_max*np.exp(-self.t/self.tau)
    
    def setvalue(self, i, j,m,n):
        for value in range(1,L*L//2):
            if not value in self.grid:
                break
        
        self.grid[i,j] = value
        self.grid[m,n] = value
 
    def place_dimer(self):
        T = self.T_max
        self.t = 0
        count = 0
        lines = []
        while T > self.T_min:
            self.t += 1
            T = self.temp_func()
            
            if self.t%10000==0: # for checking progress
                print('Temperature:', T)
            
            # dn = 1 when dimmer is added, 0 do nothing, -1 when dimer is removed
            # First two always happens but when it comes to removing it 
            # it is done with probability exp(-1/T)
            
            i,j = randint(0, L-1), randint(0,L-1)
            
            # Since it is possible to step out from boundary
            m,n = -1,-1
            while m<0 or m==L or n<0 or n==L:
                k = randint(0,4)
                m,n = i,j
                if k == 0:
                    m+=1
                elif k == 1:
                    m-=1
                elif k == 2:
                    n+=1
                elif k == 3:
                    n-=1
            
            if self.grid[i,j]==0 and self.grid[m,n]==0:
                self.setvalue(i,j,m,n)
                
                
                
                lines.append([(i,j), (m,n)])
                #Add new dimer, but skip duplicates
                #if count < 1:
                #    lines.append([(i,j),(m,n)])
                #else:
                #    for coordinate in lines:
                #        if (i,j) in coordinate:
                #            #print('skipped')
                #            pass
                #        else:
                #            #print('Added')
                #            lines.append([(i,j),(m,n)])
                        
                count +=1
            
            elif self.grid[i,j]== self.grid[m,n]: # because of previous if both aren't 0
                
                if random() < np.exp(-1/T):
                    self.grid[i,j]=0
                    self.grid[m,n]=0
                    count-=1
                    if [(i,j), (m,n)] in lines:
                        #print('Removing dimer...')
                        lines.remove([(i,j),(m,n)])
                    else:
                        pass
            else:
                pass # Otherwise do nothing
        
        return count, lines
        
    def plot_dimer(self):
        ndimers, lines = self.place_dimer()
        lc = mc.LineCollection(lines, linewidths=2)
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig('dimer1000.pdf')
        #ax.margins(0.1)
        #plt.imshow(grid, cmap='plasma')