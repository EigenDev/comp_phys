#! /usr/bin/env python

#Marcus Dupont
#Professor Tinker
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


#open the file as a readable, and close automatically upon completion to save memory
with open('data/sunspots.txt', 'r') as f: 
    dat = np.array([line.split('\t') for line in f], dtype = float) 


time = dat[:,0] #Grab the first column from the file containing the months
data = dat[:,1] #Grab the second column from the file containing the data

#Pot data
fig, axs = plt.subplots(2, 1)
axs[0].plot(time, data)
axs[0].set_title('Total Sunspot Data vs Time')
axs[0].set_ylabel('Sunspot Values')
    

def dft(samples):
    N = len(samples)
    c = np.zeros(N, dtype=complex)

    #Check if N even or odd
    if N/2 % 2 == 0:
        k_range = int(N/2 + 1)
    else:
        k_range = int((N+1)/2)
    print('Performing DFT...')
    mag_c = np.zeros(k_range,dtype=float)
    
    for k in range(k_range):
        for n in range (N):
            c[k] += samples[n]*np.exp(2j*np.pi*k*n/N)
            
        mag_c[k] = np.absolute(c[k])**2

    k_arr = range(k_range)        
    return (mag_c, k_arr)

mag_c, k = dft(data)
#Shift k array away from 0
k_shift = k[10:]
c_shift = mag_c[10:]
#Find the maximum k frequency in the range
peak_index = np.argmax(c_shift)

k_max = k_shift[peak_index]
print('The maximum occurs at a frequency: {}/month'.format(k_max))
N = len(data)
period = N/k_max
print('')
print('This corresponds to a period of {} months'.format(period))


axs[1].semilogy(k, mag_c)
axs[1].set_ylabel('$|c_k|^2$')
axs[1].set_xlabel('k')

plt.savefig('plots/sunny.pdf')
plt.show()


