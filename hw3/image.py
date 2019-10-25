#! /usr/bin/env python

#Prof Tinker
#Image Deconvolution

import numpy as np 
import matplotlib.pyplot as plt 
import scipy

blur_data = np.loadtxt('data/blur.txt')

plt.imshow(blur_data)
plt.savefig('plots/blurred_photo.pdf')
#Gaussian Width
sigma = 25


def gauss(x,y):
    return np.exp(-(x**2+y**2)/(2*sigma**2))

#Point spread array
ps_arr = np.zeros(blur_data.shape, float)
length, width = blur_data.shape

xs = np.linspace(0, length, length)
ys = np.linspace(0, width, width)


for y in range(length):
    for x in range(width):
        ps_arr[x][y] += gauss(x,y)
        ps_arr[x][-y] += gauss(x,y)
        ps_arr[-x][y] += gauss(x,y)
        ps_arr[-x][-y] += gauss(x,y)
        


plt.imshow(ps_arr, origin='lower')


bk = np.fft.rfft2(blur_data)
fk = np.fft.rfft2(ps_arr)

#print(bk.shape)
ak = np.zeros(bk.shape, complex)

#Ensure you don't divide by 1.e-3 terms in the transform
ak[fk>1e-3] = bk[fk > 1.e-3]/(length*width*fk[fk > 1.e-3])
ak[fk<=1e-3] = bk[fk<=1e-3]/(length*width)

a = np.fft.irfft2(ak)


plt.imshow(a)
plt.savefig('plots/unblurred.pdf')
plt.show()