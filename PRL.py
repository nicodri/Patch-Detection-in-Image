# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:59:11 2015

@author: drizardnicolas
"""


from math import *
from matplotlib.pylab import *

# Choose the "true" parameters.
a_true = 1
b_true = 2
c_true = -4

def pdfexp(z):#pdf non-normalized
    return exp(a_true*z+b_true*z**2+c_true**3)

   
    #unidimensional    
n = 10000
alpha = 1
vec=[]
x = 0.
vec.append(x)
innov = uniform(-alpha,alpha,n) #random inovation, uniform proposal distribution
for i in xrange(1,n):
    can = x + innov[i] #candidate
    aprob = min([1.,pdfexp(can)/pdfexp(x)]) #acceptance probability
    u = uniform(0,1)
    if u < aprob:
        x = can
        vec.append(x)

#plotting the results:
#theoretical curve

subplot(211)
title('Metropolis-Hastings')
plot(vec)
subplot(212)

hist(vec, bins=30, normed=1)
ylabel('Frequency')
xlabel('x')
title('Samples')
show()
    
    
