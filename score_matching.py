# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 10:12:46 2015

@author: drizardnicolas
"""

from __future__ import division
from math import *
from matplotlib.pylab import *
import numpy as np
import scipy.misc as misc
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import scipy.optimize as sciopt
from sympy import symbols
x = symbols('x')
from mpl_toolkits.mplot3d import Axes3D


##==================================================================================================
#---------------------------Law Definition
# Choose the "true" parameters.
theta_true=[30,-90,150,+60]

#pdf non-normalized for the data
def pdfexp(z):
    return exp(theta_true[0]*z+theta_true[1]*(z**2)+theta_true[2]*(z**3)+theta_true[3]*(z**4))

#pdf model
def pdfexpm(z,theta):
    return exp(theta[0]*z+theta[1]*z**2+theta[2]*z**3)
   

#normal law
a_true=[9,2]
def pdfn(z):
    return exp((-0.5)*((z-a_true[0])/a_true[1])**2)

def pdfnm(z,a):
    return exp((-1/2)*((z-a[0])/a[1])**2)
    
##==================================================================================================
#--------------------------Metropolis unidimensional    
n = 1000000
alpha = 1
z = 0.
vec=[]
vec.append(z)
innov = uniform(-alpha,alpha,n) #random inovation, uniform proposal distribution
for i in xrange(1,n):
    can = z + innov[i] #candidate
    aprob = min([1.,pdfexp(can)/pdfexp(z)]) #acceptance probability
    u = uniform(0,1)
    if u < aprob:
        z = can
        vec.append(z)


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


 
#==============================================================================
# #--------------------obj for exp law
# #objective function
# def obj(theta):
#     L=0
#     for i in vec:
#         L+=2*theta+6*theta_true[2]*i+0.5*(theta_true[0]+2*theta*i+3*theta_true[2]*(i**2))**2
#     return L/len(vec) # objective function to minimize
#==============================================================================

 # -------------------test obj function for normal law
def objtest(a):
    L=0
    for i in range(len(vec)):
       L=L-(1/a[1])**2+0.5*((a[0]-vec[i])/(a[1]**2))**2 
    return L/len(vec)

def objtest_scalar(a):
    L=0
    for i in range(len(vec)):
       L=L-(1/a)**2+0.5*((mu_empirical-vec[i])/(a[0]**2))**2 
    return L/len(vec)



#Empirical mean and variance computation
mu_empirical=np.mean(vec)
var_empirical=np.var(vec)

    
#score matching  
#res = minimize_scalar(objtest)
#print "The score matching estimator provide a = "+str(res.x)+" for the true parameter a_true = "+str(a_true)
#
##exp family
t=np.arange(1,10,0.1)
plot(t,objtest_scalar(t))
#
res = minimize_scalar(objtest_scalar)
print "The score matching estimator provide theta = "+str(res.x)+" for the true parameter a = ",a_true[1]

#SM for normal law with 2 parameters
a0=[1]
res = minimize(objtest_scalar, a0,method='BFGS',options={'disp': True})
print(res.x)

    
    
    
