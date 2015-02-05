# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 10:12:46 2015

@author: drizardnicolas
"""

from math import *
from matplotlib.pylab import *
import numpy as np
import scipy.misc as misc
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import scipy.optimize as sciopt
from sympy import symbols
x = symbols('x')

# Choose the "true" parameters.
theta_true=[30,90,150]

#pdf non-normalized for the data
def pdfexp(z):
    return exp(theta_true[0]*z+theta_true[1]*z**2+theta_true[2]*z**3)

#pdf model
def pdfexpm(z,theta):
    return exp(theta[0]*z+theta[1]*z**2+theta[2]*z**3)
   
a_true=9
#normal law
def pdfn(z):
    return exp((-1/2)*(z-a_true)**2)
    

#Metropolis unidimensional    
n = 1000000
alpha = 1
z = 0.
vec=[]
vec.append(z)
innov = uniform(-alpha,alpha,n) #random inovation, uniform proposal distribution
for i in xrange(1,n):
    can = z + innov[i] #candidate
    aprob = min([1.,pdfn(can)/pdfn(z)]) #acceptance probability
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


 
#--------------------obj for exp law
#objective function
def obj(theta):
    L=0
    for i in vec:
        L=2*theta_true[1]+6*theta*i+0.5*(theta_true[0]+2*theta_true[1]*i+3*theta*i**2)**2
    return L/len(vec) # objective function to minimize

# -------------------test obj function for normal law
def objtest(a):
    L=0
    for i in range(len(vec)):
       L=L+(-1)+0.5*(a-vec[i])**2 
    return L/len(vec)
#plot
t=np.arange(-100,1000,0.5)
plot(t,objtest(t))
  
#score matching  
res = minimize_scalar(objtest)
print "The score matching estimator provide a = "+str(res.x)+" for the true parameter a_true = "+str(a_true)

res = minimize(obj, theta0,method='BFGS',options={'disp': True})
#
print(res.x)

#def J(theta):
#    N=len(vec)
#    J=0
#    for k in range(N):
#        J+=2*theta[1]+1/2*(theta[0]+2*theta[1]*vec[k])**2
#    
#theta0 = np.array([1,0])
#par_sm=sciopt.fmin(J,theta0,disp=1,retall=1,xtol=0.5,ftol=0.5)
    

    
    
    
