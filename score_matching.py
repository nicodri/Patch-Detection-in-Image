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
from mpl_toolkits.mplot3d import Axes3D


##==================================================================================================
#---------------------------Law Definition
# Choose the "true" parameters.
theta_true=[-30,90,-150]

#pdf non-normalized for the data
def pdfexp(z):
    return exp(theta_true[0]*z+theta_true[1]*(z**2)+theta_true[2]*(z**3))

#pdf model
def pdfexpm(z,theta):
    return exp(theta[0]*z+theta[1]*z**2+theta[2]*z**3)
   

#normal law
a_true=[9,5]
def pdfn(z):
    return exp((-1/2)*((z-a_true[0])/a_true[1])**2)

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
    
#3D plot
def objtest_plt(a,b):
    L=0
    for i in range(len(vec)):
       L=L-(1/b)**2+0.5*((a-vec[i])/(b**2))**2 
    return L/len(vec)
    
a_0 = linspace(0, 15, 100)
a_1 = linspace(0, 100, 100)
X,Y = meshgrid(a_0,a_1)

Z=objtest_plt(a_0,a_1).T

#test1
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(1, 10, 0.25)
Y = np.arange(1, 10, 0.25)
X, Y = np.meshgrid(X, Y)
Z = objtest_plt(X,Y)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


#score matching  
#res = minimize_scalar(objtest)
#print "The score matching estimator provide a = "+str(res.x)+" for the true parameter a_true = "+str(a_true)
#
##exp family
#t=np.arange(-200,1000,0.5)
#plot(t,obj(t))
#
#res = minimize_scalar(obj)
#print "The score matching estimator provide theta = "+str(res.x)+" for the true parameter theta_true = "+str(theta_true[1])

#SM for normal law xith 2 parameters
a0=[1,3]
res = minimize(objtest, a0,method='BFGS',options={'disp': True})
print(res.x)

#def J(theta):
#    N=len(vec)
#    J=0
#    for k in range(N):
#        J+=2*theta[1]+1/2*(theta[0]+2*theta[1]*vec[k])**2
#    
#theta0 = np.array([1,0])
#par_sm=sciopt.fmin(J,theta0,disp=1,retall=1,xtol=0.5,ftol=0.5)
    

    
    
    
