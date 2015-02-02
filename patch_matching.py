# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:13:01 2015

@author: drizardnicolas
"""
from scipy import misc
import numpy as np

#load images in fray-scale layer
img = misc.imread('test.png',1)
eye=misc.imread('eye.png',1)

#-------------------Estimate the patch
#set the list of the data input
input=np.zeros((3,eye.shape[0]*eye.shape[1]))
c=0
for i in range(eye.shape[0]):
    for j in range(eye.shape[1]):
        input[0,c]=i
        input[1,c]=j
        input[2,c]=eye[i,j]
        c+=1
        
#mean
mu_eye=[]
mu_eye.append(eye.shape[0]/2)
mu_eye.append(eye.shape[1]/2)
mu_eye.append(np.mean(eye))

#covariance
cov_eye=np.cov(input)


#--------set the summed area table
#sum to compute mu
I_sum=np.zeros((img.shape[0],img.shape[1],3))
I_sum[0,0,:]=[0,0,img[0,0]]
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        I_sum[]

