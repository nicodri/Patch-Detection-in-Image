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
I_sum=np.zeros((img.shape[0],img.shape[1]))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        I_sum[i,j]=img[i,j]+(I_sum[i-1,j] if i>0 else 0)+(I_sum[i,j-1] if j>0 else 0)-(I_sum[i-1,j-1] if i>0 and j>0 else 0)

#sum to compute cov
I_cov=np.zeros((img.shape[0],img.shape[1],3,3))
temp=np.zeros((3,3))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        X=[[i,j,img[i,j]]]
        temp=np.transpose(X)*X
        I_cov[i,j,:,:]=temp+(I_cov[i-1,j,:,:] if i>0 else 0)+(I_cov[i,j-1,:,:] if j>0 else 0)-(I_cov[i-1,j-1,:,:] if i>0 and j>0 else 0)


#---------Patch Matching
#Naive approach
input_img=np.zeros((3,eye.shape[0]*eye.shape[1]))
m=0
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        input_img[0,m]=i
        input_img[1,m]=j
        input_img[2,m]=img[i,j]
        m+=1
        
i0=eye.shape[0]
i1=eye.shape[1]
for i in range(img.shape[0]-eye.shape[0]):
    for j in range(img.shape[1]-eye.shape[1]):
        mu_temp=