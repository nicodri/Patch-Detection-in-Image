# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:13:01 2015

@author: drizardnicolas
"""
from scipy import misc
import numpy as np

#load images in fray-scale layer
img = misc.imread('eye.png',1)
eye=misc.imread('eye.png',1)

i0=eye.shape[0]
j0=eye.shape[1]
eye_size=i0*j0

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
mu_eye=np.zeros((3,1))
mu_eye[0,0]=eye.shape[0]/2
mu_eye[1,0]=eye.shape[1]/2
mu_eye[2,0]=np.mean(eye)

#covariance
cov_eye=np.zeros((3,3))        
for i in range(i0):
    for j in range(j0):
        X=np.array([[i],[j],[eye[i,j]]])
        cov_eye+=np.dot((X-mu_eye),np.transpose(X-mu_eye))
cov_eye=cov_eye/(eye_size-1)
cov_eye2=np.cov(input)


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

print "Pre-processing completed"

#---------Patch Matching

#Naive approach

print "Naive Patch-Matching begins"    

#mu_tot=np.zeros((img.shape[0]-i0,img.shape[1]-j0,3))
#cov_tot=np.zeros((img.shape[0]-i0,img.shape[1]-j0,3,3))

KL=np.zeros((img.shape[0]-i0+1,img.shape[1]-j0+1))
for i in range(img.shape[0]-i0):
    print "KL line "+str(i)
    for j in range(img.shape[1]-j0):
        #patch analyzed: i to i+i0 and j to j+j0
        #mu computation
        mu_temp=np.zeros((3,1))
        mu_temp[0,0]=i+i0/2
        mu_temp[1,0]=j+j0/2
        mu_temp[2,0]=(I_sum[i+i0,j+j0]-I_sum[i,j+j0]-I_sum[i+i0,j]+I_sum[i,j])/(eye_size-1)#compute mu with the summed area table
        #cov NAIVE computation
        cov_temp=np.zeros((3,3))        
        for k in range(i0):
            for l in range(j0):
                X=np.array([[i+k],[j+l],[img[i+k,j+l]]])
                cov_temp+=np.dot((X-mu_temp),np.transpose(X-mu_temp))
        cov_temp=cov_temp/(eye_size-1)
        
        #KL symetrized computation
        
        KL[i,j]=0.25*(np.trace(np.dot(np.linalg.pinv(cov_temp),cov_eye))+np.trace(np.dot(np.linalg.pinv(cov_eye),cov_temp))+np.sum(np.dot(np.dot(np.transpose(mu_temp-mu_eye),np.linalg.pinv(cov_temp)),(mu_temp-mu_eye)))+np.sum(np.dot(np.dot(np.transpose(mu_eye-mu_temp),np.linalg.pinv(cov_eye)),(mu_eye-mu_temp)))-6)
        
        
for i in range(KL.shape[0]):
    for j in range(KL.shape[1]):
        if KL[i,j]<1:
            print(KL[i,j],i,j)