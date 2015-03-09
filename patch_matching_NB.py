# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:13:01 2015

@author: drizardnicolas
"""
from scipy import misc
import numpy as np
import math
import time

def patch_matching_NB(eye_source, img_source, zmin, zmax, dz):
#Patch maching in color
#    eye: reference patch
#    img: image where you want to find the patch eye
#    features model: (R,G,B)
#    Arguments:
#       zoom: define the size of patches considered in the img, values from zmin to zmax incrementing with dz
#       img_source: source for the image
#       eye_source: source for the patch

img_source='img/B&W/family.jpg'
eye_source='img/B&W/family_patch.png'
#--------------------------------------Loading Images
    
#load images in fray-scale layer in RGB    
img = misc.imread(img_source,1) #size (i1,j1,1), axis 1 because black and white, only the intensity value
eye=misc.imread(eye_source,1) #size (i0,j0,1)

#Start the time
start_time = time.time()

#data parameters

i0=eye.shape[0]
j0=eye.shape[1]
eye_size=i0*j0


i1=img.shape[0]
j1=img.shape[1]


print "Pre-processing begins"
#----------------------------------------Estimating the patch
            
#mean
mu_eye=np.zeros((3,1))
mu_eye[0,0]=i0/2
mu_eye[1,0]=j0/2
mu_eye[2,0]=np.mean(eye)

#covariance
cov_eye=np.zeros((3,3))        
for i in range(i0):
    for j in range(j0):
        X=np.array([[i],[j],[eye[i,j]]])
        cov_eye+=np.dot((X-mu_eye),np.transpose(X-mu_eye))
cov_eye=cov_eye/(eye_size-1)


#----------------------------------Setting the summed area table

#integral image to compute mu
I_sum=np.zeros((i1,j1))
for i in range(i1):
    for j in range(j1):
        I_sum[i,j]=img[i,j]+(I_sum[i-1,j] if i>0 else 0)+(I_sum[i,j-1] if j>0 else 0)-(I_sum[i-1,j-1] if i>0 and j>0 else 0)

pre_processing_time=time.time() - start_time

print "Pre-processing completed in "+str(pre_processing_time)+" s";


print "Patch-Matching begins"   

#setting the KL-table storing the least KL value for the patch with top-left corner (i,j) and zoom z=KL[i,j,1]
KL=np.zeros((i1-i0+1,j1-j0+1,2)) #zoom case added for consistency with other patch_matching methods
mu_temp=np.zeros((3,1))

#Fixed values because there is is no zoom, ie size of the current patch fixed
mu_temp[0,0]=i0/2
mu_temp[1,0]=j0/2

for i in range(i1-i0 +1):
    #print "KL line "+str(i)
    for j in range(j1-j0 +1):
        #patch analyzed: i to i+i0-1 and j to j+j0-1
        #mu computation with the summed area table
        sum_int=I_sum[i+i0-1,j+j0-1]-(I_sum[i-1,j+j0-1] if i>0 else 0)-(I_sum[i+i0-1,j-1] if j>0 else 0)+(I_sum[i-1,j-1] if i>0 and j>0 else 0)
        mu_temp[2,0]=(sum_int)/(eye_size)#compute mu with the summed area table            
        
        #cov NAIVE computation:
        cov_temp=np.zeros((3,3))        
        for k in range(i0):
            for l in range(j0):
                X=np.array([[k],[l],[img[i+k,j+l]]])
                cov_temp+=np.dot((X-mu_temp),np.transpose(X-mu_temp))
        cov_temp=cov_temp/(eye_size-1)
        
        
        #KL symetrized computation
        KL[i,j,0]=0.25*(np.trace(np.dot(np.linalg.pinv(cov_temp),cov_eye))+np.trace(np.dot(np.linalg.pinv(cov_eye),cov_temp))+np.sum(np.dot(np.dot(np.transpose(mu_temp-mu_eye),np.linalg.pinv(cov_temp)),(mu_temp-mu_eye)))+np.sum(np.dot(np.dot(np.transpose(mu_eye-mu_temp),np.linalg.pinv(cov_eye)),(mu_eye-mu_temp)))-6)

        #Bhattacharyya distance
        #sigma=0.5*(cov_temp+cov_eye)
        #KL[i,j,0]=0.125*(np.dot(np.dot(np.transpose(mu_eye-mu_temp),np.linalg.pinv(sigma)),(mu_eye-mu_temp))+0.5*math.log(np.linalg.det(sigma)/(math.sqrt(np.linalg.det(cov_temp)*np.linalg.det(cov_eye)))))

        KL[i,j,1]=1 #constant zoom parameter
    break            
        
patch_matching_time=time.time() - pre_processing_time - start_time

print "Patch-Matching completed in "+str(patch_matching_time)+" s"

return [KL, pre_processing_time, patch_matching_time]
    