# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:13:01 2015

@author: drizardnicolas
"""
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import math 

#load images in fray-scale layer in grey 
#    eye: reference patch
#    img: image where you want to find the patch eye
#    patchs_number: number of occurences of the patch eye in the image

img = misc.imread('test_famille.jpg',1)
eye=misc.imread('eye_famille.png',1)

#data parameters
i0=eye.shape[0]
j0=eye.shape[1]
eye_size=i0*j0


i1=img.shape[0]
j1=img.shape[1]

patchs_number=10

print "Pre-processing begins"
#----------------------------------------Estimate the patch
        
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


#----------------------------------set the summed area table

#sum to compute mu
I_sum=np.zeros((i1,j1))
for i in range(i1):
    for j in range(j1):
        I_sum[i,j]=img[i,j]+(I_sum[i-1,j] if i>0 else 0)+(I_sum[i,j-1] if j>0 else 0)-(I_sum[i-1,j-1] if i>0 and j>0 else 0)

#sum to compute cov
I_square=np.zeros((i1,j1,3,3))
for i in range(i1):
    for j in range(j1):
        X=np.array([[i],[j],[img[i,j]]]).astype(np.float64)# type unint8 unable to store X**2
        I_square[i,j,:,:]=np.dot(X,np.transpose(X))+(I_square[i-1,j,:,:] if i>0 else 0)+(I_square[i,j-1,:,:] if j>0 else 0)-(I_square[i-1,j-1,:,:] if i>0 and j>0 else 0)

print "Pre-processing completed"

#---------Patch Matching


print "Patch-Matching begins"    

#-----------------------------set the KL-table
KL=np.zeros((i1-i0+1,j1-j0+1))
mu_temp=np.zeros((3,1))
mu_temp[0,0]=i0/2
mu_temp[1,0]=j0/2
for i in range(i1-i0 +1):
    print "KL line "+str(i)
    for j in range(j1-j0 +1):
        #patch analyzed: i to i+i0-1 and j to j+j0-1
        #mu computation with the summed area table
        sum_int=I_sum[i+i0-1,j+j0-1]-(I_sum[i-1,j+j0-1] if i>0 else 0)-(I_sum[i+i0-1,j-1] if j>0 else 0)+(I_sum[i-1,j-1] if i>0 and j>0 else 0)
        mu_temp[2,0]=(sum_int)/(eye_size)#compute mu with the summed area table            
        
        #cov NAIVE computation: unable to do it with the summed area because we are using relative coordinates in the current patch
        cov_temp=np.zeros((3,3))        
        for k in range(i0):
            for l in range(j0):
                X=np.array([[k],[l],[img[i+k,j+l]]])
                cov_temp+=np.dot((X-mu_temp),np.transpose(X-mu_temp))
        cov_temp=cov_temp/(eye_size-1)
        
        
        #KL symetrized computation
        #KL[i,j]=0.25*(np.trace(np.dot(np.linalg.pinv(cov_temp),cov_eye))+np.trace(np.dot(np.linalg.pinv(cov_eye),cov_temp))+np.sum(np.dot(np.dot(np.transpose(mu_temp-mu_eye),np.linalg.pinv(cov_temp)),(mu_temp-mu_eye)))+np.sum(np.dot(np.dot(np.transpose(mu_eye-mu_temp),np.linalg.pinv(cov_eye)),(mu_eye-mu_temp)))-6)

        #Bhattacharyya distance
        sigma=0.5*(cov_temp+cov_eye)
        KL[i,j]=0.125*(np.dot(np.dot(np.transpose(mu_eye-mu_temp),np.linalg.pinv(sigma)),(mu_eye-mu_temp))+0.5*math.log(np.linalg.det(sigma)/(math.sqrt(np.linalg.det(cov_temp)*np.linalg.det(cov_eye)))))
        
print "Naive Patch-Matching completed"

#------------Find the top matchs_number patch matching the eye patch in the img based on the KL table
#check if the patch with left corner (i_test,j_test) is not to close of the patch with left corner (i_ref,j_ref) with the precision precision
def around(i_test,j_test,i_ref,j_ref,i_precision,j_precision):
    i_ret=False
    j_ret=False
    for k in range(i_precision):
        if i_test==i_ref+k or i_test==i_ref-k:
            i_ret=True
    for k in range(j_precision):
        if j_test==j_ref+k or j_test==j_ref-k:
            j_ret=True
    return i_ret*j_ret

#store the top patchs_number patch maching the input eye in the image img 
KL_min=np.zeros((3,patchs_number)) 
for  l in range(patchs_number):
    KL_min[2,l]=10000

    
for i in range(i1-i0+1):
    for j in range(j1-j0+1):
        for k in range(patchs_number):
            if KL[i,j]<KL_min[2,k]:#new top min founded,with ranking k among the top patchs_number
                #initialisation
                boolean_better=False
                boolean_worse=False
                m=0
                while (not boolean_better and m<k):# check if the new top min patch is not to close of a better patch
                    boolean_better=around(i,j,KL_min[0,m],KL_min[1,m],i0,j0)
                    m+=1   
                if not boolean_better: #current patch far enough from the other patchs
                    l=k                    
                    while (not boolean_worse and l<(patchs_number-1)):#check if the new top min patch we are storing is not to close of a worse patch. In this case we need to erase this worse patch
                        # NB: There is at most only one worse patch too close, because the patchs already stored are supposed to be far enough from each other                        
                        boolean_worse=around(i,j,KL_min[0,l],KL_min[1,l],i0,j0)
                        l+=1
                        
                    if boolean_worse:
                        #Debug zone
                        #print "Patch "+str((KL_min[0,l-1],KL_min[1,l-1]))+" too close to patch "+str((i,j))
                        for o in range(k,l-1,1):
                            KL_min[:,l-1+k-o]=KL_min[:,l-1+k-o-1]
                    else:
                        for o in range(patchs_number-k-1): #move the patch worse to the right of the K_min table
                            KL_min[:,patchs_number-o-1]=KL_min[:,patchs_number-o-1-1]
                    KL_min[0,k]=i
                    KL_min[1,k]=j
                    KL_min[2,k]=KL[i,j]
                    print str((i,j,KL_min))
                break

#Display the result
for k in range (patchs_number):
    i_min=KL_min[0,k]
    j_min=KL_min[1,k]
    print "Patch found on ("+str(i_min)+","+str(j_min)+") with the KL value "+str(KL_min[2,k])
    #Patch display
    for i in range(i0):
        img[i_min+i,j_min]=255
        img[i_min+i,j_min+j0-1]=255
    for j in range(j0):
        img[i_min,j_min+j]=255
        img[i_min+i0-1,j_min+j]=255

plt.figure(figsize=(6,6))
plt.axes([0, 0, 1, 1])
plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')

plt.show()