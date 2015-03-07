# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:00:42 2015

@author: drizardnicolas
"""


from scipy import misc
import numpy as np
import math
import time

from KLmin import KLmin
from dispresult import dispresult


#Patch maching in color
#    eye: reference patch
#    img: image where you want to find the patch eye
#    patchs_number: number of occurences of the patch eye in the image
#    zoom: define the size range of the patch analized in img, the value is z from zmin to zmax with increment dz
#    features model: (R,G,B,x,y)
#    img_result: name of the saved result

####################################################################################################################################################################################################################################################################################################################################################################################################################
#################TO DO:

#Parameters to chose before running the program
patchs_number=4
img_result='family_result.jpg'

zmin=0.8
zmax=1.2
dz=0.2

####################################################################################################################################################################################################################################################################################################################################################################################################################

#--------------------------------------Loading Images

#load images in fray-scale layer in RGB    
img = misc.imread('img/family_color_2.png') #size (i1,j1,3) 3 for RGB composante
eye=misc.imread('img/family_color_patch4.png') #size (i0,j0,3)

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
      
      
#Compute the mean of the patch
mu_eye=np.zeros((5,1))      
mu_eye[0:3,:]=np.array([[np.mean(eye[:,:,i])] for i in range(3)]) #size (3,1)
mu_eye[3,0]=(i0-1)/2.0
mu_eye[4,0]=(j0-1)/2.0

#Compute the covariance matrix of the patch eye
X=np.zeros((5,1))
cov_eye=np.zeros((5,5))
for i in range(i0):
    for j in range(j0):
        X[0:3,:]=(np.array([[eye[i,j,k]] for k in range(3)])).astype(np.float64)# type unint8 unable to store X**2
        X[3,0]=i        
        X[4,0]=j
        cov_eye+=np.dot(X,np.transpose(X))
cov_eye=(cov_eye-eye_size*np.dot(mu_eye,np.transpose(mu_eye)))/(eye_size-1.0)

#------------------------------------------Setting the summed area tables

#integral image to compute mu
I_sum=np.zeros((i1,j1,3))
for i in range(i1):
    for j in range(j1):
        I_sum[i,j,:]=img[i,j,:]+(I_sum[i-1,j,:] if i>0 else 0)+(I_sum[i,j-1,:] if j>0 else 0)-(I_sum[i-1,j-1,:] if i>0 and j>0 else 0)


#integral image to compute cov
I_square=np.zeros((i1,j1,5,5))
for i in range(i1):
    for j in range(j1):
        X[0:3,:]=(np.array([[img[i,j,k]] for k in range(3)])).astype(np.float64)# type unint8 unable to store X**2
        X[3,0]=i        
        X[4,0]=j
        I_square[i,j,:,:]=np.dot(X,np.transpose(X))+(I_square[i-1,j,:,:] if i>0 else 0)+(I_square[i,j-1,:,:] if j>0 else 0)-(I_square[i-1,j-1,:,:] if i>0 and j>0 else 0)

int_time1=time.time() - start_time

print "Pre-processing completed in "+str(int_time1)+" s";


print "Patch-Matching begins"    

#setting the KL-table storing the least KL value for the patch with top-left corner (i,j) and zoom z=KL[i,j,1]
KL=np.zeros((i1-i0+1,j1-j0+1,2))
mu_temp=np.zeros((5,1))

#Fixed values because there is is no zoom, ie size of the current patch fixed
mu_temp[3,0]=(i0-1)/2.0
mu_temp[4,0]=(j0-1)/2.0

for i in range(i1-i0 +1):
    print "Computing KL line "+str(i)
    for j in range(j1-j0 +1):
        for z in np.arange(zmin,zmax,dz):
            #patch analyzed: i to i+z*i0-1 and j to j+z*j0-1
            i_temp=int(z*i0)
            j_temp=int(z*j0)
            temp_size=i_temp*j_temp
            #check if the current patch is still inside the image img
            if i+i_temp>i1 or j+j_temp>j1:
                break
            
            #mu computation with the summed area table
            mu_temp[0:3,0]=I_sum[i+i_temp-1,j+j_temp-1,:]-(I_sum[i-1,j+j_temp-1,:] if i>0 else 0)-(I_sum[i+i_temp-1,j-1,:] if j>0 else 0)+(I_sum[i-1,j-1,:] if i>0 and j>0 else 0)
            mu_temp[0:3,0]=mu_temp[0:3,0]/(temp_size)
            
            #cov computation with the summed area table
            
            cov_temp=I_square[i+i_temp-1,j+j_temp-1,:,:]-(I_square[i-1,j+j_temp-1,:,:] if i>0 else 0)-(I_square[i+i_temp-1,j-1,:,:] if j>0 else 0)+(I_square[i-1,j-1,:,:] if i>0 and j>0 else 0)
            for r in range(3):            
                cov_temp[3,r]=cov_temp[3,r]-i*mu_temp[r,0]*eye_size
                cov_temp[r,3]=cov_temp[r,3]-i*mu_temp[r,0]*eye_size
                
                cov_temp[4,r]=cov_temp[4,r]-j*mu_temp[r,0]*eye_size
                cov_temp[r,4]=cov_temp[r,4]-j*mu_temp[r,0]*eye_size

            cov_temp[3,3]=cov_temp[3,3]-eye_size*i**2-2*eye_size*i*mu_temp[3,0]
            cov_temp[4,4]=cov_temp[4,4]-eye_size*j**2-2*eye_size*j*mu_temp[4,0]
            cov_temp[3,4]=mu_temp[3,0]*mu_temp[4,0]*i0*j0
            cov_temp[4,3]=mu_temp[3,0]*mu_temp[4,0]*i0*j0
            
            cov_temp=(cov_temp-temp_size*np.dot(mu_temp,np.transpose(mu_temp)))/(temp_size-1)

            #KL symetrized computation with 3 parameters
            KL_temp=abs(0.25*(np.trace(np.dot(np.linalg.pinv(cov_temp),cov_eye))+np.trace(np.dot(np.linalg.pinv(cov_eye),cov_temp))+np.sum(np.dot(np.dot(np.transpose(mu_temp-mu_eye),np.linalg.pinv(cov_temp)),(mu_temp-mu_eye)))+np.sum(np.dot(np.dot(np.transpose(mu_eye-mu_temp),np.linalg.pinv(cov_eye)),(mu_eye-mu_temp)))-10))#numerical value stands for 2*dimension_input, here 2*5
    
            #Bhattacharyya distance
            #sigma=0.5*(cov_temp+cov_eye)
            #KL_temp=abs(0.125*(np.dot(np.dot(np.transpose(mu_eye-mu_temp),np.linalg.pinv(sigma)),(mu_eye-mu_temp))+0.5*math.log(np.linalg.det(sigma)/(math.sqrt(np.linalg.det(cov_temp)*np.linalg.det(cov_eye))))))
            
            if KL[i,j,1]==0:
                KL[i,j,0]=KL_temp
                KL[i,j,1]=z
                
            elif KL_temp<KL[i,j,0]:
                KL[i,j,0]=KL_temp
                KL[i,j,1]=z
                
int_time2=time.time() - int_time1 - start_time

print "Patch-Matching completed in "+str(int_time2)+" s"

#-------------------------------Finding the top matchs_number patch(es) matching the eye patch in the img based on the KL table

#KL_min stores the top patchs_number patch maching the input eye in the image img

KL_min=KLmin(KL,i0,j0,i1,j1,patchs_number) 

#------------Displaying and saving the result

#aliasing img to conserve the original img
img_bis=np.zeros(img.shape)

for i in range(img.shape[0]):
    img_bis[i,:,:]=[row[:] for row in img[i,:,:]] 
img_bis=img_bis.astype(np.uint8) 

#Displaying the result and saving it with the name img_result
dispresult(img_result,img_bis,i0,j0,KL_min,patchs_number)

