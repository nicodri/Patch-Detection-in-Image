# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:00:42 2015

@author: drizardnicolas
"""


from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from around import around
import math
import time

#load images in fray-scale layer in RGB
#    eye: reference patch
#    img: image where you want to find the patch eye
#    patchs_number: number of occurences of the patch eye in the image
#    zoom: define the size range of the patch analized in img
    
img = misc.imread('img/family_color.jpg') #size (i1,j1,3) 3 for RGB composante
eye=misc.imread('img/family_color_patch.png') #size (i0,j0,3)

#Start the time
start_time = time.time()

#data parameters

i0=eye.shape[0]
j0=eye.shape[1]
eye_size=i0*j0
eye_param_size=(i0-1)*(j0-1)


i1=img.shape[0]
j1=img.shape[1]

patchs_number=9

#Zoom parameters
zmin=1
zmax=1.1
dz=0.2

print "Pre-processing begins"
#----------------------------------------Estimate the patch
      
      
#Compute the mean of the patch
mu_eye=np.zeros((14,1)) # (x,y,RGB1,RGB2,RGB3,RGB4)     
mu_eye[2:5,:]=np.array([[np.mean(eye[0:i0-1,0:j0-1,i])] for i in range(3)])#up left corner 
mu_eye[5:8,:]=np.array([[np.mean(eye[0:i0-1,1:j0,i])] for i in range(3)])#up right corner 
mu_eye[8:11,:]=np.array([[np.mean(eye[1:i0,0:j0-1,i])] for i in range(3)])#down left corner 
mu_eye[11:14,:]=np.array([[np.mean(eye[1:i0,1:j0,i])] for i in range(3)])#down right corner 

mu_eye[0,0]=i0/2.0 #one input left in comparison to the 5 parameters model
mu_eye[1,0]=j0/2.0

#Compute the covariance matrix of the patch eye
X=np.zeros((14,1))
cov_eye=np.zeros((14,14))
for i in range(1, i0):
    for j in range(1,j0):
        X[2:5,:]=(np.array([[eye[i-1,j-1,k]] for k in range(3)])).astype(np.float64)# type unint8 unable to store X**2 up left corner
        X[5:8,:]=(np.array([[eye[i-1,j,k]] for k in range(3)])).astype(np.float64)# up right corner 
        X[8:11,:]=(np.array([[eye[i,j-1,k]] for k in range(3)])).astype(np.float64)# down left corner
        X[11:14,:]=(np.array([[eye[i,j,k]] for k in range(3)])).astype(np.float64)# down right corner 
        X[0,0]=i        
        X[1,0]=j
        cov_eye+=np.dot(X,np.transpose(X))
cov_eye=(cov_eye-eye_param_size*np.dot(mu_eye,np.transpose(mu_eye)))/(eye_param_size-1.0)

#----------------------------------set the summed area table

#integral image to compute mu
I_sum=np.zeros((i1,j1,12))
for i in range(1,i1):
    for j in range(1,j1):
        I_sum[i,j,0:3]=img[i-1,j-1,:]+(I_sum[i-1,j,0:3] if i>0 else 0)+(I_sum[i,j-1,0:3] if j>0 else 0)-(I_sum[i-1,j-1,0:3] if i>0 and j>0 else 0)#up left corner
        I_sum[i,j,3:6]=img[i-1,j,:]+(I_sum[i-1,j,3:6] if i>0 else 0)+(I_sum[i,j-1,3:6] if j>0 else 0)-(I_sum[i-1,j-1,3:6] if i>0 and j>0 else 0)#up right corner
        I_sum[i,j,6:9]=img[i,j-1,:]+(I_sum[i-1,j,6:9] if i>0 else 0)+(I_sum[i,j-1,6:9] if j>0 else 0)-(I_sum[i-1,j-1,6:9] if i>0 and j>0 else 0)#down left corner
        I_sum[i,j,9:12]=img[i,j,:]+(I_sum[i-1,j,9:12] if i>0 else 0)+(I_sum[i,j-1,9:12] if j>0 else 0)-(I_sum[i-1,j-1,9:12] if i>0 and j>0 else 0)#down right corner


#integral image to compute cov
I_square=np.zeros((i1,j1,14,14))
for i in range(1,i1):
    for j in range(1,j1):
        X[2:5,:]=(np.array([[img[i-1,j-1,k]] for k in range(3)])).astype(np.float64)# type unint8 unable to store X**2 up left corner
        X[5:8,:]=(np.array([[img[i-1,j,k]] for k in range(3)])).astype(np.float64)# up right corner 
        X[8:11,:]=(np.array([[img[i,j-1,k]] for k in range(3)])).astype(np.float64)# down left corner
        X[11:14,:]=(np.array([[img[i,j,k]] for k in range(3)])).astype(np.float64)# down right corner 
        X[0,0]=i        
        X[1,0]=j
        I_square[i,j,:,:]=np.dot(X,np.transpose(X))+(I_square[i-1,j,:,:] if i>0 else 0)+(I_square[i,j-1,:,:] if j>0 else 0)-(I_square[i-1,j-1,:,:] if i>0 and j>0 else 0)

int_time1=time.time() - start_time

print "Pre-processing completed in "+str(int_time1)+" s";

#---------Patch Matching


print "Patch-Matching begins"    

#-----------------------------set the KL-table storing the least KL value for the patch with top-left corner (i,j) and zoom z=KL[i,j,1]
KL=np.zeros((i1-i0+1,j1-j0+1,2))
mu_temp=np.zeros((14,1))

#only without zoom
mu_temp[0,0]=i0/2.0
mu_temp[1,0]=j0/2.0

for i in range(i1-i0 +1):
    #print "Computing KL line "+str(i)
    for j in range(j1-j0 +1):
        for z in np.arange(zmin,zmax,dz):
            #patch analyzed: i to i+z*i0-1 and j to j+z*j0-1
            i_temp=int(z*i0)
            j_temp=int(z*j0)
            temp_size=(i_temp-1)*(j_temp-1)
            #check if the current patch is still inside the image img
            if i+i_temp>i1 or j+j_temp>j1:
                break
            
            #mu computation with the summed area table
            mu_temp[2:14,0]=I_sum[i+i_temp-1,j+j_temp-1,:]-(I_sum[i,j+j_temp-1,:] if i>0 else 0)-(I_sum[i+i_temp-1,j,:] if j>0 else 0)+(I_sum[i,j,:] if i>0 and j>0 else 0)
            mu_temp[2:14,0]=mu_temp[2:14,0]/(temp_size)
            
            #cov computation with the summed area table
            
            cov_temp=I_square[i+i_temp-1,j+j_temp-1,:,:]-(I_square[i,j+j_temp-1,:,:] if i>0 else 0)-(I_square[i+i_temp-1,j,:,:] if j>0 else 0)+(I_square[i,j,:,:] if i>0 and j>0 else 0)
            for r in range(2,14):            
                cov_temp[0,r]=cov_temp[0,r]-i*mu_temp[r,0]*eye_param_size
                cov_temp[r,0]=cov_temp[r,0]-i*mu_temp[r,0]*eye_param_size
                
                cov_temp[1,r]=cov_temp[1,r]-j*mu_temp[r,0]*eye_param_size
                cov_temp[r,1]=cov_temp[r,1]-j*mu_temp[r,0]*eye_param_size
            
            cov_temp=(cov_temp-temp_size*np.dot(mu_temp,np.transpose(mu_temp)))/(temp_size-1)
            
            cov_temp[0,0]=cov_eye[0,0]
            cov_temp[1,1]=cov_eye[1,1]
            cov_temp[0,1]=0
            cov_temp[1,0]=0
         
            #KL symetrized computation with 3 parameters
            KL_temp=abs(0.25*(np.trace(np.dot(np.linalg.pinv(cov_temp),cov_eye))+np.trace(np.dot(np.linalg.pinv(cov_eye),cov_temp))+np.sum(np.dot(np.dot(np.transpose(mu_temp-mu_eye),np.linalg.pinv(cov_temp)),(mu_temp-mu_eye)))+np.sum(np.dot(np.dot(np.transpose(mu_eye-mu_temp),np.linalg.pinv(cov_eye)),(mu_eye-mu_temp)))-28))#numerical value stands for 2*dimension_input, here 2*14
    
            #Bhattacharyya distance
            #sigma=0.5*(cov_temp+cov_eye)
            #KL_temp=abs(0.125*(np.dot(np.dot(np.transpose(mu_eye-mu_temp),np.linalg.pinv(sigma)),(mu_eye-mu_temp))+0.5*math.log(np.linalg.det(sigma)/(math.sqrt(np.linalg.det(cov_temp)*np.linalg.det(cov_eye))))))
            
            if KL[i,j,1]==0:
                KL[i,j,0]=KL_temp
                KL[i,j,1]=z
                
            elif KL_temp<KL[i,j,0]:
                #print "Change for zoom "+str(z)+" with KL value "+str(KL_temp)
                KL[i,j,0]=KL_temp
                KL[i,j,1]=z
                
int_time2=time.time() - int_time1 - start_time

print "Patch-Matching completed in "+str(int_time2)+" s"

#------------Find the top matchs_number patch matching the eye patch in the img based on the KL table

#store the top patchs_number patch maching the input eye in the image img 
KL_min=np.zeros((4,patchs_number)) 
for  l in range(patchs_number):
    KL_min[2,l]=10000

    
for i in range(i1-i0+1):
    for j in range(j1-j0+1):
        for k in range(patchs_number):
            if KL[i,j,0]<KL_min[2,k]:#new top min founded,with ranking k among the top patchs_number
                #initialisation
                boolean_better=False
                boolean_worse=False
                m=0
                while (not boolean_better and m<k):# check if the new top min patch is not to close of a better patch
                    #define the precision
                    i_precision=2*int(max(KL[i,j,1],KL_min[3,m])*i0)  
                    j_precision=2*int(max(KL[i,j,1],KL_min[3,m])*j0)
                    boolean_better=around(i,j,KL_min[0,m],KL_min[1,m],i_precision,j_precision)
                    m+=1   
                if not boolean_better: #current patch far enough from the other patchs
                    l=k                    
                    while (not boolean_worse and l<(patchs_number-1)):#check if the new top min patch we are storing is not to close of a worse patch. In this case we need to erase this worse patch
                        # NB: There is at most only one worse patch too close, because the patchs already stored are supposed to be far enough from each other                        
                        i_precision=2*int(max(KL[i,j,1],KL_min[3,l])*i0)  
                        j_precision=2*int(max(KL[i,j,1],KL_min[3,l])*j0)                        
                        boolean_worse=around(i,j,KL_min[0,l],KL_min[1,l],int(KL[i,j,1]*i0),int(KL[i,j,1]*j0))
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
                    KL_min[2,k]=KL[i,j,0]
                    KL_min[3,k]=KL[i,j,1]#store the zoom coefficient
                break

#Display the result
for k in range (patchs_number):
    i_min=KL_min[0,k]
    j_min=KL_min[1,k]
    z=KL_min[3,k]
    i_end=int(z*i0)
    j_end=int(z*j0)
    print "Patch found on ("+str(i_min)+","+str(j_min)+") with zoom "+str(z)+" and with the KL value "+str(KL_min[2,k])
    #Patch display
    for i in range(i_end):
        img[i_min+i,j_min,:]=np.array([255,255,255])
        img[i_min+i,j_min+j_end-1]=np.array([255,255,255])
    for j in range(j_end):
        img[i_min,j_min+j]=np.array([255,255,255])
        img[i_min+i_end-1,j_min+j]=np.array([255,255,255])

plt.figure(figsize=(6,6))
plt.axes([0, 0, 1, 1])
plt.imshow(img)
plt.axis('off')

plt.show()

misc.imsave('result3.jpg', img)

