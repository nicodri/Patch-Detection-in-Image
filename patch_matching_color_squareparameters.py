# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:00:42 2015

@author: drizardnicolas
"""

from scipy import misc
import numpy as np
import math
import time

def patch_matching_color_squareparameters(eye_source, img_source, zmin, zmax, dz):
    #Patch maching in color
    #    eye: reference patch
    #    img: image where you want to find the patch eye
    #    features model: (x,y,R1,G1,B1,R2,G2,B2,R3,G3,B3,R4,G4,B4)
    #    Arguments:
    #       zoom: no zoom for this model
    #       img_source: source for the image
    #       eye_source: source for the patch
    
    #--------------------------------------Loading Images
    
    #load images in fray-scale layer in RGB       
    img = misc.imread(img_source) #size (i1,j1,3) 3 for RGB composante
    eye=misc.imread(eye_source) #size (i0,j0,3)
    
    #Start the time
    start_time = time.time()
    
    #data parameters
    
    i0=eye.shape[0]
    j0=eye.shape[1]
    eye_param_size=(i0-1)*(j0-1)
    
    
    i1=img.shape[0]
    j1=img.shape[1]
    
    
    #Fixed Zoom parameters
    zmin=1
    zmax=1.1
    dz=0.2
    
    print "Pre-processing begins"
    #----------------------------------------Estimating the patch
          
          
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
    
    #-------------------------------------------Setting the summed area tables
    
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
    
    pre_processing_time=time.time() - start_time
    
    print "Pre-processing completed in "+str(pre_processing_time)+" s";
    
    
    print "Patch-Matching begins"    
    
    #setting the KL-table storing the least KL value for the patch with top-left corner (i,j) and zoom z=KL[i,j,1]
    KL=np.zeros((i1-i0+1,j1-j0+1,2))
    mu_temp=np.zeros((14,1))
    
    #Fixed values because there is is no zoom, ie size of the current patch fixed
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
                    KL[i,j,0]=KL_temp
                    KL[i,j,1]=z
                    
    patch_matching_time=time.time() - pre_processing_time - start_time
    
    print "Patch-Matching completed in "+str(patch_matching_time)+" s"
    
    return [KL, pre_processing_time, patch_matching_time]
