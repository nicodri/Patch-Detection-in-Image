# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:12:43 2015

@author: drizardnicolas
"""
import numpy as np
from scipy import misc

from KLmin import KLmin
from dispresult import dispresult
from patch_matching_color import patch_matching_color
#from patch_matching_NB import patch_matching_NB
from patch_matching_color_5parameters import patch_matching_color_5parameters
from patch_matching_color_squareparameters import patch_matching_color_squareparameters

####################################################################################################################################################################################################################################################################################################################################################################################################################
#################TO DO:

#Parameters to chose before running the program
patchs_number=12

img_result='group2_result_face_result_face.jpg'
img_source='img/group.jpg'
eye_source='img/family_color_4_patch_face.png'

#Zoom parameters
zmin=0.8
zmax=1.2
dz=0.2

####################################################################################################################################################################################################################################################################################################################################################################################################################

#load images in fray-scale layer only for the size 
img = misc.imread(img_source) 
eye=misc.imread(eye_source)

#data parameters

i0=eye.shape[0]
j0=eye.shape[1]
eye_size=i0*j0


i1=img.shape[0]
j1=img.shape[1]

#-------------------------------Patch matching method
#Method to chose among the following methods:
#   patch_matching_NB
#   patch_matching_color
#   patch_maching_color_5parameters
#   patch_matching_color_squareparameters

[KL, pre_processing_time, patch_matching_time]=patch_matching_color(eye_source, img_source, zmin, zmax, dz)


#-------------------------------Finding the top matchs_number patch(es) matching the eye patch in the img based on the KL table

#KL_min stores the top patchs_number patch maching the input eye in the image img

KL_min=KLmin(KL,i0,j0,i1,j1,patchs_number) 

#------------------------------Displaying and saving the result

#aliasing img to conserve the original img
img_bis=np.zeros(img.shape)

for i in range(img.shape[0]):
    img_bis[i,:,:]=[row[:] for row in img[i,:,:]] 
img_bis=img_bis.astype(np.uint8) 

#Displaying the result and saving it with the name img_result
dispresult(img_result,img_bis,i0,j0,KL_min,patchs_number)
