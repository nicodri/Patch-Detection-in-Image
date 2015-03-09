# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:12:52 2015

@author: drizardnicolas
"""
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

#Displaying the result and saving it with the name img_result
#Arguments:
#    patchs_number: number of occurences of the patch eye in the image
#    img_result: name of the saved result

def dispresult(img_result,img_bis,i0,j0,KL_min,patchs_number):
      
    for k in range (patchs_number):
        i_min=KL_min[0,k]
        j_min=KL_min[1,k]
        z=KL_min[3,k]
        i_end=int(z*i0)
        j_end=int(z*j0)
        print "Patch found on ("+str(i_min)+","+str(j_min)+") with zoom "+str(z)+" and with the KL value "+str(KL_min[2,k])
        #Patch display
        for i in range(i_end):
            img_bis[i_min+i,j_min,:]=np.array([255,255,255])
            img_bis[i_min+i,j_min+j_end-1]=np.array([255,255,255])
        for j in range(j_end):
            img_bis[i_min,j_min+j]=np.array([255,255,255])
            img_bis[i_min+i_end-1,j_min+j]=np.array([255,255,255])

    plt.figure(figsize=(6,6))
    plt.axes([0, 0, 1, 1])
    plt.imshow(img_bis)
    plt.axis('off')
    
    plt.show()
    
    misc.imsave(img_result, img_bis)