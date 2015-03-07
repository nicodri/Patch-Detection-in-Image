# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:12:06 2015

@author: drizardnicolas
"""
import numpy as np
from around import around

#Method to compute the KL_min table storing the #patchs_number# patchs closest to the target patch in the image
#Arguments:     KL table with the divergence value between the target patch and each patch from the image`
#               (i0,j0) dimension of the target patch
#               (i1,j1) dimension of the image
#               patchs_number: number of patch to find in the image


def KLmin(KL,i0,j0,i1,j1,patchs_number):
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
    return KL_min;
