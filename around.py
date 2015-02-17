# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 09:55:47 2015

@author: drizardnicolas
"""

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