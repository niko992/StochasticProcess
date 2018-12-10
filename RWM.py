# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:43:44 2018

@author: nikom
"""

import numpy as np
import functions as fc
import constants as cts

#Covariance Matrices definition
def Cmat(P):
    C=np.zeros(P)
    for i in range(0,P):
        C[i,i]=(i+1)**(-2)
    return C


#Random Walk Metropolis
def RWM(s2,P):
    M=np.ceil(P/2)
    C=Cmat(P)
    zero=np.zeros((P,))
    Csi=np.zeros((P,cts.N))
    Csi[:,0]=np.random.normal(zero,C)
    for i in range(1,cts.N):
        eps=np.random.normal(zero,s2*C)
        Z=Csi[:,i-1]+eps
        U=np.random.uniform(0,1)
        if U<np.min(fc.f(Z,M)/fc.f(Csi[:,i-1],M),1):
            Csi[i]=Z
        else:
            Csi[:,i]=Csi[:,i-1]
    return Csi            
        
        
        
    