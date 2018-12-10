# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:11:46 2018

@author: nikom
"""
import numpy as np
import scipy as sp
import constants as cts


#Functions
def u(x,csi):
    sinx=np.zeros((len(x),))
    for k in range(0,len(x)):
        sinx[k]=np.sin(k*np.pi*x[k])
    return np.sqrt(2)/np.pi * np.sum(csi*sinx)

def p(x,csi,M):
    x1=np.linspace(0,1,M)
    xx=np.linspace(0,x,M)
    y1=np.exp(-u(x1,csi))
    yx=np.exp(-u(xx,csi))
    S1=sp.integrate.trapz(y1,x1)
    Sx=sp.integrate.trapz(yx,xx)
    return 2*Sx/S1

def G(x,M):
    ret=np.zeros(4,1)
    for i in range(0,4):
        ret[i]=p(0.2*(i+1),x,M)
    return ret

def likelihood(x,M):
    y=np.array([0.5041, 0.8505, 1.2257, 1.4113])
    return 1/(np.sqrt(2*np.pi)*cts.sigma)*np.exp(-np.linalg.norm(y-G(x,M))/(2*cts.sigma**2))

def Pi0(csi):
    P=len(csi)
    csik2=np.zeros(P,1)
    tpik2=np.zeros(P,1)
    for k in range(0,P):
        csik2[k]=(k+1)**2*csi[k]
    for k in range(0,P):
        tpik2[k]=np.sqrt(2*np.pi*(k+1)**(-2))
    return 1/np.prod(tpik2)*np.exp(-0.5*sum(csik2))

def f(csi,M):
    return likelihood(csi,M)*Pi0(csi)