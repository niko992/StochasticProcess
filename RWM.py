# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:43:44 2018

@author: nikom
"""

import numpy as np
import functions as fct
import constants as cts
import scipy as sp
import pdb
import warnings

# Covariance Matrices definition


def Cmat(P):
    C = np.zeros((P, P))
    for i in range(P):
        C[i, i] = (i + 1)**(-2)
    return C


# Random Walk Metropolis
def RWM(s2, P):
    M = np.ceil(P / 2)
    C = Cmat(P)
    zero = np.zeros((P,))
    Csi = np.zeros((P, cts.N))
    Csi[:, 0] = np.random.multivariate_normal(zero, C)

    for i in range(1, cts.N):
        eps = np.random.multivariate_normal(zero, s2 * C)
        Z = Csi[:, i - 1] + eps
        U = np.random.uniform(0, 1)
        a = np.min([fct.f(Z, M) / fct.f(Csi[:, i - 1], M), 1])
        if U < a:
            Csi[:, i] = Z
        else:
            Csi[:, i] = Csi[:, i - 1]
    return Csi


def RWM2(s2, P):
    M = np.ceil(P / 2)
    C = Cmat(P)
    zero = np.zeros((P,))
    Csi = np.zeros((P, cts.N))
    Csi[:, 0] = np.random.multivariate_normal(zero, C)
    for i in range(1, cts.N):
        eps = np.random.multivariate_normal(zero, s2 * C)
        Z = np.sqrt(1 - s2) * Csi[:, i - 1] + eps
        U = np.random.uniform(0, 1)
        if U < np.min([fct.f(Z, M) / fct.f(Csi[:, i - 1], M), 1]):
            Csi[:, i] = Z
        else:
            Csi[:, i] = Csi[:, i - 1]
    return Csi
# To define BFGS method


def RWM3(s2, P):
    M = np.ceil(P / 2)
    C = Cmat(P)
    zero = np.zeros((P,))
    Csi = np.zeros((P, cts.N))
    Csi[:, 0] = np.random.normal(zero, C)
    csi_ref = sp.optimize.minimize(-fct.f, Csi[:, 0], method='BFGS')
    for i in range(1, cts.N):
        eps = np.random.normal(zero, H)
        (fct.f, Csi[:, i - 1])
        Z = csi_ref + eps
        U = np.random.uniform(0, 1)
        if U < np.min(fct.f(Z, M) / fct.f(Csi[:, i - 1], M), 1):
            Csi[i] = Z
        else:
            Csi[:, i] = Csi[:, i - 1]
    return Csi
