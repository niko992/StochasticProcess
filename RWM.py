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
import numdifftools as nd
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


def RWM3(s2, P):
    M = np.ceil(P / 2)
    C = Cmat(P)
    zero = np.zeros((P,))
    Csi = np.zeros((P, cts.N))
    alpha = 0.000001
    ID = np.diag(np.ones(P))
    dictionary = sp.optimize.minimize(
        fct.minus_log_posterior,  np.random.multivariate_normal(zero, C), args=(M), method='BFGS')
    pdb.set_trace()
    csi_map = dictionary['x']
    Csi[:, 0] = csi_map
    print(csi_map)
    H = dictionary['hess_inv'] + alpha*ID
    for i in range(1, cts.N):
        eps = np.random.multivariate_normal(zero, H)
        Z = csi_map + eps
        U = np.random.uniform(0, 1)
        if U < np.min([fct.f(Z, M) / fct.f(Csi[:, i - 1], M), 1]):
            Csi[:, i] = Z
        else:
            Csi[:, i] = Csi[:, i - 1]
    return Csi


def RWM4(s2, P):
    M = np.ceil(P / 2)
    C = Cmat(P)
    IP = np.diag(np.ones(P))
    Is2 = IP*s2
    zero = np.zeros((P,))
    Csi = np.zeros((P, cts.N))

    res = sp.optimize.minimize(fct.minus_log_posterior, np.random.multivariate_normal(zero, C), args=(M), method='BFGS')
    csi_map = res['x']

    Csi[:, 0] = csi_map

    def G(x):
        return fct.G(x, M)

    C_inv = np.diag([(k+1)**2 for k in range(P)])
    sqrt_C = np.diag([1/(k+1) for k in range(P)])

    gradG = nd.Jacobian(G)
    gamma = cts.sigma**2 * np.dot(gradG(csi_map).T, gradG(csi_map))
    C_gamma = np.linalg.inv(C_inv + gamma)

    H_gamma = np.dot(C, np.dot(gamma, C))
    A_gamma = np.dot(sqrt_C, np.dot(sp.linalg.sqrtm(IP-Is2+np.linalg.inv(IP+H_gamma)), sqrt_C))
    for i in range(1, cts.N):
        eps = np.random.multivariate_normal(zero, s2*C_gamma)
        Z = np.dot(A_gamma, Csi[:, i - 1]) + eps
        U = np.random.uniform(0, 1)
        if U < np.min([fct.f(Z, M) / fct.f(Csi[:, i - 1], M), 1]):
            Csi[:, i] = Z
        else:
            Csi[:, i] = Csi[:, i - 1]
    return Csi
