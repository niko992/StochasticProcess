# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:43:44 2018

@author: nikom
"""

import numpy as np
import functions as fct
import constants as cts
import scipy as sp
import numdifftools as nd
# Covariance Matrices definition


def Cmat(P):
    return np.diag([(k + 1)**(-2) for k in range(P)])


# Random Walk Metropolis
def RWM(s2, P):
    M = np.ceil(P / 2)
    C = Cmat(P)
    zero = np.zeros((P,))
    # Csi = np.zeros((P, cts.N))
    Csi_old = np.random.multivariate_normal(zero, C)
    q = np.zeros((cts.N, P))
    q[0] = Csi_old
    for i in range(1, cts.N):
        eps = np.random.multivariate_normal(zero, s2 * C)
        Z = Csi_old + eps
        U = np.random.uniform(0, 1)
        a = np.min([fct.f(Z, M) / fct.f(Csi_old, M), 1])
        if U < a:
            Csi_old = Z
        q[i] = Csi_old
    return q


def pCN(s2, P):
    M = np.ceil(P / 2)
    C = Cmat(P)
    zero = np.zeros((P,))
    Csi_old = np.random.multivariate_normal(zero, C)
    q = np.zeros((cts.N, P))
    q[0] = Csi_old
    for i in range(1, cts.N):
        eps = np.random.multivariate_normal(zero, s2 * C)
        Z = np.sqrt(1 - s2) * Csi_old + eps
        U = np.random.uniform(0, 1)
        if U < np.min([fct.f(Z, M) / fct.f(Csi_old, M), 1]):
            Csi_old = Z
        q[i] = Csi_old
    return q


def laplace_approx(s2, P):
    M = np.ceil(P / 2)
    C = Cmat(P)
    zero = np.zeros((P,))
    q = np.zeros((cts.N, P))
    alpha = 1e-12
    ID = np.diag(np.ones(P))
    dictionary = sp.optimize.minimize(
        fct.minus_log_posterior,  np.random.multivariate_normal(zero, C), args=(M), method='BFGS')
    csi_map = dictionary['x']
    Csi_old = np.random.multivariate_normal(zero, C)
    q[0] = Csi_old
    H = dictionary['hess_inv'] + alpha * ID
    for i in range(1, cts.N):
        eps = np.random.multivariate_normal(zero, H)
        Z = csi_map + eps
        U = np.random.uniform(0, 1)
        if U < np.min([fct.f(Z, M) / fct.f(Csi_old, M), 1]):
            Csi_old = Z
        q[i] = Csi_old
    return q


def gen_pCN(s2, P):
    # import pdb
    # pdb.set_trace()
    M = np.ceil(P / 2)
    C = Cmat(P)
    IP = np.diag(np.ones(P))
    Is2 = IP * s2
    zero = np.zeros((P,))
    Q = np.zeros((cts.N, P))

    res = sp.optimize.minimize(fct.minus_log_posterior, np.random.multivariate_normal(
        zero, C), args=(M), method='BFGS')
    csi_map = res['x']

    Csi_old = np.random.multivariate_normal(zero, C)
    Q[0] = Csi_old

    def G(x):
        return fct.G(x, M)

    C_inv = np.diag([(k + 1)**2 for k in range(P)])
    sqrt_C = np.diag([1 / (k + 1) for k in range(P)])
    sqrt_C_inv = np.diag([(k + 1) for k in range(P)])
    gradG = nd.Jacobian(G)
    gamma = cts.sigma**(-2) * np.dot(gradG(csi_map).T, gradG(csi_map))
    C_gamma = np.linalg.inv(C_inv + gamma)

    H_gamma = np.dot(sqrt_C, np.dot(gamma, sqrt_C))
    A_gamma = np.dot(sqrt_C, np.dot(sp.linalg.sqrtm(
        IP - Is2 + np.linalg.inv(IP + H_gamma)), sqrt_C_inv))
    for i in range(1, cts.N):
        eps = np.random.multivariate_normal(zero, s2 * C_gamma)
        Z = np.dot(A_gamma, Csi_old) + eps
        U = np.random.uniform(0, 1)
        if U < np.min([fct.f(Z, M) / fct.f(Csi_old, M), 1]):
            Csi_old = Z
        Q[i] = Csi_old
    return Q
