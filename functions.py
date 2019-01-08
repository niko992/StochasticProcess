# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:11:46 2018

@author: nikom
"""
import numpy as np
from scipy.integrate import trapz
import constants as cts
import pdb


# Functions
def u(xx, csi):
    """
    Args:
        x (np.ndarray(M, 1))
        csi (np.ndarray(P, 1))
    Returns:
        np.ndarrray(M,1)
    """
    M = len(xx)
    P = len(csi)
    sinx = np.zeros((M, P))
    for i in range(M):
        for k in range(P):
            sinx[i, k] = np.sin((k + 1) * np.pi * xx[i])
    return np.sqrt(2) / np.pi * (np.dot(sinx, csi))


def p(x, csi, M):
    """
    Args:
        x (float)
        csi (np.ndarray(P, 1))
        M (int)
    """
    x1 = np.linspace(0, 1, M)
    xx = np.linspace(0, x, M)
    y1 = np.exp(-u(x1, csi))
    yx = np.exp(-u(xx, csi))
    S1 = trapz(y1, x1)
    Sx = trapz(yx, xx)
    return 2 * Sx / S1


def G(csi, M):
    ret = np.zeros((4, 1))
    for i in range(4):
        ret[i] = p(0.2 * (i + 1), csi, M)
    return ret


def likelihood(csi, M):
    """
    Args:
        csi (np.ndarray(P, 1))
        M (int)
    """
    y = np.asarray([0.5041, 0.8505, 1.2257, 1.4113]).reshape((4, 1))
    return 1 / (np.sqrt(2 * np.pi) * cts.sigma) * np.exp(-np.linalg.norm(y - G(csi, M))**2 / (2 * cts.sigma**2))


def Pi0(csi):
    """Function returning the prior evaluated in csi
    Args:
        csi (np.array): point where the prior is evaluated
    """
    P = len(csi)
    csik2 = np.zeros((P, 1))
    tpik2 = np.zeros((P, 1))
    for k in range(P):
        csik2[k] = (k + 1)**2 * csi[k]**2
    for k in range(P):
        tpik2[k] = np.sqrt(2 * np.pi * (k + 1)**(-2))
    return 1 / np.prod(tpik2) * np.exp(-0.5 * np.sum(csik2))


def f(csi, M):
    """Function returning the non-normalized posterior evaluated in csi
    Args:
        csi (np.array): point where the non-normalized posterior is evaluated
        M (int): number of intervarls used in the numerical integration of the integral
    Returns:
        returns non-normalized posterior evaluated in the parameter csi
    """
    return likelihood(csi, M) * Pi0(csi)


def minus_log_posterior(csi, M):
    """Function returning -log(posterior) evaluated in the parameter csi
    Args:
        csi (np.array): point where the function -log(posterior) is evaluated
        M (int): number of iuntervarls used in the numerical integration of the integral
    Returns:
        returns -log(posterior) evaluated in the parameter csi
    """
    y = np.asarray([0.5041, 0.8505, 1.2257, 1.4113]).reshape((4, 1))
    log_likelihood = np.linalg.norm(y - G(csi, M))**2

    P = len(csi)
    csik2 = np.zeros((P, 1))
    for k in range(P):
        csik2[k] = (k + 1)**2 * csi[k]**2
    log_prior = 0.5 * np.sum(csik2)
    return log_likelihood + log_prior


def correlation(B, i, q):
    assert i < len(q)-B, "error in autocorrelation"
    x = q[B:]
    return np.corrcoef(np.asarray([x[:len(x)-i], x[i:]]))[0, 1]


def ESS(B, q, K):
    """Implementation of the ESS function
    Args:
        B (int): length of the burn-in
        q (np.array): array that contains the chain to analize
    """
    N = len(q)
    corr = np.zeros(N - B,)
    for i in range(K):
        corr[i] = correlation(B, i, q)
    return (N-B) / (1 + 2 * np.nansum(corr))
