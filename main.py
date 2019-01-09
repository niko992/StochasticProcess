import RWM as RWM
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import functions as fct
import constants as cts


def intexp(csi):
    x = np.linspace(0, 1, 100)
    return sp.integrate.trapz(fct.u(x, csi), x)


def csi1(csi):
    return csi[0]


def csi2(csi):
    return csi[1]


def csi10(csi):
    if len(csi) < 10:
        print("error length non sufficient")
        return
    return csi[9]


def acceptance_ratio(main_chain):
    count = 0
    N = np.alen(main_chain)
    for a, b in zip(main_chain[:-1], main_chain[1:]):
        if (a == b).all():
            count += 1
    return (N-1-count)/(N-1)


def run(s2, P, K, burnin):

    ESS = np.zeros((4, ))
    CORR = np.zeros((P, K))
    print("gen_pCN")

    main_chain = RWM.gen_pCN(s2, P)

    for i in range(K):
        for j in range(P):
            CORR[j, i] = fct.correlation(burnin, i, main_chain[:, j])
    corr = np.max(CORR, axis=0)

    chain1 = np.asarray([csi1(csi) for csi in main_chain])
    chain2 = np.asarray([csi2(csi) for csi in main_chain])
    chain3 = np.asarray([csi10(csi) for csi in main_chain])
    chain4 = np.asarray([intexp(csi) for csi in main_chain])
    ESS[0] = fct.ESS(burnin, chain1, K)
    ESS[1] = fct.ESS(burnin, chain2, K)
    ESS[2] = fct.ESS(burnin, chain3, K)
    ESS[3] = fct.ESS(burnin, chain4, K)

    return corr, ESS, main_chain


def main_s2():
    N = 10
    s2_max = 0.1
    s2_min = 0.2
    s2_vec = np.linspace(s2_min, s2_max, N)
    P = 10
    K = 100
    burnin = 500

    ESS = np.zeros((4, N))
    corr = np.zeros((N, K))
    for i, s2 in enumerate(s2_vec):
        corr[i, :], ESS[:, i], main_chain = run(s2, P, K, burnin)
        fig1 = plt.figure(1)
        for j in range(P):
            plt.plot(main_chain[:, j])
        plt.plot()
        plt.title("Random Walk Metropolis\n"
                  "Acceptance ratio: {:{}f}, s2: {:{}f}, P: {:{}d}".format(acceptance_ratio(main_chain), .2, s2, .3, P, 1))
        fig1.savefig("RMW_s2_{}_P_{}".format(i, P), format="eps")
        plt.close()

    fig1 = plt.figure(1)
    for j in range(N):
        plt.plot(corr[j, :], label="s2: {:{}f}".format(s2_vec[j], 0.3))
    plt.title("Autocorrelation")
    plt.legend()
    plt.ylim((0, 1))
    fig1.savefig("autocorr", format="eps")
    return 0


def main_ex1():
    s2 = 0.78
    P_min = 10
    P_max = 50
    P_vec = np.asarray(np.linspace(P_min, P_max, 5), dtype='int')
    N = len(P_vec)
    K = 200
    burnin = 500

    ESS = np.zeros((4, N))
    corr = np.zeros((N, K))
    for i, P in enumerate(P_vec):
        corr[i, :], ESS[:, i], main_chain = run(s2, P, K, burnin)
        fig1 = plt.figure(1)
        for j in range(P):
            plt.plot(main_chain[:, j])
        plt.plot()
        plt.title("Acceptance ratio: {:{}f}, s2: {:{}f}, P: {:{}d}".format(acceptance_ratio(main_chain), .2, s2, .3, P, 1))
        fig1.savefig("RMW_s2_{}_P_{}.eps".format(i, P), format="eps")
        plt.close()

    fig1 = plt.figure(1)
    for j in range(N):
        plt.plot(corr[j, :], label="P: {:{}d}".format(P_vec[j], 1))
    plt.title("Autocorrelation")
    plt.legend()
    plt.ylim((0, 1))
    fig1.savefig("autocorr.eps", format="eps")

    plt.close()
    fig1 = plt.figure(1)
    for j in range(4):
        plt.plot(P_vec, ESS[j, :], label="f{}".format(j, 1))
    plt.title("ESS")
    plt.legend()
    fig1.savefig("ESS.eps", format="eps")
    return 0


if __name__ == "__main__":
    main_ex1()
