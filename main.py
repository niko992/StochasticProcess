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


def run(s2, P, K, burnout):

    ESS = np.zeros((4, ))
    CORR = np.zeros((P, K))
    print("Random walk Metropolis")

    main_chain = RWM.RWM(s2, P)

    for i in range(K):
        for j in range(P):
            CORR[j, i] = fct.correlation(burnout, i, main_chain[:, j])
    corr = np.max(CORR, axis=0)

    chain1 = np.asarray([csi1(csi) for csi in main_chain])
    chain2 = np.asarray([csi2(csi) for csi in main_chain])
    chain3 = np.asarray([csi10(csi) for csi in main_chain])
    chain4 = np.asarray([intexp(csi) for csi in main_chain])
    ESS[0] = fct.ESS(burnout, chain1)
    ESS[1] = fct.ESS(burnout, chain2)
    ESS[2] = fct.ESS(burnout, chain3)
    ESS[3] = fct.ESS(burnout, chain4)

    return corr, ESS, main_chain


def main():
    N = 5
    s2_max = 0.5
    s2_min = 1e-3
    s2_vec = np.linspace(s2_min, s2_max, N)
    P = 10
    K = 100
    burnout = 500

    ESS = np.zeros((4, N))
    corr = np.zeros((N, K))
    for i, s2 in enumerate(s2_vec):
        corr[i, :], ESS[:, i], main_chain = run(s2, P, K, burnout)
        fig1 = plt.figure(1)
        for j in range(P):
            plt.plot(main_chain[:, j])
        plt.plot()
        plt.title("Random Walk Metropolis\n"
                  "Acceptance ratio: {:{}f}, s2: {:{}f}, P: {:{}d}".format(acceptance_ratio(main_chain), .2, s2, .3, P, 1))
        fig1.savefig("RMW_s2_{}_P_{}".format(i, P))
        plt.close()

    fig1 = plt.figure(1)
    for j in range(N):
        plt.plot(corr[j, :], label="s2: {:{}f}".format(s2_vec[j], 0.3))
    plt.title("Autocorrelation")
    plt.legend()
    plt.ylim((0, 1))
    fig1.savefig("autocorr")
    return 0


if __name__ == "__main__":
    main()
