import RWM as RWM
#import matplotlib
#matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import functions as fct
import constants as cts
from tqdm import tqdm
def intexp(csi):
    x=np.linspace(0,1,100)
    return sp.integrate.trapz(fct.u(x,csi),x)

def csi1(csi):
    return csi[0]

def csi2(csi):
    return csi[1]

def csi10(csi):
    if len(csi)<10:
        print("error length non sufficient")
        return
    return csi[9]


def main():
        s2=0.5
        ESS=np.zeros((4,15))
        for P in tqdm(range(10,25)):
            chain1=RWM.RWM(s2,P,csi1)
            ESS[0,P-10]=fct.ESS(cts.N,500,chain1)
            chain2=RWM.RWM(s2,P,csi2)
            ESS[1,P-10]=fct.ESS(cts.N,500,chain2)
            chain3=RWM.RWM(s2,P,csi10)
            ESS[2,P-10]=fct.ESS(cts.N,500,chain3)
            chain4=RWM.RWM(s2,P,intexp)
            ESS[3,P-10]=fct.ESS(cts.N,500,chain4)
        plt.plot(ESS[0,:])
        plt.plot(ESS[1,:])
        plt.plot(ESS[2,:])
        plt.plot(ESS[3,:])
        plt.show()

        ESS15=np.zeros(4)
        P = 15
        chain = RWM.RWM(s2, P, intexp)
        print(chain.shape)
        plt.plot(chain)
        plt.show()
        ESS15[0]=fct.ESS(cts.N,500,chain)

        chain = RWM.RWM2(s2, P, intexp)
        print(chain.shape)
        plt.plot(chain)
        plt.show()
        print(np.mean(chain))
        ESS15[1]=fct.ESS(cts.N,500,chain)

        chain = RWM.RWM3(s2, P, intexp)
        print(chain.shape)
        plt.plot(chain)
        plt.show()
        print(np.mean(chain))
        ESS15[2]=fct.ESS(cts.N,500,chain)

        chain = RWM.RWM4(s2, P, intexp)
        print(chain.shape)
        plt.plot(chain[0, :])
        plt.show()
        ESS15[3]=fct.ESS(cts.N,500,chain)

        plt.figure()
        plt.plot(ESS15)

        return

if __name__ == "__main__":
    main()
