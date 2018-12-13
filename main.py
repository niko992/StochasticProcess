import RWM as RWM
#import matplotlib
#matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import functions as fct
def intexp(csi):
    x=np.linspace(0,1,100)
    return sp.integrate.trapz(fct.u(x,csi),x)

def main():
#    s2 = 0.5
#    P = 10
#    chain = RWM.RWM(s2, P)
#    print(chain.shape)
#    plt.plot(chain[0, :])
#    plt.show()

    s2 = 0.5
    P=10
    chain = RWM.RWM2(s2, P, intexp)
    print(chain.shape)
    plt.plot(chain)
    plt.show()
    print(np.mean(chain))

#    chain = RWM.RWM4(s2, P)
#    print(chain.shape)
#    plt.plot(chain[0, :])
#    plt.show()


if __name__ == "__main__":
    main()
