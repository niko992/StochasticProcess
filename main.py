import RWM as RWM
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt


def main():
    s2 = 0.5
    P = 10
    '''chain = RWM.RWM(s2, P)
    print(chain.shape)
    plt.plot(chain[0, :])
    plt.show()

    chain = RWM.RWM2(s2, P)
    print(chain.shape)
    plt.plot(chain[0, :])
    plt.show()'''

    chain = RWM.RWM3(s2, P)
    print(chain.shape)
    plt.plot(chain[0, :])
    plt.show()


if __name__ == "__main__":
    main()
