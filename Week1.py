import numpy as np
import Utilities
import matplotlib.pyplot as plt

def opgave1():
    print(Utilities.get_group_delay(1*10**-2, 1.45332, 0.8, -0.0173))
    print(Utilities.get_group_delay(1 * 10 ** -2, 1.45601, 0.67, -0.0252))
    print(Utilities.get_group_delay(1 * 10 ** -2, 1.47012, 0.4, -0.1091))

def opgave2():
    import seaborn
    seaborn.set_style(style="darkgrid")

    lambdas = np.linspace(0.2, 1.5, 10000)
    Bs = np.array([0.6961663, 0.4079426, 0.8974794])
    Cs = np.array([0.0684043, 0.1162414, 9.896161])
    Ns = [np.sqrt(e + 1) for e in Utilities.get_sellMeier_list(lambdas, Bs, Cs)]
    seaborn.lineplot(lambdas, Ns)
    plt.xlabel("Lambda")
    plt.ylabel("n")
    plt.show()
    print(Utilities.dndlamb([0.4, 0.67, 0.8], Ns, lambdas))

def opgave4():
    tstart = [10, 20, 100, 1000]
    lambA = 800*10**-9
    lambB = 600*10**-9
    print(Utilities.get_pulse_durations(tstart,lambA, 0.49*10**-11, 0.01))
    print(Utilities.get_pulse_durations(tstart, lambB, 1.79 * 10 ** -11, 0.01))

if __name__ == "__main__":
    # opgave1()
    # opgave2()
    opgave4()




