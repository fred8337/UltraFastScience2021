import numpy as np
import Utilities
import matplotlib.pyplot as plt

## Opgave 1)

# a)

## Opgave 2)

# a)


if __name__ == "__main__":
    import seaborn
    seaborn.set_style(style="darkgrid")

    lambdas = np.linspace(0.2,1.5,10000)
    Bs = np.array([0.6961663,0.4079426,0.8974794])
    Cs = np.array([0.0684043,0.1162414,9.896161])
    Ns = [np.sqrt(e+1) for e in Utilities.get_sellMeier_list(lambdas, Bs, Cs)]
    seaborn.lineplot(lambdas,Ns)
    plt.xlabel("Lambda")
    plt.ylabel("n")
    plt.show()
    print(Utilities.dndlamb(0.8, Ns, lambdas))




