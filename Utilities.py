import numpy as np

def get_sellMeier_list(Lambdas, Bs, Cs):
    res = np.empty(np.shape(Lambdas))
    for i,Lambda in enumerate(Lambdas):
        res[i]=get_sellMeier(Lambda, Bs, Cs)
    return res

def get_sellMeier(Lambda, Bs, Cs):
    res = 0
    for (B,C) in zip(Bs,Cs):
        res += (B*Lambda**2)/(Lambda**2-C**2)
    return res

def dndlamb(Lambda, Ns, Lambs):
    grads = np.gradient(Ns, Lambs[1]-Lambs[0])
    index = np.argmin(np.abs(Lambs-Lambda))
    return grads[index]