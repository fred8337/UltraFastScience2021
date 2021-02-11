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
    res = []
    for L in Lambda:
        grads = np.gradient(Ns, Lambs[1]-Lambs[0])
        index = np.argmin(np.abs(Lambs-L))
        res.append(grads[index])
    return res

def get_group_delay(z, nLambda, Lambda, dndlamb):
    return (z/299792458)*(nLambda-Lambda*dndlamb)

def get_pulse_duration(t0,Lambda,dndlamb,z):
    #TODO: Enheder virker ikke
    phimm = (Lambda**3/(2*np.pi*299792458**2))*dndlamb*z
    return (t0*np.sqrt(1+np.power((4*np.log(2)*phimm)/((t0)**2),2)))

def get_pulse_durations(t0s,Lambda,dndlamb,z):
    res = []
    for t0 in t0s:
        res.append(get_pulse_duration(t0,Lambda,dndlamb,z))
    return res