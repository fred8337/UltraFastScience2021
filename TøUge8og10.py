import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

#OPGAVE S4
def S4():
    Ce=97*6e3 #Jm**(-3)K**(-1)
    Cl=3.5e6 #Jm**(-3)K**(-1)
    Te0=6000 #K
    tau=300*1e-15 #s
    t=np.linspace(0,2e-12,100)
    Te=(Ce/(Ce+Cl)+Cl/(Ce+Cl)*np.exp(-t/tau))*Te0
    Tl=(Ce/(Ce+Cl)-Ce/(Ce+Cl)*np.exp(-t/tau))*Te0
    plt.plot(t,Te,'k',label='$T_e$')
    plt.plot(t,Tl,'k--',label="$T_l$")
    plt.legend()
    plt.show()
    
def S6():
    elements=["Al","Au","Ag","Cu","W"]
    A=np.array([(1-0.869),(1-0.974),(1-0.980),(1-0.963),(1-0.496)])
    F0=np.array([1,1,1,1,1])  # J/m^2 #Read off
    H=np.array([29.3,32.8,24.4,42.2,86.5])*10**9
    overAlpha=np.array([7.53,12.4,12.0,12.6,23.3])*10**(-9)  # m
    F_th=(H/A*overAlpha)
    return F_th


def J2eV(J):


if "__main__"==__name__:
    omega_p2 = 4.456e31
    Gamma = 1e15
    omega = 2.344e15
    c= 3e10 #cm/s
    epsilon = 1-omega_p2/(omega**2+1j*omega*Gamma)
    print(np.sqrt(epsilon))
    kappa = np.imag(np.sqrt(epsilon))
    Q00 = 2*kappa*omega/c*5 #J/cm^3


    