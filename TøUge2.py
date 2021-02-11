import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
c=3e8
def opgave12g():
    #Not done
    P=200*1e-6
    tau=100*1e-15
    diameter=125*1e-6
    omega_0=800*1e-9
    n_2=9.8e-24
    z=0.5
    I=P/((diameter/2)**2*np.pi)
    Domega_max=4*omega_0/(c*tau)*n_2*I*np.sqrt(np.log(4))*np.exp(-1/(2))*z
    Dlambda=2*np.pi*c/Domega_max
    return Domega_max
def opgave4(lamb):
    #Calculating the real pulse duration when propagation
    #is taking into account.
    tau_0=np.array([10*1e-15,20*1e-15,100*1e-15,1*1e-12])
    z=1e-2
    if lamb==600:
        gdd=(600*1e-9)**3/(2*np.pi*c**2)*1.79e11*z
        print('her:'+ str(gdd))
    if lamb==800:
        gdd=(800*1e-9)**3/(2*np.pi*c**2)*0.49e11*z
        print('her:'+str(gdd))
    #One should find that the shorter pulses are distorted more than
    #longer pulses because of more fourier components
    #(more frequency modes)
    tau=1e15*tau_0*np.sqrt(1+(4*np.log(2)*gdd/(tau_0**2))**2)
    return tau


def opgave10():
    lamb=10.6*1e-6
    x=10.6 #Lambda i mu m
    L_laser=150*1e-6

    #Calculate the E-field of a monochromatic wave
    t=np.linspace(-200*1e-15,200*1e-15,200)
    E=np.cos(2*pi*c/(lamb)*t)
    plt.plot(t,E,label='E(t)')
    plt.title('Monofreq wave')
    plt.xlabel('t[s]')

    #Calculating the frequency spacing by first finding the refractive
    #index of CO_2
    n=1+6.99100e-2/(166.175-x**-2)+1.44720e-3/(79.609-x**-2)\
      +6.42941e-5/(56.3064-x**-2)\
      +5.21306e-5/(46.0196-x**-2)+1.46847e-6/(0.0584738-x**-2)
    vspac=c/(2*L_laser*n)
    print("Frequency spacing= "+'{:.2e}'.format(vspac))
    v_cent=c/lamb

    #Calculating the E-field when more frequency modes are taken
    #into account
    t_comp=np.linspace(-2000*1e-15,2000*1e-15,500)
    E_comp=np.cos(2*pi*v_cent*t_comp)+np.cos(2*pi*(v_cent+vspac)*t_comp)\
           +np.cos(2*pi*(v_cent-vspac)*t_comp)
    plt.figure(2)
    plt.plot(t_comp,E_comp,label='E(t)')
    plt.title('Composite Efield')
    plt.xlabel('t[s]')

    #Function to add arbitrarily more standing waves that can be added
    t_supercomp=np.linspace(-2000*1e-15,2000*1e-15,1000)
    modes=5
    E_supercomp=np.cos(2*pi*v_cent*t_supercomp)
    for n in range(1,modes):
        E_supercomp+=np.cos(2*pi*(v_cent+n*vspac)*t_supercomp)\
           +np.cos(2*pi*(v_cent-n*vspac)*t_supercomp)
    plt.figure(3)
    plt.plot(t_supercomp,E_supercomp,label='E(t)')
    plt.title('Super Composite Efield')
    plt.xlabel('t[s]')


    #The same as before but now with fase
    t_phase=np.linspace(-2000*1e-15,2000*1e-15,1000)
    modes=5
    phase_0=2*pi/modes
    E_phase=np.cos(2*pi*v_cent*t_phase)
    for n in range(1,modes):
        E_phase+=np.cos(2*pi*(v_cent+n*vspac)*t_phase+n*phase_0) \
                     +np.cos(2*pi*(v_cent-n*vspac)*t_phase+n*phase_0)

    plt.figure(4)
    plt.plot(t_phase,E_phase,label='E(t)')
    plt.title('Composite Efield phaseshifted')
    plt.xlabel('t[s]')

    #plotting intensity
    plt.figure(5)
    plt.plot(t_phase,np.abs(E_phase)**2,label='E(t)')
    plt.title('Composite Efield phaseshifted')
    plt.xlabel('t[s]')


    plt.show()
if __name__=='__main__':
    opgave10()