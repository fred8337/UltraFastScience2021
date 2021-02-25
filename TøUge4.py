import numpy as np
from numpy import pi
#Dette dokument skal svare på opgaverne i ultrafast science uge 4
# prøver for så vidt at køre SI enheder
#men det er nogle fucking ingeniør formler det her så det kan ikke
#altid lykkes
#Og så kører vi også lidt danglish
c=3e8
hbar=1.055*1e-34 #J*s
def opgave1():
    print("Opgave 1")
    #He-Ne laser
    wavelength=633*1e-9 #m
    P=20*1e-3 #W
    #Cavity, partially silvered exit mirror
    diameter=1 #mm
    #laser
    bandwidth=0.1*1e-9 #m
    divergence=1 #mrad
    #Brilliance= photons/second/[(mm^2 source area)*(mrad)**2*(0.001*bandwidth)]
    energyprphoton=2*pi*hbar*c/wavelength
    photonsprsecond=P/energyprphoton
    Brilliance=photonsprsecond/(pi*(diameter/2)**2*divergence**2*(0.001*bandwidth))
    print(Brilliance)
def opgave2():
    print("Opgave 2:")
    brilliance=2e19 #ph/s/mm**2/mrad**2/(0.001 BW)
    wavelength=633*1e-9  # m
    area=50*5e-6  # mm
    # laser
    bandwidth=1.4*1e-4  #m
    divergence=75*15*1e-6 #mrad
    # Brilliance= photons/second/[(mm^2 source area)*(mrad)**2*(0.001*bandwidth)]
    energyprphoton=1.5e4*1.602e-19
    photonsprsecond=brilliance*area*divergence**2*(10*bandwidth)
    #Det med bandwidth er fra facit, jeg forstår det ikke
    P=energyprphoton*photonsprsecond
    print(photonsprsecond,P)

def opgave3():
    print("opgave 3: ")
    E=6*1e9*1.602*1e-19 #J
    circ=844 #m
    f=352*1e6 #Hz
    I=200*1e-3 #A
    t=circ/c
    bunches=f*t
    print("bunches: " + str(bunches))
    bunchsep=circ/bunches
    print("Bunch seperation: "+str(bunchsep))
    eprsec=I/(1.602e-19)
    eprring=eprsec*t
    eprbunch=eprring/bunches
    print("Electrons in each bunch: " + str(eprbunch))
    Ekin=eprring*E
    print("E_kin: "+str(Ekin))
def opgave4():
    print("Opgave 4: ")
    E=3 #J
    I=250*1e-3 #A
    L=0.5 #m
    B=1.4 #T
    P=1.266*E**2*B**2*0.5*I*1e3 #W
    print("Power in all angles: " + str(P))
def opgave5():
    print("Opgave 5: ")
    #SLS top-up mode
    E=2.4e9*1.602e-19 #J
    L=288 #m
    #beam current drops from 401 mA to 399 mA
    I_start=0.401 #A
    I_slut=0.399 #A
    interval=3*60 #s
    #Calculate decay time
    #399=401*exp(-3 min/tau) => 3 min/ln(401/399)
    tau=interval/np.log(I_start/I_slut)
    print("Decay time: " + str(tau))
    t=L/c
    dIdt=(I_slut-I_start)/interval
    d2edt=dIdt/(1.602e-19)
    #multiply with the tme for one electron to make around trip to make sure that the change in current is from new
    #electrons disappearing.
    dedt=d2edt*L/c
    print("Change of electrons pr. second: " +str(dedt))

def opgave6():
    print("opgave 6: ")
    E=2.4 #GeV
    #Undulator magnet periodicity
    lamb_u=14*1e-3 #m
    #Deviation parameter calculated between 0.65<= K <= 1.6
    #Calculate the wavelengths of the first and second order undulator
    #radiation for the extremes og these values
    K=np.array([0.65,1.6])
    lamb_1=13.056*10*(lamb_u)/E**2*(1+K**2/2)
    lamb_2=1/2*13.056*10*(lamb_u)/E**2*(1+K**2/2)
    print(str(K**2))
    print('lamb_1: ' +str(lamb_1))
    print('lamb_2: ' + str(lamb_2))



if __name__=='__main__':
    opgave1()
    opgave2()
    opgave3()
    opgave4()
    opgave5()
    opgave6()