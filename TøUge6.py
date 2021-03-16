import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import scipy.special as ss
import scipy.integrate as si


from numpy import pi
def Y20(theta,phi):
    return 1/4*np.sqrt(5/(pi))*(3*(np.cos(theta))**2-1)
def Y21(theta, phi):
    return -1/2*np.sqrt(15/(2*pi))*np.exp(phi*1j)*np.sin(theta)*np.cos(theta)

def Y22(theta, phi):
    return 1/4*np.sqrt(15/(2*pi))*np.exp(phi*2j)*(np.sin(theta))**2

def prepState(theta,phi):
    return 1/np.sqrt(3)*(ss.sph_harm(0,0,phi,theta)+ss.sph_harm(0,2,phi,theta)+ss.sph_harm(0,4,phi,theta))

def pr02State(theta,phi):
    return 1/np.sqrt(2)*(ss.sph_harm(0,0,phi,theta)+ss.sph_harm(0,2,phi,theta))

def expecCos(P):
    f = lambda phi,theta: np.sin(theta)*np.cos(theta)**2*np.conj(P(theta,phi))*P(theta,phi)
    intval,_=si.dblquad(f,0,pi,lambda theta: 0, lambda theta: 2*pi)
    return intval
def opgaveA1():
    # 1
    thta=np.linspace(0,pi,100)
    nY20=np.abs(Y20(thta,0))
    nY21=np.abs(Y21(thta,0))
    nY22=np.abs(Y22(thta,0))
    plt.plot(thta/pi,nY20,'k--',label="$|Y20|^2$")
    plt.plot(thta/pi,nY21,'k.-',label="$|Y21|^2$")
    plt.plot(thta/pi,nY22,'k*-',label='$|Y22|^2$')
    plt.plot(thta/pi,np.cos(thta)**2*0.1,'k',label="$\cos^2$")
    plt.figure(2)
    plt.plot(thta/pi,1/5*(2*nY21**2+2*nY22**2+nY20**2))
    plt.legend()
    # 2 Allignes in some way, can seen from plot that the norm covers the entire range
    # 3
    mY20=expecCos(Y20)
    mY21=expecCos(Y21)
    mY22=expecCos(Y22)
    uniCos=np.mean((np.cos(thta))**2)
    mprep=expecCos(prepState)
    m02prep=expecCos(pr02State)
    print(m02prep)
    print("expec 20:" + str(mY20),"expec 21:" + str(mY21),"expec 22:" + str(mY22),"uniform expec: sholud be 0.33" ,"Prepared state:" + str(mprep))
    plt.show()
if __name__=="__main__":
    opgaveA1()