import scipy.optimize as so
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import random as r
import scipy.special as sse

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def gaussLinPlot(x,*p):
    A, mu, sigma,a,b, omega = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+a*x+b

def lin(x,*p):
    a, b = p
    return a*x + b

def gaussArea(xval,yval,min,max):
    index = (min < xval) & (xval < max)
    yval = yval[index]
    E = xval[index]
    A_guess = np.amax(yval)
    mu_guess = (E[-1]+E[0])/2
    sigma_guess = (E[-1]-E[0])/2
    p0 = [A_guess, mu_guess, sigma_guess]
    try:
        popt, pcov = so.curve_fit(gauss, E, yval, p0=p0,sigma=np.sqrt(yval))
        #plot(Elin,gauss(Elin,*popt), label="Peak ved "+str(popt[1]))
        #plt.errorbar(E,counts,yerr=np.sqrt(counts),fmt='.')
        area, err= quad(lambda x: gauss(x,*popt),min-20,max+20)
    except:
        area = 0
    return area


def pulsgauss(x,*p):
    A,mu,tau,omega0=p
    return A*np.exp(-2*np.log(2)/tau**2*(x-mu)**2)*np.exp(omega0*x)

def pulsgaussFit(xval,yval,min,max):
    index = (min < xval) & (xval < max)
    yval = yval[index]
    E = xval[index]
    A_guess = np.amax(yval)
    mu_guess = (E[-1]+E[0])/2
    sigma_guess = (E[-1]-E[0])/2
    omega_guess = 0
    p0 = [A_guess, mu_guess, sigma_guess,omega_guess]
    popt, pcov = so.curve_fit(pulsgauss, E, yval, p0=p0,sigma=np.sqrt(yval))
    return popt,pcov



def pulsgaussPlot(xval,yval,xmin,xmax, fig=None):
    popt,pcov=pulsgaussFit(xval,yval,xmin,xmax)
    index=(xmin<xval)&(xval<xmax)
    xval=xval[index]
    yval=yval[index]
    fignumber=plt.gcf().number
    fig=fig or r.randrange(100,1000)
    x=np.linspace(xmin,xmax,200)
    plt.figure(fig)
    plt.plot(xval,yval,'.')
    plt.plot(x,pulsgauss(x,*popt))
    plt.figure(fignumber)
    return popt,pcov

def gaussFit(xval,yval,min,max):
    index = (min < xval) & (xval < max)
    yval = yval[index]
    E = xval[index]
    A_guess = np.amax(yval)
    mu_guess = (E[-1]+E[0])/2
    sigma_guess = (E[-1]-E[0])/2
    p0 = [A_guess, mu_guess, sigma_guess]
    popt, pcov = so.curve_fit(gauss, E, yval, p0=p0,sigma=np.sqrt(yval))
    return popt,pcov,E

def gaussLinFit(xval,yval,min,max, sigma=[0,0]):
    index = (min < xval) & (xval < max)
    yval = yval[index]
    E = xval[index]
    A_guess = np.amax(yval)
    mu_guess = (E[-1]+E[0])/2
    sigma_guess = (E[-1]-E[0])/2
    a_guess=(yval[-1]-yval[0])/(E[-1]-E[0])
    b_guess=(yval[-1]-a_guess*E[-1]+yval[0]-a_guess*E[0])/2
    p0 = [A_guess, mu_guess, sigma_guess,a_guess,b_guess]
    popt, pcov = so.curve_fit(gaussLin, E, yval, p0=p0)
    if (sigma[0] == 0):
        sigma = np.sqrt(yval)
        popt, pcov = so.curve_fit(gaussLin, E, yval, p0=p0,sigma=sigma)
    return popt,pcov,index

def gaussLinArea(xval,yval,xmin,xmax):
    popt,pcov,_ = gaussLinFit(xval,yval,xmin,xmax)
    area, err = quad(lambda x: gaussLin(x,*popt) - popt[3]*x - popt[4], xmin-20,xmax+20)
    return area


    
def gaussPlot(xval,yval,xmin,xmax, fig=None):
    popt, pcov, E = gaussFit(xval,yval,xmin,xmax)
    index = (xmin < xval) & (xval < xmax)
    xval = xval[index]
    yval = yval[index]
    fignumber = plt.gcf().number
    fig = fig or r.randrange(100,1000)
    x = np.linspace(xmin,xmax,200)
    plt.figure(fig)
    plt.errorbar(xval,yval,yerr=np.sqrt(yval),fmt=".")
    plt.plot(x,gauss(x,*popt))
    plt.figure(fignumber)
    return popt,pcov,E

def gaussLinPlot(xval,yval,xmin,xmax,fig=None,sigma=[0,0]):
    popt, pcov, E = gaussLinFit(xval,yval,xmin,xmax,sigma=sigma)
    index = (xmin-20 < xval) & (xval < xmax+20)
    xval = xval[index]
    yval = yval[index]
    fignumber = plt.gcf().number
    x = np.linspace(xmin-20,xmax+20,200)
    fig = fig or r.randrange(100,1000)
    plt.figure(fig)
    plt.errorbar(xval,yval,yerr=np.sqrt(yval),fmt=".",label="Data points")
    plt.plot(x,gaussLin(x,*popt),linestyle="-",label="Fit")
    plt.plot(x,lin(x,*popt[-2:]))
    plt.axvline(x=xmin,linestyle="--",color="red",label="Region of Interest")
    plt.axvline(x=xmax,linestyle="--",color="red")
    plt.figure(fignumber)
    return popt,pcov,E

def gaussLinAreaStd(xval,yval,xmin,xmax):
    popt,pcov,_ = gaussLinFit(xval,yval,xmin,xmax)
    area, err = quad(lambda x: gaussLin(x,*popt) - popt[3]*x - popt[4], xmin-20,xmax+20)
    N = 1000
    pstd = np.zeros(len(popt))
    pdist = np.zeros([len(popt),N])
    for i in range(len(popt)):
        pstd[i] = np.sqrt(pcov[i,i])
        pdist[i,:] = np.random.normal(popt[i],pstd[i],N)
    areas = np.zeros(N)
    for i in range(N):
        areas[i],_ = quad(lambda x: gaussLin(x,*pdist[:,i])- pdist[3,i]*x - pdist[4,i], xmin-20, xmax+20)
    std = np.std(areas)
    #plt.hist(areas)
    #area = np.mean(areas)
    return area,std


"Funktionsudtryk for exp-gauss convolution"
def expGaussfunc(x,lambd,sigma,mu):
    return lambd/2*np.exp(0.5*lambd*(2*mu+lambd*sigma**2-2*x))*sse.erfc((mu+lambd*sigma**2-x)/(np.sqrt(2)*sigma))

"Funktionsudtryk for exp-gauss med lineær baggrund"
def expGaussLinFunc(x,*p):
    lambd, A,mu,sigma,a,b = p
    return A*expGaussfunc(x,lambd,sigma,mu) + a*x +b

"Funktionsudtryk for to exp-gauss og lineær baggrund"
def expGaussLinFunc2(x,*p):
    lambd,A,mu,sigma, lambd2,A2,mu2,sigma2,a,b = p
    return  A*expGaussfunc(x,lambd,sigma,mu) + A2*expGaussfunc(x,lambd2,sigma2,mu2) + a*x +b

"Fitter exp-gauss med lineær baggrund"
def expGaussLinFit(xval,yval,xmin,xmax, sigma=[0,0]):
    index = (xmin < xval) & (xval < xmax)
    yval = yval[index]
    E = xval[index]
    A_guess = np.amax(yval)
    mu_guess = (E[-1]+E[0])/2
    sigma_guess = (E[-1]-E[0])/2
    a_guess=(yval[-1]-yval[0])/(E[-1]-E[0])
    b_guess=(yval[-1]-a_guess*E[-1]+yval[0]-a_guess*E[0])/2
    lambd_guess = 0.1
    p0 = [lambd_guess, A_guess, mu_guess, sigma_guess,a_guess,b_guess]
    if (sigma[0] != 0):
        popt, pcov = so.curve_fit(expGaussLinFunc, E, yval, p0=p0,sigma=sigma[index])
    if (sigma[0] == 0):
        popt, pcov = so.curve_fit(expGaussLinFunc, E, yval, p0=p0)
    return popt,pcov,index

"Finder Areal af exp-gauss + lineær baggrund"
def expGaussLinArea(xval,yval,xmin,xmax, sigma=[0,0]):
    popt,pcov,_ = expGaussLinFit(xval,yval,xmin,xmax, sigma=sigma)
    area, err = quad(lambda x: expGaussLinFunc(x,*popt) - popt[4]*x - popt[5], xmin,xmax)
    N = 100
    pstd = np.zeros(len(popt))
    pdist = np.zeros([len(popt),N])
    for i in range(len(popt)):
        pstd[i] = np.sqrt(pcov[i,i])
        pdist[i,:] = np.random.normal(popt[i],pstd[i],N)
    areas = np.zeros(N)
    for i in range(N):
        areas[i],_ = quad(lambda x: expGaussLinFunc(x,*pdist[:,i])- pdist[4,i]*x - pdist[5,i], xmin, xmax)
    std = np.std(areas)
    return area, std

def expGaussLinArea2(xval,yval,xmin,xmax, sigma=[0,0]):
    popt,pcov,_ = expGaussLinFit2(xval,yval,xmin,xmax, sigma=sigma)
    x = np.linspace(xmin+1,xmax-1,500) #Den gik i stykker uden +1. Ved ikke hvorfor
    y = expGaussLinFunc2(x,*popt) - popt[8]*x - popt[9]
    N = 1000
    pstd = np.zeros(len(popt))
    pdist = np.zeros([len(popt),N])
    for i in range(len(popt)):
        pstd[i] = np.sqrt(pcov[i,i])
        pdist[i,:] = np.random.normal(popt[i],pstd[i],N)
    areas = np.zeros(N)
    for i in range(N):
        areas[i] = np.trapz(expGaussLinFunc2(x,*pdist[:,i])- pdist[8,i]*x - pdist[9,i], x)
    std = np.std(areas)    
    area = np.trapz(y,x)
    #area, err = quad(lambda x: expGaussLinFunc2(x,*popt) - popt[8]*x - popt[9], xmin,xmax)
    return area, std

"Plotter og fitter (optional) exp-gauss + lineær baggrund"
def expGaussLinPlot(xval,yval,xmin,xmax, sigma=[0,0],p=[0,0], acolor="orange", pcolor="red"):
    "acolor = Areal farve,  pcolor = Punkt farve"
    index = (xmin < xval) & (xval < xmax)
    "Nedenstående giver mulighed for at indsætte egen p"
    if (p[1] == 0):
        popt,pcov,index = expGaussLinFit(xval,yval,xmin,xmax, sigma=sigma)
        p = popt
    x = np.linspace(xmin,xmax,500)
    y = expGaussLinFunc(x,*p)
    ylin = p[-2]*x + p[-1]
    plt.plot(xval[index],yval[index],'.',color=pcolor)
    "Hvis der er angivet usikker plottes errobars"
    if (sigma[1] != 0):
        plt.errorbar(xval[index],yval[index],yerr=2*sigma[index],fmt=".",color=pcolor)
    plt.plot(x,y, color=acolor)
    plt.gca().fill_between(x,ylin,y, facecolor=acolor, alpha = 0.5)

"Fitter to exp-gauss med lineær baggrund"
def expGaussLinFit2(xval,yval,xmin,xmax, sigma=[0,0]):
    "Begrænser til mellem xmin og xmax"
    index = (xmin < xval) & (xval < xmax)
    yval = yval[index]
    xval = xval[index]
    "Gæt"
    sigma_guess =(xval[-1]-xval[0])/2
    a_guess=(yval[-1]-yval[0])/(xval[-1]-xval[0])
    b_guess=(yval[-1]-a_guess*xval[-1]+yval[0]-a_guess*xval[0])/2
    "for mu gæt skal vi finde peaks først"
    peaks,properties = sg.find_peaks(yval,prominence=15)
    if (len(peaks) == 1):
        peaks = np.array([peaks,peaks*1.05])
    #plot(xval[peaks],yval[peaks],"x")
    mu_guess = xval[peaks[0]]
    mu2_guess = xval[peaks[1]]
    A_guess = yval[peaks[0]]
    A2_guess = yval[peaks[1]]
    lambd_guess = 0.1
    p0 = [lambd_guess,A_guess,mu_guess,sigma_guess,   lambd_guess,A2_guess,mu2_guess,sigma_guess,   a_guess,b_guess]
    popt, pcov = so.curve_fit(expGaussLinFunc2, xval, yval, p0=p0)
    if (sigma[0] == 0):
        popt, pcov = so.curve_fit(expGaussLinFunc2, xval, yval, p0=p0)
    return popt,pcov,index
    
def expGaussLinPlot2(xval,yval,xmin,xmax,p=[0,0],sigma=[0,0]):
    popt,pcov,index = expGaussLinFit2(xval,yval,xmin,xmax, sigma=sigma)
    popt1 = np.concatenate((popt[0:4],popt[8:10]))
    popt2 = np.concatenate((popt[4:8],popt[8:10]))
    expGaussLinPlot(xval,yval,xmin,xmax, sigma=sigma,p=popt1,acolor="green")
    expGaussLinPlot(xval,yval,xmin,xmax, sigma=sigma,p=popt2)
    x = np.linspace(xmin,xmax,500)
    y = expGaussLinFunc2(x,*popt)
    plt.plot(x,y,'--',color="orange")
