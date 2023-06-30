# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:03:38 2019

@author: ronjaebner
definitions of functions for model
added: stopp for DWF taking rho1 into account
"""
import numpy as np
import event72_inp as inp
#from scipy.signal import correlate , find_peaks
from numpy import pi

def norm(data):
    
    tmp         = data - 0.5*(max(data)+min(data))
    norm_max    = abs(max(tmp))
    norm_data   = tmp/norm_max
    
    return norm_data

def lagphase(n_outp, n_inp, t_plot):
    
    tmp     = t_plot[n_inp==1]
    T1_input= tmp[0]
    
    tmp     = t_plot[n_outp==max(n_outp)]
    T1_output=max(tmp)
    
    lag     = abs(T1_output-T1_input)
    phase   = (lag/(inp.T/1000))*360
    
    return lag, phase

#def shift_corr(n,fwb, output):
#    period = inp.T                            # period of oscillations (seconds)
#    tmax = n*inp.T                               # length of time series (seconds)
#    nsamples = tmax/inp.dt
##    noise_amplitude = 0
#    t = np.linspace(0.0, tmax, nsamples, endpoint=False)
#    xcorr = correlate(fwb, output)
#    dt = np.linspace(-t[-1], t[-1], 2*nsamples-1)
#    lag = dt[xcorr.argmax()]
#    phase = 2*pi*(((0.5 + lag/period)%1.0) - 0.5)
#
#    return lag, phase

#def nonLinearity(n_data, t_plot):
#    # find 2nd maxima and minima
#    peaks = find_peaks(n_data, height = 0.999, distance = 0.5*inp.T)
#    t_0     = t_plot[peaks[0]]
#    peaks = find_peaks(-n_data, height = 0.999, distance = 0.5*inp.T)
#    t_1     = t_plot[peaks[0]]        # position of second Peak
#    tmp     = t_1-t_0[1]     # distances between maximum and all minima    
#    t_l     = tmp[tmp<0]    # tmp<0 : left  hand side
#    r_r     = tmp[tmp>0]    # tmp>0 : right hand side
#    t_l0    = abs(t_l[-1])       # last entry is closest to t_0
#    t_r0    = r_r[ 0]       # first entry is closes to t_0
#    #test    = (t_l0+t_r0)/inp.T
#    Ll      = t_l0*2        # wavelength according to left  hand side
#    Lr      = t_r0*2        # wavelength according to right hand side
#    
#    return  Ll,  Lr

def rho(temp,sal):
## linear
#    temp= 16
#    density = 1027.5*(1 + 2e-4*(temp - 5)-8e-4*(sal- 35)) # as in old model, but with fixed temperature
     
## EOS80    
#    a=  np.array([ 999.842594  , 6.793952e-2,-9.095290e-3, 1.001685e-4,-1.120083e-6, 6.536332e-9])
#    b=  np.array([   8.24493e-1,-4.0899e-3  , 7.6438e-5  ,-8.2467e-7  , 5.3875e-9])
#    c=  np.array([  -5.72466e-3, 1.0227e-4  ,-1.6546e-6])
#    d=  4.8314e-4
#    density=((a[0]+a[1]*temp + a[2]*(temp**2) +a[3]*(temp**3)+a[4]*(temp**4) +a[5]*(temp**5))+
#             (b[0]+b[1]*temp + b[2]*(temp**2) +b[3]*(temp**3)+b[4]*(temp**4))*sal+
#             (c[0]+c[1]*temp + c[2]*(temp**2))*(sal**(3/2))+
#             d*(sal**2))
    
## Simple
    density= sal
    return density

    
def water_flux_margConv(Rm, Qo, rhoa, Flux, m, EPR0, EPR1):
    Fin        = Qo-min(0,(EPR0+EPR1)) # influx compensating for fwb
    #
    Flux[0][2] = max(0,Rm[0]-Rm[2])*inp.kbg*inp.A[0]   # convection for R0>R2
    Flux[2][0] = Flux[0][2]*inp.cap 
    Flux[1][2] = max(0,(Rm[1]-Rm[2]))*inp.kbg*inp.A[1] #fixed scaling, like margins
    Flux[2][1] = (Flux[0][2]-Flux[2][0])+Flux[1][2]    # in must go out
    Flux[1][0] = max(0, (Flux[0][2]-Flux[2][0])-EPR0)  # compensating DWF and evap
    Flux[0][1] = max(0,-(Flux[0][2]-Flux[2][0])+EPR0)  # compensating DWF and evap
    Fout       = Qo + max(0,EPR0+EPR1)
    Fdown      = 0
    # Mixing Parameters
    fm0    =(rhoa<Rm[2])*min(1,(Qo<0)*Qo*inp.cfm);
    m[0]   =         max(inp.kbg, (Rm[0]-Rm[2])*inp.kstr + inp.kbg) *inp.Wc0#*inp.f_mix
    m[1]   =(1-fm0)* max(inp.kbg, (Rm[1]-Rm[2])*inp.kstr + inp.kbg) *inp.Wc1*inp.f_mix
    m[2]   =   fm0 * max(inp.kbg, (rhoa -Rm[2])*inp.kstr + inp.kbg) *inp.Wc1*inp.f_mix
    return Fin, Fout, Fdown, Flux, m
##
def param_flux(box, Flux, Param):
        #summation over all outflow*S
        #summation over all the inflows*S
    dPdt = (-sum(Flux[box,:])* Param[box]+
             sum(Flux[:,box] * Param[:])  )
    #dPdt = dPdt/Vol[box]
    return dPdt

def read_data(filename, dt, t_max):
    data = 5
    return data

def find_ampl(data):
    tmp_b=0
    tmp_t=0
    n=0
    t_n= 0
    while t_n<inp.runtime/inp.dt-(1/2)*inp.T/inp.dt :
        t_n     = (n/2)*inp.T/inp.dt #time at which Amp=0
        tmp_t   = np.append(tmp_t,int(t_n))
        tmp_b   = np.append(tmp_b,data[int(t_n)])
        n+=1
    # delete first entry
    tmp_T = np.delete(tmp_t, 0, 0)
    tmp_B = np.delete(tmp_b, 0, 0)
    # interpolate to time vector
    b_t = np.interp(inp.t_vec[:],  tmp_T, tmp_B)
    # calclate Amplitude at non-zero points in time
    # pi/4 + n*pi/2
    tmp_a=0
    tmp_t=0
    n=0
    t_n=0
    while t_n<inp.runtime/inp.dt-(1/4)*inp.T/inp.dt:
        t_n      = ((1/2+n)/4)*inp.T/inp.dt #time at which sin()!=0
        tmp_t    = np.append(tmp_t,int(t_n))
        tmp_a    = np.append(tmp_a,(data[int(t_n)]-b_t[int(t_n)])/np.sin((2*np.pi*inp.dt/inp.T)*t_n))
        n+=1
        
    # delete first entry
    tmp_T = np.delete(tmp_t, 0, 0)
    tmp_A = np.delete(tmp_a, 0, 0)
    # interpolate to time vector
    a_t = np.interp(inp.t_vec[:],  tmp_T, tmp_A)
    return a_t , b_t

def find_ampl_base(data, idx0):
    # Amplitude calculated by (max-min)/2 within one period (moving window)
    # Baseline  calculated by (max+min)/2 within one period (moving window)
    prec = 10000
    Period      = inp.T
    TimeStep    = inp.dt
    FrameStep   = Period/TimeStep*(2/11)
    FrameWidth  = Period/TimeStep
    i_end       = len(inp.t_vec)-     FrameWidth
    i           = idx0          + 0.5*FrameWidth
    a_t=np.zeros_like(inp.t_vec)
    b_t=np.zeros_like(inp.t_vec)
    t  =np.zeros_like(inp.t_vec)

    while i<i_end:
        int_0   = int(i- 0.5*FrameWidth)
        int_1   = int(i+ 0.5*FrameWidth)
        #
        a_t[int(i)]     = prec*(max(data[int_0:int_1]) - min(data[int_0:int_1]))/2
        b_t[int(i)]     = prec*(max(data[int_0:int_1]) + min(data[int_0:int_1]))/2
        t[int(i)]       = i
        #
        i+=FrameStep
    ##
    A_t = a_t[t != 0]/prec
    B_t = b_t[t != 0]/prec
    T   =   t[t != 0]*inp.dt/1000
    return A_t,  B_t , T

def find_ampl_base_neu(data, idx0, T, dt, t_vec):
    # Amplitude calculated by (max-min)/2 within one period (moving window)
    # Baseline  calculated by (max+min)/2 within one period (moving window)
    prec = 10000
    Period      = T
    TimeStep    = dt
    FrameStep   = Period/TimeStep*(2/11)
    FrameWidth  = Period/TimeStep
    i_end       = len(t_vec)-     FrameWidth
    i           = idx0          + 0.5*FrameWidth
    a_t=np.zeros_like(t_vec)
    b_t=np.zeros_like(t_vec)
    t  =np.zeros_like(t_vec)

    while i<i_end:
        int_0   = int(i- 0.5*FrameWidth)
        int_1   = int(i+ 0.5*FrameWidth)
        #
        a_t[int(i)]     = prec*(max(data[int_0:int_1]) - min(data[int_0:int_1]))/2
        b_t[int(i)]     = prec*(max(data[int_0:int_1]) + min(data[int_0:int_1]))/2
        t[int(i)]       = i
        #
        i+=FrameStep
    ##
    A_t = a_t[t != 0]/prec
    B_t = b_t[t != 0]/prec
    T   =   t[t != 0]*dt/1000
    return A_t,  B_t , T

def derivative(f,h):
    g=np.zeros(len(f))
    i=0
    while i<len(f)-2:
        i+=1
    # 'central'
        g[i]= (1/(2*h))*(f[i-1]-f[i+1])
    return g