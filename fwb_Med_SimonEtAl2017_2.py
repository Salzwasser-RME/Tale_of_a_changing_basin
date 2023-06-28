#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:39:51 2021

@author: ron

Calculating fwb of MS between 7.3Ma and 5.2

Coefficients by Simon et al., 2017

Data byLaskar et al., 2004
"""

# Import libraries
import           numpy   as np
from scipy.ndimage.filters import uniform_filter1d
from scipy import fftpack
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#%% define fwb-type and parameters

# 0-> P-E+RwC
# 1-> P-E+RnC
# 2-> RwC
# 3-> RnC
# 4-> P-E
names=                       ['PERwC','PERnC','RwC','RnC', 'P-E']
coeff_Intercept   = np.array([-2.142,-2.293, 0.657, 0.384,-2.747, 0])
coeff_Precession  = np.array([ 0    , 0    , 0    , 0    ,-0.079, 1])
coeff_Obliquity   = np.array([ 0    , 0    , 0.105, 0.005, 0.131, 1])
coeff_Eccentricity= np.array([ 0    , 0    ,-0.713,-0.211, 0.484, 1])
coeff_Precession2 = np.array([ 0.665, 0.181, 0.616, 0.201, 0.004, 1])
coeff_Obliquity2  = np.array([ 0    , 0.584,-0.845,-0.121, 0.863, 1])
coeff_Eccentricity2=np.array([ 0.256,-0.109, 1.274, 0.375,-0.772, 1])
coeff_PrecObl     = np.array([ 0    , 0    , 0    , 0    , 0    , 1])
coeff_PrecEcc     = np.array([-1.244,-0.378,-1.252,-0.421, 0.039, 1])
coeff_OblEcc      = np.array([-0.071, 0    ,-0.104, 0    ,-0.039, 1])
r2                = np.array([ 0.99 , 0.96 , 0.99 , 0.99 , 0.90 ])


#%% Data
# The result window contains two or more columns :

#     time (expressed in 103 Julian years since J2000.0, the julian year is equal to 365.25 days [help])
#     eccentricity
#     climatic precession 
#     obliquity expressed in radians
# http://vo.imcce.fr/insola/earth/online/earth/online/index.php
data=np.loadtxt("LaskarEtAl2004_EccentricityNeu.txt")
Time_vec    = data[:,0]

tmp=data[:,1]
Eccentricity=(tmp/max(tmp))
#Eccentricity=data[:,1]

tmp=data[:,2]-np.mean(data[:,2])
Precession=tmp/max(tmp)
#Precession=data[:,2]

tmp=data[:,3]- np.mean(data[:,3])
Obliquity=tmp/max(tmp)



fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(Time_vec/1000 , Eccentricity, label='Eccentricity')
ax.plot(Time_vec/1000 , Precession , label='Precession')
ax.plot(Time_vec/1000 , Obliquity , label='Obliquity ')
#ax.plot(Time_vec/1000 , test , label='test ')
plt.legend(fontsize=20)


#%%
def fwb_calc(version):
    fwb= (   
         coeff_Intercept[version]
       + coeff_Precession[version]   * Precession 
       + coeff_Obliquity[version]    * Obliquity 
       + coeff_Eccentricity[version] * Eccentricity
       + coeff_Precession2[version]  * np.multiply(Precession  , Precession  ) 
       + coeff_Obliquity2[version]   * np.multiply(Obliquity   , Obliquity   )
       + coeff_Eccentricity2[version]* np.multiply(Eccentricity, Eccentricity)
       + coeff_PrecObl[version]      * np.multiply(Precession  , Obliquity   )
       + coeff_PrecEcc[version]      * np.multiply(Precession  , Eccentricity)
       + coeff_OblEcc[version]       * np.multiply(Obliquity   , Eccentricity)
       )*10**12
    return fwb
    
#%%
version=0
PERwC= fwb_calc(version)

version=1
PERnC= fwb_calc(version)

version=2
RwCtmp= fwb_calc(version)

version=3
RnC= fwb_calc(version)

RwC=np.maximum(RwCtmp, RnC)

version=4
PE= fwb_calc(version)

#%% Save data
# data that will be saved to be used later
data2save= [Time_vec, PE, RnC, RwC]

np.savetxt("FWB_Simon2017_new.txt", data2save ,delimiter=",")


#%% Find ratio at given time
# the way of calculating the amplitude and baseline does only partially workhere
# in thiscase the period of the precession is not fixed (23000) but changes
# that way the baseline is a bit wobbly


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]


    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

data= PERwC
lmin,lmax = hl_envelopes_idx(data)
data_high= interp1d(Time_vec[lmax], data[lmax]/(2.5*10**12), kind='cubic')
data_low = interp1d(Time_vec[lmin], data[lmin]/(2.5*10**12), kind='cubic')
data_a=(data_low(Time_vec[200:20805]) - data_high(Time_vec[200:20805]))/2 
data_b=(data_low(Time_vec[200:20805]) + data_high(Time_vec[200:20805]))/2 
#PERnC_T= interp1d(Time_vec[lmax], np.diff(Time_vec[lmin]),  kind='cubic')

#%%
f, ax= plt.subplots(2,1, sharex= True) 
ax[0].set_title("reconstructed net-evaporation rate")
ax[0].set_ylabel("m/yr")
ax[1].set_ylabel("a-b-ratio [-]")
ax[1].set_xlabel("Ma")
ax[0].vlines(-7.17, min(data/(2.5*10**12) )*1.1, max(data/(2.5*10**12)  )*0.9, color= 'g', linewidth= 3,alpha= 0.7)
ax[0].plot(Time_vec[200:20805]/1000 ,data[200:20805]/(2.5*10**12)  , color= "grey", label='e')
ax[0].plot(Time_vec[200:20805]/1000 ,data_b , color= "tab:blue", linewidth= 2,  label="b")
ax[0].plot(Time_vec[200:20805]/1000 ,(data_b+data_a) , color= "tab:blue", alpha= 0.7,  label="upper/lower limit")
ax[0].plot(Time_vec[200:20805]/1000 ,(data_b-data_a) , color= "tab:blue", alpha= 0.7)#high_idx, low_idx 
#ax[0].legend()
ax[0].set_xlim([-7.3, -6.5])
ax[0].set_ylim([min(data/(2.5*10**12) )*1.1, max(data/(2.5*10**12)  )*0.9])
ax[1].set_ylim([min((data_a)/(data_b))*0.9, max((data_a)/(data_b) )*1.1])
ax[1].vlines(-7.17, min((data_a)/(data_b))*0.9, max((data_a)/(data_b) )*1.1, color= 'g', linewidth= 3,alpha= 0.7)
ax[1].plot(Time_vec[200:20805]/1000 ,(data_a)/(data_b) ,color= "tab:orange" , label="net evapo rate, ratio")#high_idx, low_idx 
ax[0].set(frame_on=False)
ax[1].set(frame_on=False)
f.subplots_adjust(hspace=0.0)
plt.savefig("FWB_Analsysis_noChad.png")
#%%
f, ax= plt.subplots(2,1, sharex= True) 
ax[0].set_title("recontructed net-evaporation, incl. Chad river")
ax[0].set_ylabel("m/yr")
ax[1].set_ylabel("a-b-ratio [-]")
ax[1].set_xlabel("Ma")
d_min=-1.15
d_max= 0
minmax=[d_min,d_max] #[min(data/(2.5*10**12) )*1.1, max(data/(2.5*10**12)  )*0.9]
ax[0].vlines(-5.97,d_min, d_max , color= 'g', linewidth= 3,alpha= 0.7)
ax[0].vlines(-5.61,d_min, d_max , color= 'g', linewidth= 3,alpha= 0.7)
ax[0].vlines(-5.55,d_min, d_max , color= 'g', linewidth= 3,alpha= 0.7)
ax[0].plot(Time_vec[200:20805]/1000 ,data[200:20805]/(2.5*10**12)  , color= "grey", label='e')
ax[0].plot(Time_vec[200:20805]/1000 ,data_b , color= "tab:blue", linewidth= 2,  label="b")
ax[0].plot(Time_vec[200:20805]/1000 ,(data_b+data_a) , color= "tab:blue", alpha= 0.7,  label="upper/lower limit")
ax[0].plot(Time_vec[200:20805]/1000 ,(data_b-data_a) , color= "tab:blue", alpha= 0.7)#high_idx, low_idx 
#ax[0].legend()
ax[0].set_xlim([-6.2, -5.3])
ax[0].set_ylim([d_min,d_max] )#([min(data/(2.5*10**12) )*1.1, max(data/(2.5*10**12)  )*0.9])
d_min=-0.1 # min((data_a)/(data_b))*0.9
d_max= 1.1   # max((data_a)/(data_b) )*1.1
ax[1].set_ylim([d_min,d_max] )#([min((data_a)/(data_b))*0.9, max((data_a)/(data_b) )*1.1])

#ax[1].vlines(-5.97, min((data_a)/(data_b))*0.9, max((data_a)/(data_b) )*1.1, color= 'g', linewidth= 3,alpha= 0.7)
ax[1].vlines(-5.97, d_min, d_max , color= 'g', linewidth= 3,alpha= 0.7)
ax[1].annotate("phase 1", xy=(-5.97, 0.85), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
ax[1].vlines(-5.61, d_min, d_max ,  color= 'g', linewidth= 3,alpha= 0.7)
ax[1].annotate("phase 2", xy=(-5.61, 0.85), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
ax[1].vlines(-5.55, d_min, d_max ,  color= 'g', linewidth= 3,alpha= 0.7)
ax[1].annotate("phase 3.1", xy=(-5.55, 0.85), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
ax[1].plot(Time_vec[200:20805]/1000 ,(data_a)/(data_b) ,color= "tab:orange" , label="net evapo rate, ratio")#high_idx, low_idx 
ax[0].set(frame_on=False)
ax[1].set(frame_on=False)
#ax[1].xlim([])
f.subplots_adjust(hspace=0.0)
plt.savefig("FWB_Analsysis_withChad.png")
#%%
plt.figure()
plt.title("reconstructed net-evaporation rate")
plt.ylabel("m/yr")
plt.xlabel("Ma")
plt.vlines(-5.97, min(data/(2.5*10**12) )*1.1, max(data/(2.5*10**12)  )*0.9, color= 'g', linewidth= 3,alpha= 0.7)
plt.vlines(-5.61, min(data/(2.5*10**12) )*1.1, max(data/(2.5*10**12)  )*0.9, color= 'g', linewidth= 3,alpha= 0.7)
plt.vlines(-5.55, min(data/(2.5*10**12) )*1.1, max(data/(2.5*10**12)  )*0.9, color= 'g', linewidth= 3,alpha= 0.7)
plt.plot(Time_vec[200:20805]/1000 ,data[200:20805]/(2.5*10**12)  , color= "grey", label='e')
plt.plot(Time_vec[200:20805]/1000 ,data_b , color= "tab:blue", linewidth= 2,  label="b")
plt.plot(Time_vec[200:20805]/1000 ,(data_b+data_a) , color= "tab:blue", alpha= 0.7,  label="upper/lower limit")
plt.plot(Time_vec[200:20805]/1000 ,(data_b-data_a) , color= "tab:blue", alpha= 0.7)#high_idx, low_idx 
#ax[0].legend()
plt.xlim([-6.2, -5.3])
plt.ylim([min(data/(2.5*10**12) )*1.1, max(data/(2.5*10**12)  )*0.9])
plt.box(False)

#ax[1].xlim([])


#%% river as percetage of riverinflux
data=RwC/PE
data1=RnC/PE
plt.figure()
plt.title("river inflow in percent of evaporational flux")
plt.ylabel("R/EP [-]")
plt.xlabel("time [Ma]")
plt.vlines(-5.97, min(data)*1.1, max(data )*0.9, color= 'g', linewidth= 3,alpha= 0.7)
plt.vlines(-5.61, min(data)*1.1, max(data )*0.9, color= 'g', linewidth= 3,alpha= 0.7)
plt.vlines(-5.55, min(data)*1.1, max(data )*0.9, color= 'g', linewidth= 3,alpha= 0.7)
plt.plot(Time_vec[200:20805]/1000 ,data[200:20805]  , color= "grey", label='with Chad')
plt.plot(Time_vec[200:20805]/1000 ,data1[200:20805] , color= "tab:blue", linewidth= 2,  label="no Chad")
plt.legend()
plt.xlim([-6.2, -5.3])
#plt.ylim([min(data/(2.5*10**12) )*1.1, max(data/(2.5*10**12)  )*0.9])
plt.box(False)
#%%
data= 10**(-6)*RnC/(60*60*24*365.25)
lmin,lmax = hl_envelopes_idx(data)
data_high= interp1d(Time_vec[lmax], data[lmax], kind='cubic')
data_low = interp1d(Time_vec[lmin], data[lmin], kind='cubic')
data_a=(-data_low(Time_vec[210:20800]) + data_high(Time_vec[210:20800]))/2 #amplitude
data_b=(data_low(Time_vec[210:20800]) + data_high(Time_vec[210:20800]))/2 #baseline

data_mean= np.mean(data)
data_min_1 = np.mean(data_low(Time_vec[13270:16900]))
data_min_2 = np.mean(data_low(Time_vec[16900:17500]))
data_min_3 = np.mean(data_low(Time_vec[17500:20800]))

plt.figure()
plt.title("reconstructed river inflow (no Chad)")
plt.ylabel("Sv")
plt.xlabel("Ma")
plt.vlines(-5.97,0, 0.1, color= 'g', linewidth= 2,alpha= 0.7)
plt.vlines(-5.61, 0, 0.1, color= 'g', linewidth=2,alpha= 0.7)
plt.vlines(-5.55,0, 0.1, color= 'g', linewidth= 2,alpha= 0.7)

plt.hlines(data_min_1 ,-5.97, -5.61, color="tab:orange")
plt.hlines(data_min_2 ,-5.61, -5.55, color="tab:orange")
plt.hlines(data_min_3 ,-5.55, -5.3, color="tab:orange")
plt.annotate("phase 3", xy=(-5.55, 0.03), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
plt.annotate("phase 1", xy=(-5.97, 0.03), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
plt.annotate("phase 2", xy=(-5.61, 0.03), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
plt.plot(Time_vec[210:20800]/1000 ,data[210:20800]  , color= "grey", label='e')
plt.plot(Time_vec[210:20800]/1000 ,data_b , color= "tab:blue", linewidth= 2,  label="b")
plt.plot(Time_vec[210:20800]/1000 ,(data_b+data_a) , color= "tab:blue", alpha= 0.7,  label="upper/lower limit")
plt.plot(Time_vec[210:20800]/1000 ,(data_b-data_a) , color= "tab:blue", alpha= 0.7)#high_idx, low_idx 
plt.annotate("mean flux ={:.3f} Sv".format(data_mean), xy=(-5.75, 0.095), xycoords="data",
                  va="center", ha="center",rotation=0,
                  bbox=dict(boxstyle="round", fc="w", color='grey'))
plt.xlim([-6.2, -5.3])
plt.ylim([0, 0.04])
plt.box(False)
#%% THIS ONE
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.ticker as tck
data= 10**(-6)*RnC/(60*60*24*365.25)
lmin,lmax = hl_envelopes_idx(data)
data_high= interp1d(Time_vec[lmax], data[lmax], kind='cubic')
data_low = interp1d(Time_vec[lmin], data[lmin], kind='cubic')
data_a=(-data_low(Time_vec[210:20800]) + data_high(Time_vec[210:20800]))/2 #amplitude
data_b=(data_low(Time_vec[210:20800]) + data_high(Time_vec[210:20800]))/2 #baseline

data_mean= np.mean(data)
data_min_1 = np.mean(data_low(Time_vec[13270:16900]))
data_min_2 = np.mean(data_low(Time_vec[16900:17500]))
data_min_3 = np.mean(data_low(Time_vec[17500:20800]))

fig, ax = plt.subplots(3,1, sharex=True)
ii=0
ax[ii].set_title("Analysis FWB (no Chad)")
ax[ii].set_ylabel("R [Sv]")

ax[ii].vlines(-5.97,0.005, 0.04, color= 'g', linewidth= 2,alpha= 0.7)
ax[ii].vlines(-5.61, 0.005, 0.04, color= 'g', linewidth=2,alpha= 0.7)
ax[ii].vlines(-5.55,0.005, 0.04, color= 'g', linewidth= 2,alpha= 0.7)

ax[ii].hlines(data_min_1 ,-5.97, -5.61, color="tab:orange")
ax[ii].hlines(data_min_2 ,-5.61, -5.55, color="tab:orange")
ax[ii].hlines(data_min_3 ,-5.55, -5.3, color="tab:orange")
ax[ii].annotate("phase 3", xy=(-5.55, 0.03), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
ax[ii].annotate("phase 1", xy=(-5.97, 0.03), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
ax[ii].annotate("phase 2", xy=(-5.61, 0.03), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
ax[ii].plot(Time_vec[210:20800]/1000 ,data[210:20800]  , color= "grey", label='e')
ax[ii].plot(Time_vec[210:20800]/1000 ,(data_b+data_a) , color= "tab:blue", alpha= 0.7,  label="upper/lower limit")
ax[ii].plot(Time_vec[210:20800]/1000 ,(data_b-data_a) , color= "tab:blue", alpha= 0.7)#high_idx, low_idx 
ax[ii].set_xlim([-6.2, -5.3])
ax[ii].set_ylim([0.005, 0.04])
ax[ii].set(frame_on=False)
# ax[ii].set(frame_on=False)
# data= 10**(-6)*PE/(60*60*24*365.25)

data= -PE/(2.5*10**12)
lmin,lmax = hl_envelopes_idx(data)
data_high= interp1d(Time_vec[lmax], data[lmax], kind='cubic')
data_low = interp1d(Time_vec[lmin], data[lmin], kind='cubic')
data_a=(-data_low(Time_vec[210:20800]) + data_high(Time_vec[210:20800]))/2 #amplitude
data_b=(data_low(Time_vec[210:20800]) + data_high(Time_vec[210:20800]))/2 #baseline

ii=1
ax[ii].set_ylabel("EP [m/yr]")

ax[ii].vlines(-5.97, 1.25, 0.75, color= 'g', linewidth= 2,alpha= 0.7)
ax[ii].vlines(-5.61, 1.25, 0.75, color= 'g', linewidth=2,alpha= 0.7)
ax[ii].vlines(-5.55,  1.25, 0.75, color= 'g', linewidth= 2,alpha= 0.7)

ax[ii].plot(Time_vec[210:20800]/1000 ,data[210:20800]  , color= "grey", label='e')
#ax[0].plot(Time_vec[210:20800]/1000 ,data_b , color= "tab:blue", linewidth= 2,  label="b")
ax[ii].plot(Time_vec[210:20800]/1000 ,(data_b+data_a) , color= "tab:blue", alpha= 0.7,  label="upper/lower limit")
ax[ii].plot(Time_vec[210:20800]/1000 ,(data_b-data_a) , color= "tab:blue", alpha= 0.7)#high_idx, low_idx 
ax[ii].set_xlim([-6.2, -5.3])
#ax[ii].set_ylim([0, 0.04])
ax[ii].set(frame_on=False)


data= RnC
Rlmin,Rlmax = hl_envelopes_idx(data)
data= -PE
Elmin,Elmax = hl_envelopes_idx(data)
# Rlmin_vec= - np.ones_like(Rlmin)
# Rlmax_vec=np.ones_like(Rlmax)
# Elmax_vec= - np.ones_like(Elmax)*0.5
# Elmin_vec=np.ones_like(Elmin)*0.5
limit =int( min(len(Rlmax), len(Elmax))-2)
lag = np.zeros(limit)
phase = np.zeros_like(lag)
ii=2
while ii <limit:
    tmp     =  np.absolute(Time_vec[Rlmax]-Time_vec[Elmax[ii]])
    ind     =  tmp.argmin()
    lag[ii] =  Time_vec[Rlmax[ind]]-Time_vec[Elmax[ii]]
    T       =  ( abs(Time_vec[Rlmax[ind-2]] - Time_vec[Rlmax[ind+2]])/4 
                +abs(Time_vec[Elmax[ii -2]] - Time_vec[Elmax[ii +2]])/4)/2
    phase[ii]= (lag[ii] /T)*2*np.pi
    if abs(phase[ii])>np.pi:
        tmp = phase[ii]
        phase[ii] = tmp - (2*np.pi*np.sign(tmp))
    ii+=1
ii=2
ax[ii].scatter(Time_vec[Elmax[0:limit]]/1000, phase/np.pi, marker= "x", s= 3,color= "tab:blue", alpha= 0.7,)
ax[ii].plot(   Time_vec[Elmax[0:limit]]/1000, phase/np.pi, color = "grey")
ax[ii].fill_between(Time_vec[Elmax[0:limit]]/1000, -0.25, 0.25, color="tab:green", alpha = 0.2)
ax[ii].fill_between(Time_vec[Elmax[0:limit]]/1000, -0.75, -1, color="tab:red", alpha = 0.2)
ax[ii].fill_between(Time_vec[Elmax[0:limit]]/1000,  0.75,  1, color="tab:red", alpha = 0.2)
ax[ii].yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
ax[ii].yaxis.set_major_locator(tck.MultipleLocator(base=1.0))
ax[ii].set(frame_on=False)
ax[ii].vlines(-5.97, -1, 1, color= 'g', linewidth= 2,alpha= 0.7)
ax[ii].vlines(-5.61, -1, 1, color= 'g', linewidth=2,alpha= 0.7)
ax[ii].vlines(-5.55, -1, 1, color= 'g', linewidth= 2,alpha= 0.7)
ax[ii].set_xlabel("Ma")
ax[ii].set_ylabel("phase shift")
#%% THIS ONE
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.ticker as tck
data= 10**(-6)*RwC/(60*60*24*365.25)
lmin,lmax = hl_envelopes_idx(data)
data_high= interp1d(Time_vec[lmax], data[lmax], kind='cubic')
data_low = interp1d(Time_vec[lmin], data[lmin], kind='cubic')
data_a=(-data_low(Time_vec[210:20800]) + data_high(Time_vec[210:20800]))/2 #amplitude
data_b=(data_low(Time_vec[210:20800]) + data_high(Time_vec[210:20800]))/2 #baseline

data_mean= np.mean(data)
data_min_1 = np.mean(data_low(Time_vec[13270:16900]))
data_min_2 = np.mean(data_low(Time_vec[16900:17500]))
data_min_3 = np.mean(data_low(Time_vec[17500:20800]))

fig, ax = plt.subplots(3,1, sharex=True)
ii=0
ax[ii].set_title("Analysis FWB (with Chad)")
ax[ii].set_ylabel("R [Sv]")

ax[ii].vlines(-5.97,0.005, 0.1, color= 'g', linewidth= 2,alpha= 0.7)
ax[ii].vlines(-5.61, 0.005, 0.1, color= 'g', linewidth=2,alpha= 0.7)
ax[ii].vlines(-5.55,0.005, 0.1, color= 'g', linewidth= 2,alpha= 0.7)

ax[ii].hlines(data_min_1 ,-5.97, -5.61, color="tab:orange")
ax[ii].hlines(data_min_2 ,-5.61, -5.55, color="tab:orange")
ax[ii].hlines(data_min_3 ,-5.55, -5.3, color="tab:orange")
ax[ii].annotate("phase 3", xy=(-5.55, 0.08), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
ax[ii].annotate("phase 1", xy=(-5.97, 0.08), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
ax[ii].annotate("phase 2", xy=(-5.61, 0.08), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
ax[ii].plot(Time_vec[210:20800]/1000 ,data[210:20800]  , color= "grey", label='e')
ax[ii].plot(Time_vec[210:20800]/1000 ,(data_b+data_a) , color= "tab:blue", alpha= 0.7,  label="upper/lower limit")
ax[ii].plot(Time_vec[210:20800]/1000 ,(data_b-data_a) , color= "tab:blue", alpha= 0.7)#high_idx, low_idx 
ax[ii].set_xlim([-6.2, -5.3])
ax[ii].set_ylim([0.005, 0.1])
ax[ii].set(frame_on=False)
# ax[ii].set(frame_on=False)
# data= 10**(-6)*PE/(60*60*24*365.25)

data= -PE/(2.5*10**12)
lmin,lmax = hl_envelopes_idx(data)
data_high= interp1d(Time_vec[lmax], data[lmax], kind='cubic')
data_low = interp1d(Time_vec[lmin], data[lmin], kind='cubic')
data_a=(-data_low(Time_vec[210:20800]) + data_high(Time_vec[210:20800]))/2 #amplitude
data_b=(data_low(Time_vec[210:20800]) + data_high(Time_vec[210:20800]))/2 #baseline

ii=1
ax[ii].set_ylabel("PE [m/yr]")

ax[ii].vlines(-5.97,  1.25, 0.75, color= 'g', linewidth= 2,alpha= 0.7)
ax[ii].vlines(-5.61,  1.25, 0.75, color= 'g', linewidth=2,alpha= 0.7)
ax[ii].vlines(-5.55,  1.25, 0.75, color= 'g', linewidth= 2,alpha= 0.7)

ax[ii].plot(Time_vec[210:20800]/1000 ,data[210:20800]  , color= "grey", label='e')
#ax[0].plot(Time_vec[210:20800]/1000 ,data_b , color= "tab:blue", linewidth= 2,  label="b")
ax[ii].plot(Time_vec[210:20800]/1000 ,(data_b+data_a) , color= "tab:blue", alpha= 0.7,  label="upper/lower limit")
ax[ii].plot(Time_vec[210:20800]/1000 ,(data_b-data_a) , color= "tab:blue", alpha= 0.7)#high_idx, low_idx 
ax[ii].set_xlim([-6.2, -5.3])
#ax[ii].set_ylim([0, 0.04])
ax[ii].set(frame_on=False)


data= RwC
Rlmin,Rlmax = hl_envelopes_idx(data)
data= -PE
Elmin,Elmax = hl_envelopes_idx(data)
# Rlmin_vec= - np.ones_like(Rlmin)
# Rlmax_vec=np.ones_like(Rlmax)
# Elmax_vec= - np.ones_like(Elmax)*0.5
# Elmin_vec=np.ones_like(Elmin)*0.5
limit =int( min(len(Rlmax), len(Elmax))-2)
lag = np.zeros(limit)
phase = np.zeros_like(lag)
ii=2
while ii <limit:
    tmp     =  np.absolute(Time_vec[Rlmax]-Time_vec[Elmax[ii]])
    ind     =  tmp.argmin()
    lag[ii] =  Time_vec[Rlmax[ind]]-Time_vec[Elmax[ii]]
    T       =  ( abs(Time_vec[Rlmax[ind-2]] - Time_vec[Rlmax[ind+2]])/4 
                +abs(Time_vec[Elmax[ii -2]] - Time_vec[Elmax[ii +2]])/4)/2
    phase[ii]= (lag[ii] /T)*2*np.pi
    if abs(phase[ii])>np.pi:
        tmp = phase[ii]
        phase[ii] = tmp - (2*np.pi*np.sign(tmp))
    ii+=1
ii=2
ax[ii].scatter(Time_vec[Elmax[0:limit]]/1000, phase/np.pi, marker= "x", s= 3,color= "tab:blue", alpha= 0.7,)
ax[ii].plot(   Time_vec[Elmax[0:limit]]/1000, phase/np.pi, color = "grey")
ax[ii].fill_between(Time_vec[Elmax[0:limit]]/1000, -0.25, 0.25, color="tab:green", alpha = 0.2)
ax[ii].fill_between(Time_vec[Elmax[0:limit]]/1000, -0.75, -1, color="tab:red", alpha = 0.2)
ax[ii].fill_between(Time_vec[Elmax[0:limit]]/1000,  0.75,  1, color="tab:red", alpha = 0.2)
ax[ii].yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
ax[ii].yaxis.set_major_locator(tck.MultipleLocator(base=1.0))
ax[ii].set(frame_on=False)
ax[ii].vlines(-5.97, -1, 1, color= 'g', linewidth= 2,alpha= 0.7)
ax[ii].vlines(-5.61, -1, 1, color= 'g', linewidth=2,alpha= 0.7)
ax[ii].vlines(-5.55, -1, 1, color= 'g', linewidth= 2,alpha= 0.7)
ax[ii].set_xlabel("Ma")
ax[ii].set_ylabel("phase shift")
#%%
# #%% Phase between evapo and river influx
# import scipy
# from scipy.signal import correlate
# #moving phaseshift 
# data1= (10**(-6)/(60*60*24*365.25))*RnC
# data2= -(10**(-6)/(60*60*24*365.25))*PE

# tmp=data1- np.mean(data1)
# D1=tmp/max(abs(tmp))
# tmp=data2- np.mean(data2)
# D2=tmp/max(abs(tmp))
# # D1 =(data1- data1.mean())/data1.std()
# # D2 =(data2- data2.mean())/data2.std()
# dt= 100
# steps = 300
# T0= 210
# T1= T0+steps
# DT= np.zeros_like(Time_vec)
# while T1 < 20800:
    
#     data1= D1
#     data2= (10**(-6)/(60*60*24*365.25))*PE[T0:T1]

#     DATA1 =D1[T0:T1]
#     DATA2 =D2[T0:T1]
    
#     xcorr = correlate(DATA1 , DATA2)
#     dt = np.arange(1-steps, steps)
    
#     idx= int((T0+T1)/2)
#     DT[idx]=abs( dt[xcorr.argmax()])
#     #DT[idx]=dt[xcorr.argmax()]
#     T0+=1
#     T1=T0+steps
# T0= 210    
# T1= 20800
# data1= (10**(-6)/(60*60*24*365.25))*RnC[T0:T1]
# data2= (10**(-6)/(60*60*24*365.25))*PE[T0:T1]
    
# DT1= DT/9
# DT2= DT/12

# # second version

# data= RnC
# Rlmin,Rlmax = hl_envelopes_idx(data)
# data= PE
# Elmin,Elmax = hl_envelopes_idx(data)

# max_len= min(len(Rlmax), len(Elmax))
# min_len= min(len(Rlmin), len(Elmin))
# DT_max= np.nan * np.ones(shape=(max_len-1))
# ind_max= np.ones(shape=(max_len-1), dtype= int )
# DT_min= np.nan * np.ones(shape=(max_len-1))
# ind_min= np.ones(shape=(max_len-1), dtype= int )
# ind=0
# while ind< max_len-1:
#     DT_max[ind] = abs((Time_vec[Rlmax[ind]]- Time_vec[Elmin[ind]])/10)-10
#     ind_max[ind]     = int((Rlmax[ind] + Elmin[ind])/2)
#     DT_min[ind] = abs((Time_vec[Rlmin[ind]]- Time_vec[Elmax[ind]])/10)-10
#     ind_min[ind]     = int((Rlmin[ind] + Elmax[ind])/2)
    
#     ind+=1
    

# f, ax = plt.subplots(3,1, sharex=True)
# ax[0].set_title("phase lag [kyr]")
# ax[0].plot(Time_vec[210:20800]/1000 ,D1[210:20800]  , color= "blue", label='RnC')
# ax[0].plot(Time_vec[210:20800]/1000 ,D2[210:20800] , color= "grey", label='EP')
# ax[0].set_ylabel("Sv")

# ax[1].set_ylabel("pi")
# ax[1].set_xlabel("Ma")
# ax[1].plot(Time_vec[210:20800]/1000,DT1[210:20800]/10)
# ax[1].plot(Time_vec[210:20800]/1000,DT2[210:20800]/10)

# ax[2].plot(Time_vec[ind_max]/1000,DT_max)
# ax[2].plot(Time_vec[ind_min]/1000,DT_min)
# ax[2].set_xlim([-6.2, -5.3])
#%% Third try
# # plotting position of peaks on 1
# # plotting position of minima on 0
# data= RnC
# Rlmin,Rlmax = hl_envelopes_idx(data)
# data= -PE
# Elmin,Elmax = hl_envelopes_idx(data)
# Rlmin_vec= - np.ones_like(Rlmin)
# Rlmax_vec=np.ones_like(Rlmax)
# Elmax_vec= - np.ones_like(Elmax)*0.5
# Elmin_vec=np.ones_like(Elmin)*0.5

# plt.figure()
# for rr in Rlmax[0:-1]:
#     plt.vlines(Time_vec[rr]/1000, 1, -0.5, color= "grey", alpha =0.5)
# for rr in Rlmin[0:-1]:
#     plt.vlines(Time_vec[rr]/1000, 0.5, -1, color= "grey", alpha =0.5)
# plt.scatter(Time_vec[Elmin]/1000, Elmin_vec, color="tab:orange")
# plt.scatter(Time_vec[Elmax]/1000, Elmax_vec, color="tab:orange")
# plt.scatter(Time_vec[Rlmin]/1000, Rlmin_vec, color="grey")
# plt.scatter(Time_vec[Rlmax]/1000, Rlmax_vec, color="grey")
# plt.xlim([-5.55, -5.33])

#%% another try
# calculate the distance to the closest peak and divide by period of that cycle

from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.ticker as tck
data= RnC
Rlmin,Rlmax = hl_envelopes_idx(data)
data= -PE
Elmin,Elmax = hl_envelopes_idx(data)
# Rlmin_vec= - np.ones_like(Rlmin)
# Rlmax_vec=np.ones_like(Rlmax)
# Elmax_vec= - np.ones_like(Elmax)*0.5
# Elmin_vec=np.ones_like(Elmin)*0.5
limit =int( min(len(Rlmax), len(Elmax))-2)
lag = np.zeros(limit)
phase = np.zeros_like(lag)
ii=2
while ii <limit:
    tmp     =  np.absolute(Time_vec[Rlmax]-Time_vec[Elmax[ii]])
    ind     =  tmp.argmin()
    lag[ii] =  Time_vec[Rlmax[ind]]-Time_vec[Elmax[ii]]
    T       =  ( abs(Time_vec[Rlmax[ind-2]] - Time_vec[Rlmax[ind+2]])/4 
                +abs(Time_vec[Elmax[ii -2]] - Time_vec[Elmax[ii +2]])/4)/2
    phase[ii]= (lag[ii] /T)*2*np.pi
    if abs(phase[ii])>np.pi:
        tmp = phase[ii]
        phase[ii] = tmp - (2*np.pi*np.sign(tmp))
    ii+=1
    
fig, ax = plt.subplots()
ax.scatter(Time_vec[Elmax[0:limit]]/1000, phase/np.pi)
ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
ax.yaxis.set_major_locator(tck.MultipleLocator(base=1.0))
ax.set_xlim([-5.973, -5.33])
#%%
data_mean= np.mean(data)
data_min_1 = np.mean(data_low(Time_vec[13270:16900]))
data_min_2 = np.mean(data_low(Time_vec[16900:17500]))
data_min_3 = np.mean(data_low(Time_vec[17500:20800]))

plt.figure()
plt.title("reconstructed river inflow (no Chad)")
plt.ylabel("Sv")
plt.xlabel("Ma")
plt.vlines(-5.97,0, 0.1, color= 'g', linewidth= 3,alpha= 0.7)
plt.vlines(-5.61, 0, 0.1, color= 'g', linewidth= 3,alpha= 0.7)
plt.vlines(-5.55,0, 0.1, color= 'g', linewidth= 3,alpha= 0.7)

plt.hlines(data_min_1 ,-5.97, -5.61, color="tab:orange")
plt.hlines(data_min_2 ,-5.61, -5.55, color="tab:orange")
plt.hlines(data_min_3 ,-5.55, -5.3, color="tab:orange")
plt.annotate("phase 3.1", xy=(-5.55, 0.08), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
plt.annotate("phase 1", xy=(-5.97, 0.08), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
plt.annotate("phase 2", xy=(-5.61, 0.08), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
plt.plot(Time_vec[210:20800]/1000 ,data[210:20800]  , color= "tab:blue",  label='e')
plt.plot(Time_vec[210:20800]/1000 ,-((10**(-6)/(60*60*24*365.25))*PE[210:20800]+0.07), color= "grey" )
plt.annotate("mean flux ={:.3f} Sv".format(data_mean), xy=(-5.75, 0.095), xycoords="data",
                  va="center", ha="center",rotation=0,
                  bbox=dict(boxstyle="round", fc="w", color='grey'))
plt.xlim([-5.97, -5.33])
plt.ylim([0, 0.04])
plt.box(False)
#%%

data= 10**(-6)*RwC/(60*60*24*365.25)
lmin,lmax = hl_envelopes_idx(data)
data_high= interp1d(Time_vec[lmax], data[lmax], kind='cubic')
data_low = interp1d(Time_vec[lmin], data[lmin], kind='cubic')
data_a=(data_low(Time_vec[210:20800]) - data_high(Time_vec[210:20800]))/2 
data_b=(data_low(Time_vec[210:20800]) + data_high(Time_vec[210:20800]))/2 

data_mean= np.mean(data)
# data_min_1 = np.mean(data_low(Time_vec[13270:16900]))
# data_min_2 = np.mean(data_low(Time_vec[16900:17500]))
# data_min_3 = np.mean(data_low(Time_vec[17500:20800]))

plt.figure()
plt.title("reconstructed river inflow (with Chad)")
plt.ylabel("Sv")
plt.xlabel("Ma")
plt.vlines(-5.97,0, 0.1, color= 'g', linewidth= 3,alpha= 0.7)
plt.vlines(-5.61, 0, 0.1, color= 'g', linewidth= 3,alpha= 0.7)
plt.vlines(-5.55,0, 0.1, color= 'g', linewidth= 3,alpha= 0.7)
plt.hlines(data_min_1 ,-5.97, -5.61, color="tab:orange")
plt.hlines(data_min_2 ,-5.61, -5.55, color="tab:orange")
plt.hlines(data_min_3 ,-5.55, -5.3, color="tab:orange")
plt.annotate("phase 3.1", xy=(-5.55, 0.08), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
plt.annotate("phase 1", xy=(-5.97, 0.08), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
plt.annotate("phase 2", xy=(-5.61, 0.08), xycoords="data",
                  va="center", ha="center",rotation=90,
                  bbox=dict(boxstyle="round", fc="w", color='g'))
plt.plot(Time_vec[210:20800]/1000 ,data[210:20800]  , color= "grey", label='e')
plt.plot(Time_vec[210:20800]/1000 ,data_b , color= "tab:blue", linewidth= 2,  label="b")
plt.plot(Time_vec[210:20800]/1000 ,(data_b+data_a) , color= "tab:blue", alpha= 0.7,  label="upper/lower limit")
plt.plot(Time_vec[210:20800]/1000 ,(data_b-data_a) , color= "tab:blue", alpha= 0.7)#high_idx, low_idx 
plt.annotate("mean flux ={:.2f} Sv".format(data_mean) , xy=(-5.75, 0.095), xycoords="data",
                  va="center", ha="center",rotation=0,
                  bbox=dict(boxstyle="round", fc="w", color='grey'))
plt.xlim([-6.2, -5.3])
plt.ylim([0, 0.1])

plt.box(False)

#%% NEW
# #%% net evaporation rate
# f, ax= plt.subplots(5,1, sharex= True)  
# i=0 
# ax[i].plot(Time_vec/1000 ,PERnC/(2.5*10**12)  , label='PERnC_r ')
# ax[i].plot(Time_vec/1000 ,PERnC_bb/(2.5*10**12) , label="net evapo rate, baseline2")
# ax[i].plot(Time_vec/1000 ,(PERnC_bb+PERnC_ab)/(2.5*10**12) , label="net evapo rate, upper")
# ax[i].plot(Time_vec/1000 ,(PERnC_bb-PERnC_ab)/(2.5*10**12) , label="net evapo rate, lower")
# ax[i].legend()
# ax[i].vlines(-7.2, min(PERnC/(2.5*10**12) ), max(PERnC/(2.5*10**12) ), color= 'r')
# i=1
# ax[i].plot(Time_vec/1000 ,PERnC/(2.5*10**12) , label="net evapo rate")
# ax[i].legend()
# ax[i].vlines(-7.2, min(PERnC/(2.5*10**12) ), max(PERnC/(2.5*10**12) ), color= 'r')
# i=2
# ax[i].plot(Time_vec/1000 ,PERnC_b/(2.5*10**12) , label="net evapo rate, baseline")
# ax[i].plot(Time_vec/1000 ,PERnC_bb/(2.5*10**12) , label="net evapo rate, baseline2")
# ax[i].legend()
# ax[i].vlines(-7.2, min(PERnC_b/(2.5*10**12)), max(PERnC_b/(2.5*10**12)), color= 'r')
# i=3
# ax[i].plot(Time_vec/1000 ,PERnC_a/(2.5*10**12) , label="net evapo rate, amplitude")
# ax[i].plot(Time_vec/1000 ,PERnC_ab/(2.5*10**12) , label="net evapo rate, amplitude2")
# ax[i].legend()
# ax[i].vlines(-7.2, min(PERnC_a/(2.5*10**12)), max(PERnC_a/(2.5*10**12)), color= 'r')
# i=4
# ax[i].plot(Time_vec/1000 ,abs(PERnC_a/PERnC_b) , label="net evapo rate, ratio")
# ax[i].plot(Time_vec/1000 ,abs(PERnC_ab/PERnC_bb) , label="net evapo rate, ratio2")
# ax[i].legend()
# ax[i].vlines(-7.2, min(abs(PERnC_a/PERnC_b) ), max(abs(PERnC_a/PERnC_b) ), color= 'r')


# PERnC_b = uniform_filter1d(data, size=frame, mode='nearest' )
# TMP=abs(data-PERnC_b)
# PERnC_a = uniform_filter1d(TMP , size=frame, mode='nearest' )
# data = PERwC
# PERwC_b = uniform_filter1d(data, size=frame, mode='nearest' )
# TMP=abs(data-PERwC_b)
# PERwC_a = uniform_filter1d(TMP , size=frame, mode='nearest' )
# fwb_min=min(min(PERwC),min(PERnC))
# fwb_max=max(max(PERwC),max(PERnC))
# a_min=min(min(PERwC_a),min(PERnC_a))
# a_max=max(max(PERwC_a),max(PERnC_a))
# b_min=min(min(PERwC_b),min(PERnC_b))
# b_max=max(max(PERwC_b),max(PERnC_b))
# r_min=min(min(PERwC_a/PERwC_b),min(PERnC_a/PERnC_b ))
# r_max=max(max(PERwC_a/PERwC_b),max(PERnC_a/PERnC_b ))

# f, ax = plt.subplots(4, 1, sharex='col', sharey='row', figsize=(40,12), dpi=200)
# ax[0].plot(Time_vec/1000 , PERnC , label='PERnC ')
# ax[0].plot(Time_vec/1000 , PERwC , label='PERwC ')
# ax[0].vlines(-7.2, fwb_min, fwb_max)
# ax[0].vlines(-6.8, fwb_min, fwb_max)
# ax[0].vlines(-6.4, fwb_min, fwb_max)
# ax[0].vlines(-6, fwb_min, fwb_max)
# ax[0].legend(fontsize=10, loc=1)
# ax[1].plot(Time_vec/1000 , PERnC_b , label='PERnC_b ')
# ax[1].plot(Time_vec/1000 , PERwC_b , label='PERwC_b ')
# ax[1].vlines(-7.2, b_min, b_max)
# ax[1].vlines(-6.8, b_min, b_max)
# ax[1].vlines(-6.4, b_min, b_max)
# ax[1].vlines(-6, b_min, b_max)
# ax[1].legend(fontsize=10, loc=1)
# ax[2].plot(Time_vec/1000 , PERnC_a , label='PERnC_a ')
# ax[2].plot(Time_vec/1000 , PERwC_a , label='PERwC_a ')
# ax[2].vlines(-7.2, a_min, a_max)
# ax[2].vlines(-6.8, a_min, a_max)
# ax[2].vlines(-6.4, a_min, a_max)
# ax[2].vlines(-6, a_min, a_max)
# ax[2].legend(fontsize=10, loc=1)
# ax[3].plot(Time_vec/1000 , PERnC_a/PERnC_b  , label='PERnC_r ')
# ax[3].plot(Time_vec/1000 , PERwC_a/PERwC_b  , label='PERwC_r ')
# ax[3].vlines(-7.2, r_min, r_max)
# ax[3].vlines(-6.8, r_min, r_max)
# ax[3].vlines(-6.4, r_min, r_max)
# ax[3].vlines(-6, r_min, r_max)
# ax[3].legend(fontsize=10, loc=4)
# ax[3].invert_xaxis()
#plt.savefig('fwb_Simon2017_ananeu')

##%% Fourier transfor of signal
#Fs=0.01 #sampling rate [1/yrs]
#n = len(Precession) # length of the signal
#k = np.arange(n)
#T = n/Fs
#frq = k/T # two sides frequency range
#frq = frq[:len(frq)//2] # one side frequency range
#
#Y = np.fft.fft(Precession)/n # dft and normalization
#Y = Y[:n//2]
#
#plt.plot(frq,abs(Y)) # plotting the spectrum
#plt.xlabel('Freq ')
#plt.ylabel('|Y(freq)|')
##%% second try
#f_s=0.01 #sampling rate [1/yrs]
#x= Precession
#X = fftpack.fft(x)
#freqs = fftpack.fftfreq(len(x)) * f_s
#
#fig, ax = plt.subplots()
#
#ax.stem(freqs, np.abs(X))
#ax.set_xlabel('Frequency in 1/yr')
#ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
#ax.set_xlim(0, 1/18000)
#ax.grid()
##ax.set_ylim(-5, 110)\
#%% Add derivative
data= PERnC/(2.5*10**12)
dt =100#yr

deriv=np.zeros_like(data)
for ii in range(0, len(data)-2):
    deriv[ii]=(data[ii]-data[ii+1])/dt
    
plt.figure()
plt.plot(Time_vec,deriv*1000)
test=np.zeros_like(data)
test[0]=data[0]
for ii in range(0, len(data)-2):
    test[ii+1]=test[ii]-deriv[ii+1]*dt
    
plt.figure()
plt.plot(data)
plt.plot(test)

f, ax= plt.subplots(2,1, sharex= True) 
ax[0].set_title("reconstructed net-evaporation rate change")
ax[0].set_ylabel("m/yr")
ax[1].set_ylabel("a-b-ratio [-]")
ax[1].set_xlabel("Ma")
ax[0].vlines(-7.17, min(data/(2.5*10**12) )*1.1, max(data/(2.5*10**12)  )*0.9, color= 'g', linewidth= 3,alpha= 0.7)
ax[0].plot(Time_vec[200:20805]/1000 ,deriv[200:20805]/(2.5*10**12)  , color= "grey", label='e')
ax[0].set_xlim([-7.3, -6.5])
ax[0].set_ylim([min(data/(2.5*10**12) )*1.1, max(data/(2.5*10**12)  )*0.9])
ax[1].set_ylim([min((data_a)/(data_b))*0.9, max((data_a)/(data_b) )*1.1])
ax[1].vlines(-7.17, min((data_a)/(data_b))*0.9, max((data_a)/(data_b) )*1.1, color= 'g', linewidth= 3,alpha= 0.7)
ax[1].plot(Time_vec[200:20805]/1000 ,(data_a)/(data_b) ,color= "tab:orange" , label="net evapo rate, ratio")#high_idx, low_idx 
ax[0].set(frame_on=False)
ax[1].set(frame_on=False)
f.subplots_adjust(hspace=0.0)
plt.savefig("FWB_Analsysis_noChad.png")
