#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:13:17 2022

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

data=np.loadtxt("LaskarEtAl2004.txt")
Time_vec    = data[:,0]

tmp=data[:,1]
Eccentricity=(tmp/max(tmp))


tmp=data[:,2]-np.mean(data[:,2])
Precession=tmp/max(tmp)


tmp=data[:,3]- np.mean(data[:,3])
Obliquity=tmp/max(tmp)



fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(Time_vec/1000 , Eccentricity, label='Eccentricity')
ax.plot(Time_vec/1000 , Precession , label='Precession')
ax.plot(Time_vec/1000 , Obliquity , label='Obliquity ')

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

data2save= [Time_vec, PE, RnC, RwC]

np.savetxt("FWB_Simon2017_new.txt", data2save ,delimiter=",")
