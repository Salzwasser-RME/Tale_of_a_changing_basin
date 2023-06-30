#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fry May 20 10:27:09 2022

@author: ronjaebner

Script to investigate the sudden change in sensitivity of various parameters 
in the Mediterranean Sea around 7.2 Ma (7.2 event)

It uses a convection driven Mediterranean Sea and a temperature less 3box
model

Inputs can be defined as cyclic, but by setting the B_ values to 0 the input
becomes linear over time.
The strait efficiency g can be defined as changing over time (3 states),
but it is not necessary to do that, g can also be constant.

The results are plotted and analysed by calculating the amplitudes over time.

-----------------------------------------------------------------------------
Tis script is used to run several restriction scenarios, save the output data
and plot the analysed results (amplitude, baseline, amplitude of ouput
                               vs amplitude of input)
it is important to make sure, that the runtime is set to 15*T > time > 8*T
The sytem needs a period to adjust (first three periods are not taken into 
account in the analysis), but 3 periods are enough to calculate the amplitude.

------------------------------------------------------------------------------
The analysis includes
- amplitude
- baseline
- amplitude baseline ration,, a-b-ratio
- parameter range

----
using the new version of the model
"""
#%% import the libraries and modules
import             numpy as np
import matplotlib.pyplot as plt
import event72_functions as func
from matplotlib.ticker import StrMethodFormatter

#%% Define directory
scenario_name ='Example'
dir_data= "../DATA/Output_"
dir_fig = "../FIGURES/"

#%% Defining run

# time
yr2sc = 24*60*60*365.25
dt    = 1
DT    = dt*yr2sc
T     = 20_000
AdjustCycle = 8
RunCycle    = 2
t_max   = (AdjustCycle+RunCycle)*T
t_vec   = np.arange(0,t_max/dt)*dt
t_plot  = t_vec*dt/1000
idx0    = int(AdjustCycle*T/dt)# Three cycles are used to let the model adjust itself
idx1    = -1

# geometry
A0      = 2.5*10**12
D0      = 1500
Dint    =  500
f       = 0.2
V    = np.array([f*A0*Dint,(1-f)*A0*Dint, A0*(D0-Dint) ])

# salinity
SA  =  36
SH  = 350

# processes 
kappa_mix   = 1*10**(-4)
kappa_conv  = 8*10**(-4)
d_mix       = 0.5*D0
A1c         = 0.9

#%% define input
# for monotonous input set A_ to 0
# for gradual increase set B: to np.linspace(start, end, t_max)
# The fwb is taken from simon et al., 2017, fig.4 by taking the  minimm and
# maximum values of the time range from 7.2 to 7.1 Ma
# Since this model does not take temperature into account the influence of rivers
# is included in A_E and B_E, while the river input is 0

# net-evaporation
B_E  = 0.8625       # m/yr,
A_E  = B_E*0.11     # m/yr, amplitude evapo

B_E_minmax  =  B_E*np.array([(1-0.11),(1+0.11)])    # m/yr,
EP          = (B_E  + np.sin((2*np.pi/T)*t_vec)*A_E)/yr2sc

# retruction
n_g = 100
g     = np.logspace(8, 3,num=n_g) 

#%% Define Arrays
arr_Sav = np.zeros((len(g),2))
arr_dS21= np.zeros((len(g),2))
arr_dS20= np.zeros((len(g),2))
arr_dS10= np.zeros((len(g),2))
arr_F02 = np.zeros((len(g),2))
arr_Q   = np.zeros((len(g),2))
arr_S0  = np.zeros((len(g),2))
arr_S1  = np.zeros((len(g),2))
arr_S2  = np.zeros((len(g),2))


arr_Savminmax = np.zeros((len(g),2))
arr_dS21minmax= np.zeros((len(g),2))
arr_dS20minmax= np.zeros((len(g),2))
arr_dS10minmax= np.zeros((len(g),2))
arr_F02minmax = np.zeros((len(g),2))
arr_Qminmax   = np.zeros((len(g),2))
arr_S0minmax  = np.zeros((len(g),2))
arr_S1minmax  = np.zeros((len(g),2))
arr_S2minmax  = np.zeros((len(g),2))
#%% Run model for all values of g


count_g=0
for gg in g:
           
    # Run model
    #S, rho, Q , F , elaps, Flux = model.box_notemp(R0,R1, EP)
    S    = np.ones((3, int(t_max/dt)))*SA
    Flux = np.zeros(   int(t_max/dt))
    Qlux = np.zeros(   int(t_max/dt))
    
    ii=0
    while ii< (t_max/dt)-2 :
        # Scenario A1
        
        Q  = gg*np.sqrt(S[1,ii]-SA)
        Qin= Q+ EP[ii]*A0

        #F02 = (S[0, ii]>S[2, ii])*kappa_conv*f*A0*(S[0,ii]-S[2,ii])/(SA*d_mix)
        F02 = (S[0, ii]>S[2, ii])*kappa_conv*f*A0*(S[0,ii]-S[2,ii])/(S[2,ii])
        
        F20 = (1-A1c)*F02
        F21 = A1c*F02
        F10 = A1c*F02 + EP[ii]*A0*f
        mix = kappa_mix*A0*(1-f)*(S[2, ii]-S[1, ii])/(d_mix)

        S[0,ii+1]= min(SH, S[0,ii] + (F10*S[1,ii] + F20*S[2,ii] 
                                      - F02*S[0,ii])*DT/V[0])
        S[1,ii+1]= min(SH, S[1,ii] + (Qin*SA + F21*S[2, ii] 
                                      -(F10 + Q)*S[1, ii]  
                                      + mix)*DT/V[1])
        S[2,ii+1]= min(SH, S[2,ii] + (F02*S[0,ii] 
                                      - (F21 + F20)*S[2,ii] 
                                      - mix)*DT/V[2])
        Flux[ii+1]=F02
        Qlux[ii+1]= Q
        ii+=1

#%%        

    ##  amplitude and base
    #find_ampl_base_neu(data, idx0, T, dt, t_vec):
    data=S[0,:]
    amp_S0, bas_S0, t_Sav =func.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=S[1,:]
    amp_S1, bas_S1, t_Sav =func.find_ampl_base_neu(data, idx0, T, dt, t_vec)

    data=S[2,:]
    amp_S2, bas_S2, t_Sav =func.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=(S[2,:]*V[2]+ S[1,:]*V[1]+ S[0,:]*V[0])/sum(V)
    amp_Sav, bas_Sav, t_Sav =func.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=S[2,:]-S[1,:]
    amp_dS21, bas_dS21, t_dS21=func.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=S[2,:]-S[0,:]
    amp_dS20, bas_dS20, t_dS20=func.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=S[0,:]-S[1,:]
    amp_dS10, bas_dS10, t_dS10=func.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=Flux
    amp_F02, bas_F02, t_F02=func.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=Qlux
    amp_Q, bas_Q, t_Q=func.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    
    
    ## Save data in array
    arr_S0[count_g,:] = np.mean(amp_S0) , np.mean(bas_S0) 
    arr_S1[count_g,:] = np.mean(amp_S1) , np.mean(bas_S1) 
    arr_S2[count_g,:] = np.mean(amp_S2) , np.mean(bas_S2) 
    arr_Sav[count_g,:] = np.mean(amp_Sav) , np.mean(bas_Sav) 
    arr_dS21[count_g,:]= np.mean(amp_dS21), np.mean(bas_dS21)
    arr_dS20[count_g,:]= np.mean(amp_dS20), np.mean(bas_dS20)
    arr_dS10[count_g,:]= np.mean(amp_dS10), np.mean(bas_dS10)
    arr_F02[count_g,:] = np.mean(amp_F02) , np.mean(bas_F02) 
    arr_Q[count_g,:]   = np.mean(amp_Q) , np.mean(bas_Q)
    
    
    # empty variables to avoid problems
    amp_Sav  = bas_Sav  = t_Sav  = []
    amp_dS21 = bas_dS21 = t_dS21 = []
    amp_dS20 = bas_dS20 = t_dS20 = []
    amp_F02  = bas_F02  = t_F02  = []
    
    # increase counter for next iteration
    count_g+=1
# end of g-loop


#%% Save arrays in files
func.safe_matrices(arr_S0  , dir_data, scenario_name,  '_S0'  )
func.safe_matrices(arr_S1  , dir_data, scenario_name,  '_S1'  )
func.safe_matrices(arr_S2  , dir_data, scenario_name,  '_S2'  )
func.safe_matrices(arr_Sav , dir_data, scenario_name,  '_Sav'  )
func.safe_matrices(arr_dS21, dir_data, scenario_name,  '_dS21' )
func.safe_matrices(arr_dS20, dir_data, scenario_name,  '_dS20' )
func.safe_matrices(arr_dS10, dir_data, scenario_name,  '_dS10' )
func.safe_matrices(arr_F02 , dir_data, scenario_name,  '_F02'  )
func.safe_matrices(arr_Q , dir_data, scenario_name  ,  '_Q'    )
func.safe_matrices(g       , dir_data, scenario_name,  '_g'    )
fwb= np.vstack(([B_E, A_E]))
func.safe_matrices(fwb     , dir_data, scenario_name,  '_EP-R0-R1'  )

#%%SALINITIES
fwb= EP*A0/yr2sc
data=fwb
amp_fwb, bas_fwb, t_fwb=func.find_ampl_base_neu(data, idx0, T, dt, t_vec)
r_fwb=amp_fwb/bas_fwb


#%%  Constant FWB
for ind in [0,1]:

    t_max = AdjustCycle*T
    EP     = (B_E_minmax[ind])/yr2sc
    
    count_g=0
    for gg in g:
               
        S    = np.ones((3, int(t_max/dt)))*SA

        ii=0
        
        while ii< (t_max/dt)-2 :
                        
            Q  = gg*np.sqrt(S[1,ii]-SA)
            Qin= Q+ EP*A0
    
            F02 = (S[0, ii]>S[2, ii])*kappa_conv*f*A0*(S[0,ii]-S[2,ii])/(S[2,ii])
            
            F20 = (1-A1c)*F02
            F21 = A1c*F02
            F10 = A1c*F02 + EP*A0*f
            mix = kappa_mix*A0*(1-f)*(S[2, ii]-S[1, ii])/(d_mix)
    
            S[0,ii+1]= min(SH, S[0,ii] + (F10*S[1,ii] + F20*S[2,ii] 
                                          - F02*S[0,ii])*DT/V[0])
            S[1,ii+1]= min(SH, S[1,ii] + (Qin*SA + F21*S[2, ii] 
                                          -(F10 + Q)*S[1, ii]  
                                          + mix)*DT/V[1])
            S[2,ii+1]= min(SH, S[2,ii] + (F02*S[0,ii] 
                                          - (F21 + F20)*S[2,ii] 
                                          - mix)*DT/V[2])
            ii+=1

    
        ## Save data in array
        arr_S0minmax[count_g,ind]     = S[0,-1] 
        arr_S1minmax[count_g,ind]     = S[1,-1] 
        arr_S2minmax[count_g,ind]     = S[2,-1] 
        arr_Savminmax[count_g,ind]    = (S[2,-1]*V[2]+ S[1,-1]*V[1]+ S[0,-1]*V[0])/sum(V)
        arr_dS21minmax[count_g,ind]   = S[2,-1] - S[1,-1] 
        arr_dS20minmax[count_g,ind]   = S[2,-1] - S[0,-1] 
        arr_dS10minmax[count_g,ind]   = S[1,-1] - S[0,-1] 
        arr_F02minmax[count_g,ind]    = F02
        arr_Qminmax[count_g,ind]      = Q
        
        
        # increase counter for next iteration
        count_g+=1
# end of g-loop



#%%with defined y-range
ylimit=[35, 50]
plt.rcParams['text.usetex'] = False
fs= 24

name ='arr_Sav'
data= arr_Sav
y  =  data[:,1]
y0 = data[:,1]+data[:,0]
y1 = data[:,1]-data[:,0]
r  = data[:,0]/data[:,1]
fig, ax = plt.subplots(figsize=(10,6))
ax.set_title("average salinity", fontsize= fs*1.2)
color = 'tab:blue'
ax.fill_between(g, y0, y1,color=color  , alpha=0.3)
ax.plot(g, y  , color=color )
ax.plot(g, y0 , color=color  , alpha=0.7,linestyle=':')
ax.plot(g, y1 , color=color  , alpha=0.7,linestyle=':')

ax.plot(g, arr_Savminmax[:,0] , color='grey'  , alpha=1,linestyle='--')
ax.plot(g, arr_Savminmax[:,1] , color='grey'  , alpha=1,linestyle='--')
ax.set_xlabel('g')
ax.set_xscale('log')
#ax.grid()
ax.set_ylabel(r'av. S [kg/$m^3$]', color=color, fontsize= fs)
ax.set_ylim([37, 110])
ax.set_xlim([5*10**3, 2*10**7])
ax.tick_params(axis='y', labelcolor=color)
#ax.set_xlabel('g in [Sv/sqrt(kg/m3)]', color='black')
#
color = 'tab:orange'
ax2   = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('a-b-ratio [-]', color=color, fontsize= fs)
ax2.tick_params(axis='y', labelcolor=color)
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
ax2.plot(g, r, color=color)

ylimit = [35,105]
yvalues=[40,50, 70, 100]
ax.set_yticks(yvalues)
ax.set_ylim(ylimit)

ylimit = [0.0, 0.04]
yvalues=[0.0, 0.02, 0.04]
ax2.set_yticks(yvalues)
ax2.set_ylim(ylimit)

color= 'tab:green'
ax.hlines(50, 5*10**3,2*10**7, color= color )
ax.fill_between(g, 35, 50, where=y0 < 50,
                color='tab:green', alpha=0.5)
ax.set_xlabel(r'low $\longleftarrow$ strait efficiency  $\longrightarrow$ high', color='black', fontsize= fs)
ax.spines['left'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=fs*0.8)
ax2.tick_params(axis='both', which='major', labelsize=fs*0.8)
plt.tight_layout()

# plt.savefig(dir_fig +'EquilibChange_' + scenario_name+ '_minmaxlim_' +name+ '.eps', format='eps')
plt.savefig(dir_fig +'EquilibChange_' + scenario_name+ '_minmaxlim_' +name+ '.png', format='png')
#%%
ylimit=[35, 50]
plt.rcParams['text.usetex'] = False
fs= 24

name ='arr_S1'
data= arr_S1
y  =  data[:,1]
y0 = data[:,1]+data[:,0]
y1 = data[:,1]-data[:,0]
r  = data[:,0]/data[:,1]
fig, ax = plt.subplots(figsize=(10,6))
ax.set_title("salinity open box", fontsize= fs*1.2)
color = 'tab:blue'
ax.fill_between(g, y0, y1,color=color  , alpha=0.3)
ax.plot(g, y  , color=color )
ax.plot(g, y0 , color=color  , alpha=0.7,linestyle=':')
ax.plot(g, y1 , color=color  , alpha=0.7,linestyle=':')
ax.plot(g, arr_S1minmax[:,0] , color='grey'  , alpha=1,linestyle='--')
ax.plot(g, arr_S1minmax[:,1] , color='grey'  , alpha=1,linestyle='--')
ax.set_xlabel('g')
ax.set_xscale('log')
#ax.grid()
ax.set_ylabel(r'S$_{open}$ [kg/$m^3$]', color=color, fontsize= fs)
ax.set_ylim([37, 110])
ax.set_xlim([5*10**3, 2*10**7])
ax.tick_params(axis='y', labelcolor=color)
#ax.set_xlabel('g in [Sv/sqrt(kg/m3)]', color='black')
#
color = 'tab:orange'
ax2   = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('a-b-ratio [-]', color=color, fontsize= fs)
ax2.tick_params(axis='y', labelcolor=color)
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
ax2.plot(g, r, color=color)


ylimit = [35,105]
yvalues=[40, 70, 100]
ax.set_yticks(yvalues)
ax.set_ylim(ylimit)

ylimit = [0.0, 0.04]
yvalues=[0.0, 0.02, 0.04]
ax2.set_yticks(yvalues)
ax2.set_ylim(ylimit)

ax.set_xlabel(r'low $\longleftarrow$ strait efficiency  $\longrightarrow$ high', color='black', fontsize= fs)
ax.spines['left'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=fs*0.8)
ax2.tick_params(axis='both', which='major', labelsize=fs*0.8)
plt.tight_layout()

# plt.savefig(dir_fig +'EquilibChange_' + scenario_name+ '_minmaxlim_' +name+ '.eps', format='eps')
plt.savefig(dir_fig +'EquilibChange_' + scenario_name+ '_minmaxlim_' +name+ '.png', format='png')
#%%
#plt.rcParams['text.usetex'] = True
fs= 24
name ='S2'

data= arr_S2
y  =  data[:,1]
y0 = data[:,1]+data[:,0]
y1 = data[:,1]-data[:,0]
r  = data[:,0]/data[:,1]
fig, ax = plt.subplots(figsize=(10,6))
ax.set_title("salinity deep box", fontsize= fs*1.2)
color = 'tab:blue'
ax.fill_between(g, y0, y1,color=color  , alpha=0.3)
ax.plot(g, y  , color=color )
ax.plot(g, y0 , color=color  , alpha=0.7,linestyle=':')
ax.plot(g, y1 , color=color  , alpha=0.7,linestyle=':')
ax.plot(g, arr_S2minmax[:,0] , color='grey'  , alpha=1,linestyle='--')
ax.plot(g, arr_S2minmax[:,1] , color='grey'  , alpha=1,linestyle='--')
ax.set_xlabel('g')
ax.set_xscale('log')

ax.set_ylabel('S$_{deep}$ [kg/m$^3$]', color=color, fontsize= fs)
#ax.set_ylim(ylimit)
ax.set_xlim([5*10**3, 2*10**7])
ax.tick_params(axis='y', labelcolor=color)

#
color = 'tab:orange'
ax2   = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('a-b-ratio [-]', color=color, fontsize= fs)
ax2.tick_params(axis='y', labelcolor=color)
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
ax2.plot(g, r, color=color)


ylimit = [35,105]
yvalues=[40,50, 70, 100]
ax.set_yticks(yvalues)
ax.set_ylim(ylimit)

ylimit = [0.0, 0.05]
yvalues=[0.0, 0.02, 0.04]
ax2.set_yticks(yvalues)
ax2.set_ylim(ylimit)



ax.set_xlabel('low $\longleftarrow$ strait efficiency  $\longrightarrow$ high', color='black', fontsize= fs)
ax.spines['left'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=fs*0.8)
ax2.tick_params(axis='both', which='major', labelsize=fs*0.8)
plt.tight_layout()
# plt.savefig(dir_fig +'EquilibChange_' + scenario_name+ '_minmaxlim_' +name + '.eps', format='eps')
plt.savefig(dir_fig +'EquilibChange_' + scenario_name+ '_minmaxlim_' +name + '.png', format='png')
#%%
#plt.rcParams['text.usetex'] = True
fs= 24
name ='arr_F02'

data= arr_F02*10**(-6)
y  =  data[:,1]
y0 = data[:,1]+data[:,0]
y1 = data[:,1]-data[:,0]
r  = data[:,0]/data[:,1]
fig, ax = plt.subplots(figsize=(10,6))
ax.set_title("convection", fontsize= fs*1.2)
color = 'tab:blue'
ax.fill_between(g, y0, y1,color=color  , alpha=0.3)
ax.plot(g, y  , color=color )
ax.plot(g, y0 , color=color  , alpha=0.7,linestyle=':')
ax.plot(g, y1 , color=color  , alpha=0.7,linestyle=':')

ax.plot(g, arr_F02minmax[:,0]*10**(-6) , color='grey'  , alpha=1,linestyle='--')
ax.plot(g, arr_F02minmax[:,1]*10**(-6) , color='grey'  , alpha=1,linestyle='--')
ax.set_xlabel('g')
ax.set_xscale('log')

ax.set_ylabel('$F_{extra->deep}$ [Sv]', color=color, fontsize= fs)
#ax.set_ylim(ylimit)
ax.set_xlim([5*10**3, 2*10**7])
ax.tick_params(axis='y', labelcolor=color)

#
color = 'tab:orange'
ax2   = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('a-b-ratio [-]', color=color, fontsize= fs)
ax2.tick_params(axis='y', labelcolor=color)
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
ax2.plot(g, r, color=color)

ylimit = [0.9,1.25]
yvalues=[1, 1.1, 1.2]
ax.set_yticks(yvalues)
ax.set_ylim(ylimit)

ylimit = [0.035, 0.125]
yvalues=[0.04, 0.08, 0.12]
ax2.set_yticks(yvalues)
ax2.set_ylim(ylimit)


ax.set_xlabel('low $\longleftarrow$ strait efficiency  $\longrightarrow$ high', color='black', fontsize= fs)
ax.spines['left'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=fs*0.8)
ax2.tick_params(axis='both', which='major', labelsize=fs*0.8)
plt.tight_layout()
# plt.savefig(dir_fig +'EquilibChange_' + scenario_name+ '_minmaxlim_' +name + '.eps', format='eps')
plt.savefig(dir_fig +'EquilibChange_' + scenario_name+ '_minmaxlim_' +name + '.png', format='png')

#%%
#plt.rcParams['text.usetex'] = True
fs= 24
name ='arr_dS21'

data= arr_dS21
y  =  data[:,1]
y0 = data[:,1]+data[:,0]
y1 = data[:,1]-data[:,0]
r  = data[:,0]/data[:,1]
fig, ax = plt.subplots(figsize=(10,6))
ax.set_title("vertical salinity difference", fontsize= fs*1.2)
color = 'tab:blue'
ax.fill_between(g, y0, y1,color=color  , alpha=0.3)
ax.plot(g, y  , color=color )
ax.plot(g, y0 , color=color  , alpha=0.7,linestyle=':')
ax.plot(g, y1 , color=color  , alpha=0.7,linestyle=':')

ax.plot(g, arr_dS21minmax[:,0], color='grey'  , alpha=1,linestyle='--')
ax.plot(g, arr_dS21minmax[:,1], color='grey'  , alpha=1,linestyle='--')
ax.set_xlabel('g')
ax.set_xscale('log')

ax.set_ylabel('S$_{deep} - S_{open}$ [kg/$m^3$]', color=color, fontsize= fs)
#ax.set_ylim(ylimit)
ax.set_xlim([5*10**3, 2*10**7])
ax.tick_params(axis='y', labelcolor=color)


#
color = 'tab:orange'
ax2   = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('a-b-ratio [-]', color=color, fontsize= fs)
ax2.tick_params(axis='y', labelcolor=color)
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
ax2.plot(g, r, color=color)

ylimit = [0.4,1.5]
yvalues=[0.5, 0.9, 1.4]
ax.set_yticks(yvalues)
ax.set_ylim(ylimit)

ylimit = [0.075, 0.15]
yvalues=[0.08, 0.11, 0.14]
ax2.set_yticks(yvalues)
ax2.set_ylim(ylimit)

ax.set_xlabel('low $\longleftarrow$ strait efficiency  $\longrightarrow$ high', color='black', fontsize= fs)
ax.spines['left'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=fs*0.8)
ax2.tick_params(axis='both', which='major', labelsize=fs*0.8)
plt.tight_layout()
# plt.savefig(dir_fig +'EquilibChange_' + scenario_name+ '_minmaxlim_' +name + '.eps', format='eps')
plt.savefig(dir_fig +'EquilibChange_' + scenario_name+ '_minmaxlim_' +name + '.png', format='png')
#%%
#plt.rcParams['text.usetex'] = True
fs= 24
name ='arr_dS10'

data= arr_dS10
y  =  data[:,1]
y0 = data[:,1]+data[:,0]
y1 = data[:,1]-data[:,0]
r  = data[:,0]/data[:,1]
fig, ax = plt.subplots(figsize=(10,6))
ax.set_title("horizontal salinity difference", fontsize= fs*1.2)#, loc='left')
color = 'tab:blue'
ax.fill_between(g, y0, y1,color=color  , alpha=0.3)
ax.plot(g, y  , color=color )
ax.plot(g, y0 , color=color  , alpha=0.7,linestyle=':')
ax.plot(g, y1 , color=color  , alpha=0.7,linestyle=':')

ax.plot(g, -arr_dS10minmax[:,0], color='grey'  , alpha=1,linestyle='--')
ax.plot(g, -arr_dS10minmax[:,1], color='grey'  , alpha=1,linestyle='--')
ax.set_xlabel('g')
ax.set_xscale('log')
ax.set_ylabel('S$_{extra} - S_{open}$ [kg/$m^3$]', color=color, fontsize= fs)
ax.set_xlim([5*10**3, 2*10**7])
ax.tick_params(axis='y', labelcolor=color)

#
color = 'tab:orange'
ax2   = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('a-b-ratio [-]', color=color, fontsize= fs)
ax2.tick_params(axis='y', labelcolor=color)
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
ax2.plot(g, r, color=color)
ylimit = [0.5, 2]
yvalues=[0.6, 1.2, 1.8]
ax.set_yticks(yvalues)
ax.set_ylim(ylimit)
ylimit = [0.07, 0.125]
yvalues=[0.08, 0.1, 0.12]
ax2.set_yticks(yvalues)
ax2.set_ylim(ylimit)

ax.set_xlabel('low $\longleftarrow$ strait efficiency  $\longrightarrow$ high', color='black', fontsize= fs)
ax.spines['left'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=fs*0.8)
ax2.tick_params(axis='both', which='major', labelsize=fs*0.8)
plt.tight_layout()
# plt.savefig(dir_fig +'EquilibChange_' + scenario_name+ '_minmaxlim_' +name + '.eps', format='eps')
plt.savefig(dir_fig +'EquilibChange_' + scenario_name+ '_minmaxlim_' +name + '.png', format='png')