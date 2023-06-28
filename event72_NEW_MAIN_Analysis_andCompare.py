#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:55:00 2021

@author: ron
Main Analysis-Script for the collaboration with Francesca Bulian to investigate the
sudden change in sensitivity of various parameters in the Mediterranean Sea
around 7.2 Ma (7.2 event)
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
import           numpy   as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
#import   event72_inp     as inp
#import   event72_model   as model
import   event72_funk_0  as funk
import   event72_data    as dat


#%% Define directory
scenario_name ='NEU__SA36_compare'
dir_data= "../DATA/Output_"
dir_fig = "../figures_hres/"




yr2sc = 24*60*60*365.25
dt=1
DT =dt*yr2sc

T =20_000
AdjustCycle=7
RunCycle = 2
t_max = (AdjustCycle+RunCycle)*T
t_vec = np.arange(0,t_max/dt)*dt
t_plot  = t_vec*dt/1000
idx0    = int(AdjustCycle*T/dt)# Three cycles are used to let the model adjust itself
idx1    = -1

A0      = 2.5*10**12
D0      = 1500
Dint    =  500
f       = 0.2
V    = np.array([f*A0*Dint,(1-f)*A0*Dint, A0*(D0-Dint) ])


SA  =  36
SH  = 350

kappa_mix   = 1*10**(-4)
kappa_conv  = 8*10**(-4)
d_mix       = 0.5*D0
A1c         = 0.9
#%% extra parameters
# scenarios A1 (convective margin)

#%% define input
# for monotonous input set A_ to 0
# for gradual increase set B: to np.linspace(start, end, t_max)
# The fwb is taken from simon et al., 2017, fig.4 by taking the  minimm and
# maximum values of the time range from 7.2 to 7.1 Ma
# Since this model does not take temperature into account the influence of rivers
# is included in A_E and B_E, while the river input is 0

B_E  = 0.8625    # m/yr,
A_E  = B_E*0.11    # m/yr, amplitude evapo

B_E_minmax  =B_E*np.array([(1-0.11),(1+0.11)])    # m/yr,

B_R0  = 0*(1-f)        # m3/s mean river
A_R0  = B_R0*0.25        # m3/s amplitude river

B_R1  = 0*f          # m3/s mean river
A_R1  = B_R1*0.25         # m3/s amplitude river

g     = np.logspace(8, 3,num=200) 

EP     = (B_E  + np.sin((2*np.pi*(dt/T))*t_vec)*A_E)/yr2sc
# R0     = B_R0 + np.sin((2*np.pi*inp.dt/inp.T)*inp.t_vec)*A_R0
# R1     = B_R1 + np.sin((2*np.pi*inp.dt/inp.T)*inp.t_vec)*A_R1
# fwb= R0+R1-EP*inp.A[2]/inp.yr2sec




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
    Flux = np.zeros(int(t_max/dt))
    Qlux = np.zeros(int(t_max/dt))
    t =0
    ii=0
    while t< t_max-dt:
        # Scenario A1
        t+=dt
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
    # # referenz figure
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(t_plot, S[0,:], 'g', label='S0')
    # ax.plot(t_plot, S[1,:], 'b', label='S1')
    # ax.plot(t_plot, S[2,:], 'r', label='S2')
    # ax.vlines(idx0*dt/1000, SA, max(S[0,:]), colors='k')
    # ax.set_ylabel("Salinity [kg/m3]")
    # ax.legend(loc='best') 
    # ax.grid()
    # ax.set_ylim([SA-1,SH])
    # ax.set_title( "Evolution of Salinity over time, log(g)=" 
    #           + str(int(np.log10(gg))))
#    plt.savefig(dir_fig +'EquilibChange_' + scenario_name
#             + str(int(np.log10(gg))) +'_Salinity')
    
    
    ##  amplitude and base
    #find_ampl_base_neu(data, idx0, T, dt, t_vec):
    data=S[0,:]
    amp_S0, bas_S0, t_Sav =funk.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=S[1,:]
    amp_S1, bas_S1, t_Sav =funk.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=S[2,:]
    amp_S2, bas_S2, t_Sav =funk.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=(S[2,:]*V[2]+ S[1,:]*V[1]+ S[0,:]*V[0])/sum(V)
    amp_Sav, bas_Sav, t_Sav =funk.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=S[2,:]-S[1,:]
    amp_dS21, bas_dS21, t_dS21=funk.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=S[2,:]-S[0,:]
    amp_dS20, bas_dS20, t_dS20=funk.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=S[0,:]-S[1,:]
    amp_dS10, bas_dS10, t_dS10=funk.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=Flux
    amp_F02, bas_F02, t_F02=funk.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    #
    data=Qlux
    amp_Q, bas_Q, t_Q=funk.find_ampl_base_neu(data, idx0, T, dt, t_vec)
    
    
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
dat.safe_matrices_72(arr_S0  , dir_data, scenario_name,  '_S0'  )
dat.safe_matrices_72(arr_S1  , dir_data, scenario_name,  '_S1'  )
dat.safe_matrices_72(arr_S2  , dir_data, scenario_name,  '_S2'  )
dat.safe_matrices_72(arr_Sav , dir_data, scenario_name,  '_Sav'  )
dat.safe_matrices_72(arr_dS21, dir_data, scenario_name,  '_dS21' )
dat.safe_matrices_72(arr_dS20, dir_data, scenario_name,  '_dS20' )
dat.safe_matrices_72(arr_dS10, dir_data, scenario_name,  '_dS10' )
dat.safe_matrices_72(arr_F02 , dir_data, scenario_name,  '_F02'  )
dat.safe_matrices_72(arr_Q , dir_data, scenario_name  ,  '_Q'    )
dat.safe_matrices_72(g       , dir_data, scenario_name,  '_g'    )
fwb= np.vstack(([B_E, A_E], [B_R0, A_R0], [B_R1, A_R1]))
dat.safe_matrices_72(fwb     , dir_data, scenario_name,  '_EP-R0-R1'  )

#%%SALINITIES
fwb= EP*A0/yr2sc
data=fwb
amp_fwb, bas_fwb, t_fwb=funk.find_ampl_base_neu(data, idx0, T, dt, t_vec)
r_fwb=amp_fwb/bas_fwb


#%%  Constant FWB
for ind in [0,1]:
    T =20_000
    t_max = 6*T
    EP     = (B_E_minmax[ind])/yr2sc
    count_g=0
    for gg in g:
               
        # Run model
        #S, rho, Q , F , elaps, Flux = model.box_notemp(R0,R1, EP)
        S    = np.ones((3, int(t_max/dt)))*SA

        t =0
        ii=0
        while t< t_max-dt:
            # Scenario A1
            t+=dt
            Q  = gg*np.sqrt(S[1,ii]-SA)
            Qin= Q+ EP*A0
    
            #F02 = (S[0, ii]>S[2, ii])*kappa_conv*f*A0*(S[0,ii]-S[2,ii])/(SA*d_mix)
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
            
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot( S[0,:], 'g', label='S0')
        # ax.plot(S[1,:], 'b', label='S1')
        # ax.plot(S[2,:], 'r', label='S2')
        # ax.set_ylabel("Salinity [kg/m3]")
        # ax.legend(loc='best') 
        # ax.grid()
        # ax.set_ylim([SA-1,SH])
        # ax.set_title( "Evolution of Salinity over time, log(g)=" 
        #           + str(int(np.log10(gg))))
        # plt.savefig(dir_fig +'EquilibChange_' + scenario_name
        #          + str(int(np.log10(gg))) +'_Salinity')
            
    
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