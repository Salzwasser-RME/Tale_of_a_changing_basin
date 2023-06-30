# -*- coding: utf-8 -*-
"""
Created on Fry May 20 16:41:27 2022

@author: ronjaebner

Script to investigate the sudden change in sensitivity of various parameters 
in the Mediterranean Sea around 7.2 Ma (7.2 event)

It uses a convection driven Mediterranean Sea and as a 3box
model

Inputs can be defined as cyclic, by setting the A_ values to 0 the input
becomes linear over time.
The strait efficiency parameter g can be defined as changing over time (3 states).

The results are plotted and analysed by calculating the amplitudes over time.

-----------------------------------------------------------------------------
This script is used to plot developement over time. No data is saved,
only figures.

# define input
# for input=const. set A_ to 0
# for gradual increase set B: to np.linspace(start, end, t_max)
"""

# import the libraries and modules
import           numpy   as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

scenario_name = 'Example'
dir_fig       = '../FIGURES/'
#%% geometry

T = 20_000
runtime= 10
dt =2
steps= T*runtime/dt
t_vec= np.linspace(0,T*runtime, num=int(steps))
yr2sc = 24*60*60*365.25

DT =dt*yr2sc

A0      = 2.5*10**12
D0      = 1500
Dint    =  500
f       = 0.2
V    = np.array([f*A0*Dint,(1-f)*A0*Dint, A0*(D0-Dint) ])
A    = np.array([f*A0,(1-f)*A0, A0 ])


SA  =  36
SH  = 350

kappa_mix   = 1*10**(-4)
kappa_conv  = 8*10**(-4)
d_mix       = 0.5*D0
A1c         = 0.9

# fwb taken from Simon et al., 2017
B_E  = 0.8625           # m/yr,
A_E1  = B_E*0.11#0.075     # m/yr, amplitude evapo, before
A_E2  = B_E*0.11#0.15     # m/yr, amplitude evapo, after



g_begin   = 1*pow(10,6)#1*pow(10,6)
g_end     = 1*pow(10,5)#1*pow(10,5) 

T1=int(3*T/dt)
T3=int(3*T/dt)
T2=int(runtime*T/dt-(T1+T3))
T4=int(7*T/dt)

# Definition 
#EP     = B_E  + np.sin((2*np.pi*inp.dt/inp.T)*inp.t_vec)*A_E
EP     = B_E  + np.hstack((np.sin((2*np.pi/T)*t_vec[0   :T4])*A_E1,
                           np.sin((2*np.pi/T)*t_vec[T4-1:-1])*A_E2 ))

# linear change in g
g_vec  =   np.hstack((np.linspace(g_begin, g_begin,T1),
                      np.linspace(g_begin, g_end  ,T2),
                      np.linspace(g_end  , g_end  ,T3)))

# logarithmic change in g
# g_vec  =   np.hstack((np.logspace(np.log10(g_begin), np.log10(g_begin),T1),
#                       np.logspace(np.log10(g_begin), np.log10(g_end)  ,T2),
#                       np.logspace(np.log10(g_end)  , np.log10(g_end)  ,T3)))


#temp= np.array(3,len(t_vec))
S       = np.ones((3,len(t_vec)))*SA
F       = np.zeros((3,len(t_vec)))

#%% Run model
#S, rho, Q , F , elaps, Flux = model.box_notemp(R0,R1, EP,g_vec)
#S    = np.ones((3, int(t_max/dt)))*SA
t =0
ii=0
while ii< len(g_vec)-1:
    # Scenario A1
    
    #Q  = g_vec[ii]*(S[1,ii]-SA)
    Q  = g_vec[ii]*np.sqrt(S[1,ii]-SA)
    Qin= Q+ EP[ii]*A0/yr2sc

    #F02 = (S[0, ii]>S[2, ii])*kappa_conv*f*A0*(S[0,ii]-S[2,ii])/(SA*d_mix)
    F02 = (S[0, ii]>S[2, ii])*kappa_conv*f*A0*(S[0,ii]-S[2,ii])/(S[2,ii])
    
    F20 = (1-A1c)*F02
    F21 = A1c*F02
    F10 = A1c*F02 + EP[ii]*A0*f/yr2sc
    mix = kappa_mix*A0*(1-f)*(S[2, ii]-S[1, ii])/d_mix

    S[0,ii+1]= min(SH, S[0,ii] + (F10*S[1,ii] + F20*S[2,ii] 
                                  - F02*S[0,ii])*DT/V[0])
    S[1,ii+1]= min(SH, S[1,ii] + (Qin*SA + F21*S[2, ii] 
                                  -(F10 + Q)*S[1, ii]  
                                  + mix)*DT/V[1])
    S[2,ii+1]= min(SH, S[2,ii] + (F02*S[0,ii] 
                                  - (F21 + F20)*S[2,ii] 
                                  - mix)*DT/V[2])
    F[1,ii+1]=F02
    F[0,ii+1]= Q
    ii+=1
    t+=dt



## ANALYSE
idx0=  int(T/dt)
idx1= -int(T/dt)
avS= (S[2,idx0:idx1]*V[2]+ S[1,idx0:idx1]*V[1]+ S[0,idx0:idx1]*V[0])/sum(V)
dS20= S[2,idx0:idx1]-S[0,idx0:idx1]
dS21= S[2,idx0:idx1]-S[1,idx0:idx1]
dS01= S[0,idx0:idx1]-S[1,idx0:idx1]
fwb= -EP*A[2]/yr2sc

#
t_plot=((t_vec[idx0:idx1])-T)/1000

#%% PLOT 
c0 = "black"
c1 = "green"
c2 = "blue"
c3 = "purple"
cS1="aquamarine"
cS0="springgreen"
cS2="mediumseagreen"
cS ="green"
cF = "mediumblue"
cO = "orange"

sc_size=25
lw=4
xvalues=np.arange(t_plot[ 0]+((1/4)*T)/1000, t_plot[-1]-((1/4)*T)/1000, (T)/1000, dtype='int')
f, axes = plt.subplots(3, 1, sharex='col', sharey='row', figsize=(20, 10), dpi=200)

ax0=axes[0].twinx()
ax0.plot(np.nan, color=cF, linewidth=lw, label="freshwater budget")
ax0.plot(t_plot, g_vec[idx0:idx1], linewidth=lw, color="grey",       label='strait efficiency')
yvalues=[-0.07, -0.06]
axes[0].set_yticks(yvalues)
axes[0].set_yticklabels(yvalues,fontsize=sc_size-5)
axes[0].plot(t_plot, fwb[idx0:idx1]*10**(-6),color=cF, linewidth=lw, label="freshwater budget")
axes[0].set_ylabel("fwb [Sv]", fontsize=sc_size-2)
axes[0].tick_params(axis='y')
axes[0].set_ylim((min(fwb[idx0:idx1]*10**(-6)))*1.05,max(fwb[idx0:idx1]*10**(-6))*0.95)
axes[0].yaxis.set_label_coords(-0.055, 0.5)

yvalues=[200000, 600000, 1000000]
ax0.set_yticks(yvalues)
ax0.set_yticklabels(yvalues,fontsize=sc_size-5)
#ax0.set_ylabel("g [ Sv/$\sqrt{m^3/kg}$ ]",fontsize=sc_size-2, color="grey")
ax0.set_ylabel(" g ",fontsize=sc_size-2, color="grey")
ax0.tick_params(axis='y', colors="grey", which='both')
ax0.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
ax0.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
ax0.yaxis.set_label_coords(1.04, 0.55)
ax0.set_yscale('log')

axes[0].set_xticks(xvalues)
axes[0].set_xlim((t_plot[0],t_plot[-1]))
axes[0].grid(axis="x")
axes[0].set_title( "input", fontsize=sc_size,fontweight="bold")

ax0.legend(loc="upper center", ncol=2, fontsize=sc_size-5,framealpha= 0.75, facecolor='white', edgecolor= 'white')
#
yvalues=[30, 35, 37, 38, 42, 46, 50]
axes[1].set_yticks(yvalues)
axes[1].set_yticklabels(yvalues,fontsize=sc_size-5)
axes[1].plot(t_plot, S[0,idx0:idx1], color=cS0, label='S0', linewidth=lw)
axes[1].plot(t_plot, S[1,idx0:idx1], color=cS1, label='S1', linewidth=lw)
axes[1].plot(t_plot, S[2,idx0:idx1], color=cS2, label='S2', linewidth=lw)
axes[1].plot(t_plot, avS, color=cS, label='avS', linewidth=lw)
axes[1].set_ylabel("salinity [kg/m3]", fontsize=sc_size-2)
axes[1].legend(loc="upper center", ncol=2, fontsize=sc_size-5,framealpha= 0.5, facecolor='white', edgecolor= 'white')
axes[1].grid(axis="x")
axes[1].set_title( "evolution of salinity over time", fontsize=sc_size,fontweight="bold")
axes[1].yaxis.set_label_coords(-0.055, 0.5)
#
axes[2].plot(t_plot,F[1,idx0:idx1]*10**(-6), color=cO, label='F02', linewidth=lw)
axes[2].tick_params(axis='y')
tmpmin= int(min(F[1,idx0:idx1]*10**(-5)))*0.09
tmpmax= int(max(F[1,idx0:idx1]*10**(-5)))*0.11

yvalues=[0.9, 1, 1.1]
axes[2].set_yticks(yvalues)
axes[2].set_ylabel("conv. [Sv]", fontsize=sc_size-2)
#axes[2].set_ylim((1.5, 2.5))
axes[2].set_yticklabels(yvalues,fontsize=sc_size-5)
axes[2].grid(axis="x")
axes[2].set_title( "convection", fontsize=sc_size,fontweight="bold")

axes[2].set_xlabel(" time in [kyrs]", fontsize=sc_size-2)
axes[2].set_xticklabels(xvalues,fontsize=sc_size-5)

for ii in [0,1,2]:
    axes[ii].spines['top'].set_visible(False)
    axes[ii].spines['right'].set_visible(False)
    axes[ii].spines['bottom'].set_visible(False)
    axes[ii].spines['left'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.spines['left'].set_visible(False)

plt.savefig(dir_fig +scenario_name+ 'Salinities')
