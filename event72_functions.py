# -*- coding: utf-8 -*-
"""
Created on Fry May 20 13:37:06 2022

@author: ronjaebner

definitions of functions for model of the 7.2 collaboration

"""
import numpy as np
from numpy import pi



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

def safe_matrices(var,name_dir, scn_name, name_var):
    name_file= (name_dir + '72event_' + scn_name
            + "_"+ name_var)
    np.savetxt(name_file, var,delimiter=",")
    return