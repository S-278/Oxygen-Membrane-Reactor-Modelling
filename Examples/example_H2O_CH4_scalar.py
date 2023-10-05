 # -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:37:31 2022

@author: k.bittner
"""

from OMR_model import Simulate_OMR
import numpy as np

'''
Example using scalar input data
Feed side gas: Steam
Sweep side gas: Methane
'''

#Model input data
T = 900         #Temperature (°C)
N_f0 = 0.2      #Feed gas molar flow rate (mol/min)
x_f0 = 'H2O:1'  #Feed gas composition
P_f = 101325    #Feed gas pressure (Pa)
N_s0 = 0.05     #Sweep gas molar flow rate (mol/min)
x_s0 = 'CH4:1'  #Sweep gas composition
P_s = 101325    #Sweep gas pressure (Pa)
A_mem = 200     #Active membrane area (cm^2)
sigma = 1       #Ambipolar conductivity (S/m)
L = 100         #Membrane thickness (mum)
Lc = 0          #Membrane characteristic length (mum)

#Model
N_f, x_f, p_o2_f, N_s, x_s, p_o2_s, j_o2, dH, x_comp, conv = Simulate_OMR(T,N_f0,x_f0,P_f,N_s0,x_s0,P_s,A_mem,sigma,L,Lc)


#Calculate additional results
i_CH4 = x_comp.index('CH4')                     #Index of CH4
i_CO = x_comp.index('CO')                       #Index of CO
i_H2O = x_comp.index('H2O')                     #Index of CO
dCH4 = (N_s0-N_s*[item[i_CH4] for item in x_s]) #Inlet-outlet CH4 flow difference
dH2O = (N_f0-N_f*[item[i_H2O] for item in x_f]) #Inlet-outlet H2O flow difference
CH4_conv = dCH4/N_s0                            #CH4 conversion
H2O_conv = dH2O/N_f0                            #H2O conversion
N_CO_s = N_s*[item[i_CO] for item in x_s]       #CO production
s_CO = N_CO_s/dCH4                              #CO selecitivity
H2_prod = 2 * j_o2                              #Hydrogen production rate on feed side


#Print results
print("\n--------------Results--------------\n")
print("Oxygen flux    = {:10.4f} mol/min".format(j_o2[0])) 
print("H2 production  = {:10.4f} mol/min\n".format(H2_prod[0])) 
print("CO selecvitivy = {:10.4f} %".format(s_CO[0]*100)) 
print("H2O conversion = {:10.4f} %".format(H2O_conv[0]*100))
print("CH4 conversion = {:10.4f} %\n".format(CH4_conv[0]*100))
print("lg(pO2_f/bar)  = {:10.4f}".format(np.log10(p_o2_f[0]/1e+5)))
print("lg(pO2_s/bar)  = {:10.4f}\n".format(np.log10(p_o2_s[0]/1e+5)))
print("Reaction heat  = {:10.4f} W".format(dH[0]))  