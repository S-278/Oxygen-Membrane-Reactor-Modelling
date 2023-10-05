 # -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:37:31 2022

@author: k.bittner
"""

from OMR_model import Simulate_OMR
import numpy as np

import matplotlib.pyplot as plt

'''
Example using array input data with varying sweep flow rate
Feed side gas: Steam
Sweep side gas: Methane
'''

#Model input data
n = 100

T = [900]*n                         #Temperature (°C)
N_f0 = [0.4]*n                      #Feed gas molar flow rate (mol/min)
x_f0 = ['H2O:1']*n                  #Feed gas composition
P_f = [101325]*n                    #Feed gas pressure (Pa)
N_s0 = np.linspace(0.0001,0.2,n)    #Sweep gas molar flow rate (mol/min)
x_s0 = ['CH4:1']*n                  #Sweep gas composition
P_s = [101325]*n                    #Sweep gas pressure (Pa)
A_mem = [200]*n                     #Active membrane area (cm^2)
sigma = [1]*n                       #Ambipolar conductivity (S/m)
L = [100]*n                         #Membrane thickness (mum)
Lc = [0]*n                          #Membrane characteristic length (mum)

#Model
N_f, x_f, p_o2_f, N_s, x_s, p_o2_s, N_o2, dH, x_comp, conv = Simulate_OMR(T,N_f0,x_f0,P_f,N_s0,x_s0,P_s,A_mem,sigma,L,Lc)


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
H2_prod = 2 * N_o2                              #Hydrogen production rate on feed side


#Plot results
fig = plt.figure()
plt.plot(N_s0,H2_prod)
plt.xlabel('Sweep gas flow rate ($mol/min$)')
plt.ylabel('$H_2$ production ($mol/min$)')
plt.xlim(0,max(N_s0))
plt.ylim(0)
plt.grid('on')
plt.show()
fig.savefig('H2_prod', dpi=600, bbox_inches='tight')

fig = plt.figure()
plt.plot(N_s0, H2O_conv*100, N_s0, CH4_conv*100, N_s0, s_CO*100)
plt.xlabel('Sweep gas flow rate ($mol/min$)')
plt.ylabel('Conversion and selectivity (%)')
plt.xlim(0,max(N_s0))
plt.ylim(0)
plt.grid('on')
plt.legend(["$H_2O$ conversion","$CH_4$ conversion","$CO$ selectivity"])
plt.show()
fig.savefig('Conversion_selectivity', dpi=600, bbox_inches='tight')

fig = plt.figure()
plt.plot(N_s0, np.log10(p_o2_f/1e+5), N_s0, np.log10(p_o2_s/1e+5))
plt.xlabel('Sweep gas flow rate ($mol/min$)')
plt.ylabel('$lg(p_{O_2}/bar)$')
plt.xlim(0,max(N_s0))
plt.ylim(-28,-12)
plt.grid('on')
plt.legend(["$p_{O_2,f}$","$p_{O_2,s}$"])
plt.show()
fig.savefig('p_O2', dpi=600, bbox_inches='tight')

fig = plt.figure()
plt.plot(N_s0, dH)
plt.xlabel('Sweep gas flow rate ($mol/min$)')
plt.ylabel('Required heat supply (W)')
plt.xlim(0,max(N_s0))
plt.ylim(0)
plt.grid('on')
plt.show()
fig.savefig('heat_supply', dpi=600, bbox_inches='tight')