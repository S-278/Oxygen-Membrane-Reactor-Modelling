# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:48:09 2022

@author: Kai Bittner
"""
import cantera as ct
import numpy as np
from scipy.optimize import fsolve
#Load GRI-Mech 3.0 mechanism
sol1 = ct.Solution('gri30.yaml')
sol2 = ct.Solution('gri30.yaml')

class ModelInputFrame:
    def __init__(self, T=0, 
                 N_f0 = 0, x_f0 = "", P_f = 0, 
                 N_s0 = 0, x_s0 = "", P_s = 0, 
                 A_mem = 0, sigma = 0, L = 0,Lc = 0):
        self.T = T
        self.N_f0 = N_f0
        self.x_f0 = x_f0
        self.P_f = P_f
        self.N_s0 = N_s0
        self.x_s0 = x_s0
        self.P_s = P_s
        self.A_mem = A_mem
        self.sigma = sigma
        self.L = L
        self.Lc = Lc
        
class ModelOutputFrame:
    def __init__(self):
        self.N_f = 0
        self.x_f = np.zeros(53)
        self.p_o2_f = 0
        self.N_s = 0
        self.x_s = np.zeros(53)
        self.p_o2_s = 0
        self.N_o2 = 0
        self.dH = 0
        self.x_comp = []
        self.conv = 0

class Experiment:
    origin : ModelInputFrame = ModelInputFrame(
        T=950, 
        N_f0 = 3.985e-4, x_f0 = "H2O:1", P_f = 101325, 
        N_s0 = 2.989e-4, x_s0 = "CH4:1", P_s = 101325, 
        A_mem = 2.41, sigma = 1.3, L = 250, Lc = 0)
    
    def __init__(self, model_input : ModelInputFrame = ModelInputFrame()):
        self.model_input : ModelInputFrame = model_input
        self.model_output : ModelOutputFrame = ModelOutputFrame()
        self.__output_valid = False

def Simulate_OMR(T, N_f0, x_f0, P_f, N_s0, x_s0, P_s, A_mem, sigma, L, Lc):
    """
    Parameters
    ----------
    T : Float (scalar or array with length n)
        Temperature in °C.
    N_f0 : Float (scalar or array with length n)
        Initial feed gas molar flow rate in mol/min.
    x_f0 : String or list of strings with length n
        Initial feed gas mole fractions specified as a string in the format 'A:x_A_f0, B:x_B_f0, C:x_C_f0, ...', where x_A_f0, x_B_f0, x_C_f0 are the mole fractions of the respective species.
    P_f : Float (scalar or array with length n)
        Feed gas pressure in Pa.
    N_s0 : Float (scalar or array with length n)
        Initial sweep gas molar flow rate in mol/min.
    x_s0 : String or list of strings with length n
        Initial sweep gas mole fractions specified as a string in the format 'A:x_A_s0, B:x_B_s0, C:x_C_s0, ...', where x_A_s0, x_B_s0, x_C_s0 are the mole fractions of the respective species.
    P_s : Float (scalar or array with length n)
        Sweep gas pressure in Pa.
    A_mem : Float (scalar or array with length n)
        Active membrane area in cm^2.
    sigma : Float (scalar or array with length n)
        Ambipolar conductivity in S/m.
    L : Float (scalar or array with length n)
        Membrane thickness in mum.
    Lc : Float (scalar or array with length n)
        Characteristic length in mum.

    Returns
    -------
    N_f : Array of floats with length n
        Feed gas molar flow rate in mol/min.
    x_f : List of array of floats with size n*53
        Mole fraction array, consisting of the mole fractions of the feed gas, related to the composition array.
    p_o2_f : Float (array with length n)
        Feed gas oxygen partial pressure in Pa.
    N_s : Float (array with length n)
        Sweep gas molar flow rate in mol/min.
    x_s : List of array of floats with size n*53
        Mole fraction array, consisting of the mole fractions of the sweep gas, related to the composition array.
    p_o2_s : oat (array with length n)
        Sweep gas oxygen partial pressure in Pa.
    p_o2_f : Float (array with length n)
        Feed gas oxygen partial pressure in Pa.
    N_o2 : Float (array with length n)
        Molar oxygen flux through the membrane in mol/min.
    dH : List of array of floats with size n*53
        Outlet-Inlet enthalpy flow difference (reaction heat) in W; If positive: The reaction is endothermic; If negative: The reaction is exothermic.
    x_comp : List of strings with size 53
        Composition array consisting of the considered species in the calculation.
    conv : Integer (array with length n)
        Check whether convergence was achieved; Equal to 1 if converged.
    """
    
    #Convert to array if values are given as scalars
    T, N_f0, x_f0, P_f, N_s0, x_s0, P_s, A_mem, sigma, L, Lc = ConvertToArray(T, N_f0, x_f0, P_f, N_s0, x_s0, P_s, A_mem, sigma, L, Lc)
    #Initialize output arrays
    N_f, x_f, p_o2_f, N_s, x_s, p_o2_s, j_o2, N_o2, dH, conv = InitializeOutputs(n=len(T)) 

    
    for i in range(0,len(T)):
        #Initialize sweep and feed gas mixtures
        mix_f,mix_s, H_in = InitializeMix(T[i],N_f0[i],x_f0[i],P_f[i],N_s0[i],x_s0[i],P_s[i])
        #Decompose sweep and feed gas mixtures into atomic components
        a_H_f,a_O_f,a_Ar_f,a_N_f,a_C_f = Decompose(mix_f)
        a_H_s,a_O_s,a_Ar_s,a_N_s,a_C_s = Decompose(mix_s)
        #Solve the gouverning equations
        j_o2[i],mix_f,mix_s,conv[i] = Solve(mix_f,mix_s,sigma[i],A_mem[i],L[i],Lc[i],a_H_f,a_O_f,a_Ar_f,a_N_f,a_C_f,a_H_s,a_O_s,a_Ar_s,a_N_s,a_C_s)
        #Calculate output values 
        x_comp,p_o2_f[i],p_o2_s[i],N_f[i],N_s[i],N_o2[i],x_f[i],x_s[i], dH[i] = CalculateOutputs(j_o2[i],mix_f,mix_s, H_in, A_mem[i])
        #Print progress
        prog = (i+1)/len(T)*100
        if len(T)>1:
            print('%.0f' % prog, '%')
    if not np.all(conv == 1):
        print("Warning: Non converged solution detected! Check conv array.")
    return(N_f, x_f, p_o2_f, N_s, x_s, p_o2_s, N_o2, dH, x_comp, conv)


def ConvertToArray(T, N_f0, x_f0, P_f, N_s0, x_s0, P_s, A_mem, sigma, L, Lc):
    T = np.atleast_1d(T)
    N_f0 = np.atleast_1d(N_f0)
    x_f0 = np.atleast_1d(x_f0)
    P_f = np.atleast_1d(P_f)
    N_s0 = np.atleast_1d(N_s0)
    x_s0 = np.atleast_1d(x_s0)
    P_s = np.atleast_1d(P_s)
    A_mem = np.atleast_1d(A_mem)
    sigma = np.atleast_1d(sigma)
    L = np.atleast_1d(L)
    Lc = np.atleast_1d(Lc)
    return(T, N_f0, x_f0, P_f, N_s0, x_s0, P_s, A_mem, sigma, L, Lc)


def InitializeOutputs(n):
    N_f = np.zeros(n)
    x_f = [None]*n
    p_o2_f = np.zeros(n)
    N_s = np.zeros(n)
    x_s = [None]*n
    p_o2_s = np.zeros(n)
    N_o2 = np.zeros(n)
    j_o2 = np.zeros(n)
    dH = np.zeros(n)
    conv = np.zeros(n)
    return(N_f, x_f, p_o2_f, N_s, x_s, p_o2_s, j_o2, N_o2,dH, conv)


def InitializeMix(T,N_f0,x_f0,P_f,N_s0,x_s0,P_s):
    #create feed side mixture
    sol1.TPX = T+273.15, P_f, x_f0
    mix_f = ct.Mixture([(sol1,N_f0/60)])
    #create sweep side mixture
    sol2.TPX = T+273.15, P_s, x_s0
    mix_s = ct.Mixture([(sol2,N_s0/60)])
    #calculate inlet enthalpy flow
    H_in=(mix_f.phase(0).enthalpy_mole*mix_f.phase_moles()[0]+mix_s.phase(0).enthalpy_mole*mix_s.phase_moles()[0])/1000
    return(mix_f,mix_s, H_in)
    

def Decompose(mix):
    a_H=0
    a_O=0
    a_Ar=0
    a_N=0
    a_C=0
    for i in range(0, len(mix.species_names)):
       a_H = a_H+mix.species_moles[i]*mix.n_atoms(i, 'H')
       a_O = a_O+mix.species_moles[i]*mix.n_atoms(i, 'O')
       a_Ar = a_Ar+mix.species_moles[i]*mix.n_atoms(i, 'Ar')
       a_N = a_N+mix.species_moles[i]*mix.n_atoms(i, 'N')
       a_C = a_C+mix.species_moles[i]*mix.n_atoms(i, 'C')
    return(a_H,a_O,a_Ar,a_N,a_C)


def Solve(mix_f,mix_s,sigma,A_mem,L,Lc,a_H_f,a_O_f,a_Ar_f,a_N_f,a_C_f,a_H_s,a_O_s,a_Ar_s,a_N_s,a_C_s):
    def EquationsToSolve(x):
        #Avoid errors
        j_o2_v = x.item()
        if j_o2_v == 0:
            j_o2_v = 1e-100
        #Initial state
        mix_f.species_moles = 'H:'+str(a_H_f)+',O:'+str(a_O_f-2*j_o2_v*A_mem*1e-4)+',AR:'+str(a_Ar_f)+',N:'+str(a_N_f)+',C:'+str(a_C_f)
        mix_s.species_moles = 'H:'+str(a_H_s)+',O:'+str(a_O_s+2*j_o2_v*A_mem*1e-4)+',AR:'+str(a_Ar_s)+',N:'+str(a_N_s)+',C:'+str(a_C_s)
        #Calculate equilibrium
        mix_f.equilibrate('TP')
        mix_s.equilibrate('TP')
        #Oxygen partial pressures
        p_o2_f = mix_f.species_moles[mix_f.species_index(0,'O2')]/mix_f.phase_moles()*mix_f.P
        p_o2_s = mix_s.species_moles[mix_s.species_index(0,'O2')]/mix_s.phase_moles()*mix_s.P
        #Avoid errors
        if p_o2_f<=0:
            p_o2_f = 1e-100
        if p_o2_s<=0:
            p_o2_s = 1e-100
        #Wagner equation in SI units
        f1 = 8.31446261815324*(mix_f.T)*sigma/(16*96485.3321233100184**2*(L+2*Lc)*1e-6)*np.log(p_o2_f/p_o2_s)/j_o2_v-1
        return f1
    #Solve equations
    j_o2,info,conv,msg =  fsolve(EquationsToSolve, x0=1e-10, full_output=1,maxfev=1000, xtol=1e-12)
    return(j_o2,mix_f,mix_s,conv)


def CalculateOutputs(j_o2,mix_f,mix_s,H_in, A_mem):
    #Composition array
    x_comp=mix_f.species_names
    #Partial pressure
    p_o2_f = mix_f.species_moles[mix_f.species_index(0,'O2')]/mix_f.phase_moles()*mix_f.P
    p_o2_s = mix_s.species_moles[mix_s.species_index(0,'O2')]/mix_s.phase_moles()*mix_s.P
    #Flow rates
    N_o2 = j_o2*A_mem*1e-4*60
    N_f=mix_f.phase_moles()[0]*60
    N_s=mix_s.phase_moles()[0]*60
    #Mole fraction array
    x_f=mix_f.species_moles/mix_f.phase_moles()
    x_s=mix_s.species_moles/mix_s.phase_moles()
    #Calculate Outlet-Inlet enthalpy flow difference
    H_out = (mix_f.phase(0).enthalpy_mole*mix_f.phase_moles()[0]+mix_s.phase(0).enthalpy_mole*mix_s.phase_moles()[0])/1000
    dH = H_out - H_in
    return(x_comp,p_o2_f,p_o2_s,N_f,N_s, N_o2,x_f,x_s,dH)