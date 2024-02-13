# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:48:09 2022

@author: Kai Bittner, Elijah Bakaleynik

This was an attempt to rework OMR_model to use two child processes
to complete the cantera mixture equilibration at each step of the root finding
in parallel. Profiling Simulate_OMR shows that 90% of the processor time
is spent on the mixture equilibration steps. Since the two mixtures are 
equilibrated independently, it was hoped that running the equilibration
in parallel CPU processes would afford substantial speedup.
This requires establishing two child worker processes, one for the feed mixture
and one for the sweep, that act as custodians of their respective mixture objects.
The children communicate with the parent process, which is the one that runs the 
root finder, using Pipes.
However, this first attempt runs several times slower than the 
original implementation, probably due to OS overhead from the amount of data 
transferred between processes (tested on Windows 11, result may be different on Unix). 
"""
import cantera as ct
import numpy as np
from scipy.optimize import fsolve
from multiprocessing import Process, Pipe
from enum import Enum
#Load GRI-Mech 3.0 mechanism
sol1 = ct.Solution('gri30.yaml')
sol2 = ct.Solution('gri30.yaml')

CONV_THRESHOLD = 1
"""
Minimum conversion coefficient output by root solver
considered to be a "converged" solution. 
"""
    
class Chamber(Enum):
    FEED = 1
    SWEEP = 2
    
def worker(chamber, pipe_end, T, N_0, x_0, P):
    sol = None
    if chamber == Chamber.FEED:
        sol = sol1
    elif chamber == Chamber.SWEEP:
        sol = sol2
    else:
        raise ValueError()
    sol.TPX = T+273.15, P, x_0
    mix = ct.Mixture([(sol,N_0/60)])
    
    H_in = mix.phase(0).enthalpy_mole*mix.phase_moles()[0]
    pipe_end.send(H_in)
    
    pipe_end.send(Decompose(mix))
    
    for species_moles in iter(pipe_end.recv, 'STOP'):
        mix.species_moles = species_moles
        mix.equilibrate('TP')
        p_o2 = mix.species_moles[mix.species_index(0,'O2')]/mix.phase_moles()*mix.P
        pipe_end.send(p_o2)

    x_comp = mix.species_names
    pipe_end.send(x_comp)
    
    p_o2 = mix.species_moles[mix.species_index(0,'O2')]/mix.phase_moles()*mix.P
    pipe_end.send(p_o2)
    
    N = mix.phase_moles()[0]*60
    pipe_end.send(N)
    
    x = mix.species_moles/mix.phase_moles()
    pipe_end.send(x)
    
    H_out = mix.phase(0).enthalpy_mole*mix.phase_moles()[0]
    pipe_end.send(H_out)

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
        parent_end_f, child_end_f = Pipe()
        parent_end_s, child_end_s = Pipe()
        worker_f = Process(target=worker, args=(Chamber.FEED, child_end_f, T[i], N_f0[i], x_f0[i], P_f[i]))
        worker_s = Process(target=worker, args=(Chamber.SWEEP, child_end_s, T[i], N_s0[i], x_s0[i], P_s[i]))
        worker_f.start()
        worker_s.start()
        
        enthalpy_flow_f = parent_end_f.recv()
        enthalpy_flow_s = parent_end_s.recv()
        H_in = (enthalpy_flow_f+enthalpy_flow_s)/1000
        
        #Decompose sweep and feed gas mixtures into atomic components
        a_H_f,a_O_f,a_Ar_f,a_N_f,a_C_f = parent_end_f.recv()
        a_H_s,a_O_s,a_Ar_s,a_N_s,a_C_s = parent_end_s.recv()
        #Solve the gouverning equations
        j_o2[i],conv[i] = Solve(parent_end_f,parent_end_s,T[i], sigma[i],A_mem[i],L[i],Lc[i],a_H_f,a_O_f,a_Ar_f,a_N_f,a_C_f,a_H_s,a_O_s,a_Ar_s,a_N_s,a_C_s)
        #Calculate output values 
        x_comp,p_o2_f[i],p_o2_s[i],N_f[i],N_s[i],N_o2[i],x_f[i],x_s[i], dH[i] = CalculateOutputs(j_o2[i],parent_end_f,parent_end_s, H_in, A_mem[i])
        
        worker_f.join()
        worker_s.join()
        #Print progress
        prog = (i+1)/len(T)*100
        if len(T)>1:
            print('%.0f' % prog, '%')
    if not np.all(conv >= CONV_THRESHOLD):
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

def Solve(pipe_f, pipe_s, T, sigma,A_mem,L,Lc,a_H_f,a_O_f,a_Ar_f,a_N_f,a_C_f,a_H_s,a_O_s,a_Ar_s,a_N_s,a_C_s):
    def EquationsToSolve(x):
        #Avoid errors
        j_o2_v = x.item()
        if j_o2_v == 0:
            j_o2_v = 1e-100
        #Initial state
        species_moles_f = 'H:'+str(a_H_f)+',O:'+str(a_O_f-2*j_o2_v*A_mem*1e-4)+',AR:'+str(a_Ar_f)+',N:'+str(a_N_f)+',C:'+str(a_C_f)
        species_moles_s = 'H:'+str(a_H_s)+',O:'+str(a_O_s+2*j_o2_v*A_mem*1e-4)+',AR:'+str(a_Ar_s)+',N:'+str(a_N_s)+',C:'+str(a_C_s)
        #Calculate equilibrium
        pipe_f.send(species_moles_f)
        pipe_s.send(species_moles_s)
        p_o2_f = pipe_f.recv()
        p_o2_s = pipe_s.recv()
        #Avoid errors
        if p_o2_f<=0:
            p_o2_f = 1e-100
        if p_o2_s<=0:
            p_o2_s = 1e-100
        #Wagner equation in SI units
        f1 = 8.31446261815324*(T)*sigma/(16*96485.3321233100184**2*(L+2*Lc)*1e-6)*np.log(p_o2_f/p_o2_s)/j_o2_v-1
        return f1
    #Solve equations
    j_o2,info,conv,msg =  fsolve(EquationsToSolve, x0=1e-10, full_output=1,maxfev=1000, xtol=1e-12)
    pipe_f.send('STOP')
    pipe_s.send('STOP')
    return(j_o2,conv)


def CalculateOutputs(j_o2,parent_end_f,parent_end_s,H_in, A_mem):
    #Composition array
    x_comp=parent_end_f.recv(); parent_end_s.recv()
    #Partial pressure
    p_o2_f = parent_end_f.recv()
    p_o2_s = parent_end_s.recv()
    #Flow rates
    N_o2 = j_o2*A_mem*1e-4*60
    N_f=parent_end_f.recv()
    N_s=parent_end_s.recv()
    #Mole fraction array
    x_f=parent_end_f.recv()
    x_s=parent_end_s.recv()
    #Calculate Outlet-Inlet enthalpy flow difference
    H_out_f = parent_end_f.recv()
    H_out_s = parent_end_s.recv()
    H_out = (H_out_f+H_out_s)/1000
    dH = H_out - H_in
    return(x_comp,p_o2_f,p_o2_s,N_f,N_s, N_o2,x_f,x_s,dH)

def find_pO2(T, P_f, N_f0, x_f0, P_s, N_s0, x_s0, j_o2):
    sol1.TPX = T+273.15, P_f, x_f0
    mix_f = ct.Mixture([(sol1,N_f0/60)])

    sol2.TPX = T+273.15, P_s, x_s0
    mix_s = ct.Mixture([(sol2,N_s0/60)])

    a_H_f,a_O_f,a_Ar_f,a_N_f,a_C_f = Decompose(mix_f)
    a_H_s,a_O_s,a_Ar_s,a_N_s,a_C_s = Decompose(mix_s)

    mix_f.species_moles = 'H:'+str(a_H_f)+',O:'+str(a_O_f-2*j_o2)+',AR:'+str(a_Ar_f)+',N:'+str(a_N_f)+',C:'+str(a_C_f)
    mix_s.species_moles = 'H:'+str(a_H_s)+',O:'+str(a_O_s+2*j_o2)+',AR:'+str(a_Ar_s)+',N:'+str(a_N_s)+',C:'+str(a_C_s)
    
    mix_f.equilibrate('TP')
    mix_s.equilibrate('TP')
    
    p_o2_f = mix_f.species_moles[mix_f.species_index(0,'O2')]/mix_f.phase_moles()*mix_f.P
    p_o2_s = mix_s.species_moles[mix_s.species_index(0,'O2')]/mix_s.phase_moles()*mix_s.P

    return p_o2_f, p_o2_s
