# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:48:09 2022

@author: Kai Bittner
"""
import cantera as ct
import numpy as np
from scipy.optimize import fsolve
import copy
import math
#Load GRI-Mech 3.0 mechanism
sol1 = ct.Solution('gri30.yaml')
sol2 = ct.Solution('gri30.yaml')

CONV_THRESHOLD = 1
"""
Minimum conversion coefficient output by root solver
considered to be a "converged" solution. 
"""

class Experiment:
    """Object-oriented user interface for Simulate_OMR
    
    An Experiment object encapsulates the inputs and outputs
    for an OMR simulation. The idea is to keep all the input/output
    parameters in one place for tidier code and easier interactive usage.
    
    An Experiment stores three kinds of data:
        - Model inputs. These are set by the user and passed to Simulate_OMR.
        - Model outputs. These are returned by Simulate_OMR and can be read 
          but not set by the user.
        - Analyzed outputs. This is data computed from the model outputs.
          Like the model outputs, it can be read but not set by the user.
          
    Workflow for using Experiment objects:
        1. Create an Experiment object. Model inputs can be passed as keyword 
           arguments to the constructor, or set on the object after calling 
           the constructor.
        2. Call the run method on the Experiment object to produce model 
           outputs.
        3. Call the analyze method on the Experiment object to produce analyzed 
           outputs.
           
    Example usage:
        experiments = []
        for temperature in numpy.linspace(900, 1000):
            # Initializing model inputs by passing keyword arguments to constructor
            experiments.append(Experiment(T=temperature)) 
        experiments[0].print_input()
        for e in experiments:
            # Direct access to model inputs as attributes
            e.P_f = 1.1 * 101325
        for e in experiments: 
            # Typical workflow
            e.run()
            e.analyze()
            e.print_analysis()
        # Direct access to analyzed outputs as attributes
        H2O_conversion = [e.H2O_conv for e in experiments]
           
    Public methods and variables:
        input_origin -- Origin point of the model parameter space. The default 
        constructor uses this as the model input. Can be modified at runtime
        to affect all future calls to the constructor.
        __init__() -- constructor
        print_input() -- print the model inputs
        run() -- call Simulate_OMR with the model inputs and record the model 
        outputs
        analyze() -- compute analyzed outputs from model outputs
        print_analysis() -- print analyzed outputs
        grid() -- generate a dense meshgrid of Experiment objects, used for 
        sampling a parameter space
    """
    
    input_origin = {
        'T' : 950, 
        'N_f0' : 3.985e-4, 'x_f0' : 'H2O:1', 'P_f' : 101325, 
        'N_s0' : 2.989e-4, 'x_s0' : 'CH4:1', 'P_s' : 101325, 
        'A_mem' : 2.41, 'sigma' : 1.3, 'L' : 250, 'Lc' : 0}
    
    def __init__(self, **kwargs):
        """Construct an Experiment object and set model inputs

        Parameters
        ----------
        **kwargs : float
            The user can optionally specify model inputs to initialize the 
            Experiment object by passing keyword arguments with the name of the 
            argument matching the name of the input variable.

        Returns
        -------
        Experiment object with initialized model inputs.

        """
        self.__model_input = copy.deepcopy(self.input_origin)
        self.__model_input.update(kwargs)
        
    def __set_input(self, key, val):
        try: del self.__model_output
        except AttributeError: pass
        try: del self.__analyzed_output
        except AttributeError: pass
        self.__model_input[key] = val
        
    def __add_input_property(name):
        return property(
            fget = lambda self: self.__model_input[name],
            fset = lambda self, newVal: self.__set_input(name, newVal))
        
    T = __add_input_property('T')
    N_f0 = __add_input_property('N_f0')
    x_f0 = __add_input_property('x_f0')
    P_f = __add_input_property(('P_f'))
    N_s0 = __add_input_property('N_s0')
    x_s0 = __add_input_property('x_s0')
    P_s = __add_input_property('P_s')
    A_mem = __add_input_property('A_mem')
    sigma = __add_input_property('sigma')
    L = __add_input_property('L')
    Lc = __add_input_property('Lc')
    
    def print_input(self):
        """Print model inputs
        
        Returns
        -------
        None.

        """
        col_template = '{: <15}{: >20}'; col_sep = ', '
        print(f'{"Reactor properties":~^50}')
        print(col_template.format('Temperature:', f'{self.T:.0f} °C'))
        print(col_template.format('Feed:', f'{self.N_f0:.2e}' + ' mol/min'), 
              col_sep, '{: >15}'.format(self.x_f0), sep='')
        print(col_template.format('Sweep:', f'{self.N_s0:.2e}' + ' mol/min'), 
              col_sep, '{: >15}'.format(self.x_s0), sep='')
        print(col_template.format('Feed pressure:', f'{self.P_f:.0f} Pa'))
        print(col_template.format('Sweep pressure:', f'{self.P_s:.0f} Pa'))
        print(f'{"Membrane properties":~^50}')
        col_template = '{: <15}{: >10}'
        print(col_template.format('Area:', str(self.A_mem) + ' cm²'), 
              col_sep, col_template.format('sigma: ', str(self.sigma) + ' S/m'), sep='')
        print(col_template.format('Thickness: ', str(self.L) + ' um'),
              col_sep, col_template.format('char. length:', str(self.Lc) + ' um'), sep='')
            
    def __get_output(self, key):
        if not hasattr(self, "_Experiment__model_output"):
            self.run()
        return self.__model_output[key]
            
    def __add_output_property(name):
        return property(
            fget = lambda self: self.__get_output(name),
            fset = None)
            
    N_f = __add_output_property('N_f')
    x_f = __add_output_property('x_f')
    p_o2_f = __add_output_property('p_o2_f')
    N_s = __add_output_property('N_s')
    x_s = __add_output_property('x_s')
    p_o2_s = __add_output_property('p_o2_s')
    N_o2 = __add_output_property('N_o2')
    dH = __add_output_property('dH')
    x_comp = __add_output_property('x_comp')
    conv = __add_output_property('conv')
    
    def run(self):
        """Run OMR model and record model outputs
        
        After any change of model inputs (including after initialization), 
        this method must be called before accessing model outputs or analyzed 
        outputs.
        
        Raises
        ------
        RuntimeError
            Raised when model fails to converge (convergence threshold 
            specified by CONV_THRESHOLD).

        Returns
        -------
        None.

        """
        self.__model_output = {}
        self.__model_output['N_f'], self.__model_output['x_f'], self.__model_output['p_o2_f'], \
        self.__model_output['N_s'], self.__model_output['x_s'], self.__model_output['p_o2_s'], \
        self.__model_output['N_o2'], self.__model_output['dH'], \
        self.__model_output['x_comp'], self.__model_output['conv'] \
        = Simulate_OMR(
            self.T,
            self.N_f0, self.x_f0, self.P_f,
            self.N_s0, self.x_s0, self.P_s,
            self.A_mem, self.sigma, self.L, self.Lc)
        for key in self.__model_output:
            if key != 'x_comp':
                self.__model_output[key] = self.__model_output[key][0]
        if self.conv < CONV_THRESHOLD:
            temp_conv = self.conv
            del self.__model_output
            raise RuntimeError(f'Simulation failed to converge with {temp_conv}')
         
    def __get_analysis(self, key):
        if not hasattr(self, "_Experiment__analyzed_output"):
            self.analyze()
        return self.__analyzed_output[key]
         
    def __add_analysis_property(name):
        return property(
            fget = lambda self: self.__get_analysis(name),
            fset = None)

    f_H2_prod = __add_analysis_property('f_H2_prod')
    s_H2_prod = __add_analysis_property('s_H2_prod')
    s_CO_prod = __add_analysis_property('s_CO_prod')
    s_CO2_prod = __add_analysis_property('s_CO2_prod')
    H2O_conv = __add_analysis_property('H2O_conv')
    CH4_conv = __add_analysis_property('CH4_conv')
    CO_sel = __add_analysis_property('CO_sel')
    O2_conv = __add_analysis_property('O2_conv')
            
    def analyze(self):
        """Compute analyzed outputs from model outputs
        
        After any change of model inputs (including after initialization), 
        this method must be called after run() and before accessing model 
        analyzed outputs.
        
        Returns
        -------
        None.

        """
        if not hasattr(self, "_Experiment__model_output"):
            self.run()
        self.__analyzed_output = {}
        self.__analyzed_output['f_H2_prod'] = self.x_f[self.x_comp.index("H2")] * self.N_f
        self.__analyzed_output['s_H2_prod'] = self.x_s[self.x_comp.index("H2")] * self.N_s
        self.__analyzed_output['s_CO_prod'] = self.x_s[self.x_comp.index("CO")] * self.N_s
        self.__analyzed_output['s_CO2_prod'] = self.x_s[self.x_comp.index("CO2")] * self.N_s
        self.__analyzed_output['H2O_conv'] = ( self.N_f0 - self.x_f[self.x_comp.index("H2O")] * self.N_f ) / self.N_f0
        self.__analyzed_output['CH4_conv'] = ( self.N_s0 - self.x_s[self.x_comp.index("CH4")] * self.N_s ) / self.N_s0
        self.__analyzed_output['CO_sel'] = self.s_CO_prod / (self.s_CO_prod + self.s_CO2_prod)
        self.__analyzed_output['O2_conv'] = (1 - (self.x_s[self.x_comp.index("O2")] / self.N_o2))
            
    def print_analysis(self):
        """Print analyzed outputs
        
        Returns
        -------
        None.

        """
        if not hasattr(self, "_Experiment__analyzed_output"):
            self.analyze()        
        col_template = '{: <20}{: >20}'; #col_sep = ', '
        print(col_template.format('Feed H2 produced:', f'{self.f_H2_prod:.2e} mol/min'))
        print(col_template.format('H2O conversion:', f'{self.H2O_conv:.0%}'))
        print(f'Sweep syngas produced: {self.s_H2_prod:.2e} mol/min H2 + {self.s_CO_prod:.2e} mol/min CO ({self.s_H2_prod/self.s_CO_prod:.2f}:1)')
        print(col_template.format('CH4 conversion:', f'{self.CH4_conv:.0%}'))
        print(col_template.format('CO selectivity:', f'{self.CO_sel:.0%}'))
        print(col_template.format('Sweep O2 conversion:', f'{self.O2_conv:.0%}'))
        print(col_template.format('Reaction heat:', f'{self.dH:.2f} W'))
        print(col_template.format('Oxygen flux:', f'{self.N_o2:.2e} mol/min'))
        
    def __print_stream(self, stream_fractions, stream_flow):
        x_sort_indices = np.flip(np.argsort(stream_fractions))[0:9]
        col_template = '{: <7}{: >20}{: >10}'
        for idx in x_sort_indices:
            print(col_template.format(
                self.x_comp[idx] + ':', 
                f'{stream_fractions[idx]*stream_flow:.2e} mol/min', 
                f'({stream_fractions[idx]:.1%})'))

    def print_feed_output(self):
        self.__print_stream(self.x_f, self.N_f)
            
    def print_sweep_output(self):
        self.__print_stream(self.x_s, self.N_s)

    def grid(**kwargs):
        """Generate meshgrid of Experiments
        
        Given input variables and their ranges, this method generates an 
        ND-array of Experiments to explore the entire specified parameter
        space. For each variable specified as a keyword argument, the returned
        array of Experiments will have an axis along which Experiments are 
        initialized with the values of the input variable.

        Parameters
        ----------
        **kwargs : ndarray
            Each keyword argument should be given as var=numpy.array(...), 
            where var specifies the name of a model input parameter and is set 
            to an ordered, regularly spaced array of values (e.g. created with
            numpy.linspace or numpy.arange). 

        Returns
        -------
        ret_arr : ndarray
            Array of Experiment objects. The order of indices is the same as 
            the order of keyword arguments.
            
        Example usage
        -------------
        exs = Experiment.grid(T=numpy.array([900,950,1000]),
                              P_f=numpy.array([101325, 1.1*101325, 1.2*101325]),
                              P_s=numpy.array([0.8*101325, 0.9*101325, 101325]))
        exs[2, 0, 1].T == 1000
        exs[2, 0, 1].P_f == 101325
        exs[2, 0, 1].P_s == 0.9*101325

        """
        shape = tuple((len(axis) for axis in kwargs.values()))
        ret_arr = np.empty(shape, dtype=Experiment)
        
        coords_1D = [np.arange(0,len(axis)) for axis in kwargs.values()]
        flat_coords = [coord_arr.flat for coord_arr in 
                       np.meshgrid(*coords_1D, copy=False)]
        
        flat_inputs = [coord_arr.flat for coord_arr in 
                       np.meshgrid(*kwargs.values(), copy=False)]
        
        
        for input_point,arr_point in zip(zip(*flat_inputs), zip(*flat_coords)):
            init_dict = {input_var:val for input_var,val in zip(kwargs.keys(), input_point)}
            ret_arr[arr_point] = Experiment(**init_dict)
            
        return ret_arr
    
class Experiment_T_dep_sigma(Experiment):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __T_to_sigma_map(cls, T: float) -> float:
        return 151.9*math.exp(-76/(8.3e-3*(T+273.15)))
        
    sigma = property(
        fget=lambda self:self.__T_to_sigma_map(self.T),
        fset=None)

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