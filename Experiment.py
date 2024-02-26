# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:15:35 2024

@author: elijah.bakaleynik
"""
from OMR_model import Simulate_OMR, CONV_THRESHOLD
import copy
import math
import numpy as np
import re
import pint; u=pint.UnitRegistry()

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
           
    Public methods and attributes:
        input_origin -- Origin point of the model parameter space. The default 
        constructor uses this as the model input. Can be modified at runtime
        to affect all future calls to the constructor.
        __init__() -- constructor
        print_input() -- print the model inputs
        run() -- call Simulate_OMR with the model inputs and record the model 
        outputs
        analyze() -- compute analyzed outputs from model outputs
        print_analysis() -- print analyzed outputs
        print_feed_output() and print_sweep_output() -- print stream compositions
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
        self._model_input = copy.deepcopy(self.input_origin)
        self._model_input.update(kwargs)
        
    def _set_input(self, key, val):
        try: del self._model_output
        except AttributeError: pass
        try: del self._analyzed_output
        except AttributeError: pass
        self._model_input[key] = val
        
    def _add_input_property(name):
        return property(
            fget = lambda self: self._model_input[name],
            fset = lambda self, newVal: self._set_input(name, newVal))
        
    T = _add_input_property('T')
    N_f0 = _add_input_property('N_f0')
    x_f0 = _add_input_property('x_f0')
    P_f = _add_input_property(('P_f'))
    N_s0 = _add_input_property('N_s0')
    x_s0 = _add_input_property('x_s0')
    P_s = _add_input_property('P_s')
    A_mem = _add_input_property('A_mem')
    sigma = _add_input_property('sigma')
    L = _add_input_property('L')
    Lc = _add_input_property('Lc')
    
    def print_input(self):
        """Print model inputs
        
        Returns
        -------
        None.

        """
        col_template = '{: <15}{: >20}'; col_sep = ', '
        print(f'{"Reactor properties":~^50}')
        print(col_template.format('Temperature:', f'{self.T:.0f} °C'))
        print(col_template.format('Feed:', f'{self.N_f0:.2e}' + ' mol/min')) 
        print('{: >35}'.format(self.x_f0))
        print(col_template.format('Sweep:', f'{self.N_s0:.2e}' + ' mol/min')) 
        print('{: >35}'.format(self.x_s0))
        print(col_template.format('Feed pressure:', f'{self.P_f:.0f} Pa'))
        print(col_template.format('Sweep pressure:', f'{self.P_s:.0f} Pa'))
        print(f'{"Membrane properties":~^50}')
        col_template = '{: <15}{: >10}'
        print(col_template.format('Area:', str(self.A_mem) + ' cm²'), 
              col_sep, col_template.format('sigma: ', str(self.sigma) + ' S/m'), sep='')
        print(col_template.format('Thickness: ', str(self.L) + ' um'),
              col_sep, col_template.format('char. length:', str(self.Lc) + ' um'), sep='')
            
    def _get_output(self, key):
        try:
            return self._model_output[key]
        except AttributeError:
            self.run()
            return self._model_output[key]
            
    def _add_output_property(name):
        return property(
            fget = lambda self: self._get_output(name),
            fset = None)
            
    N_f = _add_output_property('N_f')
    x_f = _add_output_property('x_f')
    p_o2_f = _add_output_property('p_o2_f')
    N_s = _add_output_property('N_s')
    x_s = _add_output_property('x_s')
    p_o2_s = _add_output_property('p_o2_s')
    N_o2 = _add_output_property('N_o2')
    dH = _add_output_property('dH')
    x_comp = _add_output_property('x_comp')
    conv = _add_output_property('conv')
    
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
        self._model_output = {}
        self._model_output['N_f'], self._model_output['x_f'], self._model_output['p_o2_f'], \
        self._model_output['N_s'], self._model_output['x_s'], self._model_output['p_o2_s'], \
        self._model_output['N_o2'], self._model_output['dH'], \
        self._model_output['x_comp'], self._model_output['conv'] \
        = Simulate_OMR(
            self.T,
            self.N_f0, self.x_f0, self.P_f,
            self.N_s0, self.x_s0, self.P_s,
            self.A_mem, self.sigma, self.L, self.Lc)
        for key in self._model_output:
            if key != 'x_comp':
                self._model_output[key] = self._model_output[key][0]
        if self.conv < CONV_THRESHOLD:
            temp_conv = self.conv
            del self._model_output
            raise RuntimeError(f'Simulation failed to converge with {temp_conv}')
         
    def _get_analysis(self, key):
        try:
            return self._analyzed_output[key]
        except AttributeError:
            self.analyze()
            return self._analyzed_output[key]

    def _add_analysis_property(name):
        return property(
            fget = lambda self: self._get_analysis(name),
            fset = None)

    f_H2_prod = _add_analysis_property('f_H2_prod')
    s_H2_prod = _add_analysis_property('s_H2_prod')
    s_CO_prod = _add_analysis_property('s_CO_prod')
    s_CO2_prod = _add_analysis_property('s_CO2_prod')
    H2O_conv = _add_analysis_property('H2O_conv')
    CH4_conv = _add_analysis_property('CH4_conv')
    CO_sel = _add_analysis_property('CO_sel')
    O2_conv = _add_analysis_property('O2_conv')
            
    def analyze(self):
        """Compute analyzed outputs from model outputs
        
        After any change of model inputs (including after initialization), 
        this method must be called after run() and before accessing model 
        analyzed outputs.
        
        Returns
        -------
        None.

        """
        if not hasattr(self, "_model_output"):
            self.run()
        self._analyzed_output = {}
        self._analyzed_output['f_H2_prod'] = self.x_f[self.x_comp.index("H2")] * self.N_f
        self._analyzed_output['s_H2_prod'] = self.x_s[self.x_comp.index("H2")] * self.N_s
        self._analyzed_output['s_CO_prod'] = self.x_s[self.x_comp.index("CO")] * self.N_s
        self._analyzed_output['s_CO2_prod'] = self.x_s[self.x_comp.index("CO2")] * self.N_s
        # Using regular expressions to parse x_f0 for the H2O fraction.
        # Two layers of indexing are needed: first to select the first match,
        # then to select the first pattern group 
        # (pattern has two groups to find floats written both with and without a leading 0)
        H2O_in = float(re.findall('H2O:(\d+(\.\d*)?|\.\d+)', self.x_f0)[0][0]) * self.N_f0
        H2O_out = self.x_f[self.x_comp.index("H2O")] * self.N_f
        self._analyzed_output['H2O_conv'] = (H2O_in - H2O_out) / H2O_in
        CH4_in = float(re.findall('CH4:(\d+(\.\d*)?|\.\d+)', self.x_s0)[0][0]) * self.N_s0
        CH4_out = self.x_s[self.x_comp.index("CH4")] * self.N_s
        self._analyzed_output['CH4_conv'] = (CH4_in - CH4_out) / CH4_in
        self._analyzed_output['CO_sel'] = self.s_CO_prod / (CH4_in * self.CH4_conv)
        self._analyzed_output['O2_conv'] = (self.N_o2 - self.x_s[self.x_comp.index("O2")]) / self.N_o2
            
    def print_analysis(self):
        """Print analyzed outputs
        
        Returns
        -------
        None.

        """
        if not hasattr(self, "_Experiment_analyzed_output"):
            self.analyze()        
        col_template = '{: <20}{: >20}'; #col_sep = ', '
        print(col_template.format('Feed H2 produced:', f'{self.f_H2_prod:.2e} mol/min'))
        print(col_template.format('H2O conversion:', f'{self.H2O_conv:.0%}'))
        print(col_template.format('Sweep syngas prod.:', f'{self.s_H2_prod:.2e} mol/min H2'))
        print(col_template.format('', f'{self.s_CO_prod:.2e} mol/min CO'))
        print(col_template.format('', f'({self.s_H2_prod/self.s_CO_prod:.2f}:1)'))
        print(col_template.format('CH4 conversion:', f'{self.CH4_conv:.0%}'))
        print(col_template.format('CO selectivity:', f'{self.CO_sel:.0%}'))
        print(col_template.format('Sweep O2 conversion:', f'{self.O2_conv:.0%}'))
        print(col_template.format('Reaction heat:', f'{self.dH:.2f} W'))
        print(col_template.format('O2 flux:', f'{self.N_o2:.2e} mol/min'))
        
    def _print_stream(self, stream_fractions, stream_flow):
        x_sort_indices = np.flip(np.argsort(stream_fractions))[0:9]
        col_template = '{: <7}{: >20}{: >10}'
        for idx in x_sort_indices:
            print(col_template.format(
                self.x_comp[idx] + ':', 
                f'{stream_fractions[idx]*stream_flow:.2e} mol/min', 
                f'({stream_fractions[idx]:.1%})'))

    def print_feed_output(self):
        """Print the top 10 constituents of the feed output
        
        Returns
        -------
        None.

        """
        self._print_stream(self.x_f, self.N_f)
            
    def print_sweep_output(self):
        """Print the top 10 constituents of the sweep output
        
        Returns
        -------
        None.

        """
        self._print_stream(self.x_s, self.N_s)

    def grid(**kwargs):
        """Generate meshgrid of Experiments
        
        Given input variables and their ranges, this method generates an 
        ND-array of Experiments to explore the entire specified parameter
        space. For each array passed in as a keyword argument, the returned
        array of Experiments will have an axis along which Experiments are 
        initialized with the values of the given array. Keyword arguments
        for which scalar values are passed in specify fixed parameters - 
        all Experiments in the returned array will be initialized with 
        the same value for each fixed parameter.
                
        Parameters
        ----------
        **kwargs : float or ndarray
            Axis keyword arguments should be given as var=numpy.array(...), 
            where var specifies the name of a model input parameter and the value 
            is set to an ordered array of values (e.g. created with
            numpy.linspace or numpy.arange). 
            Fixed parameter keyword arguments should be given as var=x,
            where var specifies the name of a model input parameter,
            and x is a float specifying what value this parameter should take
            for all Experiments in the returned array.

        Returns
        -------
        ret_arr : ndarray
            Array of Experiment objects. The order of indices is the same as 
            the order of keyword arguments.
            TODO: For some reason matplotlib and numpy disagree with the 
            order of indices defined above. I have tried reimplementing
            this method using numpy's native vectorize, but that somehow
            screws up the index ordering completely. Meanwhile, matplotlib
            expects the reverse index ordering convention, i.e. [z,y,x].
            Therefore, the return of this method should be transposed
            before being used for plots with more than 1 independent variable.
            
        Example usage
        -------------
        exs = Experiment.grid(T=numpy.array([900,950,1000]),
                              P_f=numpy.array([101325, 1.1*101325, 1.2*101325]),
                              P_s=numpy.array([0.8*101325, 0.9*101325, 101325]),
                              sigma=2)
        exs[2, 0, 1].T == 1000
        exs[2, 0, 1].P_f == 101325
        exs[2, 0, 1].P_s == 0.9*101325
        for e in exs.flat: e.sigma == 2

        """
        axes = dict(); fixed_params = dict()
        # Separate keyword arguments into fixed parameters and axes
        for (key,val) in kwargs.items():
            if isinstance(val, np.ndarray) and len(val) > 1:
                axes[key] = val
            else: fixed_params[key] = val
        
        shape = tuple((len(axis) for axis in axes.values()))
        ret_arr = np.empty(shape, dtype=Experiment)
        
        # Make a 1D array of indices for each axis
        coords_1D = [np.arange(0,len(axis)) for axis in axes.values()]
        # Put the index arrays through numpy.meshgrid,
        # then flatten the meshes back into 1D arrays. 
        flat_coords = [coord_arr.flat for coord_arr in 
                       np.meshgrid(*coords_1D, copy=False)]
        # zipping flat_coords now yields a sequence of index vectors
        # which can be used to index into ret_arr.
        # Iterating over this sequence allows one to iterate over all members
        # of ret_arr.
        
        # Put the axis arrays through numpy.meshgrid,
        # then flatten the meshes back into 1D arrays. 
        flat_inputs = [coord_arr.flat for coord_arr in 
                       np.meshgrid(*axes.values(), copy=False)]
        # zipping flat_inputs now yields a sequence of vectors in the parameter space
        # which can be used to populate Experiments in ret_arr.
        
        for input_point,arr_point in zip(zip(*flat_inputs), zip(*flat_coords)):
            # arr_point is an index vector specifying a member of ret_arr
            # input_point is a vector in the parameter space
            init_dict = {input_var:val for input_var,val in zip(axes.keys(), input_point)}
            init_dict.update(fixed_params)
            # Populate ret_arr at arr_point with an Experiment
            # set up at the point in parameter space given by input_point
            ret_arr[arr_point] = Experiment(**init_dict)
            
        return ret_arr
    
def spec_cond_to_sigma(cond: float, L: float, Lc: float=0) -> float:
    """Convert conductance per unit area to conductivity
    
    S/cm² -> S/m

    Parameters
    ----------
    cond : float
        In S/cm².
    L : float
        In um.
    Lc : float, optional
        In um. The default is 0.
    """
    cond *= u.S / u.cm**2
    L *= u.um; Lc *= u.um
    sigma = (cond * (L + Lc)).to(u.S / u.m)
    return sigma.magnitude

def sigma_to_spec_cond(sigma: float, L: float, Lc: float=0) -> float:
    """Convert conductivity to conductance per unit area
    
    S/m -> S/cm²

    Parameters
    ----------
    sigma : float
        In S/m.
    L : float
        In um.
    Lc : float, optional
        In um. The default is 0.
    """
    sigma *= u.S / u.m
    L *= u.um; Lc *= u.um
    cond = (sigma / (L + Lc)).to(u.S / u.cm**2)
    return cond.magnitude
    
class Experiment_T_dep_sigma(Experiment):
    """Subclass of Experiment with a temperature-dependent sigma
    
    This Experiment subclass can be used to simulate a membrane
    whose conductivity changes with temperature. 
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __T_to_sigma_map(cls, T: float) -> float:
        # These numbers are an Arrhenius fit to STF-35
        # from Fig.10 of https://ris.utwente.nl/ws/portalfiles/portal/107288677/structural.pdf
        return 151.9*math.exp(-76/(8.3e-3*(T+273.15)))
        
    sigma = property(
        fget=lambda self:self.__T_to_sigma_map(self.T),
        fset=None)
    
class ProfilingExperiment(Experiment):
    """Subclass of Experiment for profiling
    
    Uses a version of Simulate_OMR with more sub-functions,
    which makes it easier to analyze processor time.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        from OMR_model_profile import Simulate_OMR as Simulate_OMR_profile
        self._model_output = {}
        self._model_output['N_f'], self._model_output['x_f'], self._model_output['p_o2_f'], \
        self._model_output['N_s'], self._model_output['x_s'], self._model_output['p_o2_s'], \
        self._model_output['N_o2'], self._model_output['dH'], \
        self._model_output['x_comp'], self._model_output['conv'] \
        = Simulate_OMR_profile(
            self.T,
            self.N_f0, self.x_f0, self.P_f,
            self.N_s0, self.x_s0, self.P_s,
            self.A_mem, self.sigma, self.L, self.Lc)
        for key in self._model_output:
            if key != 'x_comp':
                self._model_output[key] = self._model_output[key][0]
        if self.conv < CONV_THRESHOLD:
            temp_conv = self.conv
            del self._model_output
            raise RuntimeError(f'Simulation failed to converge with {temp_conv}')
            
class LimitExperiment(Experiment):
    """Subclass of Experiment using thermodynamic limit
    
    Uses a version of Simulate_OMR which imposes the thermodynamic limit
    of membrane conductance.
    """

    def run(self):
        from OMR_model_lim import Simulate_OMR as Simulate_OMR_lim
        self._model_output = {}
        self._model_output['N_f'], self._model_output['x_f'], self._model_output['p_o2_f'], \
        self._model_output['N_s'], self._model_output['x_s'], self._model_output['p_o2_s'], \
        self._model_output['N_o2'], self._model_output['dH'], \
        self._model_output['x_comp'], self._model_output['conv'] \
        = Simulate_OMR_lim(
            self.T,
            self.N_f0, self.x_f0, self.P_f,
            self.N_s0, self.x_s0, self.P_s,
            self.A_mem, self.sigma, self.L, self.Lc)
        for key in self._model_output:
            if key != 'x_comp':
                self._model_output[key] = self._model_output[key][0]
        if self.conv < CONV_THRESHOLD:
            temp_conv = self.conv
            del self._model_output
            raise RuntimeError(f'Simulation failed to converge with {temp_conv}')


if __name__ == '__main__':
    e = LimitExperiment()
    e.run()
    pass