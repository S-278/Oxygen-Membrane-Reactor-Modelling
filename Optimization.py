# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:24:38 2024

@author: elijah.bakaleynik
"""

from Experiment import Experiment
from abc import ABC, abstractmethod
import csv
from math import tanh
from numpy import ndarray
from scipy.optimize import Bounds, OptimizeResult; import scipy.optimize
from typing import Callable
from xarray import DataArray


XA_COORDS =  [("param", ['T', 'N_f0', 'N_s0', 'P_s'])]
"""
Array coordinate convention used throughout this module.
All DataArray objects used in with this module should be created
with coords=XA_COORDS.
"""

PROGRESS_TRACKING_DIR = 'progress_tracking\\'
"""
Directory used for saving optimizer progress
"""

def extract_opt_x(exp : Experiment) -> DataArray:
    """Extract optimization space coordinates from an Experiment    

    Parameters
    ----------
    exp : Experiment
        Experiment from which the parameters under optimization
        will be read.

    Returns
    -------
    DataArray
        Contains the optimization coordinates read from the given Experiment.

    """
    ret = DataArray(coords=XA_COORDS)
    ret.loc[dict(param="T")] = exp.T
    ret.loc[dict(param="N_f0")] = exp.N_f0
    ret.loc[dict(param="N_s0")] = exp.N_s0
    if exp.P_f != exp.P_s: raise ValueError('P_f and P_s not equal')
    ret.loc[dict(param="P_s")] = exp.P_s
    return ret

def set_opt_x(exp : Experiment, xa : DataArray):
    """Set optimization coordinates on an Experiment

    Parameters
    ----------
    exp : Experiment
        Experiment on which to set optimization parameters.
    xa : DataArray
        DataArray using XA_COORDS from which the optimization coordinates
        will be read.

    Returns
    -------
    None.

    """
    exp.T=xa.sel(param="T").item()
    exp.N_f0=xa.sel(param="N_f0").item() 
    exp.N_s0=xa.sel(param="N_s0").item()
    exp.P_s=xa.sel(param="P_s").item()
    exp.P_f=exp.P_s

def extract_opt_fixed(exp : Experiment) -> dict:
    """Extract fixed parameters from an Experiment    

    Parameters
    ----------
    exp : Experiment
        Experiment from which the parameters that are kept fixed
        during optimization will be read.

    Returns
    -------
    dict
        Contains the fixed parameters read from the given Experiment.

    """
    ret = dict(x_f0=exp.x_f0, x_s0=exp.x_s0,
               A_mem=exp.A_mem, sigma=exp.sigma, 
               L=exp.L, Lc=exp.Lc)
    return ret

def set_opt_fixed(exp : Experiment, d : dict):
    """Set fixed parameters on an Experiment

    Parameters
    ----------
    exp : Experiment
        Experiment on which to set fixed parameters.
    d : dict
        dict from which the fixed parameters
        will be read.

    Returns
    -------
    None.

    """
    exp.x_f0 = d['x_f0']
    exp.x_s0 = d['x_s0']
    exp.A_mem = d['A_mem']
    exp.sigma = d['sigma']
    exp.L = d['L']
    exp.Lc = d['Lc']
    
def symmetric_inner_filter(x: float, target: float, tol: float, strength: float) -> float:
    """Produces a response of ~1 at the target and a decaying response away from the target
    
    This filter uses a sum of tanh functions to produce a response close to 1
    if x is close to the target, and a gently decaying response as x strays from the target.

    Parameters
    ----------
    x : float
        Value to be filtered.
    target : float
        If x == target, this filter will return close to 1.
    tol : float
        Describes the width of the filter.
    strength : float
        Describes the strength of the decay.

    Returns
    -------
    float
        Filter response.

    """
    halfpt_lo = target-tol
    halfpt_hi = target+tol
    return -0.5 *(                                      \
                  tanh(strength * (x - halfpt_hi)) +    \
                  tanh(strength * (-1*x + halfpt_lo))   \
                 ) + 0
        
def step_filter(x: float, target: float, tol: float) -> float:
    """Passes values above a target and rejects values below the target
    
    This filter uses a single tanh function to create a smooth step response.
    The filter will be fully open if x >= target,
    and will gradually close for x < target.

    Parameters
    ----------
    x : float
        Value to be filtered.
    target : float
        If x >= target, the filter will be open.
    tol : float
        Describes the tolerance to values slightly below the target.

    Returns
    -------
    float
        Filter response.

    """
    return 0.5 * tanh(1/tol * (x - (target - tol))) + 0.5

class Optimizer(ABC):
    """Abstract base class for an Experiment optimizer
    
    Encapsulates a global optimizer algorithm operating on 
    the Experiment parameters defined by XA_COORDS.
    At initialization an evaluation function must be provided to an Optimizer
    to use as the objective function.
    Subclasses of Optimizer should select concrete optimizer algorithms
    and add an implementation for _optimizer_call.
    
    Public methods and attributes:
        eval_funct -- Experiment evaluation function used for optimization
        fixed_params -- dictionary of fixed Experiment parameters
        optimal_x -- point in optimization space where optimum was found
        __init__() -- constructor 
        create_experiment_at() -- create Experiment at point in search space
        optimize() -- run a global optimization 
    """
    
    def __init__(self, eval_funct: Callable[[Experiment], float], run_id='test', track_progress=False):
        """Construct an Optimizer and set evaluation function

        Parameters
        ----------
        eval_funct : Callable[[Experiment], float]
            Experiment evaluation function. This function must take an Experiment
            as argument, and return a scalar score evaluating the Experiment.
            Optimization will then seek to MAXIMIZE the evaluation score.
        run_id : string, optional
            String ID used when saving files. The default is 'test'.
        track_progress : bool, optional
            Whether to track the progress of the optimization in a CSV file. 
            The default is False.

        Returns
        -------
        Optimizer object.

        """
        self.eval_funct = eval_funct
        self.run_id=run_id
        self.track_progress = track_progress
        
    def create_experiment_at(self, x : DataArray) -> Experiment:
        """Create Experiment at given point in search space
        
        The returned Experiment will be initialized with the variable parameters
        contained in x, and the fixed parameters contained in this Optimizer's
        fixed_params attribute. 
        
        Parameters
        ----------
        x : DataArray
            Point in search space. Must use XA_COORDS.

        Returns
        -------
        Experiment initialized at given point in search space.

        """
        e = Experiment(**self.fixed_params)
        xa = DataArray(data=x, coords=XA_COORDS)
        set_opt_x(e, xa)
        return e
    
    def _objective_f(self, x : ndarray) -> float:
        e = self.create_experiment_at(x)
        e.run()
        e.analyze()
        # Eval funct is inverted, because scipy optimizers minimize the objective f
        return -1 * self.eval_funct(e)
    
    @abstractmethod
    def _optimizer_call(self, init_exp: Experiment, bd: Bounds) -> OptimizeResult:
        ...
        
    def optimize(self, init_exp: Experiment, bd: Bounds) -> OptimizeResult:
        """Run a global optimization
        
        The optimization uses eval_funct as the function to maximize,
        runs within the search space bounded by bd
        and whose axes are defined by XA_COORDS,
        and uses init_exp as the initial guess.

        Parameters
        ----------
        init_exp : Experiment
            Experiment initialized with the appropriate fixed parameters
            and at the point in the search space which is to be taken as
            the initial guess.
        bd : Bounds
            Bounds object specifying the search space bounds.

        Returns
        -------
        OptimizeResult returned by underlying scipy optimizer.

        """
        self.fixed_params = extract_opt_fixed(init_exp)
        
        if self.track_progress:
            self.prog_file = open(
                PROGRESS_TRACKING_DIR + self.run_id + '_progress.csv', 
                'w+', newline='')
            self.file_writer = csv.DictWriter(self.prog_file, 
                                              XA_COORDS[0][1] + ['eval'])
            self.file_writer.writeheader()
            
        retval = self._optimizer_call(init_exp, bd)
        
        if self.track_progress:
            self.prog_file.close()
            
        self.optimal_x = DataArray(data=retval.x, coords=XA_COORDS)
        return retval
        
class DIRECT_Optimizer(Optimizer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cb_count = 0
        
    def _optimizer_call(self, init_exp: Experiment, bd: Bounds) -> OptimizeResult:
        return scipy.optimize.direct(self._objective_f, bd, 
                                     eps=1e-3, locally_biased=False, 
                                     len_tol=1e-4, vol_tol=(1/5000)**5,
                                     maxfun=50000*5, maxiter=20000,
                                     callback=self.__optimizer_cb)

    def __optimizer_cb(self, xk):
        self.cb_count += 1
        if self.cb_count % 100 == 0:
            print("CB invocation:", self.cb_count)
        if self.track_progress:
            dict_to_write = {param:val for (param,val) in zip(XA_COORDS[0][1], xk)}
            dict_to_write['eval'] = self.eval_funct(self.create_experiment_at(xk))
            self.file_writer.writerow(dict_to_write)


class DE_Optimizer(Optimizer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cb_count = 0

    def _optimizer_call(self, init_exp: Experiment, bd: Bounds) -> OptimizeResult:                    
        return scipy.optimize.differential_evolution(self._objective_f, bd,
                                                     x0=extract_opt_x(init_exp),
                                                     maxiter=int(40000/(120*5)-1), 
                                                     popsize=120, init='sobol',
                                                     atol=0.003,
                                                     mutation=(1,1.4), recombination=0.6,
                                                     callback=self.__optimizer_cb,
                                                     polish=False,
                                                     updating='immediate')
        
    def __optimizer_cb(self, xk, convergence):
        self.cb_count += 1
        print("CB invocation: " + str(self.cb_count) + ", convergence: " + str(convergence))
        if self.track_progress:
            dict_to_write = {param:val for (param,val) in zip(XA_COORDS[0][1], xk)}
            dict_to_write['eval'] = self.eval_funct(self.create_experiment_at(xk))
            self.file_writer.writerow(dict_to_write)

if __name__ == "__main__":
    RUN_ID = ''
    
    e_init = Experiment(T=1500, 
                        N_f0=4e-4, x_f0="H2O:1", P_f=101325,
                        N_s0=3e-5, x_s0="CH4:1", P_s=0.1*101325,
                        A_mem=1, sigma=0.4, L=500
                        )
    from ProcessModel import *
    import os
    pm = Spec_N_o2_PM()
    optimizer = DE_Optimizer(pm.eval_experiment, run_id=RUN_ID, track_progress=True)
    m = Metrics()
    lb = DataArray(
        data=[800, 2e-5, 101325*0.8, 2e-5, 101325*0.1],
        coords=XA_COORDS)
    ub = DataArray(
        data=[1500, 5e-4, 101325*1.2, 5e-4, 101325*1.5],
        coords=XA_COORDS)
    print("+++++++++++++++++ RUN " + RUN_ID + " ++++++++++++++++++")
    print(f'PID: {os.getpid()}')
    print("Starting optimizer...")
    res = optimizer.optimize(e_init, Bounds(lb, ub))
    print(res)
    e_opt = optimizer.create_experiment_at(res.x)
    e_opt.print_analysis()
    print(f'E eff.: {pm.get_energy_eff(e_opt)}')
    print("+++++++++++++++++ DONE ++++++++++++++++++")