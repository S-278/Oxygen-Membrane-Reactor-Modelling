# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:04:05 2023

@author: elijah.bakaleynik
"""

from OMR_model import Experiment
from scipy.optimize import Bounds, OptimizeResult; import scipy.optimize
from numpy import ndarray; import numpy
from xarray import DataArray
import pint; u=pint.UnitRegistry()
import copy
import matplotlib.pyplot as plt

XA_COORDS =  [("param", ["T", "N_f0", 'P_f', 'N_s0', 'P_s'])]

CH4_HHV = 55384 * u.kJ/u.kg * 16.04 * u.g/u.mol
CO_HHV = 10160 * u.kJ/u.kg * 28.01 * u.g/u.mol
H2_HHV = 142081 * u.kJ/u.kg * 2.016 * u.g/u.mol

glb_cb_ct = 0
glb_opt_intermediates = numpy.zeros((2000))

def optimize_wrapper(x0 : DataArray, bd : Bounds, f : callable) -> OptimizeResult:
    return scipy.optimize.direct(f, bd, 
                                 eps=1e-2, locally_biased=False, 
                                 maxfun=5000*5, maxiter=2000,
                                 len_tol=1e-4
                                 ,callback=optimize_cb
                                 )

def optimize_cb(xk):
    global glb_cb_ct
    global glb_opt_intermediates
    # glb_opt_intermediates[glb_cb_ct] = xk
    glb_cb_ct += 1
    if glb_cb_ct % 100 == 0:
        print("CB invocation:", glb_cb_ct)
        
class Metrics(dict):
    pass

class ProcessModel:
    def __init__(self):
        self.H2O_CYCLE_LOSS = 0.10
        self.H2O_PURIF_CONS = 4.52e-5 * u.kWh/u.mol
        self.H2O_PUMPING_CONS = 7.01e-6 * u.kWh / u.mol
        self.H2O_BOILING_CONS = 76.6 * u.kJ/u.mol
        self.H2O_PREHEAT_CONS = 33.6 * u.J/(u.mol * u.delta_degC)
        self.CH4_PUMPING_CONS = 6.34e-6 * u.kWh / u.mol
        self.CH4_PREHEAT_CONS = 35.7 * u.J/(u.mol * u.delta_degC)
        self.AMBIENT_T = u.Quantity(25, u.degC)
        self.HX_EFF = 0.9
        self.REACTOR_HEAT_LOSS = 0.10
        self.CONDENSER_CW_FLOW_RATIO = 2/1
        self.CONDENSER_CW_PUMPING_CONS = 0.7e-6 * u.kWh/u.mol
        self.CO2_SEP_E_CONS = 9 * u.kJ/u.mol
        self.CO2_SEP_H_CONS = 132 * u.kJ/u.mol
        self.RANKINE_EFF = 0.4
    
    def get_energy_eff(self, exp: Experiment, metrics : Metrics = None) -> float:  
        funct_params = copy.deepcopy(locals()); ProcessModel.__remove_builtins(funct_params)

        # External heat input:
        # Reactor heat supply (reaction heat + heat loss)
        reactor_heat_supply =                                                \
            exp.dH * u.W *                                                   \
            (1 + self.REACTOR_HEAT_LOSS/(1-self.REACTOR_HEAT_LOSS))
        # Input water boiling
        H2O_boil_cons = exp.N_f0 * u.mol/u.min * self.H2O_BOILING_CONS
        # Input water preheating
        H2O_preheat_cons =                                                  \
            exp.N_f0 * u.mol/u.min                                          \
            * (u.Quantity(exp.T,u.degC) - u.Quantity(100,u.degC))           \
            * (1-self.HX_EFF) * self.H2O_PREHEAT_CONS
        # Input methane preheating
        CH4_preheat_cons =                                                  \
            exp.N_s0 * u.mol/u.min                                          \
            * (u.Quantity(exp.T,u.degC) - self.AMBIENT_T)                   \
            * (1-self.HX_EFF) * self.CH4_PREHEAT_CONS
        # CO2 separation
        CO2_sep_heat_cons = exp.s_CO2_prod * u.mol/u.min * self.CO2_SEP_H_CONS
        
        ext_heat_cons = reactor_heat_supply                       \
                            + H2O_boil_cons + H2O_preheat_cons    \
                            + CH4_preheat_cons                    \
                            + CO2_sep_heat_cons
        
        # External electricity input:
        # Calculate how much electricity 
        # can be recovered from waste heat.
        waste_heat_recovered = exp.dH * u.W / ( (1-self.REACTOR_HEAT_LOSS)/self.REACTOR_HEAT_LOSS )
        elec_produced = waste_heat_recovered * self.RANKINE_EFF
        # Calculate total electricity consumption:
        # Input water purification
        H2O_pur_cons = exp.N_f0 * u.mol/u.min * self.H2O_CYCLE_LOSS * self.H2O_PURIF_CONS 
        # Input pumping
        H2O_pump_cons = exp.N_f0 * u.mol/u.min * self.H2O_PUMPING_CONS
        CH4_pump_cons = exp.N_s0 * u.mol/u.min * self.CH4_PUMPING_CONS
        # Condenser operation
        H2_condenser_cons =                             \
            exp.N_f * u.mol/u.min                       \
            * self.CONDENSER_CW_FLOW_RATIO              \
            * self.CONDENSER_CW_PUMPING_CONS
        CH4_condenser_cons=                                                         \
            exp.N_s * u.mol/u.min                                                   \
            * (exp.x_s[exp.x_comp.index("CH4")] + exp.x_s[exp.x_comp.index("CO")])  \
            * self.CONDENSER_CW_FLOW_RATIO                                          \
            * self.CONDENSER_CW_PUMPING_CONS
        # CO2 separation
        CO2_sep_elec_cons = exp.s_CO2_prod * u.mol/u.min * self.CO2_SEP_E_CONS

        elec_consumed = H2O_pur_cons                                \
                        + H2O_pump_cons + CH4_pump_cons             \
                        + H2_condenser_cons + CH4_condenser_cons    \
                        + CO2_sep_elec_cons

        # Subtract recovered electricity:
        elec_balance = elec_produced - elec_consumed
        # Discard electricity surplus but track electricity deficit:
        ext_elec_cons = -1*elec_balance if elec_balance < 0 else 0*u.W
        
        # Inflowing HHV
        HHV_in_CH4 = exp.N_s0 * u.mol/u.min * CH4_HHV
        HHV_in_tot = HHV_in_CH4
        
        # Outflowing HHV
        HHV_out_H2_f = exp.f_H2_prod * u.mol/u.min * H2_HHV
        HHV_out_H2_s = exp.s_H2_prod * u.mol/u.min * H2_HHV
        HHV_out_CO = exp.s_CO_prod * u.mol/u.min * CO_HHV
        HHV_out_CH4 = exp.N_s0 * u.mol/u.min * (1-exp.CH4_conv) * CH4_HHV
        HHV_out_tot = HHV_out_H2_f + HHV_out_H2_s + HHV_out_CO  \
                        + HHV_out_CH4
        
        P_in = ext_heat_cons + ext_elec_cons + HHV_in_tot; P_in.ito("W")
        P_out = HHV_out_tot; P_out.ito("W")
        efficiency_tot = P_out/P_in
        
        if (metrics != None):
            metrics_to_add = copy.deepcopy(locals()); ProcessModel.__remove_builtins(metrics_to_add)
            for param_name in funct_params:
                del metrics_to_add[param_name]
            del metrics_to_add["funct_params"]
            for quantity in metrics_to_add.values():
                try: quantity.ito("W")
                except pint.DimensionalityError: pass
            metrics.update(metrics_to_add)
            
        return efficiency_tot.magnitude
        
    
    def optimize_experiment(self, exp: Experiment, bd: Bounds) -> OptimizeResult:
        
        def objective_f(x : ndarray) -> float:
            xa = DataArray(data=x, coords=XA_COORDS)
            e = Experiment(
                T=xa.sel(param="T").item(),
                N_f0=xa.sel(param="N_f0").item(), 
                P_f=xa.sel(param="P_f").item(),
                N_s0=xa.sel(param="N_s0").item(),
                P_s=xa.sel(param="P_s").item(),
                x_f0=exp.x_f0, x_s0=exp.x_s0,
                A_mem=exp.A_mem, sigma=exp.sigma, L=exp.L, Lc=exp.Lc)
            e.run()
            e.analyze()
            return -1 * self.get_energy_eff(e)
        
        x0 = DataArray(
            data=[exp.T, exp.N_f0, exp.P_f, exp.N_s0, exp.P_s],
            coords=XA_COORDS)
        return optimize_wrapper(x0, bd, objective_f)
        
    def __remove_builtins(d : dict):
        for var_name in list(d):
            if (var_name.startswith('__') and var_name.endswith('__')): 
                del d[var_name]

    
if __name__ == "main":
    e = Experiment(T=900, 
                    N_f0=1e-4, x_f0="H2O:1", P_f=101325,
                    N_s0=1e-4, x_s0="CH4:1", P_s=101325)
    proc_model = ProcessModel()
    m = Metrics()    
    lb = DataArray(
        data=[800, e.A_mem * 1e-4 * 1e-2, 101325*0.9, e.A_mem * 1e-4 * 1e-2, 101325*0.9],
        coords=XA_COORDS)
    ub = DataArray(
        data=[1000, e.A_mem * 1e-4 * 100, 101325*2, e.A_mem * 1e-4 * 100, 101325*2],
        coords=XA_COORDS)
    print("+++++++++++++++++ RUN _ ++++++++++++++++++")
    print("Starting optimizer...")
    res = proc_model.optimize_experiment(e, Bounds(lb, ub))
    print(res)
    
    print("+++++++++++++++++ DONE ++++++++++++++++++")