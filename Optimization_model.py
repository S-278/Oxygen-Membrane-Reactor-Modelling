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
from math import tanh

XA_COORDS =  [("param", ["T", "N_f0", 'P_f', 'N_s0', 'P_s'])]

GAS_CONST = 8.314 * u.J/(u.mol * u.degK)
GRAV_ACCEL = 9.81 * u.m/(u.s**2)
CH4_M_Molar = 16.04 * u.g/u.mol
H2O_M_Molar =  18.015 * u.g/u.mol
CH4_HHV = 55384 * u.kJ/u.kg * CH4_M_Molar
CO_HHV = 10160 * u.kJ/u.kg * 28.01 * u.g/u.mol
H2_HHV = 142081 * u.kJ/u.kg * 2.016 * u.g/u.mol
H2Og_Cp_SLOPE = (0.000665666 * u.kJ/(u.kg*u.degK**2) * H2O_M_Molar)\
                .to(u.J/(u.mol*u.degK**2))
H2Og_Cp_INT = (1.62065 * u.kJ/(u.kg*u.degK) * H2O_M_Molar)\
              .to(u.J/(u.mol*u.degK))
CH4_Cp_SLOPE = (0.00333164 * u.kJ/(u.kg*u.degK**2) * CH4_M_Molar)\
               .to(u.J/(u.mol*u.degK**2))
CH4_Cp_INT = (1.2229 * u.kJ/(u.kg*u.degK) * CH4_M_Molar)\
             .to(u.J/(u.mol*u.degK))

glb_cb_ct = 0
glb_opt_intermediates = numpy.zeros((2000))

def optimize_wrapper(x0 : DataArray, bd : Bounds, f : callable) -> OptimizeResult:
    return scipy.optimize.direct(f, bd, 
                                 eps=1e-3, locally_biased=True, 
                                 len_tol=1e-3, vol_tol=1*10**(-3*5),
                                 maxfun=5000*5, maxiter=20000
                                 ,callback=optimize_cb
                                 )

def optimize_cb(xk):
    global glb_cb_ct
    global glb_opt_intermediates
    # glb_opt_intermediates[glb_cb_ct] = xk
    glb_cb_ct += 1
    if glb_cb_ct % 100 == 0:
        print("CB invocation:", glb_cb_ct)
        
def extract_opt_x(exp : Experiment) -> DataArray:
    ret = DataArray(coords=XA_COORDS)
    ret.loc[dict(param="T")] = exp.T
    ret.loc[dict(param="N_f0")] = exp.N_f0
    ret.loc[dict(param="P_f")] = exp.P_f
    ret.loc[dict(param="N_s0")] = exp.N_s0
    ret.loc[dict(param="P_s")] = exp.P_s
    return ret

def set_opt_x(exp : Experiment, xa : DataArray):
    exp.T=xa.sel(param="T").item()
    exp.N_f0=xa.sel(param="N_f0").item() 
    exp.P_f=xa.sel(param="P_f").item()
    exp.N_s0=xa.sel(param="N_s0").item()
    exp.P_s=xa.sel(param="P_s").item()
        
class Metrics(dict):
    
    def __str__(self):
        col_template = '{: <20}{: >20}\n'
        endl_template = '{: <5}{: >35}\n'
        ret = ''
        ret += f'{"External heat consumption":~^50}\n'
        ret += col_template.format('Reactor:', f'{self["reactor_heat_supply"].to_compact():.2f}')
        ret += col_template.format('H2O boiling:', f'{self["H2O_boil_cons"].to_compact():.2f}')
        ret += '{: <20}{: >20} {: <30}\n'.format('Preheating:', f'{self["H2O_preheat_cons"].to_compact()+self["CH4_preheat_cons"].to_compact():.2f}', f'(F:{self["H2O_preheat_cons"].to_compact():.2f}/S:{self["CH4_preheat_cons"].to_compact():.2f})')
        ret += col_template.format('CO2 sep:', f'{self["CO2_sep_heat_cons"].to_compact():.2f}')
        ret += endl_template.format('TOTAL', f'{self["ext_heat_cons"].to_compact():.2f}')
        ret += f'{"Electricity consumption":~^50}\n'
        ret += col_template.format('H2O purification:', f'{self["H2O_pur_cons"].to_compact():.2f}')
        ret += col_template.format('F pumping:', f'{self["H2Ol_pump_cons"].to_compact()+self["H2Og_pump_cons"].to_compact():.2f}')
        ret += col_template.format('S pumping:', f'{self["CH4_pump_cons"].to_compact():.2f}')
        ret += col_template.format('Condensers:', f'{self["H2_condenser_cons"].to_compact()+self["CH4_condenser_cons"].to_compact():.2f}')
        ret += col_template.format('CO2 sep:', f'{self["CO2_sep_elec_cons"].to_compact():.2f}')
        ret += endl_template.format('TOTAL', f'{self["elec_consumed"].to_compact():.2f}')
        ret += endl_template.format('BAL', f'{self["elec_balance"].to_compact():.2f}')
        return ret

class ProcessModel:
    def __init__(self):
        self.H2O_CYCLE_LOSS = 0.10
        self.H2O_PURIF_CONS = 4.52e-5 * u.kWh/u.mol
        self.H2O_BOILING_CONS = 76.6 * u.kJ/u.mol
        self.AMBIENT_T = u.Quantity(25, u.degC)
        self.AMBIENT_P = 101325 * u.Pa
        self.HX_EFF = 0.9
        self.REACTOR_HEAT_LOSS = 0.10
        self.CONDENSER_CW_FLOW_RATIO = 2/1
        self.CONDENSER_CW_PUMPING_CONS = 0.7e-6 * u.kWh/u.mol
        self.CO2_SEP_E_CONS = 9 * u.kJ/u.mol
        self.CO2_SEP_H_CONS = 132 * u.kJ/u.mol
        self.RANKINE_EFF = 0.4
        self.PUMPING_HEIGHT = 100 * u.m
        self.PUMPING_EFF = 0.7
        
        self.ambient_RT = GAS_CONST * self.AMBIENT_T.to("degK")
        self.boil_RT = GAS_CONST * u.Quantity(100, u.degC).to("degK")
        self.CH4_spec_grav_energy = CH4_M_Molar * GRAV_ACCEL * self.PUMPING_HEIGHT
        self.H2Ol_spec_grav_energy = H2O_M_Molar * GRAV_ACCEL * self.PUMPING_HEIGHT
        
        self.SYNGAS_RATIO_TARGET = 2/1
        self.SYNGAS_RATIO_TOL = 0.5
        self.SYNGAS_RATIO_FILTER_STR = 5
    
    
    def get_energy_eff(self, exp: Experiment, metrics : Metrics = None) -> float:  
        funct_params = copy.deepcopy(locals()); ProcessModel.__remove_builtins(funct_params)

        # External heat input:
        # Reactor heat supply (reaction heat + heat loss)
        reactor_heat_supply =                                                \
            exp.dH * u.W *                                                   \
            (1 + self.REACTOR_HEAT_LOSS/(1-self.REACTOR_HEAT_LOSS))
        # Input water boiling
        H2O_boil_cons = exp.N_f0 * u.mol/u.min * self.H2O_BOILING_CONS
        # Preheating:
        exp_T = u.Quantity(exp.T, u.degC).to(u.degK)
        # Input water preheating
        boil_T = u.Quantity(100,u.degC).to(u.degK)
        H2O_preheat_cons =                                      \
            exp.N_f0 * u.mol/u.min * (                          \
                  H2Og_Cp_SLOPE/2 * (exp_T**2 - boil_T**2)      \
                  + H2Og_Cp_INT * (exp_T - boil_T)              \
            ) * (1-self.HX_EFF)
        del boil_T
        # Input methane preheating
        amb_T = self.AMBIENT_T.to(u.degK)
        CH4_preheat_cons =                                \
            exp.N_s0 * u.mol/u.min * (                    \
                  CH4_Cp_SLOPE/2 * (exp_T**2 - amb_T**2)  \
                  + CH4_Cp_INT * (exp_T - amb_T)          \
            ) * (1-self.HX_EFF)
        del amb_T
        del exp_T
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
        H2Ol_pump_cons = exp.N_f0 * u.mol/u.min * self.H2Ol_spec_grav_energy / self.PUMPING_EFF
        
        H2Og_pump_cons = 0 * u.W
        if exp.P_f * u.Pa > self.AMBIENT_P:
            H2Og_pump_cons = exp.N_f0 * u.mol/u.min                             \
                             * (exp.P_f*u.Pa - self.AMBIENT_P)/self.AMBIENT_P   \
                             * self.boil_RT / self.PUMPING_EFF
                             
        CH4_pump_cons = exp.N_s0 * u.mol/u.min * (                                                  \
                              self.CH4_spec_grav_energy                                             \
                              + (exp.P_s * u.Pa - self.AMBIENT_P)/self.AMBIENT_P * self.ambient_RT  \
                        ) / self.PUMPING_EFF
        if CH4_pump_cons < 0: CH4_pump_cons.magnitude = 0
            
        # Condenser operation
        H2_condenser_cons =                             \
            exp.N_f * u.mol/u.min                       \
            * self.CONDENSER_CW_FLOW_RATIO              \
            * self.CONDENSER_CW_PUMPING_CONS            \
            / self.PUMPING_EFF
        CH4_condenser_cons=                                                         \
            exp.N_s * u.mol/u.min                                                   \
            * (exp.x_s[exp.x_comp.index("CH4")] + exp.x_s[exp.x_comp.index("CO")])  \
            * self.CONDENSER_CW_FLOW_RATIO                                          \
            * self.CONDENSER_CW_PUMPING_CONS                                        \
            / self.PUMPING_EFF
        # CO2 separation
        CO2_sep_elec_cons = exp.s_CO2_prod * u.mol/u.min * self.CO2_SEP_E_CONS

        elec_consumed = H2O_pur_cons                                        \
                        + H2Ol_pump_cons + CH4_pump_cons + H2Og_pump_cons   \
                        + H2_condenser_cons + CH4_condenser_cons            \
                        + CO2_sep_elec_cons

        # Subtract recovered electricity:
        elec_balance = elec_produced - elec_consumed
        # Discard electricity surplus but track electricity deficit:
        ext_elec_cons = -1*elec_balance if elec_balance < 0 else 0*u.W
        
        # Inflowing HHV
        HHV_in_CH4 = exp.N_s0 * u.mol/u.min * exp.CH4_conv * CH4_HHV
        HHV_in_tot = HHV_in_CH4
        
        # Outflowing HHV
        HHV_out_H2_f = exp.f_H2_prod * u.mol/u.min * H2_HHV
        HHV_out_H2_s = exp.s_H2_prod * u.mol/u.min * H2_HHV
        HHV_out_CO = exp.s_CO_prod * u.mol/u.min * CO_HHV
        HHV_out_tot = HHV_out_H2_f + HHV_out_H2_s + HHV_out_CO
        
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
    
    def syngas_ratio_filter(self, ratio: float) -> float:
        halfpt_lo = self.SYNGAS_RATIO_TARGET-self.SYNGAS_RATIO_TOL
        halfpt_hi = self.SYNGAS_RATIO_TARGET+self.SYNGAS_RATIO_TOL
        return -0.5 *(                                                              \
                      tanh(self.SYNGAS_RATIO_FILTER_STR * (ratio - halfpt_hi)) +    \
                      tanh(self.SYNGAS_RATIO_FILTER_STR * (-1*ratio + halfpt_lo))   \
                     ) + 0

    
    def eval_experiment(self, exp: Experiment) -> float:
        return self.get_energy_eff(exp) * self.syngas_ratio_filter(exp.s_H2_prod/exp.s_CO_prod)
    
    def optimize_experiment(self, exp: Experiment, bd: Bounds) -> OptimizeResult:
        
        def objective_f(x : ndarray) -> float:
            xa = DataArray(data=x, coords=XA_COORDS)
            e = Experiment(x_f0=exp.x_f0, x_s0=exp.x_s0,
                           A_mem=exp.A_mem, sigma=exp.sigma, L=exp.L, Lc=exp.Lc)
            set_opt_x(e, xa)
            e.run()
            e.analyze()
            return -1 * self.eval_experiment(e)
        
        x0 = DataArray(
            data=[exp.T, exp.N_f0, exp.P_f, exp.N_s0, exp.P_s],
            coords=XA_COORDS)
        return optimize_wrapper(x0, bd, objective_f)
        
    def __remove_builtins(d : dict):
        for var_name in list(d):
            if (var_name.startswith('__') and var_name.endswith('__')): 
                del d[var_name]

    
if __name__ == "__main__":
    e = Experiment(T=900, 
                    N_f0=1e-4, x_f0="H2O:1", P_f=101325,
                    N_s0=1e-4, x_s0="CH4:1", P_s=101325,
                    A_mem=10, sigma=5.84, L=250)
    proc_model = ProcessModel()
    m = Metrics()
    lb = DataArray(
        data=[600, e.A_mem * 1e-4 * 1e-3, 101325*0.1, e.A_mem * 1e-4 * 1e-3, 101325*0.1],
        coords=XA_COORDS)
    ub = DataArray(
        data=[1500, e.A_mem * 1e-4 * 1000, 101325*10, e.A_mem * 1e-4 * 1000, 101325*10],
        coords=XA_COORDS)
    print("+++++++++++++++++ RUN Input Origin sigma=5.84 ++++++++++++++++++")
    print("Starting optimizer...")
    res = proc_model.optimize_experiment(e, Bounds(lb, ub))
    print(res)
    set_opt_x(e, DataArray(data=res.x, coords=XA_COORDS))
    e.print_analysis()
    print(f'Optimized energy eff.: {proc_model.get_energy_eff(e):.1%}')
    print("+++++++++++++++++ DONE ++++++++++++++++++")