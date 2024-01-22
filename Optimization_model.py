# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:04:05 2023

@author: elijah.bakaleynik
"""

RUN_ID = "Input origin Kai's, narrow"
PROFILE_ID = None
if PROFILE_ID != None:
    import cProfile
    import pstats
    from OMR_model_profile import Experiment
else:
    from OMR_model import Experiment
from scipy.optimize import Bounds, OptimizeResult; import scipy.optimize
from numpy import ndarray; import numpy
from xarray import DataArray
import pint; u=pint.UnitRegistry()
import copy
from math import tanh
import csv
from abc import ABC, abstractmethod

XA_COORDS =  [("param", ['T', 'N_f0', 'P_f', 'N_s0', 'P_s'])]

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
    
def syngas_ratio_filter(ratio: float) -> float:
    SYNGAS_RATIO_TARGET = 2/1
    SYNGAS_RATIO_TOL = 0.5
    SYNGAS_RATIO_FILTER_STR = 5
    halfpt_lo = SYNGAS_RATIO_TARGET-SYNGAS_RATIO_TOL
    halfpt_hi = SYNGAS_RATIO_TARGET+SYNGAS_RATIO_TOL
    return -0.5 *(                                                         \
                  tanh(SYNGAS_RATIO_FILTER_STR * (ratio - halfpt_hi)) +    \
                  tanh(SYNGAS_RATIO_FILTER_STR * (-1*ratio + halfpt_lo))   \
                 ) + 0

             
class Metrics(dict):
    
    def __str__(self):
        col_template = '{: <20}{: >20}\n'
        endl_template = '{: <5}{: >35}\n'
        ret = ''
        ret += f'{"External heat consumption":~^50}\n'
        ret += col_template.format('Reactor:', f'{self["reactor_heat_supply"].to_compact():.2f}')
        ret += col_template.format('H2O boiling:', f'{self["H2O_boil_cons"].to_compact():.2f}')
        ret += col_template.format('Preheating:', f'{self["H2O_preheat_cons"].to_compact()+self["CH4_preheat_cons"].to_compact():.2f}')
        ret += '{: >40}\n'.format(f'(F:{self["H2O_preheat_cons"].to_compact():.2f}/S:{self["CH4_preheat_cons"].to_compact():.2f})')
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
    
    def remove_builtins(d : dict):
        for var_name in list(d):
            if (var_name.startswith('__') and var_name.endswith('__')): 
                del d[var_name]

            
class ProcessModel(ABC):
      
    @classmethod
    @abstractmethod
    def eval_experiment(cls, exp: Experiment) -> float:
        ...
            
class DefaultPM(ProcessModel):
    H2O_CYCLE_LOSS = 0.10
    H2O_PURIF_CONS = 4.52e-5 * u.kWh/u.mol
    H2O_BOILING_CONS = 76.6 * u.kJ/u.mol
    AMBIENT_T = u.Quantity(25, u.degC)
    AMBIENT_P = 101325 * u.Pa
    HX_EFF = 0.9
    REACTOR_HEAT_LOSS = 0.10
    CONDENSER_CW_FLOW_RATIO = 2/1
    CONDENSER_CW_PUMPING_CONS = 0.7e-6 * u.kWh/u.mol
    CO2_SEP_E_CONS = 9 * u.kJ/u.mol
    CO2_SEP_H_CONS = 132 * u.kJ/u.mol
    RANKINE_EFF = 0.4
    PUMPING_HEIGHT = 100 * u.m
    PUMPING_EFF = 0.7
    
    ambient_RT = GAS_CONST * AMBIENT_T.to("degK")
    boil_RT = GAS_CONST * u.Quantity(100, u.degC).to("degK")
    CH4_spec_grav_energy = CH4_M_Molar * GRAV_ACCEL * PUMPING_HEIGHT
    H2Ol_spec_grav_energy = H2O_M_Molar * GRAV_ACCEL * PUMPING_HEIGHT
    
    @classmethod
    def get_energy_eff(cls, exp: Experiment, metrics : Metrics = None) -> float:  
        funct_params = copy.deepcopy(locals()); Metrics.remove_builtins(funct_params)

        # External heat input:
        # Reactor heat supply (reaction heat + heat loss)
        reactor_heat_supply =                                                \
            exp.dH * u.W *                                                   \
            (1 + cls.REACTOR_HEAT_LOSS/(1-cls.REACTOR_HEAT_LOSS))
        # Input water boiling
        H2O_boil_cons = exp.N_f0 * u.mol/u.min * cls.H2O_BOILING_CONS
        # Preheating:
        exp_T = u.Quantity(exp.T, u.degC).to(u.degK)
        # Input water preheating
        boil_T = u.Quantity(100,u.degC).to(u.degK)
        H2O_preheat_cons =                                      \
            exp.N_f0 * u.mol/u.min * (                          \
                  H2Og_Cp_SLOPE/2 * (exp_T**2 - boil_T**2)      \
                  + H2Og_Cp_INT * (exp_T - boil_T)              \
            ) * (1-cls.HX_EFF)
        del boil_T
        # Input methane preheating
        amb_T = cls.AMBIENT_T.to(u.degK)
        CH4_preheat_cons =                                \
            exp.N_s0 * u.mol/u.min * (                    \
                  CH4_Cp_SLOPE/2 * (exp_T**2 - amb_T**2)  \
                  + CH4_Cp_INT * (exp_T - amb_T)          \
            ) * (1-cls.HX_EFF)
        del amb_T
        del exp_T
        # CO2 separation
        CO2_sep_heat_cons = exp.s_CO2_prod * u.mol/u.min * cls.CO2_SEP_H_CONS
        
        ext_heat_cons = reactor_heat_supply                       \
                            + H2O_boil_cons + H2O_preheat_cons    \
                            + CH4_preheat_cons                    \
                            + CO2_sep_heat_cons
        
        # External electricity input:
        # Calculate how much electricity 
        # can be recovered from waste heat.
        waste_heat_recovered = exp.dH * u.W / ( (1-cls.REACTOR_HEAT_LOSS)/cls.REACTOR_HEAT_LOSS )
        elec_produced = waste_heat_recovered * cls.RANKINE_EFF
        # Calculate total electricity consumption:
        # Input water purification
        H2O_pur_cons = exp.N_f0 * u.mol/u.min * cls.H2O_CYCLE_LOSS * cls.H2O_PURIF_CONS 
        # Input pumping
        H2Ol_pump_cons = exp.N_f0 * u.mol/u.min * cls.H2Ol_spec_grav_energy / cls.PUMPING_EFF
        
        H2Og_pump_cons = 0 * u.W
        if exp.P_f * u.Pa > cls.AMBIENT_P:
            H2Og_pump_cons = exp.N_f0 * u.mol/u.min                           \
                             * (exp.P_f*u.Pa - cls.AMBIENT_P)/cls.AMBIENT_P   \
                             * cls.boil_RT / cls.PUMPING_EFF
                             
        CH4_pump_cons = exp.N_s0 * u.mol/u.min * (                                                  \
                              cls.CH4_spec_grav_energy                                              \
                              + (exp.P_s * u.Pa - cls.AMBIENT_P)/cls.AMBIENT_P * cls.ambient_RT     \
                        ) / cls.PUMPING_EFF
        if CH4_pump_cons < 0: CH4_pump_cons = 0 * u.W
            
        # Condenser operation
        H2_condenser_cons =                             \
            exp.N_f * u.mol/u.min                       \
            * cls.CONDENSER_CW_FLOW_RATIO               \
            * cls.CONDENSER_CW_PUMPING_CONS             \
            / cls.PUMPING_EFF
        CH4_condenser_cons=                                                         \
            exp.N_s * u.mol/u.min                                                   \
            * (exp.x_s[exp.x_comp.index("CH4")] + exp.x_s[exp.x_comp.index("CO")])  \
            * cls.CONDENSER_CW_FLOW_RATIO                                           \
            * cls.CONDENSER_CW_PUMPING_CONS                                         \
            / cls.PUMPING_EFF
        # CO2 separation
        CO2_sep_elec_cons = exp.s_CO2_prod * u.mol/u.min * cls.CO2_SEP_E_CONS

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
            metrics_to_add = copy.deepcopy(locals()); Metrics.remove_builtins(metrics_to_add)
            for param_name in funct_params:
                del metrics_to_add[param_name]
            del metrics_to_add["funct_params"]
            for quantity in metrics_to_add.values():
                try: quantity.ito("W")
                except pint.DimensionalityError: pass
            metrics.update(metrics_to_add)
            
        return efficiency_tot.magnitude
    
    @classmethod
    def eval_experiment(cls, exp: Experiment) -> float:
        return cls.get_energy_eff(exp) * syngas_ratio_filter(exp.s_H2_prod/exp.s_CO_prod)
             
class Optimizer(ABC):
    
    def __init__(self, eval_funct: callable, run_id='test', track_progress=False):
        self.eval_funct = eval_funct
        self.run_id=run_id
        self.track_progress = track_progress
        
    def create_experiment_at(self, x) -> Experiment:
        e = Experiment(**self.fixed_params)
        xa = DataArray(data=x, coords=XA_COORDS)
        set_opt_x(e, xa)
        return e
    
    def _objective_f(self, x : ndarray) -> float:
        e = self.create_experiment_at(x)
        e.run()
        e.analyze()
        return -1 * self.eval_funct(e)
        
    @abstractmethod
    def optimize(self, init_exp: Experiment, bd: Bounds) -> OptimizeResult:
        ...
        
    def explore_local(self, target, 
                      origin=None, init_exp: Experiment = None, 
                      lookabout_range: tuple = None, num_samples: DataArray = None):
                    
        if num_samples is None:
            num_samples = DataArray(
                data=[50, 20, 20, 20, 20],
                coords=XA_COORDS)
            
        if origin is None:
            origin = self.optimal_x
            
        if lookabout_range is None:
            lookabout_range = (
                DataArray(
                    data=[-50, 
                          -origin.sel(param='N_f0').item() * 0.1,
                          -0.2*101325,
                          -origin.sel(param='N_s0').item() * 0.1,
                          -0.2*101325],
                    coords=XA_COORDS),
                DataArray(
                    data=[50, 
                          origin.sel(param='N_f0').item() * 0.1,
                          0.2*101325,
                          origin.sel(param='N_s0').item() * 0.1,
                          0.2*101325],
                    coords=XA_COORDS)
            )

        fixed_exp_params = dict()
        if init_exp is None:
            fixed_exp_params = self.fixed_params
        else:
            fixed_exp_params = dict(x_f0=init_exp.x_f0, x_s0=init_exp.x_s0,
                                    A_mem=init_exp.A_mem, sigma=init_exp.sigma, 
                                    L=init_exp.L, Lc=init_exp.Lc)
                    
        indep_var_axes = dict()
        for param in XA_COORDS[0][1]:
            indep_var_axes[param] =                                                                         \
                numpy.linspace(origin.sel(param=param).item() + lookabout_range[0].sel(param=param).item(), 
                               origin.sel(param=param).item() + lookabout_range[1].sel(param=param).item(),
                               num=num_samples.sel(param=param).item())
        
        def compute_target(exp_grid: ndarray) -> ndarray:
            vectorized_target_compute = None
            if callable(target):
                vectorized_target_compute = numpy.vectorize(target)
            elif target == 'eval':
                vectorized_target_compute = numpy.vectorize(self.eval_funct)
            else:
                vectorized_target_compute = numpy.vectorize(lambda e: getattr(e, target))
            return vectorized_target_compute(exp_grid)

        target_vs_T = compute_target(Experiment.grid(T=indep_var_axes['T'], 
                                                      N_f0=origin.sel(param='N_f0').item(),
                                                      N_s0=origin.sel(param='N_s0').item(),
                                                      P_f=origin.sel(param='P_f').item(),
                                                      P_s=origin.sel(param='P_f').item(),
                                                      **fixed_exp_params))
        target_vs_T_fig, target_vs_T_ax = plt.subplots()
        target_vs_T_ax.plot(indep_var_axes['T'], target_vs_T)
        target_vs_T_ax.spines['left'].set_position(('data', origin.sel(param='T').item()))
        target_vs_T_ax.spines['right'].set_visible(False); target_vs_T_ax.spines['top'].set_visible(False)
        target_vs_T_ax.set_title(target + ' vs. T'); target_vs_T_ax.set_xlabel('T (°C)'); target_vs_T_ax.set_ylabel(target)
        target_vs_T_fig.show()
        
        target_vs_N = compute_target(Experiment.grid(T=origin.sel(param='T').item(),
                                                      N_f0=indep_var_axes['N_f0'], 
                                                      N_s0=indep_var_axes['N_s0'],
                                                      P_f=origin.sel(param='P_f').item(),
                                                      P_s=origin.sel(param='P_f').item(),
                                                      **fixed_exp_params))
        target_vs_N_fig, target_vs_N_ax = plt.subplots(subplot_kw={'projection':'3d'})        
        target_vs_N_ax.plot_surface(*numpy.meshgrid(indep_var_axes['N_f0'], 
                                                    indep_var_axes['N_s0']),
                                    numpy.transpose(target_vs_N))
        
        # TODO: transpose bullshit
        
        # target_vs_N_ax.text(origin.sel(param='N_f0'),
        #                     origin.sel(param='N_s0'),)
        target_vs_N_ax.set_title(target + ' vs. N_f0 and N_s0'); target_vs_N_ax.set_xlabel('N_f0 (mol/min)'); target_vs_N_ax.set_ylabel('N_s0 (mol/min)'); target_vs_N_ax.set_zlabel(target)
        target_vs_N_fig.show()
        
        target_vs_P = compute_target(Experiment.grid(T=origin.sel(param='T').item(),
                                                     N_f0=origin.sel(param='N_f0').item(),
                                                     N_s0=origin.sel(param='N_s0').item(),
                                                     P_f=indep_var_axes['P_f'],
                                                     P_s=indep_var_axes['P_s'], 
                                                     **fixed_exp_params))
        target_vs_P_fig, target_vs_P_ax = plt.subplots(subplot_kw={'projection':'3d'}) 
        target_vs_P_ax.plot_surface(*numpy.meshgrid(indep_var_axes['P_f'], 
                                                   indep_var_axes['P_s']),
                                    numpy.transpose(target_vs_P))
        target_vs_P_ax.set_title(target + ' vs. P_f and P_s'); target_vs_P_ax.set_xlabel('P_f (Pa)'); target_vs_P_ax.set_ylabel('P_s (Pa)'); target_vs_P_ax.set_zlabel(target)
        target_vs_P_fig.show()
        
        # target_vs_P_f = compute_target(Experiment.grid(T=origin.sel(param='T').item(),
        #                                                N_f0=origin.sel(param='N_f0').item(),
        #                                                N_s0=origin.sel(param='N_s0').item(),
        #                                                P_f=indep_var_axes['P_f'], 
        #                                                P_s=origin.sel(param='P_s').item(), 
        #                                                **fixed_exp_params))
        # target_vs_P_f_fig, target_vs_P_f_ax = plt.subplots()
        # target_vs_P_f_ax.plot(indep_var_axes['P_f'], target_vs_P_f)
        # target_vs_P_f_ax.spines['left'].set_position(('data', origin.sel(param='P_f').item()))
        # target_vs_P_f_ax.spines['right'].set_visible(False); target_vs_P_f_ax.spines['top'].set_visible(False)
        # target_vs_P_f_ax.set_title(target + ' vs. P_f'); target_vs_P_f_ax.set_xlabel('P_f (Pa)'); target_vs_P_f_ax.set_ylabel(target)
        # target_vs_P_f_fig.show()
        
        # target_vs_P_s = compute_target(Experiment.grid(T=origin.sel(param='T').item(),
        #                                                N_f0=origin.sel(param='N_f0').item(),
        #                                                N_s0=origin.sel(param='N_s0').item(),
        #                                                P_f=origin.sel(param='P_f').item(), 
        #                                                P_s=indep_var_axes['P_s'], 
        #                                                **fixed_exp_params))
        # target_vs_P_s_fig, target_vs_P_s_ax = plt.subplots()
        # target_vs_P_s_ax.plot(indep_var_axes['P_s'], target_vs_P_s)
        # target_vs_P_s_ax.spines['left'].set_position(('data', origin.sel(param='P_s').item()))
        # target_vs_P_s_ax.spines['right'].set_visible(False); target_vs_P_s_ax.spines['top'].set_visible(False)
        # target_vs_P_s_ax.set_title(target + ' vs. P_s'); target_vs_P_s_ax.set_xlabel('P_s (Pa)'); target_vs_P_s_ax.set_ylabel(target)
        # target_vs_P_s_fig.show()
                        
class DIRECT_Optimizer(Optimizer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cb_count = 0
        
    def optimize(self, init_exp: Experiment, bd: Bounds) -> OptimizeResult:
                
        self.fixed_params = dict(
            x_f0=init_exp.x_f0, x_s0=init_exp.x_s0,
            A_mem=init_exp.A_mem, sigma=init_exp.sigma, 
            L=init_exp.L, Lc=init_exp.Lc)
        # x0 = DataArray(
        #     data=[init_exp.T, init_exp.N_f0, init_exp.P_f, init_exp.N_s0, init_exp.P_s],
        #     coords=XA_COORDS)
        
        if self.track_progress:
            self.prog_file = open('progress_tracking\\' + self.run_id + '_progress.csv', 'w+', newline='')
            self.file_writer = csv.DictWriter(self.prog_file, 
                                             fieldnames=XA_COORDS[0][1] + ['eval'])
            self.file_writer.writeheader()
            
        retval = scipy.optimize.direct(self._objective_f, bd, 
                                       eps=1e-1, locally_biased=False, 
                                       len_tol=1e-4, vol_tol=(1/5000)**5,
                                       maxfun=500000*5, maxiter=20000,
                                       callback=self.__optimizer_cb)
        
        if self.track_progress:
            self.prog_file.close()
            
        self.optimal_x = DataArray(data=retval.x, coords=XA_COORDS)
        return retval

    def __optimizer_cb(self, xk):
        self.cb_count += 1
        if self.cb_count % 100 == 0:
            print("CB invocation:", self.cb_count)
        if self.track_progress:
            dict_to_write = {param:val for (param,val) in zip(XA_COORDS[0][1], xk)}
            dict_to_write['eval'] = self.eval_funct(self.create_experiment_at(xk))
            self.file_writer.writerow(dict_to_write)


if __name__ == "__main__":
    e = Experiment(T=900, 
                    N_f0=1e-4, x_f0="H2O:1", P_f=101325,
                    N_s0=1e-4, x_s0="CH4:1", P_s=101325,
                    A_mem=10, sigma=1.3, L=250
                    )
    optimizer = DIRECT_Optimizer(DefaultPM.eval_experiment, run_id=RUN_ID, track_progress=True)
    m = Metrics()
    lb = DataArray(
        data=[800, e.A_mem * 1e-4, 101325*0.8, e.A_mem * 1e-4, 101325*0.8],
        coords=XA_COORDS)
    ub = DataArray(
        data=[1000, e.A_mem * 1e-2, 101325*1.2, e.A_mem * 1e-2, 101325*1.2],
        coords=XA_COORDS)
    print("+++++++++++++++++ RUN " + RUN_ID + " ++++++++++++++++++")
    if PROFILE_ID != None:
        print('Profiling ' + PROFILE_ID)
    print("Starting optimizer...")
    if PROFILE_ID != None:
        stats_path = 'profiling_stats\\' + PROFILE_ID + '_profile'
        cProfile.run('optimizer.optimize(e, Bounds(lb, ub))', stats_path)
        stats = pstats.Stats(stats_path)
        stats.strip_dirs().sort_stats('tottime').print_stats(100)
    else:
        res = optimizer.optimize(e, Bounds(lb, ub))
        print(res)
        set_opt_x(e, DataArray(data=res.x, coords=XA_COORDS))
        e.print_analysis()
        print(f'Optimized energy eff.: {DefaultPM.get_energy_eff(e, metrics=m):.1%}')
        print(m)
    print("+++++++++++++++++ DONE ++++++++++++++++++")