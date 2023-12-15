# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:04:05 2023

@author: elijah.bakaleynik
"""

from OMR_model import Experiment
from scipy.optimize import Bounds
import numpy
import xarray   
import pint; u=pint.UnitRegistry()

XA_COORDS =  [("param", ["T", "N_f0", 'P_f', 'N_s0', 'N_s'])]

CH4_HHV = 55384 * u.kJ/u.kg * 16.04 * u.g/u.mol
CO_HHV = 10160 * u.kJ/u.kg * 28.01 * u.g/u.mol
H2_HHV = 142081 * u.kJ/u.kg * 2.016 * u.g/u.mol

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
    
    def get_energy_eff(self, exp: Experiment) -> float:
        P_in = 0 * u.W; P_out = 0 * u.W
        
        # External heat input:
        # Reactor heat supply (reaction heat + heat loss)
        P_in += exp.dH * u.W * (1 + self.REACTOR_HEAT_LOSS/(1-self.REACTOR_HEAT_LOSS))
        # Input water boiling
        P_in += exp.N_f0 * u.mol/u.min * self.H2O_BOILING_CONS
        # Input water preheating
        P_in +=                                                         \
            exp.N_f0 * u.mol/u.min                                      \
            * (u.Quantity(exp.T,u.degC) - u.Quantity(100,u.degC))       \
            * (1-self.HX_EFF) * self.H2O_PREHEAT_CONS
        # Input methane preheating
        P_in +=                                                         \
            exp.N_s0 * u.mol/u.min                                      \
            * (u.Quantity(exp.T,u.degC) - self.AMBIENT_T)               \
            * (1-self.HX_EFF) * self.CH4_PREHEAT_CONS
        # CO2 separation
        P_in += exp.s_CO2_prod * u.mol/u.min * self.CO2_SEP_H_CONS
        
        # External electricity input:
        # Calculate how much electricity 
        # can be recovered from waste heat.
        waste_heat_recovered = exp.dH * u.W / ( (1-self.REACTOR_HEAT_LOSS)/self.REACTOR_HEAT_LOSS )
        elec_produced = waste_heat_recovered * self.RANKINE_EFF
        elec_consumed = 0 * u.W
        # Calculate total electricity consumption:
        # Input water purification
        elec_consumed += exp.N_f0 * u.mol/u.min * self.H2O_CYCLE_LOSS * self.H2O_PURIF_CONS 
        # Input pumping
        elec_consumed += exp.N_f0 * u.mol/u.min * self.H2O_PUMPING_CONS
        elec_consumed += exp.N_s0 * u.mol/u.min * self.CH4_PUMPING_CONS
        # Condenser operation
        elec_consumed +=                        \
            exp.N_f * u.mol/u.min               \
            * self.CONDENSER_CW_FLOW_RATIO      \
            * self.CONDENSER_CW_PUMPING_CONS
        elec_consumed +=                        \
            exp.N_s * u.mol/u.min               \
            * self.CONDENSER_CW_FLOW_RATIO      \
            * self.CONDENSER_CW_PUMPING_CONS
        # CO2 separation
        elec_consumed += exp.s_CO2_prod * u.mol/u.min * self.CO2_SEP_E_CONS
        # Subtract recovered electricity:
        elec_balance = elec_produced - elec_consumed
        # Discard electricity surplus but track electricity deficit:
        if elec_balance < 0: P_in += -1*elec_balance
        
        # Inflowing HHV
        P_in += exp.N_s0 * u.mol/u.min * CH4_HHV
        
        # Outflowing HHV
        P_out += exp.f_H2_prod * u.mol/u.min * H2_HHV
        P_out += exp.s_H2_prod * u.mol/u.min * H2_HHV
        P_out += exp.s_CO_prod * u.mol/u.min * CO_HHV
        P_out += exp.N_s0 * u.mol/u.min * (1-exp.CH4_conv) * CH4_HHV
        
        return P_out/P_in
        
    
    def optimize_experiment(self, exp: Experiment, bd: Bounds):
        pass

# Set up Experiment with origin of parameter space
e = Experiment(T=900, 
               N_f0=1e-4, x_f0="H2O:1", P_f=101325,
               N_s0=1e-4, x_s0="CH4:1", P_s=101325)
e.run()
e.analyze()
proc_model = ProcessModel()
proc_model.get_energy_eff(e)
# Set up parameter space bounds
lb = xarray.DataArray(
    data=[800, 0, 101325/10, 0, 101325/10],
    coords=XA_COORDS)
ub = xarray.DataArray(
    data=[1200, numpy.inf, 101325*10, numpy.inf, 101325*10],
    coords=XA_COORDS)
# Pass origin and bounds into process-aware optimizer
proc_model.optimize_experiment(e, Bounds(lb, ub))
# Display results