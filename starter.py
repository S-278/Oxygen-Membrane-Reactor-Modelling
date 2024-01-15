# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:22:46 2023

@author: elijah.bakaleynik
"""

from OMR_model import *
from Optimization_model import *

e=Experiment()
e=Experiment(T=900, 
opt = DIRECT_Optimizer(DefaultPM.eval_experiment)
m=Metrics()
e.print_analysis()
print(f'Energy eff.: {DefaultPM.get_energy_eff(e, metrics=m):.1%}')
print(m)
