# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:01:32 2024

@author: elijah.bakaleynik
"""
from experiment import ProfilingExperiment
from optimization import *
import cProfile
import pstats
from xarray import DataArray
from process_model import *
import os

PROFILE_ID = ''
STATS_DIR = 'profiling_stats\\'

print('Profiling ' + PROFILE_ID)
e_init = ProfilingExperiment(T=1500, 
                             N_f0=4e-4, x_f0="H2O:1", P_f=101325,
                             N_s0=3e-5, x_s0="CH4:1", P_s=0.1*101325,
                             A_mem=1, sigma=0.4, L=500
                             )
pm = Spec_N_o2_PM()
optimizer = DE_Optimizer(pm.eval_experiment, track_progress=False)
lb = DataArray(
    data=[800, 2e-5, 101325*0.8, 2e-5, 101325*0.1],
    coords=XA_COORDS)
ub = DataArray(
    data=[1500, 5e-4, 101325*1.2, 5e-4, 101325*1.5],
    coords=XA_COORDS)
stats_path = STATS_DIR + PROFILE_ID + '_profile'
print(f'PID: {os.getpid()}')
cProfile.run('optimizer.optimize(e_init, Bounds(lb, ub))', stats_path)
stats = pstats.Stats(stats_path)
stats.strip_dirs().sort_stats('tottime').print_stats(100)
