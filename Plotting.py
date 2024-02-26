# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:25:26 2024

@author: elijah.bakaleynik
"""
from experiment import Experiment
from optimization import *
from process_model import *
import csv
from cycler import cycler
from enum import Enum; from enum import auto as e_auto
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.ticker import PercentFormatter
from numpy import ndarray; import numpy
import os
from typing import Tuple, Dict, Callable, List
from xarray import DataArray

CACHE_PLOTDATA_EN = True
"""
If set to True, plotting functions in this module will cache 
the data to be plotted within CACHE_PLOTDATA_DIR
"""
CACHE_PLOTDATA_DIR = 'plot_cache'

PLOT_ANALYSIS_DIR = 'plot_analysis'
"""
Directory in which plotting functions in this module will
save additional analysis data.
"""

if CACHE_PLOTDATA_EN: import pickle

COLOR_SYNHELION_YLL = '#FFF385'
COLOR_SYNHELION_YL = '#f9d900'
COLOR_SYNHELION_GREY = '#AAAAAA'
COLOR_SYNHELION_ANTHRACITE = '#333333'
COLOR_CYCLE = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']

DEFAULT_COND_VALS = numpy.geomspace(0.08, 8, num=50)
"""
Default logarithmic axis of membrane specific conductances, in S/cm²
"""

"""Helper functions used by this module to cache plot data
"""
def read_cache(cache_file : str):
    retval = None
    try: 
        with open(CACHE_PLOTDATA_DIR + '\\' + cache_file, mode='rb') as f:
            retval = pickle.load(f)
    except FileNotFoundError: pass
    return retval

def write_cache(cache_file : str, obj_to_cache):
    try:            
        with open(CACHE_PLOTDATA_DIR + '\\' + cache_file, mode='w+b') as f:
            pickle.dump(obj_to_cache, f)
    except FileNotFoundError:
        os.mkdir(CACHE_PLOTDATA_DIR)
        with open(CACHE_PLOTDATA_DIR + '\\' + cache_file, mode='w+b') as f:
            pickle.dump(obj_to_cache, f)
            
def compute_target(exp_grid: ndarray, target) -> ndarray:
    """Computes the value of target over an array of Experiments
    If target is a callable, it will be called on each Experiment directly,
    otherwise target will be interpreted as the name of an attribute to query.
    """
    vectorized_target_compute = None
    if callable(target):
        vectorized_target_compute = numpy.vectorize(target)
    else:
        vectorized_target_compute = numpy.vectorize(lambda e: getattr(e, target))
    return vectorized_target_compute(exp_grid)

PARAM_COORDS = [("param", ['T', 'N_f0', 'P_f', 'N_s0', 'P_s'])]
def explore_local(target,
                  origin : DataArray, init_exp: Experiment, 
                  lookabout_range : Tuple[DataArray, DataArray] = None, 
                  num_samples: DataArray = DataArray(
                      data=[50, 20, 20, 20, 20],
                      coords=PARAM_COORDS),
                  plot_T=True, plot_N=True, plot_P=True,
                  from_cache=False) -> List[Figure]:
    """Generate plots of target over the local parameter space around origin
    
    Creates plots of the variable specified as target against various 
    axes of the optimization parameter space in the local space around origin
    within the lookabout range specified. 
    Currently, the following plots are created:
        -- Line plot of the target vs. temperature
        -- Surface plot of the target vs. N_f0 and N_s0
        -- Surface plot of the target vs. P_f and P_s

    Parameters
    ----------
    target : string or callable(Experiment) -> float 
        Dependent variable to plot. Must be either a string specifying 
        an attribute of the Experiment class to be queried,
        or a callable which will be passed an Experiment at every point plotted.
    origin : DataArray
        Origin of the independent variable space, specified as a DataArray using PARAM_COORDS.
    init_exp : Experiment
        Experiment containing the fixed Experiment parameters used to 
        compute the target at every point plotted.
    lookabout_range : Tuple[DataArray, DataArray], optional
        Tuple of DataArrays using PARAM_COORDS in the order: lower bound, upper bound;
        recording the absolute displacements below and abovev the origin, respectively.
        Entries in the lower bound must be negative or zero, 
        and entries in the upper bound must be positive or zero.
        The space on which the target will be plotted is then defined for each 
        axis as [origin + lower bound, origin + upper bound].
        If omitted, default values will be used.
    num_samples : DataArray, optional
        Number of samples to generate along each axis of the parameter space.
        Must use PARAM_COORDS. If omitted, default values will be used.
    plot_T : bool, optional
        Whether to create the temperature plot. The default is True.
    plot_N : bool, optional
        Whether to create the N_f0 and N_s0 plot. The default is True.
    plot_P : bool, optional
        Whether to create the P_f and P_s plot. The default is True.
    from_cache : bool, optional
        Whether to read in plot data from the cache. The default is False.
        Has no effect if CACHE_PLOTDATA_EN is not True.
        Any data found in the cache will be plotted directly without being recomputed.
        If CACHE_PLOTDATA_EN is True, the cache is always overwritten with 
        the data that was actually plotted.
        Note that no checking is performed as to whether the cached data 
        was generated with the same input parameters!

    Raises
    ------
    ValueError
        Raised on an invalid value in lookabout_range.

    Returns
    -------
    List of Figures
        List containing the matplotlib Figures generated.

    """
                    
    if lookabout_range is None:
        lookabout_range = (
            DataArray(
                data=[-50, 
                      -origin.sel(param='N_f0').item() * 0.1,
                      -0.2*101325,
                      -origin.sel(param='N_s0').item() * 0.1,
                      -0.2*101325],
                coords=PARAM_COORDS),
            DataArray(
                data=[50, 
                      origin.sel(param='N_f0').item() * 0.1,
                      0.2*101325,
                      origin.sel(param='N_s0').item() * 0.1,
                      0.2*101325],
                coords=PARAM_COORDS)
        )
    else:
        for step in lookabout_range[0]:
            if step > 0: raise ValueError('Positive value given for negative step')
        for step in lookabout_range[1]:
            if step < 0: raise ValueError('Negative value given for positive step')
            
    CACHE_FILE = 'explore_local.cache'

    fixed_exp_params = extract_opt_fixed(init_exp)
                
    # Create 1D arrays for the values of the independent variables
    indep_var_axes = dict()
    for param in PARAM_COORDS[0][1]:
        indep_var_axes[param] =                                                                         \
            numpy.linspace(origin.sel(param=param).item() + lookabout_range[0].sel(param=param).item(), 
                           origin.sel(param=param).item() + lookabout_range[1].sel(param=param).item(),
                           num=num_samples.sel(param=param).item())
            
    target_str = 'Target'
    if callable(target): target_str = target.__name__
    else: target_str = str(target)
        
    def plot_3d(X_flat, Y_flat, Z, 
                origin : Tuple[float, float]) -> Tuple[Figure, Axes3D]:
        # Generic function to make a 3D surface plot
        # with a given origin marked.
        # X_flat,Y_flat are 1D arrays, Z is a 2D array.
        # origin must be a tuple of x,y coordinates, 
        # which must be within the range of X_flat,Y_flat, respectively
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})        
        ax.plot_surface(*numpy.meshgrid(X_flat, Y_flat), Z)
        origin_x_idx = numpy.argmin(numpy.abs(X_flat - origin[0]))
        origin_y_idx = numpy.argmin(numpy.abs(Y_flat - origin[1]))
        ax.scatter(X_flat[origin_x_idx], Y_flat[origin_y_idx],
                   Z[origin_x_idx, origin_y_idx],
                   color='k')
        ax.text(X_flat[origin_x_idx], Y_flat[origin_y_idx],
                Z[origin_x_idx, origin_y_idx],
                'origin', color='k')
        return (fig,ax)
    
    def plot_2d(X, Y, origin : float) -> Tuple[Figure, Axes3D]:
        # Generic function to make a line plot of X and Y,
        # with the y-axis placed at origin,
        # which must be within the interval defined by X.
        fig, ax = plt.subplots()
        ax.plot(X, Y)
        ax.spines['left'].set_position(('data', origin))
        ax.spines[['right','top']].set_visible(False)
        return (fig, ax)
    
    def plot_target_vs_T(target_vs_T : ndarray = None) -> Tuple[Figure, ndarray]:
        if target_vs_T is None:
            target_vs_T = compute_target(Experiment.grid(T=indep_var_axes['T'], 
                                                         N_f0=origin.sel(param='N_f0').item(),
                                                         N_s0=origin.sel(param='N_s0').item(),
                                                         P_f=origin.sel(param='P_f').item(),
                                                         P_s=origin.sel(param='P_f').item(),
                                                         **fixed_exp_params),
                                         target)
            
        target_vs_T_fig, target_vs_T_ax = \
            plot_2d(indep_var_axes['T'], target_vs_T, 
                    origin.sel(param='T').item())
        target_vs_T_ax.set_title(target_str + ' vs. T')
        target_vs_T_ax.set_xlabel('T (°C)')
        target_vs_T_ax.set_ylabel(target_str)
        return (target_vs_T_fig, target_vs_T)
    
    def plot_target_vs_N(target_vs_N : ndarray = None) -> Tuple[Figure, ndarray]:
        if target_vs_N is None:
            target_vs_N = compute_target(Experiment.grid(T=origin.sel(param='T').item(),
                                                         N_f0=indep_var_axes['N_f0'], 
                                                         N_s0=indep_var_axes['N_s0'],
                                                         P_f=origin.sel(param='P_f').item(),
                                                         P_s=origin.sel(param='P_f').item(),
                                                         **fixed_exp_params),
                                         target)
            
        target_vs_N_fig, target_vs_N_ax = \
            plot_3d(indep_var_axes['N_f0'], indep_var_axes['N_s0'],
                    numpy.transpose(target_vs_N), # see docstring of Experiment.grid for why a transpose is needed
                    (origin.sel(param='N_f0').item(), origin.sel(param='N_s0').item()))
        target_vs_N_ax.set_title(target_str + ' vs. N_f0 and N_s0')
        target_vs_N_ax.set_xlabel('N_f0 (mol/min)')
        target_vs_N_ax.set_ylabel('N_s0 (mol/min)')
        target_vs_N_ax.set_zlabel(target_str)
        return (target_vs_N_fig, target_vs_N)
    
    def plot_target_vs_P(target_vs_P : ndarray = None) -> Tuple[Figure, ndarray]:
        if target_vs_P is None:
            target_vs_P = compute_target(Experiment.grid(T=origin.sel(param='T').item(),
                                                         N_f0=origin.sel(param='N_f0').item(),
                                                         N_s0=origin.sel(param='N_s0').item(),
                                                         P_f=indep_var_axes['P_f'],
                                                         P_s=indep_var_axes['P_s'], 
                                                         **fixed_exp_params),
                                         target)
    
        target_vs_P_fig, target_vs_P_ax = \
            plot_3d(indep_var_axes['P_f'], indep_var_axes['P_s'],
                    numpy.transpose(target_vs_P), # see docstring of Experiment.grid for why a transpose is needed
                    (origin.sel(param='P_f').item(), origin.sel(param='P_s').item()))
        target_vs_P_ax.set_title(target_str + ' vs. P_f and P_s')
        target_vs_P_ax.set_xlabel('P_f (Pa)')
        target_vs_P_ax.set_ylabel('P_s (Pa)')
        target_vs_P_ax.set_zlabel(target_str)
        return (target_vs_P_fig, target_vs_P)
    
    PLOTTERS : Dict[Plots_e, Callable[[ndarray], Tuple[Figure, ndarray]]] = \
        {Plots_e.T : plot_target_vs_T, Plots_e.N : plot_target_vs_N, Plots_e.P : plot_target_vs_P}
        
        
    data_to_plot : Dict[Plots_e, ndarray] = None
    # If cached data is enabled,
    # try to populate data_to_plot from the cache
    if CACHE_PLOTDATA_EN and from_cache:
        data_to_plot = read_cache(CACHE_FILE)

    if data_to_plot is None: data_to_plot = dict()
    figures_to_show = list(); 
    data_to_cache : Dict[Plots_e, ndarray] = dict(); 
    plots_to_make = {Plots_e.T : plot_T, Plots_e.N : plot_N, Plots_e.P : plot_P}
    for plot in Plots_e:
        if plots_to_make[plot]:
            new_fig, new_data = PLOTTERS[plot](data_to_plot.get(plot))
            data_to_cache[plot] = new_data
            figures_to_show.append(new_fig)
        
    # If caching is enabled, overwrite the cache with
    # the data returned by the plotters
    if CACHE_PLOTDATA_EN:
        write_cache(CACHE_FILE, data_to_cache)
                
    for fig in figures_to_show: fig.show()
    return figures_to_show
class Plots_e(Enum):
    T = 0
    N = e_auto()
    P = e_auto()

def tornado(pm : Scenario_PM, e_0 : Experiment, gen_metrics=False, from_cache=False) -> Figure:
    """Generate a tornado plot of process efficiency vs. process model parameters
    
    In this tornado plot, the given Experiment e_0 is plugged into the given 
    process model pm, and the process efficiency is calculated first with all
    process model parameters at their central values, then with a single parameter
    taking on a pessimistic and optimistic value.

    Parameters
    ----------
    pm : Scenario_PM
        Process model to calculate process efficiency. Must have central, pessimistic,
        and optimistic values for its model parameters.
    e_0 : Experiment
        Experiment to evaluate.
    gen_metrics : bool, optional
        Whether to generate metrics for each data point on the plot. If True,
        a csv file will be created in PLOT_ANALYSIS_DIR. The default is False.
    from_cache : bool, optional
        Whether to read in plot data from the cache. The default is False.
        The central case is never cached.
        Has no effect if CACHE_PLOTDATA_EN is not True.
        Any data found in the cache will be plotted directly without being recomputed.
        If CACHE_PLOTDATA_EN is True, the cache is always overwritten with 
        the data that was actually plotted.
        Note that no checking is performed as to whether the cached data 
        was generated with the same input parameters!

    Returns
    -------
    Figure
        Figure generated.

    """
    METRICS_FILE = 'tornado_metrics.csv'
    CACHE_FILE = 'tornado.cache'
    
    # Reset ProcessModel to central scenario for each model parameter
    for key in pm.MODEL_PARAMETERS:
        pm.scenario_def[key] = Scenarios.CENTRAL
        
    # Record result of process model when all parameters are central
    central_metrics = None
    if gen_metrics: central_metrics = Metrics()
    central_eff = pm.get_energy_eff(e_0, central_metrics)
    
    efficiencies_per_scenario = None
    
    def calculate_eff_per_param(param, scenario : Scenarios):
        # Calculate the energy efficiency when model parameter given by param
        # is set to scenario
        pm.scenario_def[param] = scenario
        to_write = None
        if gen_metrics: to_write = Metrics({'Param' : param, 'Scenario' : scenario})
        eff = pm.get_energy_eff(e_0, to_write)
        if gen_metrics: file_writer.writerow(to_write)
        pm.scenario_def[param] = Scenarios.CENTRAL # Reset the parameter back to central
        return eff
    
    # If cached data is enabled,
    # try to populate efficiencies_per_scenario from the cache
    if CACHE_PLOTDATA_EN and from_cache:
        efficiencies_per_scenario = read_cache(CACHE_FILE)
    
    # If cache was empty or disabled, compute efficiencies_per_scenario
    if efficiencies_per_scenario is None:
        efficiencies_per_scenario = dict()
        
        csv_f = None; file_writer = None
        if gen_metrics: 
            csv_f = open(PLOT_ANALYSIS_DIR + '\\' + METRICS_FILE, 'w+', newline='')
            file_writer = csv.DictWriter(csv_f, fieldnames = ['Param', 'Scenario'] + list(central_metrics))
            file_writer.writeheader()
            to_write = {'Param' : None, 'Scenario' : Scenarios.CENTRAL}; to_write.update(central_metrics)
            file_writer.writerow(to_write)
        
        # Make a vectorized function that can act on an array of parameters
        calculate_eff_vectorized = numpy.vectorize(calculate_eff_per_param, otypes=[None], excluded=[1])
        
        for scenario in [Scenarios.OPTIMISTIC, Scenarios.PESSIMISTIC]:
            # For the optimistic and pessimistic scenarios,
            # pass all model parameters into the vectorized function
            # to get an array of energy efficiencies 
            efficiencies_per_scenario[scenario] = calculate_eff_vectorized(list(pm.MODEL_PARAMETERS), scenario)
                    
        if csv_f != None: csv_f.close()
        
    # If caching is enabled, overwrite the cache 
    if CACHE_PLOTDATA_EN:
        write_cache(CACHE_FILE, efficiencies_per_scenario)
    
    tornado_fig, tornado_ax = plt.subplots(figsize=(10,4.8))#dpi=200)
    bar_lbl_format = '{:+.0%}'
    
    pess_bars = tornado_ax.barh(
        list(pm.MODEL_PARAMETERS), 
        efficiencies_per_scenario[Scenarios.PESSIMISTIC] - central_eff, 
        left=central_eff,
        color=COLOR_SYNHELION_ANTHRACITE,
    )
    tornado_ax.bar_label(
         pess_bars, 
         labels=[bar_lbl_format.format((eff - central_eff)/central_eff) for eff in efficiencies_per_scenario[Scenarios.PESSIMISTIC]],
         padding=10
    )
    
    opt_bars = tornado_ax.barh(
        list(pm.MODEL_PARAMETERS),
        efficiencies_per_scenario[Scenarios.OPTIMISTIC] - central_eff, 
        left=central_eff,
        color=COLOR_SYNHELION_YL,
    )
    tornado_ax.bar_label(
        opt_bars, 
        labels=[bar_lbl_format.format((eff - central_eff)/central_eff) for eff in efficiencies_per_scenario[Scenarios.OPTIMISTIC]],
        padding=10
    )
    
    tornado_ax.xaxis.set_major_formatter(PercentFormatter(decimals=0, xmax=1))#; tornado_ax.set_xlim(right=1)
    tornado_ax.spines[['right', 'bottom']].set_visible(False)
    tornado_ax.spines['left'].set_position(('data', central_eff))
    tornado_ax.get_xaxis().set_ticks_position('top')
    tornado_ax.get_yaxis().set_ticks_position('left')
    tornado_ax.set_xlabel('Process Efficiency', labelpad=8)
    tornado_ax.xaxis.set_label_position('top')
    lowest_eff = numpy.amin(efficiencies_per_scenario[Scenarios.PESSIMISTIC])
    low_x_lim = central_eff - 1.1 * (central_eff - lowest_eff)
    tornado_ax.set_xlim(left=low_x_lim)

    # Somewhat complicated transform procedure to ensure the y labels
    # appear on the left edge of the plot.
    # Make a transform that first goes from data coordinates to display coordinates,
    # and then from display to inches
    data_to_in_xform = tornado_ax.transData - tornado_fig.dpi_scale_trans
    y_axis_in = data_to_in_xform.transform((central_eff, 0))[0] 
    left_edge_in = data_to_in_xform.transform((low_x_lim, 0))[0]
    # Calculate the distance between the y axis at the central efficiency,
    # and the left edge of the plot, in points (1 point = 1/72 inches)
    y_lbl_pad_pts = (y_axis_in - left_edge_in) * 72
    tornado_ax.tick_params(axis='y', which='both', pad=y_lbl_pad_pts, direction='inout')
    
    plt.show()
    return tornado_fig

def cmap_vs_cond_N(target, init_exp : Experiment, 
                   N_vals : ndarray, cond_vals : ndarray = DEFAULT_COND_VALS,
                   target_str : str = None,
                   N_ratio : float = 1,
                   cond_log=False, N_log=False,
                   fmt=None,
                   from_cache=False) -> Figure:
    """Plot colormap of target over a grid of specific conductance vs. flowrate
    
    Parameters
    ----------
    target : string or callable(Experiment) -> float 
        Dependent variable to plot. Must be either a string specifying 
        an attribute of the Experiment class to be queried,
        or a callable which will be passed an Experiment at every point plotted.
    init_exp : Experiment
        Experiment from which all input parameters except sigma, N_f0, and N_s0
        will be read as fixed parameters.
    N_vals : ndarray
        Flowrate values to use for the y axis. Should be linearly or geometrically
        spaced according to N_log.
    cond_vals : ndarray, optional
        Conductance values to use for the x axis. Should be linearly or geometrically
        spaced according to cond_log. The default is DEFAULT_COND_VALS.
    target_str : str, optional
        String label for the target variable. If not supplied, this method will
        attempt to use a reasonable conversion of target.
    N_ratio : float, optional
        Sweep:feed flowrate ratio. The default is 1.
    cond_log : bool, optional
        Whether to use a logarithmic scale for the conductance on the x axis.
        The default is False.
    N_log : bool, optional
        Whether to use a logarithmic scale for the flowrate on the y axis.
        The default is False.
    fmt, optional
        Target value formatter, will be passed directly to call to Figure.colorbar.
    from_cache : bool, optional
        Whether to read in plot data from the cache. The default is False.
        Has no effect if CACHE_PLOTDATA_EN is not True.
        Any data found in the cache will be plotted directly without being recomputed.
        If CACHE_PLOTDATA_EN is True, the cache is always overwritten with 
        the data that was actually plotted.
        Note that no checking is performed as to whether the cached data 
        was generated with the same input parameters!

    Returns
    -------
    Figure
        Figure generated.

    """
    CACHE_FILE = 'cmap_cond_N.cache'
    
    if target_str is None:
        if callable(target): target_str = target.__name__
        else: target_str = str(target)

    e_grid = None
    if CACHE_PLOTDATA_EN and from_cache:
        e_grid = read_cache(CACHE_FILE)
    if e_grid is None:
        cond_mesh, N_mesh = numpy.meshgrid(cond_vals, N_vals)
        
        def experiment_helper(cond, N):
            return Experiment(A_mem=init_exp.A_mem, L=init_exp.L,
                              T=init_exp.T, P_f=init_exp.P_f, P_s=init_exp.P_s,
                              sigma=spec_cond_to_sigma(cond, init_exp.L),
                              N_f0=N, N_s0=N * N_ratio)
        e_grid = numpy.vectorize(experiment_helper, otypes=[numpy.dtype(numpy.object_)])(cond_mesh, N_mesh)
        
    
    target_data = compute_target(e_grid, target)
        
    if CACHE_PLOTDATA_EN:
        write_cache(CACHE_FILE, e_grid)
    
    # Debug tool: sets up an array of alternating values 
    # to easily see each pixel on the colormap
    # fake_counter = 0
    # def fake_helper(eff):
    #     global fake_counter
    #     fake_counter += 1
    #     return fake_counter % 2
    # fake_data = numpy.vectorize(fake_helper, otypes=[None])(e_grid)
    
    # Calculate plot area bounds
    left_x = None; right_x = None
    if cond_log:
        x_step = cond_vals[1]/cond_vals[0]
        left_x = cond_vals[0] / sqrt(x_step); right_x = cond_vals[-1] * sqrt(x_step)
    else:
        x_step = cond_vals[1] - cond_vals[0]
        left_x = cond_vals[0] - x_step/2; right_x = cond_vals[-1] + x_step/2
    bottom_y = None; top_y = None
    if N_log:
        y_step = N_vals[1]/N_vals[0]
        bottom_y = N_vals[0] / sqrt(y_step); top_y = N_vals[-1] * sqrt(y_step)
    else:
        y_step = N_vals[1] - N_vals[0]
        bottom_y = N_vals[0] - y_step/2; top_y = N_vals[-1] + y_step/2
        
    # Calculate positions of colormesh rectangle corners
    x_corners = numpy.geomspace(left_x, right_x, num=len(cond_vals)+1) if cond_log \
        else numpy.linspace(left_x, right_x, num=len(cond_vals)+1)
    y_corners = numpy.geomspace(bottom_y, top_y, num=len(N_vals)+1) if N_log \
        else numpy.linspace(bottom_y, top_y, num=len(N_vals)+1)

    cmap_fig, cmap_ax = plt.subplots()
    if cond_log: cmap_ax.set_xscale('log')
    if N_log: cmap_ax.set_yscale('log')
    cmap_ax.set_xlim((left_x, right_x)); cmap_ax.set_ylim((bottom_y, top_y))
    cmap_im = cmap_ax.pcolormesh(x_corners, y_corners, target_data)
    cmap_ax.set_xlabel('Specific conductance (S/cm²)'); cmap_ax.set_ylabel('Specific feed flowrate (mol/min/cm²)')
    cmap_fig.colorbar(cmap_im, ax=cmap_ax, fraction=0.2, label=target_str, format=fmt)
    cmap_ax.yaxis.set_major_formatter('{x:.1e}')
    cmap_fig.show()
    return cmap_fig

def annotate_axis_arrows(left_ax : Axes, right_ax : Axes,
                         left_dep_vars : List[ndarray], right_dep_vars : List[ndarray], 
                         indep_var : ndarray,
                         color='k', arrow_interval=0.1, arrow_len_px=50,
                         left_start=0.1, right_start=0.9):
    """Add arrows onto twinx axes to indicate which yaxis each curve uses
    
    For each dependent variable in left_dep_vars, adds a horizontal arrow
    onto its curve pointing left, and the same for right_dep_vars pointing right.
    The arrows are spaced horizontally by arrow_interval, and are always snapped
    to the nearest location on the x-axis for which indep_var has a value.

    Parameters
    ----------
    left_ax : Axes
    right_ax : Axes
    left_dep_vars : List[ndarray]
        List of arrays. Each array is a dependent variable plotted on left_ax.
    right_dep_vars : List[ndarray]
        List of arrays. Each array is a dependent variable plotted on right_ax.
    indep_var : ndarray
        Independent variable axis.
    color, optional
        color to be passed to the arrow object constructor. The default is 'k'.
    arrow_interval : float, optional
        Amount of space to leave between adjacent arrows, in Axes coordinates. 
        The default is 0.1.
    arrow_len_px : int, optional
        Length of each arrow in pixels. The default is 50.
    left_start : float, optional
        X-coordinate in Axes coordinates at which to start placing arrows for
        left_ax. Subsequent arrows will be placed at increasing Axes coordinates.
        The default is 0.1.
    right_start : float, optional
        X-coordinate in Axes coordinates at which to start placing arrows for
        right_ax. Subsequent arrows will be placed at decreasing Axes coordinates.
        The default is 0.9.

    Returns
    -------
    None.

    """
    for var in left_dep_vars:
        # Find the member of indep_var closest to the data coordinate 
        # we are trying to place the arrow at.
        ann_arrow_x_idx = round(left_start * (len(indep_var)-1))
        left_ax.annotate('', 
                    (indep_var[ann_arrow_x_idx], var[ann_arrow_x_idx]),
                    xycoords='data',
                    xytext=(-arrow_len_px, 0),
                    textcoords='offset pixels',
                    color=color,
                    arrowprops={'arrowstyle' : '<-', 'color' : color}
                    )
        left_start += arrow_interval
    for var in right_dep_vars:
        ann_arrow_x_idx = round(right_start * (len(indep_var)-1))
        right_ax.annotate('', 
                    (indep_var[ann_arrow_x_idx], var[ann_arrow_x_idx]),
                    xycoords='data',
                    xytext=(arrow_len_px, 0),
                    textcoords='offset pixels',
                    color=color,
                    arrowprops={'arrowstyle' : '<-', 'color' : color}
                    )
        right_start -= arrow_interval
        
def read_csv(fname : str) -> Dict[str, List[float]]:
    """Generate a data dict from a csv file
    
    Given a csv file with a header row, builds a dict mapping the 
    column headers to a list of the values in that column.

    Parameters
    ----------
    fname : str
        File name to read.

    Returns
    -------
    Dict[str, List[float]]
        dict mapping column headers to column data.

    """
    csv_data = dict()
    with open(fname, mode='r', newline='') as csv_f:
        f_reader = csv.DictReader(csv_f)
        for row_dict in f_reader:
            for key,val in row_dict.items():
                try:
                    csv_data[key].append(float(val))
                except ValueError:
                    pass
                except KeyError:
                    csv_data[key] = [float(val)]
    return csv_data

        
def rxn_perf_vs_cond(exs : ndarray = None, data_dict : Dict[str, List[float]] = None,
                     pm : Eff_PM = None, cond_log=False, prod_log=False) -> Figure:
    """Plot reaction performance metrics vs. specific conductance
    
    Takes data either as a 1D array of Experiments to run, 
    or as a data dict with the same format as that returned by read_csv, 
    but not both.

    Parameters
    ----------
    exs : ndarray, optional
        Experiments array.
    data_dict : Dict[str, List[float]], optional
        Data dict in the same format as returned by read_csv.
    pm : Eff_PM, optional
        If supplied, will be used to evaluate Experiment efficiency.
    cond_log : bool, optional
    prod_log : bool, optional

    Returns
    -------
    Figure
    """
    
    if bool(exs) ^ bool(data_dict):
        raise ValueError('Either an Experiment array or a data dict must be provided, but not both.')
        
    if data_dict is None:
        data_dict = {
            'spec_cond' : [],
            'j_o2' : [],
            'eff' : [],
            'X_CH4' : [],
            'X_H2O' : [],
            'S_CO' : [],
        }
        for e in exs:
            data_dict['spec_cond'].append(sigma_to_spec_cond(e.sigma, e.L))
            data_dict['j_o2'].append(e.N_o2 / e.A_mem)
            if pm is not None: data_dict['eff'].append(pm.get_energy_eff(e))
            data_dict['X_CH4'].append(e.CH4_conv)
            data_dict['X_H2O'].append(e.H2O_conv)
            data_dict['S_CO'].append(e.CO_sel)

    perf_fig, prod_ax = plt.subplots()
    if cond_log: prod_ax.set_xscale('log'); prod_ax.set_xlabel('Specific conductance (S/cm²)')
    percent_ax = prod_ax.twinx()
    vals_on_prod_ax = ['j_o2']
    vals_on_percent_ax = ['eff', 'X_CH4', 'X_H2O', 'S_CO']
    indep_var = data_dict['spec_cond']
    for key in vals_on_prod_ax:
        prod_ax.plot(indep_var, data_dict[key], label=key)
    for key in vals_on_percent_ax:
        percent_ax.plot(indep_var, data_dict[key], label=key)
        
    annotate_axis_arrows(prod_ax, percent_ax,
                         [data_dict[key] for key in vals_on_prod_ax], 
                         [data_dict[key] for key in vals_on_percent_ax],
                         indep_var)
    
    if prod_log: prod_ax.set_yscale('log'); prod_ax.set_ylabel('Specific flow rate (mol/min/cm²)')
    prod_ax.yaxis.set_major_formatter('{y:.1e}')
    prod_ax.legend(loc='lower right', framealpha=0.9)

    percent_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    # percent_ax.set_ylim((0.95, 1))
    percent_ax.legend(loc='center right', framealpha=0.9)
    
    perf_fig.show()
    return perf_fig

def opt_progress(data_dict : Dict[str, List[float]]) -> Figure:
    """Plot optimization progress"""

    opt_prog_fig, (T_ax, N_ax, nit_ax) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(4.8, 9.6))
    P_ax = T_ax.twinx(); nfev_ax = nit_ax.twinx()
    indep_var = data_dict['spec_cond']
    nit_ax.set_xlabel('Specific conductance (S/cm²)')
    nit_ax.set_xscale('log')

    T_ax.plot(indep_var, data_dict['T'], label='T')
    for key in ['P_f', 'P_s']:
        P_ax.plot(indep_var, data_dict[key], label=key)
    annotate_axis_arrows(T_ax, P_ax,
                         [data_dict['T']], 
                         [data_dict[key] for key in ['P_f', 'P_s']],
                         indep_var)
    T_ax.set_ylabel('Temperature (°C)')
    P_ax.set_ylabel('Pressure (Pa)'); P_ax.yaxis.set_major_formatter('{x:.1e}')
    P_ax.legend(loc='upper center')
 
    N_ax.plot(indep_var, data_dict['N_f'], label='N_f')
    N_ax.plot(indep_var, data_dict['N_s'], label='N_s')
    N_ax.set_ylabel('Specific flowrate (mol min⁻¹S⁻¹)'); N_ax.yaxis.set_major_formatter('{x:.0e}')
    N_ax.legend(loc='upper center')

    nit_ax.plot(indep_var, data_dict['nit'], label='nit')
    nfev_ax.plot(indep_var, data_dict['nit'], label='nit')
    nit_ax.set_ylabel('# iterations')
    nfev_ax.set_ylabel('# function evaluations'); nfev_ax.yaxis.set_major_formatter('{x:.1e}');# nfev_ax.set_ylim(top=3.5e4)
    
    opt_prog_fig.show()
    return opt_prog_fig

if __name__ == '__main__':
    init_exp = Experiment(A_mem=1, sigma=0.4, L=500)
    tornado(Scenario_PM(), init_exp, from_cache=True)
