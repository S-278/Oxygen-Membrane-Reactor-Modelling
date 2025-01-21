Prerequisites
--
- Numpy
- SciPy
- Cantera (tested with version 2.6.0)
- Matplotlib
- Pint
- xarray


Description
--
This is a collection of libraries using Kai Bittner's OMR model (see parent of this fork) to integrate the OMR model into a plant process model, optimize the operating point of an OMR, and visualize OMR data.

Modules and contents:
- `experiment.py`: object-oriented encapsulation for evaluation with the OMR model
- `process_model.py`: process model of an OMR plant
- `optimization.py`: optimization of an OMR plant's operating point
- `plotting.py`: some ways to visualize data from the OMR and plant process models
- `matplotlibrc`: optional runtime constants of matplotlib library
- `profiling.py`: starter script to profile optimization
- `OMR_Model_multiprc.py`: multiprocessing rework of OMR_Model
- `OMR_Model_lim.py`: fork of OMR_Model using the thermodynamic limit of infinite conductivity instead of user-specified conductivity
- `OMR_Model_profile.py`: fork of OMR_Model with more internal functions for more detailed profiling

```mermaid
---
title: Simplified relational model of classes & modules
---
classDiagram
class Experiment{
  +dict input_origin$
  +float T
  +float N_f0
  +String x_f0
  +float N_s0
  +String x_s0
  +float A_mem
  +float sigma
  +float L
  +float Lc
  +__init__(dict kwargs) Experiment
  +run() None
  +analyze() None
  +grid(dict)$ 
}
note for Experiment "Holds everything related \nto a single run of the \nOMR model in one object"
Experiment <|-- Experiment_T_dep_sigma
note for Experiment_T_dep_sigma "Like Experiment except \nsigma is temperature-dependent"

class ProcessModel {
  <<Abstract>>
  +eval_experiment(Experiment)* float
}
note for Process Model "Gives a scalar score for an Experiment"
ProcessModel ..> Experiment
class Eff_PM {
  +get_energy_eff(Experiment) float
  +eval_experiment(Experiment) float
}
ProcessModel ..|> Eff_PM
note for Eff_PM "The actual plant process model - \nevaluates energy efficiency \nof an Experiment under a \n'bandpass filter' of 2:1 syngas ratio"
Eff_PM <|-- Spec_N_o2_PM
note for Spec_N_o2_PM "Like Eff_PM but adds \nthe constraint of a \nminimum membrane \noxygen flux"
Eff_PM <|-- Scenario_PM
note for Scenario_PM "Like Eff_PM but \nmodel params have \npessimistic, central, \nand optimistic values"

class Optimizer{
  <<Abstract>>
  +__init__(Callable eval_funct) Optimizer
  +optimize(Experiment init_exp, Bounds bd) OptimizeResult
}
note for Optimizer "Optimizes the subset of Experiment \nparameters given by XA_COORDS \nto maximize a given \nevaluation function"
Optimizer ..> ProcessModel
Optimizer ..> Experiment
Optimizer ..|> DIRECT_Optimizer
Optimizer ..|> DE_Optimizer
DIRECT_Optimizer ..> scipy.optimize
DE_Optimizer ..> scipy.optimize
```
