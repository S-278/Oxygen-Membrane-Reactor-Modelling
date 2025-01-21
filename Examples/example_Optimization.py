from optimization import *
from process_model import *
import os

# Create an initialization Experiment that records the origin point of the optimization search space
e_init = Experiment(T=1000, 
                    N_f0 = 4e-4, x_f0="H2O:1",
                    N_s0 = 3e-4, x_s0="CH4:1",
                    )

# Create a process model that will act as the objective function to be optimized
pm = Eff_PM()
m = Metrics()

# Create the optimizer
optimizer = DE_Optimizer(pm.eval_experiment)

# Define lower and upper bounds for each of the dimensions of the search space
lb = DataArray(
    data=[900, 2e-4, 2e-4, 101325*0.5],
    coords=XA_COORDS)
ub = DataArray(
    data=[1100, 8e-4, 8e-4, 101325*2],
    coords=XA_COORDS)

# Tip: optimization can take 15-60 min. It's helpful to manually set the priority of the python process to high in your operating system's task manager.
print(f'PID: {os.getpid()}')

print("Starting optimizer...")
res = optimizer.optimize(e_init, Bounds(lb, ub))
print(res)
e_opt = optimizer.create_experiment_at(res.x)
e_opt.print_analysis()
opt_eff = pm.get_energy_eff(e_opt)
print(f'E eff.: {opt_eff}')
print("+++++++++++++++++ DONE ++++++++++++++++++")
