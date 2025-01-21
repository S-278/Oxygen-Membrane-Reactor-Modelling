from process_model import *

# Calculate an Experiment's energy efficiency using the Eff_PM
pm = Eff_PM()
m = Metrics() # Metrics objects are used as a nice way to store and print intermediate results from process models
e = Experiment(A_mem=10000, N_f0=4, N_s0=3)
eff = pm.get_energy_eff(e, metrics=m)
e.print_input()
print(eff)
print(m)

# Use the Scenario_PM
pm = Scenario_PM({'REACTOR_HEAT_LOSS': Scenarios.PESSIMISTIC, 'PUMPING_EFF': Scenarios.OPTIMISTIC})
# pm will use the pessimistic value for reactor heat loss and the optimistic value for pumping efficiency.
# All parameters not explicitly specified during construction will use central values.
eff = pm.get_energy_eff(e, metrics=m)
print(eff)
print(m)
