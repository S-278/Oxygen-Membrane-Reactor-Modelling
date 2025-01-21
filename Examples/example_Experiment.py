from experiment import *
import numpy

# Create, run, and analyze an Experiment
e=Experiment(T=1000, P_s=1.1*101325) # All parameters not specified explicitly when constructing an Experiment are taken from Experiment.input_origin
e.print_input()
e.P_f = 0.9 * 101325
print(e.P_f) # Input parameters can be read and written directly as attributes
e.run()
e.analyze()
e.print_analysis()
print(e.H2O_conv) # Outputs and analysis can be read but not written as attributes

# Work with Experiments in arrays
exs = Experiment.grid(T=numpy.array([900,950,1000]),
                      P_f=numpy.array([101325, 1.1*101325, 1.2*101325]),
                      sigma=2)
for row in exs:
    for e in row:
        e.print_input()
    print('\n---------------\n')

for e in exs.flat:
    e.run()
    e.analyze()

for row in exs:
    for e in row:
        e.print_analysis()
    print('\n---------------\n')
