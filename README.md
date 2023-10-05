Prequisites
--
- Numpy
- Scipy
- Cantera (tested with version 2.6.0)
- Matplotlib (used for plotting in the examples)


Description
--
The Python script OMR_model.py allows to simulate oxygen membrane reactors with continuos gas flow rates as described in our publication [1].

On both sides of the membrane, chemical equilibrium is assumed. The oxygen flux through the membrane is modelled by the Wagner equation and included into the chemical equilibrium calculation.
The entire problem is then solved as a nested problem: The inner problem is the equilibrium calculation inluding an assumed oxygen flux using Cantera [2]. 
The outer problem is a root finding problem to find the oxygen flux satisfying the Wagner equation in the equilibrium state which is done using Scipy [3].
A detailed explanation of the assumptions, limitations and equations including experimental validation can be found in our publication [1].

The implementation published here uses thermodynamic data from the Gri 3.0 mechanism [4] and subsequently considers 53 species. 
Notable species herein include: [O2, H2O, H2, CH4, CO, CO2, N2, AR]



Usage
--------------------------------------------------------------------------------------------------------------------------------------
To import the script into your Python code, download OMR_model.py and import the function Simulate_OMR at the beginning of the code as:


`from OMR_model import Simulate_OMR`

After defining the input parameters for the model [T,N_f0,x_f0,P_f,N_s0,x_s0,P_s,A_mem,sigma,L,Lc], a simulation can then be performed by calling:

`N_f, x_f, p_o2_f, N_s, x_s, p_o2_s, N_o2, dH, x_comp, conv = Simulate_OMR(T,N_f0,x_f0,P_f,N_s0,x_s0,P_s,A_mem,sigma,L,Lc)`

Examples for using the function with scalar input parameters, as well as array shaped input parameters are given in the Examples folder.



Inputs and outputs of the model
--

    Parameters (Inputs)
    ----------
    T : Float (scalar or array with length n)
        Temperature in °C.
    N_f0 : Float (scalar or array with length n)
        Initial feed gas molar flow rate in mol/min.
    x_f0 : String or list of strings with length n
        Initial feed gas mole fractions specified as a string in the format 'A:x_A_f0, B:x_B_f0, C:x_C_f0, ...', where x_A_f0, x_B_f0, x_C_f0 are the mole fractions of the respective species.
    P_f : Float (scalar or array with length n)
        Feed gas pressure in Pa.
    N_s0 : Float (scalar or array with length n)
        Initial sweep gas molar flow rate in mol/min.
    x_s0 : String or list of strings with length n
        Initial sweep gas mole fractions specified as a string in the format 'A:x_A_s0, B:x_B_s0, C:x_C_s0, ...', where x_A_s0, x_B_s0, x_C_s0 are the mole fractions of the respective species.
    P_s : Float (scalar or array with length n)
        Sweep gas pressure in Pa.
    A_mem : Float (scalar or array with length n)
        Active membrane area in cm^2.
    sigma : Float (scalar or array with length n)
        Ambipolar conductivity in S/m.
    L : Float (scalar or array with length n)
        Membrane thickness in mum.
    Lc : Float (scalar or array with length n)
        Characteristic length in mum.

    Returns (Outputs)
    -------
    N_f : Array of floats with length n
        Feed gas molar flow rate in mol/min.
    x_f : List of array of floats with size n*53
        Mole fraction array, consisting of the mole fractions of the feed gas, related to the composition array.
    p_o2_f : Float (array with length n)
        Feed gas oxygen partial pressure in Pa.
    N_s : Float (array with length n)
        Sweep gas molar flow rate in mol/min.
    x_s : List of array of floats with size n*53
        Mole fraction array, consisting of the mole fractions of the sweep gas, related to the composition array.
    p_o2_s : oat (array with length n)
        Sweep gas oxygen partial pressure in Pa.
    p_o2_f : Float (array with length n)
        Feed gas oxygen partial pressure in Pa.
    N_o2 : Float (array with length n)
        Molar oxygen flux through the membrane in mol/min.
    dH : List of array of floats with size n*53
        Outlet-Inlet enthalpy flow difference (reaction heat) in W; If positive: The reaction is endothermic; If negative: The reaction is exothermic.
    x_comp : List of strings with size 53
        Composition array consisting of the considered species in the calculation.
    conv : Integer (array with length n)
        Check whether convergence was achieved; Equal to 1 if converged.



References
--
[1] Bittner, K., Margaritis, N., Schulze-Küppers, F., Wolters, J., & Natour, G. (2023). 
    A mathematical model for initial design iterations and feasibility studies of oxygen membrane reactors by minimizing Gibbs free energy. 
    Journal of Membrane Science, 685, 121955.
    DOI: https://doi.org/10.1016/j.memsci.2023.121955

[2] David G. Goodwin, Harry K. Moffat, Ingmar Schoegl, Raymond L. Speth, and Bryan W. Weber. Cantera: An object-oriented software toolkit for chemical kinetics, 
    thermodynamics, and transport processes. 
    https://www.cantera.org, 2023. Version 3.0.0. doi:10.5281/zenodo.8137090

[3] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson,
    Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, 
    Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, 
    Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro,
    Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. 
    (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.
    DOI: https://doi.org/10.1038/s41592-019-0686-2

[4] G.P. Smith, D.M. Golden, M. Frenklach, N.W. Moriarty, B. Eiteneer, M. Goldenberg, C.T. Bowman, R.K. Hanson, S. Song, 
    W.C.J. Gardiner, V.V. Lissianski, Z. Qin, Gri-Mech 3.0~~

