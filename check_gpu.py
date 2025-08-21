import numpy as np
import matplotlib.pyplot as plt
from few.waveform import GenerateEMRIWaveform, FastSchwarzschildEccentricFlux, FastKerrEccentricEquatorialFlux
from few.utils.constants import Gpc, MRSUN_SI, YRSID_SI
from typing import Optional, Union, Callable
import tqdm

use_gpu = True #False if your computer sucks (mine does)

from stableemriderivative import StableEMRIDerivative

if not use_gpu:

    force_backend = "cpu" 
    import few
    
    #tune few configuration
    cfg_set = few.get_config_setter(reset=True)
    
    cfg_set.enable_backends("cpu")
    cfg_set.set_log_level("info");
else:
    force_backend = "gpu"
    pass #let the backend decide for itself.

from few.waveform import FastSchwarzschildEccentricFlux

print("Build the class")
waveform_derivative = StableEMRIDerivative(waveform_class=FastSchwarzschildEccentricFlux,
                                          mode_selector_kwargs=dict(mode_selection_threshold=1e-3), 
                                          inspiral_kwargs=dict(err=1e-11, max_iter=10000),
                                          force_backend = force_backend)



#=================== Now run the derivative and see what's going on ========================
m1 = 1e6
m2 = 1e1
a = 0.0 
p0 = 12.5
e0 = 0.4
xI0 = 1.0
dist = 1.0
qS = np.pi/3
phiS = np.pi/4
qK = np.pi/6
phiK = np.pi/8
Phi_phi0 = np.pi/4
Phi_theta0 = 0.0
Phi_r0 = 0.0

T = 2.0
dt = 10.0

pars_list = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
param_names = ['m1','m2','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0']

parameters = {}
for i in range(len(param_names)):
    parameters[param_names[i]] = pars_list[i]

param_to_vary = 'Phi_phi0'

delta = 1e-6  #finite difference delta for the chosen paramete
order = 4 #order of finite-difference derivative
kind = "central" #kind of finite-difference derivative

der = waveform_derivative(T = T, dt = dt, 
                    parameters = parameters, 
                    param_to_vary = param_to_vary,
                    delta = delta,
                    order = order,
                    kind = kind,
                    )

breakpoint()

                                        
