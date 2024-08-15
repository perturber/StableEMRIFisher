import numpy as np
import cupy as cp
gpu_available = True

import sys
sys.path.append("..")
from fisher import StableEMRIFisher
import matplotlib.pyplot as plt

from few.waveform import Pn5AAKWaveform, GenerateEMRIWaveform, KerrEquatorialEccentric, KerrEquatorialEccentricWaveformBase

from few.trajectory.inspiral import EMRIInspiral

from few.amplitude.ampinterp2d import AmpInterpKerrEqEcc

from few.summation.aakwave import AAKSummation
from few.summation.directmodesum import DirectModeSum  
from few.summation.interpolatedmodesum import InterpolatedModeSum

from few.utils.utility import get_separatrix, Y_to_xI, get_p_at_t
from few.utils.modeselector import ModeSelector, NeuralModeSelector

from fastlisaresponse import ResponseWrapper  # Response function 
from few.utils.utility import get_separatrix, Y_to_xI, get_p_at_t

YRSID_SI = 31558149.763545603

# M = 2e6; mu = 20.0; a = 0.5; p0 = 8.44; e0 = 0.3;
# iota0 = 0.8; Y0 = np.cos(iota0); dist = 4.0; 
# qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6; 
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0; 

# M = 5e5; mu = 5.0; a = 0.95; p0 = 10.851; e0 = 0.3;
# iota0 = np.pi/3; Y0 = np.cos(iota0); dist = 3.0; 
# qS = 0.8; phiS = 0.6; qK = 0.8; phiK = 0.4; 
# Phi_phi0 = 2.5; Phi_theta0 = 2.5; Phi_r0 = 2.5;

# M = 2e6; mu = 20.0; a = 0.5; p0 = 8.44; e0 = 0.3;
# iota0 = 0.8; Y0 = np.cos(iota0); dist = 4.0; 
# qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6; 
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0; 

# Actually seems like a reasonable approximation after changing deltas
# M = 2e6; mu = 20.0; a = 0.5; p0 = 8.44; e0 = 0.3;
# iota0 = 0.8; Y0 = np.cos(iota0); dist = 4.0; 
# qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6; 
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0; 


# Extreme point
# M = 1e7; mu = 500.0; a = 0.99; p0 = 7.95; e0 = 0.2;
# iota0 = np.pi/3; Y0 = np.cos(iota0); dist = 1.0; 
# qS = 1.5; phiS = 0.2; qK = 1.2; phiK = 0.6; 
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0; 

#set initial parameters (default parameters in FEW 5PNAAK Documentation) -- These parameters seem to work!
# M = 1e6; mu = 10; a = 0.9; p0 = 9.2; e0 = 0.2; Y0 = np.cos(0.8)
# dist = 2.0; qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0

# Schwarzschild 

# M = 1e6; mu = 10; a = 0.0; p0 = 10.64; e0 = 0.2; Y0 = 1.0
# dist = 3.0; qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0

# Fully Relativistic Kerr 
# M = 1e6; mu = 10; a = 0.9; p0 = 10.68; e0 = 0.2; Y0 = 1.0
# dist = 1.0; qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0

# Using full Kerr model
M = 1e6; mu = 10; a = 0.9; p0 = 8.58; e0 = 0.2; Y0 = 1.0
dist = 1.0; qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6
Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0

# Using full kerr eccentric model, loud SNR ~ 174
# M = 1e6; mu = 10; a = 0.9; p0 = 8.58; e0 = 0.2; Y0 = 1.0
# dist = 1.0; qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0


# Kerr with a = -0.9, RETROGRADE
# M = 1e6; mu = 50; a = -0.9; p0 = 19.2; e0 = 0.3; Y0 = 1.0
# dist = 1.0; qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6
# Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0

params = [M,mu,a,p0,e0,Y0,dist,qS,phiS,qK,phiK,Phi_phi0, Phi_theta0, Phi_r0]
# Waveform params
dt = 10.0;  # Sampling interval [seconds]
T = 2.0     # Evolution time [years]

mich = False #mich = True implies output in hI, hII long wavelength approximation

use_gpu = True
traj = EMRIInspiral(func="pn5")  # Set up trajectory module, pn5 AAK
# traj = EMRIInspiral(func="SchwarzEccFlux")  # Set up trajectory module, pn5 AAK
# traj = EMRIInspiral(func="KerrEccentricEquatorial")  # Set up trajectory module, pn5 AAK

# Compute trajectory 
if a < 0:
    a_val = -1.0 * a
    Y0_val = -1.0 * Y0
else:
    a_val = a; Y0_val = Y0

t_traj, p_traj, e_traj, Y_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj(M, mu, a_val, p0, e0, Y0_val,
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T)
traj_args = [M, mu, a, e_traj[0], Y_traj[0]]
index_of_p = 3

# Check to see what value of semi-latus rectum is required to build inspiral lasting T years. 

p_new = get_p_at_t(
    traj,
    T,
    traj_args,
    index_of_p=3,
    index_of_a=2,
    index_of_e=4,
    index_of_x=5,
    xtol=2e-12,
    rtol=8.881784197001252e-16,
    bounds=[None,13],
)


print("We require initial semi-latus rectum of ",p_new, "for inspiral lasting", T, "years")
print("Your chosen semi-latus rectum is", p0)
if p0 < p_new:
    print("Careful, the smaller body is plunging. Expect instabilities.")
else:
    print("Body is not plunging.") 
print("Final point in semilatus rectum achieved is", p_traj[-1])
print("Separatrix : ", get_separatrix(a, e_traj[-1], Y_traj[-1]))

# model_choice = "Pn5AAKWaveform"
# model_choice = "FastSchwarzschildEccentricFlux"
model_choice = "KerrEccentricEquatorialFlux"

if model_choice == "KerrEccentricEquatorialFlux":
    inspiral_kwargs = {
            "DENSE_STEPPING": 0,
            "max_init_len": int(1e3),
            "err": 1e-10,  # To be set within the class
            "use_rk4": True,
            "integrate_phases":False,
            "func": "KerrEccentricEquatorialAPEX"
        }
    # keyword arguments for summation generator (AAKSummation)
    sum_kwargs = {
        "use_gpu": True,  # GPU is availabel for this type of summation
        "pad_output": True,
    }
    amplitude_kwargs = {
        "specific_spins":[0.80, 0.9, 0.95],
        "use_gpu" : True
        }
    
    Waveform_model = GenerateEMRIWaveform(
    KerrEquatorialEccentricWaveformBase, # Define the base waveform
    EMRIInspiral, # Define the trajectory
    AmpInterpKerrEqEcc, # Define the interpolation for the amplitudes
    InterpolatedModeSum, # Define the type of summation
    ModeSelector, # Define the type of mode selection
    # when using intrinsic only , we return a list
    inspiral_kwargs=inspiral_kwargs,
    sum_kwargs=sum_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    use_gpu=use_gpu,
    frame='detector'
    )
elif model_choice == "FastSchwarzschildEccentricFlux":
    inspiral_kwargs = {
            "DENSE_STEPPING": 0,
            "max_init_len": int(1e3),
            "err": 1e-10,
            "use_rk4":True,
            "integrate_phases":False
            }  
    sum_kwargs = {
        "use_gpu": True,  # GPU is availabel for this type of summation
        "pad_output": True,
    }
    Waveform_model = GenerateEMRIWaveform(model_choice, inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)
elif model_choice == "Pn5AAKWaveform":
    inspiral_kwargs = {
            "DENSE_STEPPING": 0,
            "max_init_len": int(1e4),
            "err": 1e-10,  # To be set within the class
            # "use_rk4": True,
            # "integrate_phases":False,
            "func":"pn5"
        }
    # keyword arguments for summation generator (AAKSummation)
    sum_kwargs = {
        "use_gpu": True,  # GPU is availabel for this type of summation
        "pad_output": True,
    }
    Waveform_model = GenerateEMRIWaveform(model_choice, inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)

t0 = 20000.0   # How many samples to remove from start and end of simulations.
order = 25

orbit_file = "../lisa-on-gpu/orbit_files/equalarmlength-trailing-fit.h5"
orbit_kwargs = dict(orbit_file=orbit_file)

# 1st or 2nd or custom (see docs for custom)
tdi_gen = "1st generation"

index_lambda = 8
index_beta = 7

tdi_kwargs_esa = dict(
    orbit_kwargs=orbit_kwargs, order=order, tdi=tdi_gen, tdi_chan="AET",
    )

EMRI_TDI = ResponseWrapper(Waveform_model,T,dt,
                index_lambda,index_beta,t0=t0,
                flip_hx = True, use_gpu = use_gpu, is_ecliptic_latitude=False,
                remove_garbage = "zero", **tdi_kwargs_esa)

window_function = None
#varied parameters
param_names = ['M','mu','a','p0','e0', "scalar_charge"]
# param_names = ['M','mu','a','p0','e0']

param_args = [0]
#initialization
sef = StableEMRIFisher(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
              Phi_phi0, Phi_theta0, Phi_r0, dt, T, param_args = param_args, EMRI_waveform_gen=EMRI_TDI,
              param_names=param_names, stats_for_nerds=True, 
              filename='TestRun', CovEllipse=True)

#execution
print("Computing FM")
import time
start = time.time()
sef()
end = time.time() - start
print("Time taken to compute Fisher matrix and stable deltas is",end,"seconds")

