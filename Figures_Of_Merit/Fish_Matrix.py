import numpy as np
import cupy as cp
gpu_available = True

import sys
sys.path.append("..")
from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.noise import noise_PSD_AE, sensitivity_LWA

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

orbit_file = "../../../Github_repositories/lisa-on-gpu/orbit_files/equalarmlength-trailing-fit.h5"
orbit_kwargs = dict(orbit_file=orbit_file)

# Using full Kerr model
M = 1e6; mu = 10.0; a = 0.9; p0 = 9.05; e0 = 0.2; Y0 = np.cos(0.3)
dist = 1.0; qS = 1.5; phiS = 0.7; qK = 1.2; phiK = 0.6
Phi_phi0 = 2.0; Phi_theta0 = 3.0; Phi_r0 = 4.0

params = [M,mu,a,p0,e0,Y0,dist,qS,phiS,qK,phiK,Phi_phi0, Phi_theta0, Phi_r0]
# Waveform params
dt = 10.0;  # Sampling interval [seconds]
T = 2.0     # Evolution time [years]

use_gpu = True
# traj = EMRIInspiral(func="pn5")  # Set up trajectory module, pn5 AAK
# traj = EMRIInspiral(func="SchwarzEccFlux")  # Set up trajectory module, pn5 AAK
# traj = EMRIInspiral(func="KerrEccentricEquatorial")  # Set up trajectory module, pn5 AAK

model_choice = "Pn5AAKWaveform"
# model_choice = "FastSchwarzschildEccentricFlux"
# model_choice = "KerrEccentricEquatorialFlux"

if model_choice == "KerrEccentricEquatorialFlux":
    inspiral_kwargs = {
            "DENSE_STEPPING": 0,
            "max_init_len": int(1e3),
            "err": 1e-13,  # To be set within the class
            "use_rk4": True,
            "func":"KerrEccentricEquatorial"
        }
    # keyword arguments for summation generator (AAKSummation)
    sum_kwargs = {
        "use_gpu": True,  # GPU is availabel for this type of summation
        "pad_output": True,
    }
    amplitude_kwargs = {
        # "specific_spins":[0.80, 0.9, 0.95],
        # "use_gpu" : True
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
            "err": 1e-4,
            "use_rk4":True,
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
            "err": 1e-14,  # To be set within the class
            "use_rk4": True,
        }
    # keyword arguments for summation generator (AAKSummation)
    sum_kwargs = {
        "use_gpu": True,  # GPU is availabel for this type of summation
        "pad_output": True,
    }
    Waveform_model = GenerateEMRIWaveform(model_choice, inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)

t0 = 20000.0   # How many samples to remove from start and end of simulations.
order = 25

# 1st or 2nd or custom (see docs for custom)
tdi_gen = "1st generation"

index_lambda = 8
index_beta = 7

tdi_kwargs_esa = dict(
    orbit_kwargs=orbit_kwargs, order=order, tdi=tdi_gen, tdi_chan="AE",
    )

EMRI_TDI = ResponseWrapper(Waveform_model,T,dt,
                index_lambda,index_beta,t0=t0,
                flip_hx = True, use_gpu = use_gpu, is_ecliptic_latitude=False,
                remove_garbage = "zero", **tdi_kwargs_esa)
#varied parameters
param_names = ['M','mu','a','p0','e0','Y0','dist','qS','phiS','qK', 'phiK', 'Phi_phi0', 'Phi_theta0', 'Phi_r0']
#initialization
fish = StableEMRIFisher(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
              Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T, EMRI_waveform_gen=EMRI_TDI,
              param_names=param_names, PSD = None, stats_for_nerds=True, use_gpu=True, 
              filename="FM_file") 

#execution
fisher_matrix = fish()
breakpoint()
Cov_Matrix = np.linalg.inv(fisher_matrix)
