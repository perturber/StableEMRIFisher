from stableemrifisher.fisher import StableEMRIFisher
from few.waveform import GenerateEMRIWaveform
import matplotlib.pyplot as plt
import os
from pathlib import Path
import cupy as xp
import numpy as np
use_gpu=True

#set initial parameters
M = 1e6
mu = 10
a = 0.9
p0 = 8.05
e0 = 0.1
iota0 = 0.0 #equatorial
Y0 = np.cos(iota0)
Phi_phi0 = 2
Phi_theta0 = 3
Phi_r0 = 1.5

qS = 0.2
phiS = 0.2
qK = 0.8
phiK = 0.8
dist = 1
mich = True #mich = True implies output in hI, hII long wavelength approximation
dt = 10.0
T = 0.1

inspiral_kwargs = {
        "DENSE_STEPPING": 0,
        "max_init_len": int(1e3),
        "err": 1e-10,  # To be set within the class
        "use_rk4": True,
        }

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": True,  # GPU is available for this type of summation
    "pad_output": True,
}

amplitude_kwargs = {
    }

outdir = 'basic_usage_stability_outdir'

Path(outdir).mkdir(exist_ok=True)
waveform_model = GenerateEMRIWaveform('FastSchwarzschildEccentricFlux', return_list=True, inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)

waveform = xp.asarray(waveform_model(M, mu, a, p0, e0, Y0, 
                          dist, qS, phiS, qK, phiK, 
                          Phi_phi0, Phi_theta0, Phi_r0, 
                          mich=mich, dt=dt, T=T)).get()

plt.plot(waveform[0][-1000:])
plt.ylabel("h+")
plt.savefig(os.path.join(outdir, "waveform.png"))
plt.close()

#varied parameters
param_names = ['M','mu','p0','e0']

#initialization
fish = StableEMRIFisher(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
              Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T, EMRI_waveform_gen=waveform_model,
              param_names=param_names, stats_for_nerds=True, use_gpu=True,
              filename=outdir, CovEllipse=True, live_dangerously=False)


#execution
fish()
