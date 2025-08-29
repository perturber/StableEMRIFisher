# Import relevant EMRI packages
from few.waveform import (
    GenerateEMRIWaveform,
    FastKerrEccentricEquatorialFlux,
)
from stableemrifisher.fisher import StableEMRIFisher

from fastlisaresponse import ResponseWrapper             # Response
from lisatools.detector import EqualArmlengthOrbits

import numpy as np
# Waveform params
dt = 5.0
T = 0.01
wave_params = {
    "m1": 1e6,
    "m2": 1e1,
    "a": 0.9,
    "p0": 10,
    "e0": 0.4,
    "xI0": 1.0,
    "dist": 1.0,
    "qS": 0.2,
    "phiS": 0.8,
    "qK": 1.6,
    "phiK": 1.5,
    "Phi_phi0": 2.0,
    "Phi_theta0": 0.0,
    "Phi_r0": 3.0,
}


####=======================True Responsed waveform==========================
waveform_class = FastKerrEccentricEquatorialFlux
waveform_class_kwargs = {
    "inspiral_kwargs": {
        "err": 1e-11,
    },
    "mode_selector_kwargs": {"mode_selection_threshold": 1e-5},
}
# waveform generator setup
waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = {"return_list": False, "frame": "detector"}
# Response function set up
USE_GPU = False
tdi_kwargs = dict(
    orbits=EqualArmlengthOrbits(use_gpu=USE_GPU),
    order=25,
    tdi="2nd generation",
    tdi_chan="AE",
)  

INDEX_LAMBDA = 8
INDEX_BETA = 7

# with longer signals we care less about this
t0 = 20000.0  # throw away on both ends when our orbital information is weird

ResponseWrapper_kwargs = dict(
    Tobs = T,
    dt = dt,
    index_lambda = INDEX_LAMBDA,
    index_beta = INDEX_BETA,
    t0 = t0,
    flip_hx = True,
    use_gpu=USE_GPU,
    is_ecliptic_latitude=False,
    remove_garbage="zero",
    **tdi_kwargs
)

der_order = 4
Ndelta = 8
stability_plot = False
sef = StableEMRIFisher(waveform_class=waveform_class, 
                       waveform_class_kwargs=waveform_class_kwargs,
                       waveform_generator=waveform_generator,
                       waveform_generator_kwargs=waveform_generator_kwargs,
                       ResponseWrapper=ResponseWrapper, ResponseWrapper_kwargs=ResponseWrapper_kwargs,
                       stats_for_nerds = True, use_gpu = USE_GPU,
                        T = T, dt = dt,
                        der_order = der_order,
                        Ndelta = Ndelta,
                        stability_plot = stability_plot,
                        return_derivatives = False,
                       deriv_type='stable')

param_names = ['m1','m2','a']

delta_range = dict(
    m1 = np.geomspace(1e3, 1e-5, Ndelta),
    m2 = np.geomspace(1e-2, 1e-8, Ndelta),
    a = np.geomspace(1e-5, 1e-9, Ndelta),
)

fisher_matrix = sef(wave_params, param_names = param_names, 
                        delta_range = delta_range,
                        filename=None,
                        live_dangerously = False)


param_cov = np.linalg.inv(fisher_matrix)

for k, item in enumerate(param_names):
    print("Precision measurement in param {} is {}".format(item, param_cov[k,k]**(1/2)))

# ================= Compute fluctuation due to noise realisation ========================