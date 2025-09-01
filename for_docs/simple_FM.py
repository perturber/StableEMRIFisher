# Import relevant EMRI packages
from few.waveform import (
    GenerateEMRIWaveform,
    FastKerrEccentricEquatorialFlux,
)
from stableemrifisher.fisher import StableEMRIFisher

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

# waveform class setup
waveform_class_kwargs = {
    "inspiral_kwargs": {
        "err": 1e-11,
    },
    "mode_selector_kwargs": {"mode_selection_threshold": 1e-5},
}

# waveform generator setup
waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = {"return_list": False, "frame": "detector"}


der_order = 4
Ndelta = 8
stability_plot = False

sef = StableEMRIFisher(
    waveform_class=FastKerrEccentricEquatorialFlux,
    waveform_class_kwargs=waveform_class_kwargs,
    waveform_generator=GenerateEMRIWaveform,
    waveform_generator_kwargs=waveform_generator_kwargs,
    dt=dt,
    T=T,
    der_order=der_order,
    Ndelta=Ndelta,
    deriv_type="stable",
)


param_names = [
    "m1",
    "m2",
    "a",
]

deltas = np.array([1e-1, 1e-6, 1e-7])
delta_range = dict(
    m1=np.geomspace(1e2, 1e-3, Ndelta),
    m2=np.geomspace(1e-3, 1e-8, Ndelta),
    a=np.geomspace(1e-4, 1e-9, Ndelta),
)

fisher_matrix = sef(wave_params, param_names=param_names, delta_range=delta_range)

param_cov = np.linalg.inv(fisher_matrix)

for k, item in enumerate(param_names):
    print(
        "Precision measurement in param {} is {}".format(
            item, param_cov[k, k] ** (1 / 2)
        )
    )
