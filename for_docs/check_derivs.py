# Import relevant EMRI packages
from few.waveform import (
    GenerateEMRIWaveform,
    FastKerrEccentricEquatorialFlux,
)

# from stableemrifisher.fisher.derivatives import StableEMRIDerivative
from stableemrifisher.fisher.stablederivative import StableEMRIDerivative
from stableemrifisher.fisher.derivatives import derivative
import matplotlib.pyplot as plt
import numpy as np


YRSID_SI = 31558149.763545603
# Waveform params
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


EMRI_deriv = StableEMRIDerivative(
    FastKerrEccentricEquatorialFlux,
    waveform_generator=GenerateEMRIWaveform,
    waveform_generator_kwargs=waveform_generator_kwargs,
)

T = 0.01
dt = 10.0
kwargs = {"T": T, "dt": dt}
compute_stable_deriv = EMRI_deriv(
    parameters=wave_params,
    param_to_vary="m1",
    delta=1e-1,
    order=4,
    kind="central",
    **kwargs,
)

t = np.arange(0, T * YRSID_SI, dt)
plt.plot(t, compute_stable_deriv.real, c="blue", label="h_p: m1 partial derivative")
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Derivative", fontsize=16)
plt.title("EMRI Derivative with respect to m1", fontsize=16)
plt.grid(True)
plt.show()
