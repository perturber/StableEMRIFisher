"""Injection for exec.py."""

from few.waveform import FastKerrEccentricEquatorialFlux

waveform_class = FastKerrEccentricEquatorialFlux
use_response = True  # LISA response (2nd-gen TDI, A and E) on/off
use_gpu = True

T = 1.0  # observation time, years
dt = 10.0  # sampling interval, seconds

DeltaT = 0.01  # years; the source plunges at T + DeltaT (negative: within the observation window)
desired_snr = 50.0  # dist is set to 1 Gpc * snr_1gpc / desired_snr

# p0 and dist are computed by exec.py
emri_params = dict(
    m1=1e6,
    m2=10.0,
    a=0.9,
    e0=0.1,
    xI0=1.0,
    qS=0.8,
    phiS=2.2,
    qK=1.6,
    phiK=1.2,
    Phi_phi0=0.0,
    Phi_theta0=0.0,
    Phi_r0=0.0,
)

waveform_class_kwargs = dict(
    inspiral_kwargs=dict(err=1e-11),
    sum_kwargs=dict(pad_output=True),  # keeps every likelihood waveform at the full length
)
waveform_kwargs = dict(mode_selection_threshold=1e-5)
# without the response, drop qS, phiS, qK, phiK: they are degenerate for equatorial orbits
param_names = ["m1", "m2", "a", "p0", "e0", "dist", "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_r0"]

# optional (defaults in exec.py): add_param_args, prior_factor=10, nwalkers=4*ndim, nsteps=5000, burn=1000, seed
