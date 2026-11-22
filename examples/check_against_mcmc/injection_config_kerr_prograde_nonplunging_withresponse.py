"""Injection for exec.py."""

from few.waveform import FastKerrEccentricEquatorialFlux

save_dir = "kerr_prograde_nonplunging_withresponse"  # results folder, relative to this file

waveform_class = FastKerrEccentricEquatorialFlux
use_response = True  # LISA response (1st-gen TDI, A and E) on/off
use_gpu = True

deriv_type = "stable"  # "stable" (StableDerivative) or "direct" (DirectDerivative)
der_order = 4  # finite-difference order

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

# Eryn sampler
prior_factor = 10.0  # uniform prior: injection +/- prior_factor * Fisher sigma
prior_bounds = {}  # absolute (lower, upper) prior limits per parameter, e.g. dict(a=(-0.99, 0.99))
ntemps = 8  # parallel-tempering temperatures; the cold chain (beta = 1) is the posterior
nwalkers = 24  # per temperature; even, and at least 2 * len(param_names)
nsteps = 2000  # stored iterations, after burn-in
burn = 500  # discarded iterations
stretch_weight = 0.5  # StretchMove weight; the Gaussian move (Fisher covariance) gets 1 - stretch_weight
seed = 42  # walker start and Fisher samples in the plot; None for a random seed

# optional: add_param_args (e.g. dict(chi2=0.9) for Circ1PAT1R)
