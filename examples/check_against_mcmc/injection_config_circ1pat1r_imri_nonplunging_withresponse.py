"""Injection for exec.py."""

from few.waveform import Circ1PAT1R

save_dir = "circ1pat1r_imri_nonplunging_withresponse"  # results folder, relative to this file

waveform_class = Circ1PAT1R
use_response = True  # LISA response (1st-gen TDI, A and E) on/off
use_gpu = True

deriv_type = "stable"  # "stable" (StableDerivative) or "direct" (DirectDerivative)
der_order = 4  # finite-difference order

T = 0.25  # observation time, years
dt = 10.0  # sampling interval, seconds

DeltaT = 0.001  # years; the source plunges at T + DeltaT (negative: within the observation window)
desired_snr = 50.0  # dist is set to 1 Gpc * snr_1gpc / desired_snr

# p0 and dist are computed by exec.py
emri_params = dict(
    m1=1e6,
    m2=1e3,
    a=0.9,
    e0=0.0,  # not used by Circ1PAT1R
    xI0=1.0,  # not used by Circ1PAT1R
    qS=0.8,
    phiS=2.2,
    qK=1.6,
    phiK=1.2,
    Phi_phi0=0.0,
    Phi_theta0=0.0,  # not used by Circ1PAT1R
    Phi_r0=0.0,  # not used by Circ1PAT1R
)
add_param_args = dict(chi2=0.95)  # secondary spin

waveform_class_kwargs = dict(
    inspiral_kwargs=dict(
        evolve_primary=False,  # hold the primary's mass and spin fixed
        err=1e-11,
    ),
    sum_kwargs=dict(pad_output=True),  # keeps every likelihood waveform at the full length
)
waveform_kwargs = dict(mode_selection_threshold=1e-5)
# without the response, drop qS, phiS, qK, phiK: they are degenerate for equatorial orbits
param_names = ["m1", "m2", "a", "p0", "dist", "qS", "phiS", "qK", "phiK", "Phi_phi0", "chi2"]

# Eryn sampler
prior_factor = 10.0  # uniform prior: injection +/- prior_factor * Fisher sigma
ntemps = 8  # parallel-tempering temperatures; the cold chain (beta = 1) is the posterior
nwalkers = 24  # per temperature; even, and at least 2 * len(param_names)
nsteps = 5000  # stored iterations, after burn-in
burn = 1000  # discarded iterations
stretch_weight = 0.5  # StretchMove weight; the Gaussian move (Fisher covariance) gets 1 - stretch_weight
seed = 42  # walker start and Fisher samples in the plot; None for a random seed
