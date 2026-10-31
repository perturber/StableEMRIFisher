"""Fisher matrix of the case 1 EMRI, for the comparison with MCMC in docs/validation.

Computes the Fisher matrix through the LISA response (second-generation TDI,
A and E channels) with SciRDv1 noise including the galactic foreground, and
writes it to MCMC_FM_Data/fisher_matrices/case_1/Fisher_case_1.h5, where the
validation notebook reads it. Run from this directory; a GPU is recommended
(T = 2 years).
"""

import numpy as np
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
from few.utils.constants import YRSID_SI
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits

from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils.noise import load_psd

USE_GPU = True  # set to False to run on a CPU (slow for T = 2 years)

if USE_GPU:
    import cupy as xp
else:
    xp = np
backend = "gpu" if USE_GPU else "cpu"  # backend of the LISA response and orbits

T = 2.0  # observation time, years
dt = 5.0  # sampling interval, seconds

# case 1 source parameters
emri_params = dict(
    m1=1e6,  # primary mass, solar masses
    m2=10.0,  # secondary mass, solar masses
    a=0.998,  # primary spin (near-extremal)
    p0=7.7275,  # initial semi-latus rectum
    e0=0.73,  # initial eccentricity
    xI0=1.0,  # cosine of the initial inclination (equatorial orbit)
    dist=2.20360838037185,  # luminosity distance, Gpc (set for SNR 50)
    qS=0.8,  # sky location, polar angle
    phiS=2.2,  # sky location, azimuthal angle
    qK=1.6,  # spin orientation, polar angle
    phiK=1.2,  # spin orientation, azimuthal angle
    Phi_phi0=2.0,  # initial azimuthal phase
    Phi_theta0=0.0,  # initial polar phase
    Phi_r0=3.0,  # initial radial phase
)

# waveform model
waveform_class_kwargs = dict(
    inspiral_kwargs=dict(err=1e-11),  # trajectory integrator tolerance
    sum_kwargs=dict(pad_output=True),  # zero-pad to the full observation time if the inspiral plunges
    mode_selector_kwargs=dict(mode_selection_threshold=1e-5),
)
waveform_generator_kwargs = dict(return_list=False, frame="detector")

# LISA response: second-generation TDI on equal-arm-length orbits, A and E channels
ResponseWrapper_kwargs = dict(
    Tobs=T,
    dt=dt,
    index_lambda=8,  # position of phiS in the waveform arguments
    index_beta=7,  # position of qS in the waveform arguments
    t0=20000.0,  # seconds at each end of the signal where the response is unreliable
    remove_garbage="zero",  # zero the signal within t0 of each end
    flip_hx=True,  # FEW returns h+ - i hx
    is_ecliptic_latitude=False,  # qS is a polar angle, not a latitude
    force_backend=backend,
    orbits=EqualArmlengthOrbits(force_backend=backend),
    order=25,  # order of the Lagrange interpolation of the time delays
    tdi="2nd generation",
    tdi_chan="AE",
)

# noise: SciRDv1 PSDs of the second-generation TDI channels A and E, including the
# galactic foreground for the observation time T. The PSDs are written to the file
# once and loaded back as one interpolant returning both channels.
noise_model = load_psd(
    model="scirdv1",
    channels="AE",
    tdi2=True,
    include_foreground=True,
    stochastic_params=[T * YRSID_SI],
    filename=f"psd_scirdv1_AE_tdi2_foreground_{T}yr.npy",
    xp=xp,
)

sef = StableEMRIFisher(
    waveform_class=FastKerrEccentricEquatorialFlux,
    waveform_class_kwargs=waveform_class_kwargs,
    waveform_generator=GenerateEMRIWaveform,
    waveform_generator_kwargs=waveform_generator_kwargs,
    ResponseWrapper=ResponseWrapper,
    ResponseWrapper_kwargs=ResponseWrapper_kwargs,
    noise_model=noise_model,
    noise_kwargs={},  # the interpolant takes no kwargs
    channels=["A", "E"],
    der_order=4,
    deriv_type="stable",
    use_gpu=USE_GPU,
)

param_names = ["m1", "m2", "a", "p0", "e0", "dist", "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_r0"]

# trial step sizes for each parameter, from largest to smallest
Ndelta = 8
delta_range = dict(
    m1=np.geomspace(1e2, 1e-3, Ndelta),
    m2=np.geomspace(1e-1, 1e-7, Ndelta),
    a=np.geomspace(1e-5, 1e-9, Ndelta),
    p0=np.geomspace(1e-5, 1e-9, Ndelta),
    e0=np.geomspace(1e-5, 1e-9, Ndelta),
    qS=np.geomspace(1e-3, 1e-7, Ndelta),
    phiS=np.geomspace(1e-3, 1e-7, Ndelta),
    qK=np.geomspace(1e-3, 1e-7, Ndelta),
    phiK=np.geomspace(1e-3, 1e-7, Ndelta),
)

# writes MCMC_FM_Data/fisher_matrices/case_1/Fisher_case_1.h5 and stable_deltas_case_1.txt
fisher_matrix = sef(
    emri_params,
    param_names=param_names,
    Ndelta=Ndelta,
    delta_range=delta_range,
    filename="MCMC_FM_Data/fisher_matrices/case_1",
    suffix="case_1",
    stats_for_nerds=True,  # print the step size chosen for each parameter
)

sigma = np.sqrt(np.diag(np.linalg.inv(fisher_matrix)))
for name, s in zip(param_names, sigma):
    print(f"{name:9s} = {emri_params[name]:.6g} +/- {s:.3g}")
