# Import relevant EMRI packages
from few.waveform import (
    GenerateEMRIWaveform,
    FastKerrEccentricEquatorialFlux,
)

from fastlisaresponse import ResponseWrapper  # Response
from lisatools.detector import EqualArmlengthOrbits

import numpy as np

from stableemrifisher.fisher import StableEMRIFisher
from few.utils.constants import YRSID_SI

ONE_HOUR = 60 * 60
# ================== CASE 1 PARAMETERS ======================
T = 2.0
dt = 5.0
# EMRI Case 1 parameters as dictionary
emri_params = {
    # Masses and spin
    "m1": 1e6,        # Primary mass (solar masses)
    "m2": 10,         # Secondary mass (solar masses) 
    "a": 0.998,       # Dimensionless spin parameter (near-extremal)
    
    # Orbital parameters
    "p0": 7.7275,     # Initial semi-latus rectum
    "e0": 0.73,       # Initial eccentricity
    "xI0": 1.0,       # cos(inclination) - equatorial orbit
    
    # Source properties
    "dist": 2.20360838037185,  # Distance (Gpc) - calibrated for target SNR
    
    # Sky location (source frame)
    "qS": 0.8,        # Polar angle
    "phiS": 2.2,      # Azimuthal angle
    
    # Kerr spin orientation
    "qK": 1.6,        # Spin polar angle
    "phiK": 1.2,      # Spin azimuthal angle
    
    # Initial phases
    "Phi_phi0": 2.0,    # Azimuthal phase
    "Phi_theta0": 0.0,  # Polar phase
    "Phi_r0": 3.0,      # Radial phase
}




####=======================True Responsed waveform==========================
# waveform class setup
waveform_class = FastKerrEccentricEquatorialFlux
waveform_class_kwargs = {
    "inspiral_kwargs": {
        "err": 1e-11,
    },
    "sum_kwargs": {
        "pad_output": True
    },  # Required for plunging waveforms
    "mode_selector_kwargs": {
        "mode_selection_threshold": 1e-5
    },
}

# waveform generator setup
waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = {
    "return_list": False, 
    "frame": "detector"
}


# ========================= SET UP RESPONSE FUNCTION ===============================#
USE_GPU = True

tdi_kwargs = dict(
    orbits=EqualArmlengthOrbits(use_gpu=USE_GPU),
    order=25,  # Order of Lagrange interpolant, used for fractional delays.
    tdi="2nd generation", # Use second generation TDI variables
    tdi_chan="AE",
) 

INDEX_LAMBDA = 8
INDEX_BETA = 7

t0 = 20000.0  # throw away on both ends when our orbital information is weird

# Set up Response key word arguments
ResponseWrapper_kwargs = dict(
    Tobs=T,
    dt=dt,
    index_lambda=INDEX_LAMBDA,
    index_beta=INDEX_BETA,
    t0=t0,
    flip_hx=True,
    use_gpu=USE_GPU,
    is_ecliptic_latitude=False,
    remove_garbage="zero",
    **tdi_kwargs,
)

der_order = 4   # Fourth order derivatives

Ndelta = 8      # Check 8 possible delta values to check convergence of derivatives

# No noise model provided so will default to TDI2 A and E channels with galactic confusion noise
# extracts relevant noise model from information provided to tdi_kwargs. 

# Initialise fisher matrix
sef = StableEMRIFisher(
    # Set up waveform class 
    waveform_class=waveform_class,
    waveform_class_kwargs=waveform_class_kwargs,
    # Set up waveform generator
    waveform_generator=waveform_generator,
    waveform_generator_kwargs=waveform_generator_kwargs,
    # Set up response
    ResponseWrapper=ResponseWrapper,
    ResponseWrapper_kwargs=ResponseWrapper_kwargs,
    stats_for_nerds=True,       # Output useful information governing stability
    use_gpu=USE_GPU,            # select whether or not to use gpu
    der_order=der_order,        # derivative order
    Ndelta=Ndelta,              # delta spacing
    return_derivatives=False,   # Do not return derivatives 
    filename="MCMC_FM_Data/fisher_matrices/case_1_for_docs",
    deriv_type="stable",  # Type of derivative
)

# Specify full parameter set to compute Fisher matrix over
param_names = [
    "m1",
    "m2",
    "a",
    "p0",
    "e0",
    "dist",
    "qS",
    "phiS",
    "qK",
    "phiK",
    "Phi_phi0",
    "Phi_r0",
]

# Compute specific delta ranges
delta_range = dict(
    m1=np.geomspace(1e2, 1e-3 , Ndelta),
    m2=np.geomspace(1e-1, 1e-7, Ndelta),
    a=np.geomspace(1e-5, 1e-9, Ndelta),
    p0=np.geomspace(1e-5, 1e-9, Ndelta),
    e0=np.geomspace(1e-5, 1e-9, Ndelta),
    qS=np.geomspace(1e-3, 1e-7, Ndelta),
    phiS=np.geomspace(1e-3, 1e-7, Ndelta),
    qK=np.geomspace(1e-3, 1e-7, Ndelta),
    phiK=np.geomspace(1e-3, 1e-7, Ndelta)
)

print("Computing FM")
# Compute the fisher matrix
fisher_matrix = sef(
    emri_params, param_names=param_names, delta_range=delta_range
)

# Compute paramter covariance matrix
param_cov = np.linalg.inv(fisher_matrix)

# Print precision measurements on parameters
for k, item in enumerate(param_names):
    print(
        "Precision measurement in param {} is {}".format(
            item, param_cov[k, k] ** (1 / 2)
        )
    )
