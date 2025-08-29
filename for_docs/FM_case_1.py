# Import relevant EMRI packages
from few.waveform import (
    GenerateEMRIWaveform,
    FastKerrEccentricEquatorialFlux,
)

from fastlisaresponse import ResponseWrapper  # Response
from lisatools.detector import EqualArmlengthOrbits

import numpy as np

from stableemrifisher.fisher import StableEMRIFisher
    
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
    
    # Time domain setup
    "dt": 5.0,        # Time step (seconds)
    "T": 0.01,        # Observation time (years)
}



YRSID_SI = 31558149.763545603
ONE_HOUR = 60 * 60


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
RESPONSE_FUNCTION = True
USE_GPU = True

tdi_kwargs = dict(
    orbits=EqualArmlengthOrbits(use_gpu=USE_GPU),
    order=25,
    tdi="2nd generation",
    tdi_chan="AE",
)  # could do "AET"

INDEX_LAMBDA = 8
INDEX_BETA = 7

# with longer signals we care less about this
t0 = 20000.0  # throw away on both ends when our orbital information is weird

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
# noise setup

der_order = 4
Ndelta = 8

sef = StableEMRIFisher(
    waveform_class=waveform_class,
    waveform_class_kwargs=waveform_class_kwargs,
    waveform_generator=waveform_generator,
    waveform_generator_kwargs=waveform_generator_kwargs,
    ResponseWrapper=ResponseWrapper,
    ResponseWrapper_kwargs=ResponseWrapper_kwargs,
    dt = dt,
    T = T,
    stats_for_nerds=True,
    use_gpu=USE_GPU,
    der_order=der_order,
    Ndelta=Ndelta,
    return_derivatives=False,
    filename="MCMC_FM_Data/fisher_matrices/case_1_for_docs",
    deriv_type="stable",
)


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
derivs, fisher_matrix = sef(
    emri_params, param_names=param_names, delta_range=delta_range
)


param_cov = np.linalg.inv(fisher_matrix)

for k, item in enumerate(param_names):
    print(
        "Precision measurement in param {} is {}".format(
            item, param_cov[k, k] ** (1 / 2)
        )
    )
