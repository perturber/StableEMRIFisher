# Import relevant EMRI packages
from few.waveform import (
    GenerateEMRIWaveform,
    FastSchwarzschildEccentricFlux,
    FastKerrEccentricEquatorialFlux,
)

from few.trajectory.ode import KerrEccEqFlux

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t
from few.utils.geodesic import get_separatrix

from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils import check_if_plunging
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens

from psd_utils import write_psd_file, load_psd_from_file
from EMRI_Params import (
    m1,
    m2,
    a,
    p0,
    e0,
    xI0,
    dist,
    qS,
    phiS,
    qK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
    T,
    dt,
)

import numpy as np

try:
    import cupy as cp

    xp = cp
except ImportError:
    pass
    xp = np
import time
import os

YRSID_SI = 31558149.763545603
ONE_HOUR = 60 * 60
# Waveform params
pars_list = [
    m1,
    m2,
    a,
    p0,
    e0,
    xI0,
    dist,
    qS,
    phiS,
    qK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
]

traj = EMRIInspiral(func=KerrEccEqFlux)  # Set up trajectory module, pn5 AAK
T = check_if_plunging(
    traj,
    T,
    m1,
    m2,
    a,
    p0,
    e0,
    xI0,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
    chop_inspiral_time=0.5,
)  # Remove 30 minutes if plunging


t_traj, p_traj, e_traj, xI_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj(
    m1, m2, a, p0, e0, xI0, Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T
)

print("Final time in years is ", t_traj[-1] / YRSID_SI)
traj_args = [m1, m2, a, e_traj[0], xI_traj[0]]
# Check to see what value of semi-latus rectum is required to build inspiral lasting T years.
p_new = get_p_at_t(traj, 2, traj_args, bounds=None)


print(
    "We require initial semi-latus rectum of ",
    p_new,
    "for inspiral lasting",
    T,
    "years",
)
print("Your chosen semi-latus rectum is", p0)
if p0 < p_new:
    print("Careful, the smaller body is plunging. Expect instabilities.")
else:
    print("Body is not plunging.")
print("Final point in semilatus rectum achieved is", p_traj[-1])
print("Separatrix : ", get_separatrix(a, e_traj[-1], xI_traj[-1]))

print(
    "Separation between separatrix and final p = ",
    abs(get_separatrix(a, e_traj[-1], 1.0) - p_traj[-1]),
)
print(f"Final eccentricity = {e_traj[-1]}")

# ========================= SET UP RESPONSE FUNCTION ===============================#
RESPONSE_FUNCTION = False
USE_GPU = False
if RESPONSE_FUNCTION:
    from fastlisaresponse import ResponseWrapper  # Response
    from lisatools.detector import EqualArmlengthOrbits

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
    # ResponseWrapper setup
    data_channels = ["TDIA", "TDIE"]
    N_channels = len(data_channels)
    print("cool")
else:
    ResponseWrapper = None
    ResponseWrapper_kwargs = None
    data_channels = ["I", "II"]
    N_channels = len(data_channels)

####=======================True Responsed waveform==========================
# waveform class setup
waveform_class = FastKerrEccentricEquatorialFlux
waveform_class_kwargs = dict(
    inspiral_kwargs=dict(
        err=1e-11,
    ),
    sum_kwargs=dict(pad_output=True),  # Required for plunging waveforms
    mode_selector_kwargs=dict(mode_selection_threshold=1e-5),
)

# waveform generator setup
waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = dict(return_list=False, frame="detector")


# noise setup

run_direc = os.getcwd()
PSD_filename = "tdi2_AE_w_background.npy"

kwargs_PSD = {"stochastic_params": [T * YRSID_SI]}  # We include the background

write_psd_file(
    model="scirdv1",
    channels="AE",
    tdi2=True,
    include_foreground=True,
    filename=run_direc + PSD_filename,
    **kwargs_PSD,
)

PSD_AE_interp = load_psd_from_file(run_direc + PSD_filename, xp=xp)

# channels = [A1TDISens, E1TDISens]
# noise_model = get_sensitivity
# noise_kwargs = [{"sens_fn": channel_i} for channel_i in channels]

noise_model = PSD_AE_interp
noise_kwargs = {}
channels = ["A", "E"]

# noise_model = None
# noise_kwargs = None
# channels = None
der_order = 4
Ndelta = 8
stability_plot = False

sef = StableEMRIFisher(
    waveform_class=waveform_class,
    waveform_class_kwargs=waveform_class_kwargs,
    waveform_generator=waveform_generator,
    waveform_generator_kwargs=waveform_generator_kwargs,
    ResponseWrapper=ResponseWrapper,
    ResponseWrapper_kwargs=ResponseWrapper_kwargs,
    dt = dt,
    T = T,
    noise_model=noise_model,
    noise_kwargs=noise_kwargs,
    channels=channels,
    stats_for_nerds=True,
    use_gpu=USE_GPU,
    der_order=der_order,
    Ndelta=Ndelta,
    stability_plot=stability_plot,
    return_derivatives=True,
    #    filename="MCMC_FM_Data/fisher_matrices/plunging_EMRI",
    filename=None,
    live_dangerously=False,
    deriv_type="stable",
)

# sef = StableEMRIFisher(waveform_class=waveform_class,
#                        waveform_class_kwargs=waveform_class_kwargs,
#                        waveform_generator=waveform_generator,
#                        waveform_generator_kwargs=waveform_generator_kwargs,
#                        noise_model=noise_model, noise_kwargs=noise_kwargs, channels=channels,
#                       stats_for_nerds = True, use_gpu = USE_GPU,
#                       deriv_type='stable')

param_names = [
    # "m1",
    # "m2",
    # "a",
    # "p0",
    # "e0",
    # "dist",
    "qS",
    "phiS",
    "qK",
    "phiK",
    "Phi_phi0",
    "Phi_r0",
]
# param_names = ['m1','m2','a','p0','e0','Phi_r0']

delta_range = dict(
    m1=np.geomspace(1e-4 * m1, 1e-9 * m1, Ndelta),
    m2=np.geomspace(1e-2 * m2, 1e-7 * m2, Ndelta),
    a=np.geomspace(1e-5, 1e-9, Ndelta),
    p0=np.geomspace(1e-5, 1e-9, Ndelta),
    e0=np.geomspace(1e-5, 1e-9, Ndelta),
    qS=np.array([1e-6]),  # Flat errors in angles
    phiS=np.array([1e-6]),  # Flat errors in angles
    qK=np.array([1e-6]),  # Flat errors in angles
    phiK=np.array([1e-6]),  # Flat errors in angles
)

print("Computing FM")
start = time.time()
derivs, fisher_matrix = sef(
    *pars_list, param_names=param_names, delta_range=delta_range
)
end = time.time() - start
print("Time taken to compute Fisher matrix and stable deltas is", end, "seconds")


param_cov = np.linalg.inv(fisher_matrix)

for k, item in enumerate(param_names):
    print(
        "Precision measurement in param {} is {}".format(
            item, param_cov[k, k] ** (1 / 2)
        )
    )
