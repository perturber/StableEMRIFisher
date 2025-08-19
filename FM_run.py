# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform
from few.trajectory.ode import KerrEccEqFlux

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t
from few.utils.geodesic import get_separatrix

from stableemrifisher.fisher import StableEMRIFisher

import numpy as np
import time

YRSID_SI = 31558149.763545603

RESPONSE_FUNCTION = False
USE_GPU = False
if RESPONSE_FUNCTION:
    from fastlisaresponse import ResponseWrapper             # Response
    from lisatools.detector import EqualArmlengthOrbits

    tdi_kwargs_esa = dict(
        orbits=EqualArmlengthOrbits(use_gpu=USE_GPU),
        order=25,
        tdi="2nd generation",
        tdi_chan="AE",
    )  # could do "AET"

    INDEX_LAMBDA = 8
    INDEX_BETA = 7

    # with longer signals we care less about this
    t0 = 20000.0  # throw away on both ends when our orbital information is weird

    data_channels = ['TDIA','TDIE']
    N_channels = len(data_channels)
else:
    data_channels = ["I", "II"]
    N_channels = len(data_channels)

# Using full Kerr model
m1 = 1e6
m2 = 10
a = 0.9
p0 = 8.58
e0 = 0.2
Y0 = 1.0
dist = 1.0
qS = 1.5
phiS = 0.7
qK = 1.2
phiK = 0.0
Phi_phi0 = 2.0
Phi_theta0 = 3.0
Phi_r0 = 4.0

dt = 100.0  # Sampling interval [seconds]
T = 0.1     # Evolution time [years]

# Waveform params
params = [m1,m2,a,p0,e0,Y0,dist,qS,phiS,qK,phiK,Phi_phi0, Phi_theta0, Phi_r0]


## ===================== CHECK TRAJECTORY ====================
# 
traj = EMRIInspiral(func=KerrEccEqFlux)  # Set up trajectory module, pn5 AAK

t_traj, p_traj, e_traj, xI_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj(m1, m2, a, 
                                                                                 p0, e0, 1.0,
                                                                                 Phi_phi0=Phi_phi0, 
                                                                                 Phi_theta0=Phi_theta0, 
                                                                                 Phi_r0=Phi_r0, 
                                                                                 T=T)

traj_args = [m1, m2, a, e_traj[0], 1.0]
# Check to see what value of semi-latus rectum is required to build inspiral lasting T years.
p_new = get_p_at_t(
    traj,
    T,
    traj_args,
    bounds=None
)

print("We require initial semi-latus rectum of ",p_new, "for inspiral lasting", T, "years")
print("Your chosen semi-latus rectum is", p0)
if p0 < p_new:
    print("Careful, the smaller body is plunging. Expect instabilities.")
else:
    print("Body is not plunging.") 
print("Final point in semilatus rectum achieved is", p_traj[-1])
print("Separatrix : ", get_separatrix(a, e_traj[-1], xI_traj[-1]))

print("Separation between separatrix and final p = ",abs(get_separatrix(a,e_traj[-1],1.0) - p_traj[-1]))
print(f"Final eccentricity = {e_traj[-1]}")

if RESPONSE_FUNCTION:
    # Build the response wrapper
    print("Now going to load in class")
    EMRI_model_hp_hc = GenerateEMRIWaveform(
            "FastKerrEccentricEquatorialFlux",
            sum_kwargs=dict(pad_output=True),
            use_gpu=USE_GPU,
            return_list=False,
        )
    print("Building the responses!")
    EMRI_model = ResponseWrapper(
            EMRI_model_hp_hc,
            T,
            dt,
            INDEX_LAMBDA,
            INDEX_BETA,
            t0=t0,
            flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
            use_gpu=USE_GPU,
            is_ecliptic_latitude=False,  # False if using polar angle (theta)
            remove_garbage="zero",  # removes the beginning of the signal that has bad information
            **tdi_kwargs_esa,
        )
else:
    print("Building the waveform!")
    EMRI_model = GenerateEMRIWaveform(
            "FastKerrEccentricEquatorialFlux",
            sum_kwargs=dict(pad_output=True),
            use_gpu=USE_GPU,
            return_list=True,
        )
####=======================True Responsed waveform==========================


# varied parameters
param_names = ['m1','m2']

sef = StableEMRIFisher(EMRI_waveform_gen=EMRI_model,
                       param_names=param_names, stats_for_nerds=True,
                       filename='TestRun', CovEllipse=False)

# execution
print("Computing FM")
start = time.time()
fisher_matrix = sef(m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
                       Phi_phi0, Phi_theta0, Phi_r0, dt = dt, T = T)
end = time.time() - start
print("Time taken to compute Fisher matrix and stable deltas is", end, "seconds")

breakpoint()
param_cov = np.linalg.inv(fisher_matrix)

for k, item in enumerate(param_names):
    print("Precision measurement in param {} is {}".format(item, param_cov[k,k]))


