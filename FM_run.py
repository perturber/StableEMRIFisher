# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform, FastSchwarzschildEccentricFlux, FastKerrEccentricEquatorialFlux

from few.trajectory.ode import KerrEccEqFlux

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t
from few.utils.geodesic import get_separatrix

from stableemrifisher.fisher import StableEMRIFisher
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens

import numpy as np
import time

# Using full Kerr model
m1 = 1e6
m2 = 10
a = 0.998
p0 = 7.7275
e0 = 0.73
xI0 = 1.0
dist = 4.681378287352086
qS = 0.8
phiS = 2.2
qK = 1.6
phiK = 1.2
Phi_phi0 = 2.0
Phi_theta0 = 0.0
Phi_r0 = 4.0


dt = 100.0  # Sampling interval [seconds]
T = 0.1     # Evolution time [years]

# Waveform params
pars_list = [m1,m2,a,p0,e0,xI0,dist,qS,phiS,qK,phiK,Phi_phi0, Phi_theta0, Phi_r0]
YRSID_SI = 31558149.763545603

RESPONSE_FUNCTION = True
USE_GPU = True
if RESPONSE_FUNCTION:
    from fastlisaresponse import ResponseWrapper             # Response
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
        #waveform_gen=waveform_generator,
        Tobs = T,
        dt = dt,
        index_lambda = INDEX_LAMBDA,
        index_beta = INDEX_BETA,
        t0 = t0,
        flip_hx = True,
        use_gpu=USE_GPU,
        is_ecliptic_latitude=False,
        remove_garbage="zero",
        **tdi_kwargs
    )
    #ResponseWrapper setup
    data_channels = ['TDIA','TDIE']
    N_channels = len(data_channels)
else:
    data_channels = ["I", "II"]
    N_channels = len(data_channels)



## ===================== CHECK TRAJECTORY ====================
# 
traj = EMRIInspiral(func=KerrEccEqFlux)  # Set up trajectory module, pn5 AAK

t_traj, p_traj, e_traj, xI_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj(m1, m2, a, 
                                                                                 p0, e0, 1.0,
                                                                                 Phi_phi0=Phi_phi0, 
                                                                                 Phi_theta0=Phi_theta0, 
                                                                                 Phi_r0=Phi_r0, 
                                                                                 T=T)

breakpoint()
traj_args = [m1, m2, a, e_traj[0], 1.0]
# Check to see what value of semi-latus rectum is required to build inspiral lasting T years.
p_new = get_p_at_t(
    traj,
    2,
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


####=======================True Responsed waveform==========================
#waveform class setup
breakpoint()
waveform_class = FastKerrEccentricEquatorialFlux
waveform_class_kwargs = dict(inspiral_kwargs=dict(err=1e-11,),
                             mode_selector_kwargs=dict(mode_selection_threshold=1e-2))

#waveform generator setup
waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = dict(return_list=False)


#noise setup
channels = [A1TDISens, E1TDISens]
noise_model = get_sensitivity
noise_kwargs = [{"sens_fn": channel_i} for channel_i in channels]
breakpoint()
sef = StableEMRIFisher(waveform_class=waveform_class, 
                       waveform_class_kwargs=waveform_class_kwargs,
                       waveform_generator=waveform_generator,
                       waveform_generator_kwargs=waveform_generator_kwargs,
                       ResponseWrapper=ResponseWrapper, ResponseWrapper_kwargs=ResponseWrapper_kwargs,
                       noise_model=noise_model, noise_kwargs=noise_kwargs, channels=channels,
                      stats_for_nerds = True, use_gpu = USE_GPU,
                      deriv_type='stable')

breakpoint()
param_names = ['m1','m2','a','p0','e0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_r0']
der_order = 4
Ndelta = 8
stability_plot = True

delta_range = dict(
    m1 = np.geomspace(1e-4*m1, 1e-9*m1, Ndelta),
    m2 = np.geomspace(1e-2*m2, 1e-7*m2, Ndelta),
    p0 = np.geomspace(1e-2*p0, 1e-7*p0, Ndelta),
    e0 = np.geomspace(1e-1*e0, 1e-7*e0, Ndelta),
    qS = np.geomspace(1e-4,    1e-9,    Ndelta),
    phiS = np.geomspace(1e-4,    1e-9,    Ndelta),
    qK = np.geomspace(1e-4,    1e-9,    Ndelta),
    phiK = np.geomspace(1e-4,    1e-9,    Ndelta),
)

print("Computing FM")
start = time.time()
fisher_matrix = sef(*pars_list, param_names = param_names, 
             T = T, dt = dt, 
             der_order = der_order, 
             Ndelta = Ndelta, 
             stability_plot = stability_plot,
             delta_range = delta_range,
            live_dangerously = True)
end = time.time() - start
print("Time taken to compute Fisher matrix and stable deltas is", end, "seconds")


breakpoint()
param_cov = np.linalg.inv(fisher_matrix)

for k, item in enumerate(param_names):
    print("Precision measurement in param {} is {}".format(item, param_cov[k,k]))


