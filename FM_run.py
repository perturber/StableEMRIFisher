# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

from few.trajectory.ode import KerrEccEqFlux

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t
from few.utils.geodesic import get_separatrix

from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils import check_if_plunging
from stableemrifisher.noise import write_psd_file, load_psd_from_file

from psd_utils import (write_psd_file, load_psd_from_file)
from EMRI_Params import (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T, dt)

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
ONE_HOUR = 60*60
# Waveform params
pars_list = [m1,m2,a,p0,e0,xI0,dist,qS,phiS,qK,phiK,Phi_phi0, Phi_theta0, Phi_r0]

traj = EMRIInspiral(func=KerrEccEqFlux)  # Set up trajectory module, pn5 AAK
T = check_if_plunging(traj, T, m1, m2,a,p0,e0,xI0,Phi_phi0, Phi_theta0, Phi_r0, chop_inspiral_time = 0.5) # Remove 30 minutes if plunging


t_traj, p_traj, e_traj, xI_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj(m1, m2, a, 
                                                                                 p0, e0, xI0,
                                                                                 Phi_phi0=Phi_phi0, 
                                                                                 Phi_theta0=Phi_theta0, 
                                                                                 Phi_r0=Phi_r0, 
                                                                                 T=T)
if t_traj[-1] < T*YRSID_SI:
    print("Ah, looks like things are plunging, nightmare. redefining time T")
    end_time_seconds = (t_traj[-1]/YRSID_SI)
    T = end_time_seconds - 0.5*(ONE_HOUR)/YRSID_SI
    t_traj, p_traj, e_traj, xI_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj(m1, m2, a, 
                                                                                 p0, e0, xI0,
                                                                                 Phi_phi0=Phi_phi0, 
                                                                                 Phi_theta0=Phi_theta0, 
                                                                                 Phi_r0=Phi_r0, 
                                                                                 T=T)



print("Final time in years is ", t_traj[-1]/YRSID_SI)
traj_args = [m1, m2, a, e_traj[0], xI_traj[0]]
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

# ========================= SET UP RESPONSE FUNCTION ===============================#
RESPONSE_FUNCTION = False
USE_GPU = False
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
    ResponseWrapper = None
    ResponseWrapper_kwargs = None
    data_channels = ["I", "II"]
    N_channels = len(data_channels)

T = 0.1
####=======================True Responsed waveform==========================
#waveform class setup
waveform_class = FastKerrEccentricEquatorialFlux
waveform_class_kwargs = dict(inspiral_kwargs=dict(err=1e-11,),
                             sum_kwargs=dict(pad_output=True), # Required for plunging waveforms
                             mode_selector_kwargs=dict(mode_selection_threshold=1e-5))

#waveform generator setup
waveform_generator = GenerateEMRIWaveform
if ResponseWrapper:
    waveform_generator_kwargs = dict(return_list=False)
else:
    waveform_generator_kwargs = dict(return_list=True)



#noise setup

run_direc = os.getcwd()
PSD_filename = "tdi2_AE_w_background.npy"

kwargs_PSD = {"stochastic_params": [T*YRSID_SI]} # We include the background

write_PSD = write_psd_file(model='scirdv1', channels='AE', 
                           tdi2=True, include_foreground=True, 
                           filename = run_direc + PSD_filename, **kwargs_PSD)

PSD_AE_interp = load_psd_from_file(run_direc + PSD_filename, xp=cp)

# channels = [A1TDISens, E1TDISens]
# noise_model = get_sensitivity
# noise_kwargs = [{"sens_fn": channel_i} for channel_i in channels]

noise_model = PSD_AE_interp
noise_kwargs = {}
channels = ["A", "E"]
sef = StableEMRIFisher(waveform_class=waveform_class, 
                       waveform_class_kwargs=waveform_class_kwargs,
                       waveform_generator=waveform_generator,
                       waveform_generator_kwargs=waveform_generator_kwargs,
                       ResponseWrapper=ResponseWrapper, ResponseWrapper_kwargs=ResponseWrapper_kwargs,
                       noise_model=noise_model, noise_kwargs=noise_kwargs, channels=channels,
                       stats_for_nerds = True, use_gpu = USE_GPU,
                       deriv_type='stable')

param_names = ['m1','m2','a','p0','e0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_r0']
der_order = 4
Ndelta = 8
stability_plot = True

delta_range = dict(
    m1 = np.geomspace(1e-4*m1, 1e-9*m1, Ndelta),
    m2 = np.geomspace(1e-2*m2, 1e-7*m2, Ndelta),
    a = np.geomspace(1e-5, 1e-9, Ndelta),
    p0 = np.geomspace(1e-5, 1e-9, Ndelta),
    e0 = np.geomspace(1e-5, 1e-9, Ndelta),
    qS = np.array([1e-6]),
    phiS = np.array([1e-6]),
    qK = np.array([1e-6]),
    phiK = np.array([1e-6]),
    Phi_phi0 = np.array([1e-6]),
    Phi_r0 = np.array([1e-6]),
)

print("Computing FM")
start = time.time()
derivs, fisher_matrix = sef(*pars_list, param_names = param_names, 
             T = T, dt = dt, 
             der_order = der_order, 
             Ndelta = Ndelta, 
             stability_plot = stability_plot,
             delta_range = delta_range,
             return_derivatives = True,
             filename="MCMC_FM_Data/fisher_matrices/plunging_EMRI",
            live_dangerously = False)
end = time.time() - start
print("Time taken to compute Fisher matrix and stable deltas is", end, "seconds")


param_cov = np.linalg.inv(fisher_matrix)

for k, item in enumerate(param_names):
    print("Precision measurement in param {} is {}".format(item, param_cov[k,k]**(1/2)))

# ================= Compute fluctuation due to noise realisation ========================
def zero_pad(data):
    """
    Inputs: data stream of length N
    Returns: zero_padded data stream of new length 2^{J} for J \in \mathbb{N}
    """
    N = len(data)
    pow_2 = xp.ceil(np.log2(N))
    return xp.pad(data,(0,int((2**pow_2)-N)),'constant')

def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD):
    """
    Compute stationary noise-weighted inner product
    Inputs: sig1_f and sig2_f are signals in frequency domain 
            N_t length of padded signal in time domain
            delta_t sampling interval
            PSD Power spectral density

    Returns: Noise weighted inner product 
    """
    prefac = 4*delta_t / N_t
    sig2_f_conj = xp.conjugate(sig2_f)
    return prefac * xp.real(xp.sum((sig1_f * sig2_f_conj)/PSD))


N_params = len(param_names)
derivs_A_channel = [zero_pad(derivs[p][0]) for p in range(N_params)]
derivs_E_channel = [zero_pad(derivs[p][1]) for p in range(N_params)]

derivs_A_fft = [xp.fft.rfft(derivs_A_channel[p]) for p in range(N_params)]
derivs_E_fft = [xp.fft.rfft(derivs_E_channel[p]) for p in range(N_params)]

N_t = len(derivs_A_channel[0])

freq_bins = np.fft.rfftfreq(N_t, dt)
freq_bins[0] = freq_bins[1]

PSD_AE = PSD_AE_interp(freq_bins)

variance_noise_AE = [N_t * PSD_AE[k] / (4*dt) for k in range(N_channels)]

variance_noise_AE[0] = 2*variance_noise_AE[0]
variance_noise_AE[-1] = 2*variance_noise_AE[-1]

np.random.seed(1)
noise_f_AE_real = [xp.random.normal(0,np.sqrt(variance_noise_AE[k])) for k in range(N_channels)]
noise_f_AE_imag = [xp.random.normal(0,np.sqrt(variance_noise_AE[k])) for k in range(N_channels)]

# Compute noise in frequency domain
noise_f_AE = xp.asarray([noise_f_AE_real[k] + 1j * noise_f_AE_imag[k] for k in range(N_channels)])

for i in range(N_channels):
    noise_f_AE[i][0] = noise_f_AE[i][0].real
    noise_f_AE[i][-1] = noise_f_AE[i][-1].real

noise_f_A = noise_f_AE[0]
noise_f_E = noise_f_AE[1]

bias_vec_1 = np.array([inner_prod(derivs_A_fft[k],noise_f_A,N_t,dt,PSD_AE[0]) for p in range(N_params)])
bias_vec_1 = np.array([inner_prod(derivs_A_fft[k],noise_f_A,N_t,dt,PSD_AE[0]) for p in range(N_params)])


breakpoint()


