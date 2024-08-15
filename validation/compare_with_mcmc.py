from stableemrifisher.fisher import StableEMRIFisher
from few.waveform import GenerateEMRIWaveform
from few.utils.utility import get_p_at_t
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import os
import corner
from pathlib import Path
import cupy as xp
import numpy as np
use_gpu=True

#set initial parameters
M = 5e5
mu = 14
a = 0.8
p0 = 11.58
e0 = 0.3
iota0 = 0.8 #equatorial
Y0 = np.cos(iota0)
Phi_phi0 = 2
Phi_theta0 = 3
Phi_r0 = 4

qS = 1.5
phiS = 0.7
qK = 1.2
phiK = 0.6
dist = 2.
mich = True #mich = True implies output in hI, hII long wavelength approximation
dt = 10.0
T = 1.0

inspiral_kwargs = {
        "DENSE_STEPPING": 0,
        "max_init_len": int(1e4),
        "err": 1e-10,  # To be set within the class
        "use_rk4": False,
        }

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": True,  # GPU is available for this type of summation
    "pad_output": True,
}

amplitude_kwargs = {
    }

outdir = 'data_files/FM_results'

Path(outdir).mkdir(exist_ok=True, parents=True)

waveform_model = GenerateEMRIWaveform('Pn5AAKWaveform', inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)

waveform = xp.asarray(waveform_model(M, mu, a, p0, e0, Y0, 
                          dist, qS, phiS, qK, phiK, 
                          Phi_phi0, Phi_theta0, Phi_r0, 
                          mich=mich, dt=dt, T=T)).get()

plt.plot(waveform[:1000].real)
plt.ylabel("h+")
plt.savefig(os.path.join(outdir, "waveform.png"))
plt.close()

#varied parameters
param_names = ['M','mu','a','p0','e0','Y0','dist', 'qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0']

#initialization
fish = StableEMRIFisher(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
              Phi_phi0, Phi_theta0, Phi_r0, dt=dt, T=T, EMRI_waveform_gen=waveform_model,
              param_names=param_names, stats_for_nerds=True, use_gpu=True,
              filename=outdir, CovEllipse=True, live_dangerously=False, der_order=6, )


#execution
fish()


# Read in MCMC results

true_vals = np.array([M, mu, a, p0, e0, np.cos(iota0), dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]) # From MCMC, injection parameters

samples_14D = np.load("data_files/MCMC_results/samples_large_q_SNR_30.npy"); N_params_14D = len(samples_14D) # Read in already processed samples

# Read in FM results

Gamma_14D = np.load("data_files/FM_results/Fisher.npy") 
Gamma_inv_14D = np.linalg.inv(Gamma_14D) # Construct FM inv
FM_samples_14D = np.random.multivariate_normal(true_vals, Gamma_inv_14D, len(samples_14D[0])) # Draw samples, much easier this way.


# 14D case -- corner plot

params =[r"$M$", r"$\mu$", r"$a$", r"$p_{0}$", r"$e_{0}$", "$Y_{0}$", "$D$",r"$\theta_{S}$", r"$\phi_{S}$", r"$\theta_{K}$", r"$\phi_{K}$", 
         r"$\Phi_{\phi}$",r"$\Phi_{\theta}$", r"$\Phi_{r}$"]  # Set up labels for corner plot


corner_kwargs = dict(plot_datapoints=False,smooth1d=True,
                       labels=params, levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)), 
                       label_kwargs=dict(fontsize=40), max_n_ticks=4,
                       show_titles=False, smooth = True, labelpad = 0.4)  # Fancy plotting arguments

samples = np.column_stack(samples_14D)

figure = corner.corner(samples,bins = 30, color = 'blue', **corner_kwargs)  # MCMC corner plot

corner.corner(FM_samples_14D, fig = figure, bins = 30, color = 'red', **corner_kwargs)  # FM corner plot

axes = np.array(figure.axes).reshape((N_params_14D, N_params_14D))  # Construct axes

# Now set up vertical/horizontal lines indicating location of true parameters
for i in range(N_params_14D):
    ax = axes[i, i]
    ax.axvline(true_vals[i], color="k")
    
for yi in range(N_params_14D):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axhline(true_vals[yi], color="k")
        ax.axvline(true_vals[xi],color= "k")
        ax.plot(true_vals[xi], true_vals[yi], "sk")

# Tick sizes
for ax in figure.get_axes():
    ax.tick_params(axis='both', labelsize=18)

# Set up legend
blue_line = mlines.Line2D([], [], color='blue', label=r'MCMC')  # MCMC
red_line = mlines.Line2D([], [], color='red', label=r'Fisher')  # FM
black_line = mlines.Line2D([], [], color='black', label='True Value')  # Location of true parameter

# Place legend
plt.legend(handles=[blue_line,red_line,black_line], fontsize = 65, frameon = True, bbox_to_anchor=(-0.15, N_params_14D + 0.8), 
           loc="upper right", title = r"Comparisons: FM vs MCMC", 
           title_fontproperties = FontProperties(size = 70, weight = 'bold'))
plt.subplots_adjust(left=-0.1, bottom=-0.1, right=None, top=None, wspace=0.15, hspace=0.15) # Tidy up

plt.savefig(os.path.join(outdir, "comparison_plot.pdf"), bbox_inches="tight")  # Save plot
