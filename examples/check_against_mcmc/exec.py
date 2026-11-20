"""MCMC (Eryn) vs Fisher covariance for one injection. Usage: python exec.py <injection_config>.py"""

import importlib.util
import os
import sys

import corner
import h5py
import numpy as np
from eryn.backends import HDFBackend
from eryn.ensemble import EnsembleSampler
from eryn.moves import GaussianMove, StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from few.utils.constants import YRSID_SI
from few.utils.utility import get_p_at_t
from matplotlib.lines import Line2D

from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils.analysis import inner_product
from stableemrifisher.utils.utils import check_if_plunging

np.in1d = np.isin  # eryn 1.2.6 calls np.in1d, which numpy >= 2.4 removed

spec = importlib.util.spec_from_file_location("cfg", sys.argv[1])
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
opt = lambda name, default=None: getattr(cfg, name, default)
here = os.path.dirname(os.path.abspath(__file__))
T, dt = cfg.T, cfg.dt

use_gpu = opt("use_gpu", False)
if use_gpu:
    import cupy as xp
else:
    xp = np

response = {}
if cfg.use_response:
    from fastlisaresponse import ResponseWrapper
    from lisatools.detector import EqualArmlengthOrbits
    from stableemrifisher.utils.noise import load_psd

    backend = "gpu" if use_gpu else "cpu"
    response = dict(
        ResponseWrapper=ResponseWrapper,
        ResponseWrapper_kwargs=dict(
            Tobs=T, dt=dt, index_lambda=8, index_beta=7, t0=20000.0,
            remove_garbage="zero", flip_hx=True, is_ecliptic_latitude=False,
            force_backend=backend, orbits=EqualArmlengthOrbits(force_backend=backend),
            order=25, tdi="2nd generation", tdi_chan="AE",
        ),
        noise_model=load_psd(
            model="scirdv1", channels="AE", tdi2=True, include_foreground=True,
            stochastic_params=[T * YRSID_SI],
            filename=os.path.join(here, f"psd_scirdv1_AE_tdi2_foreground_{T}yr.npy"), xp=xp,
        ),
        noise_kwargs={},
    )

sef = StableEMRIFisher(
    waveform_class=cfg.waveform_class,
    waveform_class_kwargs=opt("waveform_class_kwargs"),
    waveform_kwargs=opt("waveform_kwargs"),
    T=T, dt=dt, der_order=4, use_gpu=use_gpu, **response,
)

# p0: plunge at T + DeltaT; dist: desired_snr, from the SNR at 1 Gpc
e, add = cfg.emri_params, dict(opt("add_param_args") or {})
inspiral_kwargs = sef.derivative.base_generator.waveform_generator.inspiral_kwargs
p0 = get_p_at_t(sef.traj_module, T + cfg.DeltaT, [e["m1"], e["m2"], e["a"], e["e0"], e["xI0"], *add.values()],
                traj_kwargs=dict(inspiral_kwargs))
order = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
params = {k: {**e, "p0": p0, "dist": 1.0}[k] for k in order}
T_obs = check_if_plunging(sef.traj_module, T, *list(params.values())[:6], *add.values(), Phi_phi0=params["Phi_phi0"],
                          Phi_theta0=params["Phi_theta0"], Phi_r0=params["Phi_r0"], chop_inspiral_time=6.0, dt=dt)
snr_1gpc = sef.compute_snr(*params.values(), *add.values(), dt=dt, T=T_obs, use_gpu=sef.use_gpu, **(opt("waveform_kwargs") or {}))
params["dist"] = 1.0 * snr_1gpc / cfg.desired_snr

fisher = sef(params, add_param_args=add or None, param_names=opt("param_names"))
names, params, wkw = sef.param_names, dict(sef.wave_params), sef.current_waveform_kwargs
cov = np.linalg.inv(fisher)
inj = np.array([params[p] for p in names])
sigma = np.sqrt(np.diag(cov))
ndim = len(names)


def log_like(x):
    r = sef.waveform - sef.derivative.generate(*{**params, **dict(zip(names, x))}.values(), **wkw)
    return -0.5 * float(inner_product(r, r, sef.PSD_funcs, sef.dt, use_gpu=sef.use_gpu))


factor = opt("prior_factor", 10.0)
lo, hi = inj - factor * sigma, inj + factor * sigma
priors = {"model_0": ProbDistContainer({i: uniform_dist(lo[i], hi[i]) for i in range(ndim)})}

rng = np.random.default_rng(opt("seed"))
nwalkers = opt("nwalkers", 4 * ndim)
start = rng.multivariate_normal(inj, cov, size=100 * nwalkers)
start = start[np.all((start > lo) & (start < hi), axis=1)][:nwalkers]

tag = "_".join(
    [cfg.waveform_class.__name__, "AE-TDI2" if cfg.use_response else "noresponse", f"T={T:g}yr", f"dt={dt:g}s",
     f"DeltaT={cfg.DeltaT:g}yr", f"SNR={cfg.desired_snr:g}"]
    + [f"{k}={v:.8g}" for k, v in params.items() if k not in ("p0", "dist")]
)
out = os.path.join(here, tag)
if os.path.exists(out + ".h5"):
    os.remove(out + ".h5")

sampler = EnsembleSampler(
    nwalkers, ndim, log_like, priors,
    moves=[(StretchMove(), 0.5), (GaussianMove({"model_0": cov}), 0.5)],
    backend=HDFBackend(out + ".h5"),
)
sampler.run_mcmc(start[None, :, None, :], opt("nsteps", 5000), burn=opt("burn", 1000), progress=True)

with h5py.File(out + ".h5", "a") as f:
    g = f.create_group("fisher")
    g["fisher"], g["covariance"], g["injection"] = fisher, cov, inj
    g["param_names"] = np.array(names, dtype="S")
    g.attrs.update(params, SNR=np.sqrt(sef.SNR2), snr_1gpc=snr_1gpc, DeltaT=cfg.DeltaT, prior_factor=factor,
                   **{"T": sef.T, "dt": sef.dt})

samples = sampler.get_chain()["model_0"][:, 0, :, 0, :].reshape(-1, ndim)
fisher_samples = rng.multivariate_normal(samples.mean(axis=0), cov, size=len(samples))
for p, s_mc, s_f in zip(names, samples.std(axis=0), sigma):
    print(f"{p:10s} sigma_MCMC = {s_mc:.4g}  sigma_Fisher = {s_f:.4g}  ratio = {s_mc / s_f:.3f}")

kw = dict(
    labels=names, bins=30, smooth=1.0, plot_datapoints=False, plot_density=False,
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
    range=list(zip(np.minimum(samples.min(0), fisher_samples.min(0)), np.maximum(samples.max(0), fisher_samples.max(0)))),
)
fig = corner.corner(samples, color="C0", truths=inj, truth_color="k", **kw)
corner.corner(fisher_samples, fig=fig, color="C3", **kw)
fig.legend(
    handles=[Line2D([], [], color=c, label=l) for c, l in [("C0", "MCMC"), ("C3", "Fisher"), ("k", "Injection")]],
    loc="upper right", fontsize=4 * ndim,
)
fig.savefig(out + ".png", dpi=300)
