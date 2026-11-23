"""MCMC (Eryn) vs Fisher covariance for one injection.

Usage: python exec.py <injection_config>.py [--fisher-only]

With --fisher-only, the Fisher matrix of a finished run is recomputed and its results updated,
keeping the run's chains.
"""

import importlib.util
import os
import shutil
import subprocess
import sys

if importlib.util.find_spec("eryn") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "eryn==1.2.6"])

import h5py
import numpy as np
from eryn.backends import HDFBackend
from eryn.ensemble import EnsembleSampler
from eryn.moves import GaussianMove, StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from few.utils.constants import YRSID_SI
from few.utils.utility import get_p_at_t
from metrics import report
from plot_results import plot_results

from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils.analysis import inner_product
from stableemrifisher.utils.utils import check_if_plunging

np.in1d = np.isin  # eryn 1.2.6 calls np.in1d, which numpy >= 2.4 removed

spec = importlib.util.spec_from_file_location("cfg", sys.argv[1])
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
opt = lambda name, default=None: getattr(cfg, name, default)
fisher_only = "--fisher-only" in sys.argv[2:]
save_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), cfg.save_dir)
os.makedirs(save_dir, exist_ok=True)
shutil.copy(sys.argv[1], save_dir)
T, dt = cfg.T, cfg.dt

use_gpu = opt("use_gpu", False)

response = {}
if cfg.use_response:
    from fastlisaresponse import ResponseWrapper
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.sensitivity import A1TDISens, E1TDISens, get_sensitivity

    backend = "gpu" if use_gpu else "cpu"
    response = dict(
        ResponseWrapper=ResponseWrapper,
        ResponseWrapper_kwargs=dict(
            Tobs=T, dt=dt, index_lambda=8, index_beta=7, t0=20000.0,
            remove_garbage="zero", flip_hx=True, is_ecliptic_latitude=False,
            force_backend=backend, orbits=EqualArmlengthOrbits(force_backend=backend),
            order=25, tdi="1st generation", tdi_chan="AE",
        ),
        # SciRDv1 noise of the A and E channels, with the galactic foreground for T
        noise_model=get_sensitivity,
        noise_kwargs=[
            dict(sens_fn=channel, model="scirdv1", stochastic_params=[T * YRSID_SI])
            for channel in (A1TDISens, E1TDISens)
        ],
    )

sef = StableEMRIFisher(
    waveform_class=cfg.waveform_class,
    waveform_class_kwargs=opt("waveform_class_kwargs"),
    waveform_kwargs=opt("waveform_kwargs"),
    deriv_type=cfg.deriv_type, der_order=cfg.der_order, T=T, dt=dt, use_gpu=use_gpu, **response,
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

fisher = sef(params, add_param_args=add or None, param_names=opt("param_names"), filename=save_dir)
names, params, wkw = sef.param_names, dict(sef.wave_params), sef.current_waveform_kwargs
cov = np.linalg.inv(fisher)
inj = np.array([params[p] for p in names])
sigma = np.sqrt(np.diag(cov))
ndim = len(names)

tag = "_".join(
    [cfg.waveform_class.__name__, "AE-TDI1" if cfg.use_response else "noresponse",
     f"{sef.deriv_type.__name__}-order{cfg.der_order}", f"T={T:g}yr", f"dt={dt:g}s",
     f"DeltaT={cfg.DeltaT:g}yr", f"SNR={cfg.desired_snr:g}"]
    + [f"{k}={v:.8g}" for k, v in params.items() if k not in ("p0", "dist")]
)
out = os.path.join(save_dir, tag)

if fisher_only:
    # replace the Fisher results of the finished run; its chains and priors are kept
    with h5py.File(out + ".h5", "a") as f:
        g = f["fisher"]
        if not np.allclose(g["injection"][:], inj, rtol=1e-10, atol=0.0):
            raise ValueError(f"the injection differs from the one of the run in {out}.h5")
        g["fisher"][...], g["covariance"][...] = fisher, cov
        g.attrs["SNR"] = np.sqrt(sef.SNR2)
    samples = HDFBackend(out + ".h5", read_only=True).get_chain()["model_0"][:, 0, :, 0, :].reshape(-1, ndim)
else:

    def log_like(x):
        r = sef.waveform - sef.derivative.generate(*{**params, **dict(zip(names, x))}.values(), **wkw)
        return -0.5 * float(inner_product(r, r, sef.PSD_funcs, sef.dt, use_gpu=sef.use_gpu))

    # uniform priors: injection +/- prior_factor * sigma, clipped to the absolute prior_bounds
    factor = cfg.prior_factor
    bounds = np.array([cfg.prior_bounds.get(p, (-np.inf, np.inf)) for p in names], dtype=float)
    lo = np.maximum(inj - factor * sigma, bounds[:, 0])
    hi = np.minimum(inj + factor * sigma, bounds[:, 1])
    priors = {"model_0": ProbDistContainer({i: uniform_dist(lo[i], hi[i]) for i in range(ndim)})}

    # walkers start from the Fisher Gaussian; coordinates outside the priors are redrawn uniformly
    rng = np.random.default_rng(cfg.seed)
    nwalkers, ntemps = cfg.nwalkers, cfg.ntemps
    start = rng.multivariate_normal(inj, cov, size=ntemps * nwalkers)
    outside = (start <= lo) | (start >= hi)
    start[outside] = rng.uniform(np.broadcast_to(lo, start.shape)[outside], np.broadcast_to(hi, start.shape)[outside])

    if os.path.exists(out + ".h5"):
        os.remove(out + ".h5")
    sampler = EnsembleSampler(
        nwalkers, ndim, log_like, priors, tempering_kwargs=dict(ntemps=ntemps),
        moves=[(StretchMove(), cfg.stretch_weight), (GaussianMove({"model_0": cov}), 1.0 - cfg.stretch_weight)],
        backend=HDFBackend(out + ".h5"),
    )
    sampler.run_mcmc(start.reshape(ntemps, nwalkers, 1, ndim), cfg.nsteps, burn=cfg.burn, progress=True)

    with h5py.File(out + ".h5", "a") as f:
        g = f.create_group("fisher")
        g["fisher"], g["covariance"], g["injection"] = fisher, cov, inj
        g["param_names"] = np.array(names, dtype="S")
        g["prior_lower"], g["prior_upper"] = lo, hi
        g.attrs.update(params, SNR=np.sqrt(sef.SNR2), snr_1gpc=snr_1gpc, DeltaT=cfg.DeltaT, prior_factor=factor,
                       **{"T": sef.T, "dt": sef.dt})
    samples = sampler.get_chain()["model_0"][:, 0, :, 0, :].reshape(-1, ndim)

for p, s_mc, s_f in zip(names, samples.std(axis=0), sigma):
    print(f"{p:10s} sigma_MCMC = {s_mc:.4g}  sigma_Fisher = {s_f:.4g}  ratio = {s_mc / s_f:.3f}")

plot_results(out + ".h5")
report(out + ".h5")
