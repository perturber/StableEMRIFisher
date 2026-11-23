"""Fisher matrices with different inspiral err, der_order and step grids vs an existing MCMC run.

Usage: python fisher_convergence.py <injection_config>.py

The injection (including p0 and dist) is read from the finished run in the config's save_dir, so
every Fisher matrix describes exactly the source the MCMC sampled. Outputs go to
<save_dir>/fisher_convergence/: one folder per case (Fisher.h5, stable_deltas.txt, stability
plots), one corner plot per case against the MCMC samples, and summary.txt. A case whose
Fisher.h5 already exists is not recomputed.
"""

import copy
import glob
import importlib.util
import os
import sys

import h5py
import matplotlib
import numpy as np
from eryn.backends import HDFBackend
from few.utils.constants import YRSID_SI

from metrics import gaussian_kl, gaussian_w2
from plot_results import plot_results
from stableemrifisher.fisher import StableEMRIFisher

matplotlib.use("Agg")
np.in1d = np.isin  # eryn 1.2.6 calls np.in1d, which numpy >= 2.4 removed

# (inspiral err, der_order, deep grid): the deep grid runs the trial steps of the masses and orbital
# parameters down to 1e-12 of their value (default 1e-9), with twice as many steps (16 instead of 8)
CASES = [(1e-11, 4, False), (1e-13, 4, False), (1e-11, 8, False), (1e-13, 8, False), (1e-13, 8, True)]
ORBITAL = {"m1", "m2", "a", "p0", "e0", "xI0"}

spec = importlib.util.spec_from_file_location("cfg", sys.argv[1])
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
opt = lambda name, default=None: getattr(cfg, name, default)
save_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), cfg.save_dir)
out_dir = os.path.join(save_dir, "fisher_convergence")
os.makedirs(out_dir, exist_ok=True)

# the finished run: its injection, window and MCMC samples
run = [p for p in glob.glob(os.path.join(save_dir, "*.h5")) if os.path.basename(p) != "Fisher.h5"][0]
with h5py.File(run, "r") as f:
    names = [n.decode() for n in f["fisher/param_names"][:]]
    attrs = dict(f["fisher"].attrs)
order = ["m1", "m2", "a", "p0", "e0", "xI0", "dist", "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0"]
params = {k: float(attrs[k]) for k in order}
add = {k: float(attrs[k]) for k in (opt("add_param_args") or {})}
inj = np.array([{**params, **add}[p] for p in names])
chain = HDFBackend(run, read_only=True).get_chain()["model_0"][:, 0, :, 0, :].reshape(-1, len(names))
sigma_mcmc = chain.std(axis=0)

T, dt, use_gpu = cfg.T, cfg.dt, opt("use_gpu", False)
response = {}
if cfg.use_response:  # as in exec.py
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
        noise_model=get_sensitivity,
        noise_kwargs=[
            dict(sens_fn=channel, model="scirdv1", stochastic_params=[T * YRSID_SI])
            for channel in (A1TDISens, E1TDISens)
        ],
    )

summary = []
for err, der_order, deep in CASES:
    label = f"err{err:.0e}_order{der_order}" + ("_deepgrid" if deep else "")
    case_dir = os.path.join(out_dir, label)
    if os.path.exists(os.path.join(case_dir, "Fisher.h5")):
        with h5py.File(os.path.join(case_dir, "Fisher.h5"), "r") as f:
            fisher = f["Fisher"][:]
        print(f"{label}: reusing {case_dir}/Fisher.h5", flush=True)
    else:
        waveform_class_kwargs = copy.deepcopy(opt("waveform_class_kwargs") or {})
        waveform_class_kwargs.setdefault("inspiral_kwargs", {})["err"] = err
        sef = StableEMRIFisher(
            waveform_class=cfg.waveform_class, waveform_class_kwargs=waveform_class_kwargs,
            waveform_kwargs=opt("waveform_kwargs"), deriv_type=cfg.deriv_type, der_order=der_order,
            T=T, dt=dt, use_gpu=use_gpu, **response,
        )
        ndelta, delta_range = 8, None
        if deep:
            values = {**params, **add}
            ndelta = 16
            delta_range = {p: np.geomspace(1e-4 * values[p], 1e-12 * values[p], ndelta)
                           for p in names if p in ORBITAL and values[p] != 0.0}
        fisher = sef(dict(params), add_param_args=add or None, param_names=names, Ndelta=ndelta,
                     delta_range=delta_range, stability_plot=True, stats_for_nerds=True, filename=case_dir)
        if not np.isclose(sef.T, attrs["T"]):
            print(f"WARNING: observation time {sef.T} differs from the MCMC run's {attrs['T']}")
    cov = np.linalg.inv(fisher)
    sigma_f = np.sqrt(np.diag(cov))

    # joint KL and W2 of the MCMC samples against this Fisher Gaussian (whitened, as in metrics.py)
    corr = cov / np.outer(sigma_f, sigma_f)
    z = np.linalg.solve(np.linalg.cholesky(corr), ((chain - inj) / sigma_f).T).T
    kl, w2 = gaussian_kl(z.mean(0), np.cov(z.T)), gaussian_w2(z.mean(0), np.cov(z.T))

    title = f"inspiral err = {err:.0e}, der_order = {der_order}" + (", deep grid" if deep else "")
    plot_results(run, cov=cov, out=os.path.join(out_dir, f"corner_{label}.png"), title=title)
    summary.append((title, kl, w2, sigma_mcmc / sigma_f))
    print(f"{title}: joint KL = {kl:.4f}, joint W2 = {w2:.3f}", flush=True)

lines = [f"MCMC run: {os.path.basename(run)}", "",
         "sigma_MCMC / sigma_Fisher (rows: inspiral err, der_order; joint KL and W2 as in metrics.py)", ""]
width = max(len(t) for t, *_ in summary)
lines.append(" " * width + "  " + "".join(f"{n:>9s}" for n in names) + f"{'KL':>9s}{'W2':>8s}")
for title, kl, w2, ratio in summary:
    lines.append(f"{title:{width}s}  " + "".join(f"{r:9.3f}" for r in ratio) + f"{kl:9.4f}{w2:8.3f}")
text = "\n".join(lines) + "\n"
print("\n" + text)
with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
    fh.write(text)
