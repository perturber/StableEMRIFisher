"""Corner plot of an exec.py run: MCMC posterior vs Fisher Gaussian times the run's uniform prior.

Usage: python fisher_prior.py <run>.h5

The Fisher approximation to the posterior is the Fisher Gaussian (centred on the injection) times
the uniform prior box of the MCMC run, a truncated Gaussian, sampled by keeping the draws of the
Fisher Gaussian that lie inside the box. Writes <run>_fisher_prior.png next to <run>.h5.
"""

import sys

import corner
import h5py
import matplotlib.pyplot as plt
import numpy as np
from eryn.backends import HDFBackend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from stableemrifisher.utils.plot import cov_ellipse

np.in1d = np.isin  # eryn 1.2.6 calls np.in1d, which numpy >= 2.4 removed

MCMC_COLOR, FISHER_COLOR, LIKELIHOOD_COLOR = "C0", "C3", "0.4"
LEVELS = (1 - np.exp(-0.5), 1 - np.exp(-2))  # 1 and 2 sigma of a 2D Gaussian


def fisher_prior_samples(inj, cov, lower, upper, n, seed=0):
    """n draws of the Fisher Gaussian N(inj, cov) restricted to the box [lower, upper]."""
    rng = np.random.default_rng(seed)
    kept, n_kept = [], 0
    while n_kept < n:
        draws = rng.multivariate_normal(inj, cov, size=2 * n)
        draws = draws[np.all((draws > lower) & (draws < upper), axis=1)]
        kept.append(draws)
        n_kept += len(draws)
    return np.concatenate(kept)[:n]


def plot_fisher_prior(path, n_sigma=4.0):
    with h5py.File(path, "r") as f:
        g = f["fisher"]
        cov, inj = g["covariance"][:], g["injection"][:]
        lower, upper = g["prior_lower"][:], g["prior_upper"][:]
        names = [n.decode() for n in g["param_names"][:]]
    ndim = len(names)
    chain = HDFBackend(path, read_only=True).get_chain()["model_0"][:, 0, :, 0, :].reshape(-1, ndim)
    truncated = fisher_prior_samples(inj, cov, lower, upper, n=4 * len(chain))

    mcmc, fp = chain - inj, truncated - inj  # offsets from the injection
    sigma_f = np.sqrt(np.diag(cov))
    sigma_mcmc, sigma_fp = mcmc.std(axis=0), fp.std(axis=0)
    span = np.maximum(np.abs(np.percentile(mcmc, [0.05, 99.95], axis=0)).max(axis=0),
                      np.abs(np.percentile(fp, [0.05, 99.95], axis=0)).max(axis=0))
    ranges = list(zip(-1.1 * span, 1.1 * span))

    common = dict(bins=40, smooth=1.0, levels=LEVELS, plot_datapoints=False, plot_density=False, range=ranges)
    fig = corner.corner(
        mcmc, labels=[rf"$\Delta${n}" for n in names], color=MCMC_COLOR, fill_contours=True,
        contour_kwargs=dict(linewidths=0.6), max_n_ticks=3, label_kwargs=dict(fontsize=14),
        hist_kwargs=dict(histtype="stepfilled", density=True, alpha=0.45, edgecolor="none"), **common,
    )
    corner.corner(
        fp, fig=fig, color=FISHER_COLOR, fill_contours=False, contour_kwargs=dict(linewidths=1.5),
        hist_kwargs=dict(histtype="step", density=True, linewidth=1.8), **common,
    )

    # the Fisher Gaussian without the prior, for comparison
    axes = np.array(fig.axes).reshape(ndim, ndim)
    for i in range(ndim):
        ax = axes[i, i]
        x = np.linspace(*ranges[i], 400)
        ax.plot(x, np.exp(-0.5 * (x / sigma_f[i]) ** 2) / (np.sqrt(2 * np.pi) * sigma_f[i]),
                color=LIKELIHOOD_COLOR, lw=1.2, ls="--")
        ax.set_title(rf"$\sigma_{{\rm MCMC}}/\sigma_{{\rm F}} = {sigma_mcmc[i] / sigma_f[i]:.2f}$" "\n"
                     rf"$\sigma_{{\rm MCMC}}/\sigma_{{\rm F \times prior}} = {sigma_mcmc[i] / sigma_fp[i]:.2f}$",
                     fontsize=10)
        for j in range(i + 1, ndim):
            for n_std in (1, 2):
                cov_ellipse((0.0, 0.0), cov[np.ix_([i, j], [i, j])], axes[j, i], n_std=n_std,
                            facecolor="none", edgecolor=LIKELIHOOD_COLOR, lw=1.0, ls="--")

    fig.legend(
        handles=[
            Patch(color=MCMC_COLOR, alpha=0.6, label=r"MCMC (1$\sigma$, 2$\sigma$)"),
            Line2D([], [], color=FISHER_COLOR, lw=1.8, label=r"Fisher $\times$ prior"),
            Line2D([], [], color=LIKELIHOOD_COLOR, lw=1.2, ls="--", label="Fisher (no prior)"),
        ],
        loc="upper right", fontsize=4 + 1.5 * ndim, frameon=False,
    )
    out = path[: -len(".h5")] + "_fisher_prior.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    width = max(len(n) for n in names)
    print(f"{'':{width}s}  {'sigma_MCMC/sigma_F':>19s}  {'sigma_MCMC/sigma_(F x prior)':>29s}")
    for n, sm, sf, sp in zip(names, sigma_mcmc, sigma_f, sigma_fp):
        print(f"{n:{width}s}  {sm / sf:19.3f}  {sm / sp:29.3f}")
    return out


if __name__ == "__main__":
    print(plot_fisher_prior(sys.argv[1]))
