"""Corner plot of an exec.py run: MCMC posterior (filled) vs Fisher Gaussian (lines). Usage: python plot_results.py <run>.h5"""

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

MCMC_COLOR, FISHER_COLOR = "C0", "C3"


def plot_results(path, n_sigma=4.0, cov=None, out=None, title=None):
    """Write <run>.png next to <run>.h5: offsets from the injection, 1 and 2 sigma contours.

    `cov` replaces the run's Fisher covariance (same parameters), `out` the output path, and
    `title` is written above the plot.
    """
    with h5py.File(path, "r") as f:
        inj = f["fisher/injection"][:]
        cov = f["fisher/covariance"][:] if cov is None else np.asarray(cov)
        names = [n.decode() for n in f["fisher/param_names"][:]]
    chain = HDFBackend(path, read_only=True).get_chain()["model_0"]
    offsets = chain[:, 0, :, 0, :].reshape(-1, len(names)) - inj  # cold chain
    ndim = len(names)

    sigma_fisher, sigma_mcmc = np.sqrt(np.diag(cov)), offsets.std(axis=0)
    half_width = n_sigma * np.maximum(sigma_fisher, sigma_mcmc)
    fig = corner.corner(
        offsets,
        labels=[rf"$\Delta${n}" for n in names],
        range=list(zip(-half_width, half_width)),
        bins=40,
        smooth=1.0,
        color=MCMC_COLOR,
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),  # 1 and 2 sigma of a 2D Gaussian
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
        contour_kwargs=dict(linewidths=0.6),
        hist_kwargs=dict(histtype="stepfilled", density=True, alpha=0.45, edgecolor="none"),
        max_n_ticks=3,
        label_kwargs=dict(fontsize=14),
    )
    axes = np.array(fig.axes).reshape(ndim, ndim)

    for i in range(ndim):
        ax = axes[i, i]
        x = np.linspace(-half_width[i], half_width[i], 400)
        gauss = np.exp(-0.5 * (x / sigma_fisher[i]) ** 2) / (np.sqrt(2 * np.pi) * sigma_fisher[i])
        ax.plot(x, gauss, color=FISHER_COLOR, lw=1.8)
        ax.set_ylim(0, 1.15 * max(ax.get_ylim()[1], gauss.max()))
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(rf"$\sigma_{{\rm MCMC}}/\sigma_{{\rm F}} = {sigma_mcmc[i] / sigma_fisher[i]:.2f}$", fontsize=11)
        for j in range(i + 1, ndim):
            ax = axes[j, i]
            for n_std in (1, 2):
                cov_ellipse((0.0, 0.0), cov[np.ix_([i, j], [i, j])], ax, n_std=n_std,
                            facecolor="none", edgecolor=FISHER_COLOR, lw=1.5)
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.axhline(0, color="k", lw=0.8, ls="--")

    fig.legend(
        handles=[
            Patch(color=MCMC_COLOR, alpha=0.6, label=r"MCMC (1$\sigma$, 2$\sigma$)"),
            Line2D([], [], color=FISHER_COLOR, lw=1.8, label=r"Fisher (1$\sigma$, 2$\sigma$)"),
            Line2D([], [], color="k", lw=0.8, ls="--", label="Injection"),
        ],
        loc="upper right",
        fontsize=4 + 1.5 * ndim,
        frameon=False,
    )
    if title is not None:
        fig.suptitle(title, x=0.98, y=0.86, ha="right", fontsize=4 + 1.5 * ndim)
    out = out or path[: -len(".h5")] + ".png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot_results(sys.argv[1]))
