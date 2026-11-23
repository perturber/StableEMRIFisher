"""Fisher vs MCMC agreement (KL divergence, Wasserstein distance). Usage: python metrics.py [<run>.h5 ...]

With no arguments, every run in the subfolders of this folder is used. All metrics are computed in
coordinates where the Fisher Gaussian (centred on the injection) is a standard normal, so they are
dimensionless and do not depend on the units of the parameters.
"""

import glob
import os
import sys

import h5py
import numpy as np
from eryn.backends import HDFBackend
from eryn.utils.utility import get_integrated_act
from scipy.linalg import sqrtm
from scipy.stats import norm

np.in1d = np.isin  # eryn 1.2.6 calls np.in1d, which numpy >= 2.4 removed


def gaussian_kl(mean, cov):
    """KL(N(mean, cov) || N(0, I))."""
    d = len(mean)
    return 0.5 * (np.trace(cov) + mean @ mean - d - np.linalg.slogdet(cov)[1])


def gaussian_w2(mean, cov):
    """2-Wasserstein distance between N(mean, cov) and N(0, I)."""
    return np.sqrt(max(mean @ mean + np.trace(cov) + len(mean) - 2 * np.trace(sqrtm(cov).real), 0.0))


def marginal_w1(x):
    """1-Wasserstein distance between the samples x and N(0, 1)."""
    x = np.sort(x)
    return np.mean(np.abs(x - norm.ppf((np.arange(x.size) + 0.5) / x.size)))


def whitened_samples(path, discard_act=5):
    """Cold-chain samples, whitened by the Fisher matrix; also the names and effective sample size."""
    with h5py.File(path, "r") as f:
        cov, inj = f["fisher/covariance"][:], f["fisher/injection"][:]
        names = [n.decode() for n in f["fisher/param_names"][:]]
    chain = HDFBackend(path, read_only=True).get_chain()["model_0"][:, 0, :, 0, :]  # (steps, walkers, ndim)
    # autocorrelation time in steps; short chains can give an unreliable estimate below one step
    tau = max(float(np.max(get_integrated_act(chain, average=True))), 1.0)
    chain = chain[min(int(np.ceil(discard_act * tau)), chain.shape[0] // 2):]
    n_eff = chain.shape[0] * chain.shape[1] / tau

    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)
    # z = L^{-1} (theta - inj) / sigma with corr = L L^T: the Fisher Gaussian becomes N(0, I)
    L = np.linalg.cholesky(corr)
    x = (chain.reshape(-1, len(names)) - inj) / sigma
    z = np.linalg.solve(L, x.T).T
    return names, x, z, n_eff


def metrics(path, n_floor=200, seed=0):
    names, x, z, n_eff = whitened_samples(path)
    joint = dict(kl=gaussian_kl(z.mean(0), np.cov(z.T)), w2=gaussian_w2(z.mean(0), np.cov(z.T)))
    per_param = {
        n: dict(
            sigma_ratio=x[:, i].std(),
            bias=x[:, i].mean(),
            w1=marginal_w1(x[:, i]),
            kl=gaussian_kl(np.array([x[:, i].mean()]), np.array([[x[:, i].var()]])),
        )
        for i, n in enumerate(names)
    }

    # noise floor: the same metrics for n_eff independent draws from the Fisher Gaussian itself
    rng, d, n = np.random.default_rng(seed), len(names), int(n_eff)
    floor = {"kl": [], "w2": [], "w1": []}
    for _ in range(n_floor):
        s = rng.standard_normal((n, d))
        floor["kl"].append(gaussian_kl(s.mean(0), np.cov(s.T)))
        floor["w2"].append(gaussian_w2(s.mean(0), np.cov(s.T)))
        floor["w1"].append(marginal_w1(s[:, 0]))
    floor = {k: (np.mean(v), np.percentile(v, 95)) for k, v in floor.items()}
    return dict(names=names, n_eff=n_eff, joint=joint, per_param=per_param, floor=floor)


def table(headers, rows):
    """Plain-text table: the first column left-aligned, the others right-aligned."""
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]
    line = lambda cells: "  ".join(
        str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(cells, widths))
    )
    return "\n".join([line(headers), "  ".join("-" * w for w in widths)] + [line(r) for r in rows])


def report(path):
    """Metrics of one run, printed and written to metrics.txt in the run's folder."""
    m = metrics(path)
    run = os.path.basename(os.path.dirname(os.path.abspath(path)))
    f, j = m["floor"], m["joint"]
    rows = [
        [n, f"{v['sigma_ratio']:.3f}", f"{v['bias']:+.3f}", f"{v['w1']:.3f}", f"{v['kl']:.4f}"]
        for n, v in m["per_param"].items()
    ]
    text = "\n".join([
        run,
        "",
        "Units: the Fisher Gaussian, centred on the injection, is a standard normal (offsets in Fisher sigma).",
        f"Effective samples: {m['n_eff']:.0f}",
        "",
        table(["", "value", "noise floor (mean)", "noise floor (95%)"], [
            ["joint KL(MCMC || Fisher)", f"{j['kl']:.4f}", f"{f['kl'][0]:.4f}", f"{f['kl'][1]:.4f}"],
            ["joint W2", f"{j['w2']:.3f}", f"{f['w2'][0]:.3f}", f"{f['w2'][1]:.3f}"],
            ["marginal W1 (largest)", f"{max(v['w1'] for v in m['per_param'].values()):.3f}",
             f"{f['w1'][0]:.3f}", f"{f['w1'][1]:.3f}"],
        ]),
        "",
        table(["parameter", "sigma_MCMC/sigma_F", "bias/sigma_F", "W1", "KL (1D)"], rows),
        "",
    ])
    print("\n" + text)
    with open(os.path.join(os.path.dirname(os.path.abspath(path)), "metrics.txt"), "w") as fh:
        fh.write(text)
    return run, m


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    paths = sys.argv[1:] or sorted(p for p in glob.glob(os.path.join(here, "*", "*.h5")) if not p.endswith("Fisher.h5"))
    results = [report(p) for p in paths]
    if len(results) > 1:
        rows = []
        for run, m in results:
            worst = max(m["per_param"], key=lambda n: m["per_param"][n]["w1"])
            f = m["floor"]
            rows.append([run, f"{m['joint']['kl']:.4f}", f"{m['joint']['w2']:.3f}",
                         f"{m['per_param'][worst]['w1']:.3f} ({worst})", f"{f['kl'][1]:.4f}", f"{f['w2'][1]:.3f}",
                         f"{f['w1'][1]:.3f}"])
        print("\nSummary (floors: 95th percentile)\n")
        print(table(["run", "joint KL", "joint W2", "largest marginal W1", "KL floor", "W2 floor", "W1 floor"], rows))
