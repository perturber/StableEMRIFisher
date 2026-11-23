"""The step-size search over TEST_PARAMS and dist, and the plots and covariance."""

import numpy as np
import pytest

from cases import TEST_PARAMS
from conftest import assert_only_sky_degenerate

pytestmark = pytest.mark.slow

NDELTA = 8


@pytest.fixture(scope="module")
def search_run(sef, model):
    param_names = TEST_PARAMS + ["dist"]  # dist has an analytic derivative and no search
    F = sef(model.wave_params, param_names=param_names, Ndelta=NDELTA)
    return dict(
        F=F,
        param_names=param_names,
        deltas=dict(sef.deltas),
        relerr=dict(sef.relerr),
        SNR2=sef.SNR2,
        fisher_check=sef.fisher_check,
        analytic=[p for p in param_names if not sef.derivative.needs_step_size(p)],
    )


def test_step_sizes(search_run, model):
    for p in search_run["param_names"]:
        delta = search_run["deltas"][p]
        if p in search_run["analytic"]:
            assert delta == 0.0
        else:
            assert 0.0 < delta <= 0.1 * abs(model.wave_params[p] or 1.0), p
            assert search_run["relerr"][p] < 1e-2, (
                f"step-size search for {p} did not converge: relative error "
                f"{search_run['relerr'][p]:.1e}"
            )


def test_searched_fisher(search_run, model):
    F = search_run["F"]
    assert_only_sky_degenerate(F, search_run["param_names"], search_run["fisher_check"])
    i = search_run["param_names"].index("dist")
    np.testing.assert_allclose(
        F[i, i] * model.wave_params["dist"] ** 2 / search_run["SNR2"], 1.0, rtol=1e-12
    )


def test_searched_steps_reproduce_fisher(sef, model, search_run):
    F = sef(model.wave_params, param_names=search_run["param_names"], deltas=search_run["deltas"])
    np.testing.assert_allclose(F, search_run["F"], rtol=1e-12, atol=0.0)


def test_plots_and_covariance(sef, model, tmp_path):
    param_names = ["m1", "p0", "dist"]
    F = sef(
        model.wave_params,
        param_names=param_names,
        Ndelta=4,
        stability_plot=True,
        CovEllipse=True,
        filename=str(tmp_path),
    )
    # covariance @ F is the identity in the parameters scaled by their Fisher
    # errors; unscaled, its elements carry the units of m1 (~1e6) and p0 (~10)
    scale = np.sqrt(np.diag(F))
    np.testing.assert_allclose(
        (sef.covariance @ F) * np.outer(scale, 1.0 / scale), np.eye(len(F)), atol=1e-8
    )
    for p in ("m1", "p0"):  # dist has an analytic derivative and no search
        assert (tmp_path / f"stability_{p}.png").exists()
    assert (tmp_path / "covariance_ellipses.png").exists()
