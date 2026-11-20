"""The full Fisher-matrix pipeline on FEW models, on the CPU and the GPU.

The waveform is proportional to 1 / dist, so the distance element is known
exactly in terms of the SNR: Gamma_dist,dist = SNR^2 / dist^2. The other
elements are checked against the inner products of the derivatives, and the
stable-derivative Fisher matrix against the direct-derivative one.
"""

import h5py
import numpy as np
import pytest

from stableemrifisher.utils.analysis import check_fisher, fisher_matrix
from stableemrifisher.utils.utils import check_if_plunging, fishinv

from cases import DT, TEST_PARAMS
from conftest import assert_only_sky_degenerate

pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def fisher_run(sef, model):
    """Fisher matrix over TEST_PARAMS and dist, at fixed step sizes."""
    param_names = TEST_PARAMS + ["dist"]
    deltas = model.case.deltas(param_names)
    F = sef(model.wave_params, param_names=param_names, deltas=deltas, keep_derivatives=True)
    return dict(
        F=F,
        param_names=param_names,
        deltas=deltas,
        SNR2=sef.SNR2,
        PSD=sef.PSD_funcs,
        derivatives=sef.derivatives,
        fisher_check=sef.fisher_check,
    )


def test_distance_element_is_snr_squared(fisher_run, model):
    i = fisher_run["param_names"].index("dist")
    dist = model.wave_params["dist"]
    np.testing.assert_allclose(
        fisher_run["F"][i, i] * dist**2 / fisher_run["SNR2"], 1.0, rtol=1e-12, atol=0.0
    )


def test_fisher_is_symmetric(fisher_run):
    F = fisher_run["F"]
    assert np.array_equal(F, F.T)


def test_fisher_is_not_degenerate(fisher_run):
    assert_only_sky_degenerate(
        fisher_run["F"], fisher_run["param_names"], fisher_run["fisher_check"]
    )


def test_fisher_is_matrix_of_derivative_inner_products(fisher_run, backend):
    expected = fisher_matrix(fisher_run["derivatives"], fisher_run["PSD"], DT, use_gpu=backend.use_gpu)
    np.testing.assert_allclose(fisher_run["F"], expected, rtol=1e-12, atol=0.0)


def test_fisher_is_reproducible(sef, model, fisher_run):
    F = sef(model.wave_params, param_names=fisher_run["param_names"], deltas=fisher_run["deltas"])
    np.testing.assert_allclose(F, fisher_run["F"], rtol=1e-12, atol=0.0)


def test_stable_matches_direct_fisher(make_sef, backend, model, fisher_run):
    direct_sef = make_sef(backend, deriv_type="direct")
    F_direct = direct_sef(
        model.wave_params,
        param_names=fisher_run["param_names"],
        deltas=fisher_run["deltas"],
    )
    F = fisher_run["F"]
    scale = 1.0 / np.sqrt(np.diag(F))
    difference = np.abs(F - F_direct) * np.outer(scale, scale)
    i, j = np.unravel_index(difference.argmax(), difference.shape)
    names = fisher_run["param_names"]
    assert difference.max() < 1e-5, (
        f"largest normalized difference {difference.max():.1e}, "
        f"for ({names[i]}, {names[j]})"
    )


def test_output_files(sef, model, tmp_path):
    param_names = ["m1", "dist", "Phi_phi0"]
    deltas = model.case.deltas(param_names)
    # single-step grids go through the step-size search, which writes the steps
    F = sef(
        model.wave_params,
        param_names=param_names,
        delta_range={p: [deltas[p]] for p in param_names},
        filename=str(tmp_path),
        suffix="test",
        save_derivatives=True,
    )
    with h5py.File(tmp_path / "Fisher_test.h5", "r") as f:
        np.testing.assert_array_equal(f["Fisher"][()], F)
        assert f["derivatives"].shape[0] == len(param_names)
    saved_deltas = (tmp_path / "stable_deltas_test.txt").read_text()
    assert all(p in saved_deltas for p in param_names)


def test_plunge_check(sef, model):
    params = model.wave_params
    traj_params = [params[p] for p in ("m1", "m2", "a", "p0", "e0", "xI0")]
    assert check_if_plunging(sef.traj_module, 0.01, *traj_params) == 0.01
    T = check_if_plunging(sef.traj_module, 10.0, *traj_params, chop_inspiral_time=6.0)
    assert 0.0 < T < 10.0


# ----------------------------------------- Fisher-matrix helpers, without FEW


def test_fishinv_on_badly_scaled_matrix():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(4, 4))
    A = B @ B.T + 4 * np.eye(4)  # well conditioned
    J = np.diag([1e6, 1.0, 1.0, 1.0])
    F = np.linalg.inv(J) @ A @ np.linalg.inv(J)  # Fisher matrix for M = 1e6 * x
    np.testing.assert_allclose(fishinv(1e6, F, 0), J @ np.linalg.inv(A) @ J, rtol=1e-10)


@pytest.mark.parametrize(
    "F, key, expected",
    [
        (np.diag([1.0, 2.0, 3.0]), "positive_definite", True),
        (np.diag([1.0, 0.0, 3.0]), "no_effect", ["b"]),
        (np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "degenerate", [["a", "b"]]),
        (np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "negative", True),
    ],
    ids=["positive-definite", "no-effect", "degenerate", "negative"],
)
def test_check_fisher(F, key, expected):
    check = check_fisher(F, ["a", "b", "c"])
    if key == "degenerate":
        assert [params for params, _ in check[key]] == expected
    else:
        assert check[key] == expected
    if key != "positive_definite":
        assert not check["positive_definite"]
