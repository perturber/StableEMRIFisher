"""The full 1-year Fisher matrix of a Kerr EMRI with the LISA response reproduces a stored reference.

The reference (tests/data/kerr_prograde_nonplunging_withresponse) is the Fisher matrix of the
run of examples/check_against_mcmc/injection_config_kerr_prograde_nonplunging_withresponse.py,
computed by exec.py on a GPU, with its step sizes. Here it is recomputed with the same set-up and
step sizes, and the two Fisher Gaussians N(0, F^-1) are compared with the KL divergence.
Runs only with -m slow, a GPU and the LISA response packages.
"""

import ast
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import RESPONSE_INSTALL_HINT, response_status

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

DATA = Path(__file__).parent / "data" / "kerr_prograde_nonplunging_withresponse"

# KL divergence below which the two Fisher matrices agree up to numerical error: a change of
# about 0.1% in the variance along any direction gives a KL divergence of about 2.5e-7
KL_TOLERANCE = 1e-6


def gaussian_kl(F_ref, F_new):
    """KL(N(0, F_ref^-1) || N(0, F_new^-1)), computed with both normalised by diag(F_ref)."""
    scale = 1.0 / np.sqrt(np.diag(F_ref))
    A = F_ref * np.outer(scale, scale)
    B = F_new * np.outer(scale, scale)
    M = np.linalg.solve(A, B)  # A^-1 B, same trace and determinant as B A^-1
    return 0.5 * (np.trace(M) - len(M) - np.linalg.slogdet(M)[1])


def test_gaussian_kl_of_identical_and_scaled_matrices():
    F = np.array([[4.0, 1.0], [1.0, 3.0]])
    assert abs(gaussian_kl(F, F)) < 1e-14
    # every variance halved: 0.5 * d * (2 - 1 - ln 2)
    np.testing.assert_allclose(gaussian_kl(F, 2 * F), (1 - np.log(2)), rtol=1e-12)


@pytest.fixture(scope="module")
def reference():
    with h5py.File(DATA / "Fisher.h5", "r") as f:
        F = f["Fisher"][:]
    info = json.loads((DATA / "injection.json").read_text())
    info["deltas"] = ast.literal_eval((DATA / "stable_deltas.txt").read_text())
    return F, info


def test_full_fisher_matches_reference(gpu, reference):
    ok, text = response_status()
    if not ok:
        pytest.skip(f"LISA response packages: {text}. {RESPONSE_INSTALL_HINT}")

    from few.utils.constants import YRSID_SI
    from few.waveform import FastKerrEccentricEquatorialFlux
    from fastlisaresponse import ResponseWrapper
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.sensitivity import A1TDISens, E1TDISens, get_sensitivity

    from stableemrifisher.fisher import StableEMRIFisher

    F_ref, info = reference
    T, dt = info["T"], info["dt"]
    # as in examples/check_against_mcmc/exec.py and the injection config
    sef = StableEMRIFisher(
        waveform_class=FastKerrEccentricEquatorialFlux,
        waveform_class_kwargs=dict(
            inspiral_kwargs=dict(err=1e-11),
            sum_kwargs=dict(pad_output=True),
            force_backend=gpu.few_backend,
        ),
        waveform_kwargs=dict(mode_selection_threshold=1e-5),
        ResponseWrapper=ResponseWrapper,
        ResponseWrapper_kwargs=dict(
            Tobs=T, dt=dt, index_lambda=8, index_beta=7, t0=20000.0,
            remove_garbage="zero", flip_hx=True, is_ecliptic_latitude=False,
            force_backend=gpu.few_backend, orbits=EqualArmlengthOrbits(force_backend=gpu.few_backend),
            order=25, tdi="1st generation", tdi_chan="AE",
        ),
        noise_model=get_sensitivity,
        noise_kwargs=[
            dict(sens_fn=channel, model="scirdv1", stochastic_params=[T * YRSID_SI])
            for channel in (A1TDISens, E1TDISens)
        ],
        deriv_type="stable",
        der_order=4,
        T=T,
        dt=dt,
        use_gpu=True,
    )
    F = sef(dict(info["wave_params"]), param_names=info["param_names"], deltas=info["deltas"])

    assert sef.T == pytest.approx(info["T_window"], rel=1e-12)
    assert np.sqrt(sef.SNR2) == pytest.approx(info["SNR"], rel=1e-9)
    kl = gaussian_kl(F_ref, F)
    assert kl < KL_TOLERANCE, f"KL divergence to the reference Fisher matrix: {kl:.2e}"
