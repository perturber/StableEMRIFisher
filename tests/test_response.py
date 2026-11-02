"""The Fisher matrix with the LISA response (fastlisaresponse and lisatools)."""

import numpy as np
import pytest

from cases import DT, T_OBS
from conftest import RESPONSE_INSTALL_HINT, response_status
from stableemrifisher.utils.analysis import inner_product

pytestmark = pytest.mark.slow

PARAM_NAMES = ["m1", "p0", "dist", "qS", "phiK", "Phi_phi0"]


@pytest.fixture(scope="module")
def make_response_sef(model, backend):
    ok, text = response_status()
    if not ok:
        pytest.skip(f"LISA response packages: {text}. {RESPONSE_INSTALL_HINT}")

    from fastlisaresponse import ResponseWrapper
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.sensitivity import A1TDISens, E1TDISens, get_sensitivity

    from stableemrifisher.fisher import StableEMRIFisher

    def make(deriv_type):
        return StableEMRIFisher(
            waveform_class=model.waveform_class,
            waveform_class_kwargs=model.generator_kwargs(backend),
            ResponseWrapper=ResponseWrapper,
            ResponseWrapper_kwargs=dict(
                Tobs=T_OBS,
                dt=DT,
                index_lambda=8,
                index_beta=7,
                t0=10000.0,
                remove_garbage="zero",
                flip_hx=True,
                is_ecliptic_latitude=False,
                orbits=EqualArmlengthOrbits(),
                order=20,
                tdi="1st generation",
                tdi_chan="AE",
                force_backend=backend.few_backend,
            ),
            noise_model=get_sensitivity,
            noise_kwargs=[{"sens_fn": A1TDISens}, {"sens_fn": E1TDISens}],
            deriv_type=deriv_type,
            der_order=4,
            use_gpu=backend.use_gpu,
        )

    return make


@pytest.fixture(scope="module")
def response_run(make_response_sef, model):
    sef = make_response_sef("stable")
    deltas = model.case.deltas(PARAM_NAMES)
    F = sef(model.wave_params, param_names=PARAM_NAMES, deltas=deltas, keep_derivatives=True)
    return dict(F=F, sef=sef, deltas=deltas)


def test_response_fisher(response_run, model):
    sef, F = response_run["sef"], response_run["F"]
    assert sef.waveform.shape[0] == 2  # A and E
    assert sef.fisher_check["positive_definite"]
    i = PARAM_NAMES.index("dist")
    np.testing.assert_allclose(
        F[i, i] * model.wave_params["dist"] ** 2 / sef.SNR2, 1.0, rtol=1e-12
    )


def test_response_stable_matches_direct(make_response_sef, response_run, model, backend):
    stable_sef = response_run["sef"]
    direct_sef = make_response_sef("direct")
    direct_sef(
        model.wave_params,
        param_names=PARAM_NAMES,
        deltas=response_run["deltas"],
        keep_derivatives=True,
    )

    def ip(a, b):
        return float(inner_product(a, b, stable_sef.PSD_funcs, DT, use_gpu=backend.use_gpu))

    for p, d_stable, d_direct in zip(PARAM_NAMES, stable_sef.derivatives, direct_sef.derivatives):
        difference = d_stable - d_direct
        mismatch = ip(difference, difference) / ip(d_direct, d_direct)
        assert mismatch < 1e-8, f"{p}: mismatch {mismatch:.1e}"
