"""Waveform derivatives on FEW models, on the CPU and the GPU.

- The stable derivative builds the waveform itself, from FEW's trajectory,
  amplitudes and mode selection. Its distance derivative is exactly -h / dist,
  so -dist * (dh/d dist) must reproduce FEW's waveform to machine precision,
  whichever way the modes are chosen.
- For every other parameter, the stable derivative must agree with the
  direct finite difference of FEW's waveform. Both use the modes FEW selects
  for the reference waveform: the stable derivative always, the direct one
  with fix_modes (its default).
"""

import pytest

from cases import RELATIVE_STEPS
from stableemrifisher.derivatives import StableDerivative
from stableemrifisher.fisher.defaults import default_param_names
from stableemrifisher.utils.analysis import inner_product
from stableemrifisher.utils.noise import generate_psd, sensitivity_lwa

pytestmark = pytest.mark.fast

ORDER = 4
# (Delta|Delta) / (dh|dh) of the difference Delta of the two derivatives; they
# agree to better than 1e-12 at the steps in cases.RELATIVE_STEPS
MAX_MISMATCH = 1e-10


def psd_for(waveform, dt, backend):
    return generate_psd(
        waveform, dt, noise_PSD=sensitivity_lwa, noise_kwargs={}, use_gpu=backend.use_gpu
    )


def max_relative_difference(a, b, backend):
    xp = backend.xp
    return float(xp.abs(a - b).max() / xp.abs(b).max())


def selected_modes(stable):
    """The (l, m, n) modes chosen by the stable derivative at its reference point."""
    return [(int(l), int(m), int(n)) for l, m, n in stable.engine.cache["mode_selection"]]


def internal_waveform(stable, dist):
    """The waveform the stable derivative builds, from its distance derivative."""
    return -dist * stable("dist", 0.0)


@pytest.mark.parametrize(
    "mode_kwargs",
    [
        {},
        dict(mode_selection_threshold=1e-3),
        dict(mode_selection=[(2, 2, 0), (2, 2, 1), (3, 3, 0)]),
    ],
    ids=["few-default", "call-threshold", "mode-list"],
)
def test_stable_waveform_matches_few(sef, model, backend, waveform_kwargs, mode_kwargs):
    stable = sef.derivative
    params = model.wave_params
    stable.set_reference(params, **waveform_kwargs, **mode_kwargs)
    assert max_relative_difference(
        internal_waveform(stable, params["dist"]), stable.waveform, backend
    ) < 1e-12


def test_stable_waveform_matches_few_with_init_threshold(model, backend, waveform_kwargs):
    """FEW's waveform ignores a threshold given at initialization, and so must the derivative."""
    stable = StableDerivative(
        waveform_class=model.waveform_class,
        waveform_generator_kwargs=model.generator_kwargs(
            backend, mode_selector_kwargs=dict(mode_selection_threshold=1e-2)
        ),
        use_gpu=backend.use_gpu,
    )
    params = model.wave_params
    stable.set_reference(params, **waveform_kwargs)
    assert max_relative_difference(
        internal_waveform(stable, params["dist"]), stable.waveform, backend
    ) < 1e-12


def assert_stable_matches_direct(sef, direct, params, backend, waveform_kwargs, param, delta):
    stable = sef.derivative
    stable.set_reference(params, **waveform_kwargs)
    d_stable = stable(param, delta, order=ORDER)

    direct.set_reference(params, **waveform_kwargs)
    assert direct.modes == selected_modes(stable)
    d_direct = direct(param, delta, order=ORDER)

    PSD = psd_for(stable.waveform, waveform_kwargs["dt"], backend)

    def ip(a, b):
        return float(inner_product(a, b, PSD, waveform_kwargs["dt"], use_gpu=backend.use_gpu))

    difference = d_stable - d_direct
    mismatch = ip(difference, difference) / ip(d_direct, d_direct)
    assert mismatch < MAX_MISMATCH, (
        f"stable and direct derivatives with respect to {param} differ: "
        f"mismatch {mismatch:.1e} at step {delta:.1e}"
    )


@pytest.mark.parametrize("param", list(RELATIVE_STEPS))
def test_stable_matches_direct(sef, direct, model, backend, waveform_kwargs, param):
    few_waveform = sef.derivative.base_generator.waveform_generator
    if param not in default_param_names(few_waveform):
        pytest.skip(f"{param} is not a parameter of {model.case.id}")
    delta = model.case.deltas([param])[param]
    assert_stable_matches_direct(
        sef, direct, model.wave_params, backend, waveform_kwargs, param, delta
    )


def test_deriv_kwargs_reach_the_derivative(make_sef, model, backend, waveform_kwargs):
    unfixed = make_sef(backend, deriv_type="direct", deriv_kwargs=dict(fix_modes=False))
    assert unfixed.derivative.fix_modes is False
    unfixed.derivative.set_reference(model.wave_params, **waveform_kwargs)
    assert unfixed.derivative.modes is None
    assert "mode_selection" not in unfixed.derivative.waveform_kwargs


def test_fixed_modes_across_a_mode_change(sef, direct, model, backend, waveform_kwargs):
    """With this threshold and step, FEW keeps one mode fewer at qS - 2 delta
    than at the reference (Kerr source of cases.py). Without fixed modes, the
    direct derivative is then off by order unity."""
    params = model.wave_params
    assert_stable_matches_direct(
        sef,
        direct,
        params,
        backend,
        dict(waveform_kwargs, mode_selection_threshold=1e-3),
        "qS",
        1e-3 * params["qS"],
    )
