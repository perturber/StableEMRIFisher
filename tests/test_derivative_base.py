"""The derivative interface, on a toy waveform whose derivatives are known exactly.

The toy generator returns h+ - i hx = A exp(-i (2 pi f t + phi0)) / dist like a
FEW generator, so these tests need neither FEW nor a GPU.
"""

import numpy as np
import pytest

from stableemrifisher.derivatives import (
    DerivativeBase,
    DirectDerivative,
    StableDerivative,
    get_derivative_class,
)

pytestmark = pytest.mark.fast

PARAMS = dict(A=2.0, f=1e-3, phi0=0.5, dist=1.5)
KWARGS = dict(T=1000.0, dt=10.0)


class ToyGenerator:
    def __init__(self, waveform_class=None, **kwargs):
        pass

    def __call__(self, A, f, phi0, dist, T, dt):
        t = np.arange(0.0, T, dt)
        return A * np.exp(-1j * (2 * np.pi * f * t + phi0)) / dist


def toy_derivatives():
    t = np.arange(0.0, KWARGS["T"], KWARGS["dt"])
    h = ToyGenerator()(**PARAMS, **KWARGS)
    exact = {
        "A": h / PARAMS["A"],
        "f": -2j * np.pi * t * h,
        "phi0": -1j * h,
        "dist": -h / PARAMS["dist"],
    }
    # in the (channels, samples) layout [h+, hx]
    return {p: np.array([d.real, -d.imag]) for p, d in exact.items()}


@pytest.fixture
def direct():
    derivative = DirectDerivative(waveform_class=None, waveform_generator=ToyGenerator)
    derivative.set_reference(PARAMS, **KWARGS)
    return derivative


def test_waveform_is_split_into_plus_and_cross(direct):
    h = ToyGenerator()(**PARAMS, **KWARGS)
    np.testing.assert_array_equal(direct.waveform, [h.real, -h.imag])


@pytest.mark.parametrize("param", ["A", "f", "phi0", "dist"])
@pytest.mark.parametrize("kind", ["central", "forward", "backward"])
def test_direct_derivative(direct, param, kind):
    delta = 1e-4 * PARAMS[param]
    expected = toy_derivatives()[param]
    np.testing.assert_allclose(
        direct(param, delta, order=8, kind=kind),
        expected,
        rtol=0,
        atol=1e-8 * np.abs(expected).max(),
    )


def test_distance_derivative_is_analytic(direct):
    assert not direct.needs_step_size("dist")
    np.testing.assert_allclose(
        direct("dist", 0.0), toy_derivatives()["dist"], rtol=1e-15, atol=0.0
    )


def test_errors(direct):
    with pytest.raises(ValueError):
        direct("not_a_parameter", 1e-3)
    with pytest.raises(ValueError):
        direct("f", 1e-3, order=3)
    unset = DirectDerivative(waveform_class=None, waveform_generator=ToyGenerator)
    with pytest.raises(RuntimeError):
        unset("f", 1e-3)


def test_derivative_class_lookup():
    assert get_derivative_class("stable") is StableDerivative
    assert get_derivative_class("direct") is DirectDerivative
    assert get_derivative_class(DirectDerivative) is DirectDerivative
    for bad in ("bogus", DerivativeBase.__init__, int):
        with pytest.raises(ValueError):
            get_derivative_class(bad)


def test_stable_derivative_rejects_aak():
    class Pn5AAKWaveform:  # stands in for few.waveform.Pn5AAKWaveform
        pass

    assert not StableDerivative.supports_waveform(Pn5AAKWaveform)
    assert DirectDerivative.supports_waveform(Pn5AAKWaveform)
