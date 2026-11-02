"""Finite-difference stencils: exactness on polynomials and convergence order."""

import numpy as np
import pytest

from stableemrifisher.derivatives.finite_difference import (
    KINDS,
    ORDERS,
    apply_stencil,
    check_stencil,
    offsets,
)

pytestmark = pytest.mark.fast


def stencil_derivative(f, x0, delta, order, kind):
    steps = np.array([f(x0 + offset) for offset in offsets(delta, order, kind)])
    return apply_stencil(steps, delta, order, kind)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("order", ORDERS)
def test_exact_for_polynomials_up_to_order(order, kind):
    """A stencil of order n differentiates polynomials of degree <= n exactly."""
    poly = np.polynomial.Polynomial(np.random.default_rng(order).normal(size=order + 1))
    x0 = 0.3
    np.testing.assert_allclose(
        stencil_derivative(poly, x0, 0.05, order, kind), poly.deriv()(x0), rtol=1e-9
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("order", ORDERS)
def test_convergence_order(order, kind):
    """Halving the step divides the error by 2**order.

    For a polynomial of degree order + 1 the error is exactly C delta**order
    (all higher derivatives vanish), so the ratio is exact up to round-off.
    """
    poly = np.polynomial.Polynomial([0.0] * (order + 1) + [1.0])  # x**(order + 1)
    x0 = 0.3
    errors = [
        abs(stencil_derivative(poly, x0, delta, order, kind) - poly.deriv()(x0))
        for delta in (0.1, 0.05)
    ]
    assert np.log2(errors[0] / errors[1]) == pytest.approx(order, abs=0.01)


def test_applies_to_arrays():
    t = np.linspace(0.0, 1.0, 50)

    def f(x):
        return np.array([np.sin(x * t), np.cos(x * t)])

    derivative = stencil_derivative(f, 2.0, 1e-3, 4, "central")
    expected = np.array([t * np.cos(2.0 * t), -t * np.sin(2.0 * t)])
    assert derivative.shape == (2, 50)
    np.testing.assert_allclose(derivative, expected, atol=1e-10)


def test_offsets_layout():
    np.testing.assert_array_equal(offsets(0.5, 4, "central"), [-1.0, -0.5, 0.5, 1.0])
    np.testing.assert_array_equal(offsets(0.5, 2, "forward"), [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(offsets(0.5, 2, "backward"), [-1.0, -0.5, 0.0])


@pytest.mark.parametrize("order, kind", [(3, "central"), (4, "sideways")])
def test_invalid_stencils_are_rejected(order, kind):
    with pytest.raises(ValueError):
        check_stencil(order, kind)
