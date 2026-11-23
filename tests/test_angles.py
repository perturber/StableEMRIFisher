"""Analytic derivatives of the viewing angle and antenna patterns, against finite differences."""

import numpy as np
import pytest

from stableemrifisher.utils.deriv_utils.deriv_angles import (
    fplus_fcross,
    fplus_fcross_derivs,
    viewing_angle_partials,
)

pytestmark = pytest.mark.fast

ANGLES = ("qS", "phiS", "qK", "phiK")
# generic sky positions and spin orientations, away from the poles
SKY = [
    dict(qS=np.pi / 3, phiS=np.pi / 4, qK=np.pi / 6, phiK=np.pi / 8),
    dict(qS=2.1, phiS=5.0, qK=0.7, phiK=3.3),
    dict(qS=1.2, phiS=0.4, qK=2.6, phiK=4.9),
]
THETA_KEYS = {
    "qS": "del theta_src / del qS",
    "phiS": "del theta_src / del phi_S",
    "qK": "del theta_src / del qK",
    "phiK": "del theta_src / del phi_K",
}


def numerical_derivative(f, angles, name, h=1e-4):
    """Fourth-order central difference of f(**angles) with respect to `name`."""

    def at(offset):
        return np.asarray(f(**dict(angles, **{name: angles[name] + offset})))

    return (at(-2 * h) - 8 * at(-h) + 8 * at(h) - at(2 * h)) / (12 * h)


@pytest.mark.parametrize("angles", SKY)
@pytest.mark.parametrize("name", ANGLES)
def test_viewing_angle_partials(angles, name):
    analytic = viewing_angle_partials(**angles)[THETA_KEYS[name]]
    numerical = numerical_derivative(
        lambda **a: viewing_angle_partials(**a)["theta_src"], angles, name
    )
    np.testing.assert_allclose(analytic, numerical, atol=1e-10)


@pytest.mark.parametrize("angles", SKY)
@pytest.mark.parametrize("name", ANGLES)
def test_antenna_pattern_derivatives(angles, name):
    derivs = fplus_fcross_derivs(**angles, with_respect_to=name)
    analytic = [
        derivs[f"d{F}/d{name}"] for F in ("FplusI", "FcrossI", "FplusII", "FcrossII")
    ]
    numerical = numerical_derivative(fplus_fcross, angles, name)
    np.testing.assert_allclose(analytic, numerical, atol=1e-10)
