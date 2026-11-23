"""The step-size search warns when the chosen step is at an edge of the trial grid."""

import logging

import numpy as np
import pytest

from stableemrifisher.fisher.stepsize import stable_deltas

pytestmark = pytest.mark.fast

GRID = np.geomspace(1e-1, 1e-5, 9)


class ModelDerivative:
    """A derivative D(delta) = v + truncation * delta**2 + noise / delta, with fixed vectors."""

    def __init__(self, truncation=0.0, noise=0.0):
        rng = np.random.default_rng(0)
        self.v, self.w, self.n = rng.normal(size=(3, 50))
        self.truncation, self.noise = truncation, noise

    def needs_step_size(self, param):
        return True

    def __call__(self, param, delta, order, kind):
        return self.v + self.truncation * delta**2 * self.w + self.noise / delta * self.n


def search(derivative, caplog):
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="stableemrifisher"):
        deltas, _, _ = stable_deltas(
            derivative, {"x": GRID}, lambda a, b: float(np.dot(a, b)), 4, {"x": "central"}
        )
    edge_warnings = [r.getMessage() for r in caplog.records if "outside the grid" in r.getMessage()]
    return deltas["x"], edge_warnings


def test_minimum_at_smallest_step_warns(caplog):
    delta, warnings = search(ModelDerivative(truncation=1.0), caplog)
    assert delta == GRID[-1]
    assert len(warnings) == 1 and "smallest" in warnings[0] and "smaller" in warnings[0]


def test_minimum_at_largest_step_warns(caplog):
    delta, warnings = search(ModelDerivative(noise=1e-6), caplog)
    assert delta == GRID[1]  # the largest step that can be chosen: the first is only compared against
    assert len(warnings) == 1 and "largest" in warnings[0] and "larger" in warnings[0]


def test_minimum_inside_the_grid_does_not_warn(caplog):
    delta, warnings = search(ModelDerivative(truncation=1.0, noise=1e-9), caplog)
    assert GRID[-1] < delta < GRID[1]
    assert warnings == []
