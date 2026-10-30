"""Finite-difference stencils shared by all derivative classes.

A stencil of a given ``order`` and ``kind`` combines the function evaluated at
the parameter offsets returned by :func:`offsets` into a first derivative:

    df/dx ~ sum_k c_k f(x + offsets_k) / delta

Coefficients from Fornberg (1988): https://doi.org/10.1090/S0025-5718-1988-0935077-0
"""

import numpy as np

KINDS = ("central", "forward", "backward")
ORDERS = (2, 4, 6, 8)

# Each array is ordered like the offsets returned by `offsets` (most negative
# first). Central stencils skip the zero offset, forward stencils start at it
# and backward stencils end at it.
STENCILS = {
    "central": {
        2: np.asarray([-1 / 2, 1 / 2]),
        4: np.asarray([1 / 12, -2 / 3, 2 / 3, -1 / 12]),
        6: np.asarray([-1 / 60, 3 / 20, -3 / 4, 3 / 4, -3 / 20, 1 / 60]),
        8: np.asarray(
            [1 / 280, -4 / 105, 1 / 5, -4 / 5, 4 / 5, -1 / 5, 4 / 105, -1 / 280]
        ),
    },
    "forward": {
        2: np.asarray([-3 / 2, 2, -1 / 2]),
        4: np.asarray([-25 / 12, 4, -3, 4 / 3, -1 / 4]),
        6: np.asarray([-49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6]),
        8: np.asarray(
            [-761 / 280, 8, -14, 56 / 3, -35 / 2, 56 / 5, -14 / 3, 8 / 7, -1 / 8]
        ),
    },
    "backward": {
        2: np.asarray([1 / 2, -2, 3 / 2]),
        4: np.asarray([1 / 4, -4 / 3, 3, -4, 25 / 12]),
        6: np.asarray([1 / 6, -6 / 5, 15 / 4, -20 / 3, 15 / 2, -6, 49 / 20]),
        8: np.asarray(
            [1 / 8, -8 / 7, 14 / 3, -56 / 5, 35 / 2, -56 / 3, 14, -8, 761 / 280]
        ),
    },
}


def check_stencil(order, kind):
    """Raise a ValueError if no stencil exists for this order and kind."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}.")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}.")


def offsets(delta, order, kind):
    """Parameter offsets at which the function must be evaluated.

    Args:
        delta (float): finite-difference step size.
        order (int): order of the stencil, one of ORDERS.
        kind (str): one of KINDS.

    Returns:
        np.ndarray: the offsets, ordered like the stencil coefficients.
    """
    check_stencil(order, kind)
    if kind == "central":
        # symmetric positions around 0, excluding 0
        half = order // 2
        positions = list(range(-half, 0)) + list(range(1, half + 1))
    elif kind == "forward":
        positions = list(range(order + 1))
    else:
        positions = list(range(-order, 1))
    return np.array(positions) * delta


def apply_stencil(func_steps, delta, order, kind, xp=np):
    """Combine function evaluations at the stencil offsets into a derivative.

    Args:
        func_steps: array whose first axis runs over the offsets from `offsets`.
        delta (float): finite-difference step size.
        order (int): order of the stencil, one of ORDERS.
        kind (str): one of KINDS.
        xp: array module (numpy or cupy) that `func_steps` belongs to.

    Returns:
        The derivative, with the shape of a single entry of `func_steps`.
    """
    check_stencil(order, kind)
    return (
        xp.tensordot(xp.asarray(STENCILS[kind][order]), func_steps, axes=(0, 0))
        / delta
    )
