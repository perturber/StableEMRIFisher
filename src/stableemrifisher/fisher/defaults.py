"""Default settings StableEMRIFisher falls back on when the user gives none."""

import numpy as np

# Parameters of the Fisher matrix for each FEW waveform family, keyed by the
# (descriptor, background) of the FEW waveform class.
DEFAULT_PARAM_NAMES = {
    ("eccentric equatorial", "Kerr"): [
        "m1", "m2", "a", "p0", "e0", "dist",
        "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_r0",
    ],
    ("eccentric equatorial", "Schwarzschild"): [
        "m1", "m2", "p0", "e0", "dist",
        "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_r0",
    ],
    ("eccentric inclined", "Kerr"): [
        "m1", "m2", "a", "p0", "e0", "xI0", "dist",
        "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_r0",
    ],
    ("eccentric inclined", "Schwarzschild"): [
        "m1", "m2", "p0", "e0", "xI0", "dist",
        "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_r0",
    ],
}

# Parameters whose value is below the lower (above the upper) threshold are
# differentiated with forward (backward) stencils, so that no finite-difference
# step leaves the allowed range of the parameter.
STENCIL_BOUNDS = {
    "a": [0.05, 0.95],
    "e0": [0.01, 0.7],
    "Phi_phi0": [0.1, 2 * np.pi * 0.9],
    "Phi_r0": [0.1, 2 * np.pi * 0.9],
    "Phi_theta0": [0.1, 2 * np.pi * 0.9],
    "qS": [0.1, np.pi * 0.9],
    "qK": [0.1, np.pi * 0.9],
    "phiS": [0.1, 2 * np.pi * 0.9],
    "phiK": [0.1, 2 * np.pi * 0.9],
}


def default_param_names(waveform):
    """Default Fisher parameters for a FEW waveform class instance.

    Args:
        waveform: FEW waveform class instance (with `descriptor` and
            `background` attributes).

    Returns:
        list of str: parameter names.

    Raises:
        ValueError: if there is no default for this waveform family.
    """
    key = (waveform.descriptor, waveform.background)
    try:
        return list(DEFAULT_PARAM_NAMES[key])
    except KeyError:
        raise ValueError(
            f"No default param_names for a {key[0]} {key[1]} waveform. "
            "Pass param_names explicitly."
        ) from None


def stencil_kind(param, value, default_kind="central", bounds=STENCIL_BOUNDS):
    """Finite-difference kind for a parameter, given its value.

    Returns "forward" ("backward") if the value is at or below the lower
    (above the upper) threshold in `bounds`, and `default_kind` otherwise.
    """
    if param in bounds:
        if value <= bounds[param][0]:
            return "forward"
        if value > bounds[param][1]:
            return "backward"
    return default_kind


def default_delta_grid(param, value, Ndelta):
    """Trial step sizes for the step-size search, from largest to smallest.

    Relative to the parameter value: from 1e-4 to 1e-9 for the masses and the
    orbital parameters, from 1e-1 to 1e-6 for the others (angles, phases).
    For a parameter equal to zero, absolute steps from 1e-4 to 1e-9.

    Args:
        param (str): parameter name.
        value (float): parameter value.
        Ndelta (int): number of trial steps.

    Returns:
        np.ndarray: geometrically spaced steps, shape (Ndelta,).
    """
    if value == 0.0:
        return np.geomspace(1e-4, 1e-9, Ndelta)
    if param in {"m1", "m2", "a", "p0", "e0", "xI0"}:
        return np.geomspace(1e-4 * value, 1e-9 * value, Ndelta)
    return np.geomspace(1e-1 * value, 1e-6 * value, Ndelta)
