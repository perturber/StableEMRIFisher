"""Choice of finite-difference step sizes from the stability of the Fisher diagonal."""

import logging

import numpy as np

logger = logging.getLogger("stableemrifisher")


def stable_deltas(derivative, delta_grids, inner, order, kinds):
    """Pick, for each parameter, the step size at which Gamma_ii is most stable.

    For each trial step delta_k (from largest to smallest), the diagonal Fisher
    element Gamma_ii(delta_k) = (dh/dx_i | dh/dx_i) is computed. The relative
    change between consecutive steps,

        relerr_k = |Gamma_ii(delta_k) - Gamma_ii(delta_{k-1})| / Gamma_ii(delta_k),

    is small where the derivative has converged but is not yet dominated by
    numerical noise. The step with the smallest relerr_k is chosen.

    Parameters whose derivative is analytic (see
    `DerivativeBase.needs_step_size`) get a step of 0.0.

    Args:
        derivative (DerivativeBase): derivative with its reference point set.
        delta_grids (dict): parameter name -> trial step sizes, from largest
            to smallest. A single step is used as given.
        inner (callable): inner(a, b) -> float, the noise-weighted inner product.
        order (int): order of the finite-difference stencil.
        kinds (dict): parameter name -> stencil kind ("central", "forward" or
            "backward").

    Returns:
        tuple:
            deltas (dict): parameter name -> chosen step size.
            relerr (dict): parameter name -> relerr at the chosen step, an
                estimate of the relative error of Gamma_ii (0.0 for analytic
                derivatives; absent for single-step grids).
            curves (dict): parameter name -> (grid, Gamma, index of the chosen
                step), for the parameters that were searched.
    """
    logger.info("calculating stable deltas...")
    deltas = {}
    relerr_min = {}
    curves = {}

    for param_name, grid in delta_grids.items():

        if not derivative.needs_step_size(param_name):
            # analytic derivative, no step size to choose
            deltas[param_name] = 0.0
            relerr_min[param_name] = 0.0
            continue

        grid = np.asarray(grid)
        if len(grid) == 1:
            # a single step size: nothing to compare it against
            deltas[param_name] = grid[0].item()
            continue

        Gamma = []
        for delta_k in grid:
            del_k = derivative(param_name, delta_k, order=order, kind=kinds[param_name])
            Gammai = inner(del_k, del_k)
            logger.debug("Gamma_ii for %s: %s", param_name, Gammai)
            if np.isnan(Gammai):
                Gamma.append(0.0)  # handle nan's
                logger.warning(
                    "NaN type encountered during "
                    "Fisher calculation! Replacing with 0.0."
                )
            else:
                Gamma.append(Gammai)
        Gamma = np.array(Gamma)

        if (Gamma[1:] == 0.0).all():  # handle non-contributing parameters
            relerr = list(np.ones(len(Gamma) - 1))
        else:
            relerr = []
            for m in range(1, len(Gamma)):
                if Gamma[m - 1] == 0.0:  # handle partially null contributors
                    relerr.append(1.0)
                else:
                    relerr.append(np.abs(Gamma[m] - Gamma[m - 1]) / Gamma[m])
        logger.debug("relative errors for %s: %s", param_name, relerr)

        relerr_min_i = relerr.index(min(relerr))
        if relerr[relerr_min_i] >= 0.01:
            logger.warning(
                "minimum relative error is greater "
                "than 1%% for %s. Fisher may be unstable!",
                param_name,
            )

        # +1 because relerr starts from the second step
        deltas_min_i = relerr_min_i + 1
        deltas[param_name] = grid[deltas_min_i].item()
        relerr_min[param_name] = relerr[relerr_min_i]
        curves[param_name] = (grid, Gamma, deltas_min_i)

    logger.debug("stable deltas: %s", deltas)
    return deltas, relerr_min, curves
