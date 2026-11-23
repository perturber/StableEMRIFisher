"""General helpers: windowing, padding, plunge check and Fisher-matrix inversion."""

import logging

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from few.utils.constants import YRSID_SI

logger = logging.getLogger("stableemrifisher")


def tukey(N, alpha=0.5, use_gpu=False):
    """
    Tukey (tapered cosine) window of length N.

    The window is 1 in the middle and rises/falls as a half cosine over a
    fraction alpha/2 of the samples at each end. alpha = 1 gives a Hann
    window. Multiply it into a time series (or pass it as `window` to
    `inner_product`) to suppress the spectral leakage caused by the abrupt
    start and end of the signal.

    Args:
        N (int): number of samples.
        alpha (float): fraction of the window that is tapered, 0 < alpha <= 1.
            Default 0.5.
        use_gpu (bool): return a CuPy array. Default False.

    Returns:
        array: the window, shape (N,).
    """
    if use_gpu:
        xp = cp
    else:
        xp = np

    t = xp.linspace(0.0, 1.0, N)
    window = xp.ones(N)
    condition1 = (t > (1 - alpha / 2)) & (t <= 1)
    condition2 = (t >= 0) & (t < alpha / 2)
    window[condition1] = 0.5 * (
        1 + xp.cos(2 * xp.pi / alpha * ((t[condition1] - 1 + alpha / 2) - 1))
    )
    window[condition2] = 0.5 * (
        1 + xp.cos(2 * xp.pi / alpha * (t[condition2] - alpha / 2))
    )
    return window


def padding(a, b, use_gpu=False):
    """
    Make time series `a` the same length as time series `b`.

    `a` is zero-padded at the end if it is shorter than `b`, and truncated if
    it is longer. Multi-channel arrays are treated channel by channel. Used
    to align perturbed waveforms (which can end earlier, e.g. if the
    perturbed system plunges sooner) with the reference waveform before
    finite differencing.

    Args:
        a (array): time series to adjust, shape (N_a,) or (N_channels, N_a).
        b (array): reference time series with the same number of dimensions,
            shape (N_b,) or (N_channels, N_b).
        use_gpu (bool): inputs are CuPy arrays. Default False.

    Returns:
        array: `a` with the length of `b`.
    """
    if use_gpu:
        xp = cp
    else:
        xp = np

    a = xp.asarray(a)
    b = xp.asarray(b)

    assert a.ndim == b.ndim

    if a.ndim > 1:
        a_temp = []

        for i in range(len(a)):
            if len(a[i]) < len(b[i]):
                a_temp.append(xp.concatenate((a[i], xp.zeros(len(b[i]) - len(a[i])))))

            elif len(a[i]) > len(b[i]):
                a_temp.append(a[i][: len(b[i])])

            else:
                a_temp.append(a[i])

        a = xp.array(a_temp)

    else:
        if len(a) < len(b):
            a = xp.concatenate((a, xp.zeros(len(b) - len(a))))
        elif len(a) > len(b):
            a = a[: len(b)]

    return a


def fishinv(M, Fisher, index_of_M=0):
    """
    Invert a Fisher matrix after reparametrising one parameter M -> ln M.

    A large parameter such as the primary mass (M ~ 1e6) makes the Fisher
    matrix badly conditioned. Changing it to ln M rescales its row and column
    by M, the matrix is inverted in those coordinates, and the covariance is
    transformed back:

        J = diag(1, ..., M, ..., 1)        (dM/d ln M = M at index_of_M)
        F_lnM = J^T F J
        C = J (F_lnM)^-1 J^T

    Args:
        M (float): value of the parameter at index `index_of_M`.
        Fisher (np.ndarray): Fisher matrix, shape (N_params, N_params).
        index_of_M (int): index of that parameter in the Fisher matrix.
            Default 0.

    Returns:
        np.ndarray: the covariance matrix (inverse Fisher matrix) in the
        original parameters.
    """

    # Jacobian for Fisher = partial old/partial new, going from M -> lnM

    J = np.eye(len(Fisher))
    J[index_of_M, index_of_M] = M

    Fisher_lnM = J.T @ Fisher @ J

    Fisher_lnM_inv = np.linalg.inv(Fisher_lnM)

    # Jacobian for Covariance = partial new/partial old, going from lnM -> M

    Fisher_inv = J.T @ Fisher_lnM_inv @ J

    return Fisher_inv


def check_if_plunging(
    traj,
    T,
    m1,
    m2,
    a,
    p0,
    e0,
    xI0,
    *add_args,
    Phi_phi0=0.0,
    Phi_theta0=0.0,
    Phi_r0=0.0,
    chop_inspiral_time=0.0,
    **traj_kwargs,
):
    """
    Shorten the observation time if the compact object plunges before T.

    Runs the trajectory for T years. If it ends earlier than T (the compact
    object plunges), the returned time is the plunge time minus
    `chop_inspiral_time` hours; the waveform close to the plunge is hard to
    differentiate stably, so trimming it makes the Fisher matrix more stable.
    Otherwise T is returned unchanged.

    A trajectory ending less than 1 s before T does not count as a plunge,
    since FEW trajectories can stop slightly short of T even without one.

    Args:
        traj: FEW trajectory module instance (e.g. `EMRIInspiral`).
        T (float): observation time in years.
        m1, m2, a, p0, e0, xI0: trajectory parameters.
        *add_args: any further trajectory parameters, passed on positionally.
        Phi_phi0, Phi_theta0, Phi_r0 (float): initial phases. Default 0.
        chop_inspiral_time (float): hours to remove before the plunge.
            Default 0.
        **traj_kwargs: further kwargs for `traj`, e.g. dt.

    Returns:
        float: observation time in years.
    """
    ONE_HOUR = 60 * 60

    out = traj(
        m1,
        m2,
        a,
        p0,
        e0,
        xI0,
        *add_args,
        Phi_phi0=Phi_phi0,
        Phi_theta0=Phi_theta0,
        Phi_r0=Phi_r0,
        T=T,
        **traj_kwargs,
    )
    t_traj = out[0]
    if t_traj[-1] < T * YRSID_SI - 1.0:
        T_new = (t_traj[-1] - chop_inspiral_time * ONE_HOUR) / YRSID_SI
        logger.warning("Body is plunging! Expect instabilities.")
        logger.warning(
            "Removed last %s hours of inspiral. New evolution time: %s years",
            chop_inspiral_time,
            T_new,
        )
        return T_new

    logger.debug("Body is not plunging, Fisher should be stable.")
    return T
