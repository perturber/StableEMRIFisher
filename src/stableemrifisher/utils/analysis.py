"""Noise-weighted inner products and quantities derived from them.

The inner product of two time series is computed in two stages:
`fourier_coefficients` transforms each time series (the expensive step, one
FFT per channel), and `inner_product_fd` combines two sets of coefficients
(an element-wise product and a sum). `inner_product` runs both stages; when
the same time series enters several inner products, as in a Fisher matrix,
its coefficients are computed once and reused.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


@dataclass
class FourierCoefficients:
    """Fourier coefficients of a time series, as returned by `fourier_coefficients`.

    Attributes:
        values: the coefficients dt * rfft(window * a) at the kept positive
            frequencies, shape (N_channels, n_freq). For a complex input
            (h+ - i hx), a pair (plus, cross) of such arrays, the transforms
            of its real and imaginary parts.
        df (float): frequency resolution 1 / (N dt) of the time series.
        freq_mask (np.ndarray of bool): which positive frequencies (without
            f = 0) are kept.
    """

    values: Any
    df: float
    freq_mask: np.ndarray


def _frequency_mask(N, dt, fmin, fmax, freq_mask, use_gpu):
    """Mask over the positive frequencies (without f = 0) of a length-N series."""
    if freq_mask is not None:
        # assume the f = 0 component is already stripped
        return freq_mask

    xp = cp if use_gpu else np
    freq = xp.fft.rfftfreq(N) / dt
    if use_gpu and hasattr(freq, "get"):
        freq = freq.get()

    freq_mask = np.full(len(freq), True, dtype=bool)
    if fmin is not None:
        freq_mask &= freq > fmin
    if fmax is not None:
        freq_mask &= freq < fmax
    return freq_mask[1:]  # skip f = 0


def fourier_coefficients(
    a,
    dt,
    window=None,
    fmin=None,
    fmax=None,
    freq_mask=None,
    use_gpu=False,
):
    """
    Fourier coefficients of a time series, for use in `inner_product_fd`.

    Computes a_c(f) = dt * rfft(window * a_c), the discrete approximation of
    the continuous Fourier transform of each channel c, at the positive
    frequencies np.fft.rfftfreq(N, dt)[1:] selected by fmin, fmax or freq_mask.
    A complex input is read as h+ - i hx, and its real and imaginary parts are
    transformed separately.

    Args:
        a (array): time series, shape (N_channels, N) or (N,).
        dt, window, fmin, fmax, freq_mask, use_gpu: as in `inner_product`.

    Returns:
        FourierCoefficients: the coefficients, with the frequency resolution
        and mask they were computed with.
    """
    xp = cp if use_gpu else np

    a = xp.atleast_2d(a)
    N = a.shape[1]
    df = (N * dt) ** -1
    freq_mask = _frequency_mask(N, dt, fmin, fmax, freq_mask, use_gpu)

    if window is not None:
        a = a * xp.atleast_2d(xp.asarray(window))

    def transform(x):
        return (dt * xp.fft.rfft(x, axis=-1)[:, 1:])[:, freq_mask]

    if xp.iscomplexobj(a):
        values = (transform(a.real), transform(a.imag))
    else:
        values = transform(a)
    return FourierCoefficients(values, df, freq_mask)


def inner_product_fd(a_f, b_f, PSD, use_gpu=False):
    """
    Noise-weighted inner product from two sets of Fourier coefficients.

    Computes (a|b) = 4 df Re sum_c sum_f conj(a_c(f)) b_c(f) / S_c(f) from the
    output of `fourier_coefficients`, without any FFT.

    Args:
        a_f, b_f (FourierCoefficients): coefficients of the two time series,
            computed with the same length, dt and frequency mask.
        PSD (array or list): one-sided PSD of each channel, as in
            `inner_product`.
        use_gpu (bool, optional): coefficients are CuPy arrays. Default False.

    Returns:
        float: the inner product.
    """
    if a_f.df != b_f.df or not np.array_equal(a_f.freq_mask, b_f.freq_mask):
        raise ValueError(
            "The Fourier coefficients were computed on different frequency grids."
        )
    if isinstance(a_f.values, tuple) != isinstance(b_f.values, tuple):
        raise ValueError(
            "Cannot combine the coefficients of a complex and a real time series."
        )

    xp = cp if use_gpu else np
    PSD = xp.atleast_2d(xp.asarray(PSD))
    df = a_f.df
    freq_mask = a_f.freq_mask

    if isinstance(a_f.values, tuple):
        a_fft_plus, a_fft_cross = a_f.values
        b_fft_plus, b_fft_cross = b_f.values
        inner_prod = (
            4
            * df
            * (
                (a_fft_plus.conj() * b_fft_plus + a_fft_cross * b_fft_cross.conj()).real
                / PSD[:, freq_mask]
            ).sum()
        )
    else:
        a_fft, b_fft = a_f.values, b_f.values
        # Compute inner products over given channels
        inner_prod = 4 * df * ((a_fft.conj() * b_fft).real / PSD[:, freq_mask]).sum()

    if use_gpu:
        inner_prod = inner_prod.get()
    return inner_prod


def inner_product(
    a,
    b,
    PSD,
    dt,
    window=None,
    fmin=None,
    fmax=None,
    freq_mask=None,
    normalize=False,
    use_gpu=False,
):
    """
    Noise-weighted inner product (a|b) of two time series, summed over channels.

    Computes

        (a|b) = 4 df Re sum_c sum_f conj(a_c(f)) b_c(f) / S_c(f),

    where a_c(f) = dt * rfft(a_c) is the discrete approximation of the
    continuous Fourier transform of channel c, S_c is the one-sided PSD of
    that channel and df = 1 / (N dt). The sum runs over the positive
    frequencies of the real FFT, without the f = 0 bin.

    A complex input is read as h+ - i hx: its real part (h+) and imaginary
    part (-hx) are treated as two separate channels that share the PSD of
    that row.

    This runs `fourier_coefficients` on both time series (once, if `b` is
    `a`) and combines them with `inner_product_fd`. To reuse the coefficients
    of a time series in several inner products, call those two functions
    directly.

    Args:
        a (array): first time series, shape (N_channels, N) or (N,).
        b (array): second time series, same shape as `a`.
        PSD (array or list): one-sided PSD of each channel, evaluated at the
            frequencies np.fft.rfftfreq(N, dt)[1:] (as returned by
            `generate_psd`), shape (N_channels, N // 2).
        dt (float): sampling interval in seconds.
        window (array, optional): window multiplied into both time series
            before the FFT, e.g. from `tukey`. Default None (no window).
        fmin (float, optional): only frequencies above fmin (Hz) are summed.
        fmax (float, optional): only frequencies below fmax (Hz) are summed.
        freq_mask (array of bool, optional): precomputed mask over the
            frequencies without f = 0. Overrides fmin and fmax.
        normalize (bool, optional): if True, return the overlap
            (a|b) / sqrt((a|a)(b|b)) instead.
        use_gpu (bool, optional): inputs are CuPy arrays. Default False.

    Returns:
        float: the inner product (or the overlap if `normalize` is True).
    """
    xp = cp if use_gpu else np
    same = b is a

    # a complex `a` makes the inner product split both inputs into h+ and hx
    if xp.iscomplexobj(a) and not xp.iscomplexobj(b):
        b = xp.asarray(b).astype(complex)

    a_f = fourier_coefficients(a, dt, window, fmin, fmax, freq_mask, use_gpu)
    b_f = a_f if same else fourier_coefficients(b, dt, window, fmin, fmax, freq_mask, use_gpu)

    inner_prod = inner_product_fd(a_f, b_f, PSD, use_gpu)

    if normalize:
        den1 = inner_product_fd(a_f, a_f, PSD, use_gpu)
        den2 = inner_product_fd(b_f, b_f, PSD, use_gpu)
        return inner_prod / np.sqrt(den1 * den2)

    return inner_prod


def fisher_matrix_from_coefficients(coefficients, PSD, use_gpu=False):
    """
    Fisher matrix F_ij = (dh/dx_i | dh/dx_j) from the Fourier coefficients of
    the waveform derivatives.

    Each derivative is transformed only once (by `fourier_coefficients`), and
    the n(n+1)/2 elements are computed with `inner_product_fd`.

    Args:
        coefficients (sequence of FourierCoefficients): coefficients of the
            derivative with respect to each parameter.
        PSD, use_gpu: as in `inner_product_fd`.

    Returns:
        np.ndarray: the symmetric Fisher matrix, shape (N_params, N_params).
    """
    xp = cp if use_gpu else np
    PSD = xp.atleast_2d(xp.asarray(PSD))

    npar = len(coefficients)
    Fisher = np.zeros((npar, npar), dtype=np.float64)
    for i in range(npar):
        for j in range(i, npar):
            Fisher[i, j] = np.float64(
                np.real(inner_product_fd(coefficients[i], coefficients[j], PSD, use_gpu))
            )
            Fisher[j, i] = Fisher[i, j]
    return Fisher


def fisher_matrix(
    derivatives,
    PSD,
    dt,
    window=None,
    fmin=None,
    fmax=None,
    freq_mask=None,
    use_gpu=False,
):
    """
    Fisher matrix from waveform derivatives, F_ij = (dh/dx_i | dh/dx_j).

    Each derivative is Fourier transformed once, see
    `fisher_matrix_from_coefficients`.

    Args:
        derivatives (sequence of arrays): derivative of the waveform with
            respect to each parameter, each of shape (N_channels, N).
        PSD, dt, window, fmin, fmax, freq_mask, use_gpu: as in `inner_product`.

    Returns:
        np.ndarray: the symmetric Fisher matrix, shape (N_params, N_params).
    """
    coefficients = [
        fourier_coefficients(d, dt, window, fmin, fmax, freq_mask, use_gpu)
        for d in derivatives
    ]
    return fisher_matrix_from_coefficients(coefficients, PSD, use_gpu)


def check_fisher(Fisher, param_names, rtol=1e-12, component_tol=0.1):
    """
    Find the degeneracies of a Fisher matrix, without modifying it.

    The matrix is first normalized by its diagonal, C_ij = F_ij / sqrt(F_ii F_jj),
    so that the result does not depend on the units of the parameters. An
    eigenvalue of C below `rtol` times the largest one marks a combination of
    parameters that the data cannot constrain; the parameters with an
    eigenvector component above `component_tol` are reported as degenerate. A
    parameter with F_ii = 0 has no effect on the waveform at all.

    Args:
        Fisher (np.ndarray): Fisher matrix, shape (N_params, N_params).
        param_names (list of str): names of the parameters, in matrix order.
        rtol (float): relative eigenvalue threshold. Default 1e-12.
        component_tol (float): eigenvector component threshold. Default 0.1.

    Returns:
        dict with keys:
            "no_effect" (list of str): parameters with F_ii = 0.
            "degenerate" (list of (list of str, float)): for each unconstrained
                combination, the parameters involved and the relative
                eigenvalue.
            "positive_definite" (bool): True if the matrix is invertible and
                positive definite.
            "negative" (bool): True if the matrix has an eigenvalue below
                -rtol (relative), i.e. is not positive semi-definite.
    """
    Fisher = np.asarray(Fisher, dtype=np.float64)
    names = np.asarray(param_names)
    diag = np.diag(Fisher)

    no_effect = [str(n) for n in names[diag == 0.0]]
    keep = diag > 0.0
    scale = 1.0 / np.sqrt(diag[keep])
    C = Fisher[np.ix_(keep, keep)] * np.outer(scale, scale)

    degenerate = []
    negative = bool((diag < 0.0).any())
    if C.size:
        eigvals, eigvecs = np.linalg.eigh(C)
        rel = eigvals / eigvals.max()
        negative = negative or bool((rel < -rtol).any())
        for k in np.where(np.abs(rel) < rtol)[0]:
            involved = [
                str(n) for n in names[keep][np.abs(eigvecs[:, k]) > component_tol]
            ]
            degenerate.append((involved, float(rel[k])))

    return {
        "no_effect": no_effect,
        "degenerate": degenerate,
        "positive_definite": not (no_effect or degenerate or negative),
        "negative": negative,
    }


def optimal_snr(waveform, PSD, dt, window=None, fmin=None, fmax=None, freq_mask=None, use_gpu=False):
    """
    Optimal signal-to-noise ratio of a waveform, sqrt((h|h)).

    Args:
        waveform (array): the signal h, shape (N_channels, N) or (N,).
        PSD, dt, window, fmin, fmax, freq_mask, use_gpu: as in `inner_product`.

    Returns:
        float: the optimal SNR, summed in quadrature over the channels.
    """

    return np.sqrt(
        inner_product(
            waveform,
            waveform,
            PSD,
            dt,
            window=window,
            fmin=fmin,
            fmax=fmax,
            freq_mask=freq_mask,
            use_gpu=use_gpu,
        )
    )
