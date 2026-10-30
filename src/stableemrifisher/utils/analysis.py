"""Noise-weighted inner products and quantities derived from them."""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


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

    a = xp.atleast_2d(a)
    b = xp.atleast_2d(b)
    PSD = xp.atleast_2d(xp.asarray(PSD))
    N = a.shape[1]
    df = (N * dt) ** -1

    # 1. Frequency masking
    if freq_mask is None:
        freq = xp.fft.rfftfreq(N) / dt
        if use_gpu and hasattr(freq, "get"):
            freq = freq.get()

        freq_mask = np.full(len(freq), True, dtype=bool)
        if fmin is not None:
            freq_mask &= freq > fmin
        if fmax is not None:
            freq_mask &= freq < fmax
        freq_mask = freq_mask[1:]  # skip f = 0
    # else: assume the f = 0 component is already stripped.

    # 2. Windowing & Template FFT
    if window is not None:
        window = xp.atleast_2d(xp.asarray(window))
        a_in = a * window
        b_in = b * window
    else:
        a_in = a
        b_in = b

    if xp.iscomplexobj(a_in):
        a_fft_plus = (dt * xp.fft.rfft(a_in.real, axis=-1)[:, 1:])[:, freq_mask]
        a_fft_cross = (dt * xp.fft.rfft(a_in.imag, axis=-1)[:, 1:])[:, freq_mask]

        b_fft_plus = (dt * xp.fft.rfft(b_in.real, axis=-1)[:, 1:])[:, freq_mask]
        b_fft_cross = (dt * xp.fft.rfft(b_in.imag, axis=-1)[:, 1:])[:, freq_mask]

        inner_prod = (
            4
            * df
            * (
                (a_fft_plus.conj() * b_fft_plus + a_fft_cross * b_fft_cross.conj()).real
                / PSD[:, freq_mask]
            ).sum()
        )

    else:
        a_fft = (dt * xp.fft.rfft(a_in, axis=-1)[:, 1:])[:, freq_mask]
        b_fft = (dt * xp.fft.rfft(b_in, axis=-1)[:, 1:])[:, freq_mask]

        # Compute inner products over given channels
        inner_prod = 4 * df * ((a_fft.conj() * b_fft).real / PSD[:, freq_mask]).sum()

    if use_gpu:
        inner_prod = inner_prod.get()

    if normalize:
        den1 = inner_product(a, a, PSD, dt, window, fmin, fmax, freq_mask, normalize=False, use_gpu=use_gpu)
        den2 = inner_product(b, b, PSD, dt, window, fmin, fmax, freq_mask, normalize=False, use_gpu=use_gpu)
        return inner_prod / np.sqrt(den1 * den2)

    return inner_prod


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

    Args:
        derivatives (sequence of arrays): derivative of the waveform with
            respect to each parameter, each of shape (N_channels, N).
        PSD, dt, window, fmin, fmax, freq_mask, use_gpu: as in `inner_product`.

    Returns:
        np.ndarray: the symmetric Fisher matrix, shape (N_params, N_params).
    """
    npar = len(derivatives)
    Fisher = np.zeros((npar, npar), dtype=np.float64)
    for i in range(npar):
        for j in range(i, npar):
            Fisher[i, j] = np.float64(
                np.real(
                    inner_product(
                        derivatives[i],
                        derivatives[j],
                        PSD,
                        dt,
                        window=window,
                        fmin=fmin,
                        fmax=fmax,
                        freq_mask=freq_mask,
                        use_gpu=use_gpu,
                    )
                )
            )
            Fisher[j, i] = Fisher[i, j]
    return Fisher


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
