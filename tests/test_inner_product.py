"""Noise-weighted inner products, checked against closed-form results.

The signals are sinusoids at exact Fourier frequencies, in white noise of
PSD S0, for which the inner product is known exactly: a sinusoid of amplitude
A at a positive frequency below Nyquist has (h|h) = A^2 N dt / S0, and
sinusoids at different frequencies, or sine and cosine at the same frequency,
are orthogonal.
"""

import numpy as np
import pytest

from stableemrifisher.utils.analysis import (
    fisher_matrix,
    fourier_coefficients,
    inner_product,
    inner_product_fd,
    optimal_snr,
)
from stableemrifisher.utils.utils import tukey

pytestmark = pytest.mark.fast

N = 4096
DT = 10.0
S0 = 1e-40
RTOL = 1e-12


def sinusoid(k, amplitude=1e-20, phase=0.0):
    """Sinusoid at the k-th Fourier frequency k / (N dt)."""
    return amplitude * np.sin(2 * np.pi * k * np.arange(N) / N + phase)


def psd(backend, n_channels=1, level=S0):
    return [backend.xp.full(N // 2, level) for _ in range(n_channels)]


def close(a, b, rtol=RTOL):
    return np.isclose(float(a), float(b), rtol=rtol, atol=0.0)


def random_channels(seed, n_channels=2):
    return np.random.default_rng(seed).normal(scale=1e-20, size=(n_channels, N))


@pytest.mark.parametrize("k", [1, 37, N // 2 - 1])
def test_sinusoid_norm(backend, k):
    A = 3e-20
    h = backend.xp.asarray(sinusoid(k, A))
    expected = A**2 * N * DT / S0
    assert close(inner_product(h, h, psd(backend), DT, use_gpu=backend.use_gpu), expected)
    assert close(optimal_snr(h, psd(backend), DT, use_gpu=backend.use_gpu), np.sqrt(expected))


def test_orthogonality(backend):
    xp = backend.xp
    a = xp.asarray(sinusoid(37))
    b = xp.asarray(sinusoid(80))
    c = xp.asarray(sinusoid(37, phase=np.pi / 2))
    norm = inner_product(a, a, psd(backend), DT, use_gpu=backend.use_gpu)
    for other in (b, c):
        overlap = inner_product(a, other, psd(backend), DT, use_gpu=backend.use_gpu)
        assert abs(float(overlap)) < 1e-12 * float(norm)


def test_symmetry_linearity_and_scaling(backend):
    xp = backend.xp
    a = xp.asarray(random_channels(1))
    b = xp.asarray(random_channels(2))
    S = psd(backend, 2)

    def ip(x, y, PSD=S, **kwargs):
        return inner_product(x, y, PSD, DT, use_gpu=backend.use_gpu, **kwargs)

    assert close(ip(a, b), ip(b, a))
    assert close(ip(2.5 * a + b, b), 2.5 * ip(a, b) + ip(b, b))
    assert close(ip(a, b, PSD=psd(backend, 2, 2 * S0)), ip(a, b) / 2)
    assert close(ip(a, a, normalize=True), 1.0)
    assert float(ip(a, a)) > 0.0


def test_channels_add(backend):
    xp = backend.xp
    a = random_channels(3)
    b = random_channels(4)
    total = inner_product(xp.asarray(a), xp.asarray(b), psd(backend, 2), DT, use_gpu=backend.use_gpu)
    per_channel = [
        inner_product(xp.asarray(a[i]), xp.asarray(b[i]), psd(backend), DT, use_gpu=backend.use_gpu)
        for i in range(2)
    ]
    assert close(total, sum(per_channel))


def test_complex_input_is_plus_and_cross(backend):
    """h+ - i hx gives the same inner product as the two channels [h+, hx]."""
    xp = backend.xp
    a = random_channels(5)
    b = random_channels(6)
    complex_ip = inner_product(
        xp.asarray(a[0] - 1j * a[1]), xp.asarray(b[0] - 1j * b[1]), psd(backend), DT,
        use_gpu=backend.use_gpu,
    )
    real_ip = inner_product(xp.asarray(a), xp.asarray(b), psd(backend, 2), DT, use_gpu=backend.use_gpu)
    assert close(complex_ip, real_ip)


def test_frequency_limits(backend):
    xp = backend.xp
    k = 100
    f_k = k / (N * DT)
    h = xp.asarray(sinusoid(k))
    full = inner_product(h, h, psd(backend), DT, use_gpu=backend.use_gpu)

    def ip(**kwargs):
        return inner_product(h, h, psd(backend), DT, use_gpu=backend.use_gpu, **kwargs)

    # outside the signal's frequency, only FFT round-off remains
    def vanishes(x):
        return abs(float(x)) < 1e-12 * float(full)

    assert close(ip(fmin=0.5 * f_k, fmax=2 * f_k), full)
    assert vanishes(ip(fmin=1.5 * f_k))
    assert vanishes(ip(fmax=0.5 * f_k))

    freq = np.fft.rfftfreq(N, DT)[1:]
    assert close(ip(freq_mask=(freq > 0.5 * f_k) & (freq < 2 * f_k)), full)
    assert vanishes(ip(freq_mask=freq > 1.5 * f_k))


def test_window(backend):
    xp = backend.xp
    a = xp.asarray(random_channels(7))
    b = xp.asarray(random_channels(8))
    w = tukey(N, alpha=0.2, use_gpu=backend.use_gpu)
    windowed = inner_product(a, b, psd(backend, 2), DT, window=w, use_gpu=backend.use_gpu)
    by_hand = inner_product(w * a, w * b, psd(backend, 2), DT, use_gpu=backend.use_gpu)
    assert close(windowed, by_hand)


def test_fisher_matrix_is_matrix_of_inner_products(backend):
    xp = backend.xp
    derivatives = [xp.asarray(random_channels(seed)) for seed in range(10, 14)]
    S = psd(backend, 2)
    F = fisher_matrix(derivatives, S, DT, use_gpu=backend.use_gpu)
    expected = np.array(
        [[float(inner_product(a, b, S, DT, use_gpu=backend.use_gpu)) for b in derivatives] for a in derivatives]
    )
    assert np.array_equal(F, F.T)
    np.testing.assert_allclose(F, expected, rtol=RTOL, atol=0.0)


def test_coefficients_on_different_grids_are_rejected(backend):
    xp = backend.xp
    a = fourier_coefficients(xp.asarray(sinusoid(5)), DT, use_gpu=backend.use_gpu)
    shorter = fourier_coefficients(xp.asarray(sinusoid(5)[: N // 2]), DT, use_gpu=backend.use_gpu)
    complex_ = fourier_coefficients(xp.asarray(sinusoid(5) + 0j), DT, use_gpu=backend.use_gpu)
    with pytest.raises(ValueError):
        inner_product_fd(a, shorter, psd(backend), use_gpu=backend.use_gpu)
    with pytest.raises(ValueError):
        inner_product_fd(a, complex_, psd(backend), use_gpu=backend.use_gpu)
