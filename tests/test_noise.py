"""PSDs on the waveform's frequency grid, one per channel of the waveform."""

import numpy as np
import pytest

from stableemrifisher.utils.noise import generate_psd

pytestmark = pytest.mark.fast

N = 1000
DT = 10.0


def flat_psd(f, level=1.0):
    return np.full(len(f), level)


def two_channel_psd(f):
    return np.array([np.full(len(f), 1.0), np.full(len(f), 2.0)])


@pytest.mark.parametrize("n_channels", [1, 2, 3])
def test_one_psd_for_every_channel(backend, n_channels):
    waveform = backend.xp.zeros((n_channels, N))
    PSD = generate_psd(waveform, DT, noise_PSD=flat_psd, noise_kwargs={}, use_gpu=backend.use_gpu)
    assert len(PSD) == n_channels
    for psd in PSD:
        assert psd.shape == (N // 2,)
        assert float(psd[0]) == 1.0


def test_one_psd_per_call(backend):
    waveform = backend.xp.zeros((2, N))
    PSD = generate_psd(
        waveform, DT, noise_PSD=flat_psd, noise_kwargs=[{"level": 1.0}, {"level": 2.0}],
        use_gpu=backend.use_gpu,
    )
    assert [float(psd[0]) for psd in PSD] == [1.0, 2.0]


def test_several_channels_per_call(backend):
    waveform = backend.xp.zeros((2, N))
    PSD = generate_psd(waveform, DT, noise_PSD=two_channel_psd, noise_kwargs={}, use_gpu=backend.use_gpu)
    assert [float(psd[0]) for psd in PSD] == [1.0, 2.0]


def test_extra_psds_are_dropped(backend):
    PSD = generate_psd(
        backend.xp.zeros(N), DT, noise_PSD=two_channel_psd, noise_kwargs={}, use_gpu=backend.use_gpu
    )
    assert [float(psd[0]) for psd in PSD] == [1.0]


def test_too_few_psds_are_rejected(backend):
    waveform = backend.xp.zeros((3, N))
    with pytest.raises(ValueError, match="3 channels"):
        generate_psd(
            waveform, DT, noise_PSD=flat_psd, noise_kwargs=[{"level": 1.0}, {"level": 2.0}],
            use_gpu=backend.use_gpu,
        )
