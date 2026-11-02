"""The GPU gives the same results as the CPU. Skipped without a GPU."""

import numpy as np
import pytest

from conftest import CPU
from stableemrifisher.fisher.defaults import default_param_names
from stableemrifisher.utils.analysis import inner_product

pytestmark = [pytest.mark.fast, pytest.mark.gpu]


def test_inner_product(gpu):
    rng = np.random.default_rng(0)
    a, b = rng.normal(scale=1e-20, size=(2, 2, 4096))
    PSD = np.full((2, 2048), 1e-40)
    on_cpu = inner_product(a, b, PSD, 10.0)
    on_gpu = inner_product(
        gpu.xp.asarray(a), gpu.xp.asarray(b), gpu.xp.asarray(PSD), 10.0, use_gpu=True
    )
    np.testing.assert_allclose(float(on_gpu), float(on_cpu), rtol=1e-12)


def test_waveform_and_fisher(make_sef, model, gpu):
    cpu_sef = make_sef(CPU)
    gpu_sef = make_sef(gpu)
    param_names = default_param_names(cpu_sef.derivative.base_generator.waveform_generator)
    deltas = model.case.deltas(param_names)

    F_cpu = cpu_sef(model.wave_params, param_names=param_names, deltas=deltas)
    F_gpu = gpu_sef(model.wave_params, param_names=param_names, deltas=deltas)

    waveform_gpu = gpu.to_numpy(gpu_sef.waveform)
    assert np.abs(waveform_gpu - cpu_sef.waveform).max() < 1e-8 * np.abs(cpu_sef.waveform).max()

    scale = 1.0 / np.sqrt(np.diag(F_cpu))
    difference = np.abs(F_gpu - F_cpu) * np.outer(scale, scale)
    assert difference.max() < 1e-6
