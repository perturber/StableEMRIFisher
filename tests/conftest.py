"""Shared fixtures and the environment report of the StableEMRIFisher test suite.

Every test is marked either `fast` or `slow`. The two marks are levels of
thoroughness: the fast level (the default, a few minutes on a CPU) runs the
fast tests, and the slow level runs them together with the more extensive
checks (step-size search, LISA response):

    pytest              # fast tests (the default; same as pytest -m fast)
    pytest -m slow      # fast and slow tests

Tests that can run on a GPU run once per backend; the GPU runs are skipped,
with the reason, when no GPU is available. The report at the top of the run
says which backends and optional packages were found.
"""

import functools
import importlib.metadata
import platform
import subprocess
import sys
from dataclasses import dataclass

import matplotlib
import numpy as np
import pytest

from cases import DT, T_OBS, WAVEFORM_CASES

# write plots to files without opening windows
matplotlib.use("Agg")

SPEED_MARKS = ("fast", "slow")

RESPONSE_INSTALL_HINT = (
    'Linux: pip install -e ".[response]"; macOS: build both packages from '
    "source first (see the README section on the LISA response)"
)
GPU_INSTALL_HINT = 'Linux x86_64 with an NVIDIA GPU: pip install -e ".[cuda12x]" (or ".[cuda11x]")'

# versions the examples (and the response tests) are written against
RESPONSE_PINS = {"lisaanalysistools": "1.2.0", "fastlisaresponse": "1.1.17"}


def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


@functools.lru_cache(maxsize=None)
def gpu_status():
    """(FEW CUDA backend name or None, description of the GPU setup)."""
    try:
        import cupy
    except ImportError:
        return None, "CuPy not installed"
    try:
        n_devices = cupy.cuda.runtime.getDeviceCount()
    except Exception as e:  # no driver, no device, broken CUDA install
        return None, f"CuPy installed, but no usable CUDA device ({e})"
    if n_devices == 0:
        return None, "CuPy installed, but no CUDA device found"
    name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
    name = name.decode() if isinstance(name, bytes) else name

    import few

    for backend in ("cuda12x", "cuda11x"):
        if few.has_backend(backend):
            return backend, f"{name} (FEW backend {backend})"
    return None, (
        f"{name} found, but FEW has no CUDA backend; install "
        "fastemriwaveforms-cuda12x (or -cuda11x)"
    )


@functools.lru_cache(maxsize=None)
def response_status():
    """(True if the LISA response can be used, description)."""
    missing = [dist for dist in RESPONSE_PINS if _version(dist) is None]
    if missing:
        return False, f"{', '.join(missing)} not installed"

    # Loading both packages into one process can crash it (e.g. the PyPI macOS
    # wheels), so try it in a separate process first.
    probe = (
        "import lisatools.sensitivity, fastlisaresponse; "
        "from fastlisaresponse import ResponseWrapper; "
        "from lisatools.detector import EqualArmlengthOrbits; "
        "EqualArmlengthOrbits()"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        detail = (result.stderr.strip().splitlines() or [f"exit code {result.returncode}"])[-1]
        return False, f"installed, but loading them together fails: {detail}"
    return True, "available"


def _response_versions():
    parts = []
    for dist, pin in RESPONSE_PINS.items():
        version = _version(dist)
        if version is None:
            parts.append(f"{dist} not installed")
        elif version != pin:
            parts.append(f"{dist} {version} (the examples need {pin})")
        else:
            parts.append(f"{dist} {version}")
    return ", ".join(parts)


def environment_report():
    import stableemrifisher

    gpu_backend, gpu_text = gpu_status()
    response_ok, response_text = response_status()
    lines = [
        f"StableEMRIFisher {stableemrifisher.__version__} | "
        f"FEW {_version('fastemriwaveforms')} | numpy {np.__version__} | "
        f"Python {platform.python_version()} | {platform.system()} {platform.machine()}",
        f"GPU:      {gpu_text}" + ("" if gpu_backend else f". {GPU_INSTALL_HINT}"),
        f"Response: {_response_versions()}"
        + ("" if response_ok else f". {response_text}. {RESPONSE_INSTALL_HINT}"),
        "Note: FEW downloads its data files the first time a waveform model is "
        "used, which needs an internet connection and can take a few minutes.",
    ]
    return lines


def pytest_report_header(config):
    return environment_report()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # runs before pytest deselects tests by mark (-m)
    unmarked = [
        item.nodeid
        for item in items
        if sum(item.get_closest_marker(m) is not None for m in SPEED_MARKS) != 1
    ]
    if unmarked:
        raise pytest.UsageError(
            "Every test must be marked either 'fast' or 'slow' (exactly one): "
            + ", ".join(unmarked)
        )
    # the slow level includes the fast tests
    for item in items:
        if item.get_closest_marker("fast") is not None:
            item.add_marker(pytest.mark.slow)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    gpu_backend, gpu_text = gpu_status()
    response_ok, response_text = response_status()
    not_tested = []
    if gpu_backend is None:
        not_tested.append(f"GPU ({gpu_text}). {GPU_INSTALL_HINT}")
    if not response_ok:
        not_tested.append(f"LISA response ({response_text}). {RESPONSE_INSTALL_HINT}")
    markexpr = config.getoption("markexpr")
    if markexpr == "fast":
        not_tested.append('slow tests (run them with: pytest -m slow)')

    terminalreporter.section("StableEMRIFisher environment")
    for line in environment_report()[:3]:
        terminalreporter.write_line(line)
    if not_tested:
        terminalreporter.write_line("Not tested in this run:")
        for line in not_tested:
            terminalreporter.write_line(f"  - {line}")


# ------------------------------------------------------------------ helpers

SKY_ANGLES = ("qS", "phiS", "qK", "phiK")


def assert_only_sky_degenerate(F, param_names, check):
    """Check a Fisher matrix computed without a response.

    Without a response, the waveform depends on qS, phiS, qK and phiK only
    through the viewing angle theta_src and the polarization angle psi, so two
    combinations of the sky angles are unconstrained when all four are in the
    matrix. Every other parameter must be constrained.

    Args:
        F (np.ndarray): the Fisher matrix.
        param_names (list of str): its parameters.
        check (dict): its `check_fisher` report.
    """
    from stableemrifisher.utils.analysis import check_fisher

    assert not check["negative"] and not check["no_effect"]
    n_sky = len(set(param_names) & set(SKY_ANGLES))
    assert len(check["degenerate"]) == max(0, n_sky - 2), check["degenerate"]
    for params, _ in check["degenerate"]:
        assert set(params) <= set(SKY_ANGLES), params

    keep = [i for i, p in enumerate(param_names) if p not in SKY_ANGLES]
    sub = F[np.ix_(keep, keep)]
    assert check_fisher(sub, [param_names[i] for i in keep])["positive_definite"]
    scale = 1.0 / np.sqrt(np.diag(sub))
    assert np.linalg.eigvalsh(sub * np.outer(scale, scale)).min() > 0.0


# ------------------------------------------------------------------ fixtures


@dataclass(frozen=True)
class Backend:
    """Array backend of a test: numpy on the CPU or CuPy on a GPU."""

    name: str
    use_gpu: bool
    few_backend: str

    @property
    def xp(self):
        if self.use_gpu:
            import cupy

            return cupy
        return np

    def to_numpy(self, a):
        return a.get() if self.use_gpu and hasattr(a, "get") else np.asarray(a)


CPU = Backend("cpu", False, "cpu")


def gpu_backend():
    """The GPU Backend, or skip the test if there is none."""
    few_backend, reason = gpu_status()
    if few_backend is None:
        pytest.skip(f"GPU not available: {reason}")
    return Backend("gpu", True, few_backend)


@pytest.fixture(
    scope="session", params=["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]
)
def backend(request):
    """Runs a test on the CPU and, if available, on the GPU."""
    return CPU if request.param == "cpu" else gpu_backend()


@pytest.fixture(scope="session")
def gpu():
    """The GPU backend, for tests comparing it with the CPU."""
    return gpu_backend()


@dataclass(frozen=True)
class Model:
    """A waveform case with its resolved FEW class."""

    case: object
    waveform_class: type

    @property
    def wave_params(self):
        return dict(self.case.wave_params)

    def generator_kwargs(self, backend, **extra):
        kwargs = dict(self.case.waveform_class_kwargs, force_backend=backend.few_backend)
        kwargs.update(extra)
        return kwargs


@pytest.fixture(scope="session", params=WAVEFORM_CASES, ids=lambda c: c.id)
def model(request):
    """Runs a test once per waveform model in cases.WAVEFORM_CASES."""
    import few.waveform

    case = request.param
    waveform_class = getattr(few.waveform, case.waveform_class_name, None)
    if waveform_class is None:
        pytest.skip(
            f"{case.waveform_class_name} needs FEW >= {case.min_few_version} "
            f"(installed: {_version('fastemriwaveforms')})"
        )
    return Model(case, waveform_class)


@pytest.fixture(scope="session")
def make_sef(model):
    """Factory for StableEMRIFisher instances on the test model, without a response."""
    from stableemrifisher.fisher import StableEMRIFisher
    from stableemrifisher.utils.noise import sensitivity_lwa

    def make(backend, **kwargs):
        options = dict(
            waveform_class=model.waveform_class,
            waveform_class_kwargs=model.generator_kwargs(backend),
            noise_model=sensitivity_lwa,
            noise_kwargs={},
            T=T_OBS,
            dt=DT,
            der_order=4,
            use_gpu=backend.use_gpu,
        )
        options.update(kwargs)
        return StableEMRIFisher(**options)

    return make


@pytest.fixture(scope="session")
def sef(make_sef, backend):
    """StableEMRIFisher with the stable derivative, shared by the tests."""
    return make_sef(backend)


@pytest.fixture(scope="session")
def direct(model, backend):
    """DirectDerivative on the test model, shared by the tests."""
    from stableemrifisher.derivatives import DirectDerivative

    return DirectDerivative(
        waveform_class=model.waveform_class,
        waveform_generator_kwargs=model.generator_kwargs(backend),
        use_gpu=backend.use_gpu,
    )


@pytest.fixture(scope="session")
def waveform_kwargs():
    return dict(T=T_OBS, dt=DT)
