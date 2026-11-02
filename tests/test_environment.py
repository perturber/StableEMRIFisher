"""The installation itself: version metadata, GPU setup and the optional LISA response."""

import tomllib
from pathlib import Path

import pytest

from conftest import RESPONSE_INSTALL_HINT, gpu_status, response_status

pytestmark = pytest.mark.fast


def test_version_matches_pyproject():
    import stableemrifisher

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject, "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    assert stableemrifisher.__version__ == version, (
        f"The installed StableEMRIFisher reports version "
        f"{stableemrifisher.__version__}, but pyproject.toml says {version}. "
        "Reinstall it (pip install -e .) to update the installed metadata."
    )


def test_public_modules_import():
    import stableemrifisher.derivatives  # noqa: F401
    import stableemrifisher.fisher  # noqa: F401
    import stableemrifisher.utils  # noqa: F401


def test_gpu_setup_is_complete():
    """A GPU that CuPy can use must also be usable by FEW."""
    few_backend, text = gpu_status()
    if few_backend is None and "FEW has no CUDA backend" not in text:
        pytest.skip(f"no GPU: {text}")
    assert few_backend is not None, text


def test_response_packages_load_together():
    ok, text = response_status()
    if "not installed" in text:
        pytest.skip(f"LISA response packages: {text}. {RESPONSE_INSTALL_HINT}")
    assert ok, f"LISA response packages: {text}. {RESPONSE_INSTALL_HINT}"
