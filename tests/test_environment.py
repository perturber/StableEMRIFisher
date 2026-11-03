"""The installation itself: version metadata, GPU setup and the optional LISA response."""

import re
import tomllib
from pathlib import Path

import pytest

from conftest import RESPONSE_INSTALL_HINT, gpu_status, response_status

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[1]


def pyproject_version():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_version_matches_pyproject():
    import stableemrifisher

    version = pyproject_version()
    assert stableemrifisher.__version__ == version, (
        f"The installed StableEMRIFisher reports version "
        f"{stableemrifisher.__version__}, but pyproject.toml says {version}. "
        "Reinstall it (pip install -e .) to update the installed metadata."
    )


def test_citation_names_the_current_version():
    """The citation example in the README and CITATION.cff cite this version.

    After a version change in pyproject.toml, update the version (only) in the
    README's citation section and in CITATION.cff; the DOI is the one for all
    versions and stays the same.
    """
    version = pyproject_version()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    citation = readme.split("<!-- citation-start")[1].split("<!-- citation-end")[0]
    cited = re.findall(r"\bv(\d+\.\d+\.\d+)\b", citation)
    assert len(cited) >= 3, "no version found in the README's citation example"
    assert set(cited) == {version}, (
        f"The README's citation example names version(s) {sorted(set(cited))}, "
        f"but pyproject.toml says {version}."
    )

    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    cff_version = re.search(r"^version:\s*(\S+)", cff, re.MULTILINE).group(1)
    assert cff_version == version, (
        f"CITATION.cff names version {cff_version}, but pyproject.toml says {version}."
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
