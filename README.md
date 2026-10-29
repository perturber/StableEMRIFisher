# StableEMRIFisher (SEF)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/Default%20reversed.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/_static/Default.png">
    <img src="docs/_static/Default.png" alt="StableEMRIFisher Logo" width="300"/>
  </picture>
</p>

<!-- overview-start: this section is also included in docs/index.rst -->
**StableEMRIFisher** (SEF) is a Python package for computing stable Fisher information matrices for Extreme Mass Ratio Inspiral (EMRI) gravitational wave sources, on both CPUs and GPUs. It computes robust numerical derivatives of EMRI waveforms using finite differencing, and returns Fisher matrices that can be used for accelerated parameter estimation analyses or population inference.
<!-- overview-end -->

StableEMRIFisher uses the [FastEMRIWaveforms](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms) package. Please refer to the [documentation](https://stableemrifisher.readthedocs.io) to know how to get started.

## Key Features

<!-- features-start: this section is also included in docs/index.rst -->
- **Stable Numerical Derivatives**: Robust finite difference methods for parameter derivatives of adiabatic Kerr eccentric equatorial waveform models.
- **Stability Checks**: Careful validation of the numerical stability and convergence of the derivatives.
- **GPU Acceleration**: Efficient computation on both CPU (NumPy) and GPU (CuPy) backends.
- **EMRI Waveforms**: Integration with [FastEMRIWaveforms](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms) for accurate EMRI modelling.
- **Fisher Information Matrix Analysis**: Complete parameter estimation uncertainty analysis.
- **Response Function**: Uses [fastlisaresponse](https://github.com/mikekatz04/lisa-on-gpu) to efficiently compute the response of the LISA detector to EMRI waveforms.
- **LISA Noise Models**: Built-in support for LISA detector noise, from an analytic sensitivity curve to TDI noise PSDs from LISAanalysistools.
- **Validated**: Compared against MCMC parameter estimation studies.
<!-- features-end -->

## Installation

<!-- install-requirements-start: this section is also included in docs/installation.rst -->
StableEMRIFisher requires **Python 3.12 or later**. Its main dependency is **FastEMRIWaveforms (FEW) v2.0.0 or later**, the state-of-the-art framework for generating computationally efficient and accurate EMRI waveform models for LISA data analysis. FEW is installed automatically with StableEMRIFisher (for both CPU and GPU).

StableEMRIFisher also installs the following dependencies automatically:

- **NumPy**: Array computation and numerical operations
- **SciPy**: Scientific computing utilities
- **h5py**: HDF5 file support for data storage
- **Cython**: C extensions for Python, needed to build the LISA response packages
- **matplotlib**: Plots for assessing the stability of numerical derivatives
- **corner**: Corner plots of parameter covariances
- **setuptools**: Building `lisa-on-gpu` and `LISAanalysistools` from source
<!-- install-requirements-end -->

<!-- 
## Quick Install (CPU-only)
```bash
# Install from PyPI (coming soon)
pip install stableemrifisher
```
-->

### Development Installation (Recommended)

<!-- install-source-start: this section is also included in docs/installation.rst -->
Clone the repository and install it in editable mode. With conda:

```bash
# Create and activate environment
conda create -n sef_env python=3.12
conda activate sef_env

# Clone and install
git clone https://github.com/perturber/StableEMRIFisher.git
cd StableEMRIFisher
pip install -e .
```
<!-- install-source-end -->

### GPU-Accelerated Installation

<!-- install-gpu-start: this section is also included in docs/installation.rst -->
For GPU acceleration, install with CUDA support. **Note**: GPU support requires Linux x86_64 systems with NVIDIA GPUs and appropriate CUDA drivers.

```bash
# For CUDA 11.x (Linux x86_64 only)
pip install -e ".[cuda11x]"

# For CUDA 12.x (Linux x86_64 only)
pip install -e ".[cuda12x]"
```

For full development, documentation and GPU support, install all of the optional dependencies. For example, on a CUDA 12.x system:

```bash
pip install -e ".[dev,docs,cuda12x]"
```
<!-- install-gpu-end -->

### StableEMRIFisher with the LISA Response

<!-- install-response-start: this section is also included in docs/installation.rst -->
StableEMRIFisher can optionally include the LISA response function, to compute the Fisher matrix in the LISA detector frame. This needs two additional packages:

- **[LISAanalysistools](https://github.com/mikekatz04/LISAanalysistools)** (`lisaanalysistools` on PyPI): LISA orbits and LISA noise power spectral densities (PSDs) that describe the instrumental noise.
- **[lisa-on-gpu](https://github.com/mikekatz04/lisa-on-gpu)** (`fastlisaresponse` on PyPI): the GPU-accelerated time-domain LISA response function.

The examples need specific versions of both packages (`lisaanalysistools` 1.2.0 and `fastlisaresponse` 1.1.17), because newer `lisaanalysistools` releases have restructured the API the examples use. On Linux, install them with the `response` extra:

```bash
# CPU (Linux)
pip install -e ".[response]"

# GPU (CUDA 12.x)
pip install lisaanalysistools-cuda12x==1.2.0 fastlisaresponse-cuda12x==1.1.17
```

> **macOS: the `response` extra does not install these packages.** Their pre-built macOS wheels each bundle a separate copy of the GCC C++ runtime, and loading both into one Python process crashes it (the Jupyter kernel dies when `StableEMRIFisher` sets up the response). On macOS the `response` extra therefore skips them, and you need to **build both packages from source before installing StableEMRIFisher**, preferably `lisaanalysistools` 1.2.0 and `fastlisaresponse` 1.1.17.

Building from source links both packages against the system C++ runtime, which avoids the crash. These versions have no source releases on PyPI, so build them from their GitHub tags. You need a C++ compiler, for example from conda (`conda install -c conda-forge cxx-compiler`, then re-activate the environment). Then:

```bash
# 1. Build lisaanalysistools from source. Its LATW submodule is listed with an SSH
#    address; these variables fetch it over HTTPS instead, for this command only.
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0='url.https://github.com/.insteadOf' GIT_CONFIG_VALUE_0='git@github.com:' \
    pip install "lisaanalysistools @ git+https://github.com/mikekatz04/LISAanalysistools.git@v1.2.0"

# 2. Build fastlisaresponse against that lisaanalysistools. --no-build-isolation makes it
#    compile against the installed lisaanalysistools, so its build tools must be installed first.
pip install scikit-build-core setuptools-scm astropy cmake ninja
pip install --no-build-isolation \
    "fastlisaresponse @ git+https://github.com/mikekatz04/lisa-on-gpu.git@v1.1.17"

# 3. Install StableEMRIFisher
pip install -e .
```

If the packaged GPU builds don't work on your system, install both packages from source by following the instructions in their repositories.

LISAanalysistools is optional. Without it, StableEMRIFisher still imports and runs, but the PSD helpers `write_psd_file` and `load_psd` are not available, so you need to pass your own `noise_model` (or load a pre-computed PSD file with `load_psd_from_file`).
<!-- install-response-end -->

### Verifying the Installation

<!-- install-verify-start: this section is also included in docs/installation.rst -->
Test your installation by running:

```python
import stableemrifisher
from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils.noise import LISATOOLS_AVAILABLE

print(f"StableEMRIFisher {stableemrifisher.__version__}")
print(f"LISAanalysistools available: {LISATOOLS_AVAILABLE}")

# Check if GPU support is available
try:
    import cupy as cp
    print(f"GPU support available: {cp.cuda.is_available()}")
except ImportError:
    print("GPU support not available (CuPy not installed)")
```
<!-- install-verify-end -->

## Documentation

**Full documentation is available at [stableemrifisher.readthedocs.io](https://stableemrifisher.readthedocs.io)**

- [Installation Guide](https://stableemrifisher.readthedocs.io/en/latest/installation.html)
- [Quick Start Tutorial](https://stableemrifisher.readthedocs.io/en/latest/quickstart.html)
- [API Reference](https://stableemrifisher.readthedocs.io/en/latest/api/index.html)
- [Validation Studies](https://stableemrifisher.readthedocs.io/en/latest/validation/validation.html)
- [Example notebooks and scripts](https://github.com/perturber/StableEMRIFisher/tree/main/examples)

### Building Documentation Locally

<!-- build-docs-start: this section is also included in docs/installation.rst -->
Install the documentation dependencies, then build the docs with Sphinx:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make clean
make html

# View documentation
open _build/html/index.html  # macOS
# or
xdg-open _build/html/index.html  # Linux
```

The generated documentation is in `docs/_build/html`; open `index.html` in a web browser.
<!-- build-docs-end -->

## Contributing

<!-- contributing-start: this section is also included in docs/contribution.rst -->
We welcome contributions! Please:

1.  **Report bugs** via [GitHub Issues](https://github.com/perturber/StableEMRIFisher/issues)
2.  **Suggest features** through issue discussions
3.  **Submit pull requests** with improvements
4.  **Improve documentation** and examples
5.  **Validate results** by sharing your comparison studies

For development, install StableEMRIFisher in editable mode with the development and documentation dependencies (add `cuda12x` if you are using GPUs):

```bash
git clone https://github.com/perturber/StableEMRIFisher.git
cd StableEMRIFisher
pip install -e ".[dev,docs]"
```
<!-- contributing-end -->

## Citation

<!-- citation-start: this section is also included in docs/citation.rst -->
If you use StableEMRIFisher in your research, please cite it. Each release of StableEMRIFisher is archived on Zenodo with its own DOI. **We recommend citing the exact version of the code you used**, which makes your results reproducible and helps us debug any issues you report. You can check your installed version with:

```python
import stableemrifisher
print(stableemrifisher.__version__)
```

For the correct citation of any release, including the latest one, see the [StableEMRIFisher Zenodo page](https://doi.org/10.5281/zenodo.23449876). The "Versions" panel there lists every release, and the "Citation" box gives the citation in many styles (APA, BibTeX, and others). For v1.0.0, an APA-style citation reads:

> Kejriwal, S., Burke, O., Chapman-Bird, C., & J. K. Chua, A. (2026). StableEMRIFisher (Version v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23449876

and the corresponding BibTeX entry is:
```bibtex
@software{Kejriwal_StableEMRIFisher_2026,
  author       = {Kejriwal, Shubham and Burke, Ollie and Chapman-Bird, Christian and J. K. Chua, Alvin},
  title        = {StableEMRIFisher},
  version      = {v1.0.0},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23449876},
  url          = {https://doi.org/10.5281/zenodo.23449876}
}
```

The same metadata is in our `CITATION.cff` file, and you can also copy it via the **"Cite this repository"** button in the right sidebar of this GitHub page.

Please also cite the underlying [FastEMRIWaveforms](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms) package and other dependencies as appropriate.

<details>
<summary>FastEMRIWaveforms BibTeX entries</summary>

```bibtex
@article{Chua:2020stf,
   author = "Chua, Alvin J. K. and Katz, Michael L. and Warburton, Niels and Hughes, Scott A.",
   title = "{Rapid generation of fully relativistic extreme-mass-ratio-inspiral waveform templates for LISA data analysis}",
   eprint = "2008.06071",
   archivePrefix = "arXiv",
   primaryClass = "gr-qc",
   doi = "10.1103/PhysRevLett.126.051102",
   journal = "Phys. Rev. Lett.",
   volume = "126",
   number = "5",
   pages = "051102",
   year = "2021"
}

@article{Katz:2021yft,
   author = "Katz, Michael L. and Chua, Alvin J. K. and Speri, Lorenzo and Warburton, Niels and Hughes, Scott A.",
   title = "{Fast extreme-mass-ratio-inspiral waveforms: New tools for millihertz gravitational-wave data analysis}",
   eprint = "2104.04582",
   archivePrefix = "arXiv",
   primaryClass = "gr-qc",
   doi = "10.1103/PhysRevD.104.064047",
   journal = "Phys. Rev. D",
   volume = "104",
   number = "6",
   pages = "064047",
   year = "2021"
}

@article{Speri:2023jte,
   author = "Speri, Lorenzo and Katz, Michael L. and Chua, Alvin J. K. and Hughes, Scott A. and Warburton, Niels and Thompson, Jonathan E. and Chapman-Bird, Christian E. A. and Gair, Jonathan R.",
   title = "{Fast and Fourier: Extreme Mass Ratio Inspiral Waveforms in the Frequency Domain}",
   eprint = "2307.12585",
   archivePrefix = "arXiv",
   primaryClass = "gr-qc",
   doi = "10.3389/fams.2023.1266739",
   journal = "Front. Appl. Math. Stat.",
   volume = "9",
   year = "2024"
}

@article{Chapman-Bird:2025xtd,
   author = "Chapman-Bird, Christian E. A. and others",
   title = "{The Fast and the Frame-Dragging: Efficient waveforms for asymmetric-mass eccentric equatorial inspirals into rapidly-spinning black holes}",
   eprint = "2506.09470",
   archivePrefix = "arXiv",
   primaryClass = "gr-qc",
   month = "6",
   year = "2025"
}
```

</details>
<!-- citation-end -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

<!-- support-start: this section is also included in docs/index.rst -->
- **Getting started**: Follow the [Quick Start guide](https://stableemrifisher.readthedocs.io/en/latest/quickstart.html) for a detailed walkthrough, and see the [example notebooks and scripts](https://github.com/perturber/StableEMRIFisher/tree/main/examples)
- **API**: See the [API Reference](https://stableemrifisher.readthedocs.io/en/latest/api/index.html) for complete documentation of every function and class
- **Questions**: Open a [GitHub Discussion](https://github.com/perturber/StableEMRIFisher/discussions)
- **Bug Reports**: Submit a [GitHub Issue](https://github.com/perturber/StableEMRIFisher/issues)
<!-- support-end -->

## Acknowledgments

<!-- acknowledgments-start: this section is also included in docs/citation.rst -->
StableEMRIFisher development has been supported by:
- **National University of Singapore**: IT Research and Computing Group
- **Institute for Mathematical Science, National University of Singapore**: Workshop on Mathematical Methods for the General Relativistic Two-Body Problem
- **Centre National de la Recherche Scientifique (CNRS)**: Laboratoire des 2 Infinis - Toulouse scientific computing
- **European Research Council**: ERC Consolidator Grant "NewNear" (Grant No. 864035)
- **European Research Council**: ERC Synergy Grant "GWSky" (Grant No. 101167314), funded by the European Union
- **Max Planck Institute for Gravitational Physics (Albert Einstein Institute), Potsdam**

Special thanks to the FastEMRIWaveforms development team and the LISA Consortium.
<!-- acknowledgments-end -->
