Installation
============

StableEMRIFisher requires Python 3.10 or later and has several key 
dependencies. The first main dependency is **FastEMRIWaveforms (FEW)**, 
which is the current state-of-the-art framework for generating 
computationally efficient and accurate EMRI waveform models suitable for LISA 
data analysis.

Quick Installation with UV (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`UV <https://github.com/astral-sh/uv>`_ is a fast Python package manager that provides better dependency resolution and faster installs:

**Installation:**

.. code-block:: bash

   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Clone the repository
   git clone https://github.com/perturber/StableEMRIFisher.git
   cd StableEMRIFisher

   # Create virtual environment and install dependencies
   uv venv
   uv sync --dev

   # For GPU support (Linux x86_64 only)
   uv sync --extra cuda12x --dev

**Alternative manual installation with UV:**

.. code-block:: bash

   # Create virtual environment
   uv venv

   # Install in development mode (CPU only)
   uv pip install -e ".[docs,dev]"

   # Or install with GPU support (Linux x86_64 only)
   uv pip install -e ".[cuda12x,docs,dev]"

**Common UV development commands:**

.. code-block:: bash

   # Run tests
   uv run pytest

   # Format and lint code
   uv run black .
   uv run pylint stableemrifisher/

   # Build documentation
   cd docs && uv run make html

   # Add dependencies
   uv add numpy          # adds to main dependencies
   uv add --dev pytest   # adds to dev dependencies

   # Update dependencies
   uv lock --upgrade

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

StableEMRIFisher works on both CPUs and GPUs. If installing directly from source, you can use either UV (recommended) or pip:

**With UV (recommended):**

.. code-block:: bash

   git clone https://github.com/perturber/StableEMRIFisher.git
   cd StableEMRIFisher
   
   # CPU version
   uv pip install -e ".[docs,dev]"
   
   # GPU version (Linux x86_64 only)
   uv pip install -e ".[cuda12x,docs,dev]"

**With pip:**

.. code-block:: bash

   git clone https://github.com/perturber/StableEMRIFisher.git
   cd StableEMRIFisher
   # Install CPU version 
   pip install -e ".[cpu]"
   # Install with CUDA 11.x support if available
   pip install -e ".[cuda11x]"  
   # Install with CUDA 12.x support if available
   pip install -e ".[cuda12x]"

.. note::
   This package requires the latest (v2.0.0) FastEMRIWaveforms (FEW) package to be installed. 
   Installing StableEMRIFisher will install FastEMRIWaveforms by default (for both CPU and GPU).

StableEMRIFFisher will automatically install the following dependencies:

* **NumPy**: Array computation and numerical operations
* **SciPy**: Scientific computing utilities
* **h5py**: HDF5 file format support for data storage
* **Cython**: C-Extensions for python. Essential if the user wants to incorporate the LISA response.
* **matplotlib**: To help assess stability of numerical derivatives
* **setuptools**: To help build `lisa-on-gpu` and `LISAAnalysisTools` from source. 


For full development, documentation and GPU support, we recommend installing all of the dependencies:

**With UV:**

.. code-block:: bash

   # Install with CUDA 12.x support if available
   uv pip install -e ".[dev,docs,cuda12x]"

**With pip:**

.. code-block:: bash

   # Install with CUDA 12.x support if available
   pip install -e ".[dev,docs,cuda12x]"


Building Documentation Locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build the documentation, make sure that the documentation dependencies are installed.

**With UV:**

.. code-block:: bash

   cd docs
   uv run make clean
   uv run make html
   # View documentation
   open _build/html/index.html  # macOS
   # or
   xdg-open _build/html/index.html  # Linux

**With traditional tools:**

.. code-block:: bash

   cd docs
   make clean
   make html
   # View documentation
   open _build/html/index.html  # macOS
   # or
   xdg-open _build/html/index.html  # Linux

Within `docs/_build/html` you can find the generated documentation. The file `index.html` can be opened in a web browser. 


Response Function
~~~~~~~~~~~~~~~~~~

The package `StableEMRIFisher` can optionally include the LISA response function to compute the Fisher matrix in the LISA detector frame. This requires two additional packages to be installed from source:

* **LISAAnalysisTools**: To interface with the LISA response function and generate LISA-based Power Spectral Densities that describe the instrumental noise

  .. code-block:: bash
  
     git clone https://github.com/mikekatz04/LISAanalysistools.git
     python scripts/prebuild.py
     python setup.py install

* **LISA-on-gpu**: GPU-accelerated time-domain LISA response function
  
  .. code-block:: bash
  
     git clone https://github.com/mikekatz04/lisa-on-gpu.git
     python scripts/prebuild.py
     python setup.py install

Verifying Installation
~~~~~~~~~~~~~~~~~~~~~~

Test your installation by running:

.. code-block:: bash

   # With UV
   uv run python -c "
   import stableemrifisher
   from stableemrifisher.fisher import StableEMRIFisher
   
   # Check if GPU support is available
   try:
       import cupy as cp
       print(f'GPU support available: {cp.cuda.is_available()}')
   except ImportError:
       print('GPU support not available (CuPy not installed)')
   
   # Check FEW installation
   try:
       import few
       print('FastEMRIWaveforms successfully imported')
   except ImportError:
       print('ERROR: FastEMRIWaveforms not found - please install FEW')
   "

   # Or with traditional Python
   python -c "
   import stableemrifisher
   from stableemrifisher.fisher import StableEMRIFisher
   
   # Check if GPU support is available
   try:
       import cupy as cp
       print(f'GPU support available: {cp.cuda.is_available()}')
   except ImportError:
       print('GPU support not available (CuPy not installed)')
   
   # Check FEW installation
   try:
       import few
       print('FastEMRIWaveforms successfully imported')
   except ImportError:
       print('ERROR: FastEMRIWaveforms not found - please install FEW')
   "

   
