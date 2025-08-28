Installation
============

StableEMRIFisher requires Python 3.9 or later and has several key 
dependencies. The first main dependency is **FastEMRIWaveforms (FEW)**, 
which is the current state-of-the-art framework for generating 
computationally efficient and accurate EMRI waveform models suitable for LISA 
data analysis. 

Pip Installation on PyPi
~~~~~~~~~~~~~~~~~~~~~~~

You can install StableEMRIFisher directly from PyPi using pip:

.. code-block:: bash

   pip install stableemrifisher

**Note:** This will automatically install FastEMRIWaveforms version 2.0.0 and required 
dependencies.

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

Our package StableEMRIFisher is both CPU and GPU agnostic. If installing directly from source, you can do so by cloning the repository and running the 
following commands. 

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

For full development, documentation and GPU support, we recommend installing all of the dependencies 
.. code-block:: bash

   # Install with CUDA 12.x support if available
   pip install -e ".[dev, docs, cuda12x]"

Our package `StableEMRIFFisher` will automatically install the following dependencies

* **NumPy**: Array computation and numerical operations
* **SciPy**: Scientific computing utilities
* **h5py**: HDF5 file format support for data storage
* **cython**: C-Extensions for python. Essential if the user wants to incorporate the LISA response.
* **matplotlib**: To help assess stability of numerical derivatives
* **setuptools**: TO help build `lisa-on-gpu` and `LISAAnalysisTools` from source. 

Building Documentation Locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To build the documentation, make sure that the documentation dependencies are installed.

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

.. code-block:: python

   import stableemrifisher
   from stableemrifisher.fisher import StableEMRIFisher
   
   # Check if GPU support is available
   try:
       import cupy as cp
       print(f"GPU support available: {cp.cuda.is_available()}")
   except ImportError:
       print("GPU support not available (CuPy not installed)")
   
   # Check FEW installation
   try:
       import few
       print("FastEMRIWaveforms successfully imported")
   except ImportError:
       print("ERROR: FastEMRIWaveforms not found - please install FEW")

   