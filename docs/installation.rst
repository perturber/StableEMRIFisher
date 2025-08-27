Installation
============

StableEMRIFisher requires Python 3.9 or later and has several key dependencies for gravitational wave modeling and computation.

Requirements
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

The package requires the following dependencies that will be automatically installed:

* **NumPy**: Array computation and numerical operations
* **SciPy**: Scientific computing utilities
* **h5py**: HDF5 file format support for data storage

Required External Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You must install these packages separately before using StableEMRIFisher:

* **FastEMRIWaveforms (FEW)**: Core EMRI waveform generation
  
  .. code-block:: bash
  
     # Install from PyPI
     pip install fastemriwaveforms
     
     # Or from source
     git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
     cd FastEMRIWaveforms
     pip install .

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For enhanced functionality, you may also install:

* **CuPy**: GPU acceleration support
  
  .. code-block:: bash
  
     # For CUDA 11.x
     pip install cupy-cuda11x
     
     # For CUDA 12.x  
     pip install cupy-cuda12x

* **LISAAnalysisTools**: Additional LISA detector utilities
  
  .. code-block:: bash
  
     pip install git+https://github.com/mikekatz04/LISAanalysistools.git

* **LISA-on-gpu**: GPU-accelerated LISA response functions
  
  .. code-block:: bash
  
     pip install git+https://github.com/mikekatz04/lisa-on-gpu.git

Installing StableEMRIFisher
---------------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install stableemrifisher

From Source
~~~~~~~~~~~

For the latest development version or if you want to contribute:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/perturber/StableEMRIFisher.git
   cd StableEMRIFisher
   
   # Install in development mode
   pip install -e .

Verifying Installation
----------------------

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

Building Documentation (Optional)
----------------------------------

To build the documentation locally:

.. code-block:: bash

   # Install documentation dependencies
   pip install sphinx sphinx_rtd_theme nbsphinx
   
   # Build documentation
   cd docs
   make html
   
   # View documentation
   open _build/html/index.html  # macOS
   # or
   xdg-open _build/html/index.html  # Linux

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'few'**

This means FastEMRIWaveforms is not installed. Follow the FEW installation instructions above.

**CUDA/GPU Issues**

If you encounter GPU-related errors:

1. Ensure you have a compatible NVIDIA GPU
2. Install the correct CUDA toolkit version
3. Install the matching CuPy version
4. Set ``use_gpu=False`` to fall back to CPU computation

**Memory Issues**

For large parameter spaces or long waveforms:

1. Reduce the observation time ``T``
2. Increase the time step ``dt`` 
3. Use fewer derivative points in finite difference calculations
4. Enable GPU computation to access more memory

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/perturber/StableEMRIFisher/issues>`_
2. Review the FastEMRIWaveforms installation guide
3. Open a new issue with your error message and system details
