StableEMRIFisher Documentation
==============================

**Stable EMRI Fisher Matrix Calculator**

StableEMRIFisher (SEF) is a Python package for computing stable Fisher matrices using numerical derivatives for Extreme Mass Ratio Inspiral (EMRI) waveform models in the FastEMRIWaveforms (FEW) package.

The Fisher matrix is a fundamental tool in gravitational wave parameter estimation, providing estimates of parameter uncertainties and correlations. This package implements stable numerical derivative methods to compute Fisher matrices for EMRI systems, which are key sources for space-based gravitational wave detectors like LISA.

.. note::
   This package requires the FastEMRIWaveforms (FEW) package to be installed separately. 
   See the installation guide for details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   tutorials/index
   api/index
   examples
   validation
   citation

Key Features
------------

* **Stable Numerical Derivatives**: Robust finite difference methods for parameter derivatives
* **GPU/CPU Support**: Efficient computation on both CPU (NumPy) and GPU (CuPy) backends
* **EMRI Waveforms**: Integration with FastEMRIWaveforms for accurate EMRI modeling
* **Fisher Matrix Analysis**: Complete parameter estimation uncertainty analysis
* **LISA Noise Models**: Built-in support for LISA detector noise characteristics
* **Validation Tools**: Comparison utilities against MCMC parameter estimation

Quick Start
-----------

Once installed, you can compute a Fisher matrix for an EMRI system:

.. code-block:: python

   from stableemrifisher.fisher import StableEMRIFisher
   
   # Initialize the Fisher matrix calculator
   fisher_calc = StableEMRIFisher(
       waveform_model="AAKSummation",
       use_gpu=True
   )
   
   # Compute Fisher matrix for specific parameters
   fisher_matrix = fisher_calc(
       m1=1e6,      # Primary mass (solar masses)
       m2=10.0,     # Secondary mass (solar masses)
       a=0.9,       # Spin parameter
       p0=12.0,     # Initial separation
       e0=0.2,      # Initial eccentricity
       Y0=1.0,      # Initial inclination angle
       dist=1.0,    # Luminosity distance (Gpc)
       qS=0.5,      # Sky location parameter
       phiS=1.0,    # Sky location parameter
       qK=0.3,      # Spin direction parameter
       phiK=2.0,    # Spin direction parameter
       Phi_phi0=0.0,    # Initial orbital phase
       Phi_theta0=0.0,  # Initial polar angle
       Phi_r0=0.0,      # Initial radial phase
       dt=10.0,     # Time step (seconds)
       T=1.0        # Observation time (years)
   )

Getting Help
------------

* Check the :doc:`quickstart` guide for a detailed walkthrough
* Browse the :doc:`tutorials/index` for in-depth examples
* Refer to the :doc:`api/index` for complete API documentation
* See :doc:`examples` for ready-to-run code samples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
