StableEMRIFisher Documentation
==============================

**Stable EMRI Fisher Matrix Calculator**

StableEMRIFisher (SEF) is a Python package for computing stable Fisher matrices 
using numerical derivatives for Extreme Mass Ratio Inspiral (EMRI) waveform models 
in the FastEMRIWaveforms (FEW) package.

The Fisher matrix is a fundamental tool in gravitational wave parameter estimation, providing estimates of parameter uncertainties and correlations. 
This package implements stable numerical derivative methods to compute Fisher matrices 
for EMRI systems.

**Stable Numerical Derivatives**
A gravitational waveform from an EMRI system at infinity is given by

.. math::

   h(t;\boldsymbol{\theta}) = h_+ - ih_\times = \sum_{l,m,n,k} A_{lmnk}(t;\boldsymbol{\theta}) \exp(-i\Phi_{mnk}(t;\boldsymbol{\theta}))

Here $A_{lmnk}$ are the slowly varying amplitudes and $\Phi_{mnk}$ are the slowly evolving phases.
Parameter derivatives $\partial_{i} = \partial/\partial\theta^i$ are given by

.. math::

   \begin{align}
   \partial_{i}h(t;\boldsymbol{\theta}) &= \sum_{lmnk} (\partial_{i}A_{lmnk}(t;\boldsymbol{\theta}) -i A_{lmnk}\partial_{i}\Phi_{mnk})\exp(-i\Phi_{mnk}(t;\boldsymbol{\theta})) \\
   \partial_{i}h(t;\boldsymbol{\theta}) &= \sum_{lmnk} A_{lmnk}^{\prime}\exp(-i\Phi_{mnk}(t;\boldsymbol{\theta}))
   \end{align}

with effective amplitudes $A_{lmnk}^{\prime} = \partial_{i}A_{lmnk}(t;\boldsymbol{\theta}) -i A_{lmnk}\partial_{i}\Phi_{mnk}$. 

These effective amplitudes are constructed using finite differences and then splined. The oscillatory waveform is then built using the 
effective amplitudes $A^{\prime}_{lmnk}$ and the original phases $\Phi_{mnk}$. This method avoids direct finite differencing of the oscillatory waveform,
resulting in more stable numerical derivatives. 

.. note::
   This package requires the latest (v2.0.0) FastEMRIWaveforms (FEW) package to be installed. 
   Installing StableEMRIFisher will install FastEMRIWaveforms by default (for both CPU and GPU).
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
   contribution
   citation


Key Features
------------

* **Stable Numerical Derivatives**: Robust finite difference methods for parameter derivatives
* **GPU/CPU Support**: Efficient computation on both CPU (NumPy) and GPU (CuPy) backends
* **EMRI Waveforms**: Integration with FastEMRIWaveforms for accurate EMRI modeling
* **Fisher Matrix Analysis**: Complete parameter estimation uncertainty analysis
* **LISA Noise Models**: Built-in support for LISA detector noise characteristics
* **Validation Tools**: Comparison utilities against MCMC parameter estimation

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
