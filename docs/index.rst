SEF: StableEMRIFisher
==============================

.. image:: _static/Default.png
   :align: center
   :alt: Default
   :width: 400px

Stable EMRI Fisher Matrix Calculator
------------------------------------

..
   Sections pulled in with ``.. include:: ../README.md`` come from README.md at build
   time, so the README is the single source for them. Edit them there.

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- overview-start: this section is also included in docs/index.rst -->
   :end-before: <!-- overview-end -->

The Fisher information matrix has elements: 

.. math::

   \Gamma_{ij} := \Bigg(\partial_i h(t;\boldsymbol{\theta})\Bigg|\partial_j h(t;\boldsymbol{\theta})\Bigg)
   
where 

.. math::

   \partial_i h(t;\boldsymbol{\theta}) := \frac{\partial h(t;\boldsymbol{\theta})}{\partial \theta^i}.
   
We calculate this derivative for a given parameter :math:`\theta^i` using a stable numerical derivative scheme.

Stable Numerical Derivatives
----------------------------

A gravitational waveform from an EMRI system at infinity is given by

.. math::

   h(t;\boldsymbol{\theta}) = h_+ - ih_\times = \sum_{l,m,n,k} A_{lmnk}(t;\boldsymbol{\theta}) \exp(-i\Phi_{mnk}(t;\boldsymbol{\theta}))

Here :math:`A_{lmnk}` are the slowly varying amplitudes and :math:`\Phi_{mnk}` are the slowly evolving phases.
Parameter derivatives are given by

.. math::

   \partial_{i}h(t;\boldsymbol{\theta}) &= \sum_{lmnk} (\partial_{i}A_{lmnk} -i A_{lmnk}\partial_{i}\Phi_{mnk})\exp(-i\Phi_{mnk}(t;\boldsymbol{\theta})) \\
   \partial_{i}h(t;\boldsymbol{\theta}) &= \sum_{lmnk} A_{lmnk}^{\prime}\exp(-i\Phi_{mnk}(t;\boldsymbol{\theta}))

with effective amplitudes :math:`A_{lmnk}^{\prime} = \partial_{i}A_{lmnk}(t;\boldsymbol{\theta}) -i A_{lmnk}\partial_{i}\Phi_{mnk}`.

These effective amplitudes are constructed using finite differences and then splined. The oscillatory waveform is then built using the 
effective amplitudes :math:`A^{\prime}_{lmnk}` and the original phases :math:`\Phi_{mnk}`. This method avoids direct finite differencing of the oscillatory waveform,
resulting in more stable numerical derivatives. 

.. note::
   This package requires FastEMRIWaveforms (FEW) v2.0.0 or later.
   Installing StableEMRIFisher will install FastEMRIWaveforms by default (for both CPU and GPU).
   See the :doc:`installation` guide for details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index
   installation
   quickstart
   validation/validation
   contribution
   citation


Key Features
------------

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- features-start: this section is also included in docs/index.rst -->
   :end-before: <!-- features-end -->

Getting Help
------------

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- support-start: this section is also included in docs/index.rst -->
   :end-before: <!-- support-end -->

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
