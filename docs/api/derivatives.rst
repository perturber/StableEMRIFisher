Derivatives Module
==================

Waveform derivatives. Every derivative class derives from ``DerivativeBase``, builds the waveform model it differentiates (including the LISA response, if any), and returns derivatives of the detector waveform with one row per channel. ``StableEMRIFisher`` selects the class with ``deriv_type``: ``"stable"`` for ``StableDerivative`` and ``"direct"`` for ``DirectDerivative``, or any ``DerivativeBase`` subclass.

Base class
----------

.. automodule:: stableemrifisher.derivatives.base
   :members:
   :special-members: __init__, __call__

Stable derivative
-----------------

.. autoclass:: stableemrifisher.derivatives.stable.StableDerivative
   :members:
   :show-inheritance:

Direct derivative
-----------------

.. automodule:: stableemrifisher.derivatives.direct
   :members:
   :show-inheritance:

Finite differences
------------------

.. automodule:: stableemrifisher.derivatives.finite_difference
   :members:

Selecting a derivative
----------------------

.. autofunction:: stableemrifisher.derivatives.get_derivative_class
