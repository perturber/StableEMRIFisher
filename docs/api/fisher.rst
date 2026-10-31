Fisher Matrix Module
====================

The Fisher matrix computation: ``StableEMRIFisher``, which runs the pipeline, and the helpers it uses to choose finite-difference step sizes.

StableEMRIFisher
----------------

.. autoclass:: stableemrifisher.fisher.StableEMRIFisher
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

Step-size search
----------------

.. automodule:: stableemrifisher.fisher.stepsize
   :members:

Defaults
--------

Default parameters, step-size grids and stencil bounds used when the user gives none.

.. automodule:: stableemrifisher.fisher.defaults
   :members:
