Analysis Module
===============

Noise-weighted inner products, SNRs and Fisher matrices. The inner product is computed in two stages: ``fourier_coefficients`` Fourier transforms a time series, and ``inner_product_fd`` combines two sets of coefficients. ``inner_product`` runs both; when the same time series enters several inner products, as in a Fisher matrix, its coefficients can be computed once and reused.

.. automodule:: stableemrifisher.utils.analysis
   :members:
