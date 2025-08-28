Quick Start Guide
=================

The key features of the `StableEMRIFisher` package is twofold 

- **Robust Numerical Derivatives:** Cheap and Robust EMRI waveform derivatives 
- **Fisher Matrix Computations** Fisher matrix calculations for accelerated parameter inference. 

This quick-start guide will walk you through computing numerical derivatives and ultimately Fisher matrices.

Numerical Derivatives of Waveforms
----------------------------------

Here's a complete example of computing a Fisher matrix for an EMRI system:

.. code-block:: python

   import numpy as np
   from stableemrifisher.fisher import StableEMRIFisher
   
   # Initialize the Fisher matrix calculator
   fisher_calc = StableEMRIFisher(
       waveform_model="AAKSummation",  # EMRI waveform model
       use_gpu=True,                   # Use GPU if available
       dt_factor=1.0,                  # Time step scaling factor
       delta_factor=1e-6               # Finite difference step size
   )
   
   # Define EMRI system parameters
   parameters = {
       'm1': 1e6,           # Primary mass (solar masses)
       'm2': 10.0,          # Secondary mass (solar masses) 
       'a': 0.9,            # Dimensionless spin parameter
       'p0': 12.0,          # Initial orbital separation (M)
       'e0': 0.2,           # Initial eccentricity
       'Y0': 1.0,           # Initial inclination cosine
       'dist': 1.0,         # Luminosity distance (Gpc)
       'qS': 0.5,           # Sky location θ parameter
       'phiS': 1.0,         # Sky location φ parameter
       'qK': 0.3,           # Spin direction θ parameter
       'phiK': 2.0,         # Spin direction φ parameter
       'Phi_phi0': 0.0,     # Initial orbital phase
       'Phi_theta0': 0.0,   # Initial polar angle phase
       'Phi_r0': 0.0,       # Initial radial phase
       'dt': 10.0,          # Time step (seconds)
       'T': 1.0             # Observation time (years)
   }
   
   # Compute the Fisher matrix
   fisher_matrix = fisher_calc(**parameters)
   
   print(f"Fisher matrix shape: {fisher_matrix.shape}")
   print(f"Condition number: {np.linalg.cond(fisher_matrix):.2e}")

Parameter Uncertainties
-----------------------

Extract parameter uncertainties from the Fisher matrix:

.. code-block:: python

   # Compute covariance matrix (inverse of Fisher matrix)
   try:
       covariance = np.linalg.inv(fisher_matrix)
       
       # Parameter uncertainties (1-sigma)
       uncertainties = np.sqrt(np.diag(covariance))
       
       # Parameter names in order
       param_names = ['m1', 'm2', 'a', 'p0', 'e0', 'Y0', 'dist', 
                      'qS', 'phiS', 'qK', 'phiK', 'Phi_phi0', 
                      'Phi_theta0', 'Phi_r0']
       
       print("Parameter uncertainties (1-sigma):")
       for name, uncertainty in zip(param_names, uncertainties):
           print(f"{name}: {uncertainty:.2e}")
           
   except np.linalg.LinAlgError:
       print("Fisher matrix is singular - cannot compute uncertainties")

Correlation Matrix
------------------
