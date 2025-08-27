Quick Start Guide
=================

This guide will walk you through your first Fisher matrix calculation with StableEMRIFisher.

Basic Fisher Matrix Calculation
--------------------------------

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

Examine parameter correlations:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Compute correlation matrix
   correlation = np.zeros_like(covariance)
   for i in range(len(uncertainties)):
       for j in range(len(uncertainties)):
           correlation[i, j] = covariance[i, j] / (uncertainties[i] * uncertainties[j])
   
   # Plot correlation matrix
   fig, ax = plt.subplots(figsize=(10, 8))
   im = ax.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1)
   
   # Add parameter labels
   ax.set_xticks(range(len(param_names)))
   ax.set_yticks(range(len(param_names))) 
   ax.set_xticklabels(param_names, rotation=45)
   ax.set_yticklabels(param_names)
   
   # Add colorbar
   plt.colorbar(im, ax=ax, label='Correlation')
   plt.title('Parameter Correlation Matrix')
   plt.tight_layout()
   plt.show()

Signal-to-Noise Ratio
----------------------

Calculate the matched-filter SNR:

.. code-block:: python

   # Compute SNR using the built-in method
   snr = fisher_calc.SNRcalc_SEF(**parameters)
   print(f"Signal-to-noise ratio: {snr:.1f}")
   
   # SNR is also related to the Fisher matrix trace
   fisher_trace = np.trace(fisher_matrix)
   print(f"Fisher matrix trace: {fisher_trace:.2e}")

Stability Check
---------------

Verify the numerical stability of your calculation:

.. code-block:: python

   # Check Fisher matrix properties
   eigenvalues = np.linalg.eigvals(fisher_matrix)
   condition_number = np.linalg.cond(fisher_matrix)
   
   print(f"Minimum eigenvalue: {np.min(eigenvalues):.2e}")
   print(f"Maximum eigenvalue: {np.max(eigenvalues):.2e}")
   print(f"Condition number: {condition_number:.2e}")
   
   # Good Fisher matrices should have:
   # - All positive eigenvalues
   # - Reasonable condition number (< 1e12)
   
   if np.min(eigenvalues) <= 0:
       print("WARNING: Fisher matrix has non-positive eigenvalues!")
   
   if condition_number > 1e12:
       print("WARNING: Fisher matrix is poorly conditioned!")

Different Waveform Models
-------------------------

Try different EMRI waveform models:

.. code-block:: python

   # Available models in FastEMRIWaveforms
   models = ["AAKSummation", "Pn5AAKWaveform"]
   
   for model in models:
       print(f"\\nTesting {model}:")
       
       fisher_calc_model = StableEMRIFisher(
           waveform_model=model,
           use_gpu=True
       )
       
       try:
           fisher_matrix = fisher_calc_model(**parameters)
           snr = fisher_calc_model.SNRcalc_SEF(**parameters)
           print(f"  SNR: {snr:.1f}")
           print(f"  Condition number: {np.linalg.cond(fisher_matrix):.2e}")
       except Exception as e:
           print(f"  Error: {e}")

Next Steps
----------

Now that you've computed your first Fisher matrix:

1. **Explore Parameter Space**: Try different EMRI configurations
2. **Study Tutorials**: Work through the detailed :doc:`tutorials/index`
3. **Read the API**: Browse the complete :doc:`api/index` reference
4. **Run Examples**: Check out more :doc:`examples`
5. **Validate Results**: Compare with :doc:`validation` studies

Tips for Success
-----------------

* **Start Small**: Begin with shorter observation times (T ~ 0.1-1 year)
* **Check Stability**: Always verify Fisher matrix conditioning
* **Use GPU**: Enable GPU acceleration for faster computation
* **Tune Parameters**: Adjust ``delta_factor`` if derivatives seem unstable
* **Save Results**: Store Fisher matrices in HDF5 format for later analysis

.. code-block:: python

   # Save Fisher matrix for later use
   import h5py
   
   with h5py.File('fisher_results.h5', 'w') as f:
       f.create_dataset('fisher_matrix', data=fisher_matrix)
       f.create_dataset('parameters', data=list(parameters.values()))
       f.attrs['param_names'] = param_names
       f.attrs['snr'] = snr
