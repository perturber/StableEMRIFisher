Basic Usage Tutorial
====================

This tutorial covers the fundamental concepts of Fisher matrix analysis for EMRI parameter estimation using StableEMRIFisher.

What is a Fisher Matrix?
------------------------

The Fisher Information Matrix (FIM) is a key tool in parameter estimation theory. For gravitational wave analysis, it provides:

1. **Parameter Uncertainties**: Diagonal elements give variance estimates
2. **Parameter Correlations**: Off-diagonal elements show parameter degeneracies  
3. **Detectability**: The trace relates to signal-to-noise ratio
4. **Network Analysis**: Combining multiple detectors

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

For a gravitational wave signal :math:`h(\\theta)` with parameters :math:`\\theta_i`, the Fisher matrix is:

.. math::

   \\Gamma_{ij} = \\left( \\frac{\\partial h}{\\partial \\theta_i} \\bigg| \\frac{\\partial h}{\\partial \\theta_j} \\right)

where :math:`(a|b)` is the noise-weighted inner product:

.. math::

   (a|b) = 4 \\text{Re} \\int_0^{\\infty} \\frac{\\tilde{a}(f)^* \\tilde{b}(f)}{S_n(f)} df

Setting Up Your First Calculation
----------------------------------

Import Required Modules
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from stableemrifisher.fisher import StableEMRIFisher
   from stableemrifisher.noise import lisa_psd

Create the Fisher Calculator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize with default settings
   fisher_calc = StableEMRIFisher(
       waveform_model="AAKSummation",  # EMRI waveform model
       use_gpu=False,                  # Start with CPU
       dt_factor=1.0,                  # Time resolution factor
       delta_factor=1e-6               # Finite difference step
   )

Define Your EMRI System
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Typical stellar-mass object around supermassive black hole
   params = {
       'm1': 1e6,        # SMBH mass: 1 million solar masses
       'm2': 10.0,       # Compact object: 10 solar masses
       'a': 0.9,         # High SMBH spin
       'p0': 12.0,       # Initial separation: 12 M
       'e0': 0.2,        # Moderate eccentricity
       'Y0': 1.0,        # cos(inclination) = 1 (edge-on)
       'dist': 1.0,      # Distance: 1 Gpc
       'qS': 0.5,        # Sky location (polar)
       'phiS': 1.0,      # Sky location (azimuthal) 
       'qK': 0.3,        # Spin direction (polar)
       'phiK': 2.0,      # Spin direction (azimuthal)
       'Phi_phi0': 0.0,  # Initial orbital phase
       'Phi_theta0': 0.0, # Initial polar phase
       'Phi_r0': 0.0,    # Initial radial phase
       'dt': 10.0,       # Time step: 10 seconds
       'T': 1.0          # Observation: 1 year
   }

Compute the Fisher Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # This may take a few minutes for the first calculation
   print("Computing Fisher matrix...")
   fisher_matrix = fisher_calc(**params)
   
   print(f"Fisher matrix shape: {fisher_matrix.shape}")
   print(f"Matrix condition number: {np.linalg.cond(fisher_matrix):.2e}")

Understanding the Results
-------------------------

Parameter Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Parameter names in order
   param_names = ['m1', 'm2', 'a', 'p0', 'e0', 'Y0', 'dist', 
                  'qS', 'phiS', 'qK', 'phiK', 'Phi_phi0', 
                  'Phi_theta0', 'Phi_r0']
   
   # Compute covariance matrix
   try:
       cov_matrix = np.linalg.inv(fisher_matrix)
       uncertainties = np.sqrt(np.diag(cov_matrix))
       
       print("\\nParameter uncertainties (1-sigma):")
       print("-" * 40)
       for name, param_val, uncertainty in zip(param_names, params.values(), uncertainties):
           fractional = uncertainty / abs(param_val) if param_val != 0 else float('inf')
           print(f"{name:12}: {uncertainty:.2e} ({fractional:.1%})")
           
   except np.linalg.LinAlgError:
       print("ERROR: Fisher matrix is singular!")

Signal-to-Noise Ratio
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate the optimal matched-filter SNR
   snr = fisher_calc.SNRcalc_SEF(**params)
   print(f"\\nSignal-to-noise ratio: {snr:.1f}")
   
   # Rule of thumb: SNR > 8 needed for reliable detection
   if snr > 8:
       print("✓ Signal is detectable by LISA")
   else:
       print("✗ Signal may be too weak for detection")

Visualizing Results
-------------------

Correlation Matrix
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute correlation matrix
   correlations = np.zeros_like(cov_matrix)
   for i in range(len(uncertainties)):
       for j in range(len(uncertainties)):
           correlations[i,j] = cov_matrix[i,j] / (uncertainties[i] * uncertainties[j])
   
   # Plot correlations
   fig, ax = plt.subplots(figsize=(12, 10))
   im = ax.imshow(correlations, cmap='RdBu_r', vmin=-1, vmax=1)
   
   # Labels and formatting
   ax.set_xticks(range(len(param_names)))
   ax.set_yticks(range(len(param_names)))
   ax.set_xticklabels(param_names, rotation=45, ha='right')
   ax.set_yticklabels(param_names)
   
   # Add correlation values
   for i in range(len(param_names)):
       for j in range(len(param_names)):
           text = ax.text(j, i, f'{correlations[i,j]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)
   
   plt.colorbar(im, ax=ax, label='Correlation Coefficient')
   plt.title('Parameter Correlation Matrix')
   plt.tight_layout()
   plt.show()

Error Ellipses
~~~~~~~~~~~~~~

.. code-block:: python

   # 2D error ellipse for mass parameters
   from matplotlib.patches import Ellipse
   from scipy.stats import chi2
   
   # Extract mass parameter covariance submatrix
   mass_indices = [0, 1]  # m1, m2 indices
   mass_cov = cov_matrix[np.ix_(mass_indices, mass_indices)]
   
   # Eigenvalues and eigenvectors for ellipse orientation
   eigenvals, eigenvecs = np.linalg.eigh(mass_cov)
   angle = np.degrees(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]))
   
   # Confidence levels
   confidence_levels = [0.68, 0.95]  # 1-sigma, 2-sigma
   colors = ['blue', 'red']
   
   fig, ax = plt.subplots(figsize=(8, 6))
   
   for conf, color in zip(confidence_levels, colors):
       # Chi-squared scaling for confidence ellipse
       scale = chi2.ppf(conf, df=2)
       width = 2 * np.sqrt(scale * eigenvals[0])
       height = 2 * np.sqrt(scale * eigenvals[1])
       
       ellipse = Ellipse(
           xy=(params['m1'], params['m2']),
           width=width, height=height, angle=angle,
           facecolor='none', edgecolor=color, linewidth=2,
           label=f'{conf:.0%} confidence'
       )
       ax.add_patch(ellipse)
   
   # Mark true values
   ax.plot(params['m1'], params['m2'], 'ko', markersize=8, label='True values')
   
   ax.set_xlabel('Primary Mass $m_1$ ($M_\\odot$)')
   ax.set_ylabel('Secondary Mass $m_2$ ($M_\\odot$)')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.title('Mass Parameter Error Ellipses')
   plt.show()

Exploring Parameter Space
-------------------------

Mass Ratio Effects
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Study how mass ratio affects uncertainties
   mass_ratios = np.logspace(-3, -1, 10)  # q = m2/m1 from 0.001 to 0.1
   snrs = []
   mass_uncertainties = []
   
   for q in mass_ratios:
       # Keep total mass fixed, vary ratio
       params_q = params.copy()
       params_q['m2'] = q * params_q['m1']
       
       try:
           fisher_q = fisher_calc(**params_q)
           cov_q = np.linalg.inv(fisher_q)
           
           snr_q = fisher_calc.SNRcalc_SEF(**params_q)
           sigma_m1 = np.sqrt(cov_q[0, 0])
           
           snrs.append(snr_q)
           mass_uncertainties.append(sigma_m1 / params_q['m1'])
           
       except:
           snrs.append(np.nan)
           mass_uncertainties.append(np.nan)
   
   # Plot results
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   
   ax1.loglog(mass_ratios, snrs, 'o-')
   ax1.set_xlabel('Mass Ratio $q = m_2/m_1$')
   ax1.set_ylabel('SNR')
   ax1.grid(True)
   ax1.set_title('SNR vs Mass Ratio')
   
   ax2.loglog(mass_ratios, mass_uncertainties, 'o-', color='red')
   ax2.set_xlabel('Mass Ratio $q = m_2/m_1$')
   ax2.set_ylabel('Fractional $m_1$ Uncertainty')
   ax2.grid(True)
   ax2.set_title('Mass Uncertainty vs Mass Ratio')
   
   plt.tight_layout()
   plt.show()

Best Practices
--------------

Numerical Stability
~~~~~~~~~~~~~~~~~~~

1. **Check Condition Number**: Fisher matrices with condition numbers > 1e12 may be unreliable
2. **Monitor Eigenvalues**: All eigenvalues should be positive
3. **Validate Derivatives**: Ensure finite differences are converged

.. code-block:: python

   def check_fisher_stability(fisher_matrix):
       \"\"\"Check numerical properties of Fisher matrix.\"\"\"
       eigenvals = np.linalg.eigvals(fisher_matrix)
       cond_num = np.linalg.cond(fisher_matrix)
       
       print(f"Eigenvalue range: [{np.min(eigenvals):.2e}, {np.max(eigenvals):.2e}]")
       print(f"Condition number: {cond_num:.2e}")
       
       if np.min(eigenvals) <= 0:
           print("⚠️  WARNING: Non-positive eigenvalues detected!")
       if cond_num > 1e12:
           print("⚠️  WARNING: Matrix is poorly conditioned!")
       if np.min(eigenvals) > 0 and cond_num < 1e10:
           print("✓ Fisher matrix appears numerically stable")
   
   check_fisher_stability(fisher_matrix)

Parameter Selection
~~~~~~~~~~~~~~~~~~~

Start with well-measured parameters and gradually add more:

.. code-block:: python

   # Start with just masses and distance
   essential_params = ['m1', 'm2', 'dist']
   
   # Add orbital parameters
   orbital_params = essential_params + ['a', 'p0', 'e0']
   
   # Include all extrinsic parameters
   all_params = list(params.keys())
   
   print("Parameter sets to try:")
   print(f"Essential: {essential_params}")
   print(f"Orbital: {orbital_params}")
   print(f"Complete: {all_params}")

This concludes the basic usage tutorial. Continue with :doc:`parameter_estimation` for more advanced analysis techniques.
