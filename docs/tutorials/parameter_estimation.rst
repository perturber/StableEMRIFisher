Parameter Estimation Tutorial
=============================

Advanced techniques for EMRI parameter estimation using Fisher matrices.

Understanding Parameter Uncertainties
--------------------------------------

The Fisher Information Matrix provides the Cramér-Rao lower bound on parameter estimation uncertainties. For a gravitational wave signal with parameters θ, the Fisher matrix is:

.. math::

   \\Gamma_{ij} = \\left( \\frac{\\partial h}{\\partial \\theta_i} \\bigg| \\frac{\\partial h}{\\partial \\theta_j} \\right)

where the inner product is defined as:

.. math::

   (a|b) = 4 \\text{Re} \\int_0^{\\infty} \\frac{\\tilde{a}(f)^* \\tilde{b}(f)}{S_n(f)} df

Parameter Covariance and Correlations
--------------------------------------

The parameter covariance matrix is the inverse of the Fisher matrix:

.. code-block:: python

   import numpy as np
   from stableemrifisher.fisher import StableEMRIFisher
   
   # Compute Fisher matrix
   fisher_calc = StableEMRIFisher(waveform_model="AAKSummation")
   fisher_matrix = fisher_calc(**parameters)
   
   # Extract covariance matrix
   covariance = np.linalg.inv(fisher_matrix)
   
   # Parameter uncertainties (1-sigma)
   uncertainties = np.sqrt(np.diag(covariance))
   
   # Correlation coefficients
   correlations = np.zeros_like(covariance)
   for i in range(len(uncertainties)):
       for j in range(len(uncertainties)):
           correlations[i,j] = covariance[i,j] / (uncertainties[i] * uncertainties[j])

Confidence Regions
------------------

Multi-dimensional confidence regions follow chi-squared distributions:

.. code-block:: python

   from scipy.stats import chi2
   import matplotlib.pyplot as plt
   from matplotlib.patches import Ellipse
   
   def plot_confidence_ellipse(covariance_2d, center, confidence=0.68, ax=None):
       \"\"\"Plot confidence ellipse for 2D parameter space.\"\"\"
       
       if ax is None:
           fig, ax = plt.subplots()
       
       # Eigendecomposition for ellipse orientation
       eigenvals, eigenvecs = np.linalg.eigh(covariance_2d)
       angle = np.degrees(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]))
       
       # Scale factor for confidence level
       scale = chi2.ppf(confidence, df=2)
       width = 2 * np.sqrt(scale * eigenvals[0])
       height = 2 * np.sqrt(scale * eigenvals[1])
       
       # Create ellipse
       ellipse = Ellipse(xy=center, width=width, height=height, 
                        angle=angle, facecolor='none', 
                        edgecolor='blue', linewidth=2)
       ax.add_patch(ellipse)
       
       return ax
   
   # Example: Mass parameter confidence ellipse
   mass_cov = covariance[:2, :2]  # m1, m2 covariance
   mass_center = [parameters['m1'], parameters['m2']]
   
   fig, ax = plt.subplots(figsize=(8, 6))
   plot_confidence_ellipse(mass_cov, mass_center, confidence=0.68, ax=ax)
   plot_confidence_ellipse(mass_cov, mass_center, confidence=0.95, ax=ax)
   
   ax.plot(parameters['m1'], parameters['m2'], 'ro', markersize=8)
   ax.set_xlabel('Primary Mass $m_1$ ($M_\\odot$)')
   ax.set_ylabel('Secondary Mass $m_2$ ($M_\\odot$)')
   plt.show()

Parameter Transformations
-------------------------

Some parameter combinations are better measured than others. Consider transformations that reduce correlations:

Chirp Mass and Mass Ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def transform_to_chirp_mass(m1, m2):
       \"\"\"Transform individual masses to chirp mass and mass ratio.\"\"\"
       total_mass = m1 + m2
       reduced_mass = m1 * m2 / total_mass
       chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
       mass_ratio = m2 / m1
       
       return chirp_mass, mass_ratio
   
   def chirp_mass_jacobian(m1, m2):
       \"\"\"Jacobian matrix for chirp mass transformation.\"\"\"
       total = m1 + m2
       
       # dMc/dm1 and dMc/dm2
       dMc_dm1 = (3/5) * (m2/m1)**(3/5) * (total)**(-1/5) - (1/5) * (m1*m2)**(3/5) * total**(-6/5)
       dMc_dm2 = (3/5) * (m1/m2)**(3/5) * (total)**(-1/5) - (1/5) * (m1*m2)**(3/5) * total**(-6/5)
       
       # dq/dm1 and dq/dm2  
       dq_dm1 = -m2 / m1**2
       dq_dm2 = 1 / m1
       
       jacobian = np.array([[dMc_dm1, dMc_dm2],
                           [dq_dm1, dq_dm2]])
       return jacobian
   
   # Transform covariance to chirp mass space
   J = chirp_mass_jacobian(parameters['m1'], parameters['m2'])
   mass_cov_original = covariance[:2, :2]
   mass_cov_chirp = J @ mass_cov_original @ J.T
   
   print(f"Original mass correlation: {correlations[0,1]:.3f}")
   chirp_corr = mass_cov_chirp[0,1] / np.sqrt(mass_cov_chirp[0,0] * mass_cov_chirp[1,1])
   print(f"Chirp mass-ratio correlation: {chirp_corr:.3f}")

Detectability Analysis
----------------------

Signal-to-Noise Ratio
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def snr_scaling_analysis():
       \"\"\"Study how SNR scales with EMRI parameters.\"\"\"
       
       base_params = {
           'm1': 1e6, 'm2': 10.0, 'a': 0.9, 'p0': 12.0, 'e0': 0.2,
           'Y0': 1.0, 'dist': 1.0, 'qS': 0.5, 'phiS': 1.0,
           'qK': 0.3, 'phiK': 2.0, 'Phi_phi0': 0.0, 
           'Phi_theta0': 0.0, 'Phi_r0': 0.0, 'dt': 10.0, 'T': 1.0
       }
       
       # Vary distance
       distances = np.logspace(-1, 1, 20)  # 0.1 to 10 Gpc
       snrs = []
       
       for dist in distances:
           params = base_params.copy()
           params['dist'] = dist
           
           snr = fisher_calc.SNRcalc_SEF(**params)
           snrs.append(snr)
       
       plt.figure(figsize=(8, 6))
       plt.loglog(distances, snrs, 'o-')
       plt.axhline(y=8, color='r', linestyle='--', label='Detection threshold')
       plt.xlabel('Distance (Gpc)')
       plt.ylabel('SNR')
       plt.legend()
       plt.grid(True)
       plt.title('EMRI Detectability vs Distance')
       plt.show()
       
       return distances, snrs

Network Analysis
----------------

Multiple Detector Combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For detector networks, Fisher matrices add:

.. code-block:: python

   def multi_detector_fisher():
       \"\"\"Combine Fisher matrices from multiple detectors.\"\"\"
       
       # LISA constellation (simplified)
       detectors = ['LISA_A', 'LISA_E', 'LISA_T']
       
       total_fisher = np.zeros((14, 14))
       
       for detector in detectors:
           # In practice, each detector would have different noise curves
           # and response functions
           fisher_detector = fisher_calc(**parameters)  # Simplified
           total_fisher += fisher_detector
       
       # Combined uncertainties are better than single detector
       combined_cov = np.linalg.inv(total_fisher)
       combined_uncertainties = np.sqrt(np.diag(combined_cov))
       
       single_cov = np.linalg.inv(fisher_matrix)
       single_uncertainties = np.sqrt(np.diag(single_cov))
       
       improvement = single_uncertainties / combined_uncertainties
       
       print("Uncertainty improvement from detector network:")
       param_names = ['m1', 'm2', 'a', 'p0', 'e0', 'Y0', 'dist',
                      'qS', 'phiS', 'qK', 'phiK', 'Phi_phi0', 
                      'Phi_theta0', 'Phi_r0']
       
       for name, imp in zip(param_names, improvement):
           print(f"{name}: {imp:.1f}x better")

Systematic Studies
------------------

Parameter Space Exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def parameter_space_study():
       \"\"\"Study Fisher matrix properties across parameter space.\"\"\"
       
       # Vary mass ratio
       mass_ratios = np.logspace(-4, -1, 20)
       condition_numbers = []
       snrs = []
       
       for q in mass_ratios:
           params = parameters.copy()
           params['m2'] = q * params['m1']
           
           try:
               fisher_q = fisher_calc(**params)
               cond_num = np.linalg.cond(fisher_q)
               snr = fisher_calc.SNRcalc_SEF(**params)
               
               condition_numbers.append(cond_num)
               snrs.append(snr)
           except:
               condition_numbers.append(np.nan)
               snrs.append(np.nan)
       
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
       
       ax1.loglog(mass_ratios, snrs, 'o-')
       ax1.set_xlabel('Mass Ratio $q = m_2/m_1$')
       ax1.set_ylabel('SNR')
       ax1.grid(True)
       ax1.set_title('SNR vs Mass Ratio')
       
       ax2.loglog(mass_ratios, condition_numbers, 'o-', color='red')
       ax2.axhline(y=1e12, color='k', linestyle='--', 
                   label='Poor conditioning threshold')
       ax2.set_xlabel('Mass Ratio $q = m_2/m_1$')
       ax2.set_ylabel('Condition Number')
       ax2.legend()
       ax2.grid(True)
       ax2.set_title('Matrix Conditioning vs Mass Ratio')
       
       plt.tight_layout()
       plt.show()

Advanced Analysis Techniques
----------------------------

Principal Component Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identify the best-measured parameter combinations:

.. code-block:: python

   def fisher_pca_analysis(fisher_matrix, param_names):
       \"\"\"Principal component analysis of Fisher matrix.\"\"\"
       
       # Eigendecomposition
       eigenvals, eigenvecs = np.linalg.eigh(fisher_matrix)
       
       # Sort by eigenvalue (largest first)
       idx = np.argsort(eigenvals)[::-1]
       eigenvals = eigenvals[idx]
       eigenvecs = eigenvecs[:, idx]
       
       print("Principal Components (best to worst measured):")
       print("-" * 60)
       
       for i in range(len(eigenvals)):
           print(f"PC {i+1}: eigenvalue = {eigenvals[i]:.2e}")
           
           # Show parameter contributions
           contributions = eigenvecs[:, i]
           sorted_idx = np.argsort(np.abs(contributions))[::-1]
           
           print("  Main contributors:")
           for j in sorted_idx[:5]:  # Top 5 contributors
               if np.abs(contributions[j]) > 0.1:
                   print(f"    {param_names[j]}: {contributions[j]:+.3f}")
           print()
       
       return eigenvals, eigenvecs

This concludes the parameter estimation tutorial. See :doc:`stability_analysis` for numerical considerations.
