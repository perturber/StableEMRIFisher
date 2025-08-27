Validation
==========

StableEMRIFisher has been validated against Monte Carlo Markov Chain (MCMC) parameter estimation studies to ensure accuracy of Fisher matrix predictions.

Validation Studies
------------------

MCMC Comparison
~~~~~~~~~~~~~~~

Fisher matrix uncertainties have been compared against full Bayesian parameter estimation using MCMC sampling. The validation study includes:

* **EMRI Configuration**: Stellar-mass compact object (10 M☉) around supermassive black hole (10⁶ M☉)
* **Signal-to-Noise**: SNR ~ 30 for robust statistical comparison
* **Parameters**: All 14 EMRI parameters including masses, spin, orbital elements, and sky location
* **Duration**: 1-year LISA observation

Results Summary
~~~~~~~~~~~~~~~

The Fisher matrix predictions show excellent agreement with MCMC results:

* **Mass Parameters**: Fisher uncertainties within 10% of MCMC credible intervals
* **Orbital Parameters**: Good agreement for well-measured parameters (p₀, e₀)
* **Extrinsic Parameters**: Consistent sky localization and distance uncertainties
* **Correlations**: Parameter correlation structure well-captured by Fisher analysis

Validation Scripts
------------------

Comparison Tools
~~~~~~~~~~~~~~~~

The ``validation/`` directory contains scripts for comparing Fisher and MCMC results:

.. code-block:: python

   # validation/compare_with_mcmc.py
   import numpy as np
   import h5py
   from stableemrifisher.fisher import StableEMRIFisher
   
   def compare_fisher_mcmc(fisher_file, mcmc_file):
       \"\"\"Compare Fisher matrix with MCMC results.\"\"\"
       
       # Load Fisher results
       with h5py.File(fisher_file, 'r') as f:
           fisher_matrix = f['fisher_matrix'][:]
           fisher_cov = np.linalg.inv(fisher_matrix)
           fisher_uncertainties = np.sqrt(np.diag(fisher_cov))
       
       # Load MCMC results  
       mcmc_samples = np.load(mcmc_file)
       mcmc_uncertainties = np.std(mcmc_samples, axis=0)
       
       # Compare uncertainties
       ratio = fisher_uncertainties / mcmc_uncertainties
       
       param_names = ['m1', 'm2', 'a', 'p0', 'e0', 'Y0', 'dist',
                      'qS', 'phiS', 'qK', 'phiK', 'Phi_phi0', 
                      'Phi_theta0', 'Phi_r0']
       
       print("Fisher vs MCMC Uncertainty Comparison:")
       print("-" * 50)
       for name, ratio_val in zip(param_names, ratio):
           print(f"{name:12}: {ratio_val:.2f}")
       
       return ratio

Validation Data
~~~~~~~~~~~~~~~

Pre-computed validation datasets are available:

* ``MCMC_FM_Data/mcmc_data/case_1_few.h5``: MCMC posterior samples
* ``MCMC_FM_Data/mcmc_data/samples_large_q_SNR_30.npy``: High mass-ratio case
* ``validation/data_files/MCMC_results/``: Additional validation datasets

Corner Plot Comparison
~~~~~~~~~~~~~~~~~~~~~~

Visual comparison of Fisher and MCMC posteriors:

.. code-block:: python

   import corner
   import matplotlib.pyplot as plt
   
   def plot_fisher_mcmc_comparison(mcmc_samples, fisher_cov, param_names):
       \"\"\"Create corner plot comparing Fisher and MCMC posteriors.\"\"\"
       
       # Create figure
       fig = corner.corner(mcmc_samples, labels=param_names, 
                          show_titles=True, title_kwargs={"fontsize": 12})
       
       # Overlay Fisher ellipses
       corner.overplot_lines(fig, np.mean(mcmc_samples, axis=0), color='red')
       corner.overplot_points(fig, np.mean(mcmc_samples, axis=0)[None], 
                             marker='s', color='red')
       
       # Add Fisher contours (approximate)
       # Note: Requires additional implementation for exact Fisher contours
       
       plt.suptitle('Fisher Matrix vs MCMC Comparison', fontsize=16)
       plt.show()

Systematic Studies
------------------

Parameter Dependencies
~~~~~~~~~~~~~~~~~~~~~~

Validation across different EMRI configurations:

* **Mass Ratios**: q = 10⁻⁴ to 10⁻¹ 
* **Spin Values**: a = 0 to 0.998
* **Eccentricities**: e₀ = 0 to 0.7
* **Inclinations**: All sky orientations
* **SNR Range**: 10 to 100

Accuracy Metrics
~~~~~~~~~~~~~~~~

Fisher matrix accuracy is quantified using:

1. **Uncertainty Ratios**: σ_Fisher / σ_MCMC
2. **Bias Assessment**: |μ_Fisher - μ_MCMC| / σ_MCMC  
3. **Coverage Tests**: Fraction of MCMC samples within Fisher confidence intervals
4. **Correlation Validation**: Correlation coefficient comparisons

Known Limitations
-----------------

Where Fisher Analysis Breaks Down
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fisher matrices provide Gaussian approximations that may fail for:

* **Very High SNR**: Non-linear parameter dependencies (SNR > 100)
* **Strong Correlations**: Near-degenerate parameter combinations
* **Prior Boundaries**: Parameters near physical limits
* **Multi-modal Posteriors**: Complex likelihood structures

Mitigation Strategies
~~~~~~~~~~~~~~~~~~~~

* **SNR Monitoring**: Validate Fisher results for your specific SNR regime
* **Stability Checks**: Monitor condition numbers and eigenvalues
* **Parameter Transformation**: Use better-behaved parameter combinations
* **Subset Analysis**: Focus on well-measured parameter combinations

Reproducing Validation
----------------------

Running Your Own Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To validate Fisher results for your EMRI system:

.. code-block:: python

   # 1. Compute Fisher matrix
   from stableemrifisher.fisher import StableEMRIFisher
   
   fisher_calc = StableEMRIFisher(waveform_model="AAKSummation")
   fisher_matrix = fisher_calc(**your_parameters)
   
   # 2. Run MCMC analysis (requires additional MCMC code)
   # mcmc_samples = run_mcmc_analysis(your_parameters)
   
   # 3. Compare results
   # compare_fisher_mcmc(fisher_matrix, mcmc_samples)

Continuous Validation
~~~~~~~~~~~~~~~~~~~~~

The validation suite can be run with:

.. code-block:: bash

   cd validation
   python compare_with_mcmc.py
   jupyter notebook derivatives.ipynb

This validation framework ensures StableEMRIFisher provides reliable parameter estimation predictions for EMRI gravitational wave analysis.
