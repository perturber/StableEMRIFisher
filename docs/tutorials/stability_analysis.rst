Stability Analysis Tutorial
============================

Ensuring numerical stability and reliability in Fisher matrix calculations.

Understanding Numerical Stability
----------------------------------

Fisher matrix calculations involve numerical differentiation and matrix operations that can be sensitive to:

1. **Finite Difference Errors**: Step size selection for derivatives
2. **Matrix Conditioning**: Inversion of nearly singular matrices  
3. **Parameter Scaling**: Different parameter magnitudes affecting precision
4. **Waveform Evaluation**: Numerical accuracy of underlying waveform generation

Finite Difference Stability
----------------------------

Choosing Optimal Step Sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The finite difference step size (δ) must balance truncation and round-off errors:

.. code-block:: python

   import numpy as np
   from stableemrifisher.fisher import StableEMRIFisher
   import matplotlib.pyplot as plt
   
   def derivative_convergence_test(param_name='m1'):
       \"\"\"Test convergence of numerical derivatives with step size.\"\"\"
       
       base_params = {
           'm1': 1e6, 'm2': 10.0, 'a': 0.9, 'p0': 12.0, 'e0': 0.2,
           'Y0': 1.0, 'dist': 1.0, 'qS': 0.5, 'phiS': 1.0,
           'qK': 0.3, 'phiK': 2.0, 'Phi_phi0': 0.0, 
           'Phi_theta0': 0.0, 'Phi_r0': 0.0, 'dt': 10.0, 'T': 0.1
       }
       
       fisher_calc = StableEMRIFisher(waveform_model="AAKSummation")
       
       # Generate reference waveform
       h_ref = fisher_calc._generate_waveform(**base_params)
       
       # Test different step sizes
       delta_factors = np.logspace(-8, -3, 20)
       derivatives = []
       
       param_value = base_params[param_name]
       
       for delta_factor in delta_factors:
           delta = delta_factor * abs(param_value)
           
           # Forward difference
           params_plus = base_params.copy()
           params_plus[param_name] = param_value + delta
           h_plus = fisher_calc._generate_waveform(**params_plus)
           
           # Backward difference  
           params_minus = base_params.copy()
           params_minus[param_name] = param_value - delta
           h_minus = fisher_calc._generate_waveform(**params_minus)
           
           # Central difference
           dh_dp = (h_plus - h_minus) / (2 * delta)
           
           # Inner product with itself as measure
           inner_product = np.real(np.vdot(dh_dp, dh_dp))
           derivatives.append(inner_product)
       
       # Plot convergence
       plt.figure(figsize=(10, 6))
       plt.loglog(delta_factors, derivatives, 'o-')
       plt.xlabel('Relative Step Size $\\delta$')
       plt.ylabel('$||\\partial h/\\partial \\theta||^2$')
       plt.title(f'Derivative Convergence for {param_name}')
       plt.grid(True)
       
       # Find optimal step size (plateau region)
       optimal_idx = np.argmin(np.abs(np.diff(np.log(derivatives))))
       optimal_delta = delta_factors[optimal_idx]
       plt.axvline(optimal_delta, color='red', linestyle='--', 
                   label=f'Optimal δ ≈ {optimal_delta:.1e}')
       plt.legend()
       plt.show()
       
       return delta_factors, derivatives, optimal_delta

Adaptive Step Size Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def adaptive_step_size(fisher_calc, params, param_name, tolerance=1e-3):
       \"\"\"Automatically determine optimal step size for parameter.\"\"\"
       
       param_value = params[param_name]
       base_delta = 1e-6 * abs(param_value)
       
       # Test multiple step sizes
       test_deltas = base_delta * np.array([0.1, 0.5, 1.0, 2.0, 5.0])
       derivatives = []
       
       for delta in test_deltas:
           # Compute derivative with this step size
           params_plus = params.copy()
           params_plus[param_name] = param_value + delta
           
           params_minus = params.copy()  
           params_minus[param_name] = param_value - delta
           
           h_plus = fisher_calc._generate_waveform(**params_plus)
           h_minus = fisher_calc._generate_waveform(**params_minus)
           
           dh_dp = (h_plus - h_minus) / (2 * delta)
           norm = np.linalg.norm(dh_dp)
           derivatives.append(norm)
       
       # Find most stable region (minimal relative variation)
       relative_vars = []
       for i in range(1, len(derivatives)-1):
           rel_var = abs(derivatives[i+1] - derivatives[i-1]) / derivatives[i]
           relative_vars.append(rel_var)
       
       # Choose step size with minimal variation
       min_var_idx = np.argmin(relative_vars) + 1
       optimal_delta = test_deltas[min_var_idx]
       
       print(f"Parameter {param_name}:")
       print(f"  Tested step sizes: {test_deltas}")
       print(f"  Derivative norms: {derivatives}")
       print(f"  Optimal step size: {optimal_delta:.2e}")
       
       return optimal_delta

Matrix Conditioning Analysis
----------------------------

Diagnosing Ill-Conditioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_matrix_conditioning(fisher_matrix, param_names):
       \"\"\"Comprehensive analysis of Fisher matrix conditioning.\"\"\"
       
       # Basic properties
       eigenvals = np.linalg.eigvals(fisher_matrix)
       cond_num = np.linalg.cond(fisher_matrix)
       
       print("Fisher Matrix Conditioning Analysis")
       print("=" * 50)
       print(f"Matrix size: {fisher_matrix.shape}")
       print(f"Condition number: {cond_num:.2e}")
       print(f"Log condition number: {np.log10(cond_num):.1f}")
       
       # Eigenvalue analysis
       eigenvals_sorted = np.sort(eigenvals)[::-1]
       print(f"\\nEigenvalue spectrum:")
       print(f"  Largest: {eigenvals_sorted[0]:.2e}")
       print(f"  Smallest: {eigenvals_sorted[-1]:.2e}")
       print(f"  Ratio: {eigenvals_sorted[0]/eigenvals_sorted[-1]:.2e}")
       
       # Check for problematic eigenvalues
       if np.min(eigenvals) <= 0:
           print("\\n⚠️  WARNING: Non-positive eigenvalues detected!")
           negative_count = np.sum(eigenvals <= 0)
           print(f"   Number of non-positive eigenvalues: {negative_count}")
       
       if cond_num > 1e12:
           print("\\n⚠️  WARNING: Matrix is poorly conditioned!")
           print("   Consider:")
           print("   - Reducing parameter space")
           print("   - Parameter transformations")
           print("   - Increasing SNR")
       
       # Parameter contributions to conditioning
       print(f"\\nWorst-conditioned parameter combinations:")
       try:
           cov_matrix = np.linalg.inv(fisher_matrix)
           uncertainties = np.sqrt(np.diag(cov_matrix))
           
           # Sort by uncertainty (largest = worst measured)
           worst_indices = np.argsort(uncertainties)[::-1]
           
           for i in range(min(5, len(worst_indices))):
               idx = worst_indices[i]
               print(f"  {param_names[idx]}: σ = {uncertainties[idx]:.2e}")
               
       except np.linalg.LinAlgError:
           print("  Cannot compute - matrix is singular!")
       
       return eigenvals, cond_num

Regularization Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def regularize_fisher_matrix(fisher_matrix, regularization=1e-12):
       \"\"\"Apply Tikhonov regularization to improve conditioning.\"\"\"
       
       # Add small diagonal terms
       n = fisher_matrix.shape[0]
       regularized = fisher_matrix + regularization * np.eye(n)
       
       # Check improvement
       original_cond = np.linalg.cond(fisher_matrix)
       regularized_cond = np.linalg.cond(regularized)
       
       print(f"Regularization applied:")
       print(f"  Original condition number: {original_cond:.2e}")
       print(f"  Regularized condition number: {regularized_cond:.2e}")
       print(f"  Improvement factor: {original_cond/regularized_cond:.1f}")
       
       return regularized
   
   def truncated_svd_fisher(fisher_matrix, threshold=1e-12):
       \"\"\"Use SVD to remove problematic modes.\"\"\"
       
       U, s, Vt = np.linalg.svd(fisher_matrix)
       
       # Threshold small singular values
       s_regularized = np.where(s > threshold * s[0], s, threshold * s[0])
       
       # Reconstruct matrix
       fisher_regularized = U @ np.diag(s_regularized) @ Vt
       
       print(f"SVD regularization:")
       print(f"  Original singular values: {s[:5]} ...")
       print(f"  Regularized singular values: {s_regularized[:5]} ...")
       
       return fisher_regularized

Parameter Space Reduction
-------------------------

Identifying Important Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def parameter_importance_analysis(fisher_calc, base_params):
       \"\"\"Determine which parameters contribute most to Fisher matrix.\"\"\"
       
       param_names = list(base_params.keys())
       n_params = len(param_names)
       
       # Compute full Fisher matrix
       full_fisher = fisher_calc(**base_params)
       full_cond = np.linalg.cond(full_fisher)
       
       print("Parameter Importance Analysis")
       print("=" * 40)
       print(f"Full parameter space condition number: {full_cond:.2e}")
       
       # Test removing each parameter
       importance_scores = []
       
       for i, param_name in enumerate(param_names):
           # Create reduced parameter set
           reduced_params = {k: v for j, (k, v) in enumerate(base_params.items()) if j != i}
           
           try:
               # Would need modified Fisher calculation for subset
               # This is simplified - actual implementation would require
               # parameter index mapping
               reduced_indices = [j for j in range(n_params) if j != i]
               reduced_fisher = full_fisher[np.ix_(reduced_indices, reduced_indices)]
               
               reduced_cond = np.linalg.cond(reduced_fisher)
               improvement = full_cond / reduced_cond
               
               importance_scores.append(improvement)
               print(f"Without {param_name:12}: condition = {reduced_cond:.2e}, "
                     f"improvement = {improvement:.1f}x")
               
           except:
               importance_scores.append(0)
               print(f"Without {param_name:12}: failed to compute")
       
       # Rank parameters by improvement when removed
       ranked_indices = np.argsort(importance_scores)[::-1]
       
       print(f"\\nParameters ranked by conditioning impact:")
       for i, idx in enumerate(ranked_indices[:5]):
           print(f"{i+1}. {param_names[idx]} (improvement: {importance_scores[idx]:.1f}x)")
       
       return importance_scores

Validation and Cross-Checks
----------------------------

Richardson Extrapolation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def richardson_extrapolation_derivative(fisher_calc, params, param_name):
       \"\"\"Use Richardson extrapolation for higher-order derivative estimates.\"\"\"
       
       param_value = params[param_name]
       h = 1e-6 * abs(param_value)  # Base step size
       
       # Compute derivatives at h and h/2
       def central_difference(step_size):
           params_plus = params.copy()
           params_plus[param_name] = param_value + step_size
           
           params_minus = params.copy()
           params_minus[param_name] = param_value - step_size
           
           h_plus = fisher_calc._generate_waveform(**params_plus)
           h_minus = fisher_calc._generate_waveform(**params_minus)
           
           return (h_plus - h_minus) / (2 * step_size)
       
       # First and second order estimates
       D1 = central_difference(h)
       D2 = central_difference(h/2)
       
       # Richardson extrapolation: D_improved = D2 + (D2 - D1)/3
       D_richardson = D2 + (D2 - D1) / 3
       
       # Compare inner products
       norm_D1 = np.linalg.norm(D1)
       norm_D2 = np.linalg.norm(D2)
       norm_richardson = np.linalg.norm(D_richardson)
       
       print(f"Richardson extrapolation for {param_name}:")
       print(f"  D(h):   ||dh/dp|| = {norm_D1:.6e}")
       print(f"  D(h/2): ||dh/dp|| = {norm_D2:.6e}")
       print(f"  Richardson: ||dh/dp|| = {norm_richardson:.6e}")
       print(f"  Convergence: {abs(norm_D2 - norm_D1)/norm_D1:.2e}")
       
       return D_richardson

Monte Carlo Error Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def monte_carlo_fisher_uncertainty(fisher_calc, params, n_trials=10):
       \"\"\"Estimate Fisher matrix uncertainty using Monte Carlo sampling.\"\"\"
       
       fisher_matrices = []
       
       for trial in range(n_trials):
           # Add small random perturbations to parameters
           perturbed_params = params.copy()
           
           for key, value in params.items():
               if key in ['dt', 'T']:  # Don't perturb time parameters
                   continue
               noise_level = 1e-8 * abs(value)
               perturbed_params[key] = value + np.random.normal(0, noise_level)
           
           try:
               fisher_trial = fisher_calc(**perturbed_params)
               fisher_matrices.append(fisher_trial)
           except:
               print(f"Trial {trial} failed")
               continue
       
       if len(fisher_matrices) > 1:
           fisher_matrices = np.array(fisher_matrices)
           
           # Compute statistics
           mean_fisher = np.mean(fisher_matrices, axis=0)
           std_fisher = np.std(fisher_matrices, axis=0)
           
           print(f"Monte Carlo Fisher Matrix Analysis ({len(fisher_matrices)} trials):")
           print(f"  Mean condition number: {np.linalg.cond(mean_fisher):.2e}")
           print(f"  Std of diagonal elements: {np.mean(std_fisher.diagonal()):.2e}")
           
           return mean_fisher, std_fisher
       else:
           print("Insufficient successful trials for statistics")
           return None, None

Best Practices Summary
----------------------

Numerical Stability Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Step Size Validation**:
   - Test derivative convergence with varying step sizes
   - Use adaptive step size selection when possible
   - Verify Richardson extrapolation convergence

2. **Matrix Conditioning**:
   - Monitor condition numbers (should be < 1e12)
   - Check for positive eigenvalues
   - Apply regularization if needed

3. **Parameter Space**:
   - Start with well-measured parameter subsets
   - Use parameter transformations to reduce correlations
   - Remove poorly constrained parameters if necessary

4. **Cross-Validation**:
   - Compare with analytical derivatives when available
   - Use Monte Carlo sampling to estimate uncertainties
   - Validate against MCMC results for known cases

.. code-block:: python

   def stability_check_pipeline(fisher_calc, params):
       \"\"\"Complete stability analysis pipeline.\"\"\"
       
       print("=== FISHER MATRIX STABILITY ANALYSIS ===\\n")
       
       # 1. Compute Fisher matrix
       print("1. Computing Fisher matrix...")
       fisher_matrix = fisher_calc(**params)
       
       # 2. Basic conditioning check
       print("\\n2. Matrix conditioning analysis:")
       eigenvals, cond_num = analyze_matrix_conditioning(fisher_matrix, list(params.keys()))
       
       # 3. Derivative convergence tests
       print("\\n3. Testing derivative convergence:")
       for param in ['m1', 'm2', 'dist']:  # Test key parameters
           try:
               optimal_delta = adaptive_step_size(fisher_calc, params, param)
           except:
               print(f"   {param}: convergence test failed")
       
       # 4. Parameter importance
       print("\\n4. Parameter importance analysis:")
       importance_scores = parameter_importance_analysis(fisher_calc, params)
       
       # 5. Final recommendations
       print("\\n5. Recommendations:")
       if cond_num > 1e12:
           print("   ⚠️  Consider parameter space reduction")
       if np.min(eigenvals) <= 0:
           print("   ⚠️  Matrix has negative eigenvalues - check derivatives")
       if cond_num < 1e10:
           print("   ✅ Matrix appears numerically stable")
       
       return fisher_matrix

This concludes the stability analysis tutorial. Continue with :doc:`gpu_acceleration` for performance optimization.
