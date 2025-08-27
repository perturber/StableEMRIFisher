Custom Waveforms Tutorial
=========================

Integrating custom EMRI waveform models with StableEMRIFisher.

Introduction to Custom Waveforms
---------------------------------

While StableEMRIFisher comes with support for standard FastEMRIWaveforms models, you may want to integrate custom waveform models for specialized research.

Supported Waveform Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom waveforms must implement the FastEMRIWaveforms interface:

.. code-block:: python

   class CustomEMRIWaveform:
       def __init__(self, **kwargs):
           # Initialize your waveform generator
           pass
       
       def __call__(self, m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, 
                    Phi_phi0, Phi_theta0, Phi_r0, dt, T, **kwargs):
           # Generate waveform given parameters
           # Returns: (h_plus, h_cross, time_array, frequency_array)
           pass

Integrating Custom Models
-------------------------

Method 1: Direct Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from stableemrifisher.fisher import StableEMRIFisher
   
   class MyCustomWaveform:
       \"\"\"Example custom EMRI waveform generator.\"\"\"
       
       def __init__(self, use_gpu=False, **kwargs):
           self.use_gpu = use_gpu
           # Initialize your custom waveform model
           
       def __call__(self, m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
                    Phi_phi0, Phi_theta0, Phi_r0, dt, T, **kwargs):
           \"\"\"Generate custom EMRI waveform.\"\"\"
           
           # Your custom waveform generation code here
           # This is a placeholder - replace with actual implementation
           
           import numpy as np
           
           # Time array
           t_max = T * 365.25 * 24 * 3600  # Convert years to seconds
           time_array = np.arange(0, t_max, dt)
           
           # Frequency evolution (example - replace with your model)
           f_initial = 1e-4  # Initial frequency
           f_final = 1e-2    # Final frequency
           frequency_array = np.linspace(f_initial, f_final, len(time_array))
           
           # Waveform strain (example - replace with your model)
           phase = 2 * np.pi * np.cumsum(frequency_array) * dt
           amplitude = dist * 1e-21  # Scale with distance
           
           h_plus = amplitude * np.cos(phase)
           h_cross = amplitude * np.sin(phase)
           
           return h_plus, h_cross, time_array, frequency_array
   
   # Use custom waveform with StableEMRIFisher
   def create_custom_fisher_calculator():
       \"\"\"Create Fisher calculator with custom waveform.\"\"\"
       
       # You would need to modify StableEMRIFisher to accept custom waveforms
       # This is a conceptual example
       
       custom_waveform = MyCustomWaveform(use_gpu=False)
       
       # In practice, you might need to subclass StableEMRIFisher
       class CustomFisherCalculator(StableEMRIFisher):
           def __init__(self, custom_waveform_func, **kwargs):
               super().__init__(**kwargs)
               self.waveform_generator = custom_waveform_func
       
       return CustomFisherCalculator(custom_waveform)

Method 2: Wrapper Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a wrapper that translates between your waveform and the expected interface:

.. code-block:: python

   class WaveformWrapper:
       \"\"\"Wrapper to adapt custom waveforms to StableEMRIFisher interface.\"\"\"
       
       def __init__(self, custom_waveform_class, **init_kwargs):
           self.waveform = custom_waveform_class(**init_kwargs)
           
       def generate_waveform(self, params):
           \"\"\"Generate waveform from parameter dictionary.\"\"\"
           
           # Extract parameters
           m1 = params['m1']
           m2 = params['m2']
           a = params['a']
           # ... extract all other parameters
           
           # Call custom waveform
           h_plus, h_cross, t_array, f_array = self.waveform(**params)
           
           # Convert to expected format if necessary
           return h_plus, h_cross, t_array, f_array
   
   # Usage example
   custom_wrapper = WaveformWrapper(MyCustomWaveform, use_gpu=False)

Analytical Waveform Models
--------------------------

Post-Newtonian Waveforms
~~~~~~~~~~~~~~~~~~~~~~~~

Example implementation of a simple PN-inspired EMRI waveform:

.. code-block:: python

   class PostNewtonianEMRI:
       \"\"\"Simple post-Newtonian EMRI waveform for testing.\"\"\"
       
       def __init__(self, pn_order=2, use_gpu=False):
           self.pn_order = pn_order
           self.use_gpu = use_gpu
           
           if use_gpu:
               import cupy as cp
               self.xp = cp
           else:
               import numpy as np
               self.xp = np
       
       def __call__(self, m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
                    Phi_phi0, Phi_theta0, Phi_r0, dt, T, **kwargs):
           \"\"\"Generate PN EMRI waveform.\"\"\"
           
           # Physical constants
           G = 6.67430e-11  # m^3 kg^-1 s^-2
           c = 299792458    # m/s
           M_sun = 1.98847e30  # kg
           
           # Convert to SI units
           M1 = m1 * M_sun
           M2 = m2 * M_sun
           M_total = M1 + M2
           mu = M1 * M2 / M_total
           eta = mu / M_total
           
           # Time array
           t_max = T * 365.25 * 24 * 3600
           time_array = self.xp.arange(0, t_max, dt)
           
           # Frequency evolution (PN approximation)
           # This is highly simplified - real EMRI evolution is much more complex
           f_orbit_initial = 1.0 / (2 * self.xp.pi * (G * M_total / c**3)**(1/2) * p0**(3/2))
           
           # Approximate frequency evolution
           beta = (96 * self.xp.pi / 5) * eta * (G * M_total / c**3)**(5/3)
           f_orbit = f_orbit_initial * (1 + beta * f_orbit_initial**(8/3) * time_array)**(-3/8)
           
           # Waveform strain
           r_distance = dist * 3.086e22  # Convert Gpc to meters
           amplitude = (G * mu / c**2) * (G * M_total / (c**3 * r_distance))
           
           phase = 2 * self.xp.pi * self.xp.cumsum(f_orbit) * dt
           
           # Include angular factors (simplified)
           angular_factor = self.xp.sin(2 * qS) * self.xp.cos(2 * phiS)
           
           h_plus = amplitude * angular_factor * self.xp.cos(phase)
           h_cross = amplitude * angular_factor * self.xp.sin(phase)
           
           return h_plus, h_cross, time_array, f_orbit

Numerical Waveform Integration
------------------------------

For waveforms requiring numerical integration:

.. code-block:: python

   class NumericalEMRIWaveform:
       \"\"\"EMRI waveform with numerical orbital evolution.\"\"\"
       
       def __init__(self, use_gpu=False, **kwargs):
           self.use_gpu = use_gpu
           
           if use_gpu:
               import cupy as cp
               self.xp = cp
           else:
               import numpy as np
               self.xp = np
       
       def orbital_evolution(self, m1, m2, a, p0, e0, t_array):
           \"\"\"Solve orbital evolution equations numerically.\"\"\"
           
           from scipy.integrate import solve_ivp
           
           def orbital_derivatives(t, y):
               \"\"\"Derivatives for orbital evolution.\"\"\"
               p, e, phase = y
               
               # Simplified radiation reaction (placeholder)
               # Real EMRI evolution requires sophisticated treatment
               dpdt = -1e-10 * p  # Orbital decay
               dedt = -1e-11 * e  # Eccentricity decay
               dphasedt = 2 * np.pi / (p**(3/2))  # Orbital frequency
               
               return [dpdt, dedt, dphasedt]
           
           # Initial conditions
           y0 = [p0, e0, 0.0]
           
           # Solve evolution
           sol = solve_ivp(orbital_derivatives, [t_array[0], t_array[-1]], y0,
                          t_eval=t_array, method='DOP853', rtol=1e-10)
           
           return sol.y[0], sol.y[1], sol.y[2]  # p(t), e(t), phase(t)
       
       def __call__(self, m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
                    Phi_phi0, Phi_theta0, Phi_r0, dt, T, **kwargs):
           \"\"\"Generate numerically evolved EMRI waveform.\"\"\"
           
           # Time array
           t_max = T * 365.25 * 24 * 3600
           time_array = np.arange(0, t_max, dt)
           
           # Solve orbital evolution
           p_t, e_t, phase_t = self.orbital_evolution(m1, m2, a, p0, e0, time_array)
           
           # Convert to GPU if needed
           if self.use_gpu:
               time_array = self.xp.asarray(time_array)
               p_t = self.xp.asarray(p_t)
               e_t = self.xp.asarray(e_t)
               phase_t = self.xp.asarray(phase_t)
           
           # Compute waveform from orbital elements
           # This is a simplified example
           r_distance = dist * 3.086e22
           amplitude = 1e-21 / r_distance
           
           h_plus = amplitude * self.xp.cos(phase_t)
           h_cross = amplitude * self.xp.sin(phase_t)
           
           # Frequency from orbital motion
           frequency_array = self.xp.gradient(phase_t) / (2 * self.xp.pi * dt)
           
           return h_plus, h_cross, time_array, frequency_array

Validation and Testing
----------------------

Testing Custom Waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def validate_custom_waveform(waveform_func, test_params):
       \"\"\"Validate custom waveform implementation.\"\"\"
       
       # Test basic functionality
       try:
           h_plus, h_cross, t_array, f_array = waveform_func(**test_params)
           print("✓ Waveform generation successful")
       except Exception as e:
           print(f"✗ Waveform generation failed: {e}")
           return False
       
       # Check output formats
       if len(h_plus) != len(t_array):
           print("✗ h_plus length mismatch")
           return False
       
       if len(h_cross) != len(t_array):
           print("✗ h_cross length mismatch")
           return False
       
       # Check for NaN or infinite values
       if np.any(~np.isfinite(h_plus)):
           print("✗ h_plus contains NaN or inf")
           return False
       
       if np.any(~np.isfinite(h_cross)):
           print("✗ h_cross contains NaN or inf")
           return False
       
       # Check frequency evolution
       if len(f_array) > 1:
           if not np.all(np.diff(f_array) >= 0):
               print("⚠ Frequency not monotonically increasing")
       
       print("✓ Custom waveform validation passed")
       return True

Comparison with Standard Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compare_waveforms(custom_waveform, standard_model, params):
       \"\"\"Compare custom waveform with standard FEW model.\"\"\"
       
       # Generate custom waveform
       h_plus_custom, h_cross_custom, t_custom, f_custom = custom_waveform(**params)
       
       # Generate standard waveform (conceptual - requires FEW setup)
       # h_plus_std, h_cross_std, t_std, f_std = standard_model(**params)
       
       # Compute overlap or other comparison metrics
       # This would require proper implementation of overlap integrals
       
       print("Waveform comparison completed")

Performance Considerations
--------------------------

Optimization Tips
~~~~~~~~~~~~~~~~~

1. **Vectorization**: Use array operations instead of loops
2. **Memory Management**: Pre-allocate arrays when possible
3. **GPU Utilization**: Ensure computations stay on GPU if using CuPy
4. **Caching**: Cache expensive computations like orbital evolution
5. **Approximations**: Use appropriate approximations for your accuracy needs

.. code-block:: python

   class OptimizedCustomWaveform:
       \"\"\"Example of optimized custom waveform.\"\"\"
       
       def __init__(self, use_gpu=False, cache_size=100):
           self.use_gpu = use_gpu
           self.cache = {}
           self.cache_size = cache_size
           
           if use_gpu:
               import cupy as cp
               self.xp = cp
           else:
               import numpy as np
               self.xp = np
       
       def _cache_key(self, params):
           \"\"\"Generate cache key from parameters.\"\"\"
           return hash(tuple(sorted(params.items())))
       
       def __call__(self, **params):
           \"\"\"Cached waveform generation.\"\"\"
           
           cache_key = self._cache_key(params)
           
           if cache_key in self.cache:
               return self.cache[cache_key]
           
           # Generate waveform
           result = self._generate_waveform(**params)
           
           # Cache result
           if len(self.cache) < self.cache_size:
               self.cache[cache_key] = result
           
           return result

This tutorial provides a framework for integrating custom EMRI waveforms. For production use, ensure thorough validation against known models and parameter estimation studies.
