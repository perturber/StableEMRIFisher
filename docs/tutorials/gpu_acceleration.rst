GPU Acceleration Tutorial
=========================

Leveraging GPU computation for faster Fisher matrix calculations.

Introduction to GPU Acceleration
---------------------------------

StableEMRIFisher supports GPU acceleration through CuPy, providing significant speedups for large parameter spaces and long observation times.

Performance Benefits
~~~~~~~~~~~~~~~~~~~~

Typical speedup factors:

* **Small problems** (T < 0.1 year): 2-5x speedup
* **Medium problems** (T ~ 1 year): 5-15x speedup  
* **Large problems** (T > 2 years): 10-50x speedup

Memory advantages:

* **GPU Memory**: Access to 8-80 GB depending on GPU
* **Parallel Processing**: Thousands of cores vs. CPU dozens
* **Memory Bandwidth**: ~1 TB/s vs. ~100 GB/s for CPU

Setting Up GPU Acceleration
----------------------------

Installation Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install CuPy for your CUDA version
   pip install cupy-cuda11x  # For CUDA 11.x
   # or
   pip install cupy-cuda12x  # For CUDA 12.x
   
   # Verify GPU availability
   python -c "import cupy as cp; print(f'GPU available: {cp.cuda.is_available()}')"

Basic GPU Usage
~~~~~~~~~~~~~~~

.. code-block:: python

   from stableemrifisher.fisher import StableEMRIFisher
   
   # Enable GPU acceleration
   fisher_calc = StableEMRIFisher(
       waveform_model="AAKSummation",
       use_gpu=True  # Enable GPU
   )
   
   # Check GPU status
   try:
       import cupy as cp
       print(f"GPU Device: {cp.cuda.Device().id}")
       print(f"GPU Memory: {cp.get_default_memory_pool().used_bytes() / 1e9:.1f} GB")
   except ImportError:
       print("CuPy not available - falling back to CPU")

Performance Optimization
------------------------

Memory Management
~~~~~~~~~~~~~~~~~

GPU memory is limited and requires careful management:

.. code-block:: python

   import cupy as cp
   
   def gpu_memory_info():
       """Print current GPU memory usage."""
       mempool = cp.get_default_memory_pool()
       print(f"GPU Memory Used: {mempool.used_bytes() / 1e9:.2f} GB")
       print(f"GPU Memory Total: {mempool.total_bytes() / 1e9:.2f} GB")
       
       # Free unused memory
       mempool.free_all_blocks()
   
   # Monitor memory usage
   gpu_memory_info()
   
   # Compute Fisher matrix
   fisher_matrix = fisher_calc(**params)
   
   # Check memory after computation
   gpu_memory_info()

Batch Processing
~~~~~~~~~~~~~~~~

For parameter space studies, process in batches to avoid memory overflow:

.. code-block:: python

   def batch_fisher_calculation(param_grid, batch_size=10):
       """Compute Fisher matrices in batches to manage GPU memory."""
       
       fisher_calc = StableEMRIFisher(waveform_model="AAKSummation", use_gpu=True)
       results = []
       
       for i in range(0, len(param_grid), batch_size):
           batch = param_grid[i:i+batch_size]
           batch_results = []
           
           for params in batch:
               try:
                   fisher = fisher_calc(**params)
                   snr = fisher_calc.SNRcalc_SEF(**params)
                   
                   batch_results.append({
                       'params': params,
                       'fisher': fisher,
                       'snr': snr
                   })
                   
               except Exception as e:
                   print(f"Batch item failed: {e}")
                   continue
           
           results.extend(batch_results)
           
           # Free GPU memory between batches
           if 'cp' in globals():
               cp.get_default_memory_pool().free_all_blocks()
       
       return results

Performance Benchmarking
------------------------

CPU vs GPU Comparison
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   
   def benchmark_cpu_vs_gpu(params, num_runs=3):
       """Compare CPU and GPU performance."""
       
       # CPU benchmark
       fisher_cpu = StableEMRIFisher(waveform_model="AAKSummation", use_gpu=False)
       
       cpu_times = []
       for run in range(num_runs):
           start_time = time.time()
           fisher_matrix_cpu = fisher_cpu(**params)
           cpu_time = time.time() - start_time
           cpu_times.append(cpu_time)
       
       # GPU benchmark
       fisher_gpu = StableEMRIFisher(waveform_model="AAKSummation", use_gpu=True)
       
       gpu_times = []
       for run in range(num_runs):
           start_time = time.time()
           fisher_matrix_gpu = fisher_gpu(**params)
           gpu_time = time.time() - start_time
           gpu_times.append(gpu_time)
       
       # Results
       avg_cpu_time = np.mean(cpu_times)
       avg_gpu_time = np.mean(gpu_times)
       speedup = avg_cpu_time / avg_gpu_time
       
       print(f"CPU Time: {avg_cpu_time:.2f} ± {np.std(cpu_times):.2f} seconds")
       print(f"GPU Time: {avg_gpu_time:.2f} ± {np.std(gpu_times):.2f} seconds")
       print(f"Speedup: {speedup:.1f}x")
       
       return speedup

Scaling Studies
~~~~~~~~~~~~~~~

.. code-block:: python

   def scaling_study():
       """Study GPU performance scaling with problem size."""
       
       # Different observation times
       observation_times = [0.1, 0.5, 1.0, 2.0, 4.0]
       
       results = {
           'T': observation_times,
           'cpu_time': [],
           'gpu_time': [],
           'speedup': []
       }
       
       base_params = {
           'm1': 1e6, 'm2': 10.0, 'a': 0.9, 'p0': 12.0, 'e0': 0.2,
           'Y0': 1.0, 'dist': 1.0, 'qS': 0.5, 'phiS': 1.0,
           'qK': 0.3, 'phiK': 2.0, 'Phi_phi0': 0.0,
           'Phi_theta0': 0.0, 'Phi_r0': 0.0, 'dt': 10.0
       }
       
       for T in observation_times:
           params = base_params.copy()
           params['T'] = T
           
           speedup = benchmark_cpu_vs_gpu(params, num_runs=1)
           results['speedup'].append(speedup)
       
       # Plot results
       import matplotlib.pyplot as plt
       
       plt.figure(figsize=(10, 6))
       plt.semilogx(observation_times, results['speedup'], 'o-', linewidth=2, markersize=8)
       plt.xlabel('Observation Time (years)')
       plt.ylabel('GPU Speedup Factor')
       plt.title('GPU Performance Scaling')
       plt.grid(True, alpha=0.3)
       plt.show()

Troubleshooting GPU Issues
--------------------------

Common Problems
~~~~~~~~~~~~~~~

**Out of Memory Errors**

.. code-block:: python

   # Solution 1: Reduce observation time or increase time step
   params['T'] = 0.5  # Reduce from 1.0 year
   params['dt'] = 20.0  # Increase from 10.0 seconds
   
   # Solution 2: Free GPU memory more aggressively
   import cupy as cp
   cp.get_default_memory_pool().free_all_blocks()
   
   # Solution 3: Fall back to CPU for problematic cases
   try:
       fisher_matrix = fisher_calc_gpu(**params)
   except cp.cuda.memory.OutOfMemoryError:
       print("GPU out of memory - falling back to CPU")
       fisher_calc_cpu = StableEMRIFisher(use_gpu=False)
       fisher_matrix = fisher_calc_cpu(**params)

**CUDA Errors**

.. code-block:: python

   def safe_gpu_calculation(params, max_retries=3):
       """Robust GPU calculation with error handling."""
       
       for attempt in range(max_retries):
           try:
               fisher_calc = StableEMRIFisher(use_gpu=True)
               return fisher_calc(**params)
               
           except Exception as e:
               print(f"Attempt {attempt + 1} failed: {e}")
               
               if attempt < max_retries - 1:
                   # Clear GPU state and retry
                   try:
                       import cupy as cp
                       cp.get_default_memory_pool().free_all_blocks()
                       cp.cuda.Device().synchronize()
                   except:
                       pass
               else:
                   # Final fallback to CPU
                   print("All GPU attempts failed - using CPU")
                   fisher_calc_cpu = StableEMRIFisher(use_gpu=False)
                   return fisher_calc_cpu(**params)

Performance Tips
~~~~~~~~~~~~~~~~

1. **Warm-up**: First GPU calculation includes initialization overhead
2. **Batch Size**: Balance memory usage vs. computation efficiency
3. **Memory Pooling**: Reuse GPU memory allocations when possible
4. **Precision**: Consider float32 vs. float64 tradeoffs
5. **Monitoring**: Track GPU utilization and memory usage

.. code-block:: python

   # GPU performance monitoring
   def monitor_gpu_performance():
       \"\"\"Monitor GPU performance during calculation.\"\"\"
       try:
           import GPUtil
           gpus = GPUtil.getGPUs()
           if gpus:
               gpu = gpus[0]
               print(f"GPU Utilization: {gpu.load * 100:.1f}%")
               print(f"GPU Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
               print(f"GPU Temperature: {gpu.temperature}°C")
       except ImportError:
           print("GPUtil not available for monitoring")

Multi-GPU Support
-----------------

For systems with multiple GPUs:

.. code-block:: python

   def multi_gpu_fisher(param_list, gpu_ids=None):
       """Distribute Fisher calculations across multiple GPUs."""
       
       if gpu_ids is None:
           import cupy as cp
           gpu_ids = list(range(cp.cuda.runtime.getDeviceCount()))
       
       results = []
       
       for i, params in enumerate(param_list):
           gpu_id = gpu_ids[i % len(gpu_ids)]
           
           with cp.cuda.Device(gpu_id):
               fisher_calc = StableEMRIFisher(use_gpu=True)
               fisher_matrix = fisher_calc(**params)
               results.append(fisher_matrix)
       
       return results

This tutorial covers GPU acceleration fundamentals. For advanced numerical considerations, see :doc:`stability_analysis`.
