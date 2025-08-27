Examples
========

Ready-to-run examples demonstrating StableEMRIFisher capabilities.

Code Examples
-------------

The following Python scripts are available in the ``examples/`` directory:

Basic Usage Examples
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/basic_usage_no_stability_check.py
   :language: python
   :caption: examples/basic_usage_no_stability_check.py

.. literalinclude:: ../examples/basic_usage_stability_check.py
   :language: python  
   :caption: examples/basic_usage_stability_check.py

Jupyter Notebook Examples
--------------------------

Interactive notebooks for detailed exploration:

SEF Inner Product Example
~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrates the inner product calculations used in Fisher matrix computation.

.. note::
   The notebook ``examples/SEF_inner_product_example.ipynb`` contains detailed 
   explanations of the mathematical foundations and implementation details.

Main Derivative Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

The main analysis notebook showing derivative computation and validation:

.. note::
   See ``StableEMRIDerivative_FEW2April.ipynb`` in the repository root for 
   comprehensive derivative analysis and validation studies.

Running the Examples
--------------------

To run the Python scripts:

.. code-block:: bash

   cd examples
   python basic_usage_stability_check.py

To run the Jupyter notebooks:

.. code-block:: bash

   jupyter notebook SEF_inner_product_example.ipynb

Additional Examples
-------------------

More examples can be found in:

* **validation/**: Scripts comparing Fisher matrix results with MCMC chains
* **MCMC_FM_Data/**: Data processing and comparison notebooks  
* **TestRun/**: Quick test scripts for development

Example Data
------------

Example Fisher matrix outputs and validation data are stored in HDF5 format:

* ``TestRun/Fisher.h5``: Sample Fisher matrix results
* ``MCMC_FM_Data/mcmc_data/case_1_few.h5``: MCMC comparison data

Loading Example Results
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import h5py
   import numpy as np
   
   # Load pre-computed Fisher matrix
   with h5py.File('TestRun/Fisher.h5', 'r') as f:
       fisher_matrix = f['fisher_matrix'][:]
       parameters = f['parameters'][:]
       
   print(f"Loaded Fisher matrix shape: {fisher_matrix.shape}")
   print(f"Condition number: {np.linalg.cond(fisher_matrix):.2e}")
