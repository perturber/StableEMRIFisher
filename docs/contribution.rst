Contributing and Development
=============================

For the latest development version and/or you want to build the documentation locally

**With UV (recommended):**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/perturber/StableEMRIFisher.git
   cd StableEMRIFisher
   
   # Set up development environment
   uv venv
   uv sync --dev
   
   # Or install manually
   uv pip install -e ".[docs,dev]"  # Add cuda12x for GPU support

**With pip:**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/perturber/StableEMRIFisher.git
   cd StableEMRIFisher
   
   # Install in development mode
   pip install -e ".[docs,dev]"  # Maybe want cuda12x here as well if using GPUs