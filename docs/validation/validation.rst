Validation
==========

StableEMRIFisher has been validated against Monte Carlo Markov Chain (MCMC) parameter estimation studies to ensure accuracy of Fisher matrix predictions.

Fisher Matrix vs MCMC Comparison
---------------------------------

For this specific example, we consider an EMRI from the FEWv2 paper given by the first row in Tab.I in the latest FEW paper. We have applied the LISA response function and use second generation TDI variables (over A and E) with a ower spectral density
given by SciRDv1 with the galactic confusion noise included. The source has SNR 50, with parameters below

.. table:: EMRI Source Parameters
    :widths: 25 25 25 25

    +------------------+------------------+------------------+------------------+
    | Parameter        | Value            | Parameter        | Value            |
    +==================+==================+==================+==================+
    | :math:`m_1`      | :math:`1×10^6`   | :math:`m_2`      | :math:`10`       |
    | (M☉)             | :math:`M_☉`      | (M☉)             | :math:`M_☉`      |
    +------------------+------------------+------------------+------------------+
    | :math:`a`        | 0.998            | :math:`p_0`      | 7.7275           |
    +------------------+------------------+------------------+------------------+
    | :math:`e_0`      | 0.73             | :math:`xI_0`     | 1.0              |
    +------------------+------------------+------------------+------------------+
    | :math:`d_L`      | 2.204            | :math:`q_S`      | 0.8              |
    | (Gpc)            |                  |                  |                  |
    +------------------+------------------+------------------+------------------+
    | :math:`φ_S`      | 2.2              | :math:`q_K`      | 1.6              |
    +------------------+------------------+------------------+------------------+
    | :math:`φ_K`      | 1.2              | :math:`Φ_{φ0}`   | 2.0              |
    +------------------+------------------+------------------+------------------+
    | :math:`Φ_{θ0}`   | 0.0              | :math:`Φ_{r0}`   | 3.0              |
    +------------------+------------------+------------------+------------------+



The following interactive analysis compares Fisher matrix parameter uncertainties with full MCMC posterior distributions, including quantitative goodness-of-fit measures.

.. toctree::
   :maxdepth: 1
   
    check_fisher_against_mcmc_executed.ipynb

The notebook shows pre-executed results for fast documentation builds. To update the analysis, re-run the notebook and regenerate the executed version.

.. note::
   
   **Updating Validation Results**
   
   To refresh the validation analysis with new data or updated code:
   
   .. code-block:: bash
   
      # Quick update (from project root):
      ./docs/validation/update_validation_notebook.sh
      
      # Or manually (from docs/validation/):
      jupyter nbconvert --to notebook --execute check_fisher_against_mcmc.ipynb --output check_fisher_against_mcmc_executed.ipynb
   
   Then rebuild the documentation to see the updated results.
