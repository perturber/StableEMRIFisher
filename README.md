# Stable EMRI Fisher Calculator

Calculates a stable Fisher matrix using numerical derivatives for the various EMRI waveform models in the FastEMRIWaveforms (FEW) package (https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms). 

### Installation and Execution
Clone the git repository, and install the environment with `pip` via
```console
pip install .
```
For development work, you can use an editable installation with the `-e` flag:
```console
pip install -e .
```

To use this package you will require the following dependences; follow their installation instructions to get started:

- FastEMRIWaveforms (https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms)

- (Optional) LISAAnalysisTools (https://github.com/mikekatz04/LISAanalysistools)

- (Optional) LISA-on-gpu (https://github.com/mikekatz04/lisa-on-gpu)

### Example usage
See the `examples` directory for examples of basic usage of this package, and `validation` for a comparison against some MCMC results.

Documentation coming soon...

### Citation
```
@software{kejriwal_2024_sef,
  author       = {Shubham Kejriwal and
                  Christian Chapman-Bird and
                  Ollie Burke},
  title        = {perturber/StableEMRIFisher.},
  publisher    = {Github},
  year         = "manuscript under preparation",
  url          = {https://github.com/perturber/StableEMRIFisher}
}
```
<!-- 
### Processing the data

1. The directory `MCMC_FM_Data` contains a jupyter notebook `Process_Results.ipynb' that can be used to generate corner plots to check the MCMC simulation alongside the FM results.  
2. It is necessary to create directories: `data\_files/FM\_results` and `data\_files/MCMC\_results` and add the relevant data to those directories. The notebook reads in data from these two directories and produces a corner plot. The data is large, these will need to be sent separately. 
3. The directory `MCMC\_FM\_Data/plots` will have plots saved to it. 

 -->
