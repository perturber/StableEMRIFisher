# Stable EMRI Fisher Calculator

## In this branch, we calculate the Fisher matrices for subsonic interactions between the accretion disk and a compact object on eccentric orbits.

*The current version of this class needs GPU to work*.

Calculates a stable Fisher matrix using numerical derivatives for the 14-dimensional parameter space of the 5PNAAK waveform model in the FastEMRIWaveforms (FEW) package (https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms). 

### Installation and Execution
Simply download as a zip file, extract, and run in your favorite FEW environment. No build is required. 

`fisher.py` contains the StableEMRIFisher class definition, supporting function definitions, and FEW imports. 

`StableEMRIFisher_Example.ipynb` contains a simple example with results in the `TestRun` folder.

### Processing the data

1. The directory `MCMC_FM_Data` contains a jupyter notebook `Process_Results.ipynb' that can be used to generate corner plots to check the MCMC simulation alongside the FM results.  
2. It is necessary to create directories: `data\_files/FM\_results` and `data\_files/MCMC\_results` and add the relevant data to those directories. The notebook reads in data from these two directories and produces a corner plot. The data is large, these will need to be sent separately. 
3. The directory `MCMC\_FM\_Data/plots` will have plots saved to it. 


