# FisherCalculator
Calculates stable deltas for an 8th-order central finite-difference derivative and the corresponding Fisher elements for the full 14 dimensional parameter space of the 5PNAAK waveform model in the FastEMRIWaveforms package (https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms). 

### Installation
Simply download as a zip file, extract, and run in your favorite FEW environment. No build is required. 

`FisherCalculator.py` contains the class definition, supporting function definitions, and imports. 

`FisherCalculator_Example.ipynb` provides a simple usage example. 

# Processing the data

1. The directory `MCMC_FM_Data` contains a jupyter notebook `Process_Results.ipynb' that can be used to generate corner plots to check the MCMC simulation alongside the FM results.  
2. It is necessary to create directories: `data\_files/FM\_results` and `data\_files/MCMC\_results` and add the relevant data to those directories. The notebook reads in data from these two directories and produces a corner plot. The data is large, these will need to be sent separately. 
3. The directory `MCMC\_FM\_Data/plots` will have plots saved to it. 


