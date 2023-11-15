# FisherCalculator
Calculates stable deltas for an 8th-order central finite-difference derivative and the corresponding Fisher elements.

### Installation
Simply download as a zip file, extract, and run in your favorite FEW environment. No build is required*. 

*Note that the implementation depends on dense-stepping trajectories. One must tweak FEW to set a sufficient dense-step size. Results here used step size dt = 3000 sec, by recompiling FEW after replacing the file /FastEMRIWaveforms/src/Inspiral.cc to the one provided in this repository (``Inspiral.cc``).  

`FisherCalculator.py` contains the class definition, supporting function definitions, and imports. 

`FisherCalculator_Example.ipynb` runs through a simple example of using the Fisher class. 

