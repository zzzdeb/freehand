# Freehand ultrasound without external trackers

This repository contains algorithms to train deep neural networks, using scans of freehand ultrasound image frames acquired with ground-truth frame locations from external spatial trackers. The aim is to reconstruct the spatial frame locations or relative transformation between them, on the newly acquired scans.

The most up-to-date code is in the `dev0` branch, where the `train.py` and `test.py` under the `scripts` folder can be adapted with local data path. The conda environment required to run the code is detailed in [requirements](/doc/requirements.md).

The data used in the following papers can be downloaded [here](https://doi.org/10.5281/zenodo.7740734).

"Qi et al. Trackerless freehand ultrasound with sequence modelling and auxiliary transformation over past and future frames" 

"Qi et al. Long-term Dependency for 3D Reconstruction of Freehand Ultrasound Without External Tracker" 
