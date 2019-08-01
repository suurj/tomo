# Python tomography library

## Introduction
This tiny Python library can be used to simulate and reconstruct X-ray tomography measurements. The library contains methods to calculate maximum a posteriori and conditional mean estimates out of sinograms of a given image.

## Depedencies
- Python 3.6+
- Cython and C++ compiler with OPENMP support
- Numpy
- Scipy: Scipy sparse, Scipy signal, Scipy optimize
- PyWT
- Tqdm
- Skimage (scikit-image)
- Matplotlib
- H5py
- Pathlib
- Argparse

## Usage
The main script can be used with or without command line arguments. When command line arguments are used, only one instance of the tomography class is created and only one reconstruction can be done. Alternatively one can enter the desired calculations directly to the main.py, if multiple calculations are done and multiple instances of measurements are created.
### Tomography class
The script's core is the tomography class, which is initialized by giving the target image (filename), size (targetsize), noise level (noise) and measurement angles (itheta).  Inverse crimes can be avoided by simulating the  sinogram by a different grid and reconstructing the image with a second one.  
Parameter crimefree is False by default, dimbig and N_thetabig are the dimensions of the simulation sinogram. Preferrably they should be primes. 
The lhdev parameter refers to the sinogram measurement likelihood sigma. By default it's None, which means that the sigma is assumed to be proportional to the noise level: it's (sinogram_maxvalue * noise) ^ 2. If the noise level is 0, then the sigma is (sinogram_maxvalue * 0.01) ^ 2.

### Methods
After the class is initialized, the calculations itself can be run. 