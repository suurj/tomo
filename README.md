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
The main script can be used with or without command line arguments. When command line arguments are used, only one instance of the tomography class is created and only one reconstruction can be done. Alternatively one can type and call the desired calculations directly from the main.py, if multiple calculations are done and multiple instances of measurements are created.
### Tomography class
The script's core is the tomography class, which is initialized by giving the target image (_filename_), size (_targetsize_), noise level (_noise_) and measurement angles (_itheta_).  Inverse crimes can be avoided by simulating the  sinogram by a different grid and reconstructing the image with a second one.  
Parameter _crimefree_ is False by default, _dimbig_ and _N\_thetabig_ are the dimensions of the simulation sinogram. Preferrably they should be primes. 
The _lhdev_ parameter refers to the sinogram measurement likelihood sigma. By default it's None, which means that the sigma is assumed to be proportional to the noise level: it's (sinogram\_maxvalue * noise) ^ 2. If the noise level is 0, then the sigma is (sinogram\_maxvalue * 0.01) ^ 2.

There are two ways to enter the  measurement angles: user can enter either one integer (which then is the number of angles between 0 and 180 degress) or three integers, which will refer the first angle, last angle and the number of angles between them, respectively.

### Methods
After the class is initialized, the calculations itself can be run. The names of methods are rather self-descriptive. The methods of the tomography class beginning with  with the word _map_ refer to MAP estimates with different priors, _mwg_-starting methods refer to CM estimation by Metropolis-within-Gibbs (SCAM) and _hmc_-beginning methods refer to CM estimation by Hamiltonian Monte Carlo. 

For all methods, the most important  parameter is the prior's regularization parameter. Depending on the prior, it can be selected also so that the reconstructions remain approximately the same even if the resolution of the image is increased. The second important parameter is technical. With the _retim_ parameter one can select what the methods actually return. The _retim_ is by default True for all methods, which means that the methods return only the reconstructed image. If the parameter is False, an instance of _container_ class is returned. The class can be feed as an argument to the _saveresult_ method, which then saves the result to a hdf5 file for later analysis.

###

