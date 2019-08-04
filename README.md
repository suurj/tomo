# Python tomography library

## Introduction
This tiny Python library can be used to simulate and reconstruct X-ray tomography measurements. The library contains methods to calculate maximum a posteriori and conditional mean estimates out of sinograms of a given image.

## Dependencies
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
The _container_ consists of attributes which are gone through when saving to the result file. The class has only one method in addition to the init, _finish_. An instance of this class  is supposed to be created when the calculation is started and the finish method is called when it's completed. This saves the time difference to one attribute.

- tomography.\_\_init\_\_filename, targetsize=128, itheta=50, noise=0.0,  commonprefix="", dimbig = 607, N\_thetabig=421, crimefree=False,lhdev=None)
    -One might increase dimbig  and N\_thetabig from their default values, if the targetsize is near to 500.
    
- map\_tikhonov( alpha=1.0, order=1,maxiter=400,retim=True)
    -MAP function for Tikhonov regularization. Order means the order of the discrete derivative (1 or 2).  

-  map\_tv( alpha=1.0, maxiter=400,retim=True)
    -MAP function for total variation regularization.

- map\_cauchy(alpha=0.05, maxiter=400,retim=True)
    -MAP function for Cauchy difference prior. Default alpha value of 0.05 seems to have rather strong effect.
    
- map\_wavelet(alpha=1.0, type='haar', maxiter=400,levels=3 ,retim=True):
    -MAP function for total wavelet regularization. Default DWT level is 3, but might be increased for rather big images. On the other hand, 64x64, might be better with 2 levels only.
    
- hmcmc\_tikhonov( alpha, M=100, Madapt=20, order=1,mapstart=False,thinning=1,retim=True)
    - HMC function for CM estimation with Tikhonov regularization. Since MAP and CM should converge to the same solution with Gaussian priors, this function is just for testing purposes and thus not interesting. 
   - Note that _M_ and _Madapt_ are almost certainly too small even for HMC, but enough to verify that the function works.
    
- hmcmc\_tv(alpha, M=100, Madapt=20,mapstart=False,thinning=1,retim=True)
    - HMC function for CM estimation with TV regularization.
    
- hmcmc\_cauchy(alpha, M=100, Madapt=20,thinning=1,mapstart=True,retim=True)
    - HMC function for CM estimation with Cauchy prior.
    
- hmcmc\wavelet( alpha, M=100, Madapt=20, type='haar',levels=3,mapstart=False,thinning=1,retim=True)
    - HMC function for CM estimation with Wavelet prior.    