# Python tomography library

## Depedencies
- Python 3.6+
- Cython
- C++ compiler with OPENMP support
- Numpy
- Scipy: Scipy sparse, Scipy signal, Scipy optimize
- PyWT
- Tqdm
- Skimage (scikit-image)
- Matplotlib
- H5py
- Argparse

## Usage
The script's core is the tomography class, which is initialized by giving the target image (filename), size (targetsize), noise level (noise) and measurement angles (itheta).  Inverse crimes can be avoided by simulating the  sinogram by a different grid and reconstructing the image with a second one.  
Parameter crimefree is False by default, dimbig and N_thetabig are the dimensions of the simulation sinogram.

It's possible to pass arguments 