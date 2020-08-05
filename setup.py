from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    ext_modules = cythonize([Extension("cyt",sources=["cyt.pyx"],extra_link_args=["-O3","-fopenmp"],language="c++",extra_compile_args=["-O3", "-march=native","-fopenmp"],include_dirs=[numpy.get_include()]),
                            Extension("matrices",sources=["matrices.pyx"],extra_link_args=["-O3","-fopenmp"],language="c++",extra_compile_args=["-O3", "-march=native","-fopenmp"],include_dirs=[numpy.get_include()])])
    
)
#setup(
#    ext_modules = cythonize("cyt.pyx")
    
#)
#"-fopenmp",
'''
setup(
  name = 'Cytti',
  ext_modules=[
    Extension('cyt',
              sources=['cyt.pyx'],
              extra_compile_args=['-O3'],
              language='c')
    ],
  cmdclass = {'build_ext': build_ext}
)
'''
