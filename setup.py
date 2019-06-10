from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    ext_modules = cythonize(Extension("cyt",sources=["cyt.pyx"],extra_link_args=["-O3"],language="c++",extra_compile_args=["-O3", "-march=native"]))
    
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