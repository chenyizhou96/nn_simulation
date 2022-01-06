from distutils.core import Extension, setup
from Cython.Build import cythonize


ext = Extension(name="TGSL", sources=["cython_binding.pyx"],extra_compile_args=["-std=c++17"],
    extra_link_args=["-std=c++17"])
setup(ext_modules=cythonize(ext,  language_level=3),
    include_dirs= ['../../external_libraries/eigen', '../../tgsl/library/',
                    '../../external_libraries/cxxopts'])