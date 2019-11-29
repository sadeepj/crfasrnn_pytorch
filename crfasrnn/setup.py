from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='permuto_cpp',
      ext_modules=[cpp_extension.CppExtension('permuto_cpp', ['permuto.cpp', 'permutohedral.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
