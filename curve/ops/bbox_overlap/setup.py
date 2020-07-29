from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='bbox_overlap_cpu',
      ext_modules=[CppExtension('bbox_overlap_cpu', ['src/bbox_overlap_cpu.cpp'])],
      cmdclass={'build_ext': BuildExtension})
