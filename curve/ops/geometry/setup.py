from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='pip_cuda',
    ext_modules=[
        CUDAExtension('pip_cuda', [
            'src/pip_cuda.cpp',
            'src/pip_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})

setup(
    name='iou_cuda',
    ext_modules=[
        CUDAExtension('iou_cuda', [
            'src/iou_cuda.cpp',
            'src/iou_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
