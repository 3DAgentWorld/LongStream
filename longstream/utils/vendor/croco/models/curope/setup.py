from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

all_cuda_archs = cuda.get_gencode_flags().replace("compute=", "arch=").split()

setup(
    name="curope",
    ext_modules=[
        CUDAExtension(
            name="curope",
            sources=[
                "curope.cpp",
                "kernels.cu",
            ],
            extra_compile_args=dict(
                nvcc=["-O3", "--ptxas-options=-v", "--use_fast_math"] + all_cuda_archs,
                cxx=["-O3"],
            ),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
