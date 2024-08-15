# from future.utils import iteritems
import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import shutil


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if "CUDAHOME" in os.environ or "CUDA_HOME" in os.environ:
        home = (
            os.environ["CUDAHOME"]
            if "CUDAHOME" in os.environ
            else os.environ["CUDA_HOME"]
        )
        nvcc = pjoin(home, "bin", "nvcc")
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise EnvironmentError(
                "The nvcc binary could not be "
                "located in your $PATH. Either add it to your path, "
                "or set $CUDAHOME"
            )
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": pjoin(home, "include"),
        "lib64": pjoin(home, "lib64"),
    }
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError(
                "The CUDA %s path could not be " "located in %s" % (k, v)
            )

    return cudaconfig


def customize_compiler_for_nvcc(self):

    # track all the object files generated with cuda device code
    self.cuda_object_files = []

    # Tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        # generate a special object file that will contain linked in
        # relocatable device code
        if src == "zzzzzzzzzzzzzzzz.cu":
            self.set_executable("compiler_so", CUDA["nvcc"])
            postargs = extra_postargs["nvcclink"]
            cc_args = self.cuda_object_files[1:]
            src = self.cuda_object_files[0]
        elif os.path.splitext(src)[1] == ".cu":
            self.set_executable("compiler_so", CUDA["nvcc"])
            postargs = extra_postargs["nvcc"]
            self.cuda_object_files.append(obj)
        else:
            postargs = extra_postargs["gcc"]
        super(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile


# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


try:
    CUDA = locate_cuda()
    run_cuda_install = True
except OSError:
    run_cuda_install = False

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


# lib_gsl_dir = "/opt/local/lib"
# include_gsl_dir = "/opt/local/include"

# find detector source files from installed distribution.
import lisatools

path_to_lisatools = lisatools.__file__.split("__init__.py")[0]
path_to_lisatools_cutils = path_to_lisatools + "cutils/"

# if installing for CUDA, build Cython extensions for gpu modules
if run_cuda_install:

    # gpu_extension = dict(
    #     libraries=["cudart", "cublas", "cusparse"],
    #     library_dirs=[CUDA["lib64"]],
    #     runtime_library_dirs=[CUDA["lib64"]],
    #     language="c++",
    #     # This syntax is specific to this build system
    #     # we're only going to use certain compiler args with nvcc
    #     # and not with gcc the implementation of this trick is in
    #     # customize_compiler()
    #     extra_compile_args={
    #         "gcc": ["-std=c++11"],  # '-g'],
    #         "nvcc": [
    #             "-arch=sm_70",
    #             # "-gencode=arch=compute_30,code=sm_30",
    #             # "-gencode=arch=compute_50,code=sm_50",
    #             # "-gencode=arch=compute_52,code=sm_52",
    #             # "-gencode=arch=compute_60,code=sm_60",
    #             # "-gencode=arch=compute_61,code=sm_61",
    #             "-gencode=arch=compute_70,code=sm_70",
    #             #'-gencode=arch=compute_75,code=sm_75',
    #             #'-gencode=arch=compute_75,code=compute_75',
    #             "-std=c++11",
    #             "--default-stream=per-thread",
    #             "--ptxas-options=-v",
    #             "-c",
    #             "--compiler-options",
    #             "'-fPIC'",
    #             # "-G",
    #             # "-g",
    #             # "-O0",
    #             # "-lineinfo",
    #         ],  # for debugging
    #     },
    #     include_dirs=[
    #         numpy_include,
    #         CUDA["include"],
    #         "include/",
    #     ],
    # )

    response_ext = Extension(
        "fastlisaresponse.pyresponse_gpu",
        sources=[
            path_to_lisatools_cutils + "src/Detector.cu",
            "src/LISAResponse.cu",
            "src/responselisa.pyx",
            "zzzzzzzzzzzzzzzz.cu",
        ],
        library_dirs=[CUDA["lib64"]],
        language="c++",
        libraries=["cudart", "cudadevrt"],
        runtime_library_dirs=[CUDA["lib64"]],
        extra_compile_args={
            "gcc": ["-std=c++11"],
            "nvcc": ["-arch=sm_80", "-rdc=true", "--compiler-options", "'-fPIC'"],
            "nvcclink": [
                "-arch=sm_80",
                "--device-link",
                "--compiler-options",
                "'-fPIC'",
            ],
        },
        include_dirs=[
            numpy_include,
            CUDA["include"],
            "include",
            path_to_lisatools_cutils + "include",
        ],
    )

cpu_extension = dict(
    libraries=[],
    language="c++",
    # This syntax is specific to this build system
    # we're only going to use certain compiler args with nvcc
    # and not with gcc the implementation of this trick is in
    # customize_compiler()
    extra_compile_args={
        "gcc": ["-std=c++11"],
    },  # '-g'],
    include_dirs=[numpy_include, "./include", path_to_lisatools_cutils + "include"],
)

response_cpu_ext = Extension(
    "fastlisaresponse.pyresponse_cpu",
    sources=[
        path_to_lisatools_cutils + "src/Detector.cpp",
        "src/LISAResponse.cpp",
        "src/responselisa_cpu.pyx",
    ],
    **cpu_extension
)


cpu_extensions = [response_cpu_ext]  # , detector_ext]

if run_cuda_install:
    extensions = [response_ext] + cpu_extensions

else:
    extensions = cpu_extensions

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fastlisaresponse",
    author="Michael Katz",
    author_email="mikekatz04@gmail.com",
    ext_modules=extensions,
    # Inject our custom trigger
    packages=["fastlisaresponse", "fastlisaresponse.utils"],
    cmdclass={"build_ext": custom_build_ext},
    # Since the package has c code, the egg cannot be zipped
    zip_safe=False,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="1.0.7",
    url="https://github.com/mikekatz04/lisa-on-gpu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Environment :: GPU :: NVIDIA CUDA",
        "Natural Language :: English",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.12",
)
