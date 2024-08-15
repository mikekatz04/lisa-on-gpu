# fastlisaresponse: Generic LISA response function for GPUs

This code base provides a GPU-accelerated version of the generic time-domain LISA response function. The GPU-acceleration allows this code to be used directly in Parameter Estimation.

Please see the [documentation](https://mikekatz04.github.io/lisa-on-gpu/) for further information on these modules. The code can be found on Github [here](https://github.com/mikekatz04/lisa-on-gpu). It can be found on [Zenodo](https://zenodo.org/record/3981654#.XzS_KRNKjlw).

If you use all or any parts of this code, please cite [arXiv:2204.06633](https://arxiv.org/abs/2204.06633). See the [documentation](https://mikekatz04.github.io/lisa-on-gpu/) to properly cite specific modules.


## Getting Started

Install with pip (CPU only for now):
```
pip install fastlisaresponse
```
To import fastlisaresponse:

```
from fastlisaresponse import ResponseWrapper
```

See [examples notebook](https://github.com/mikekatz04/lisa-on-gpu/blob/master/examples/fast_LISA_response_tutorial.ipynb).


### Prerequisites

Now (version 1.0.7) `fastlisaresponse` requires the newest version of [LISA Analysis Tools](github.com/mikekatz04/LISAanalysistools). You can run `pip install lisaanalysistools`.

To install this software for CPU usage, you need Python >3.4 and NumPy. To run the examples, you will also need jupyter and matplotlib. We generally recommend installing everything, including gcc and g++ compilers, in the conda environment as is shown in the examples here. This generally helps avoid compilation and linking issues. If you use your own chosen compiler, you will need to make sure all necessary information is passed to the setup command (see below). You also may need to add information to the `setup.py` file.

To install this software for use with NVIDIA GPUs (compute capability >2.0), you need the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [CuPy](https://cupy.chainer.org/). The CUDA toolkit must have cuda version >8.0. Be sure to properly install CuPy within the correct CUDA toolkit version. Make sure the nvcc binary is on `$PATH` or set it as the `CUDAHOME` environment variable.


### Installing


Install with pip (CPU only for now):
```
pip install fastlisaresponse
```

To install from source:

0) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) if you do not have it.

1) Create a virtual environment.

```
conda create -n lisa_resp_env -c conda-forge gcc_linux-64 gxx_linux-64 numpy Cython scipy jupyter ipython h5py matplotlib python=3.12
conda activate lisa_resp_env
```

    If on MACOSX, substitute `gcc_linux-64` and `gxx_linus-64` with `clang_osx-64` and `clangxx_osx-64`.

    If you want a faster install, you can install the python packages (numpy, Cython, scipy, tqdm, jupyter, ipython, h5py, requests, matplotlib) with pip.

2) Clone the repository.

```
git clone https://github.com/mikekatz04/lisa-on-gpu.git
cd lisa-on-gpu
```

3) If using GPUs, use pip to [install cupy](https://docs-cupy.chainer.org/en/stable/install.html). 

```
pip install cupy-12x
```

4) Run install. Make sure CUDA is on your PATH.

```
python scripts/prebuild.py
pip install .
```

## Running the Tests

Run the example notebook or the tests using `unittest` from the main directory of the code:
```
python -m unittest discover
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/mikekatz04/lisa-on-gpu/tags).

Current Version: 1.0.7

## Authors

* **Michael Katz**
* Jean-Baptiste Bayle
* Alvin J. K. Chua
* Michele Vallisneri

### Contibutors

* Maybe you!

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* It was also supported in part through the computational resources and staff contributions provided for the Quest/Grail high performance computing facility at Northwestern University.
