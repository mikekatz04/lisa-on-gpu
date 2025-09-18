from __future__ import annotations
import dataclasses
import enum
import types
import typing
import abc
from typing import Optional, Sequence, TypeVar, Union
from ..utils.exceptions import *

from gpubackendtools.gpubackendtools import BackendMethods, CpuBackend, Cuda11xBackend, Cuda12xBackend
from gpubackendtools.exceptions import *

@dataclasses.dataclass
class FastLISAResponseBackendMethods(BackendMethods):
    get_response_wrap: typing.Callable[(...), None]
    get_tdi_delays_wrap: typing.Callable[(...), None]


class FastLISAResponseBackend:
    get_response_wrap: typing.Callable[(...), None]
    get_tdi_delays_wrap: typing.Callable[(...), None]

    def __init__(self, fastlisaresponse_backend_methods):

        # set direct fastlisaresponse methods
        # pass rest to general backend
        assert isinstance(fastlisaresponse_backend_methods, FastLISAResponseBackendMethods)

        self.get_response_wrap = fastlisaresponse_backend_methods.get_response_wrap
        self.get_tdi_delays_wrap = fastlisaresponse_backend_methods.get_tdi_delays_wrap


class FastLISAResponseCpuBackend(CpuBackend, FastLISAResponseBackend):
    """Implementation of the CPU backend"""
    
    _backend_name = "fastlisaresponse_backend_cpu"
    _name = "fastlisaresponse_cpu"
    def __init__(self, *args, **kwargs):
        CpuBackend.__init__(self, *args, **kwargs)
        FastLISAResponseBackend.__init__(self, self.cpu_methods_loader())

    @staticmethod
    def cpu_methods_loader() -> FastLISAResponseBackendMethods:
        try:
            import fastlisaresponse_backend_cpu.responselisa
            
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cpu' backend could not be imported."
            ) from e

        numpy = FastLISAResponseCpuBackend.check_numpy()

        return FastLISAResponseBackendMethods(
            get_response_wrap=fastlisaresponse_backend_cpu.responselisa.get_response_wrap,
            get_tdi_delays_wrap=fastlisaresponse_backend_cpu.responselisa.get_tdi_delays_wrap,
            xp=numpy,
        )


class FastLISAResponseCuda11xBackend(Cuda11xBackend, FastLISAResponseBackend):

    """Implementation of CUDA 11.x backend"""
    _backend_name : str = "fastlisaresponse_backend_cuda11x"
    _name = "fastlisaresponse_cuda11x"

    def __init__(self, *args, **kwargs):
        Cuda11xBackend.__init__(self, *args, **kwargs)
        FastLISAResponseBackend.__init__(self, self.cuda11x_module_loader())
        
    @staticmethod
    def cuda11x_module_loader():
        try:
            import fastlisaresponse_backend_cuda11x.responselisa

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda11x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda11x' backend requires cupy", pip_deps=["cupy-cuda11x"]
            ) from e

        return FastLISAResponseBackendMethods(
            get_response_wrap=fastlisaresponse_backend_cuda11x.responselisa.get_response_wrap,
            get_tdi_delays_wrap=fastlisaresponse_backend_cuda11x.responselisa.get_tdi_delays_wrap,
            xp=cupy,
        )

class FastLISAResponseCuda12xBackend(Cuda12xBackend, FastLISAResponseBackend):
    """Implementation of CUDA 12.x backend"""
    _backend_name : str = "fastlisaresponse_backend_cuda12x"
    _name = "fastlisaresponse_cuda12x"
    
    def __init__(self, *args, **kwargs):
        Cuda12xBackend.__init__(self, *args, **kwargs)
        FastLISAResponseBackend.__init__(self, self.cuda12x_module_loader())
        
    @staticmethod
    def cuda12x_module_loader():
        try:
            import fastlisaresponse_backend_cuda12x.responselisa

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda12x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda12x' backend requires cupy", pip_deps=["cupy-cuda12x"]
            ) from e

        return FastLISAResponseBackendMethods(
            get_response_wrap=fastlisaresponse_backend_cuda12x.responselisa.get_response_wrap,
            get_tdi_delays_wrap=fastlisaresponse_backend_cuda12x.responselisa.get_tdi_delays_wrap,
            xp=cupy,
        )


KNOWN_BACKENDS = {
    "cuda12x": FastLISAResponseCuda12xBackend,
    "cuda11x": FastLISAResponseCuda11xBackend,
    "cpu": FastLISAResponseCpuBackend,
}

"""List of existing backends, per default order of preference."""
# TODO: __all__ ?


