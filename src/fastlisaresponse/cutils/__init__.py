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
    # TDSplineTDIWaveformWrap: object
    FDSplineTDIWaveformWrap: object
    GBTDIonTheFlyWrap: object
    LISAResponseWrap: object
    LISAResponse: object
    OrbitsWrap: object
    TDIConfigWrap: object
    TDIConfig: object
    CubicSplineWrap: object
    WDMDomainWrap: object
    WaveletLookupTableWrap: object
    GBComputationGroupWrap: object
    TDITypeDict: object

class FastLISAResponseBackend:
    # TDSplineTDIWaveformWrap: object
    FDSplineTDIWaveformWrap: object
    GBTDIonTheFlyWrap: object
    LISAResponseWrap: object
    LISAResponse: object
    OrbitsWrap: object
    TDIConfigWrap: object
    TDIConfig: object
    CubicSplineWrap: object
    WDMDomainWrap: object
    WaveletLookupTableWrap: object
    GBComputationGroupWrap: object
    TDITypeDict: object
    
    def __init__(self, fastlisaresponse_backend_methods):

        # set direct fastlisaresponse methods
        # pass rest to general backend
        assert isinstance(fastlisaresponse_backend_methods, FastLISAResponseBackendMethods)

        # self.TDSplineTDIWaveformWrap = fastlisaresponse_backend_methods.TDSplineTDIWaveformWrap
        self.FDSplineTDIWaveformWrap = fastlisaresponse_backend_methods.FDSplineTDIWaveformWrap
        self.GBTDIonTheFlyWrap = fastlisaresponse_backend_methods.GBTDIonTheFlyWrap
        self.OrbitsWrap = fastlisaresponse_backend_methods.OrbitsWrap
        self.TDIConfigWrap = fastlisaresponse_backend_methods.TDIConfigWrap
        self.TDIConfig = fastlisaresponse_backend_methods.TDIConfig
        self.CubicSplineWrap = fastlisaresponse_backend_methods.CubicSplineWrap
        self.LISAResponseWrap = fastlisaresponse_backend_methods.LISAResponseWrap
        self.LISAResponse = fastlisaresponse_backend_methods.LISAResponse
        self.WDMDomainWrap = fastlisaresponse_backend_methods.WDMDomainWrap
        self.WaveletLookupTableWrap = fastlisaresponse_backend_methods.WaveletLookupTableWrap
        self.GBComputationGroupWrap = fastlisaresponse_backend_methods.GBComputationGroupWrap
        self.TDITypeDict = fastlisaresponse_backend_methods.TDITypeDict

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
            import fastlisaresponse_backend_cpu.tdionthefly

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cpu' backend could not be imported."
            ) from e

        numpy = FastLISAResponseCpuBackend.check_numpy()

        tmp = {
            "XYZ": fastlisaresponse_backend_cpu.tdionthefly.TDI_XYZ,
            "AET": fastlisaresponse_backend_cpu.tdionthefly.TDI_AET,
            "AE": fastlisaresponse_backend_cpu.tdionthefly.TDI_AE,
        }
        return FastLISAResponseBackendMethods(
            # TDSplineTDIWaveformWrap=fastlisaresponse_backend_cpu.tdionthefly.FDSplineTDIWaveformWrap,
            FDSplineTDIWaveformWrap=fastlisaresponse_backend_cpu.tdionthefly.FDSplineTDIWaveformWrapCPU,
            GBTDIonTheFlyWrap=fastlisaresponse_backend_cpu.tdionthefly.GBTDIonTheFlyWrapCPU,
            LISAResponseWrap=fastlisaresponse_backend_cpu.responselisa.LISAResponseWrapCPU,
            LISAResponse=fastlisaresponse_backend_cpu.responselisa.LISAResponseCPU,
            OrbitsWrap=fastlisaresponse_backend_cpu.responselisa.OrbitsWrapCPU_responselisa,
            TDIConfig=fastlisaresponse_backend_cpu.responselisa.TDIConfigCPU,
            TDIConfigWrap=fastlisaresponse_backend_cpu.responselisa.TDIConfigWrapCPU,
            CubicSplineWrap=fastlisaresponse_backend_cpu.responselisa.CubicSplineWrapCPU_responselisa,
            WDMDomainWrap=fastlisaresponse_backend_cpu.tdionthefly.WDMDomainWrapCPU,
            WaveletLookupTableWrap=fastlisaresponse_backend_cpu.tdionthefly.WaveletLookupTableWrapCPU,
            GBComputationGroupWrap=fastlisaresponse_backend_cpu.tdionthefly.GBComputationGroupWrapCPU,
            TDITypeDict=tmp,
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
            import fastlisaresponse_backend_cuda11x.tdionthefly

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

        tmp = {
            "XYZ": fastlisaresponse_backend_cuda11x.tdionthefly.TDI_XYZ,
            "AET": fastlisaresponse_backend_cuda11x.tdionthefly.TDI_AET,
            "AE": fastlisaresponse_backend_cuda11x.tdionthefly.TDI_AE,
        }
        return FastLISAResponseBackendMethods(
            # TDSplineTDIWaveformWrap=fastlisaresponse_backend_cpu.tdionthefly.FDSplineTDIWaveformWrap,
            FDSplineTDIWaveformWrap=fastlisaresponse_backend_cuda11x.tdionthefly.FDSplineTDIWaveformWrapGPU,
            GBTDIonTheFlyWrap=fastlisaresponse_backend_cuda11x.tdionthefly.GBTDIonTheFlyWrapGPU,
            LISAResponseWrap=fastlisaresponse_backend_cuda11x.responselisa.LISAResponseWrapGPU,
            LISAResponse=fastlisaresponse_backend_cuda11x.responselisa.LISAResponseGPU,
            OrbitsWrap=fastlisaresponse_backend_cuda11x.responselisa.OrbitsWrapGPU_responselisa,
            TDIConfig=fastlisaresponse_backend_cuda11x.responselisa.TDIConfigGPU,
            TDIConfigWrap=fastlisaresponse_backend_cuda11x.responselisa.TDIConfigWrapGPU,
            CubicSplineWrap=fastlisaresponse_backend_cuda11x.responselisa.CubicSplineWrapGPU_responselisa,
            WDMDomainWrap=fastlisaresponse_backend_cuda11x.tdionthefly.WDMDomainWrapGPU,
            WaveletLookupTableWrap=fastlisaresponse_backend_cuda11x.tdionthefly.WaveletLookupTableWrapGPU,
            GBComputationGroupWrap=fastlisaresponse_backend_cuda11x.tdionthefly.GBComputationGroupWrapGPU,
            TDITypeDict=tmp,
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
            import fastlisaresponse_backend_cuda12x.tdionthefly

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
        tmp = {
            "XYZ": fastlisaresponse_backend_cuda12x.tdionthefly.TDI_XYZ,
            "AET": fastlisaresponse_backend_cuda12x.tdionthefly.TDI_AET,
            "AE": fastlisaresponse_backend_cuda12x.tdionthefly.TDI_AE,
        }
        return FastLISAResponseBackendMethods(
            # TDSplineTDIWaveformWrap=fastlisaresponse_backend_cpu.tdionthefly.FDSplineTDIWaveformWrap,
            FDSplineTDIWaveformWrap=fastlisaresponse_backend_cuda12x.tdionthefly.FDSplineTDIWaveformWrapGPU,
            GBTDIonTheFlyWrap=fastlisaresponse_backend_cuda12x.tdionthefly.GBTDIonTheFlyWrapGPU,
            LISAResponseWrap=fastlisaresponse_backend_cuda12x.responselisa.LISAResponseWrapGPU,
            LISAResponse=fastlisaresponse_backend_cuda12x.responselisa.LISAResponseGPU,
            OrbitsWrap=fastlisaresponse_backend_cuda12x.responselisa.OrbitsWrapGPU_responselisa,
            TDIConfig=fastlisaresponse_backend_cuda12x.responselisa.TDIConfigGPU,
            TDIConfigWrap=fastlisaresponse_backend_cuda12x.responselisa.TDIConfigWrapGPU,
            CubicSplineWrap=fastlisaresponse_backend_cuda12x.responselisa.CubicSplineWrapGPU_responselisa,
            WDMDomainWrap=fastlisaresponse_backend_cuda12x.tdionthefly.WDMDomainWrapGPU,
            WaveletLookupTableWrap=fastlisaresponse_backend_cuda12x.tdionthefly.WaveletLookupTableWrapGPU,
            GBComputationGroupWrap=fastlisaresponse_backend_cuda12x.tdionthefly.GBComputationGroupWrapGPU,
            TDITypeDict=tmp,
            xp=cupy,
        )


"""List of existing backends, per default order of preference."""
# TODO: __all__ ?


