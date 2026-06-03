from __future__ import annotations
import dataclasses
import enum
import types
import typing
import abc
from typing import Optional, Sequence, TypeVar, Union
from ..utils.exceptions import *

from gpubackendtools.gpubackendtools import BackendMethods, CpuBackend, Cuda11xBackend, Cuda12xBackend, Cuda13xBackend
from gpubackendtools.exceptions import *

@dataclasses.dataclass
class FastLISAResponseBackendMethods(BackendMethods):
    TDSplineTDIWaveformWrap: object
    FDSplineTDIWaveformWrap: object
    GBTDIonTheFlyWrap: object
    SOBBHTDIonTheFlyWrap: object
    LISAResponseWrap: object
    LISAResponse: object
    OrbitsWrap: object
    TDIConfigWrap: object
    TDIConfig: object
    CubicSplineWrap: object
    WDMSettingsWrap: object
    WDMDomainWrap: object
    FDDomainWrap: object
    WaveletLookupTableWrap: object
    GBComputationGroupWrap: object
    SOBBHComputationGroupWrap: object
    TDITypeDict: object

class FastLISAResponseBackend:
    TDSplineTDIWaveformWrap: object
    FDSplineTDIWaveformWrap: object
    GBTDIonTheFlyWrap: object
    SOBBHTDIonTheFlyWrap: object
    LISAResponseWrap: object
    LISAResponse: object
    OrbitsWrap: object
    TDIConfigWrap: object
    TDIConfig: object
    CubicSplineWrap: object
    WDMSettingsWrap: object
    WDMDomainWrap: object
    FDDomainWrap: object
    WaveletLookupTableWrap: object
    GBComputationGroupWrap: object
    SOBBHComputationGroupWrap: object
    TDITypeDict: object
    
    def __init__(self, fastlisaresponse_backend_methods):

        # set direct fastlisaresponse methods
        # pass rest to general backend
        assert isinstance(fastlisaresponse_backend_methods, FastLISAResponseBackendMethods)

        self.TDSplineTDIWaveformWrap = fastlisaresponse_backend_methods.TDSplineTDIWaveformWrap
        self.FDSplineTDIWaveformWrap = fastlisaresponse_backend_methods.FDSplineTDIWaveformWrap
        self.GBTDIonTheFlyWrap = fastlisaresponse_backend_methods.GBTDIonTheFlyWrap
        self.SOBBHTDIonTheFlyWrap = fastlisaresponse_backend_methods.SOBBHTDIonTheFlyWrap
        self.OrbitsWrap = fastlisaresponse_backend_methods.OrbitsWrap
        self.TDIConfigWrap = fastlisaresponse_backend_methods.TDIConfigWrap
        self.TDIConfig = fastlisaresponse_backend_methods.TDIConfig
        self.CubicSplineWrap = fastlisaresponse_backend_methods.CubicSplineWrap
        self.LISAResponseWrap = fastlisaresponse_backend_methods.LISAResponseWrap
        self.LISAResponse = fastlisaresponse_backend_methods.LISAResponse
        self.WDMSettingsWrap = fastlisaresponse_backend_methods.WDMSettingsWrap
        self.WDMDomainWrap = fastlisaresponse_backend_methods.WDMDomainWrap
        self.FDDomainWrap = fastlisaresponse_backend_methods.FDDomainWrap
        self.WaveletLookupTableWrap = fastlisaresponse_backend_methods.WaveletLookupTableWrap
        self.GBComputationGroupWrap = fastlisaresponse_backend_methods.GBComputationGroupWrap
        self.SOBBHComputationGroupWrap = fastlisaresponse_backend_methods.SOBBHComputationGroupWrap
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
            # Phase 3E (2026-06-02): response classes (LISAResponseWrap,
            # TDIConfigWrap, OrbitsWrap_responselisa, CubicSplineWrap_responselisa)
            # were absorbed by LISAanalysistools and now live in
            # `lisatools_backend_cpu.pycppdetector`. The `responselisa`
            # pybind11 module no longer exists.
            import lisatools_backend_cpu.pycppdetector as _lat_pd
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
            TDSplineTDIWaveformWrap=fastlisaresponse_backend_cpu.tdionthefly.TDSplineTDIWaveformWrapCPU,
            FDSplineTDIWaveformWrap=fastlisaresponse_backend_cpu.tdionthefly.FDSplineTDIWaveformWrapCPU,
            GBTDIonTheFlyWrap=fastlisaresponse_backend_cpu.tdionthefly.GBTDIonTheFlyWrapCPU,
            SOBBHTDIonTheFlyWrap=fastlisaresponse_backend_cpu.tdionthefly.SOBBHTDIonTheFlyWrapCPU,
            LISAResponseWrap=_lat_pd.LISAResponseWrapCPU,
            LISAResponse=_lat_pd.LISAResponseCPU,
            OrbitsWrap=_lat_pd.OrbitsWrapCPU_responselisa,
            TDIConfig=_lat_pd.TDIConfigCPU,
            TDIConfigWrap=_lat_pd.TDIConfigWrapCPU,
            CubicSplineWrap=_lat_pd.CubicSplineWrapCPU_responselisa,
            WDMSettingsWrap=_lat_pd.WDMSettingsWrapCPU,
            WDMDomainWrap=fastlisaresponse_backend_cpu.tdionthefly.WDMDomainWrapCPU,
            FDDomainWrap=_lat_pd.FDDomainWrapCPU,
            WaveletLookupTableWrap=fastlisaresponse_backend_cpu.tdionthefly.WaveletLookupTableWrapCPU,
            GBComputationGroupWrap=fastlisaresponse_backend_cpu.tdionthefly.GBComputationGroupWrapCPU,
            SOBBHComputationGroupWrap=fastlisaresponse_backend_cpu.tdionthefly.SOBBHComputationGroupWrapCPU,
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
            # Phase 3E: response classes now in lisatools_backend_cuda11x.pycppdetector.
            import lisatools_backend_cuda11x.pycppdetector as _lat_pd
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
            TDSplineTDIWaveformWrap=fastlisaresponse_backend_cuda11x.tdionthefly.TDSplineTDIWaveformWrapGPU,
            FDSplineTDIWaveformWrap=fastlisaresponse_backend_cuda11x.tdionthefly.FDSplineTDIWaveformWrapGPU,
            GBTDIonTheFlyWrap=fastlisaresponse_backend_cuda11x.tdionthefly.GBTDIonTheFlyWrapGPU,
            SOBBHTDIonTheFlyWrap=fastlisaresponse_backend_cuda11x.tdionthefly.SOBBHTDIonTheFlyWrapGPU,
            LISAResponseWrap=_lat_pd.LISAResponseWrapGPU,
            LISAResponse=_lat_pd.LISAResponseGPU,
            OrbitsWrap=_lat_pd.OrbitsWrapGPU_responselisa,
            TDIConfig=_lat_pd.TDIConfigGPU,
            TDIConfigWrap=_lat_pd.TDIConfigWrapGPU,
            CubicSplineWrap=_lat_pd.CubicSplineWrapGPU_responselisa,
            WDMSettingsWrap=_lat_pd.WDMSettingsWrapGPU,
            WDMDomainWrap=fastlisaresponse_backend_cuda11x.tdionthefly.WDMDomainWrapGPU,
            FDDomainWrap=_lat_pd.FDDomainWrapGPU,
            WaveletLookupTableWrap=fastlisaresponse_backend_cuda11x.tdionthefly.WaveletLookupTableWrapGPU,
            GBComputationGroupWrap=fastlisaresponse_backend_cuda11x.tdionthefly.GBComputationGroupWrapGPU,
            SOBBHComputationGroupWrap=fastlisaresponse_backend_cuda11x.tdionthefly.SOBBHComputationGroupWrapGPU,
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
            # Phase 3E: response classes now in lisatools_backend_cuda12x.pycppdetector.
            import lisatools_backend_cuda12x.pycppdetector as _lat_pd
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
            TDSplineTDIWaveformWrap=fastlisaresponse_backend_cuda12x.tdionthefly.TDSplineTDIWaveformWrapGPU,
            FDSplineTDIWaveformWrap=fastlisaresponse_backend_cuda12x.tdionthefly.FDSplineTDIWaveformWrapGPU,
            GBTDIonTheFlyWrap=fastlisaresponse_backend_cuda12x.tdionthefly.GBTDIonTheFlyWrapGPU,
            SOBBHTDIonTheFlyWrap=fastlisaresponse_backend_cuda12x.tdionthefly.SOBBHTDIonTheFlyWrapGPU,
            LISAResponseWrap=_lat_pd.LISAResponseWrapGPU,
            LISAResponse=_lat_pd.LISAResponseGPU,
            OrbitsWrap=_lat_pd.OrbitsWrapGPU_responselisa,
            TDIConfig=_lat_pd.TDIConfigGPU,
            TDIConfigWrap=_lat_pd.TDIConfigWrapGPU,
            CubicSplineWrap=_lat_pd.CubicSplineWrapGPU_responselisa,
            WDMSettingsWrap=_lat_pd.WDMSettingsWrapGPU,
            WDMDomainWrap=fastlisaresponse_backend_cuda12x.tdionthefly.WDMDomainWrapGPU,
            FDDomainWrap=_lat_pd.FDDomainWrapGPU,
            WaveletLookupTableWrap=fastlisaresponse_backend_cuda12x.tdionthefly.WaveletLookupTableWrapGPU,
            GBComputationGroupWrap=fastlisaresponse_backend_cuda12x.tdionthefly.GBComputationGroupWrapGPU,
            SOBBHComputationGroupWrap=fastlisaresponse_backend_cuda12x.tdionthefly.SOBBHComputationGroupWrapGPU,
            TDITypeDict=tmp,
            xp=cupy,
        )

class FastLISAResponseCuda13xBackend(Cuda13xBackend, FastLISAResponseBackend):
    """Implementation of CUDA 13.x backend"""
    _backend_name : str = "fastlisaresponse_backend_cuda13x"
    _name = "fastlisaresponse_cuda13x"
    
    def __init__(self, *args, **kwargs):
        Cuda13xBackend.__init__(self, *args, **kwargs)
        FastLISAResponseBackend.__init__(self, self.cuda13x_module_loader())
        
    @staticmethod
    def cuda13x_module_loader():
        try:
            # Phase 3E: response classes now in lisatools_backend_cuda13x.pycppdetector.
            import lisatools_backend_cuda13x.pycppdetector as _lat_pd

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda13x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda13x' backend requires cupy", pip_deps=["cupy-cuda13x"]
            ) from e
        return FastLISAResponseBackendMethods(
            LISAResponseWrap=_lat_pd.LISAResponseWrapGPU,
            LISAResponseBase=_lat_pd.LISAResponseBaseGPU,
            OrbitsWrap=_lat_pd.OrbitsWrapGPU_responselisa,
            xp=cupy,
        )
        
        
KNOWN_BACKENDS = {
    "cuda13x": FastLISAResponseCuda13xBackend,
    "cuda12x": FastLISAResponseCuda12xBackend,
    "cuda11x": FastLISAResponseCuda11xBackend,
    "cpu": FastLISAResponseCpuBackend,
}

# The JAX backend lives in ``fastlisaresponse.jax`` (one directory up)
# and is gated on ``import jax`` at module load time. Aggregating it
# in here keeps ``KNOWN_BACKENDS`` as the single registry users /
# config validators look at, even though the implementation lives
# outside ``cutils``.
try:
    from ..jax import FastLISAResponseJaxBackend as _FastLISAResponseJaxBackend
    if _FastLISAResponseJaxBackend is not None:
        KNOWN_BACKENDS["jax"] = _FastLISAResponseJaxBackend
except (ImportError, ModuleNotFoundError):
    pass

"""List of existing backends, per default order of preference."""
# TODO: __all__ ?


