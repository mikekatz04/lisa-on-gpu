"""Retired at Phase 3L.7k (2026-06-04).

The ``fastlisaresponse_<flavor>`` backend family that used to live here
has been deleted. Its responsibilities split into three places:

* LISA-response Wraps (Orbits, TDIConfig, WDM/FD/Spline, LISAResponseWrap,
  TDITypeDict) -> ``lisatools.cutils.LISAToolsBackend``
  (lisatools.get_backend("cpu") etc.). LAT's ``pycppdetector`` module
  now exposes ``TDI_XYZ`` / ``TDI_AET`` / ``TDI_AE`` directly.
* GB-specific Wraps (GBTDIonTheFlyWrap, GBComputationGroupWrap,
  GBGPUComputationWrap + the sharedmem accessors) -> ``gbgpu.cutils.GBGPUBackend``
  (``gbgpu.get_backend("cpu")``). Composes LAT-side automatically.
* SOBBH + BBH-specific Wraps (SOBBHTDIonTheFlyWrap, SOBBHComputationGroupWrap,
  BBHxComputationWrap + bound methods) -> ``bbhx.cutils.BBHxBackend``
  (``bbhx.get_backend("cpu")``). Composes LAT-side automatically.

Existing code that hit ``fastlisaresponse.get_backend(...)`` will get
``AttributeError`` and should migrate to one of the three above. Code
that hit ``self.backend.X`` from a ``FastLISAResponseParallelModule``
descendant keeps working: the LAT class now resolves ``force_backend``
strings to ``lisatools_<flavor>`` (which carries the LISA-response
Wraps directly).
"""
