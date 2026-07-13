# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## ⚠️ Deprecated — pure-Python husk (Phase 3L.7n, 2026-06-04)

`fastlisaresponse` has been fully deprecated. **Do not add code here.** All
functionality moved to `lisatools.response` / `lisatools.jax` / `gbgpu` /
`bbhx`; the entire C++ surface and the Python shim chain were deleted.
`import fastlisaresponse` now only emits a `DeprecationWarning` and exposes
`__version__` / `__version_tuple__` / `_is_editable` / `cutils` (a
retirement-notice stub). An editable install builds a ~6.4 KB pure-Python
wheel: `CMakeLists.txt` is `LANGUAGES NONE`, and there are **zero**
`.cu`/`.cxx`/`.hpp`/`.hh`/`.pyx` files under `src/`.

**Phase 3L.7o** (one release cycle out) deletes the `lisa-on-gpu/` directory
entirely.

## Where the old API went

| Old (`fastlisaresponse.*`) | New home |
|---|---|
| `response` (`pyResponseTDI`, `ResponseWrapper`, `ecliptic_to_icrs`) | `lisatools.response.directresponse` |
| `tdionfly`, `tdiconfig`, `utils.parallelbase` | `lisatools.response.{tdionfly,tdiconfig,parallelbase}` |
| `jax.{base,projection,tdi_config,amp_phase_extract}`, `jax.wdm.*` | `lisatools.jax.response.*`, `lisatools.jax.wdm.*` |
| `cutils` C++ (`LISAResponse`, `TDIonTheFly`, WDM/FD/spline, bindings) | `lisatools/cutils/` (GB parts → `gbgpu/cutils/`, SOBBH → `bbhx/cutils/`) |
| `jax.sources.ucb`, `jax.wdm.{kernels,heterodyne_kernels,fast_inner_heterodyne}` | `gbgpu.jax.{sources,wdm}.*` |
| `jax.sources.sobbh` | `bbhx.jax.sources.sobbh` |

## What still lives here (until 3L.7o)

`src/fastlisaresponse/__init__.py` (the DeprecationWarning),
`cutils/__init__.py` (retirement-notice docstring), `{_version,_editable}.py`
packaging hooks, `CMakeLists.txt` (`LANGUAGES NONE`) + `pyproject.toml`, and
historical `tests/` / `README.md` / `examples/` / `docs/` kept for reference.

## LISA Analysis Tools–wide rules

This repo has **no native code**, so the C++/CUDA conventions shared across
LISA Analysis Tools (backend-implementation hierarchy, no-backend-strings,
host→device wrapper upload, CPU/GPU class-name aliasing, deepcopy/pickle
safety) do not apply here — they live in
[`../LISAanalysistools/docs/conventions.md`](../LISAanalysistools/docs/conventions.md)
(canonical) and govern the repos that still own native code
(`LISAanalysistools`, `GBGPU`, `BBHx`, `GPUBackendTools`,
`FastEMRIWaveforms`). See
[`../LISAanalysistools/docs/architecture-map.md`](../LISAanalysistools/docs/architecture-map.md)
for the cross-repo map.
