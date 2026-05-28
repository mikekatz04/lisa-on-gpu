# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Backend implementation hierarchy (sprint-wide rule)

When implementing or modifying an algorithm that exists across multiple
backends (GPU C++ / CPU C++ / JAX), follow this hierarchy:

1. **GPU C++ (CUDA) leads.** This is the canonical performance target
   and reference implementation. New algorithms and optimizations are
   designed for the GPU first; CPU and JAX paths follow.

2. **CPU C++ mirrors GPU C++ as closely as possible.** Same kernel
   structure, same algorithm, same data flow — use `#ifdef __CUDACC__`
   or shared compile-time macros (`CUDA_SHARED`, `THREAD_START_X`,
   `BLOCK_INCR_X`, …) to bridge platform differences. The CPU path
   exists primarily for testing and CPU-only environments; it must
   not diverge in algorithm or output beyond floating-point order of
   operations.

3. **CPU C++ must reproduce the overall lisatools computation.**
   Against the lisatools reference (e.g. `FDSignal.transform`,
   `TDSignal.transform`, `XYZ2SensitivityMatrix`), match to machine
   precision (≤ 1e-15 mismatch) in direct modes; cache/approximation
   modes have documented per-feature error budgets.

4. **JAX may diverge internally** — design it to be JAX-efficient.
   JAX-CPU and JAX-GPU compilation targets may even differ. Use
   JAX-native idioms (`jax.lax.scan`, `jax.vmap`, static-shape
   `dynamic_slice` + masks, functional carries) rather than
   mechanically translating CUDA shared memory / register caches.

5. **JAX must match C++ inner-product outputs.** End-to-end
   likelihood quantities (`<d|h>`, `<h|h>`, swap_ll 5 terms) must
   match the C++ to floating-point precision (reldiff ≲ 1e-12) on
   representative test cases. Intermediate quantities (raw templates,
   per-chunk WDM coefficients) may differ at FP precision due to
   summation order — validate at the inner-product level.

**Workflow for a new feature.** GPU C++ → CPU C++ via `#ifdef` → JAX
with JAX-native idioms → cross-backend inner-product validation.


## No backend strings as function kwargs (sprint-wide rule)

Backend selection MUST happen at instantiation, not in method
signatures. Subclass :class:`FastLISAResponseParallelModule` (or
equivalent ParallelModuleBase descendant) with ``force_backend=...``;
all methods dispatch via ``self.backend`` / ``self.backend.xp`` /
``self.backend.name``.

- **Allowed:** ``GBWDMHeterodyne(force_backend="cpu")``,
  ``GBFDComputations(force_backend="jax")``.
- **Forbidden in method signatures:** ``backend="jax"``,
  ``backend="cpp"``, ``use_cpp=True``, ``use_jax=True``.
- Method names MAY carry a backend suffix (e.g. ``get_ll_grad_jax``)
  when the implementation is intrinsically tied to that backend, but
  the caller picks which method to call -- not a runtime kwarg.

One instance = one backend keeps ``xp`` arrays, kernels, and dispatch
consistent. Cross-backend usage (e.g. evaluating gradients on a CPU
instance via a separate JAX setup) belongs to a separate instance,
not a shared method-level flag.


## Host→device upload of class-wrapper objects (sprint-wide rule)

Pybind11 wrapper classes in this codebase (``OrbitsWrap_responselisa``,
``TDIConfigWrap``, ``WDMSettingsWrap``, ``WDMDomainWrap``,
``FDDomainWrap``, ``AnalysisContainerArrayWrap``, …) store their
underlying C++ instance via plain ``new`` on the **host** heap, e.g.

```cpp
class OrbitsWrap_responselisa : public ReturnPointerBase {
    Orbits *orbits;
    OrbitsWrap_responselisa(...) {
        orbits = new Orbits(..., _ltt_arr_device_ptr, ...);
        //       ^^^^^^^^^^ host allocation; pointer fields inside
        //                  may already point to device memory.
    }
};
```

The pointer fields inside the struct (``Orbits::ltt_arr``,
``WDMDomain::wdm_data``, ``TDIConfig::unit_starts``) are device
pointers extracted from cupy arrays via
``return_pointer_and_check_length``. But **the struct itself lives on
the host**.

A CUDA kernel parameter of type ``Orbits *`` therefore cannot be the
host pointer ``orbits_wrap->orbits`` directly. Dereferencing it from
device code (``orbits->ltt_t0``) reads garbage and triggers an illegal
memory access -- typically with a faulting address in the canonical
Linux PIE/heap range (``0x55555...``) and a sanitizer message of the
form "X bytes after the nearest allocation" with a wildly OOB delta
(tens of TB). That delta is **not** an off-by-one; it means the device
dereferenced a host address.

The required upload pattern (mirrors
``LISAResponse.cu:419-433``):

```cpp
#ifdef __CUDACC__
    Orbits *orbits_gpu = nullptr;
    gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(Orbits)));
    gpuErrchk(cudaMemcpy(orbits_gpu, orbits, sizeof(Orbits),
                         cudaMemcpyHostToDevice));

    TDIConfig *tdi_config_gpu = nullptr;
    gpuErrchk(cudaMalloc(&tdi_config_gpu, sizeof(TDIConfig)));
    gpuErrchk(cudaMemcpy(tdi_config_gpu, tdi_config, sizeof(TDIConfig),
                         cudaMemcpyHostToDevice));

    // ...repeat for every host-side wrapper struct accessed on device:
    //    WDMSettings, WDMDomain, FDDomain, etc.

    my_kernel<<<...>>>(orbits_gpu, tdi_config_gpu, ...);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(orbits_gpu));
    gpuErrchk(cudaFree(tdi_config_gpu));
#else
    // CPU branch keeps the host pointers unchanged.
    my_kernel(orbits, tdi_config, ...);
#endif
```

Rules:

1. **Every** struct constructed via ``new`` on the host that the kernel
   dereferences (i.e. reads scalar fields or pointer fields off of
   ``this``) must be copied to device with ``cudaMalloc`` +
   ``cudaMemcpy(..., cudaMemcpyHostToDevice)`` before the kernel
   launch.
2. The device-side pointer fields *inside* the uploaded struct survive
   the shallow copy; do **not** also try to upload those.
3. Free the device-side struct copies after the kernel sync, before
   returning.
4. The CPU branch (``#else``) does not copy -- it passes the host
   pointer directly into the (host-compiled) kernel.
5. This applies to every CUDA wrapper across the sprint tree --
   existing legacy kernels in ``LISAResponse.cu`` and ``Detector.cu``
   already follow it; new chunked-het / chunked-FD / WDM impl
   wrappers must do the same.

When debugging an IMA whose faulting address starts with
``0x55555...`` and whose "nearest allocation" delta is in the TB
range, the first hypothesis should be a missing wrapper upload --
not an indexing bug in the kernel.


## CPU/GPU class-name aliasing (sprint-wide rule)

Every C++ class that is compiled into **both** the CPU and the GPU
shared object (one per backend wheel) MUST have a per-backend
``#define`` alias at the top of the header that declares it, so the
two builds emit **distinct C++ type names** for the same logical
class. This applies to **two** layers:

**(a) The pybind11 wrapper classes** -- anything passed to
``py::class_<...>(m, "...")``. Block lives at the top of
``src/fastlisaresponse/cutils/binding_tof.hpp``:

```cpp
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
#define GBTDIonTheFlyWrap         GBTDIonTheFlyWrapGPU
#define SOBBHTDIonTheFlyWrap      SOBBHTDIonTheFlyWrapGPU
#define FDSplineTDIWaveformWrap   FDSplineTDIWaveformWrapGPU
#define TDSplineTDIWaveformWrap   TDSplineTDIWaveformWrapGPU
#define WaveletLookupTableWrap    WaveletLookupTableWrapGPU
#define WDMSettingsWrap           WDMSettingsWrapGPU
#define WDMDomainWrap             WDMDomainWrapGPU
#define FDDomainWrap              FDDomainWrapGPU
#define GBComputationGroupWrap    GBComputationGroupWrapGPU
#define SOBBHComputationGroupWrap SOBBHComputationGroupWrapGPU
#else
// ...CPU suffixes...
#endif
```

**(b) The underlying C++ classes** that the wrappers hold a pointer
to -- ``Orbits``, ``WDMSettings``, ``WDMDomain``, ``TDIConfig``,
``GBTDIonTheFly``, ``GBComputationGroup``, etc. Same block pattern,
at the top of each header that declares them:

- ``Detector.hpp`` (in LISAanalysistools) aliases ``Orbits`` →
  ``OrbitsGPU`` / ``OrbitsCPU``.
- ``src/fastlisaresponse/cutils/TDIonTheFly.hh`` aliases
  ``GBTDIonTheFly``, ``SOBBHTDIonTheFly``, ``FDSplineTDIWaveform``,
  ``TDSplineTDIWaveform``, ``WaveletLookupTable``, ``WDMSettings``,
  ``WDMDomain``, ``FDDomain``, ``GBComputationGroup``
  (``SOBBHComputationGroup`` should be added when next touched).

After preprocessing, the GPU build defines ``class WDMSettingsGPU``
and the CPU build defines ``class WDMSettingsCPU``: distinct C++
types with distinct ``typeid``s and distinct mangled symbol names.

Rules:

1. **Every class -- wrapper or underlying -- that ends up in both
   shared objects must appear in the relevant header's ``#define``
   block, with both GPU and CPU branches.** When you add a new
   class to a backend-shared header, add it to the block in the same
   commit.
2. **Both branches of the ``#if/#else`` must have the same set of
   entries.** A missing CPU- or GPU-branch entry (e.g. an alias
   present only on the GPU side) silently produces a backend-asymmetric
   class name, which is exactly the situation the rule prevents.
3. The pybind11 registration line ``py::class_<FooWrap>(m, "FooWrapGPU"
   / "FooWrapCPU")`` in the ``.cxx`` binding source must be guarded by
   the same ``#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)``
   toggle so the Python-visible name tracks the C++ alias.
4. **Inheritance only works through the alias if both the base and
   derived class names are in the ``#define`` block.** Example:
   ``class WDMDomain : public WDMSettings`` works correctly only when
   *both* ``WDMSettings`` and ``WDMDomain`` are aliased; otherwise the
   GPU build links ``WDMDomainGPU`` against the (still-unaliased)
   ``WDMSettings``, while the CPU build links ``WDMDomainCPU`` against
   the same ``WDMSettings`` -- the base type collides across the two
   shared objects even though the derived names differ.
5. Plain helper structs that never escape a single translation unit
   (e.g. a file-static ``OrbitsSplineCache``) do NOT need aliasing --
   only types whose symbols end up in the .so's exported interface
   (held by wrappers, referenced by pybind11, instantiated by
   templates exported from the shared object) need it.

**Rationale: ensures we do not duplicate imported symbols across the
CPU and GPU imports.** Both backends ship as separate plugin wheels
(``lisaanalysistools-cuda12x``, ``-cpu``, …) that load into the same
Python interpreter. If both shared objects declare ``class
WDMSettings``, they emit the same mangled C++ symbols and the same
``typeid``. Effects:

- pybind11's global type registry, keyed by ``typeid``, sees a
  collision: the second registration is rejected or silently shadows
  the first.
- The dynamic linker may resolve one shared object's call to
  ``WDMSettings::method`` against the *other* shared object's vtable,
  producing wrong-arch device calls or stack corruption.
- Inheritance edges registered with pybind11 reference the wrong
  base typeid, breaking ``isinstance`` / downcasts at the Python
  layer.

Aliasing forces every backend-specific symbol to be distinct end to
end, so the CPU and GPU plugin wheels are ABI-independent and
side-loadable in the same process. This is what allows
``has_backend("cpu")`` and ``has_backend("cuda12x")`` to both be true
simultaneously.
