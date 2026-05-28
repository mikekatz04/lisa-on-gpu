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
   or shared compile-time macros (`CUDA_SHARED`, `THREAD_START`,
   `BLOCK_INCR`, …) to bridge platform differences. The CPU path
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
