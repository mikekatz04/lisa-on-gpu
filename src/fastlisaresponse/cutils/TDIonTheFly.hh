#ifndef __TDI_ON_THE_FLY_HH__
#define __TDI_ON_THE_FLY_HH__

#include "Detector.hpp"
#include "Interpolate.hh"
#include "LISAResponse.hh"
#include "gbt_global.h"
// Phase 3L (2026-06-02..03): FDDomain + WDMSettings + WDMDomain +
// LISATDIonTheFly (base class) + OrbitsSplineCache moved to LAT. Their
// class definitions + CPU/GPU aliases now live in LAT headers
// (fd_domain.hh, wdm_settings.hh, wdm_domain.hh, lat_tdi_on_the_fly.hh).
// WaveletLookupTable still inherits from WDMSettings; the derived classes
// below (GBTDIonTheFly, SOBBHTDIonTheFly, FDSplineTDIWaveform,
// TDSplineTDIWaveform) inherit from LISATDIonTheFly via the LAT include.
#include "fd_domain.hh"
#include "wdm_settings.hh"
#include "wdm_domain.hh"
#include "lat_tdi_on_the_fly.hh"
#include "lat_spline_tdi_waveform.hh"
// Phase 3L.7c (2026-06-04): GBTDIonTheFly class declaration + its
// aliasing macros + gb_run_wave_tdi_wrap / gb_run_fd_wave_tdi_wrap
// function decls now live in GBGPU's gb_tdi_on_the_fly.hh. The
// include path for ${GBGPU_CUTILS} is wired into the four
// fastlisaresponse_{cpu,gpu}_tdionthefly{,_static} CMake targets so
// the include below resolves. GBTDIonTheFly method bodies +
// gb_run_wave_tdi kernel/wrap + gb_run_fd_wave_tdi kernel/wrap +
// FD helpers (gbfd_*) all STILL live in this repo's TDIonTheFly.cu
// for now -- subsequent Phase 3L.7 slices move them to GBGPU.
#include "gb_tdi_on_the_fly.hh"


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
// GBTDIonTheFly + GBComputationGroup aliasing moved to GBGPU's
// gb_tdi_on_the_fly.hh (Phase 3L.7c + 3L.7f.1, 2026-06-04). Included
// transitively above.
#define SOBBHTDIonTheFly SOBBHTDIonTheFlyGPU
// FDSplineTDIWaveform + TDSplineTDIWaveform aliases moved to LAT at Phase 3L.6.
// WaveletLookupTable disabled at Phase 3L (2026-06-02) -- lookup-table spline path is being retired.
#else
// GBTDIonTheFly + GBComputationGroup aliasing moved to GBGPU's
// gb_tdi_on_the_fly.hh (Phase 3L.7c + 3L.7f.1, 2026-06-04). Included
// transitively above.
#define SOBBHTDIonTheFly SOBBHTDIonTheFlyCPU
// FDSplineTDIWaveform + TDSplineTDIWaveform aliases moved to LAT at Phase 3L.6.
// WaveletLookupTable disabled at Phase 3L (2026-06-02) -- lookup-table spline path is being retired.
#endif

#define TDI_XYZ 1
#define TDI_AET 2
#define TDI_AE 3


// OrbitsSplineCache struct moved to LAT at Phase 3L.5 (2026-06-03).
// Definition lives in lisatools/cutils/lat_tdi_on_the_fly.hh; included
// at the top of this header.

// LISATDIonTheFly base class moved to LAT at Phase 3L.5 (2026-06-03).
// Class definition + method declarations + CPU/GPU alias for
// LISATDIonTheFly + OrbitsSplineCache live in
// lisatools/cutils/lat_tdi_on_the_fly.hh; method bodies are compiled
// out of lisatools/cutils/lat_tdi_on_the_fly.cu (copy-compiled
// in-place by this repo's CMakeLists, same pattern as LISAResponse.cu).
//
// Stub kept ONLY so the historical class declaration is the canonical
// reference for change-history purposes; the actual class lives in LAT.
#if 0
class LISATDIonTheFly{
    public:
        Orbits *orbits;
        TDIConfig *tdi_config;
        int inc_index;
        int psi_index;
        int lam_index;
        int beta_index;
        int N_store;

        CUDA_DEVICE
        void run_wave_tdi(
            void *buffer, int buffer_length, cmplx *tdi_channels_arr,
            double *tdi_amp, double *tdi_phase, double *phi_ref,
            double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
        );
        CUDA_CALLABLE_MEMBER
        LISATDIonTheFly(Orbits *orbits_, TDIConfig *tdi_config_, int inc_index_, int psi_index_, int lam_index_, int beta_index_)
        {
            orbits = orbits_;
            tdi_config = tdi_config_;
            inc_index = inc_index_;
            psi_index = psi_index_;
            lam_index = lam_index_;
            beta_index = beta_index_;
        };
        CUDA_CALLABLE_MEMBER
        ~LISATDIonTheFly();
        CUDA_DEVICE
        void print_orbits_tdi();
        CUDA_DEVICE
        void fill_link_arrays(int *link_space_craft_rec, int *link_space_craft_em);
        // CUDA_DEVICE
        // void LISA_polarization_tensor(double costh, double phi, double *eplus, double *ecross, double *k);
        CUDA_DEVICE
        void get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels);

        // Heterodyned-phi_ref variant of get_tdi. Returns the same
        // (tdi_amp, tdi_phase) outputs but with phi_ref replaced by
        // ``phi_ref_het[i] = phi_ref[i] - 2*pi*f0_grid*t_arr[i]``, where
        // ``f0_grid`` is the snapped chunk-rfft carrier. The residual is
        // slow over the chunk (O(1 rad)) so it can be unwrapped/splined
        // robustly even at coarse N -- this is the prerequisite for the
        // within-chunk source-signal spline cache.
        //
        // Only meaningful for sources with a well-defined constant
        // carrier (GB / SOBBH). Sources without a constant ``f0``
        // (MBHB chirp, EMRI) should call ``get_tdi`` instead -- carrier
        // subtraction is undefined there.
        CUDA_DEVICE
        void get_tdi_heterodyned(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref_het, double *params, double *t_arr, int N, int bin_i, int nchannels, double f0_grid);

        // ---- Orbit-cache variants ------------------------------------
        // Same outputs as their non-cached siblings, but the orbit
        // lookups (``orbits->get_pos`` and ``orbits->get_light_travel_time``)
        // are redirected to ``cache`` -- a cubic-spline cache built once
        // per chunk per block via ``populate_orbit_spline_cache``. Use
        // these when the chunk + the source's get_tdi inner-loop delays
        // are all within the cache window (true by construction in the
        // chunked-het kernels).
        //
        // Window checks against the orbits' raw global tables are
        // skipped -- by construction the cache is built from t-values
        // inside the source's valid orbit window, so any t inside the
        // chunk is in-bounds.
        CUDA_DEVICE
        void get_tdi_Xf_single_cached(cmplx *tdi_channel, double t, double *params, Vec k, Vec u, Vec v, int *link_Space_craft_rec, int *link_Space_craft_em, int bin_i, OrbitsSplineCache *cache);

        CUDA_DEVICE
        void get_tdi_Xf_cached(cmplx *tdi_channels_arr, double *params, double *t_data, int N, int bin_i, int *link_Space_craft_rec, int *link_Space_craft_em, Vec k, Vec u, Vec v, OrbitsSplineCache *cache);

        CUDA_DEVICE
        void get_tdi_cached(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double *phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels, OrbitsSplineCache *cache);

        CUDA_DEVICE
        void get_tdi_heterodyned_cached(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double *phi_ref_het, double *params, double *t_arr, int N, int bin_i, int nchannels, double f0_grid, OrbitsSplineCache *cache);

        // Raw variants of get_tdi[_cached]: fill tdi_channels_arr
        // (nchannels * N raw complex samples) and phi_ref (N, UN-heterodyned
        // -- get_phase_ref(t_i) straight from the source, NO carrier
        // subtraction), but skip the per-channel amplitude/phase extract
        // + unwrap.
        //
        // The carrier subtraction MUST happen on the caller side, AFTER
        // any extract that wants the OLD get_tdi convention. The inner
        // ``new_extract_amplitude_and_phase`` consumes phiR via
        // ``remainder(phiR, 2*pi)``, which is NOT invariant under shifts
        // by 2*pi*f0*t (the carrier offset isn't a multiple of 2*pi), so
        // passing a heterodyne-subtracted phi_ref here would change the
        // unwrapping decision and the resulting Dphi by a per-sample
        // amount that does NOT cancel against the downstream
        // ``dphi_ref + phi0_chunk`` term -- it would offset the
        // slow-signal phase off the chunk-FFT grid.
        //
        // Used by the chunked-het spline path so the caller can perform
        // extract + unwrap one channel at a time into single-channel
        // coefficient buffers (~6 KB / kernel shared-mem reduction).
        CUDA_DEVICE
        void get_tdi_raw(cmplx *tdi_channels_arr, double *phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels);
        CUDA_DEVICE
        void get_tdi_raw_cached(cmplx *tdi_channels_arr, double *phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels, OrbitsSplineCache *cache);
        // CUDA_DEVICE
        // void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i);
        // CUDA_DEVICE
        // void get_tdi_sub(cmplx *M, int n, int N, int a, int b, int c, double t, double* tarray, double *amp_tdi_vals, double *phase_tdi_vals, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm);
        // CUDA_DEVICE
        // void get_tdi_n(cmplx *X, cmplx *Y, cmplx *Z, double* phi_ref, double *params, double t, int m, int N, double costh, double phi, double cosi, double psi, int bin_i);
        // CUDA_DEVICE
        // void get_t_tdi(double *t_out, double *kr, double *Larm, double t, int a, int b, int c, int n);
        CUDA_DEVICE
        void get_tdi_Xf(cmplx *tdi_channels_arr, double *params, double *t_data, int N, int bin_i, int *link_space_craft_rec, int *link_space_craft_em, Vec k, Vec u, Vec v);
        CUDA_DEVICE
        void get_tdi_Xf_single(cmplx *tdi_channel, double t, double *params, Vec k, Vec u, Vec v, int *link_space_craft_rec, int *link_space_craft_em, int bin_i);
        CUDA_DEVICE
        void extract_amplitude_and_phase(double *flip, double *pjump, int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR);
        CUDA_DEVICE
        void new_extract_amplitude_and_phase(int *count, bool *fix_count, double *flip, double *pjump, int Ns, double *As, double *Dphi, cmplx *M, double *phiR);
        CUDA_CALLABLE_MEMBER
	int get_tdi_buffer_size(int N);
        CUDA_DEVICE
        void unwrap_phase(int N, double *phase);
        CUDA_DEVICE
        void new_unwrap_phase(double *ph_correct_buffer, int N, double *phase);
        CUDA_DEVICE
        void new_extract_phase(cmplx *M, double *phiR, int N, double *t_arr);
        CUDA_DEVICE
        double get_phase_ref(double t, double *params, int bin_i);
        CUDA_DEVICE
        void get_hp_hc(double *hp, double *hc, double t, double *params, double phase_change, int bin_i);
        CUDA_DEVICE
        void get_sky_vectors(Vec *k, Vec *u, Vec *v, double *params);
        CUDA_DEVICE
        void xi_projections(double *xi_p, double *xi_c, Vec u, Vec v, Vec n);
        CUDA_DEVICE
        virtual double get_amp(double t, double *params, int bin_i);
        CUDA_DEVICE
        virtual double get_phase(double t, double *params, int bin_i);
        CUDA_DEVICE
        virtual double get_f(double t, double *params, int bin_i);
        CUDA_DEVICE
        virtual double get_fdot(double t, double *params, int bin_i);
};
#endif // === end LISATDIonTheFly stub (moved to LAT) ===

// `class GBTDIonTheFly : public LISATDIonTheFly` moved to GBGPU at Phase
// 3L.7c (2026-06-04). Declaration now lives in
// `gbgpu/src/gbgpu/cutils/gb_tdi_on_the_fly.hh`, consumed transitively
// via the `#include "gb_tdi_on_the_fly.hh"` near the top of this file.
//
// `gb_run_wave_tdi_wrap` + `gb_run_fd_wave_tdi_wrap` function
// declarations also moved to that header. Method bodies (ucb_*,
// get_*, dtor, get_gb_buffer_size, get_gb_fd_buffer_size) + kernel
// launchers (gb_run_wave_tdi_kernel, gb_run_fd_wave_tdi_kernel) +
// FD helpers (gbfd_*) all STILL live in this repo's TDIonTheFly.cu
// pending the next Phase 3L.7 slice.


// Stellar-origin black-hole binary TDI-on-the-fly. Mirrors GBTDIonTheFly:
// per-time-sample (amplitude, phase) device methods feed into the shared
// LISATDIonTheFly::get_tdi -> get_hp_hc projection. The intrinsic post-Newtonian
// content (intrinsic_quantities, phase_fn, time_to_merger_fn, tau_to_x_fn) is
// ported verbatim from the prototype sobbh_intrinsic_Ladeeda.cpp; only style
// (CUDA decorators, no std:: qualifiers) has been adapted.
//
// Parameter layout (n_params = 11):
//   0: m1            (solar masses)
//   1: m2            (solar masses)
//   2: s1            (dimensionless spin component, primary)
//   3: s2            (dimensionless spin component, secondary)
//   4: distance      (parsecs)
//   5: f_low         (Hz, GW frequency at t_ref)
//   6: phi_c         (rad, reference orbital phase)
//   7: inc           (rad, inclination)
//   8: psi           (rad, polarization)
//   9: lam           (rad, ecliptic longitude)
//  10: beta          (rad, ecliptic latitude)
class SOBBHTDIonTheFly : public LISATDIonTheFly{
    public:
        double T;
        double t_ref;
        int m1_index;
        int m2_index;
        int s1_index;
        int s2_index;
        int distance_index;
        int f_low_index;
        int phi_c_index;
        // f0_index is an ALIAS for f_low_index so source-class-agnostic
        // kernels (e.g. fast_wdm_inner_heterodyne) can read
        // ``src->f0_index`` uniformly across GB and SOBBH variants.
        int f0_index;

        CUDA_CALLABLE_MEMBER
        SOBBHTDIonTheFly(Orbits *orbits_, TDIConfig *tdi_config_, double T_, double t_ref_) : LISATDIonTheFly(orbits_, tdi_config_, 7, 8, 9, 10)
        {
            T = T_;
            t_ref = t_ref_;
            m1_index = 0;
            m2_index = 1;
            s1_index = 2;
            s2_index = 3;
            distance_index = 4;
            f_low_index = 5;
            phi_c_index = 6;
            f0_index = 5;          // alias of f_low_index for unified kernels
        };
        CUDA_CALLABLE_MEMBER
        ~SOBBHTDIonTheFly();
        // Intrinsic-quantity helpers (PN expansions ported from sobbh_intrinsic_Ladeeda.cpp).
        CUDA_DEVICE
        double sobbh_phase_fn(double x, double sigma, double delta, double eta, double s);
        CUDA_DEVICE
        double sobbh_time_to_merger_fn(double x, double sigma, double delta, double eta, double s);
        CUDA_DEVICE
        double sobbh_tau_to_x_fn(double tau, double sigma, double delta, double eta, double s);
        // Per-sample on-the-fly evaluators. amplitude/phase are GW-quadrupole
        // conventions: phase = 2 * (phi_c - phase_fn(x)), amp = 2 M eta x / D
        // (positive); get_hp_hc folds in -cos / -sin and the (1+cos^2 iota),
        // 2 cos(iota) factors.
        CUDA_DEVICE
        double sobbh_amplitude(double t, double *params);
        CUDA_DEVICE
        double sobbh_phase(double t, double *params);
        CUDA_DEVICE
        double sobbh_f(double t, double *params);
        CUDA_DEVICE
        double sobbh_fdot(double t, double *params);
        CUDA_CALLABLE_MEMBER
	int get_sobbh_buffer_size(int N);
        CUDA_DEVICE
        double get_amp(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_phase(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_f(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_fdot(double t, double *params, int bin_i);
};

void sobbh_run_wave_tdi_wrap(SOBBHTDIonTheFly *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);


// TDSplineTDIWaveform + FDSplineTDIWaveform classes moved to LAT at
// Phase 3L.6 (2026-06-03). Class definitions + CPU/GPU aliases live in
// lisatools/cutils/lat_spline_tdi_waveform.hh; method bodies + host
// launchers (td_spline_run_wave_tdi_wrap, fd_spline_run_wave_tdi_wrap)
// compile out of lat_spline_tdi_waveform.cu (copy-compiled in-place by
// this repo's CMakeLists).

// WDMSettings class moved to LAT at Phase 3L (2026-06-02). Definition
// lives in lisatools/cutils/wdm_settings.hh; included at the top of
// this header.

// WDMDomain class moved to LAT at Phase 3L (2026-06-02). Definition
// lives in lisatools/cutils/wdm_domain.hh; included at the top of
// this header. All 12 method bodies are header-inline there (no
// out-of-line .cu bodies remain).


// ---------------------------------------------------------------------------
// FDDomain  -- mirror of WDMDomain for the heterodyne FD path.
//
// Holds the frequency-domain data array (complex, length n_rfft per channel,
// num_data instances) and an inverse-covariance array invC (3x3 cross-channel
// for tdi_type == TDI_XYZ; diagonal for TDI_AET / TDI_AE).  The convention is
// the lisatools rfft grid (df = 1/Tobs), with the active band specified by
// [ind_min, ind_max] inclusive.  Inner products are the standard lisatools
// formula  (a|b) = 4 Re sum_{c1,c2} sum_k conj(a_c1[k]) b_c2[k] invC[c1,c2][k] * df.
// FDDomain class moved to LAT at Phase 3L (2026-06-02). Definition lives
// in lisatools/cutils/fd_domain.hh; included at the top of this header.


// Lookup table kind. Selects how linear_interp indexes the coefficient buffer
// and which extra sign corrections get_w_mn_lookup applies.
//   PER_N      — legacy. Table is (Nt, num_fdot, num_f); layer_n offsets into
//                the time axis. No extra dn-sign needed (the per-n parity
//                swap is baked into each n slice at build time).
//   N_REF_ONLY — Plan A. Table is (num_fdot, num_f) — only the (m_ref, n_ref)
//                pixel was extracted at build time. layer_n is unused for
//                indexing; eval applies (-1)^(layer_n - n_ref) to translate
//                from the built n_ref pixel to the desired layer_n.
enum LookupKind : int { LOOKUP_PER_N = 0, LOOKUP_N_REF_ONLY = 1 };

#if 0  // === WaveletLookupTable disabled at Phase 3L (2026-06-02) -- lookup-table spline path retiring ===
class WaveletLookupTable : public WDMSettings{
  public:
    double *c_nm_all;
    double *s_nm_all;

    int num_f;
    int num_fdot;
    double df_interp;
    double dfdot_interp;
    double min_f_scaled;
    double min_fdot;
    // Build-time reference frequency layer (the layer the lookup table was
    // built at). At lookup time, sources whose central layer has the
    // OPPOSITE parity from m_ref pick up an overall (-1) on the WDM
    // coefficient — see the parity-correction comment in get_w_mn_lookup.
    int m_ref;
    // Build-time reference time pixel. Only consulted for kind==N_REF_ONLY
    // (the dn-sign correction). Pass 0 for PER_N.
    int n_ref;
    // Dispatch flag: see LookupKind. Stored as int for CUDA portability.
    int kind;

    CUDA_CALLABLE_MEMBER
    WaveletLookupTable(double *c_nm_all_, double *s_nm_all_, int num_f_, int num_fdot_, double df_interp_, double dfdot_interp_, double min_f_scaled_, double min_fdot_,
        double layer_df_, double layer_dt_, int Nf_, int Nt_, int num_channel_, int ind_min_t_, int ind_max_t_, int ind_min_f_, int ind_max_f_, int m_ref_, int n_ref_, int kind_): WDMSettings(layer_df_, layer_dt_, Nf_, Nt_, num_channel_, ind_min_t_, ind_max_t_, ind_min_f_, ind_max_f_) {
        // n * Nf + m
        c_nm_all = c_nm_all_;
        s_nm_all = s_nm_all_;
        num_f = num_f_;
        num_fdot = num_fdot_;
        df_interp = df_interp_;
        dfdot_interp = dfdot_interp_;
        min_f_scaled = min_f_scaled_;
        min_fdot = min_fdot_;
        m_ref = m_ref_;
        n_ref = n_ref_;
        kind = kind_;
    };
    CUDA_DEVICE
    double get_wdm_in_channel_over_layers(cmplx tdi_channel_val, double f, double fdot, int m, int n);
    CUDA_DEVICE
    double linear_interp(double f_scaled, double fdot, double *z_vals, int layer_n);
    CUDA_DEVICE
    double get_w_mn_lookup(cmplx tdi_channel_val, double f, double fdot, int layer_m, int layer_n);
};
#endif  // === end WaveletLookupTable disabled ===

void fd_spline_run_wave_tdi_wrap(FDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);



// Parallel API for SOBBH sources. Mirrors GBComputationGroup's chunked-
// heterodyne family with `sobbh_` prefixes. Both classes route through the
// templated `wdm_het_*_kernel<SourceT>` so 95% of the C++ infrastructure is
// shared -- only the per-block source-class construction
// (`GBTDIonTheFly` vs `SOBBHTDIonTheFly`) differs.
//
// The pre-existing per-pixel-lookup path (gb_wdm_fill_global / get_ll /
// swap_ll) is GB-only and has no SOBBH equivalent here; for SOBBH the
// chunked-heterodyne path IS the canonical entry point.
class SOBBHComputationGroup{
  public:
    void sobbh_wdm_het_fill_global_wrap(
        double *template_fill,
        Orbits *orbits, TDIConfig *tdi_config,
        WDMSettings *wdm_settings,
        double *params_all, double *factors_all,
        double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
        int *chunk_n_global_offset,
        double *wdm_window,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref,
        double tukey_alpha,
        int grid_dim, int N_cp_sig, int N_cp_orbit);

    void sobbh_wdm_het_get_ll_wrap(
        double *d_h_out, double *h_h_out,
        Orbits *orbits, TDIConfig *tdi_config,
        WDMSettings *wdm_settings,
        double *params_all,
        int *data_index_all, int *noise_index_all,
        double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
        int *chunk_n_global_offset,
        double *wdm_window,
        double *data_d, double *invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha,
        int grid_dim, int N_cp_sig, int N_cp_orbit,
        int *binary_perm, int *group_starts, int *group_ends,
        int *group_m_lo, int *group_m_hi, int n_groups);

    void sobbh_wdm_het_swap_ll_wrap(
        double *d_h_add_out, double *d_h_remove_out,
        double *add_add_out, double *remove_remove_out, double *add_remove_out,
        Orbits *orbits, TDIConfig *tdi_config,
        WDMSettings *wdm_settings,
        double *params_add_all, double *params_remove_all,
        int *data_index_all, int *noise_index_all,
        double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
        int *chunk_n_global_offset,
        double *wdm_window,
        double *data_d, double *invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha,
        int grid_dim, int N_cp_sig, int N_cp_orbit,
        int *binary_perm, int *group_starts, int *group_ends,
        int *group_m_lo, int *group_m_hi, int n_groups,
        int *pair_m_lo_b, int *pair_m_hi_b);

    // F-stat (chunked-heterodyne); see GBComputationGroup::gb_wdm_het_get_fstat_ll_wrap.
    void sobbh_wdm_het_get_fstat_ll_wrap(
        double *N_arr_re_out, double *N_arr_im_out,
        double *M_mat_re_out, double *M_mat_im_out,
        Orbits *orbits, TDIConfig *tdi_config,
        WDMSettings *wdm_settings,
        double *params_all,
        int *data_index_all, int *noise_index_all,
        double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
        int *chunk_n_global_offset,
        double *wdm_window,
        double *data_d, double *invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha,
        int grid_dim, int m_band_half_width);
};

#endif // __TDI_ON_THE_FLY_HH__
