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


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define GBTDIonTheFly GBTDIonTheFlyGPU
#define SOBBHTDIonTheFly SOBBHTDIonTheFlyGPU
// FDSplineTDIWaveform + TDSplineTDIWaveform aliases moved to LAT at Phase 3L.6.
// WaveletLookupTable disabled at Phase 3L (2026-06-02) -- lookup-table spline path is being retired.
#define GBComputationGroup GBComputationGroupGPU
#else
#define GBTDIonTheFly GBTDIonTheFlyCPU
#define SOBBHTDIonTheFly SOBBHTDIonTheFlyCPU
// FDSplineTDIWaveform + TDSplineTDIWaveform aliases moved to LAT at Phase 3L.6.
// WaveletLookupTable disabled at Phase 3L (2026-06-02) -- lookup-table spline path is being retired.
#define GBComputationGroup GBComputationGroupCPU
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

class GBTDIonTheFly : public LISATDIonTheFly{
    public:
        double T;
        double t_ref;
        int amplitude_index;
        int f0_index;
        int fdot0_index;
        int fddot0_index;
        int phi0_index;

        CUDA_CALLABLE_MEMBER
        GBTDIonTheFly(Orbits *orbits_, TDIConfig *tdi_config_, double T_, double t_ref_) : LISATDIonTheFly(orbits_, tdi_config_, 5, 6, 7, 8)
        {
            T = T_;
            t_ref = t_ref_;
            amplitude_index = 0;
            f0_index = 1;
            fdot0_index = 2;
            fddot0_index = 3;
            phi0_index = 4;
        };
        CUDA_CALLABLE_MEMBER
        ~GBTDIonTheFly();
        // CUDA_DEVICE
        // void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i) override;
        CUDA_DEVICE
        double ucb_amplitude(double t, double *params);
        CUDA_DEVICE
        double ucb_phase(double t, double *params);
        CUDA_DEVICE
        double ucb_fdot(double t, double *params);
        CUDA_DEVICE
        double ucb_f(double t, double *params);
        CUDA_CALLABLE_MEMBER
	int get_gb_buffer_size(int N);
        // Total bytes of dynamic shared memory needed by the heterodyne FD
        // kernel: params + tdi_channels_arr (complex, also FFT scratch) +
        // tdi_amp + tdi_phase + phi_ref + get_tdi scratch.  N here is the
        // sparse FFT length (must be a power of two).
        CUDA_CALLABLE_MEMBER
	int get_gb_fd_buffer_size(int N, int nchannels);
        // double get_phase_ref(double t, double *params, int bin_i);
        // CUDA_DEVICE
        // void run_wave_tdi(
        //     cmplx *tdi_channels_arr, 
        //     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
        //     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
        // );
        CUDA_DEVICE
        double get_amp(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_phase(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_f(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_fdot(double t, double *params, int bin_i);
};

void gb_run_wave_tdi_wrap(GBTDIonTheFly *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);

// Heterodyned frequency-domain GB TDI on the fly.
//
//   For each GB binary, builds the slow positive-frequency complex signal
//     s_c(tau) = A_c(tau) * exp(+i (phi_c(tau) + phi_ref(tau) - 2*pi*f0_grid*tau))
//   on a sparse, power-of-two-length time grid tau_n = n * dt_sparse, with
//   dt_sparse = Tobs / N_sparse and f0_grid = round(f0/df) * df,
//   df = 1/Tobs.  All three (or `nchannels`) channels live in shared memory
//   simultaneously so any XYZ-style cross-channel post-processing can happen
//   before the data hits global memory.  An in-place shared-memory radix-2
//   FFT then yields
//     X_het_c[m] = 0.5 * dt_sparse * FFT[s_c][m],
//   which equals the dense rfft of the real GB time series at dense bin
//   (k_f0 + m), bins outside +/- N_sparse/2 of f0_grid being zero.
//
// X_het output layout: (num_bin, nchannels, N_sparse) complex doubles,
//                       FFT-order (DC at index 0).
// k_f0_out: (num_bin,) integer dense rfft bin closest to f0.
// f0_grid_out: (num_bin,) double snapped carrier frequency [Hz].
//
// Requires N_sparse to be a power of two and t_ref == t_start; the caller
// should pass tau = t_local = absolute_t - t_start in t_arr_sparse (but the
// kernel just reads f0, Tobs, N_sparse and t_start to rebuild tau).
// tukey_alpha: scipy.signal.windows.tukey alpha applied to the slow
// signal before the sparse FFT. 0.0 = rectangular. Pass the same alpha
// used to window the dense rfft so the dense/sparse FD paths produce
// matching inner products.
void gb_run_fd_wave_tdi_wrap(GBTDIonTheFly *tdi_on_fly,
    cmplx *X_het, int *k_f0_out, double *f0_grid_out,
    double *params, double t_start, double Tobs,
    int N_sparse, int num_bin, int n_params, int nchannels,
    double tukey_alpha);


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

class GBComputationGroup{
  public:

    // Spline-path mirror of gb_wdm_fill_global_wrap. Replaces per-WDM-pixel
    // fast_wdm_inner calls with cubic-spline interpolation of get_tdi outputs
    // on a coarse uniform time grid of spacing `coarse_dt` (seconds). Output
    // template_fill is bit-compatible with the direct path's output up to
    // cubic-spline interpolation error.
    // gb_wdm_spline_fill_global_wrap disabled at Phase 3L (2026-06-02) -- WaveletLookupTable retiring.

    // Chunked-heterodyne family. Replaces the per-pixel WaveletLookupTable path
    // with a per-chunk dense-rfft + WDM xform built on the slow signal
    // (heterodyne to f0_grid). Geometry (chunk_t_starts / keep_lo / keep_hi /
    // n_global_offset / wdm_window) is precomputed on the host -- see
    // ``gb_wdm_het.compute_chunk_geometry`` / ``compute_wdm_window``.
    //
    // ``grid_dim`` selects the launch grid (number of CUDA blocks). The Python
    // helper ``chunked_het_grid_dim()`` picks an A100/H100-optimal value; pass
    // anything > 0 on CPU (ignored). Workspaces are allocated and freed inside
    // this wrapper.
    void gb_wdm_het_fill_global_wrap(
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
        int grid_dim, int N_cp_sig, int N_cp_orbit,
        int m_band_half_width);

    void gb_wdm_het_get_ll_wrap(
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
        int *group_m_lo, int *group_m_hi, int n_groups,
        int m_band_half_width);

    void gb_wdm_het_swap_ll_wrap(
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
        int *pair_m_lo_b, int *pair_m_hi_b,
        int m_band_half_width);

    // F-stat (chunked-heterodyne). Builds the 4 Cornish & Crowder '05 basis
    // filters per binary and writes:
    //   N_arr_re/im_out :  (num_bin, 4)  -- <d|A_i> (im always 0 for real WDM)
    //   M_mat_re/im_out :  (num_bin, 10) -- upper-triangle <A_i|A_j> (i<=j)
    // Python computes F = N^T M^{-1} N / 2 from these.
    void gb_wdm_het_get_fstat_ll_wrap(
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

    // Spline-path mirrors. `coarse_dt` (seconds) sets the coarse-grid spacing
    // for the cubic-spline window builder (smaller -> more accurate / more
    // get_tdi work). Python computes coarse_dt from a user knob
    // `coarse_pts_per_year` (typical 256).
    // gb_wdm_spline_get_ll_wrap, gb_wdm_spline_swap_ll_wrap disabled at Phase 3L (2026-06-02).

    // Chain-rule parameter gradients of the two likelihood kernels.
    //
    //   grad_out             (num_bin, nparams)  -- dL/dtheta for get_ll
    //   grad_{add,remove}_out (num_bin, nparams) -- d(ll_diff)/d(theta_{add,remove}) for swap_ll
    //
    // The per-parameter central-difference step size is supplied via
    // ``param_eps`` (length nparams).  Passing eps <= 0 freezes that
    // parameter (gradient stays zero).

    // Spline-path mirror of gb_wdm_get_ll_grad_wrap. Three spline slots in
    // shared memory (base, plus, minus); per-parameter inner loop rebuilds
    // the plus/minus slots from `params + eps_k e_k` and `params - eps_k e_k`
    // while the base slot is reused. `eps_k <= 0` freezes parameter k as
    // with the direct path. Shared-memory footprint is constant in nparams.
    // gb_wdm_spline_get_ll_grad_wrap disabled at Phase 3L (2026-06-02).


    // Spline analog of gb_wdm_eval_inputs_wrap. Builds ONE coarse-grid spline
    // window of WDM_SPLINE_L points starting at t_window_start with spacing
    // coarse_dt, then evaluates the splines at every tn in tn_arr. Outputs
    // are in the same convention as gb_wdm_eval_inputs_wrap.
    void gb_wdm_spline_eval_inputs_wrap(
        Orbits *orbits, TDIConfig *tdi_config,
        double *params_all, double *tn_arr,
        int num_bin, int nparams, int num_t, int nchannels,
        double T, double t_ref,
        double t_window_start, double coarse_dt,
        double *amp_out, double *phi_out, double *f_out, double *fdot_out,
        double *phase_ref_out);

    // ------------------------------------------------------------------
    // FD analogs of the WDM methods above.
    //
    // gb_fd_fill_global_wrap: add (or subtract -- via factors_all[bin_i] in {+1,-1})
    //    each GB's heterodyne FD piece into a global rfft-grid template buffer
    //    of shape (num_data, num_channel, n_rfft) addressed by data_index_all.
    //
    // gb_fd_get_ll_wrap: compute (d|h) and (h|h) per binary using the
    //    standard lisatools FD inner product
    //         (a|b) = 4 Re sum_{c1,c2} sum_k conj(a_c1[k]) b_c2[k] invC[c1,c2][k] df
    //    with cross-channel invC for tdi_type=TDI_XYZ and diagonal invC for
    //    tdi_type=TDI_AET / TDI_AE.
    //
    // gb_fd_swap_ll_wrap: same accumulators as the WDM swap, restricted to the
    //    union of the add- and remove-source sparse supports.
    //
    // All three share the same per-source heterodyne FD pass that
    // GBFDTDIonTheFly already uses.  N_sparse must be a power of two.
    void gb_fd_fill_global_wrap(cmplx *template_fill,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_all, int *data_index_all, double *factors_all,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels);

    void gb_fd_get_ll_wrap(double *d_h_out, double *h_h_out,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_all, int *data_index_all, int *noise_index_all,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    void gb_fd_swap_ll_wrap(
        double *d_h_add_out, double *d_h_remove_out,
        double *add_add_out, double *remove_remove_out, double *add_remove_out,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_add_all, double *params_remove_all,
        int *data_index_all, int *noise_index_all,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    // Chain-rule parameter gradients of the two FD likelihood kernels.
    // Same convention as the WDM counterparts:
    //   grad_out             (num_bin, nparams)  -- dL/dtheta for get_ll
    //   grad_{add,remove}_out (num_bin, nparams) -- d(ll_diff)/d(theta_{add,remove})
    // Per-parameter central-FD step is supplied via ``param_eps_*`` arrays
    // (length nparams).  Passing eps_k <= 0 freezes parameter k (grad stays 0).
    // Both routines are CPU-fully-wired; the GPU branch prints a TODO and
    // returns (matching gb_fd_swap_ll_wrap's status).
    void gb_fd_get_ll_grad_wrap(double *grad_out,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_all, int *data_index_all, int *noise_index_all,
        double *param_eps,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    void gb_fd_swap_ll_grad_wrap(
        double *grad_add_out, double *grad_remove_out,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_add_all, double *params_remove_all,
        int *data_index_all, int *noise_index_all,
        double *param_eps_add, double *param_eps_remove,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    // ------------------------------------------------------------------
    // Signal-heterodyne (v2 polyphase) family.
    //
    // First port of the v2 polyphase signal-het Python prototype at
    // ``LISAanalysistools/scripts/gb_chunked_het/gb_signal_het_wdm_v2.py``.
    // Takes the candidate's precomputed ``rfft(Tukey * td)`` and the
    // reference's precomputed ``c0_sparse / A0 / A1 / B0 / B1`` (bin-folded
    // at construction) and returns per-binary ``<d|h>``, ``<h|h>`` from
    // the bin-folded inner-product accumulator (v1-style sparse path:
    //    <d|h> = sum_{c',m,b} A0[c',m,b] * r[c',m,b] + A1[c',m,b] * dr/dn[c',m,b]
    // where r and dr/dn are evaluated at sparse bin centres without
    // carrier de-rotation; matches the Python prototype to FP precision
    // at DF0=0 and ~1% relative residual at DF0/layer_df=0.05).
    //
    // Active m-band = m_floor +/- m_active_half_width (m_floor =
    // floor(f0/layer_df), default half-width = 2 => 5 layers).
    //
    // Per-channel-index conventions (XYZ tdi_type=0; AE/AET tdi_type=1
    // uses diagonal B0/B1 of shape (num_data, nch, Nf_active, Nt_layer)
    // instead of (num_data, nch, nch, Nf_active, Nt_layer)).
    void gb_signal_het_get_ll_wrap(
        double *d_h_out,
        double *h_h_out,
        cmplx  *fd_rfft_all,
        cmplx  *c0_sparse_all,
        cmplx  *A0_all,
        cmplx  *A1_all,
        cmplx  *B0_all,
        cmplx  *B1_all,
        double *wdm_window,
        int    *n_sparse_local_arr,
        double *params_cand_all,
        double *params_ref_all,
        int    *data_index_all,
        int     num_bin, int num_data,
        int     nparams, int f0_idx, int fdot_idx,
        int     Nf, int Nt, int Nf_active, int Nt_active,
        int     Nt_layer, int N_sparse_t, int stride,
        int     ind_min_t, int ind_min_f,
        int     m_active_half_width,
        double  layer_df, double dt,
        int     nchannels, int tdi_type,
        int     n_rfft);

    // Stage 2a: signal_het_get_ll consuming the SPARSE carrier-removed FD
    // (the output of GBTDIonTheFly::run_fd_wave_tdi -- length N_sparse_fd per
    // (binary, channel), centred at the per-binary k_f0 absolute-FD bin).
    // Polyphase fold iterates only over the N_sparse_fd nonzero bins,
    // implicit zero everywhere else. Eliminates per-source dense FD storage.
    //
    // X_het_all[bin, ch, i] is the absolute FD value at bin
    //     k_abs = k_f0_all[bin] + (i - N_sparse_fd/2)
    // i.e. the dense rfft restricted to a window of N_sparse_fd bins around
    // f0. In production (Stage 2b) this array is filled in-kernel from the
    // source-class heterodyned sparse rfft; here it is an input for
    // validation against the dense-FD path.
    void gb_signal_het_get_ll_sparse_wrap(
        double *d_h_out,
        double *h_h_out,
        cmplx  *X_het_all,
        int    *k_f0_all,
        cmplx  *c0_sparse_all,
        cmplx  *A0_all,
        cmplx  *A1_all,
        cmplx  *B0_all,
        cmplx  *B1_all,
        double *wdm_window,
        int    *n_sparse_local_arr,
        double *params_cand_all,
        double *params_ref_all,
        int    *data_index_all,
        int     num_bin, int num_data,
        int     nparams, int f0_idx, int fdot_idx,
        int     Nf, int Nt, int Nf_active, int Nt_active,
        int     Nt_layer, int N_sparse_t, int stride,
        int     ind_min_t, int ind_min_f,
        int     m_active_half_width,
        double  layer_df, double dt,
        int     nchannels, int tdi_type,
        int     N_sparse_fd);

    // Stage 2b -- in-kernel sparse-FD signal-het. Fuses the existing
    // ``gb_run_fd_wave_tdi`` (sparse heterodyned rfft) with the polyphase +
    // bin-fold pipeline. X_het is allocated in a transient per-call buffer
    // (heap on CPU; would be per-block shared memory on GPU at Stage 3).
    // No per-source FD storage in global memory.
    // tukey_alpha is set by the caller to match the alpha applied to the
    // dense rfft(Tukey * td) on the analysis side; 0.05 is the
    // recommended/default value pushed in from Python.
    void gb_signal_het_get_ll_in_kernel_wrap(
        GBTDIonTheFly *tdi_on_fly,
        double *d_h_out, double *h_h_out,
        cmplx  *c0_sparse_all,
        cmplx  *A0_all, cmplx *A1_all,
        cmplx  *B0_all, cmplx *B1_all,
        double *wdm_window,
        int    *n_sparse_local_arr,
        double *params_cand_all,
        double *params_ref_all,
        int    *data_index_all,
        int     num_bin, int num_data,
        int     nparams, int f0_idx, int fdot_idx,
        int     Nf, int Nt, int Nf_active, int Nt_active,
        int     Nt_layer, int N_sparse_t, int stride,
        int     ind_min_t, int ind_min_f,
        int     m_active_half_width,
        double  layer_df, double dt,
        double  T_obs, double t_start,
        int     nchannels, int tdi_type,
        int     N_sparse_fd, double tukey_alpha);

    // Signal-het fill_global. Same FD + polyphase + r_sparse machinery as
    // get_ll, but reconstructs the dense template via the heterodyne
    // identity (linear-interp r_demod -> re-rotate carrier -> multiply by
    // stored c0_dense_complex on the full active band -> take real part)
    // and scatters into template_fill at the (m_global, n_global) WDM
    // positions of each active layer. Caller pre-zeroes / accumulates.
    void gb_signal_het_fill_global_sparse_wrap(
        double *template_fill,
        cmplx  *X_het_all, int *k_f0_all,
        cmplx  *c0_sparse_all,
        cmplx  *c0_dense_complex_all,
        double *wdm_window, int *n_sparse_local_arr,
        double *params_cand_all, double *params_ref_all,
        double *factors_all,
        int    *data_index_all,
        int     num_bin, int num_data,
        int     nparams, int f0_idx, int fdot_idx,
        int     Nf, int Nt, int Nf_active, int Nt_active,
        int     Nt_layer, int N_sparse_t, int stride,
        int     ind_min_t, int ind_min_f,
        int     m_active_half_width,
        double  layer_df, double dt,
        int     nchannels,
        int     N_sparse_fd);

    void gb_signal_het_fill_global_in_kernel_wrap(
        GBTDIonTheFly *tdi_on_fly,
        double *template_fill,
        cmplx  *c0_sparse_all,
        cmplx  *c0_dense_complex_all,
        double *wdm_window, int *n_sparse_local_arr,
        double *params_cand_all, double *params_ref_all,
        double *factors_all,
        int    *data_index_all,
        int     num_bin, int num_data,
        int     nparams, int f0_idx, int fdot_idx,
        int     Nf, int Nt, int Nf_active, int Nt_active,
        int     Nt_layer, int N_sparse_t, int stride,
        int     ind_min_t, int ind_min_f,
        int     m_active_half_width,
        double  layer_df, double dt,
        double  T_obs, double t_start,
        int     nchannels,
        int     N_sparse_fd, double tukey_alpha);

    // Signal-het central-difference gradient of logL = d_h - 0.5*h_h over
    // candidate params. param_eps[k] is the per-parameter finite-difference
    // step; eps_k <= 0 freezes dimension k. Also reports the central
    // d_h_central/h_h_central so callers get logL alongside the gradient
    // in a single pass.
    void gb_signal_het_get_ll_grad_in_kernel_wrap(
        GBTDIonTheFly *tdi_on_fly,
        double *grad_out,
        double *d_h_central, double *h_h_central,
        cmplx  *c0_sparse_all,
        cmplx  *A0_all, cmplx *A1_all,
        cmplx  *B0_all, cmplx *B1_all,
        double *wdm_window, int *n_sparse_local_arr,
        double *params_cand_all, double *params_ref_all,
        int    *data_index_all,
        double *param_eps,
        int     num_bin, int num_data,
        int     nparams, int f0_idx, int fdot_idx,
        int     Nf, int Nt, int Nf_active, int Nt_active,
        int     Nt_layer, int N_sparse_t, int stride,
        int     ind_min_t, int ind_min_f,
        int     m_active_half_width,
        double  layer_df, double dt,
        double  T_obs, double t_start,
        int     nchannels, int tdi_type,
        int     N_sparse_fd, double tukey_alpha);
};


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
