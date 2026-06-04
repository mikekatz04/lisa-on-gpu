// WDMSplineHelpers.hh
//
// Layer-by-layer WDM transform kernels for spline-based TDI-on-the-fly
// waveforms (mirrors GB special-move semantics but for broadband, MBHB-like
// sources whose evolution is described by amp(t) and freq(t) cubic splines).
//
// Pipeline per binary (one CUDA block / one CPU-loop iteration):
//   1. Synthesize TD samples on the fly using FDSplineTDIWaveform + the LISA
//      response, written into a per-block global scratch buffer of size
//      Nf*Nt cmplx per channel.
//   2. Take the full FFT of the TD scratch via a 2-stage row-column
//      decomposition (Nt FFTs of size Nf, then Nf FFTs of size Nt).  Each
//      stage uses Bluestein in shared memory so general-even Nf, Nt work.
//   3. For each WDM frequency layer m in [m_min, m_max]: gather the Nt-bin
//      slice from the FD scratch, multiply by the precomputed Cmm window,
//      run an IFFT (Bluestein, shared memory) of size Nt_layer, apply the
//      (m+n)-parity sign/real-imag selector on the *global* n grid, and
//      either atomic-add the resulting w_mn into a global template or
//      accumulate <d|h>, <h|h>, etc. via WDMDomain helpers.  The layer's
//      w_mn never escapes shared memory.
//
// Narrow-window per-(src, layer) hooks are provided: when
// narrow_widths[bin_i, m] != 0, that layer uses an Nt_narrow-sized slice
// centered on narrow_centers[bin_i, m] (must be even).  The phase/parity
// computation always uses the global n index.
//
// All code uses the THREAD_START_X / BLOCK_INCR_X / CUDA_SYNC_THREADS macros so
// it compiles cleanly as single-threaded on CPU and block-cooperative on
// GPU.

#ifndef __WDM_SPLINE_HELPERS_HH__
#define __WDM_SPLINE_HELPERS_HH__

#include "TDIonTheFly.hh"
#include "gbt_global.h"

// FFT machinery (radix-2 + cufftdx + dispatcher) moved to LAT at Phase
// 3L.7a slice 2a (2026-06-04). `wdm_spline_radix2_fft`,
// `wdm_cufftdx_*`, `cufftdx_block_fft`, `wdm_fft_dispatch`, and
// `wdm_cufftdx_max_scratch` all live in
// `lisatools/cutils/lat_wdm_fft.hh` now. They're consumed here so the
// downstream Bluestein + synth + extract helpers (which stay in this
// file -- spline-path-specific) keep working without source changes,
// and so the new LAT-side chunked-het kernels can `#include
// "lat_wdm_fft.hh"` directly without dragging in this whole file.
#include "lat_wdm_fft.hh"

#ifndef WDM_SPLINE_MAX_NARROW_WIDTHS
#define WDM_SPLINE_MAX_NARROW_WIDTHS 8
#endif

// ---------------------------------------------------------------------------
// Small POD helper passed by value to kernels: precomputed Bluestein chirps
// + log2 lengths for each transform size we use within a launch.  Up to two
// "primary" sizes (Nf and Nt) plus WDM_SPLINE_MAX_NARROW_WIDTHS extras.
//
// chirp_n_*: length N, holds w_k = exp(-i * pi * k^2 / N) (forward) or its
//   conjugate (inverse) -- we build both signs at host setup time.
// chirp_M_fft_*: length M = next_pow2(2N-1), holds FFT( padded sequence b_k ),
//   reused across many forward/inverse Bluestein evaluations.
// ---------------------------------------------------------------------------

class WDMSplineBluesteinTable {
  public:
    // --- "big" sizes used by the 2-stage TD->FD FFT ---
    int    Nf;           // first-stage transform size
    int    Mf;           // bluestein pad length for size Nf
    int    log2Mf;
    cmplx *chirp_n_Nf;   // (Nf,)   forward chirp factors
    cmplx *chirp_M_Nf;   // (Mf,)   FFT of zero-padded b_k for size Nf
    int    Nt;           // second-stage transform size = "full" layer length
    int    Mt;           // bluestein pad length for size Nt
    int    log2Mt;
    cmplx *chirp_n_Nt;
    cmplx *chirp_M_Nt;
    // --- narrow-width entries (up to WDM_SPLINE_MAX_NARROW_WIDTHS) ---
    int    num_narrow;
    int    narrow_widths[WDM_SPLINE_MAX_NARROW_WIDTHS];
    int    narrow_Ms    [WDM_SPLINE_MAX_NARROW_WIDTHS];
    int    narrow_log2Ms[WDM_SPLINE_MAX_NARROW_WIDTHS];
    cmplx *narrow_chirp_n [WDM_SPLINE_MAX_NARROW_WIDTHS];
    cmplx *narrow_chirp_M [WDM_SPLINE_MAX_NARROW_WIDTHS];

    CUDA_CALLABLE_MEMBER
    WDMSplineBluesteinTable() : Nf(0), Mf(0), log2Mf(0),
                                 chirp_n_Nf(NULL), chirp_M_Nf(NULL),
                                 Nt(0), Mt(0), log2Mt(0),
                                 chirp_n_Nt(NULL), chirp_M_Nt(NULL),
                                 num_narrow(0) {
        for (int i = 0; i < WDM_SPLINE_MAX_NARROW_WIDTHS; ++i) {
            narrow_widths[i] = 0;
            narrow_Ms[i] = 0;
            narrow_log2Ms[i] = 0;
            narrow_chirp_n[i] = NULL;
            narrow_chirp_M[i] = NULL;
        }
    }

    CUDA_CALLABLE_MEMBER
    int find_narrow_index(int width) const {
        for (int i = 0; i < num_narrow; ++i)
            if (narrow_widths[i] == width) return i;
        return -1;
    }
};

// ---------------------------------------------------------------------------
// Three flavors of the per-layer finishing step.  Each "operates on a layer"
// after the IFFT + parity has produced w_mn for nchannels in shared memory.
// ---------------------------------------------------------------------------
enum WDMSplineKernelMode {
    WDM_SPLINE_FILL_GLOBAL = 0,
    WDM_SPLINE_GET_LL      = 1,
    WDM_SPLINE_SWAP_LL     = 2,
};

// ---------------------------------------------------------------------------
// Host helpers (declared in TDIonTheFly.hh, defined in TDIonTheFly.cu).
// Build the Bluestein chirp arrays for a given N and the WDM analysis window
// (matches WDMSettings.phitilde + setup_window in lisatools/domains.py).
// These are pure-host routines (numpy-free); the kernel reads the resulting
// arrays directly.
// ---------------------------------------------------------------------------

void wdm_spline_build_bluestein_chirps(int N, cmplx *chirp_n_out, cmplx *chirp_M_out, int *M_out, int *log2M_out);
void wdm_spline_build_window         (int Nt_layer, double *window_out);

int  wdm_spline_pad_length (int N);   // next power of 2 >= 2N - 1, with N>=2
int  wdm_spline_log2_pad   (int N);

// ---------------------------------------------------------------------------
// Three host wrappers, one per kernel mode.  Same signatures so the Python
// wrapper can dispatch without duplicating glue code.
//
// All scratch is allocated by the host and passed in:
//   global_scratch         (num_blocks * nchannels * N) cmplx
// (where N = Nf * Nt and num_blocks = launch-time grid dim x).
// ---------------------------------------------------------------------------

void fd_spline_wdm_fill_global_wrap(
    double *template_fill,                    // (num_data, num_channel, Nf_active, Nt_active)
    cmplx  *global_scratch,                   // (num_blocks, nchannels, Nf*Nt) cmplx
    int     num_blocks,                       // = launch grid_dim_x (or 1 on CPU)
    const WDMSplineBluesteinTable *bluestein, // precomputed chirps
    double *window_full,                      // (Nt,) Cmm
    double *window_narrow,                    // (num_narrow, Nt_max_narrow) — flat layout
    int     window_narrow_stride,             // = Nt_max_narrow (or 0 if no narrow)
    int    *narrow_widths,                    // (num_bin, Nf_active) or NULL
    int    *narrow_centers,                   // (num_bin, Nf_active) or NULL
    FDSplineTDIWaveform *tdi_on_fly,
    WDMDomain *wdm,
    double *params, int *data_index_all, double *factors_all,
    double dt,
    int num_bin, int n_params, int nchannels);

void fd_spline_wdm_get_ll_wrap(
    double *d_h_out, double *h_h_out,
    cmplx  *global_scratch, int num_blocks,
    const WDMSplineBluesteinTable *bluestein,
    double *window_full, double *window_narrow, int window_narrow_stride,
    int    *narrow_widths, int *narrow_centers,
    FDSplineTDIWaveform *tdi_on_fly, WDMDomain *wdm,
    double *params, int *data_index_all, int *noise_index_all,
    double dt,
    int num_bin, int n_params, int nchannels, int tdi_type);

void fd_spline_wdm_swap_ll_wrap(
    double *d_h_add_out, double *d_h_remove_out,
    double *add_add_out, double *remove_remove_out, double *add_remove_out,
    cmplx  *global_scratch, int num_blocks,
    const WDMSplineBluesteinTable *bluestein,
    double *window_full, double *window_narrow, int window_narrow_stride,
    int    *narrow_widths_add, int *narrow_centers_add,
    int    *narrow_widths_rem, int *narrow_centers_rem,
    FDSplineTDIWaveform *tdi_on_fly, WDMDomain *wdm,
    double *params_add, double *params_rem,
    int *data_index_all, int *noise_index_all,
    double dt,
    int num_bin, int n_params, int nchannels, int tdi_type);


// ===========================================================================
// IMPLEMENTATION
// ===========================================================================
//
// The following definitions are inlined into the translation unit that
// includes this header (i.e. TDIonTheFly.cu).  They are not exported as
// stand-alone symbols.  They rely on macros and types defined by
// TDIonTheFly.hh / gbt_global.h being already in scope (cmplx, CUDA_DEVICE,
// THREAD_START_X, etc.).
// ---------------------------------------------------------------------------

#ifdef WDM_SPLINE_HELPERS_IMPLEMENTATION

#include <cmath>
#include <cstring>

// --- power-of-2 ceiling helpers (host) -------------------------------------
inline int wdm_spline_pad_length(int N)
{
    int need = 2 * N - 1;
    int m = 1;
    while (m < need) m <<= 1;
    return m;
}

inline int wdm_spline_log2_pad(int N)
{
    int M = wdm_spline_pad_length(N);
    int r = 0;
    int t = M;
    while ((t >>= 1) != 0) ++r;
    return r;
}

// --- host: build Bluestein chirp arrays for size N -------------------------
//
// chirp_n[k]   = exp(-i * pi * k^2 / N)        k = 0..N-1
// b_k          = exp(+i * pi * k^2 / N)        k = -(N-1)..(N-1), 0 else
// chirp_M[*]   = FFT(b padded to length M)     (zero-padded)
//
// The convolution form of the DFT is:
//
//   X[k] = chirp_n[k] * sum_n  ( x[n] * chirp_n[n] ) * b[k-n]
//
// where the inner sum is computed as IFFT( FFT(a) * FFT(b) ).  See
// Bluestein 1968 / Rabiner-Schafer-Rader 1969 for the chirp-z form.
//
// For the IFFT we negate the chirp signs (chirp_n -> conj, b -> conj) on the
// kernel side so we reuse the same precomputed arrays.

inline void wdm_spline_build_bluestein_chirps(int N, cmplx *chirp_n_out, cmplx *chirp_M_out, int *M_out, int *log2M_out)
{
    int M = wdm_spline_pad_length(N);
    int log2M = wdm_spline_log2_pad(N);
    if (M_out)     *M_out     = M;
    if (log2M_out) *log2M_out = log2M;

    // chirp_n[k] = exp(-i pi k^2 / N) -- forward sign
    for (int k = 0; k < N; ++k) {
        double phase = -M_PI * ((double) k * (double) k) / (double) N;
        chirp_n_out[k] = cmplx(cos(phase), sin(phase));
    }

    // Build b in a temp buffer (length M, zero-padded), then FFT.
    // b[0]      = 1
    // b[k]      = exp(+i pi k^2 / N)        for k = 1..N-1
    // b[M-k]    = b[k]                      mirror
    // b[k]      = 0                         for k in [N, M-N]
    cmplx *b = new cmplx[M];
    for (int k = 0; k < M; ++k) b[k] = cmplx(0.0, 0.0);
    b[0] = cmplx(1.0, 0.0);
    for (int k = 1; k < N; ++k) {
        double phase = M_PI * ((double) k * (double) k) / (double) N;
        cmplx v(cos(phase), sin(phase));
        b[k]     = v;
        b[M - k] = v;
    }

    // In-place radix-2 FFT of b -- use a host-side simple Cooley-Tukey.
    // Bit reversal:
    int log2M_local = log2M;
    for (int n = 0; n < M; ++n) {
        int r = 0, x = n;
        for (int i = 0; i < log2M_local; ++i) { r = (r << 1) | (x & 1); x >>= 1; }
        if (r > n) { cmplx t = b[n]; b[n] = b[r]; b[r] = t; }
    }
    for (int s = 1; s <= log2M_local; ++s) {
        int mm  = 1 << s;
        int mh  = mm >> 1;
        double base = -2.0 * M_PI / (double) mm;
        for (int k = 0; k < (M >> 1); ++k) {
            int g  = k / mh;
            int j  = k - g * mh;
            int i0 = g * mm + j;
            int i1 = i0 + mh;
            double th = base * (double) j;
            cmplx w(cos(th), sin(th));
            cmplx u = b[i0];
            cmplx v = w * b[i1];
            b[i0] = u + v;
            b[i1] = u - v;
        }
    }
    for (int k = 0; k < M; ++k) chirp_M_out[k] = b[k];
    delete[] b;
}

// --- host: build the WDM analysis window matching domains.py phitilde -----
//
// dOmega_s = pi / Nf (matches domains.py setup_window with Nf = Nt_layer/2;
// for the spline kernel we use the *layer*-IFFT length, so the host caller
// must pass Nf consistent with the layer size).
//
// For our use we want the same window as lisatools (Cornish-Romano with
// beta_incomplete-regularized roll-off, WAVELET_FILTER_CONSTANT=4), so we
// use the regularized incomplete beta function I_x(a,a).  For a=4 we can
// compute this directly from a degree-9 polynomial -- but to avoid a custom
// math dep we use a power-series evaluation that converges quickly for
// x in [0, 1].
//
// For arbitrary Nt_layer (general even), the window is parametrized purely
// by `omega = (2 pi / Nt_layer) * (i - Nt_layer/2)` for i in [0, Nt_layer).

namespace _wdm_spline_window {
    // Regularized incomplete beta I_x(a, b) for integer a = b = 4.
    // Closed form via lgamma is overkill; we use the series:
    //   I_x(a, b) = x^a (1-x)^b * sum_{j=0}^{inf} C(a+b-1, a+j) x^j / B(a,b)
    // For a = b = 4, B(a,b) = 1 / 140.  We compute the sum from the
    // recurrence to ~30 terms; that's > 1e-15 accurate for x in [0, 0.5]
    // and we use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) for x > 0.5.
    inline double betainc_aa_4(double x)
    {
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;
        double xx = (x > 0.5) ? (1.0 - x) : x;
        // series: x^a (1-x)^b / B(a,b) * sum_{j=0}^{N} (a+b-1 choose a-1+j) * (-x)^j ... too complex.
        // Use direct numerical integration via series for the regularized lower:
        //   I_x(a,b) = sum_{j=a}^{a+b-1} C(a+b-1, j) x^j (1-x)^(a+b-1-j)
        // For a = b = 4, a+b-1 = 7.
        // sum_{j=4}^{7} C(7,j) x^j (1-x)^(7-j)
        const double comb[4] = {35.0, 35.0, 21.0, 7.0};  // C(7,4), C(7,5), C(7,6), C(7,7)
        double y = 1.0 - xx;
        double p = xx*xx*xx*xx;          // xx^4
        double q = y*y*y;                // y^3
        double s = 0.0;
        // j=4: 35 * xx^4 * y^3
        s += comb[0] * p * q;
        // j=5: 35 * xx^5 * y^2
        p *= xx; q /= y;
        s += comb[1] * p * q;
        // j=6: 21 * xx^6 * y
        p *= xx; q /= y;
        s += comb[2] * p * q;
        // j=7: 7 * xx^7
        p *= xx;
        s += comb[3] * p;
        if (x > 0.5) s = 1.0 - s;
        return s;
    }

    inline double phitilde(double omega, double dOmega, double A_param)
    {
        // Match WDMSettings.phitilde in domains.py.  insDOM = 1/sqrt(dOmega),
        // A = dOmega/4, B = dOmega - 2A.  Outside [A, A+B]: zero.  Inside
        // [-A, A]: insDOM.  Inside [A, A+B] (and mirror): insDOM * cos(I_x*pi/2).
        double insDOM = 1.0 / sqrt(dOmega);
        double A = A_param;
        double B = dOmega - 2.0 * A;
        double abs_om = fabs(omega);
        if (abs_om < A) return insDOM;
        if (abs_om >= A && abs_om <= A + B) {
            double xx = (abs_om - A) / B;
            double y  = betainc_aa_4(xx);
            return insDOM * cos(y * M_PI / 2.0);
        }
        return 0.0;
    }
}

inline void wdm_spline_build_window(int Nt_layer, double *window_out)
{
    // The window depends only on the layer IFFT length.  For the *full*
    // transform Nt_layer = Nt and dOmega_s = pi / Nf — but since we pass it
    // by length here we use the canonical relation dOmega_s = 2*pi / Nt_layer
    // (i.e. omega step is 2 pi / Nt_layer so the bin range is exactly pi).
    // For the narrow case the same formula applies with the smaller Nt_layer
    // and a correspondingly *wider* normalization (insDOM = 1/sqrt(dOmega_s)),
    // which is what we want -- a narrower time wavelet has wider frequency
    // support.
    double dOmega_s = 2.0 * M_PI / (double) Nt_layer;
    double A = dOmega_s / 4.0;
    for (int i = 0; i < Nt_layer; ++i) {
        double omega = (2.0 * M_PI / (double) Nt_layer) * (double) (i - Nt_layer / 2);
        window_out[i] = _wdm_spline_window::phitilde(omega, dOmega_s, A);
    }
}


// wdm_spline_radix2_fft + cufftdx wrappers + wdm_fft_dispatch +
// wdm_cufftdx_max_scratch moved to LAT (lat_wdm_fft.hh) at Phase
// 3L.7a slice 2a (2026-06-04). They are consumed here via the
// `#include "lat_wdm_fft.hh"` near the top of this file, so the
// remaining spline-path helpers (wdm_spline_bluestein_fft,
// wdm_spline_synth_and_fft, wdm_spline_extract_layer) keep working
// without any source-level rewiring.


// ---------------------------------------------------------------------------
// Device: Bluestein FFT (general N).  Inputs/outputs in `a` (length N).
// `workspace` must have length 2*M, used for the padded sequence and an FFT
// scratch.  `chirp_n` (length N) and `chirp_M_fft` (length M) come from the
// host-precomputed tables.
//
// Forward:  X[k] = chirp_n[k] * IFFT_M( FFT_M( pad_M(x * chirp_n) ) * chirp_M_fft )[k] / M
// Inverse:  conjugate the chirp_n and chirp_M_fft entries to flip sign, and
// scale the final result by 1/N at the end.
// ---------------------------------------------------------------------------
CUDA_DEVICE
inline void wdm_spline_bluestein_fft(cmplx *a, cmplx *workspace,
                                     const cmplx *chirp_n, const cmplx *chirp_M_fft,
                                     int N, int M, int log2M, bool inverse)
{
    // Build pad = (x_k * chirp_n[k]) for k in [0, N), zeros elsewhere.
    // For inverse we use conj(chirp_n[k]).
    cmplx *pad = workspace;
    for (int k = THREAD_START_X; k < M; k += BLOCK_INCR_X) pad[k] = cmplx(0.0, 0.0);
    CUDA_SYNC_THREADS;
    for (int k = THREAD_START_X; k < N; k += BLOCK_INCR_X) {
        cmplx cn = chirp_n[k];
        if (inverse) cn = cmplx(cn.real(), -cn.imag());
        pad[k] = a[k] * cn;
    }
    CUDA_SYNC_THREADS;

    // FFT_M(pad) in place
    wdm_spline_radix2_fft(pad, M, log2M, false);
    CUDA_SYNC_THREADS;

    // Pointwise multiply by chirp_M_fft (conj for inverse)
    for (int k = THREAD_START_X; k < M; k += BLOCK_INCR_X) {
        cmplx cM = chirp_M_fft[k];
        if (inverse) cM = cmplx(cM.real(), -cM.imag());
        pad[k] = pad[k] * cM;
    }
    CUDA_SYNC_THREADS;

    // IFFT_M -> pad now holds the chirp-domain convolution (with /M built in)
    wdm_spline_radix2_fft(pad, M, log2M, true);
    CUDA_SYNC_THREADS;

    // Write back: a[k] = pad[k] * chirp_n[k] (conj for inverse); /N for inverse.
    double scale = inverse ? (1.0 / (double) N) : 1.0;
    for (int k = THREAD_START_X; k < N; k += BLOCK_INCR_X) {
        cmplx cn = chirp_n[k];
        if (inverse) cn = cmplx(cn.real(), -cn.imag());
        cmplx out = pad[k] * cn;
        a[k] = cmplx(out.real() * scale, out.imag() * scale);
    }
    CUDA_SYNC_THREADS;
}


// ---------------------------------------------------------------------------
// Device helper: synthesize the TD waveform for one binary into a global
// scratch buffer, then do the 2-stage row-column FFT in place.  After this
// returns, `scratch` (length Nf*Nt cmplx, per channel) holds the full FD of
// the binary's TDI channel `chan` in the index order  FD[k_f + Nf*k_t]
// where k = k_f + Nf*k_t.
//
// Layout choice:
//   n = a * Nt + b,   a in [0, Nf), b in [0, Nt)
//   X[k_f + Nf * k_t]: this is what the row-column 2D FFT naturally produces
//   from this layout with the canonical sign convention.
// ---------------------------------------------------------------------------
CUDA_DEVICE
inline void wdm_spline_synth_and_fft(
    cmplx *scratch,                         // (Nf*Nt) cmplx, per channel
    cmplx *shared_buf,                      // shared-mem buffer, big enough for max(Nf, Nt) bluestein workspace
    FDSplineTDIWaveform *tdi_on_fly,
    double *params,
    int *link_space_craft_rec, int *link_space_craft_em,
    Vec k_sky, Vec u_sky, Vec v_sky,
    double dt, int Nf, int Nt, int bin_i, int chan,
    const cmplx *chirp_n_Nf, const cmplx *chirp_M_Nf, int Mf, int log2Mf,
    const cmplx *chirp_n_Nt, const cmplx *chirp_M_Nt, int Mt, int log2Mt)
{
    // Shared layout: [ row_buf (max(Nf,Nt) cmplx) | bluestein workspace (2 * max(Mf,Mt) cmplx) ]
    int maxN = (Nf > Nt) ? Nf : Nt;
    cmplx *row_buf  = shared_buf;
    cmplx *blu_work = shared_buf + maxN;

    int N_total = Nf * Nt;
    bool Nf_pow2 = ((Nf & (Nf - 1)) == 0);
    bool Nt_pow2 = ((Nt & (Nt - 1)) == 0);
    int log2_Nf = 0; { int t = Nf; while ((t >>= 1) != 0) ++log2_Nf; }
    int log2_Nt = 0; { int t = Nt; while ((t >>= 1) != 0) ++log2_Nt; }

    // ---------------- Stage A: Nt FFTs of size Nf -----------------------
    // For each column b in [0, Nt), evaluate TD samples a*Nt+b in shared row_buf,
    // FFT them (size Nf), apply twiddle, store back into scratch[a, b] (col-stride Nt).
    cmplx I(0.0, 1.0);
    for (int b = 0; b < Nt; ++b)
    {
        // Synthesize the column into row_buf
        for (int a = THREAD_START_X; a < Nf; a += BLOCK_INCR_X) {
            int n_sample = a * Nt + b;
            double t = (double) n_sample * dt;
            // FDSplineTDIWaveform represents amp(t) * exp(i * 2*pi*f(t)*t),
            // projected through the LISA TDI response.  We evaluate via the
            // existing get_tdi_Xf_single helper (returns the cmplx TDI value
            // for the requested channel + time).
            cmplx tdi_val(0.0, 0.0);
            tdi_on_fly->get_tdi_Xf_single(&tdi_val, t, params,
                                          k_sky, u_sky, v_sky,
                                          link_space_craft_rec, link_space_craft_em, bin_i);
            // For the FD step we need *real* samples (the TD waveform is real
            // by construction; the imaginary part of TDI_Xf_single is the
            // Hilbert-transform pair used for FD plotting -- we drop it).
            row_buf[a] = cmplx(tdi_val.real(), 0.0);
        }
        CUDA_SYNC_THREADS;

        if (Nf_pow2) {
            wdm_spline_radix2_fft(row_buf, Nf, log2_Nf, false);
        } else {
            wdm_spline_bluestein_fft(row_buf, blu_work,
                                     chirp_n_Nf, chirp_M_Nf,
                                     Nf, Mf, log2Mf, false);
        }
        CUDA_SYNC_THREADS;

        // Twiddle: S'[a, b] = S[a, b] * exp(-i 2*pi * a * b / N) and write to scratch
        for (int a = THREAD_START_X; a < Nf; a += BLOCK_INCR_X) {
            double th = -2.0 * M_PI * (double) a * (double) b / (double) N_total;
            cmplx w(cos(th), sin(th));
            scratch[a * Nt + b] = row_buf[a] * w;
        }
        CUDA_SYNC_THREADS;
    }

    // ---------------- Stage B: Nf FFTs of size Nt -----------------------
    // For each row k_f, load scratch[k_f, :] (Nt values), FFT, write back.
    for (int kf = 0; kf < Nf; ++kf) {
        for (int b = THREAD_START_X; b < Nt; b += BLOCK_INCR_X) {
            row_buf[b] = scratch[kf * Nt + b];
        }
        CUDA_SYNC_THREADS;
        if (Nt_pow2) {
            wdm_spline_radix2_fft(row_buf, Nt, log2_Nt, false);
        } else {
            wdm_spline_bluestein_fft(row_buf, blu_work,
                                     chirp_n_Nt, chirp_M_Nt,
                                     Nt, Mt, log2_Nt, false);
        }
        CUDA_SYNC_THREADS;
        for (int b = THREAD_START_X; b < Nt; b += BLOCK_INCR_X) {
            scratch[kf * Nt + b] = row_buf[b];
        }
        CUDA_SYNC_THREADS;
    }
}


// ---------------------------------------------------------------------------
// Device helper: extract one WDM layer's w_mn coefficients into a shared
// buffer.  Inputs: FD scratch already populated by wdm_spline_synth_and_fft.
//
//   `slice` buffer:    Nt_layer cmplx (shared)
//   `blu_work`:        bluestein workspace, 2*M cmplx (shared)
//   `window`:          Nt_layer doubles (read-only)
//   `w_mn_out`:        Nt_layer doubles (shared write)
//
// Layer-slice bins for layer m: k = m * Nt_layer/2 + i for i in [-Nt_layer/2, Nt_layer/2)
// (matches WDMSettings.get_shift_map but with Nt_layer in place of the full Nt).
//
// In the row-column 2D FFT layout, bin k = k_f + Nf*k_t maps to:
//   k_f = k mod Nf
//   k_t = k / Nf
// We do the gather one-by-one (Nt_layer << N_total, so the cost is small
// compared to the FFT).  Negative-frequency wraparound: standard rfft trick
// (k < 0  -> conj(X[-k]),  k > N/2 -> conj(X[N-k])) mirroring what
// domains.py does at lines 778-795.
// ---------------------------------------------------------------------------
CUDA_DEVICE
inline void wdm_spline_extract_layer(
    double *w_mn_out, cmplx *slice, cmplx *blu_work,
    const cmplx *fd_scratch,                  // size Nf*Nt cmplx, full FD
    const double *window,                     // size Nt_layer
    int Nf, int Nt, int N_total,
    int Nt_layer, int M_layer, int log2M_layer,
    const cmplx *chirp_n_layer, const cmplx *chirp_M_layer,
    int m,                                    // layer index (global, 0..Nf)
    int n_global_start,                       // first global pixel index of the output
    double kappa, double dt_data)
{
    // 1. Gather Nt_layer FD bins around m * Nt_layer/2, with Hermitian wrap
    int center = m * (Nt_layer / 2);
    for (int i = THREAD_START_X; i < Nt_layer; i += BLOCK_INCR_X) {
        int kk = center + (i - Nt_layer / 2);
        bool conj_it = false;
        int  k_use   = kk;
        if (kk < 0) { k_use = -kk; conj_it = true; }
        else if (kk > N_total / 2) { k_use = N_total - kk; conj_it = true; }
        cmplx v;
        if (k_use >= 0 && k_use < N_total) {
            int kf = k_use % Nf;
            int kt = k_use / Nf;
            v = fd_scratch[kf * Nt + kt];
        } else {
            v = cmplx(0.0, 0.0);
        }
        if (conj_it) v = cmplx(v.real(), -v.imag());
        // Match domains.py:790 -- divide by dt for the "before_ifft" scaling
        v = cmplx(v.real() / dt_data, v.imag() / dt_data);
        // Multiply by window
        slice[i] = cmplx(v.real() * window[i], v.imag() * window[i]);
    }
    CUDA_SYNC_THREADS;

    // 2. Inverse FFT of size Nt_layer (Bluestein for general even)
    bool Nt_pow2 = ((Nt_layer & (Nt_layer - 1)) == 0);
    int log2N_layer = 0; { int t = Nt_layer; while ((t >>= 1) != 0) ++log2N_layer; }
    if (Nt_pow2) {
        wdm_spline_radix2_fft(slice, Nt_layer, log2N_layer, true);
    } else {
        wdm_spline_bluestein_fft(slice, blu_work,
                                 chirp_n_layer, chirp_M_layer,
                                 Nt_layer, M_layer, log2M_layer, true);
    }
    CUDA_SYNC_THREADS;

    // 3. (m+n)-parity extraction on the *global* n grid.
    //    w_mn = kappa * sign((-1)^((m+1)*n_global)) *
    //                ( ((m + n_global) & 1) ? imag(IFFT[i]) : real(IFFT[i]) )
    //    Zero out the (m == 0 || m == Nf) && ((m + n_global) & 1) "edge" pixels
    //    -- the caller handles DC/Nyquist merging separately.
    for (int i = THREAD_START_X; i < Nt_layer; i += BLOCK_INCR_X) {
        int n_global = n_global_start + i;
        int parity   = (m + n_global) & 1;          // 0 -> real, 1 -> imag
        int phase_p  = ((m + 1) * n_global) & 1;     // 0 -> +1, 1 -> -1
        double val   = parity ? slice[i].imag() : slice[i].real();
        double sign  = phase_p ? -1.0 : +1.0;
        bool   edge_zero = ((m == 0) || (m == /*Nf*/-1)) && parity;  // Nf handled by caller
        if (edge_zero) {
            w_mn_out[i] = 0.0;
        } else {
            w_mn_out[i] = kappa * sign * val;
        }
    }
    CUDA_SYNC_THREADS;
}

#endif  // WDM_SPLINE_HELPERS_IMPLEMENTATION

#endif  // __WDM_SPLINE_HELPERS_HH__
