// chunked_het_bench.cxx
//
// Minimal standalone timing harness for the new shared-memory-only WDM
// chunked-heterodyne kernels in TDIonTheFly.cu:
//
//   wdm_het_get_ll_kernel        -> GBComputationGroup::gb_wdm_het_get_ll_wrap
//   wdm_het_fill_global_kernel   ->                       :: gb_wdm_het_fill_global_wrap
//   wdm_het_swap_ll_kernel       ->                       :: gb_wdm_het_swap_ll_wrap
//   wdm_het_get_fstat_ll_kernel  ->                       :: (TODO public wrap; called direct here)
//
// The harness uses *synthetic* inputs:
//
//   * Orbits: dense linear-equilateral mock (analytic, no astro fidelity).
//   * TDIConfig: a minimal 1-channel, 2-unit TDI definition.
//   * GB params: random in physically sensible ranges (f0 mid-band, etc.).
//   * data_d / invC: zero-filled (correctness invalid; timing exercises the
//     same code path as a real run).
//
// The script ``build_and_run.sh`` compiles this against the lisa-on-gpu
// source tree directly (no Python build, no install). On CPU it runs with
// the existing macros from ``GPUBackendTools/gbt_global.h`` collapsing the
// thread / block axes to single sequential loops.
//
// The point of this harness is per-kernel wall-clock timing, NOT
// correctness. Validate correctness via the existing CPU
// ``gb_chunked_test_script.py`` flow.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Include the kernel source directly. This pulls in TDIonTheFly.cu and its
// transitive dependencies (Detector.cu / LISAResponse.cu / etc.) -- we are a
// single-TU build keyed off the same source files the wheel uses, with no
// linking against installed wheels.
#include "TDIonTheFly.hh"
#include "Detector.hpp"
#include "LISAResponse.hh"

// -----------------------------------------------------------------------------
// Time helpers
// -----------------------------------------------------------------------------
struct Stopwatch {
    using clock = std::chrono::steady_clock;
    clock::time_point t0;
    void start() { t0 = clock::now(); }
    double stop_ms() {
        const auto t1 = clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// -----------------------------------------------------------------------------
// Synthetic input builders
// -----------------------------------------------------------------------------
// LISA equilateral-triangle analytic orbit at constant armlength L.
// Returns flat n_arr (sc_N, 6, 3) and ltt_arr (ltt_N, 6) and x_arr
// (sc_N, 3, 3) and link / sc_r / sc_e arrays.
static void build_mock_orbits(
    int sc_N, double sc_t0, double sc_dt,
    int ltt_N, double ltt_t0, double ltt_dt,
    std::vector<double>& n_arr,    // (sc_N, 6, 3)
    std::vector<double>& ltt_arr,  // (ltt_N, 6)
    std::vector<double>& x_arr,    // (sc_N, 3, 3)
    std::vector<int>&    links,    // (6,)
    std::vector<int>&    sc_r,     // (6,)
    std::vector<int>&    sc_e,     // (6,)
    double armlength)
{
    // Fixed equilateral triangle in xy plane, rotating slowly.
    const double L = armlength;
    n_arr.assign((size_t) sc_N * 6 * 3, 0.0);
    ltt_arr.assign((size_t) ltt_N * 6, L / Clight);
    x_arr.assign((size_t) sc_N * 3 * 3, 0.0);

    for (int i = 0; i < sc_N; ++i) {
        const double t  = sc_t0 + i * sc_dt;
        const double w  = 2.0 * M_PI / (3.16e7);   // 1 yr period
        const double th = w * t;
        // Spacecraft positions on equilateral triangle with rotation
        for (int sc = 0; sc < 3; ++sc) {
            const double phi = th + 2.0 * M_PI * sc / 3.0;
            x_arr[((size_t) i * 3 + sc) * 3 + 0] = L * cos(phi);
            x_arr[((size_t) i * 3 + sc) * 3 + 1] = L * sin(phi);
            x_arr[((size_t) i * 3 + sc) * 3 + 2] = 0.0;
        }
        // Unit normals along each link
        const int link_pairs[6][2] = {
            {0, 1}, {1, 2}, {2, 0}, {0, 2}, {1, 0}, {2, 1}};
        for (int l = 0; l < 6; ++l) {
            const int a = link_pairs[l][0];
            const int b = link_pairs[l][1];
            double dx[3];
            for (int k = 0; k < 3; ++k) {
                dx[k] = x_arr[((size_t) i * 3 + b) * 3 + k]
                       - x_arr[((size_t) i * 3 + a) * 3 + k];
            }
            const double r = sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
            for (int k = 0; k < 3; ++k) {
                n_arr[((size_t) i * 6 + l) * 3 + k] = dx[k] / r;
            }
        }
    }

    // Valid LISA link IDs in the order expected by Orbits::get_link_ind:
    //   12 -> 0, 23 -> 1, 31 -> 2, 13 -> 3, 32 -> 4, 21 -> 5.
    links = {12, 23, 31, 13, 32, 21};
    sc_r  = {2, 3, 1, 3, 2, 1};   // receiver spacecraft per link
    sc_e  = {1, 2, 3, 1, 3, 2};   // emitter spacecraft per link
}

// Minimal 2nd-gen-ish single-channel TDI definition: one unit of length 2
// per channel, 3 channels (X, Y, Z) using simple Michelson combinations.
static void build_mock_tdi_config(
    std::vector<int>&    unit_starts,
    std::vector<int>&    unit_lengths,
    std::vector<int>&    tdi_base_link,
    std::vector<int>&    tdi_link_combinations,
    std::vector<double>& tdi_signs_in,
    std::vector<int>&    channels,
    int& num_units, int& num_channels)
{
    num_channels = 3;
    num_units    = 3;
    // 3 single-unit TDI channels: each consists of a base link with one
    // combination link (no nested arms, no negative link IDs -- valid for
    // the orbit's get_link_ind path).
    unit_starts   = {0, 1, 2};
    unit_lengths  = {1, 1, 1};
    tdi_base_link = {12, 23, 31};
    tdi_link_combinations = {21, 32, 13};
    tdi_signs_in  = {+1.0, +1.0, +1.0};
    channels      = {0, 1, 2};
}

// -----------------------------------------------------------------------------
// Run the timing harness
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // ---- Knobs -------------------------------------------------------------
    int num_bin   = 1000;
    int n_chunks  = 16;
    int Nf        = 4096;
    int Nt_sub    = 256;
    int N_sparse  = 256;
    int Nt        = 1024;            // global WDM time dimension
    int nchannels = 3;
    int tdi_type  = TDI_XYZ;
    double dt     = 10.0;            // s (typical mojito sampling)
    double T_chunk_factor = 1.0;     // T_chunk = N_chunk_td * dt
    int repeats   = 3;
    int grid_dim  = 0;               // 0 -> default

    // Crude argv parser: --num_bin=N --grid_dim=N ...
    auto parse_int = [&] (const char* key, int& out) {
        for (int i = 1; i < argc; ++i) {
            const char* a = argv[i];
            const size_t klen = strlen(key);
            if (strncmp(a, key, klen) == 0 && a[klen] == '=') {
                out = atoi(a + klen + 1);
            }
        }
    };
    parse_int("--num_bin",   num_bin);
    parse_int("--n_chunks",  n_chunks);
    parse_int("--Nf",        Nf);
    parse_int("--Nt_sub",    Nt_sub);
    parse_int("--N_sparse",  N_sparse);
    parse_int("--Nt",        Nt);
    parse_int("--repeats",   repeats);
    parse_int("--grid_dim",  grid_dim);

    const int log2_N_sparse = (int) log2(N_sparse);
    const int log2_Nt_sub   = (int) log2(Nt_sub);
    const int N_chunk_td    = Nf * Nt_sub;
    const double T_chunk    = T_chunk_factor * N_chunk_td * dt;
    const double T          = (double) Nt * dt + (double) n_chunks * T_chunk;
    const int n_rfft_chunk  = N_chunk_td / 2 + 1;
    const int nparams       = 9;
    const int Nf_active     = (Nf < 256) ? Nf : 256;
    const int Nt_active     = Nt;

    printf("[bench] num_bin=%d n_chunks=%d Nf=%d Nt=%d Nt_sub=%d N_sparse=%d\n",
           num_bin, n_chunks, Nf, Nt, Nt_sub, N_sparse);
    printf("[bench] repeats=%d grid_dim=%d  Nf_active=%d Nt_active=%d\n",
           repeats, grid_dim, Nf_active, Nt_active);

    // ---- Build orbits + TDI config -----------------------------------------
    const int    sc_N  = 64, ltt_N = 64;
    const double sc_t0 = -0.5 * T,  sc_dt = T / (sc_N - 1);
    const double ltt_t0 = sc_t0,    ltt_dt = sc_dt;
    std::vector<double> n_arr, ltt_arr, x_arr;
    std::vector<int>    links, sc_r, sc_e;
    build_mock_orbits(sc_N, sc_t0, sc_dt, ltt_N, ltt_t0, ltt_dt,
                       n_arr, ltt_arr, x_arr, links, sc_r, sc_e,
                       2.5e9);

    Orbits orbits(sc_t0, sc_dt, sc_N, ltt_t0, ltt_dt, ltt_N,
                   n_arr.data(), ltt_arr.data(), x_arr.data(),
                   links.data(), sc_r.data(), sc_e.data(),
                   2.5e9);

    std::vector<int>    unit_starts, unit_lengths, tdi_base_link;
    std::vector<int>    tdi_link_combinations, channels;
    std::vector<double> tdi_signs_in;
    int num_units = 0, num_channels = 0;
    build_mock_tdi_config(unit_starts, unit_lengths, tdi_base_link,
                           tdi_link_combinations, tdi_signs_in, channels,
                           num_units, num_channels);
    TDIConfig tdi_config(unit_starts.data(), unit_lengths.data(),
                          tdi_base_link.data(), tdi_link_combinations.data(),
                          tdi_signs_in.data(), channels.data(),
                          num_units, num_channels);

    // WDMSettings: active band centered around mid-Nf.
    const int ind_min_t = 0;
    const int ind_max_t = Nt - 1;
    const int ind_min_f = (Nf - Nf_active) / 2;
    const int ind_max_f = ind_min_f + Nf_active - 1;
    const double layer_df = 1.0 / (2.0 * (double) Nf * dt);
    const double layer_dt = 2.0 * (double) Nf * dt;
    WDMSettings wdm_settings(layer_df, layer_dt, Nf, Nt, nchannels,
                              ind_min_t, ind_max_t, ind_min_f, ind_max_f);

    // ---- Allocate inputs / outputs (host-side) -----------------------------
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> uA(1e-22, 1e-21);
    std::uniform_real_distribution<double> uphi(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> uiota(0.0, M_PI);
    std::uniform_real_distribution<double> upsi(0.0, M_PI);
    std::uniform_real_distribution<double> ulam(-M_PI, M_PI);
    std::uniform_real_distribution<double> ubeta(-M_PI / 2, M_PI / 2);
    std::uniform_real_distribution<double> uf0(
        ((double) (ind_min_f + Nf_active / 4)) * layer_df,
        ((double) (ind_min_f + 3 * Nf_active / 4)) * layer_df);

    std::vector<double> params(num_bin * nparams);
    for (int b = 0; b < num_bin; ++b) {
        double* p = &params[b * nparams];
        p[0] = uA(rng);          // A
        p[1] = uf0(rng);         // f0
        p[2] = 1e-18;            // fdot
        p[3] = 0.0;              // fddot
        p[4] = uphi(rng);        // phi0
        p[5] = uiota(rng);       // iota
        p[6] = upsi(rng);        // psi
        p[7] = ulam(rng);        // lam
        p[8] = ubeta(rng);       // beta
    }

    std::vector<double> factors(num_bin, 1.0);
    std::vector<int>    data_idx (num_bin, 0);
    std::vector<int>    noise_idx(num_bin, 0);

    // Chunk geometry (uniform half-overlap chunks across [0, T))
    std::vector<double> chunk_t_starts(n_chunks);
    std::vector<int>    chunk_keep_lo (n_chunks);
    std::vector<int>    chunk_keep_hi (n_chunks);
    std::vector<int>    chunk_n_global_off(n_chunks);
    const int per_chunk_n = Nt / n_chunks;
    for (int j = 0; j < n_chunks; ++j) {
        chunk_t_starts[j]    = j * T_chunk - 0.5 * T;
        chunk_keep_lo[j]     = (j == 0)            ? 0       : (Nt_sub / 4);
        chunk_keep_hi[j]     = (j == n_chunks - 1) ? Nt_sub  : (3 * Nt_sub / 4);
        chunk_n_global_off[j] = j * per_chunk_n;
    }

    std::vector<double> wdm_window(Nt_sub, 1.0);  // rect; bench-only

    // Active-band data + invC (XYZ): zero-filled (correctness invalid)
    const size_t n_data_bytes =
        (size_t) nchannels * Nf_active * Nt_active * sizeof(double);
    const size_t n_invC_bytes =
        (size_t) nchannels * nchannels * Nf_active * Nt_active * sizeof(double);
    std::vector<double> data_d (n_data_bytes / sizeof(double), 0.0);
    std::vector<double> invC   (n_invC_bytes / sizeof(double), 0.0);
    for (size_t k = 0; k < invC.size(); ++k) invC[k] = 1.0;  // any nonzero

    // Outputs
    std::vector<double> d_h_out (num_bin, 0.0);
    std::vector<double> h_h_out (num_bin, 0.0);
    std::vector<double> dh_add  (num_bin, 0.0);
    std::vector<double> dh_rem  (num_bin, 0.0);
    std::vector<double> aa      (num_bin, 0.0);
    std::vector<double> rr      (num_bin, 0.0);
    std::vector<double> ar      (num_bin, 0.0);
    std::vector<double> template_fill(
        (size_t) nchannels * Nf * Nt, 0.0);

    // Layer-group stubs (kernels accept-and-ignore in the new design)
    std::vector<int> binary_perm  (num_bin, 0);
    std::vector<int> group_starts (1, 0);
    std::vector<int> group_ends   (1, 0);
    std::vector<int> group_m_lo   (1, 0);
    std::vector<int> group_m_hi   (1, 0);
    std::vector<int> pair_m_lo_b  (1, 0);
    std::vector<int> pair_m_hi_b  (1, 0);
    const int n_groups = 0;

    // ---- Timing loops ------------------------------------------------------
    GBComputationGroup comp;
    Stopwatch sw;

    auto bench = [&] (const char* name, auto&& fn) {
        // Warm-up
        fn();
        double best_ms = 1e30, sum_ms = 0.0;
        for (int r = 0; r < repeats; ++r) {
            sw.start();
            fn();
            const double ms = sw.stop_ms();
            sum_ms  += ms;
            if (ms < best_ms) best_ms = ms;
        }
        const double avg_ms = sum_ms / repeats;
        printf("[bench] %-28s  best=%9.2f ms   avg=%9.2f ms   per_binary=%7.2f us\n",
                name, best_ms, avg_ms, avg_ms / num_bin * 1000.0);
    };

    bench("get_ll_wdm", [&] () {
        comp.gb_wdm_het_get_ll_wrap(
            d_h_out.data(), h_h_out.data(),
            &orbits, &tdi_config, &wdm_settings,
            params.data(), data_idx.data(), noise_idx.data(),
            chunk_t_starts.data(), chunk_keep_lo.data(), chunk_keep_hi.data(),
            chunk_n_global_off.data(), wdm_window.data(),
            data_d.data(), invC.data(),
            n_chunks, num_bin, nparams,
            Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
            nchannels, n_rfft_chunk,
            T_chunk, dt, T, /*t_ref=*/0.0, tdi_type,
            /*tukey_alpha=*/-1.0, grid_dim, /*N_cp_sig=*/0, /*N_cp_orbit=*/0,
            binary_perm.data(), group_starts.data(), group_ends.data(),
            group_m_lo.data(), group_m_hi.data(), n_groups);
    });

    bench("fill_global_wdm", [&] () {
        // Re-zero template each call (caller's contract; cheap).
        std::memset(template_fill.data(), 0,
                     template_fill.size() * sizeof(double));
        comp.gb_wdm_het_fill_global_wrap(
            template_fill.data(),
            &orbits, &tdi_config, &wdm_settings,
            params.data(), factors.data(),
            chunk_t_starts.data(), chunk_keep_lo.data(), chunk_keep_hi.data(),
            chunk_n_global_off.data(), wdm_window.data(),
            n_chunks, num_bin, nparams,
            Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
            nchannels, n_rfft_chunk,
            T_chunk, dt, T, /*t_ref=*/0.0,
            /*tukey_alpha=*/-1.0, grid_dim, /*N_cp_sig=*/0, /*N_cp_orbit=*/0);
    });

    // swap_ll: feed two copies of params as add + rem
    bench("swap_ll_wdm", [&] () {
        comp.gb_wdm_het_swap_ll_wrap(
            dh_add.data(), dh_rem.data(),
            aa.data(), rr.data(), ar.data(),
            &orbits, &tdi_config, &wdm_settings,
            params.data(), params.data(),
            data_idx.data(), noise_idx.data(),
            chunk_t_starts.data(), chunk_keep_lo.data(), chunk_keep_hi.data(),
            chunk_n_global_off.data(), wdm_window.data(),
            data_d.data(), invC.data(),
            n_chunks, num_bin, nparams,
            Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
            nchannels, n_rfft_chunk,
            T_chunk, dt, T, /*t_ref=*/0.0, tdi_type,
            /*tukey_alpha=*/-1.0, grid_dim, /*N_cp_sig=*/0, /*N_cp_orbit=*/0,
            binary_perm.data(), group_starts.data(), group_ends.data(),
            group_m_lo.data(), group_m_hi.data(), n_groups,
            pair_m_lo_b.data(), pair_m_hi_b.data());
    });

    // F-stat: now via the public wrap on GBComputationGroup.
    std::vector<double> N_re((size_t) num_bin * 4,  0.0);
    std::vector<double> N_im((size_t) num_bin * 4,  0.0);
    std::vector<double> M_re((size_t) num_bin * 10, 0.0);
    std::vector<double> M_im((size_t) num_bin * 10, 0.0);
    bench("get_fstat_ll_wdm", [&] () {
        comp.gb_wdm_het_get_fstat_ll_wrap(
            N_re.data(), N_im.data(), M_re.data(), M_im.data(),
            &orbits, &tdi_config, &wdm_settings,
            params.data(), data_idx.data(), noise_idx.data(),
            chunk_t_starts.data(), chunk_keep_lo.data(), chunk_keep_hi.data(),
            chunk_n_global_off.data(), wdm_window.data(),
            data_d.data(), invC.data(),
            n_chunks, num_bin, nparams,
            Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
            nchannels, n_rfft_chunk,
            T_chunk, dt, T, /*t_ref=*/0.0, tdi_type,
            /*tukey_alpha=*/-1.0, grid_dim, /*m_band_half_width=*/1);
    });

    printf("[bench] done\n");
    return 0;
}
