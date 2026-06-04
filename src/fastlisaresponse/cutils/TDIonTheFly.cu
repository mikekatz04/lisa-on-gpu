#include "TDIonTheFly.hh"
#include "Detector.hpp"
#include "LISAResponse.hh"
#include "Interpolate.hh"
// LAT-owned chunked-het primitives header (Phase 3L.7a, 2026-06-04+).
// Slices 1, 2a, 2b, 3-prep all consumed from here:
// - FAST_WDM_* macros + WDMHet*Bufs PODs + arena macros (slice 1)
// - WDM FFT machinery via lat_wdm_fft.hh (slice 2a)
// - chunked-het helpers (wdm_fit_cubic_spline,
//   populate_orbit_spline_cache, fast_wdm_inner_heterodyne_*,
//   gb_chunk_fd_to_wdm) (slice 2b)
// - NUM_THREADS_HERE + FAST_WDM_K_PER_THREAD_MAX (slice 3-prep)
#include "lat_chunked_het_kernels.hh"
#define WDM_SPLINE_HELPERS_IMPLEMENTATION
#include "WDMSplineHelpers.hh"
#include <string>
#include <unistd.h>
#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif

// TODO: GET RID OF THIS ??!!!
#define C_SI 299792458.;

// NUM_THREADS_HERE block-size knob moved to LAT at Phase 3L.7a slice
// 3-prep (2026-06-04); see lat_chunked_het_kernels.hh included near
// the top of this file. Definition is preserved as 128 (GPU) / 1
// (CPU) -- same value used by every chunked-het + spline-path kernel
// in this file.

// === LISATDIonTheFly method bodies moved to LAT at Phase 3L.5 (2026-06-03) ===
// All ~26 method bodies now live in
//   LISAanalysistools/src/lisatools/cutils/lat_tdi_on_the_fly.cu
// (copy-compiled in-place by this repo's CMakeLists, same pattern
// as LISAResponse.cu since Phase 3E). The blocks below are retained
// as #if 0 for historical reference and will be removed once the
// carve-out is fully validated.
#if 0
CUDA_DEVICE
LISATDIonTheFly::~LISATDIonTheFly()
{
    return;
}
#endif  // === end LISATDIonTheFly dtor moved to LAT ===

// CUDA_DEVICE
// void LISATDIonTheFly::get_t_tdi(double *t_out, double *kr, double *Larm, double t, int a, int b, int c, int n)
// {
  
//     t_out[0] = t - kr[a] - 2.0 * Larm[c] - 2.0 * Larm[b];
//     t_out[1] = t- kr[b] - Larm[c] - 2.0 * Larm[b];
//     t_out[2] = t - kr[c]-Larm[b]-2.0*Larm[c];
//     t_out[3] = t - kr[a]-2.0*Larm[b];
//     t_out[4] = t - kr[a]-2.0*Larm[c];
//     t_out[5] = t - kr[c]- Larm[b];
//     t_out[6] = t - kr[b]- Larm[c];
//     t_out[7] = t - kr[a];

//     // for (int i = 0; i < 8; i += 1) printf("CHECKCHECK: %d %e, %e\n", i, t, t_out[i]);
      
// }

// CUDA_DEVICE
// static void hplus_and_hcross(double t, double phase, double amp,  double Aplus, double Across, double cos2psi, double sin2psi,  double *hp, double *hc, double *hpf, double *hcf)
// {
//     double cp = cos(phase);
//     double sp = sin(phase);

//     *hp  = amp * ( Aplus*cos2psi*cp  + Across*sin2psi*sp );
//     *hc  = amp * ( Across*cos2psi*sp - Aplus*sin2psi*cp  );
    
//     *hpf = amp * (-Aplus*cos2psi*sp  + Across*sin2psi*cp );
//     *hcf = amp * ( Across*cos2psi*cp + Aplus*sin2psi*sp  );              
// }


// CUDA_DEVICE
// void LISATDIonTheFly::get_tdi_sub(cmplx *M, int n, int N, int a, int b, int c, double t_orig, double* tarray, double *amp_tdi_vals, double *phase_tdi_vals, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm)
// {
//     double t, f, amp, phase;
//     double hp, hc, hpf, hcf;

//     M[n] = 0.0;
    
//     // if(freq_spline) /* mbh */
//     // {
//     //     //For TDI we want the overall amplitude scaled out of the waveform as it passes through zero
//     //     amp = 1.0; //spline_interpolation(amp_spline,  tarray[n]);
//     //     f   = spline_interpolation(freq_spline, tarray[n]);
//     // }

//     // first index
//     t = tarray[0];
//     amp = amp_tdi_vals[0];
//     phase = phase_tdi_vals[0];
    
//     //  if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
//     hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
//     // printf("%d %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e \n", n, t_orig, t, hp, hc, hpf, hcf, phase, Aplus, Across, cos2psi, sin2psi);
              
//     // M[n] += hp*Apm[b]+hc*Acm[b];
//     // M[n] -= hp*App[c]+hc*Acp[c];
//     // Mf[n] += hpf*Apm[b]+hcf*Acm[b];
//     // Mf[n] -= hpf*App[c]+hcf*Acp[c];
//     cmplx I(0.0, 1.0);
//     // printf("CHECKING: %d %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", n, t, phase, amp, Aplus, Across, Apm[b], Acm[b]);
    
//     M[n] += (hp*Apm[b]+hc*Acm[b]) + I * (hpf*Apm[b]+hcf*Acm[b]);
//     M[n] -= (hp*App[c]+hc*Acp[c]) + I * (hpf*App[c]+hcf*Acp[c]);
    
//     t = tarray[1];
//     amp = amp_tdi_vals[1];
//     phase = phase_tdi_vals[1];
    
//     // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
//     hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
//     // M[n] -= hp*Apm[c]+hc*Acm[c];
//     // M[n] += hp*App[c]+hc*Acp[c];
//     // Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
//     // Mf[n] += hpf*App[c]+hcf*Acp[c];
//     M[n] -= (hp*Apm[c]+hc*Acm[c]) + I * (hpf*Apm[c]+hcf*Acm[c]);
//     M[n] += (hp*App[c]+hc*Acp[c]) + I * (hpf*App[c]+hcf*Acp[c]);
    
//     t = tarray[2];
//     amp = amp_tdi_vals[2];
//     phase = phase_tdi_vals[2];
//     // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
//     hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
//     // M[n] += hp*App[b]+hc*Acp[b];
//     // M[n] -= hp*Apm[b]+hc*Acm[b];
//     // Mf[n] += hpf*App[b]+hcf*Acp[b];
//     // Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
//     M[n] += (hp*App[b]+hc*Acp[b]) + I * (hpf*App[b]+hcf*Acp[b]);
//     M[n] -= (hp*Apm[b]+hc*Acm[b]) + I * (hpf*Apm[b]+hcf*Acm[b]);

//     t = tarray[3];
//     amp = amp_tdi_vals[3];
//     phase = phase_tdi_vals[3];
//     // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
//     hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
//     // M[n] -= hp*Apm[b]+hc*Acm[b];
//     // M[n] += hp*Apm[c]+hc*Acm[c];
//     // Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
//     // Mf[n] += hpf*Apm[c]+hcf*Acm[c];
//     M[n] -= (hp*Apm[b]+hc*Acm[b]) + I * (hpf*Apm[b]+hcf*Acm[b]);
//     M[n] += (hp*Apm[c]+hc*Acm[c]) + I * (hpf*Apm[c]+hcf*Acm[c]);
    
//     t = tarray[4];
//     amp = amp_tdi_vals[4];
//     phase = phase_tdi_vals[4];
//     // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
//     hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
//     // M[n] += hp*App[c]+hc*Acp[c];
//     // M[n] -= hp*App[b]+hc*Acp[b];
//     // Mf[n] += hpf*App[c]+hcf*Acp[c];
//     // Mf[n] -= hpf*App[b]+hcf*Acp[b];
//     M[n] += (hp*App[c]+hc*Acp[c]) + I * (hpf*App[c]+hcf*Acp[c]);
//     M[n] -= (hp*App[b]+hc*Acp[b]) + I * (hpf*App[b]+hcf*Acp[b]);
    
//     t = tarray[5];
//     amp = amp_tdi_vals[5];
//     phase = phase_tdi_vals[5];
//     // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
//     hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
//     // M[n] -= hp*App[b]+hc*Acp[b];
//     // M[n] += hp*Apm[b]+hc*Acm[b];
//     // Mf[n] -= hpf*App[b]+hcf*Acp[b];
//     // Mf[n] += hpf*Apm[b]+hcf*Acm[b];
//     M[n] -= (hp*App[b]+hc*Acp[b]) + I * (hpf*App[b]+hcf*Acp[b]);
//     M[n] += (hp*Apm[b]+hc*Acm[b]) + I * (hpf*Apm[b]+hcf*Acm[b]);
    
//     t = tarray[6];
//     amp = amp_tdi_vals[6];
//     phase = phase_tdi_vals[6];
//     // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
//     hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
//     // M[n] += hp*Apm[c]+hc*Acm[c];
//     // M[n] -= hp*App[c]+hc*Acp[c];
//     // Mf[n] += hpf*Apm[c]+hcf*Acm[c];
//     // Mf[n] -= hpf*App[c]+hcf*Acp[c];
//     M[n] += (hp*Apm[c]+hc*Acm[c]) + I * (hpf*Apm[c]+hcf*Acm[c]);
//     M[n] -= (hp*App[c]+hc*Acp[c]) + I * (hpf*App[c]+hcf*Acp[c]);
    
//     t = tarray[7];
//     amp = amp_tdi_vals[7];
//     phase = phase_tdi_vals[7];
//     // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
//     hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
//     // M[n] -= hp*Apm[c]+hc*Acm[c];
//     // M[n] += hp*App[b]+hc*Acp[b];
//     // Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
//     // Mf[n] += hpf*App[b]+hcf*Acp[b];  
//     M[n] -= (hp*Apm[c]+hc*Acm[c]) + I * (hpf*Apm[c]+hcf*Acm[c]);
//     M[n] += (hp*App[b]+hc*Acp[b]) + I * (hpf*App[b]+hcf*Acp[b]);

//     // printf("WAAAAT: %d %.12e %.12e %.12e %.12e \n", n, t_orig, t, M[n].real(), M[n].imag());

// }


// // void LISA_polarization_tensor(double costh, double phi, double eplus[4][4], double ecross[4][4], double k[4])
// CUDA_DEVICE
// void LISATDIonTheFly::LISA_polarization_tensor(double costh, double phi, double *eplus, double *ecross, double *k)
// {
//     /*   Gravitational Wave basis vectors   */
//     double u[3],v[3];

//     /*   Sky location angles   */
//     double sinth = sqrt(1.0 - costh*costh);
//     double cosph = cos(phi);
//     double sinph = sin(phi);

    
//     /*   Tensor construction for building slowly evolving LISA response   */
//     //Gravitational Wave source basis vectors
//     u[0] =  costh*cosph;  u[1] =  costh*sinph;  u[2] = -sinth;
//     v[0] =  sinph;        v[1] = -cosph;        v[2] =  0.;
//     k[0] = -sinth*cosph;  k[1] = -sinth*sinph;  k[2] = -costh;
    
//     // printf("CHECKECKECEKCEK: %e %e %e %e %e\n", k[0], k[1], k[2], costh, cosph);

//     //GW polarization basis tensors
//     /*
//      * LDC convention:
//      * https://gitlab.in2p3.fr/LISA/LDC/-/blob/develop/ldc/waveform/fastGB/GB.cc
//      */
//     for(int i=0; i<3;i++)
//     {
//         for(int j=0; j<3;j++)
//         {
//             eplus[i * 3 + j]  = v[i]*v[j] - u[i]*u[j];
//             ecross[i * 3 + j] = u[i]*v[j] + v[i]*u[j];
//         }
//     }

// }


// CUDA_DEVICE
// void LISATDIonTheFly::get_tdi_n(cmplx *X, cmplx *Y, cmplx *Z, double* phi_ref, double *params, double t, int m, int N, double costh, double phi, double cosi, double psi, int bin_i)
// {
//     // printf("bef: inside: %d \n", m);
//     /*   Indicies    */
//     int i,j;

//     Vec k; 
//     double kdotr[3], kdotn[3];
//     double dplus[3], dcross[3];
    
//     /*   Polarization basis tensors   */
//     double eplus[3 * 3], ecross[3 * 3];
    
//     double cos2psi, sin2psi;
//     double Aplus, Across;
//     double App[3], Apm[3], Acp[3], Acm[3];
//     double t_tdi[8], amp_tdi[8], phase_tdi[8];
    
//     /*   Allocating Arrays   */
    
//     // TODO: move outside this function
//     cos2psi = cos(2.*psi);  
//     sin2psi = sin(2.*psi);
    
//     Aplus = 0.5*(1.+cosi*cosi);
//     Across = -cosi;

//     LISA_polarization_tensor(costh, phi, &eplus[0], &ecross[0], &k[0]);

//     Vec x[3]; 
//     Vec n[3]; // TODO: What about other direction down arms?
//     double L[3];
//     int sc, link;

//     // position vectors for each spacecraft (converted to seconds)
//     for(i=0; i<3; i++)
//     {   
//         sc = i + 1;
//         x[i] = orbits->get_pos(t, sc) / C_SI;
//         // x[i][0] = spline_interpolation(orbit->dx[i],t)/CLIGHT;
//         // x[i][1] = spline_interpolation(orbit->dy[i],t)/CLIGHT;
//         // x[i][2] = spline_interpolation(orbit->dz[i],t)/CLIGHT;
//         // printf("WHATAA?:%.12e %.12e %.12e %.12e\n", t, x[i].x, x[i].y, x[i].z);
//     }

//     n[0] = x[1] - x[2];
//     n[1] = x[2] - x[0];
//     n[2] = x[0] - x[1];

//     // arm vectors
//     for(i=0; i<3; i++)
//     {   
//         link = orbits->get_link_from_arr(i);
//         // printf("need TO CHECK THIS. Probably wrong link order?.\n");

//         // n[i] = orbits->get_normal_unit_vec(t, link); 
//         // L[i] = orbits->get_light_travel_time(t, link);

//         L[i] = n[i].dot(n[i]);
//         L[i] = sqrt(L[i]);

//         n[i] = n[i] /  L[i];
//         // n[0][i] = x[1][i] - x[2][i];
//         // n[1][i] = x[2][i] - x[0][i];
//         // n[2][i] = x[0][i] - x[1][i];
//     }

//     // //arm lengths
//     // for(i=0; i<3; i++) 
//     // {
//     //     // L[i]=0.0;
//     //     // for(j=0; j<3; j++)  L[i] += n[i][j]*n[i][j];
//     //     // L[i] = sqrt(L[i]);
//     //     sc = i + 1;
        
//     // }

//     //normalize arm vectors  
//     // Orbit class vectors are normalized.      
//     // for(i=0; i<3; i++)
//     //     for(j=0;j<3;j++)
//     //         n[i][j] /= L[i];
    


//     // k dot r_i (source direction spacecraft locations)
//     for(i=0; i<3; i++)
//     {
//         kdotr[i] = k.dot(x[i]);
//         // for (int j= 0; j<3; j+=1)
//         //     printf("CHECK K: %d %d %d %e %e %e\n", m, i, j, kdotr[i], x[i][j], k[j]);
              
//     }
    
//     // k dot n_i (source direction w/ arm vectors)
//     for(i=0; i<3; i++)
//     {
//         kdotn[i] = k.dot(n[i]);
//     }        
    
//     //Antenna primitives
//     for(i=0; i<3; i++)
//     {
//         dplus[i] = 0.0;
//         dcross[i] = 0.0;
//         for(j=0; j<3; j++)
//         {
//             for(int l=0; l<3; l++)
//             {
//                 dplus[i]  += (n[i][j] * n[i][l]) * eplus[j * 3 + l];
//                 dcross[i] += (n[i][j] * n[i][l]) * ecross[j * 3 + l];
//             }
//         }
//     }
    
//     //Full Antenna patterns
//     for(i=0; i<3; i++)
//     {
//         App[i] = 0.5 * dplus[i]  / (1.0 + kdotn[i]);
//         Apm[i] = 0.5 * dplus[i]  / (1.0 - kdotn[i]);
//         Acp[i] = 0.5 * dcross[i] / (1.0 + kdotn[i]);
//         Acm[i] = 0.5 * dcross[i] / (1.0 - kdotn[i]);
//     }
//     // printf("af2: inside: %e %e %e %e %e\n", t, dplus[0], dcross[0], kdotn[0], n[0].x);
    
//     double time_sc = t - kdotr[0];
//     get_phase_ref(t, time_sc, &phi_ref[0], &params[0], 1, bin_i, m);

//     get_t_tdi(&t_tdi[0], &kdotr[0], &L[0], t, 0, 1, 2, m);
//     get_amp_and_phase(t, &t_tdi[0], &amp_tdi[0], &phase_tdi[0], &params[0], 8, bin_i);
//     // printf("af33: inside: %d %e \n", m, phi_ref[m]);
//     get_tdi_sub(X, m, N, 0, 1, 2, t, &t_tdi[0], &amp_tdi[0], &phase_tdi[0], Aplus, Across, cos2psi, sin2psi, &App[0], &Apm[0], &Acp[0], &Acm[0], &kdotr[0], &L[0]);

//     // printf("af44: inside: %d %.12e %.12e %.12e \n", m, X[m].real(), X[m].imag(), phi_ref[m]);
    
//     get_t_tdi(&t_tdi[0], &kdotr[0], &L[0], t, 1, 2, 0, m);
//     get_amp_and_phase(t, &t_tdi[0], &amp_tdi[0], &phase_tdi[0], &params[0], 8, bin_i);
//     get_tdi_sub(Y, m, N, 1, 2, 0, t, &t_tdi[0], &amp_tdi[0], &phase_tdi[0], Aplus, Across, cos2psi, sin2psi, &App[0], &Apm[0], &Acp[0], &Acm[0], &kdotr[0], &L[0]);

//     // printf("af4: inside: %d \n", m);
    
//     get_t_tdi(&t_tdi[0], &kdotr[0], &L[0], t, 2, 0, 1, m);
//     get_amp_and_phase(t, &t_tdi[0], &amp_tdi[0], &phase_tdi[0], &params[0], 8, bin_i);
//     get_tdi_sub(Z, m, N, 2, 0, 1, t, &t_tdi[0], &amp_tdi[0], &phase_tdi[0], Aplus, Across, cos2psi, sin2psi, &App[0], &Apm[0], &Acp[0], &Acm[0], &kdotr[0], &L[0]);

//     // printf("af66: inside: %d \n", m);
    
// }


// === WDMDomain method bodies moved to LAT at Phase 3L (2026-06-02) ===
// All 12 method bodies are now header-inline in
//   LISAanalysistools/src/lisatools/cutils/wdm_domain.hh
// The block below is retained as `#if 0` for historical reference and
// will be removed once the carve-out is fully validated.
#if 0
CUDA_DEVICE
int WDMDomain::get_pixel_index(int m, int n, int channel, int data_index)
{
    if (data_index >= num_data)
    {
#ifdef __CUDACC__  
#else
        throw std::invalid_argument("data_index is larger than available data instances.");
#endif
    }
    return ((data_index * num_channel + channel) * Nf_active + (m - ind_min_f)) * Nt_active + (n - ind_min_t);
}

CUDA_DEVICE
int WDMDomain::get_pixel_index_noise(int m, int n, int channel, int noise_index)
{
    if (noise_index >= num_noise)
    {
#ifdef __CUDACC__  
#else
        throw std::invalid_argument("noise_index is larger than available noise instances.");
#endif
    }
    return ((noise_index * num_channel + channel) * Nf_active + (m - ind_min_f)) * Nt_active + (n - ind_min_t);
}

CUDA_DEVICE
int WDMDomain::get_pixel_index_noise_cross_channel(int m, int n, int channel_i, int channel_j, int noise_index)
{
    return (((noise_index * num_channel + channel_i) * num_channel + channel_j) * Nf_active + (m - ind_min_f)) * Nt_active + (n - ind_min_t);
}

CUDA_DEVICE
double WDMDomain::get_pixel_data_value(int m, int n, int channel,  int data_index)
{
    return wdm_data[get_pixel_index(m, n, channel, data_index)];
}

CUDA_DEVICE
double WDMDomain::get_pixel_noise_value(int m, int n, int channel, int noise_index)
{
    return wdm_noise[get_pixel_index_noise(m, n, channel, noise_index)];
}

CUDA_DEVICE
double WDMDomain::get_pixel_noise_value_cross_channel(int m, int n, int channel_i, int channel_j, int noise_index)
{
    return wdm_noise[get_pixel_index_noise_cross_channel(m, n, channel_i, channel_j, noise_index)];
}

CUDA_DEVICE
void WDMDomain::get_inner_product_value(double *d_h, double *h_h, double wdm_template_nm, int m, int n, int channel, int data_index, int noise_index)
{
    double wdm_data_nm = get_pixel_data_value(m, n, channel, data_index);
    double wdm_noise_nm = get_pixel_noise_value(m, n, channel, noise_index);
    double val_d_h = wdm_data_nm * wdm_template_nm * wdm_noise_nm * 0.25;
    double val_h_h = wdm_template_nm * wdm_template_nm * wdm_noise_nm * 0.25;
    
    *d_h = val_d_h;
    *h_h = val_h_h;
}

CUDA_DEVICE
void WDMDomain::get_inner_product_value_cross_channel(double *d_h, double *h_h, double wdm_template_nm_i, double wdm_template_nm_j, int m, int n, int channel_i, int channel_j, int data_index, int noise_index)
{
    // assume data is channel_i, template is channel_j
    // printf("CHECK14 %d %d %d %d\n", n, m, channel_i, channel_j);
    
    double wdm_data_nm_i = get_pixel_data_value(m, n, channel_i, data_index);
    // printf("CHECK15 %d %d %d %d\n", n, m, channel_i, channel_j);
    
    double wdm_noise_nm_ij = get_pixel_noise_value_cross_channel(m, n, channel_i, channel_j, noise_index);
    // printf("CHECK16 %d %d %d %d %e %e %e\n", n, m, channel_i, channel_j, wdm_data_nm_i, wdm_template_nm_j, wdm_noise_nm_ij);
    // if ((n == 1000) & (channel_i == 0) && (channel_j == 0)) printf("CHECHCHECK: %d %d %d %d %e %e %e %e\n", m, n, channel_i, channel_j, wdm_template_nm_i, wdm_template_nm_j, wdm_data_nm_i, wdm_noise_nm_ij);
    
    // 0.25 factor is needed. Check python code
    double val_d_h = wdm_data_nm_i * wdm_template_nm_j * wdm_noise_nm_ij * 0.25;
    double val_h_h = wdm_template_nm_i * wdm_template_nm_j * wdm_noise_nm_ij * 0.25;
    // printf("CHECK16 %e %e %d %d %d %d %e %e %e\n", val_d_h, val_h_h, n, m, channel_i, channel_j, wdm_data_nm_i, wdm_template_nm_j, wdm_noise_nm_ij);
    
    *d_h = val_d_h;
    *h_h = val_h_h;
}

#endif  // === end WDMDomain method bodies (first segment) moved to LAT ===

#if 0  // === WaveletLookupTable disabled at Phase 3L (2026-06-02) -- lookup-table spline path retiring ===
CUDA_DEVICE
double WaveletLookupTable::linear_interp(double f_scaled, double fdot, double *z_vals, int layer_n)
{
    // PER_N      table is (Nt, num_fdot, num_f) — offset by layer_n.
    // N_REF_ONLY table is     (num_fdot, num_f) — no per-n axis; the
    //            (-1)^(layer_n - n_ref) correction is applied in
    //            get_w_mn_lookup, not here.
    double *z_slice = z_vals;
    if (kind == LOOKUP_PER_N) {
        z_slice += (size_t)layer_n * (size_t)num_fdot * (size_t)num_f;
    }

    if (num_fdot > 1)
    {
        int f_index = int((f_scaled - min_f_scaled) / df_interp) ;
        int fdot_index = int((fdot - min_fdot) / dfdot_interp) ;
        bool bad = false;

        // printf("CHECK18 %e %e %d %d %d %d %e %e %e %e\n", f_scaled, fdot, f_index, fdot_index, num_f, num_fdot, df_interp, dfdot_interp, min_f_scaled, min_fdot);

        if ((f_index < 0) || (f_index >= num_f) || (fdot_index < 0) || (fdot_index >= num_fdot))
        {
            bad = true;
    #ifdef __CUDACC__
            f_index = 0;
            fdot_index = 0;

    #else
            // throw std::invalid_argument("Asking for value outside interp domain.");
    #endif
        }

        if (bad)
        {
            return 0.0;
        }
        double x1 = df_interp * f_index;
        double x2 = df_interp * (f_index + 1);
        double y1 = df_interp * fdot_index;
        double y2 = df_interp * (fdot_index + 1);

        double z11 = z_slice[fdot_index * num_f + f_index];
        double z12 = z_slice[(fdot_index + 1) * num_f + f_index];
        double z21 = z_slice[fdot_index * num_f + (f_index + 1)];
        double z22 = z_slice[(fdot_index + 1) * num_f + (f_index + 1)];

        double f_x_y1 = (x2 - f_scaled) / (x2 - x1) * z11 + (f_scaled - x1) / (x2 - x1) * z21;
        double f_x_y2 = (x2 - f_scaled) / (x2 - x1) * z21 + (f_scaled - x1) / (x2 - x1) * z22;

        double f_xy = (y2 - fdot) / (y2 - y1) * f_x_y1 + (fdot - y1) / (y2 - y1) * f_x_y2;
        return f_xy;
    }
    else
    {
        int f_index = int((f_scaled - min_f_scaled) / df_interp) ;
        double x1 = (df_interp * f_index) + min_f_scaled;
        double x2 = (df_interp * (f_index + 1)) + min_f_scaled;
        double z1 = z_slice[f_index];
        double z2 = z_slice[f_index + 1];

        double f_y = z1 + (f_scaled - x1) * (z2 - z1) / (x2 - x1);
        return f_y;
    }
}

CUDA_DEVICE
double WaveletLookupTable::get_w_mn_lookup(cmplx tdi_channel_val, double f, double fdot, int layer_m, int layer_n)
{
    double f_scaled = f - layer_m * layer_df;
    double _c_nm = linear_interp(f_scaled, fdot, c_nm_all, layer_n);
    double _s_nm = linear_interp(f_scaled, fdot, s_nm_all, layer_n);
    double c_nm, s_nm;

    // N_REF_ONLY: the table was built at a single (m_ref, n_ref) pixel,
    // so n-translation is recovered via (-1)^(layer_n - n_ref) on both
    // sin and cos coefficients. (Exact for fdot=0; an approximation for
    // chirp — see Plan A notes in WDM_FDOT_LOOKUP_PLAN.md.) PER_N tables
    // already carry the per-n information so no dn-sign is needed.
    if (kind == LOOKUP_N_REF_ONLY)
    {
        if (((layer_n - n_ref) & 1) != 0)
        {
            _c_nm = -_c_nm;
            _s_nm = -_s_nm;
        }
    }

    bool is_m_plus_n_even = (layer_m + layer_n) % 2 == 0;

    // Build pre-applies an (m+n)-parity sin/cos swap to the table.
    // Lookup undoes it based on the LOOKUP pixel's (layer_m + layer_n) parity:
    //   (m+n) odd  → no swap (table already has the swapped values).
    //   (m+n) even → swap to undo the build swap.
    if (!is_m_plus_n_even)
    {
        s_nm = _s_nm;
        c_nm = _c_nm;
    }
    else
    {
        s_nm = _c_nm;
        c_nm = _s_nm;
    }

    double w_mn = c_nm * tdi_channel_val.real() + s_nm * tdi_channel_val.imag();

    // m-parity correction. The Python build pre-multiplies each m_diff
    // block by (-1)^(m_diff_build) so that linear_interp across f_norm
    // block boundaries is smooth. The block reached at lookup is
    // m_diff_build = -m_diff_eval = m_source - layer_m, so the build flip
    // imprints (-1)^(m_source - layer_m) onto the looked-up value.
    // Combined with the original FFT-mirror correction
    // (-1)^(m_source - m_ref), the net eval sign is
    // (-1)^((layer_m - m_ref) parity). Applies to BOTH per_n and
    // n_ref_only builds — see get_wdm_coeffs in lisatools/domains.py.
    if (((layer_m - m_ref) & 1) != 0)
    {
        w_mn = -w_mn;
    }

    return w_mn;
}

CUDA_DEVICE
double WaveletLookupTable::get_wdm_in_channel_over_layers(cmplx tdi_channel_val, double f, double fdot, int m, int n)
{
    // printf("CHECK66 %d %e %e %e %e %e\n", n, f[0], f[1], f[2], avg_f, wdm->layer_df);

    if ((m >= ind_min_f) && (m < ind_max_f))
    {
        // for (int layer_m = layer_m_here; layer_m <= layer_m_here; layer_m += 1)
        return get_w_mn_lookup(tdi_channel_val, f, fdot, m, n);
    }
    else
    {
        return 0.0;
    }
}
#endif  // === end WaveletLookupTable disabled ===

// === WDMDomain method bodies (second segment) moved to LAT at Phase 3L ===
#if 0
CUDA_DEVICE
void WDMDomain::add_ip_contrib(double *d_h_tmp, double *h_h_tmp, double *w_mn, int layer_m, int n, int data_index, int noise_index, int tdi_type)
{
#ifdef __CUDACC__
    int tid = threadIdx.x;
#else
    int tid = 0;
#endif

    // printf("CHECK11 %d %d\n", n, layer_m);

    double d_h_val = 0.0;
    double h_h_val = 0.0;
    if (tdi_type == TDI_XYZ)
    {
        for (int channel_i = 0; channel_i < 3; channel_i += 1)
        {
            for (int channel_j = 0; channel_j < 3; channel_j += 1)
            {

                // TODO: change from 9 to 6 calculations?
                get_inner_product_value_cross_channel(&d_h_val, &h_h_val, w_mn[channel_i], w_mn[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);                
                d_h_tmp[tid] += d_h_val;
                h_h_tmp[tid] += h_h_val;  
                
            }
        } 
    }
    else if (tdi_type == TDI_AET)
    {
        // AET: three orthogonal channels, diagonal noise. The caller is
        // responsible for providing AET-projected data/template values and
        // a diagonal-only noise buffer; both the CPU and CUDA builds run
        // the same loop.
        for (int channel_i = 0; channel_i < 3; channel_i += 1)
        {
            get_inner_product_value(&d_h_val, &h_h_val, w_mn[channel_i], layer_m, n, channel_i, data_index, noise_index);
            d_h_tmp[tid] += d_h_val;
            h_h_tmp[tid] += h_h_val;
        }
    }
    else if (tdi_type == TDI_AE)
    {
        // AE: two orthogonal channels (T dropped). Same loop body as AET
        // but truncated to channels {0,1}; the caller must pre-project.
        for (int channel_i = 0; channel_i < 2; channel_i += 1)
        {
            get_inner_product_value(&d_h_val, &h_h_val, w_mn[channel_i], layer_m, n, channel_i, data_index, noise_index);
            d_h_tmp[tid] += d_h_val;
            h_h_tmp[tid] += h_h_val;
        }
    }
}

CUDA_DEVICE
void WDMDomain::add_ip_swap_contrib(double *d_h_add_acc, double *d_h_remove_acc, double *add_add_acc, double *remove_remove_acc, double *add_remove_acc, double *w_mn_add, double *w_mn_remove, int layer_m, int n, int data_index, int noise_index, int tdi_type)
{
    // Accumulators are per-thread scalars (register-resident in the caller). We
    // sum into local temporaries here and write them back at the end, so the
    // hot channel loop touches no shared/global memory and the previous
    // 5xNUM_THREADS_HERE shared staging buffer is gone.
    double d_h_add_local = 0.0;
    double d_h_remove_local = 0.0;
    double add_add_local = 0.0;
    double remove_remove_local = 0.0;
    double add_remove_local = 0.0;

    double d_h_val = 0.0;
    double hh_val = 0.0;

    int nchannels = 3;
    if (tdi_type == TDI_AE) nchannels = 2;

    if (tdi_type == TDI_XYZ)
    {
        for (int channel_i = 0; channel_i < 3; channel_i += 1)
        {
            for (int channel_j = 0; channel_j < 3; channel_j += 1)
            {
                get_inner_product_value_cross_channel(&d_h_val, &hh_val, w_mn_add[channel_i], w_mn_add[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);
                d_h_add_local += d_h_val;
                add_add_local += hh_val;

                get_inner_product_value_cross_channel(&d_h_val, &hh_val, w_mn_remove[channel_i], w_mn_remove[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);
                d_h_remove_local += d_h_val;
                remove_remove_local += hh_val;

                // <h_add|h_remove>: only hh_val (= add_i * remove_j * noise_ij) is needed.
                get_inner_product_value_cross_channel(&d_h_val, &hh_val, w_mn_add[channel_i], w_mn_remove[channel_j], layer_m, n, channel_i, channel_j, data_index, noise_index);
                add_remove_local += hh_val;
            }
        }
    }
    else if ((tdi_type == TDI_AET) || (tdi_type == TDI_AE))
    {
        // AET/AE: orthogonal channels, diagonal per-pixel noise. AET keeps
        // all three channels, AE drops T via nchannels=2. Caller must
        // supply data/template/noise in the projected basis. Same loop on
        // CPU and CUDA.
        for (int channel_i = 0; channel_i < nchannels; channel_i += 1)
        {
            get_inner_product_value(&d_h_val, &hh_val, w_mn_add[channel_i], layer_m, n, channel_i, data_index, noise_index);
            d_h_add_local += d_h_val;
            add_add_local += hh_val;

            get_inner_product_value(&d_h_val, &hh_val, w_mn_remove[channel_i], layer_m, n, channel_i, data_index, noise_index);
            d_h_remove_local += d_h_val;
            remove_remove_local += hh_val;

            get_inner_product_value_cross_channel(&d_h_val, &hh_val, w_mn_add[channel_i], w_mn_remove[channel_i], layer_m, n, channel_i, channel_i, data_index, noise_index);
            add_remove_local += hh_val;
        }
    }
    else
    {
#ifdef __CUDACC__
#else
        throw std::invalid_argument("Incorrect TDI type.");
#endif
    }

    *d_h_add_acc += d_h_add_local;
    *d_h_remove_acc += d_h_remove_local;
    *add_add_acc += add_add_local;
    *remove_remove_acc += remove_remove_local;
    *add_remove_acc += add_remove_local;
}


// -----------------------------------------------------------------------------
//  Per-pixel chain-rule contributions to dL/dtheta_k.
//
//  For a Gaussian log-likelihood L = -1/2 < d - h | d - h > in the WDM domain
//  the analytic gradient is the inner product of the residual with the
//  parameter derivative of the template,
//
//      dL/dtheta_k = 4 * sum_{m,n,c}  (w_d - w_h)_{m n c}  *  (dw_h/dtheta_k)_{m n c}  *  N^{-1}
//
//  where N^{-1} is the appropriate per-pixel noise weighting (cross-channel
//  for XYZ, diagonal for AET/AE).  We approximate dw_h/dtheta_k by central
//  finite difference *of the waveform itself*,
//
//      dw_h/dtheta_k(p) = (w_+ - w_-) / (2 eps_k)  +  O(eps^2 d^3 w/dtheta^3),
//
//  and use the *true* un-perturbed wavelet coefficient w_h_CENTER as the
//  residual anchor.  This gives an unbiased chain rule whenever the
//  central FD of w is unbiased (i.e. for polynomial-degree-2 dependence
//  central FD is exact and the kernel matches jax.grad to round-off; for
//  higher-order or sinusoidal dependence the only error is the O(eps^2)
//  truncation in the FD derivative itself).
//
//  The outer factor of 4 is supplied by the calling kernel at block-reduce
//  time, just like d_h_out / h_h_out above.  The caller supplies the
//  per-channel central template w_mn[c] and the per-channel FD derivative
//  dw_mn_dk[c] = (w_+ - w_-)/(2 eps_k).
// -----------------------------------------------------------------------------

CUDA_DEVICE
void WDMDomain::add_grad_contrib(double *grad_acc_k, const double *w_mn, const double *dw_mn_dk,
                                  int layer_m, int n, int data_index, int noise_index, int tdi_type)
{
    double local_acc = 0.0;
    if (tdi_type == TDI_XYZ)
    {
        for (int ci = 0; ci < 3; ci += 1)
        {
            double w_d_i = get_pixel_data_value(layer_m, n, ci, data_index);
            double r_i = w_d_i - w_mn[ci];
            for (int cj = 0; cj < 3; cj += 1)
            {
                double N_ij = get_pixel_noise_value_cross_channel(layer_m, n, ci, cj, noise_index);
                local_acc += r_i * dw_mn_dk[cj] * N_ij * 0.25;
            }
        }
    }
    else if ((tdi_type == TDI_AET) || (tdi_type == TDI_AE))
    {
#ifndef __CUDACC__
        // AET path -- see add_ip_contrib comment.
#endif
        int nchannels = (tdi_type == TDI_AE) ? 2 : 3;
        for (int c = 0; c < nchannels; c += 1)
        {
            double w_d = get_pixel_data_value(layer_m, n, c, data_index);
            double N_c = get_pixel_noise_value(layer_m, n, c, noise_index);
            local_acc += (w_d - w_mn[c]) * dw_mn_dk[c] * N_c * 0.25;
        }
    }
    *grad_acc_k += local_acc;
}


// Swap-likelihood per-pixel chain-rule contribution on one side (add or remove).
//
// For ll_diff = L(after) - L(before) with the post-swap residual
//
//    r_after = w_d - w_h_add + w_h_remove,
//
//  d(ll_diff)/d(theta_add[k])    = +4 sum_{m,n,c} (r_after)_{m n c} (dw_add/dtheta_k)_{m n c} * N^{-1}
//  d(ll_diff)/d(theta_remove[k]) = -4 sum_{m,n,c} (r_after)_{m n c} (dw_rem/dtheta_k)_{m n c} * N^{-1}
//
// The caller passes `sign` (=+1 for add side, =-1 for remove side), the center
// wavelet coefficients of *both* templates at this pixel (zero if the other
// template is out of its layer/orbit support) and the FD derivative of the
// side that is being differentiated.
CUDA_DEVICE
void WDMDomain::add_swap_grad_contrib_one_side(
    double *grad_acc_k, double sign,
    const double *w_mn_add, const double *w_mn_rem, const double *dw_mn_dk,
    int layer_m, int n, int data_index, int noise_index, int tdi_type)
{
    double local_acc = 0.0;
    if (tdi_type == TDI_XYZ)
    {
        for (int ci = 0; ci < 3; ci += 1)
        {
            double w_d_i = get_pixel_data_value(layer_m, n, ci, data_index);
            double r_i = w_d_i - w_mn_add[ci] + w_mn_rem[ci];
            for (int cj = 0; cj < 3; cj += 1)
            {
                double N_ij = get_pixel_noise_value_cross_channel(layer_m, n, ci, cj, noise_index);
                local_acc += sign * r_i * dw_mn_dk[cj] * N_ij * 0.25;
            }
        }
    }
    else if ((tdi_type == TDI_AET) || (tdi_type == TDI_AE))
    {
#ifndef __CUDACC__
        // AET path -- see add_grad_contrib comment.
#endif
        int nchannels = (tdi_type == TDI_AE) ? 2 : 3;
        for (int c = 0; c < nchannels; c += 1)
        {
            double w_d = get_pixel_data_value(layer_m, n, c, data_index);
            double N_c = get_pixel_noise_value(layer_m, n, c, noise_index);
            double r_c = w_d - w_mn_add[c] + w_mn_rem[c];
            local_acc += sign * r_c * dw_mn_dk[c] * N_c * 0.25;
        }
    }
    *grad_acc_k += local_acc;
}
#endif  // === end WDMDomain method bodies (second segment) moved to LAT ===


#define N_PARAMS_MAX 20


#ifdef __CUDACC__
CUDA_DEVICE
double block_reduce(double *array)
{
     // Specialize BlockReduce for a 1D block of 128 threads of type int
    using BlockReduce = cub::BlockReduce<double, NUM_THREADS_HERE>;
    int tid = threadIdx.x;
    // Allocate shared memory for BlockReduce
    CUDA_SHARED typename BlockReduce::TempStorage temp_storage;
    CUDA_SYNC_THREADS;
    double thread_data = array[tid];
    double output = BlockReduce(temp_storage).Sum(thread_data);
    return output;
}

// Scalar-input variant of block_reduce: reduces a per-thread register value
// without going through a NUM_THREADS_HERE shared staging array. Only the cub
// TempStorage stays in __shared__, which is smaller than the staging array.
CUDA_DEVICE
double block_reduce_scalar(double thread_data)
{
    using BlockReduce = cub::BlockReduce<double, NUM_THREADS_HERE>;
    CUDA_SHARED typename BlockReduce::TempStorage temp_storage;
    CUDA_SYNC_THREADS;
    return BlockReduce(temp_storage).Sum(thread_data);
}
#endif

// =============================================================================
// Spline-based WDM kernels
// -----------------------------------------------------------------------------
//
// The direct-path fast_wdm_inner (below) calls get_tdi_Xf_single + numerical
// central differences three times per WDM time pixel. The spline path replaces
// that with: (1) an evenly-spaced coarse grid (~256 pts/yr by default) on
// which the existing LISATDIonTheFly::get_tdi already builds smooth
// (tdi_amp, tdi_phase, phi_ref) via new_extract_amplitude_and_phase +
// new_unwrap_phase; (2) cubic splines fit cooperatively in shared memory via
// fit_cubic_spline_pcr; (3) evaluation of the splines at every WDM time pixel
// to rebuild the same (tdi_channel_val, f, fdot=0) triple fast_wdm_inner
// returns.
//
// To keep shared memory bounded, we slide a WDM_SPLINE_L-point window across
// the source's WDM time range. Windows overlap by 1 in t (last point of window
// k = first point of window k+1), so every WDM pixel is owned by exactly one
// window. Within a window the spline is self-consistent for reconstructing the
// raw M and its phase derivative; across windows we do not need to patch the
// 2pi branch of tdi_phase because the lookup consumes
//   M' = conj(M_raw * exp(-i pi/2)) = i * conj(M_raw),
// which is invariant under Dphi -> Dphi + 2 pi, and the frequency is computed
// from the within-window spline derivative.
//
// Convention (matches new_extract_amplitude_and_phase / fast_wdm_inner):
//   M_raw            = amp * exp(-i (tdi_phase + phi_ref))         (amp signed)
//   tdi_channel_val  = conj(M_raw exp(-i pi/2))
//                    = (-amp sin(theta),  +amp cos(theta))
//                       with theta = tdi_phase + phi_ref
//   f                = -(1/2 pi) d/dt arg(M_raw)
//                    =  (1/2 pi) (d tdi_phase / dt + d phi_ref / dt)
//   fdot             = 0   (matches the direct path)
// =============================================================================

#define WDM_SPLINE_L 32

// Pointers into shared memory carving out one spline slot: 3 amp splines
// + 3 dphi splines + 1 phi_ref spline, all length WDM_SPLINE_L on the same
// coarse t-grid. amp_y[c] / dphi_y[c] / phi_ref_y are the (signed amp /
// tdi_phase / phi_ref) values get_tdi writes; c1/c2/c3 are filled by
// fit_cubic_spline_pcr.
struct WDMSplineSet
{
    double *t_grid;                                       // [L]
    double *amp_y[3];                                     // [L] each
    double *dphi_y[3];                                    // [L] each
    double *phi_ref_y;                                    // [L]
    double *amp_c1[3], *amp_c2[3], *amp_c3[3];            // [L] each
    double *dphi_c1[3], *dphi_c2[3], *dphi_c3[3];         // [L] each
    double *phi_ref_c1, *phi_ref_c2, *phi_ref_c3;         // [L] each
};

// Lay out a spline slot inside three contiguous shared buffers:
//   amp_y_buf[3*L]  (channels 0/1/2 back-to-back -- get_tdi writes here)
//   dphi_y_buf[3*L] (same)
//   phi_ref_y_buf[L]
//   coefs_buf[21*L] (7 splines x 3 coefs)
//   t_grid_buf[L]   (shared coarse time grid)
CUDA_DEVICE inline
void wdm_spline_set_init(WDMSplineSet *S,
                         double *t_grid_buf,
                         double *amp_y_buf, double *dphi_y_buf, double *phi_ref_y_buf,
                         double *coefs_buf)
{
    S->t_grid = t_grid_buf;
    for (int c = 0; c < 3; ++c) S->amp_y[c]  = amp_y_buf  + c * WDM_SPLINE_L;
    for (int c = 0; c < 3; ++c) S->dphi_y[c] = dphi_y_buf + c * WDM_SPLINE_L;
    S->phi_ref_y = phi_ref_y_buf;
    int off = 0;
    for (int c = 0; c < 3; ++c) { S->amp_c1[c]  = coefs_buf + off; off += WDM_SPLINE_L; }
    for (int c = 0; c < 3; ++c) { S->amp_c2[c]  = coefs_buf + off; off += WDM_SPLINE_L; }
    for (int c = 0; c < 3; ++c) { S->amp_c3[c]  = coefs_buf + off; off += WDM_SPLINE_L; }
    for (int c = 0; c < 3; ++c) { S->dphi_c1[c] = coefs_buf + off; off += WDM_SPLINE_L; }
    for (int c = 0; c < 3; ++c) { S->dphi_c2[c] = coefs_buf + off; off += WDM_SPLINE_L; }
    for (int c = 0; c < 3; ++c) { S->dphi_c3[c] = coefs_buf + off; off += WDM_SPLINE_L; }
    S->phi_ref_c1 = coefs_buf + off; off += WDM_SPLINE_L;
    S->phi_ref_c2 = coefs_buf + off; off += WDM_SPLINE_L;
    S->phi_ref_c3 = coefs_buf + off; off += WDM_SPLINE_L;
}

// Fit one cubic spline of length L cooperatively. Wraps the GPU PCR variant
// and the CPU Thomas variant so the same call site compiles under both
// builds. pcr_scratch is unused on the CPU side.
CUDA_DEVICE inline
void fit_one_spline(double *x, double *y,
                    double *c1, double *c2, double *c3,
                    double *B, double *pcr_scratch, int L)
{
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    fit_cubic_spline_pcr(x, y, c1, c2, c3, B, pcr_scratch, L,
                         CUBIC_SPLINE_LINEAR_SPACING);
#else
    (void) pcr_scratch;
    fit_cubic_spline_thomas(x, y, c1, c2, c3, B, L,
                            CUBIC_SPLINE_LINEAR_SPACING);
#endif
}

// Cooperatively populate S with splines for the WDM_SPLINE_L coarse points
// starting at t_window_start with spacing coarse_dt. Returns true on success;
// false if any coarse point falls outside orbits/light-travel-time support
// (in that case S's coefficients are left in an indeterminate state and the
// caller should skip the WDM pixels owned by this window).
//
// All scratch buffers are caller-allocated (typically in shared):
//   tdi_chan_scratch [3*L]   cmplx (consumed by get_tdi as channels storage)
//   pcr_scratch      [8*L]   double (GPU only; ignored on CPU)
//   B_scratch        [L]     double
//   get_tdi_scratch  [tof.get_tdi_buffer_size(L)] bytes (flip/pjump/count/fix_count)
CUDA_DEVICE
bool build_wdm_spline_window(
    GBTDIonTheFly &tof, WDMSplineSet *S,
    double *params, int bin_i,
    double t_window_start, double coarse_dt,
    cmplx *tdi_chan_scratch,
    double *pcr_scratch, double *B_scratch,
    void *get_tdi_scratch, int get_tdi_scratch_len)
{
    const int L = WDM_SPLINE_L;
    const int nchannels = 3;

    for (int i = THREAD_START_X; i < L; i += BLOCK_INCR_X)
    {
        S->t_grid[i] = t_window_start + (double) i * coarse_dt;
    }
    CUDA_SYNC_THREADS;

    // get_tdi writes:
    //   tdi_chan_scratch [3*L]      raw M values per channel (we discard)
    //   S->amp_y[0..2]  ([3*L] contiguous) signed amp per channel
    //   S->dphi_y[0..2] ([3*L] contiguous) tdi_phase per channel
    //   S->phi_ref_y    ([L])               phi_ref
    tof.get_tdi(get_tdi_scratch, get_tdi_scratch_len,
                tdi_chan_scratch, S->amp_y[0], S->dphi_y[0], S->phi_ref_y,
                params, S->t_grid, L, bin_i, nchannels);
    CUDA_SYNC_THREADS;

    // Out-of-orbit-bounds check: get_tdi_Xf_single leaves tdi_chan == 0 when
    // an orbit/light-travel-time window check fails. Even one such point in
    // the window means we cannot trust the fit -- skip it.
    CUDA_SHARED bool any_bad;
    if (THREAD_ZERO) any_bad = false;
    CUDA_SYNC_THREADS;
    for (int i = THREAD_START_X; i < L; i += BLOCK_INCR_X)
    {
        if ((tdi_chan_scratch[i].real() == 0.0) &&
            (tdi_chan_scratch[i].imag() == 0.0))
        {
            any_bad = true;
        }
    }
    CUDA_SYNC_THREADS;
    if (any_bad) return false;

    for (int c = 0; c < nchannels; ++c)
    {
        fit_one_spline(S->t_grid, S->amp_y[c],
                       S->amp_c1[c], S->amp_c2[c], S->amp_c3[c],
                       B_scratch, pcr_scratch, L);
        CUDA_SYNC_THREADS;
        fit_one_spline(S->t_grid, S->dphi_y[c],
                       S->dphi_c1[c], S->dphi_c2[c], S->dphi_c3[c],
                       B_scratch, pcr_scratch, L);
        CUDA_SYNC_THREADS;
    }
    fit_one_spline(S->t_grid, S->phi_ref_y,
                   S->phi_ref_c1, S->phi_ref_c2, S->phi_ref_c3,
                   B_scratch, pcr_scratch, L);
    CUDA_SYNC_THREADS;

    return true;
}

// Evaluate the spline set at one WDM time pixel `tn`. Writes the three
// channel values of tdi_channel_val (in the same conj/exp(-i pi/2) rotated
// convention fast_wdm_inner produces) and the per-channel f / fdot=0.
// Returns false if tn falls outside the window's coarse grid (caller should
// not have asked for that pixel, but the guard prevents OOB segment access).
CUDA_DEVICE inline
bool eval_wdm_spline_pixel(const WDMSplineSet *S, double tn,
                           cmplx *tdi_channel_val, double *f, double *fdot)
{
    const int L = WDM_SPLINE_L;
    double t0_grid = S->t_grid[0];
    double dx = S->t_grid[1] - t0_grid;
    int idx = (int) floor((tn - t0_grid) / dx);
    if (idx < 0) idx = 0;
    if (idx > L - 2) idx = L - 2;
    double t0 = S->t_grid[idx];

    double y0   = S->phi_ref_y[idx];
    double cc1  = S->phi_ref_c1[idx];
    double cc2  = S->phi_ref_c2[idx];
    double cc3  = S->phi_ref_c3[idx];
    CubicSplineSegment seg_phiref(t0, y0, cc1, cc2, cc3, CUBIC_SPLINE_LINEAR_SPACING);
    double phi_ref_val   = seg_phiref.eval(tn);
    double dphi_ref_dt   = seg_phiref.eval_single_derivative(tn);

    for (int c = 0; c < 3; ++c)
    {
        CubicSplineSegment seg_amp(t0,
            S->amp_y[c][idx], S->amp_c1[c][idx], S->amp_c2[c][idx], S->amp_c3[c][idx],
            CUBIC_SPLINE_LINEAR_SPACING);
        CubicSplineSegment seg_dphi(t0,
            S->dphi_y[c][idx], S->dphi_c1[c][idx], S->dphi_c2[c][idx], S->dphi_c3[c][idx],
            CUBIC_SPLINE_LINEAR_SPACING);
        double amp        = seg_amp.eval(tn);
        double dphi_val   = seg_dphi.eval(tn);
        double ddphi_dt   = seg_dphi.eval_single_derivative(tn);

        double theta = dphi_val + phi_ref_val;
        double s = sin(theta);
        double cs = cos(theta);
        tdi_channel_val[c] = cmplx(-amp * s, +amp * cs);
        f[c]    = (ddphi_dt + dphi_ref_dt) / (2.0 * M_PI);
        fdot[c] = 0.0;
    }
    return true;
}

// =============================================================================
// Alternative window builder: spline `Re(M*exp(i*phi_ref))` and
// `Im(M*exp(i*phi_ref))` per channel + phi_ref, instead of the get_tdi
// (amp, tdi_phase) decomposition. Memory layout reuses WDMSplineSet exactly
// (amp_y[c] holds Re(M_demod_c); dphi_y[c] holds Im(M_demod_c)) so the storage
// is unchanged.
//
// Why this variant exists -- the get_tdi-based builder runs
// new_extract_amplitude_and_phase, which makes a discrete local-min decision
// (is_min = (As[i] < As[i-1]) && (As[i] < As[i+1])) and a count*pi cumsum
// from it. A tiny parameter perturbation can flip an is_min test at one
// coarse pt, shifting tdi_phase by pi from there onwards. The cubic spline
// of tdi_phase then has a pi-jump that is THE WRONG SIGN for a smooth
// function of (theta+eps) vs (theta-eps), and the chain-rule central FD
// 1/(2 eps_k) amplifies this into a large gradient error.
//
// The demodulated form sidesteps this entirely: Re and Im of M_demod are
// continuous functions of both t and theta (no discrete decisions). The
// cubic spline of each remains smooth across windows and under theta
// perturbations, so the FD-based gradient is well-conditioned.
//
// Per coarse pt the builder does:
//     M_k     = get_tdi_Xf_single(t_k, params, k, u, v, ..., bin_i)
//     phi_ref = get_phase_ref(t_k, params, bin_i)
//     re[c]   = Re(M_k[c] * exp(i*phi_ref))
//     im[c]   = Im(M_k[c] * exp(i*phi_ref))
// then fits 7 splines (3 re + 3 im + phi_ref).
//
// At a WDM pixel tn the evaluator reconstructs:
//     M_demod[c]    = re_eval[c] + i * im_eval[c]
//     M_raw[c]      = M_demod[c] * exp(-i * phi_ref_eval)
//     tdi_channel_val[c] = i * conj(M_raw[c])    (same convention)
// and computes:
//     f[c] = -(1/(2*pi)) * d arg(M_raw[c])/dt
//          =  (1/(2*pi)) * [ dphi_ref/dt - (re*dim/dt - im*dre/dt)/(re^2+im^2) ]
//     fdot[c] = 0
// =============================================================================
CUDA_DEVICE
bool build_wdm_demod_spline_window(
    GBTDIonTheFly &tof, WDMSplineSet *S,
    double *params, int bin_i,
    double t_window_start, double coarse_dt,
    cmplx *tdi_chan_scratch,
    double *pcr_scratch, double *B_scratch,
    int *link_Space_craft_rec, int *link_Space_craft_em)
{
    const int L = WDM_SPLINE_L;
    const int nchannels = 3;

    for (int i = THREAD_START_X; i < L; i += BLOCK_INCR_X)
        S->t_grid[i] = t_window_start + (double) i * coarse_dt;
    CUDA_SYNC_THREADS;

    Vec k_vec(0.0, 0.0, 0.0), u_vec(0.0, 0.0, 0.0), v_vec(0.0, 0.0, 0.0);
    tof.get_sky_vectors(&k_vec, &u_vec, &v_vec, params);

    // Per-coarse-pt raw M[3] + phi_ref evaluation, demodulated into
    // (re, im) per channel. We share the work across threads with
    // THREAD_START_X / BLOCK_INCR_X strides; each thread is responsible for
    // a subset of the L points.
    CUDA_SHARED bool any_bad;
    if (THREAD_ZERO) any_bad = false;
    CUDA_SYNC_THREADS;

    for (int i = THREAD_START_X; i < L; i += BLOCK_INCR_X)
    {
        double t = S->t_grid[i];
        cmplx M[3];
        tof.get_tdi_Xf_single(M, t, params, k_vec, u_vec, v_vec,
                              link_Space_craft_rec, link_Space_craft_em, bin_i);
        // get_tdi_Xf_single leaves M == 0 when an orbit/LTT window fails;
        // mark window as bad so the caller can skip.
        if ((M[0].real() == 0.0) && (M[0].imag() == 0.0)) any_bad = true;

        double phi_ref = tof.get_phase_ref(t, params, bin_i);
        S->phi_ref_y[i] = phi_ref;

        double c_phr = cos(phi_ref);
        double s_phr = sin(phi_ref);
        for (int c = 0; c < nchannels; ++c)
        {
            // M_demod = M * exp(i*phi_ref) = M * (c_phr + i*s_phr)
            double Mr = M[c].real();
            double Mi = M[c].imag();
            S->amp_y[c][i]  = Mr * c_phr - Mi * s_phr;   // Re(M_demod_c)
            S->dphi_y[c][i] = Mr * s_phr + Mi * c_phr;   // Im(M_demod_c)
        }
        (void) tdi_chan_scratch;  // not needed for this builder
    }
    CUDA_SYNC_THREADS;
    if (any_bad) return false;

    for (int c = 0; c < nchannels; ++c)
    {
        fit_one_spline(S->t_grid, S->amp_y[c],
                       S->amp_c1[c], S->amp_c2[c], S->amp_c3[c],
                       B_scratch, pcr_scratch, L);
        CUDA_SYNC_THREADS;
        fit_one_spline(S->t_grid, S->dphi_y[c],
                       S->dphi_c1[c], S->dphi_c2[c], S->dphi_c3[c],
                       B_scratch, pcr_scratch, L);
        CUDA_SYNC_THREADS;
    }
    fit_one_spline(S->t_grid, S->phi_ref_y,
                   S->phi_ref_c1, S->phi_ref_c2, S->phi_ref_c3,
                   B_scratch, pcr_scratch, L);
    CUDA_SYNC_THREADS;

    return true;
}

// Evaluator paired with build_wdm_demod_spline_window. Reconstructs M_raw
// from the spline-interpolated (Re_demod, Im_demod) and phi_ref, then
// applies the same conj/i rotation fast_wdm_inner does, and computes f
// from d arg(M_raw)/dt analytically (no central differences in time).
CUDA_DEVICE inline
bool eval_wdm_demod_spline_pixel(const WDMSplineSet *S, double tn,
                                  cmplx *tdi_channel_val, double *f, double *fdot)
{
    const int L = WDM_SPLINE_L;
    double t0_grid = S->t_grid[0];
    double dx = S->t_grid[1] - t0_grid;
    int idx = (int) floor((tn - t0_grid) / dx);
    if (idx < 0) idx = 0;
    if (idx > L - 2) idx = L - 2;
    double t0 = S->t_grid[idx];

    CubicSplineSegment seg_phiref(t0,
        S->phi_ref_y[idx], S->phi_ref_c1[idx], S->phi_ref_c2[idx], S->phi_ref_c3[idx],
        CUBIC_SPLINE_LINEAR_SPACING);
    double phi_ref_val = seg_phiref.eval(tn);
    double dphi_ref_dt = seg_phiref.eval_single_derivative(tn);

    double c_phr = cos(phi_ref_val);
    double s_phr = sin(phi_ref_val);

    for (int c = 0; c < 3; ++c)
    {
        CubicSplineSegment seg_re(t0,
            S->amp_y[c][idx], S->amp_c1[c][idx], S->amp_c2[c][idx], S->amp_c3[c][idx],
            CUBIC_SPLINE_LINEAR_SPACING);
        CubicSplineSegment seg_im(t0,
            S->dphi_y[c][idx], S->dphi_c1[c][idx], S->dphi_c2[c][idx], S->dphi_c3[c][idx],
            CUBIC_SPLINE_LINEAR_SPACING);
        double re   = seg_re.eval(tn);
        double im   = seg_im.eval(tn);
        double dre  = seg_re.eval_single_derivative(tn);
        double dim  = seg_im.eval_single_derivative(tn);

        // M_raw = (re + i*im) * exp(-i*phi_ref)
        //       = (re + i*im) * (c_phr - i*s_phr)
        //       = (re*c_phr + im*s_phr) + i*(im*c_phr - re*s_phr)
        double Mr_raw = re * c_phr + im * s_phr;
        double Mi_raw = im * c_phr - re * s_phr;

        // tdi_channel_val = i * conj(M_raw) = (Im(M_raw), Re(M_raw))
        //                = (Mi_raw, Mr_raw)
        tdi_channel_val[c] = cmplx(Mi_raw, Mr_raw);

        // f = -(1/2pi) d arg(M_raw)/dt.
        //   arg(M_raw) = arg(M_demod) - phi_ref  (mod 2pi)
        //   d arg(M_demod)/dt = (re*dim - im*dre) / (re^2 + im^2)
        double mag2 = re * re + im * im;
        double darg_demod_dt = (mag2 > 0.0) ? ((re * dim - im * dre) / mag2) : 0.0;
        double darg_raw_dt = darg_demod_dt - dphi_ref_dt;
        f[c] = -darg_raw_dt / (2.0 * M_PI);
        fdot[c] = 0.0;
    }
    return true;
}

// Number of windows needed to cover the WDM active time range and the chosen
// coarse spacing. Windows are length WDM_SPLINE_L, overlap by 1, so each
// "step" between windows is (WDM_SPLINE_L - 1) * coarse_dt.
CUDA_DEVICE inline
int wdm_spline_num_windows(int n_min, int n_max, double layer_dt, double coarse_dt)
{
    double t_span = (double)(n_max - n_min) * layer_dt;
    double step = (double)(WDM_SPLINE_L - 1) * coarse_dt;
    int K = (int) ceil(t_span / step);
    if (K < 1) K = 1;
    return K;
}

// First (inclusive) and last (inclusive) WDM pixel index that belongs to
// window `k` for the run defined by (n_min, n_max, layer_dt, t_ref,
// t_active_start = t_ref + n_min*layer_dt, coarse_dt, K).
//
// Each pixel is owned by exactly one window: window k owns pixels with
//   tn in [t_window_start_k, t_window_start_{k+1})
// except the last window which extends through n_max.
CUDA_DEVICE inline
void wdm_spline_window_pixel_range(int k, int K,
                                   int n_min, int n_max,
                                   double layer_dt, double t_ref,
                                   double t_active_start, double coarse_dt,
                                   int *n_lo, int *n_hi)
{
    double step = (double)(WDM_SPLINE_L - 1) * coarse_dt;
    double t_window_start = t_active_start + (double) k * step;
    double t_window_end   = (k == K - 1) ?
        ((double) (n_max + 1) * layer_dt + t_ref) :
        (t_active_start + (double)(k + 1) * step);

    // Half-open assignment: window k owns pixels with
    //   t_window_start_k <= n*layer_dt + t_ref < t_window_end_k.
    // For non-integer alignment between coarse_dt and layer_dt, ceil-1
    // (not floor) is the largest n satisfying the strict-less-than upper
    // bound. The +1 offset on the last window's t_window_end ensures n_max
    // gets included (n*layer_dt + t_ref < (n_max+1)*layer_dt + t_ref).
    int lo = (int) ceil((t_window_start - t_ref) / layer_dt);
    int hi = (int) ceil((t_window_end - t_ref) / layer_dt) - 1;
    if (lo < n_min) lo = n_min;
    if (hi > n_max) hi = n_max;
    *n_lo = lo;
    *n_hi = hi;
}

CUDA_DEVICE


// ============================================================================
// fast_wdm_inner -- heterodyne / chunked-FFT path
// ============================================================================
//
// Recommended Tukey alphas (mirrors the Python helper
// :func:`check_shortened_wdm.recommended_tukey_alpha`; see Test G sweep
// for the data justifying these). The taper is alpha/2 of the
// ``N_sparse`` window on each end; tuned to fit inside the n_pad
// overlap region of the chunk stitch (alpha < 2*n_pad/Nt_sub).
//
//   FAST_WDM_TUKEY_ALPHA_TD          = 0.02   // TD-based chunked stitch
//   FAST_WDM_TUKEY_ALPHA_HET_WIDE    = 0.01   // FD-heterodyne, N_sparse >= 512
//   FAST_WDM_TUKEY_ALPHA_HET_NARROW  = 0.05   // FD-heterodyne, N_sparse  < 512
//
// Don't go above ~0.1 -- past that the taper bites into the chunk
// interior and biases stitched pixels.
//
// Pass ``FAST_WDM_TUKEY_ALPHA_AUTO`` (= -1.0) to ``tukey_alpha`` to
// trigger the equivalent auto-pick (see ``fast_wdm_inner_heterodyne``
// dispatcher). Pass ``0.0`` to disable Tukey explicitly (rectangular
// window).
//
// ============================================================================
//
// Prepares the WDM waveform inputs *without* going through the dense
// time-domain rfft of the full observation. Instead, for a single GB source:
//
//   1. Sparse-evaluate ``tdi_amp``, ``tdi_phase``, ``phase_ref`` at
//      ``N_sparse`` (power-of-two) points spanning a sub-window
//      ``[chunk_t_start, chunk_t_start + T_chunk)``.
//   2. Snap the GB carrier to the chunk's rfft grid: ``k_f0 = round(f0
//      / df_chunk)``, ``f0_grid = k_f0 * df_chunk`` with
//      ``df_chunk = 1/T_chunk``.
//   3. Build the slow positive-freq complex signal per channel
//
//          s_c[i] = tdi_amp[c, i]
//                   * exp(+I * (tdi_phase[c, i] + phase_ref[i]
//                               - 2*pi*f0_grid * (i * dt_sparse)))
//
//      then (optionally) multiply by a Tukey window of length ``N_sparse``
//      with the taper confined to ``tukey_alpha/2`` of each end. The taper
//      kills out-of-band spectral leakage so a wider ``N_sparse`` is not
//      required to cover the WDM analysis band -- empirically gives ~1000x
//      mismatch improvement at small ``N_sparse``.
//   4. Run an in-place complex FFT of length ``N_sparse`` on ``s_c``
//      (reuse :func:`wdm_spline_radix2_fft` from WDMSplineHelpers.hh).
//   5. Place ``X_het[c, m] = 0.5 * dt_sparse * fft(s_c)[m]`` into the
//      chunk's dense rfft grid at bins ``k_f0 + fftfreq(N_sparse).astype(int)``
//      (i.e. ``m_in_fft_order`` runs ``[0, 1, ..., N/2-1, -N/2, ..., -1]``).
//
// The downstream chunk-FD --> chunk-WDM step (per-layer window-and-iFFT)
// is left to existing WDM machinery; this device function only fills the
// chunk's rfft buffer.
//
// Caller responsibilities:
//   * ``chunk_fd_out`` -- pre-zeroed buffer of length
//     ``nchannels * n_rfft_chunk``. Only ``N_sparse`` bins around
//     ``k_f0`` are written; the rest stay zero.
//   * All workspace pointers (``t_sparse_buf``, ``tdi_amp_buf``,
//     ``tdi_phase_buf``, ``phi_ref_buf``, ``tdi_channels_buf``,
//     ``slow_buf``) sized as documented; allocated in shared memory if
//     called from a single block.
//   * ``get_tdi_buffer`` of length ``get_tdi_buffer_len`` -- the
//     scratch ``LISATDIonTheFly::get_tdi`` requires (see
//     ``get_tdi_buffer_size``).
//   * ``N_sparse`` must be a power of two; ``log2_N_sparse`` matches.
//
// Multi-chunk dispatch: launch one block per chunk and pass per-chunk
// ``chunk_t_start`` + a per-chunk slice of ``chunk_fd_out``. The kernel
// :func:`gb_heterodyne_chunk_kernel` below does this for a single
// source's chunked WDM build; the host wrapper
// :func:`GBComputationGroup::gb_heterodyne_chunk_prepare_wrap` is
// declared in binding_tof.hpp (TBD; not added by this commit).
//
// CPU vs CUDA: ``CUDA_DEVICE``, ``THREAD_START_X`` and ``BLOCK_INCR_X``
// macros let the same code run as a serial CPU function or a CUDA
// per-block routine. The FFT helper (``wdm_spline_radix2_fft``) is
// already dual-mode.
//
// Per the Python reference (``check_shortened_wdm.py`` Test G/H, dated
// 2026-05), ``N_sparse=1024, tukey_alpha=0.0`` reproduces mm5/mm2 ~
// 5e-13 (matches the dense TD->FD->WDM floor); ``N_sparse=64,
// tukey_alpha=0.05`` reaches mm5/mm2 ~ 1e-7 -- the small Tukey collapses
// the heterodyne-band requirement.
// ----------------------------------------------------------------------------

// FAST_WDM_TUKEY_ALPHA_* + FAST_WDM_N_SPARSE_MAX + FAST_WDM_NCHANNELS_MAX
// + FAST_WDM_NT_SUB_MAX + FAST_WDM_N_CP_SIG_MAX + FAST_WDM_HET_GRID_DIM_X_DEFAULT
// moved to LAT at Phase 3L.7a slice 1 (2026-06-04). They live in
// `lisatools/cutils/lat_chunked_het_kernels.hh`, already included near
// the top of this file.

// FAST_WDM_K_PER_THREAD_MAX moved to LAT at Phase 3L.7a slice 3-prep
// (2026-06-04); see lat_chunked_het_kernels.hh included near the top
// of this file.

// ---------------------------------------------------------------------------
// Threading model notes for upcoming gb_wdm_het_* kernels
// ---------------------------------------------------------------------------
//
// NUM_THREADS vs N_sparse: the radix-2 FFT helper
// (``wdm_spline_radix2_fft`` in WDMSplineHelpers.hh) already strides via
// THREAD_START_X / BLOCK_INCR_X, so a block can be sized with NUM_THREADS
// != N_sparse. The only constraint is N_sparse >= NUM_THREADS (each
// thread handles N_sparse / NUM_THREADS sequential samples). On CPU
// NUM_THREADS = 1, the loops serialise.
//
// d_h / h_h accumulation pattern (for the upcoming get_ll / swap_ll
// chunked variants -- not in this commit):
//   #ifdef __CUDACC__
//     // each thread: partial[NUM_THREADS] accumulator in shared mem
//     // -> CUB block-reduce -> thread 0 atomicAdd to source's d_h[bin_i]
//   #else
//     // single-threaded loop: regular += into d_h[bin_i]
//   #endif
// This collapses atomic contention to one atomicAdd per block per
// source, instead of one per WDM pixel.
// ---------------------------------------------------------------------------

// wdm_fit_cubic_spline + populate_orbit_spline_cache moved to LAT at
// Phase 3L.7a slice 2b (2026-06-04). See
// lisatools/cutils/lat_chunked_het_kernels.hh included near the top.
//
// OrbitsSplineCache struct is declared in lat_tdi_on_the_fly.hh (LAT)
// so that LISATDIonTheFly's cached member functions can take it as a
// parameter.


// === OrbitsSplineCache evaluation helpers moved to LAT at Phase 3L.5 ===
// _orbit_cache_seg, _orbit_cache_link_index, cache_get_light_travel_time,
// cache_get_pos now live in
//   LISAanalysistools/src/lisatools/cutils/lat_tdi_on_the_fly.cu
// (copy-compiled in-place by this repo's CMakeLists). The only callers
// were the LISATDIonTheFly cached method bodies, which also moved to LAT.
// The block below is retained as #if 0 for historical reference.
#if 0
// Inline helper: locate the segment index for an arbitrary t in the cache's
// uniform grid, clamped to [0, N_cp - 2].
CUDA_DEVICE
inline int _orbit_cache_seg(const OrbitsSplineCache *c, double t)
{
    int seg = (int) ((t - c->t_cp0) / c->dt_cp);
    if (seg < 0)             seg = 0;
    if (seg > c->N_cp - 2)   seg = c->N_cp - 2;
    return seg;
}

CUDA_DEVICE
inline int _orbit_cache_link_index(int link)
{
    // Mirrors Orbits::get_link_ind. Returns -1 on bad link (caller bug).
    switch (link) {
        case 12: return 0;
        case 23: return 1;
        case 31: return 2;
        case 13: return 3;
        case 32: return 4;
        case 21: return 5;
        default: return -1;
    }
}

CUDA_DEVICE
inline double cache_get_light_travel_time(const OrbitsSplineCache *c,
                                           double t, int link)
{
    const int link_i = _orbit_cache_link_index(link);
    const int seg    = _orbit_cache_seg(c, t);
    const int p      = link_i * c->N_cp + seg;
    const double dx  = t - c->t_cp[seg];
    return c->ltt_y[p]
         + c->ltt_c1[p] * dx
         + c->ltt_c2[p] * dx * dx
         + c->ltt_c3[p] * dx * dx * dx;
}

CUDA_DEVICE
inline Vec cache_get_pos(const OrbitsSplineCache *c, double t, int sc)
{
    const int seg = _orbit_cache_seg(c, t);
    const double dx = t - c->t_cp[seg];
    const int base = (sc - 1) * 3;   // sc in 1..3
    double v[3];
    for (int xyz = 0; xyz < 3; ++xyz) {
        const int p = (base + xyz) * c->N_cp + seg;
        v[xyz] = c->pos_y[p]
              + c->pos_c1[p] * dx
              + c->pos_c2[p] * dx * dx
              + c->pos_c3[p] * dx * dx * dx;
    }
    return Vec(v[0], v[1], v[2]);
}
#endif  // === end OrbitsSplineCache helpers moved to LAT ===


// fast_wdm_inner_heterodyne_spline + fast_wdm_inner_heterodyne +
// fast_wdm_inner_heterodyne_direct moved to LAT at Phase 3L.7a slice 2b
// (2026-06-04); see lisatools/cutils/lat_chunked_het_kernels.hh
// included near the top of this file. The GB-specific KERNEL launcher
// `fast_wdm_inner_heterodyne_kernel` below stays in this repo (it
// instantiates `GBTDIonTheFly` directly) and moves to GBGPU at Phase
// 3L.7. The templated `fast_wdm_inner_heterodyne_direct<SourceT>` that
// the kernel below calls is resolved through the LAT header include.


// Kernel: dispatches one block per chunk for a single GB source. Now uses
// the direct (pointwise get_tdi_Xf_single) inner path -- shared workspace
// is just slow_buf (~24 KB) plus the small link-spacecraft arrays.
//
//   chunk_fd_all          (n_chunks, nchannels, n_rfft_chunk)  zero-init by host
//   chunk_t_starts        (n_chunks,)
//
// get_tdi_scratch_all / get_tdi_scratch_len_per_block are kept in the
// signature for Python-binding compat but unused by this kernel
// (get_tdi_Xf_single does not allocate scratch).
//
// NUM_THREADS may be < N_sparse: THREAD_START_X / BLOCK_INCR_X stride.
CUDA_KERNEL
inline void fast_wdm_inner_heterodyne_kernel(
    cmplx *chunk_fd_all,            // (n_chunks, nchannels, n_rfft_chunk)
    Orbits *orbits, TDIConfig *tdi_config, double T, double t_ref,
    double *params,                 // (9,) single source
    double *chunk_t_starts,         // (n_chunks,)
    int n_chunks, int bin_i,
    double T_chunk, int N_sparse, int log2_N_sparse,
    int n_rfft_chunk, int nchannels, double tukey_alpha,
    void   *get_tdi_scratch_all,
    int     get_tdi_scratch_len_per_block
)
{
    (void) get_tdi_scratch_all;
    (void) get_tdi_scratch_len_per_block;

    GBTDIonTheFly gb(orbits, tdi_config, T, t_ref);

    // Per-block shared workspace -- just the FFT staging buffer now.
    CUDA_SHARED cmplx slow_buf[FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];

    // Sky vectors + link arrays: per-source, time-independent. Compute once.
    CUDA_SHARED int link_sc_rec[NLINKS];
    CUDA_SHARED int link_sc_em [NLINKS];
    gb.fill_link_arrays(link_sc_rec, link_sc_em);

    Vec k_sky(0.0, 0.0, 0.0), u_sky(0.0, 0.0, 0.0), v_sky(0.0, 0.0, 0.0);
    gb.get_sky_vectors(&k_sky, &u_sky, &v_sky, params);
    CUDA_SYNC_THREADS;

    for (int j = BLOCK_START_X; j < n_chunks; j += GRID_INCR_X) {
        cmplx *chunk_fd = &chunk_fd_all[j * nchannels * n_rfft_chunk];

        fast_wdm_inner_heterodyne_direct(
            chunk_fd, gb, params, bin_i, gb.f0_index,
            chunk_t_starts[j], T_chunk,
            N_sparse, log2_N_sparse, n_rfft_chunk, nchannels, tukey_alpha,
            slow_buf, k_sky, u_sky, v_sky,
            link_sc_rec, link_sc_em);
        CUDA_SYNC_THREADS;
    }
}


// gb_chunk_fd_to_wdm moved to LAT at Phase 3L.7a slice 2b (2026-06-04);
// see lisatools/cutils/lat_chunked_het_kernels.hh included near the
// top of this file. Despite the historical `gb_` prefix the function is
// source-agnostic -- it takes a heterodyned FD chunk and returns the
// per-layer WDM coefficients via iFFT + parity-sign + Re/Im pick.


// WDMHetDirectBufs + WDMHetSplineBufs + WDM_HET_PATH_BYTES /
// WDM_HET_PATH_ALIGN moved to LAT at Phase 3L.7a slice 1 (2026-06-04);
// see lisatools/cutils/lat_chunked_het_kernels.hh included near the
// top of this file. The structs are file-static-only consumers
// (chunked-het kernels) so no aliasing / pybind11 registration is
// needed.


// ============================================================================
// gb_wdm_het_fill_global_kernel  (Phase 2b -- chunked-heterodyne fill_global)
// ============================================================================
//
// Mirrors :func:`gb_wdm_fill_global_kernel` but uses the chunked
// FD-heterodyne path (no per-pixel lookup table). For each binary,
// iterates over time-window chunks; per chunk:
//
//   1. fast_wdm_inner_heterodyne(...)  -> chunk's dense rfft
//                                          (N_sparse bins around k_f0
//                                           populated; rest zero)
//   2. gb_chunk_fd_to_wdm(...)         -> chunk WDM (nchannels, Nf, Nt_sub)
//   3. stitch into template_fill[chan, m, n_global] with the standard
//      interior rule:
//         keep [n_pad, Nt_sub - n_pad) for middle chunks
//         + extend to 0 / Nt_sub for first / last chunk
//
// Partial-slide handling (the Nt is not always a multiple of step =
// Nt_sub - 2*n_pad): the host pre-computes ``chunk_t_starts`` and
// ``chunk_keep_lo`` / ``chunk_keep_hi`` per chunk so the kernel is
// stitch-aware without re-deriving the geometry.
//
// Outer loop: binaries.  Inner loop: chunks.
//   ``factors_all[bin_i]`` is a per-source multiplicative scalar
//   applied at the accumulation step (mirrors fill_global's interface).
//
// (Stub: see Phase 2 plan; actual body wires the helpers and stitches.
// Wave-table window must be precomputed on the host and passed via
// ``wdm_window``.)

// =============================================================================
// OLD chunked-het kernels (Phase 2b/c/d) -- preserved for reference but
// disabled. Replaced by the shared-memory-only kernels further down. See the
// header block before the new kernels for the rewrite rationale.
// =============================================================================
#if 0 // ---- OLD KERNELS BEGIN ----
template <class SourceT>
CUDA_KERNEL
void wdm_het_fill_global_kernel(
    double *template_fill,         // (nchannels, Nf, Nt) global WDM template
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,     // Nf, Nt, ind_min_f, ind_min_t,
                                   // Nf_active, Nt_active, layer_df, ...
    double *params_all,            // (num_bin * nparams,)
    double *factors_all,           // (num_bin,)
    double *chunk_t_starts,        // (n_chunks,)
    int    *chunk_keep_lo,         // (n_chunks,)
    int    *chunk_keep_hi,         // (n_chunks,)
    int    *chunk_n_global_offset, // (n_chunks,) -- global n_pixel for chunk-pixel keep_lo
    double *wdm_window,            // (Nt_sub,) precomputed phitilde samples
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    double tukey_alpha,
    int N_cp_sig,                  // 0 -> direct path; >0 -> source-signal spline cache
    int N_cp_orbit,                // 0 -> raw orbits; >0 -> orbit spline cache
    // workspaces (host-allocated, per-chunk or per-block):
    cmplx  *ws_chunk_fd_all,       // (n_chunks * nchannels * n_rfft_chunk)
    cmplx  *ws_layer_scratch_all,  // (n_chunks * Nt_sub)
    double *ws_chunk_wdm_all,      // (n_chunks * nchannels * Nf * Nt_sub)
    cmplx  *ws_tdi_channels_all,   // (n_chunks * nchannels * N_sparse) -- heap-resident
                                   //   substitute for the per-block tdi_channels_buf;
                                   //   write-once / read-once scratch from get_tdi.
    void   *get_tdi_scratch_all,
    int     get_tdi_scratch_len_per_block
)
{
    SourceT src(orbits, tdi_config, T, t_ref);

    // Shadow the WDM constants for readable loop bodies (no duplicated
    // signature parameters -- the kernel reads everything from
    // ``wdm_settings``).
    const int Nf = wdm_settings->Nf;
    const int Nt = wdm_settings->Nt;

    // Per-block shared workspace for the heterodyne primitives. Sized at
    // FAST_WDM_N_SPARSE_MAX / FAST_WDM_NCHANNELS_MAX so the kernel JITs
    // once and dispatches against any (N_sparse, nchannels) below the
    // maxima. tdi_channels_buf moved to heap (`ws_tdi_channels_all`) to
    // keep static shared <= 48 KB default budget on A100/V100 without
    // needing cudaFuncSetAttribute opt-in.
    //
    // Direct-path and spline-path buffers are OVERLAID via a union: only
    // one branch runs per (chunk, binary) (selected by use_spline_cache),
    // so they are never simultaneously live. This collapses ~16 KB of
    // duplicated shared-mem footprint per kernel. The spline-path amp /
    // phase coefficient stacks are also single-channel (amp/phase are fit
    // + evaluated one channel at a time inside fast_wdm_inner_heterodyne_spline),
    // saving another ~6 KB vs. the old per-channel-stacks layout.
    CUDA_SHARED alignas(WDM_HET_PATH_ALIGN) char path_arena[WDM_HET_PATH_BYTES];
    WDMHetDirectBufs *direct = reinterpret_cast<WDMHetDirectBufs *>(path_arena);
    WDMHetSplineBufs *spline = reinterpret_cast<WDMHetSplineBufs *>(path_arena);
    CUDA_SHARED cmplx  slow_buf        [FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    const bool use_spline_cache = (N_cp_sig > 0 && N_cp_sig <= FAST_WDM_N_CP_SIG_MAX
                                   && N_cp_sig < N_sparse);

    // Orbit spline cache (populated once per chunk; reused across binaries).
    CUDA_SHARED double orbit_t_cp_buf  [FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_y_buf [6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c1_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c2_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c3_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_y_buf [9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c1_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c2_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c3_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_B_buf     [FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pcr_buf   [8 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED OrbitsSplineCache orbit_cache_storage;
    const bool use_orbit_cache = (N_cp_orbit > 0 && N_cp_orbit <= FAST_WDM_N_CP_ORBIT_MAX);

    // Per-block CUDA_SHARED layer iFFT scratch -- avoids the per-element
    // global-mem latency that the previous heap-resident
    // ``ws_layer_scratch_all`` slice incurred. On CPU CUDA_SHARED stubs
    // to nothing so this lands on the stack; sized at
    // FAST_WDM_NT_SUB_MAX (256 GPU / 4096 CPU).
    CUDA_SHARED cmplx layer_scratch[FAST_WDM_NT_SUB_MAX];


    // OUTER: chunks on grid.Z. See the equivalent comment block in
    // ``wdm_het_get_ll_kernel`` for the 3D-grid rationale and the
    // (BLOCK_START_X, j) scratch-slot key. fill_global has no per-binary
    // contention on template_fill (each binary writes a distinct global
    // pixel set), so atomic-free writes remain correct under the new
    // X-axis parallelism.
    for (int j = BLOCK_START_Z; j < n_chunks; j += GRID_INCR_Z) {
        const size_t scratch_slot = (size_t) BLOCK_START_X * (size_t) n_chunks
                                     + (size_t) j;
        cmplx  *chunk_fd       = &ws_chunk_fd_all[scratch_slot * nchannels * n_rfft_chunk];
        // layer_scratch lives in CUDA_SHARED (declared at kernel top).
        double *w_chunk        = &ws_chunk_wdm_all[scratch_slot * nchannels * Nf * Nt_sub];
        cmplx  *tdi_channels_buf = &ws_tdi_channels_all[scratch_slot * nchannels * N_sparse];
        void   *get_tdi_scratch = (char *) get_tdi_scratch_all
            + scratch_slot * (size_t) get_tdi_scratch_len_per_block;

        const int keep_lo        = chunk_keep_lo[j];
        const int keep_hi        = chunk_keep_hi[j];
        const int n_global_lo    = chunk_n_global_offset[j];
        const double chunk_t0    = chunk_t_starts[j];

        // Populate the orbit cache once per chunk (if enabled). All
        // binaries in this chunk reuse the same shared-mem splines.
        OrbitsSplineCache *orbit_cache_ptr = nullptr;
        if (use_orbit_cache) {
            populate_orbit_spline_cache(
                &orbit_cache_storage, orbits,
                chunk_t0, T_chunk, N_cp_orbit,
                orbit_t_cp_buf,
                orbit_ltt_y_buf, orbit_ltt_c1_buf, orbit_ltt_c2_buf, orbit_ltt_c3_buf,
                orbit_pos_y_buf, orbit_pos_c1_buf, orbit_pos_c2_buf, orbit_pos_c3_buf,
                orbit_B_buf, orbit_pcr_buf);
            CUDA_SYNC_THREADS;
            orbit_cache_ptr = &orbit_cache_storage;
        }

        // INNER: binaries -- grid-strided over the X axis. Each X
        // block walks BLOCK_START_X, BLOCK_START_X + GRID_INCR_X, ...,
        // sharing this chunk's orbit cache across the binaries it
        // visits. On CPU GRID_INCR_X = 1, so the loop covers
        // [0, num_bin) sequentially (the original single-block path).
        const double layer_df = 1.0 / (2.0 * (double) Nf * dt);
        for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X) {
            double *params = &params_all[(size_t) bin_i * nparams];
            const double factor = factors_all[bin_i];

            // Per-source narrow band -- matches the GROUP_BAND_LAYERS=5
            // convention used by get_ll / swap_ll: ``[m_floor - 3,
            // m_floor + 3)`` (asymmetric in the mm5 sense; see sprint
            // root CLAUDE.md). Layers outside stay at the caller's
            // pre-zero of template_fill (and at zero in w_chunk).
            // This is the physical narrow-band GB template model --
            // edge effects beyond the band are spectral leakage we
            // already validate via mm5 ~1e-9.
            const int m_floor_src = (int) (params[1] / layer_df);
            int bin_m_lo = m_floor_src - 3;
            int bin_m_hi = m_floor_src + 3;
            if (bin_m_lo < 0)   bin_m_lo = 0;
            if (bin_m_hi > Nf)  bin_m_hi = Nf;

            // Zero per-chunk workspaces -- only the live region. See the
            // equivalent optimization in wdm_het_get_ll_kernel for the
            // full rationale (the chunk_fd zero band must cover the
            // gb_chunk_fd_to_wdm READ range, not just the heterodyne
            // write band).
            //
            //   chunk_fd: read by gb_chunk_fd_to_wdm at
            //             k in [m*half_Nt_sub - half_Nt_sub,
            //                  m*half_Nt_sub + half_Nt_sub) for each
            //             m in [bin_m_lo, bin_m_hi). Combined range
            //             [(bin_m_lo - 1)*half_Nt_sub,
            //              (bin_m_hi + 1)*half_Nt_sub). Boundary (m_lo<=0
            //             or m_hi>=Nf) falls back to full zero since the
            //             Hermitian fold pulls reads from the opposite
            //             end of the buffer.
            //   w_chunk:  read by the template_fill stitch (line ~2720)
            //             only at m in [bin_m_lo, bin_m_hi). Zero just
            //             that band.
            const int half_Nt_sub_zero = Nt_sub / 2;
            int kfd_lo, kfd_hi;
            if (bin_m_lo <= 0 || bin_m_hi >= Nf) {
                kfd_lo = 0;
                kfd_hi = n_rfft_chunk;
            } else {
                kfd_lo = (bin_m_lo - 1) * half_Nt_sub_zero;
                kfd_hi = (bin_m_hi + 1) * half_Nt_sub_zero;
                if (kfd_lo < 0)             kfd_lo = 0;
                if (kfd_hi > n_rfft_chunk)  kfd_hi = n_rfft_chunk;
            }
            const int kfd_band = (kfd_hi > kfd_lo) ? (kfd_hi - kfd_lo) : 0;
            for (int idx = THREAD_START_X; idx < nchannels * kfd_band;
                 idx += BLOCK_INCR_X) {
                const int c = idx / kfd_band;
                const int k = kfd_lo + (idx - c * kfd_band);
                chunk_fd[c * n_rfft_chunk + k] = cmplx(0.0, 0.0);
            }

            const int wm_band = (bin_m_hi > bin_m_lo)
                                ? (bin_m_hi - bin_m_lo) : 0;
            for (int idx = THREAD_START_X;
                 idx < nchannels * wm_band * Nt_sub;
                 idx += BLOCK_INCR_X) {
                const int c    = idx / (wm_band * Nt_sub);
                const int rem  = idx - c * wm_band * Nt_sub;
                const int mloc = rem / Nt_sub;
                const int n    = rem - mloc * Nt_sub;
                const int m    = bin_m_lo + mloc;
                w_chunk[((size_t) c * Nf + m) * Nt_sub + n] = 0.0;
            }
            CUDA_SYNC_THREADS;

            // 1) heterodyne FD for this (chunk, binary)
            if (use_spline_cache) {
                fast_wdm_inner_heterodyne_spline(
                    chunk_fd, &src, params, bin_i, src.f0_index,
                    chunk_t0, T_chunk,
                    N_sparse, log2_N_sparse, N_cp_sig,
                    n_rfft_chunk, nchannels, tukey_alpha,
                    spline->t_cp_buf,
                    spline->amp_y_buf, spline->amp_c1_buf,
                    spline->amp_c2_buf, spline->amp_c3_buf,
                    spline->phase_y_buf, spline->phase_c1_buf,
                    spline->phase_c2_buf, spline->phase_c3_buf,
                    spline->dphi_ref_y_buf, spline->dphi_ref_c1_buf,
                    spline->dphi_ref_c2_buf, spline->dphi_ref_c3_buf,
                    spline->B_buf, spline->pcr_scratch,
                    spline->phi_ref_un_het_buf,
                    spline->tdi_channels_cp_buf, slow_buf,
                    spline->extract_scratch_buf,
                    (int) sizeof(spline->extract_scratch_buf),
                    orbit_cache_ptr
                );
            } else {
                fast_wdm_inner_heterodyne(
                    chunk_fd, &src, params, bin_i, src.f0_index,
                    chunk_t0, T_chunk,
                    N_sparse, log2_N_sparse, n_rfft_chunk, nchannels, tukey_alpha,
                    direct->t_sparse_buf, direct->tdi_amp_buf,
                    direct->tdi_phase_buf, direct->phi_ref_buf,
                    tdi_channels_buf, slow_buf,
                    get_tdi_scratch, get_tdi_scratch_len_per_block,
                    orbit_cache_ptr
                );
            }
            CUDA_SYNC_THREADS;

            // 2) chunk FD -> chunk WDM. Restrict to the per-source
            // narrow band [bin_m_lo, bin_m_hi) computed above.
            gb_chunk_fd_to_wdm(
                w_chunk, chunk_fd, wdm_window,
                Nf, Nt_sub, log2_Nt_sub, n_rfft_chunk, dt, nchannels,
                layer_scratch,
                /*m_lo=*/bin_m_lo, /*m_hi=*/bin_m_hi
            );
            CUDA_SYNC_THREADS;

            // 3) stitch into template_fill. Only m in [bin_m_lo, bin_m_hi)
            //    has nonzero w_chunk for this source; layers outside
            //    that band are at zero from the pre-zero of w_chunk.
            //    template_fill[c, m, n_global_lo + (n - keep_lo)] +=
            //        factor * w_chunk[c, m, n]   for n in [keep_lo, keep_hi)
            for (int c = 0; c < nchannels; ++c) {
                for (int m = bin_m_lo; m < bin_m_hi; ++m) {
                    for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                         n_loc += BLOCK_INCR_X) {
                        const int n_glob = n_global_lo + (n_loc - keep_lo);
                        const size_t dst = ((size_t) c * Nf + m) * Nt + n_glob;
                        template_fill[dst] +=
                            factor * w_chunk[(size_t) c * Nf * Nt_sub + (size_t) m * Nt_sub + n_loc];
                    }
                }
            }
            CUDA_SYNC_THREADS;
        }
    }
}


// ============================================================================
// gb_wdm_het_get_ll_kernel  (Phase 2c -- chunked-heterodyne get_ll)
// ============================================================================
//
// Computes per-source ``<d|h>`` and ``<h|h>`` over the WDM domain using
// the chunked FD-heterodyne template build. Designed for high CUDA
// efficiency under the constraints discussed:
//
//   * Outer loop = chunks. PSD and data are read once per chunk into
//     shared memory; ~``nchannels * Nf * Nt_sub`` doubles per chunk =
//     ~24 KB at (3, 64, 128) -- fits.
//   * Inner loop = binaries. Each binary reuses the same shared PSD /
//     data, paying only one global read for its own params.
//   * Per (chunk, binary) pixel-loop: each thread accumulates a partial
//     sum of d_h_partial and h_h_partial into shared per-thread arrays.
//   * After the per-binary loop:
//       #ifdef __CUDACC__
//         CUB block-reduce d_h_partial / h_h_partial -> single value
//         thread 0 atomicAdd into d_h_out[bin_i], h_h_out[bin_i]
//       #else
//         simple serial reduction (NUM_THREADS = 1 on CPU)
//       #endif
//
// Partial-slide handling: the host pre-builds chunk geometry arrays
// (chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset)
// such that only the *new* pixels of a partial-slide chunk are summed
// (the overlap with the previous full chunk is excluded). This makes
// the kernel-side stitching logic identical for full and partial
// chunks.
//
// NUM_THREADS vs N_sparse: the FFT helper accepts NUM_THREADS <
// N_sparse (sequential stride). Host can pick NUM_THREADS independently
// of N_sparse provided N_sparse >= NUM_THREADS.
//
// (Stub; body filled in once gb_wdm_het_fill_global_kernel is
// validated. Algorithm and data flow documented above.)
template <class SourceT>
CUDA_KERNEL
void wdm_het_get_ll_kernel(
    double *d_h_out, double *h_h_out,        // (num_bin,) outputs (host pre-zero'd)
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,               // Nf, Nt, ind_min_f, ind_min_t,
                                             // Nf_active, Nt_active, ...
    double *params_all,                      // (num_bin * nparams,)
    int    *data_index_all, int *noise_index_all,
    double *chunk_t_starts,                  // (n_chunks,)
    int    *chunk_keep_lo, int *chunk_keep_hi,
    int    *chunk_n_global_offset,
    double *wdm_window,                      // (Nt_sub,)
    double *data_d, double *invC,            // ACTIVE-band layout (matches the
                                             // natural lisatools output of
                                             // AnalysisContainerArray):
                                             //   data_d : (nchannels, Nf_active, Nt_active)
                                             //   invC (TDI_XYZ):
                                             //     (nchannels, nchannels, Nf_active, Nt_active)
                                             //   invC (TDI_AET/AE):
                                             //     (nchannels, Nf_active, Nt_active) diagonal
                                             // Pixels outside the active band
                                             // (m, n) are skipped by the
                                             // accumulator (zero contribution).
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,                         // TDI_XYZ / TDI_AET / TDI_AE
    double tukey_alpha,
    int N_cp_sig,                            // 0 -> direct path; >0 -> spline cache
    int N_cp_orbit,                          // 0 -> raw orbits; >0 -> orbit spline cache
    // Per-block heap workspace:
    cmplx  *ws_chunk_fd_all,                 // (n_chunks * nchannels * n_rfft_chunk)
    cmplx  *ws_layer_scratch_all,            // (n_chunks * Nt_sub)
    double *ws_chunk_wdm_all,                // (n_chunks * nchannels * Nf * Nt_sub)
    cmplx  *ws_tdi_channels_all,             // (n_chunks * nchannels * N_sparse) -- heap
                                             //   tdi_channels_buf, see fill_global.
    void   *get_tdi_scratch_all,
    int     get_tdi_scratch_len_per_block,
    // Layer-grouping (chunk's outer is unchanged; if n_groups > 0 we
    // insert an intermediate group loop, iterate only binaries in the
    // group, and restrict the m-band accumulator to [group_m_lo,
    // group_m_hi). With n_groups == 0 the existing per-binary all-m
    // path is used.
    int    *binary_perm,                     // (num_bin,) -- kernel reads binary_perm[bin_iter]
    int    *group_starts,                    // (n_groups,)
    int    *group_ends,                      // (n_groups,) exclusive
    int    *group_m_lo,                      // (n_groups,) inclusive
    int    *group_m_hi,                      // (n_groups,) exclusive
    int     n_groups
)
{
    SourceT src(orbits, tdi_config, T, t_ref);

    // Shadow the WDM constants for readable loop bodies (no duplicated
    // signature parameters -- the kernel reads everything from
    // ``wdm_settings``).
    const int Nf         = wdm_settings->Nf;
    const int Nt         = wdm_settings->Nt;
    const int ind_min_f  = wdm_settings->ind_min_f;
    const int ind_min_t  = wdm_settings->ind_min_t;
    const int Nf_active  = wdm_settings->Nf_active;
    const int Nt_active  = wdm_settings->Nt_active;

    // Heterodyne shared workspace (per block / per chunk). The
    // tdi_channels_buf slab has been moved to heap to keep static shared
    // under the 48 KB default budget; see fill_global comments.
    //
    // Direct-path and spline-path buffers are OVERLAID via a union: only
    // one branch runs per (chunk, binary) (selected by use_spline_cache),
    // so they are never simultaneously live. This collapses ~16 KB of
    // duplicated shared-mem footprint. The spline-path amp / phase
    // coefficient stacks are also single-channel (amp/phase fit + eval
    // happens one channel at a time inside fast_wdm_inner_heterodyne_spline),
    // saving another ~6 KB vs. the old per-channel-stacks layout.
    CUDA_SHARED alignas(WDM_HET_PATH_ALIGN) char path_arena[WDM_HET_PATH_BYTES];
    WDMHetDirectBufs *direct = reinterpret_cast<WDMHetDirectBufs *>(path_arena);
    WDMHetSplineBufs *spline = reinterpret_cast<WDMHetSplineBufs *>(path_arena);
    const bool use_spline_cache = (N_cp_sig > 0 && N_cp_sig <= FAST_WDM_N_CP_SIG_MAX
                                   && N_cp_sig < N_sparse);
    CUDA_SHARED cmplx  slow_buf        [FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];

    // Orbit spline cache (populated once per chunk; reused across binaries).
    CUDA_SHARED double orbit_t_cp_buf  [FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_y_buf [6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c1_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c2_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c3_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_y_buf [9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c1_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c2_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c3_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_B_buf     [FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pcr_buf   [8 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED OrbitsSplineCache orbit_cache_storage;
    const bool use_orbit_cache = (N_cp_orbit > 0 && N_cp_orbit <= FAST_WDM_N_CP_ORBIT_MAX);

    // Per-block CUDA_SHARED layer iFFT scratch -- avoids the per-element
    // global-mem latency that the previous heap-resident
    // ``ws_layer_scratch_all`` slice incurred. On CPU CUDA_SHARED stubs
    // to nothing so this lands on the stack; sized at
    // FAST_WDM_NT_SUB_MAX (256 GPU / 4096 CPU).
    CUDA_SHARED cmplx layer_scratch[FAST_WDM_NT_SUB_MAX];

    // Per-thread partial accumulators (one slot per thread; reduced at end).
    // Sized at a generous upper bound; THREAD_START_X / BLOCK_INCR_X controls
    // the active extent.
    CUDA_SHARED double partial_dh[FAST_WDM_N_SPARSE_MAX];   // reuse N_sparse cap
    CUDA_SHARED double partial_hh[FAST_WDM_N_SPARSE_MAX];

    // Per-thread data/invC cache REMOVED -- the dynamic slot indexing
    // prevented NVCC from keeping the arrays in registers (they spilled
    // to CUDA local memory, DRAM-backed with L1 caching), making the
    // cache equivalent in latency to direct global reads. On CPU the
    // cache had ``WDM_KEEP_PER_THREAD_MAX = 1`` and the code already
    // took the direct-read branch. The narrow-band per-block working
    // set fits comfortably in L1/L2, so direct reads here are fast
    // automatically via the hardware cache. See profile + design
    // discussion in the sprint root.

    // OUTER: chunks on grid.Z (default ``gridDim.z == n_chunks``).
    //
    // 3D-grid layout (see FAST_WDM_HET_GRID_DIM_X_DEFAULT and the impl
    // wrapper below): blockIdx.z indexes chunks (grid-stride to
    // tolerate gridDim.z < n_chunks); blockIdx.x indexes binaries with
    // a grid-stride that lets a single X block process multiple
    // binaries from the same group, reusing the chunk-local orbit
    // cache and m-band metadata across them. The per-(x, z) heap
    // scratch slot is ``BLOCK_START_X * n_chunks + j`` -- unique per
    // (x-block, chunk) so independent (x, z) blocks never collide.
    for (int j = BLOCK_START_Z; j < n_chunks; j += GRID_INCR_Z) {
        const size_t scratch_slot = (size_t) BLOCK_START_X * (size_t) n_chunks
                                     + (size_t) j;
        cmplx  *chunk_fd      = &ws_chunk_fd_all[scratch_slot * nchannels * n_rfft_chunk];
        // layer_scratch lives in CUDA_SHARED (declared at kernel top).
        double *w_chunk       = &ws_chunk_wdm_all[scratch_slot * nchannels * Nf * Nt_sub];
        cmplx  *tdi_channels_buf = &ws_tdi_channels_all[scratch_slot * nchannels * N_sparse];
        void   *get_tdi_scratch = (char *) get_tdi_scratch_all
            + scratch_slot * (size_t) get_tdi_scratch_len_per_block;

        const int keep_lo     = chunk_keep_lo[j];
        const int keep_hi     = chunk_keep_hi[j];
        const int n_global_lo = chunk_n_global_offset[j];
        const double chunk_t0 = chunk_t_starts[j];
        const int n_pixels    = keep_hi - keep_lo;

        // Populate orbit cache once per chunk if enabled.
        OrbitsSplineCache *orbit_cache_ptr = nullptr;
        if (use_orbit_cache) {
            populate_orbit_spline_cache(
                &orbit_cache_storage, orbits,
                chunk_t0, T_chunk, N_cp_orbit,
                orbit_t_cp_buf,
                orbit_ltt_y_buf, orbit_ltt_c1_buf, orbit_ltt_c2_buf, orbit_ltt_c3_buf,
                orbit_pos_y_buf, orbit_pos_c1_buf, orbit_pos_c2_buf, orbit_pos_c3_buf,
                orbit_B_buf, orbit_pcr_buf);
            CUDA_SYNC_THREADS;
            orbit_cache_ptr = &orbit_cache_storage;
        }

        // Choose iteration structure:
        //   n_groups == 0 -> ungrouped: iterate binaries directly, m in [0, Nf)
        //   n_groups  > 0 -> grouped: middle loop over groups, inner over
        //                    binaries in group, m restricted to band.
        const int n_group_iter = (n_groups > 0) ? n_groups : 1;
        for (int g = 0; g < n_group_iter; ++g) {
            int bin_iter_lo, bin_iter_hi, m_lo, m_hi;
            if (n_groups > 0) {
                bin_iter_lo = group_starts[g];
                bin_iter_hi = group_ends  [g];
                m_lo = group_m_lo[g];   if (m_lo < 0)  m_lo = 0;
                m_hi = group_m_hi[g];   if (m_hi > Nf) m_hi = Nf;
            } else {
                bin_iter_lo = 0;
                bin_iter_hi = num_bin;
                m_lo = 0;
                m_hi = Nf;
            }

            // (Per-thread populate-pass removed; direct global reads in
            // the accumulator below pick up L1/L2 caching automatically.)

            // INNER: binaries in this group, grid-strided across the X
            // dimension. Each X block walks ``bin_iter_lo +
            // BLOCK_START_X, bin_iter_lo + BLOCK_START_X + GRID_INCR_X,
            // ...`` -- so adjacent X blocks process adjacent binaries
            // within the same group, sharing the chunk-local orbit
            // cache and m-band metadata. With GRID_INCR_X = 1 on CPU
            // the loop walks the whole [bin_iter_lo, bin_iter_hi)
            // range sequentially (single-block behaviour).
            for (int bin_iter = bin_iter_lo + BLOCK_START_X;
                 bin_iter < bin_iter_hi;
                 bin_iter += GRID_INCR_X) {
                const int bin_i = (n_groups > 0) ? binary_perm[bin_iter] : bin_iter;
                double *params = &params_all[(size_t) bin_i * nparams];

                // Zero per-chunk workspaces -- BUT only the live region the
                // downstream code actually reads/writes. The full-buffer zero
                // (524k cmplx + 1M doubles per channel) was burning ~50 MB
                // of HBM write traffic per binary, the dominant per-binary
                // cost on A100 at moderate gridDim.x.
                //
                //   chunk_fd: must be zero wherever gb_chunk_fd_to_wdm
                //             READS it. For m in [m_lo, m_hi) that read
                //             range (pre-Hermitian-fold) is
                //             [m_lo * half_Nt_sub - half_Nt_sub,
                //              m_hi * half_Nt_sub + half_Nt_sub).
                //             ``fast_wdm_inner_heterodyne`` writes inside
                //             that range at k = k_f0 +/- N_sparse/2; cells
                //             it leaves untouched must be 0 (otherwise
                //             they carry the previous binary's heterodyne
                //             values). When m_lo == 0 or m_hi >= Nf the
                //             Hermitian fold pulls reads from the opposite
                //             end -- fall back to a full zero in that rare
                //             boundary case.
                //   w_chunk:  gb_chunk_fd_to_wdm writes inside [m_lo, m_hi)
                //             and the accumulator only reads inside the
                //             active band intersection
                //             [max(m_lo, ind_min_f), min(m_hi, ind_max_f+1)).
                //             Zeroing the read intersection is sufficient.
                //
                // For typical narrow-band setups this is ~1000x less
                // zero traffic than the full-buffer zero.
                const int half_Nt_sub_zero = Nt_sub / 2;
                int kfd_lo, kfd_hi;
                if (m_lo <= 0 || m_hi >= Nf) {
                    // Boundary case: the Hermitian fold makes the read
                    // pattern span the full chunk_fd. Fall back to full zero.
                    kfd_lo = 0;
                    kfd_hi = n_rfft_chunk;
                } else {
                    kfd_lo = (m_lo - 1) * half_Nt_sub_zero;
                    kfd_hi = (m_hi + 1) * half_Nt_sub_zero;
                    if (kfd_lo < 0)             kfd_lo = 0;
                    if (kfd_hi > n_rfft_chunk)  kfd_hi = n_rfft_chunk;
                }
                const int kfd_band = (kfd_hi > kfd_lo) ? (kfd_hi - kfd_lo) : 0;
                for (int idx = THREAD_START_X; idx < nchannels * kfd_band;
                     idx += BLOCK_INCR_X) {
                    const int c = idx / kfd_band;
                    const int k = kfd_lo + (idx - c * kfd_band);
                    chunk_fd[c * n_rfft_chunk + k] = cmplx(0.0, 0.0);
                }

                int wm_lo = (m_lo < ind_min_f) ? ind_min_f : m_lo;
                int wm_hi = (m_hi > ind_min_f + Nf_active)
                              ? (ind_min_f + Nf_active) : m_hi;
                const int wm_band = (wm_hi > wm_lo) ? (wm_hi - wm_lo) : 0;
                for (int idx = THREAD_START_X;
                     idx < nchannels * wm_band * Nt_sub;
                     idx += BLOCK_INCR_X) {
                    const int c    = idx / (wm_band * Nt_sub);
                    const int rem  = idx - c * wm_band * Nt_sub;
                    const int mloc = rem / Nt_sub;
                    const int n    = rem - mloc * Nt_sub;
                    const int m    = wm_lo + mloc;
                    w_chunk[((size_t) c * Nf + m) * Nt_sub + n] = 0.0;
                }
                partial_dh[THREAD_START_X] = 0.0;
                partial_hh[THREAD_START_X] = 0.0;
                CUDA_SYNC_THREADS;

                // 1) heterodyne FD
                if (use_spline_cache) {
                    fast_wdm_inner_heterodyne_spline(
                        chunk_fd, &src, params, bin_i, src.f0_index,
                        chunk_t0, T_chunk,
                        N_sparse, log2_N_sparse, N_cp_sig,
                        n_rfft_chunk, nchannels, tukey_alpha,
                        spline->t_cp_buf,
                        spline->amp_y_buf, spline->amp_c1_buf,
                        spline->amp_c2_buf, spline->amp_c3_buf,
                        spline->phase_y_buf, spline->phase_c1_buf,
                        spline->phase_c2_buf, spline->phase_c3_buf,
                        spline->dphi_ref_y_buf, spline->dphi_ref_c1_buf,
                        spline->dphi_ref_c2_buf, spline->dphi_ref_c3_buf,
                        spline->B_buf, spline->pcr_scratch,
                        spline->phi_ref_un_het_buf,
                        spline->tdi_channels_cp_buf, slow_buf,
                        spline->extract_scratch_buf,
                        (int) sizeof(spline->extract_scratch_buf),
                        orbit_cache_ptr
                    );
                } else {
                    fast_wdm_inner_heterodyne(
                        chunk_fd, &src, params, bin_i, src.f0_index,
                        chunk_t0, T_chunk,
                        N_sparse, log2_N_sparse, n_rfft_chunk, nchannels, tukey_alpha,
                        direct->t_sparse_buf, direct->tdi_amp_buf,
                        direct->tdi_phase_buf, direct->phi_ref_buf,
                        tdi_channels_buf, slow_buf,
                        get_tdi_scratch, get_tdi_scratch_len_per_block,
                        orbit_cache_ptr
                    );
                }
                CUDA_SYNC_THREADS;

                // 2) chunk FD -> chunk WDM. Restrict the WDM transform's
                // outer m-loop to the group's [m_lo, m_hi) band -- the
                // narrow-band GB inner product only reads those layers
                // (see accumulator below), so iterating elsewhere is
                // pure overhead. Layers outside stay at the pre-zero.
                gb_chunk_fd_to_wdm(
                    w_chunk, chunk_fd, wdm_window,
                    Nf, Nt_sub, log2_Nt_sub, n_rfft_chunk, dt, nchannels,
                    layer_scratch,
                    /*m_lo=*/m_lo, /*m_hi=*/m_hi
                );
                CUDA_SYNC_THREADS;

                // 3) per-pixel accumulation, m restricted to the group band.
                //    Dispatch on tdi_type to mirror the lisatools layout:
                //      TDI_XYZ   -> invC is (nchannels, nchannels,
                //                            Nf_active, Nt_active)
                //                    (full Hermitian inverse), inner product
                //                    is sum_{c1,c2} d[c1]*h[c2]*invC[c1,c2].
                //      TDI_AET/AE -> invC is (nchannels, Nf_active, Nt_active)
                //                    diagonal, inner product is
                //                    sum_c d[c]*h[c]*invC[c].
                //    data_d follows the same active-band layout in both
                //    cases ((nchannels, Nf_active, Nt_active)). Pixels with
                //    m or n_glob outside the active band contribute zero
                //    and are skipped at the loop level.
                //
                //    For real-only WDM, lisatools' 4*dc prefactor = 1, so
                //    these sums are bit-equal to lisatools.diagnostic.inner_product.
                const int ind_max_f_excl = ind_min_f + Nf_active;
                const int ind_max_t_excl = ind_min_t + Nt_active;
                const int m_lo_act = (m_lo < ind_min_f) ? ind_min_f : m_lo;
                const int m_hi_act = (m_hi > ind_max_f_excl) ? ind_max_f_excl : m_hi;
                if (tdi_type == TDI_XYZ) {
                    for (int m = m_lo_act; m < m_hi_act; ++m) {
                        const int m_act = m - ind_min_f;
                        for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                             n_loc += BLOCK_INCR_X) {
                            const int n_glob = n_global_lo + (n_loc - keep_lo);
                            if (n_glob < ind_min_t || n_glob >= ind_max_t_excl) continue;
                            const int n_act = n_glob - ind_min_t;
                            double w_arr[FAST_WDM_NCHANNELS_MAX];
                            double d_arr[FAST_WDM_NCHANNELS_MAX];
                            for (int c = 0; c < nchannels; ++c) {
                                const size_t g_w  = ((size_t) c * Nf + m) * Nt_sub + n_loc;
                                const size_t g_dt = ((size_t) c * Nf_active + m_act)
                                                     * Nt_active + n_act;
                                w_arr[c] = w_chunk[g_w];
                                d_arr[c] = data_d[g_dt];
                            }
                            for (int c1 = 0; c1 < nchannels; ++c1) {
                                for (int c2 = 0; c2 < nchannels; ++c2) {
                                    const size_t g_inv = (((size_t) c1 * nchannels + c2)
                                                           * Nf_active + m_act)
                                                          * Nt_active + n_act;
                                    const double inv = invC[g_inv];
                                    partial_dh[THREAD_START_X] += d_arr[c1] * w_arr[c2] * inv;
                                    partial_hh[THREAD_START_X] += w_arr[c1] * w_arr[c2] * inv;
                                }
                            }
                        }
                    }
                } else {
                    for (int c = 0; c < nchannels; ++c) {
                        for (int m = m_lo_act; m < m_hi_act; ++m) {
                            const int m_act = m - ind_min_f;
                            for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                                 n_loc += BLOCK_INCR_X) {
                                const int n_glob = n_global_lo + (n_loc - keep_lo);
                                if (n_glob < ind_min_t || n_glob >= ind_max_t_excl) continue;
                                const int n_act = n_glob - ind_min_t;
                                const size_t g_w  = ((size_t) c * Nf + m) * Nt_sub + n_loc;
                                const size_t g_dt = ((size_t) c * Nf_active + m_act)
                                                     * Nt_active + n_act;
                                const double w   = w_chunk[g_w];
                                const double d   = data_d[g_dt];
                                const double inv = invC  [g_dt];
                                partial_dh[THREAD_START_X] += d * w * inv;
                                partial_hh[THREAD_START_X] += w * w * inv;
                            }
                        }
                    }
                }
                CUDA_SYNC_THREADS;

                // 4) Block-reduce partials and add to global outputs.
                //
                // On CUDA: ideally CUB BlockReduce<double, BLOCK_DIM>::Sum.
                // Since BLOCK_DIM is a compile-time arg to cub::BlockReduce
                // and we want to keep this kernel parametric in NUM_THREADS,
                // we use a manual block reduction that works for any
                // power-of-2 block size up to FAST_WDM_N_SPARSE_MAX. The
                // host launches with a power-of-2 NUM_THREADS to satisfy
                // this. On CPU NUM_THREADS = 1 -> the inner reduction loop
                // is a no-op.
#ifdef __CUDACC__
                // tree-reduction over partial_dh / partial_hh in shared mem
                for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                    if (THREAD_START_X < stride) {
                        partial_dh[THREAD_START_X] += partial_dh[THREAD_START_X + stride];
                        partial_hh[THREAD_START_X] += partial_hh[THREAD_START_X + stride];
                    }
                    CUDA_SYNC_THREADS;
                }
                if (THREAD_START_X == 0) {
                    atomicAdd(&d_h_out[bin_i], partial_dh[0]);
                    atomicAdd(&h_h_out[bin_i], partial_hh[0]);
                }
#else
                // CPU: NUM_THREADS = 1 by THREAD_START_X convention; partial_dh[0]
                // already holds the full sum from the single thread.
                d_h_out[bin_i] += partial_dh[0];
                h_h_out[bin_i] += partial_hh[0];
#endif
                CUDA_SYNC_THREADS;
            } // bin_iter
        } // g (group)
    } // chunk
}


// ============================================================================
// gb_wdm_het_swap_ll_kernel  (Phase 2d -- chunked-heterodyne swap_ll)
// ============================================================================
//
// Same chunked-outer-loop / shared-memory recipe as get_ll, but with
// add/remove pairs:
//
//   d_h_add  / d_h_remove
//   add_add  / remove_remove / add_remove
//
// Per chunk, per pair (add_bin, remove_bin):
//   build w_add (heterodyne path on params_add)
//   build w_rem (heterodyne path on params_rem)
//   accumulate the five contributions using the shared PSD/data slabs:
//      d_h_add_acc        += sum d * w_add * inv_C
//      d_h_remove_acc     += sum d * w_rem * inv_C
//      add_add_acc        += sum w_add * w_add * inv_C
//      remove_remove_acc  += sum w_rem * w_rem * inv_C
//      add_remove_acc     += sum w_add * w_rem * inv_C
//   reduce + atomicAdd into the five per-pair outputs.
//
// (Stub; same status as get_ll.)
template <class SourceT>
CUDA_KERNEL
void wdm_het_swap_ll_kernel(
    double *d_h_add_out, double *d_h_remove_out,
    double *add_add_out, double *remove_remove_out, double *add_remove_out,
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,               // Nf, Nt, ind_min_f, ind_min_t,
                                             // Nf_active, Nt_active, ...
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset,
    double *wdm_window,
    double *data_d, double *invC,            // ACTIVE-band layout:
                                             //   data_d (C, Nf_active, Nt_active)
                                             //   invC (TDI_XYZ)
                                             //     (C, C, Nf_active, Nt_active)
                                             //   invC (TDI_AET/AE)
                                             //     (C, Nf_active, Nt_active)
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,                         // TDI_XYZ / TDI_AET / TDI_AE
    double tukey_alpha,
    int N_cp_sig,                            // 0 -> direct; >0 -> source-signal spline cache
    int N_cp_orbit,                          // 0 -> raw orbits; >0 -> orbit spline cache
    // Per-block heap workspace: need TWO chunk_fd / chunk_wdm slabs
    // per chunk (one for the add template, one for the remove).
    // tdi_channels_buf is also on heap (shared by add/remove since it is
    // overwritten by each get_tdi call before being consumed).
    cmplx  *ws_chunk_fd_add_all,    cmplx  *ws_chunk_fd_rem_all,
    cmplx  *ws_layer_scratch_all,
    double *ws_chunk_wdm_add_all,   double *ws_chunk_wdm_rem_all,
    cmplx  *ws_tdi_channels_all,    // (n_chunks * nchannels * N_sparse)
    void   *get_tdi_scratch_all,    int     get_tdi_scratch_len_per_block,
    // Layer-grouping (same convention as get_ll: n_groups == 0 -> per-pair
    // all-m loop; n_groups > 0 -> middle group loop, m-band restricted).
    // The "binary index" here is the (add, remove) pair index; both
    // params_add_all[bin_i] and params_remove_all[bin_i] are read.
    int    *binary_perm,
    int    *group_starts, int *group_ends,
    int    *group_m_lo,   int *group_m_hi,
    int     n_groups,
    // Two-pass swap_ll: per-pair (sorted order, indexed by bin_iter)
    // remove-template m-band, used in pass 2 to pick up the d_h_rem and
    // rem_rem pixels that lie outside the group's add band.
    int    *pair_m_lo_b, int *pair_m_hi_b
)
{
    SourceT src(orbits, tdi_config, T, t_ref);

    // Shadow the WDM constants for readable loop bodies (no duplicated
    // signature parameters -- the kernel reads everything from
    // ``wdm_settings``).
    const int Nf         = wdm_settings->Nf;
    const int Nt         = wdm_settings->Nt;
    const int ind_min_f  = wdm_settings->ind_min_f;
    const int ind_min_t  = wdm_settings->ind_min_t;
    const int Nf_active  = wdm_settings->Nf_active;
    const int Nt_active  = wdm_settings->Nt_active;

    // Heterodyne shared workspace (tdi_channels_buf moved to heap). The
    // direct-path and spline-path buffers are OVERLAID via a union -- only
    // one branch runs per binary (add + remove templates both go through
    // the same branch back-to-back inside the binary loop, never mixed
    // direct/spline). Saves ~16 KB shared per kernel; spline amp/phase
    // coeff stacks are single-channel (fit+eval per channel inside
    // fast_wdm_inner_heterodyne_spline) for another ~6 KB.
    CUDA_SHARED alignas(WDM_HET_PATH_ALIGN) char path_arena[WDM_HET_PATH_BYTES];
    WDMHetDirectBufs *direct = reinterpret_cast<WDMHetDirectBufs *>(path_arena);
    WDMHetSplineBufs *spline = reinterpret_cast<WDMHetSplineBufs *>(path_arena);
    const bool use_spline_cache = (N_cp_sig > 0 && N_cp_sig <= FAST_WDM_N_CP_SIG_MAX
                                   && N_cp_sig < N_sparse);
    CUDA_SHARED cmplx  slow_buf        [FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    CUDA_SHARED double partial_dh_add[FAST_WDM_N_SPARSE_MAX];
    CUDA_SHARED double partial_dh_rem[FAST_WDM_N_SPARSE_MAX];
    CUDA_SHARED double partial_aa    [FAST_WDM_N_SPARSE_MAX];
    CUDA_SHARED double partial_rr    [FAST_WDM_N_SPARSE_MAX];
    CUDA_SHARED double partial_ar    [FAST_WDM_N_SPARSE_MAX];

    // Per-thread local cache for data + invC. Same convention as get_ll
    // -- populated once per (chunk, group) on GPU; disabled on CPU (cap
    // = 1) so the kernel falls back to direct reads on single-thread CPU.
    // Per-thread data/invC cache REMOVED here too (see get_ll kernel for
    // the rationale). Direct global reads pick up automatic L1/L2 caching
    // and avoid the spill-to-local-mem footgun.

    // Orbit spline cache (populated once per chunk; reused across binaries).
    CUDA_SHARED double orbit_t_cp_buf  [FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_y_buf [6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c1_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c2_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c3_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_y_buf [9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c1_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c2_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c3_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_B_buf     [FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pcr_buf   [8 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED OrbitsSplineCache orbit_cache_storage;
    const bool use_orbit_cache = (N_cp_orbit > 0 && N_cp_orbit <= FAST_WDM_N_CP_ORBIT_MAX);

    // Per-block CUDA_SHARED layer iFFT scratch -- avoids the per-element
    // global-mem latency that the previous heap-resident
    // ``ws_layer_scratch_all`` slice incurred. On CPU CUDA_SHARED stubs
    // to nothing so this lands on the stack; sized at
    // FAST_WDM_NT_SUB_MAX (256 GPU / 4096 CPU).
    CUDA_SHARED cmplx layer_scratch[FAST_WDM_NT_SUB_MAX];

    // OUTER: chunks on grid.Z. See ``wdm_het_get_ll_kernel`` for the
    // 3D-grid rationale and (BLOCK_START_X, j) scratch-slot key.
    for (int j = BLOCK_START_Z; j < n_chunks; j += GRID_INCR_Z) {
        const size_t scratch_slot = (size_t) BLOCK_START_X * (size_t) n_chunks
                                     + (size_t) j;
        cmplx  *chunk_fd_add  = &ws_chunk_fd_add_all [scratch_slot * nchannels * n_rfft_chunk];
        cmplx  *chunk_fd_rem  = &ws_chunk_fd_rem_all [scratch_slot * nchannels * n_rfft_chunk];
        // layer_scratch lives in CUDA_SHARED (declared at kernel top).
        double *w_chunk_add   = &ws_chunk_wdm_add_all[scratch_slot * nchannels * Nf * Nt_sub];
        double *w_chunk_rem   = &ws_chunk_wdm_rem_all[scratch_slot * nchannels * Nf * Nt_sub];
        cmplx  *tdi_channels_buf = &ws_tdi_channels_all[scratch_slot * nchannels * N_sparse];
        void   *get_tdi_scratch = (char *) get_tdi_scratch_all
            + scratch_slot * (size_t) get_tdi_scratch_len_per_block;

        const int keep_lo     = chunk_keep_lo[j];
        const int keep_hi     = chunk_keep_hi[j];
        const int n_global_lo = chunk_n_global_offset[j];
        const double chunk_t0 = chunk_t_starts[j];

        // Populate orbit cache once per chunk if enabled.
        OrbitsSplineCache *orbit_cache_ptr = nullptr;
        if (use_orbit_cache) {
            populate_orbit_spline_cache(
                &orbit_cache_storage, orbits,
                chunk_t0, T_chunk, N_cp_orbit,
                orbit_t_cp_buf,
                orbit_ltt_y_buf, orbit_ltt_c1_buf, orbit_ltt_c2_buf, orbit_ltt_c3_buf,
                orbit_pos_y_buf, orbit_pos_c1_buf, orbit_pos_c2_buf, orbit_pos_c3_buf,
                orbit_B_buf, orbit_pcr_buf);
            CUDA_SYNC_THREADS;
            orbit_cache_ptr = &orbit_cache_storage;
        }

        const int n_group_iter = (n_groups > 0) ? n_groups : 1;
        for (int g = 0; g < n_group_iter; ++g) {
            int bin_iter_lo, bin_iter_hi, m_lo, m_hi;
            if (n_groups > 0) {
                bin_iter_lo = group_starts[g];
                bin_iter_hi = group_ends  [g];
                m_lo = group_m_lo[g];   if (m_lo < 0)  m_lo = 0;
                m_hi = group_m_hi[g];   if (m_hi > Nf) m_hi = Nf;
            } else {
                bin_iter_lo = 0;
                bin_iter_hi = num_bin;
                m_lo = 0;
                m_hi = Nf;
            }

            // (Per-thread cache populate removed; direct reads below.)

            // INNER: binaries -- grid-strided on the X axis (see the
            // equivalent comment in ``wdm_het_get_ll_kernel``).
        for (int bin_iter = bin_iter_lo + BLOCK_START_X;
             bin_iter < bin_iter_hi;
             bin_iter += GRID_INCR_X) {
            const int bin_i = (n_groups > 0) ? binary_perm[bin_iter] : bin_iter;
            double *params_add = &params_add_all   [(size_t) bin_i * nparams];
            double *params_rem = &params_remove_all[(size_t) bin_i * nparams];

            // Zero workspaces and partials. Same live-region-only
            // optimization as wdm_het_get_ll_kernel: the chunk_fd zero band
            // must cover the gb_chunk_fd_to_wdm READ range (which is
            // determined by [m_lo, m_hi), the group band -- same for both
            // add and rem templates), not just the heterodyne write band.
            // Boundary fold cases fall back to a full zero.
            const int half_Nt_sub_zero = Nt_sub / 2;
            int kfd_lo, kfd_hi;
            if (m_lo <= 0 || m_hi >= Nf) {
                kfd_lo = 0;
                kfd_hi = n_rfft_chunk;
            } else {
                kfd_lo = (m_lo - 1) * half_Nt_sub_zero;
                kfd_hi = (m_hi + 1) * half_Nt_sub_zero;
                if (kfd_lo < 0)             kfd_lo = 0;
                if (kfd_hi > n_rfft_chunk)  kfd_hi = n_rfft_chunk;
            }
            const int kfd_band = (kfd_hi > kfd_lo) ? (kfd_hi - kfd_lo) : 0;
            for (int idx = THREAD_START_X; idx < nchannels * kfd_band;
                 idx += BLOCK_INCR_X) {
                const int c = idx / kfd_band;
                const int k = kfd_lo + (idx - c * kfd_band);
                chunk_fd_add[c * n_rfft_chunk + k] = cmplx(0.0, 0.0);
                chunk_fd_rem[c * n_rfft_chunk + k] = cmplx(0.0, 0.0);
            }

            int wm_lo = (m_lo < ind_min_f) ? ind_min_f : m_lo;
            int wm_hi = (m_hi > ind_min_f + Nf_active)
                          ? (ind_min_f + Nf_active) : m_hi;
            const int wm_band = (wm_hi > wm_lo) ? (wm_hi - wm_lo) : 0;
            for (int idx = THREAD_START_X;
                 idx < nchannels * wm_band * Nt_sub;
                 idx += BLOCK_INCR_X) {
                const int c    = idx / (wm_band * Nt_sub);
                const int rem  = idx - c * wm_band * Nt_sub;
                const int mloc = rem / Nt_sub;
                const int n    = rem - mloc * Nt_sub;
                const int m    = wm_lo + mloc;
                w_chunk_add[((size_t) c * Nf + m) * Nt_sub + n] = 0.0;
                w_chunk_rem[((size_t) c * Nf + m) * Nt_sub + n] = 0.0;
            }
            partial_dh_add[THREAD_START_X] = 0.0;
            partial_dh_rem[THREAD_START_X] = 0.0;
            partial_aa    [THREAD_START_X] = 0.0;
            partial_rr    [THREAD_START_X] = 0.0;
            partial_ar    [THREAD_START_X] = 0.0;
            CUDA_SYNC_THREADS;

            // Add template
            if (use_spline_cache) {
                fast_wdm_inner_heterodyne_spline(
                    chunk_fd_add, &src, params_add, bin_i, src.f0_index,
                    chunk_t0, T_chunk,
                    N_sparse, log2_N_sparse, N_cp_sig,
                    n_rfft_chunk, nchannels, tukey_alpha,
                    spline->t_cp_buf,
                    spline->amp_y_buf, spline->amp_c1_buf,
                    spline->amp_c2_buf, spline->amp_c3_buf,
                    spline->phase_y_buf, spline->phase_c1_buf,
                    spline->phase_c2_buf, spline->phase_c3_buf,
                    spline->dphi_ref_y_buf, spline->dphi_ref_c1_buf,
                    spline->dphi_ref_c2_buf, spline->dphi_ref_c3_buf,
                    spline->B_buf, spline->pcr_scratch,
                    spline->phi_ref_un_het_buf,
                    spline->tdi_channels_cp_buf, slow_buf,
                    spline->extract_scratch_buf,
                    (int) sizeof(spline->extract_scratch_buf),
                    orbit_cache_ptr
                );
            } else {
                fast_wdm_inner_heterodyne(
                    chunk_fd_add, &src, params_add, bin_i, src.f0_index,
                    chunk_t0, T_chunk,
                    N_sparse, log2_N_sparse, n_rfft_chunk, nchannels, tukey_alpha,
                    direct->t_sparse_buf, direct->tdi_amp_buf,
                    direct->tdi_phase_buf, direct->phi_ref_buf,
                    tdi_channels_buf, slow_buf,
                    get_tdi_scratch, get_tdi_scratch_len_per_block,
                    orbit_cache_ptr
                );
            }
            CUDA_SYNC_THREADS;
            gb_chunk_fd_to_wdm(
                w_chunk_add, chunk_fd_add, wdm_window,
                Nf, Nt_sub, log2_Nt_sub, n_rfft_chunk, dt, nchannels,
                layer_scratch,
                /*m_lo=*/m_lo, /*m_hi=*/m_hi
            );
            CUDA_SYNC_THREADS;

            // Remove template
            if (use_spline_cache) {
                fast_wdm_inner_heterodyne_spline(
                    chunk_fd_rem, &src, params_rem, bin_i, src.f0_index,
                    chunk_t0, T_chunk,
                    N_sparse, log2_N_sparse, N_cp_sig,
                    n_rfft_chunk, nchannels, tukey_alpha,
                    spline->t_cp_buf,
                    spline->amp_y_buf, spline->amp_c1_buf,
                    spline->amp_c2_buf, spline->amp_c3_buf,
                    spline->phase_y_buf, spline->phase_c1_buf,
                    spline->phase_c2_buf, spline->phase_c3_buf,
                    spline->dphi_ref_y_buf, spline->dphi_ref_c1_buf,
                    spline->dphi_ref_c2_buf, spline->dphi_ref_c3_buf,
                    spline->B_buf, spline->pcr_scratch,
                    spline->phi_ref_un_het_buf,
                    spline->tdi_channels_cp_buf, slow_buf,
                    spline->extract_scratch_buf,
                    (int) sizeof(spline->extract_scratch_buf),
                    orbit_cache_ptr
                );
            } else {
                fast_wdm_inner_heterodyne(
                    chunk_fd_rem, &src, params_rem, bin_i, src.f0_index,
                    chunk_t0, T_chunk,
                    N_sparse, log2_N_sparse, n_rfft_chunk, nchannels, tukey_alpha,
                    direct->t_sparse_buf, direct->tdi_amp_buf,
                    direct->tdi_phase_buf, direct->phi_ref_buf,
                    tdi_channels_buf, slow_buf,
                    get_tdi_scratch, get_tdi_scratch_len_per_block,
                    orbit_cache_ptr
                );
            }
            CUDA_SYNC_THREADS;
            gb_chunk_fd_to_wdm(
                w_chunk_rem, chunk_fd_rem, wdm_window,
                Nf, Nt_sub, log2_Nt_sub, n_rfft_chunk, dt, nchannels,
                layer_scratch,
                /*m_lo=*/m_lo, /*m_hi=*/m_hi
            );
            CUDA_SYNC_THREADS;

            // Accumulate the 5 quantities -- active-band indexing for both
            // dispatch branches (see get_ll comments for the layout):
            //   TDI_XYZ   -> invC (C, C, Nf_active, Nt_active), full sum.
            //   TDI_AET/AE -> invC (C, Nf_active, Nt_active) diagonal.
            // data_d is always (C, Nf_active, Nt_active). Pixels outside
            // the active band contribute zero and are skipped.
            const int ind_max_f_excl = ind_min_f + Nf_active;
            const int ind_max_t_excl = ind_min_t + Nt_active;
            const int m_lo_act = (m_lo < ind_min_f) ? ind_min_f : m_lo;
            const int m_hi_act = (m_hi > ind_max_f_excl) ? ind_max_f_excl : m_hi;
            if (tdi_type == TDI_XYZ) {
                for (int m = m_lo_act; m < m_hi_act; ++m) {
                    const int m_act = m - ind_min_f;
                    for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                         n_loc += BLOCK_INCR_X) {
                        const int n_glob = n_global_lo + (n_loc - keep_lo);
                        if (n_glob < ind_min_t || n_glob >= ind_max_t_excl) continue;
                        const int n_act = n_glob - ind_min_t;
                        double wa_arr[FAST_WDM_NCHANNELS_MAX];
                        double wr_arr[FAST_WDM_NCHANNELS_MAX];
                        double d_arr [FAST_WDM_NCHANNELS_MAX];
                        for (int c = 0; c < nchannels; ++c) {
                            const size_t g_w  = ((size_t) c * Nf + m) * Nt_sub + n_loc;
                            const size_t g_dt = ((size_t) c * Nf_active + m_act)
                                                 * Nt_active + n_act;
                            wa_arr[c] = w_chunk_add[g_w];
                            wr_arr[c] = w_chunk_rem[g_w];
                            d_arr [c] = data_d[g_dt];
                        }
                        for (int c1 = 0; c1 < nchannels; ++c1) {
                            for (int c2 = 0; c2 < nchannels; ++c2) {
                                const size_t g_inv = (((size_t) c1 * nchannels + c2)
                                                       * Nf_active + m_act)
                                                      * Nt_active + n_act;
                                const double inv = invC[g_inv];
                                partial_dh_add[THREAD_START_X] += d_arr[c1] * wa_arr[c2] * inv;
                                partial_dh_rem[THREAD_START_X] += d_arr[c1] * wr_arr[c2] * inv;
                                partial_aa    [THREAD_START_X] += wa_arr[c1] * wa_arr[c2] * inv;
                                partial_rr    [THREAD_START_X] += wr_arr[c1] * wr_arr[c2] * inv;
                                partial_ar    [THREAD_START_X] += wa_arr[c1] * wr_arr[c2] * inv;
                            }
                        }
                    }
                }
            } else {
                for (int c = 0; c < nchannels; ++c) {
                    for (int m = m_lo_act; m < m_hi_act; ++m) {
                        const int m_act = m - ind_min_f;
                        for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                             n_loc += BLOCK_INCR_X) {
                            const int n_glob = n_global_lo + (n_loc - keep_lo);
                            if (n_glob < ind_min_t || n_glob >= ind_max_t_excl) continue;
                            const int n_act = n_glob - ind_min_t;
                            const size_t g_w  = ((size_t) c * Nf + m) * Nt_sub + n_loc;
                            const size_t g_dt = ((size_t) c * Nf_active + m_act)
                                                 * Nt_active + n_act;
                            const double wa  = w_chunk_add[g_w];
                            const double wr  = w_chunk_rem[g_w];
                            const double d   = data_d[g_dt];
                            const double inv = invC  [g_dt];
                            partial_dh_add[THREAD_START_X] += d * wa * inv;
                            partial_dh_rem[THREAD_START_X] += d * wr * inv;
                            partial_aa    [THREAD_START_X] += wa * wa * inv;
                            partial_rr    [THREAD_START_X] += wr * wr * inv;
                            partial_ar    [THREAD_START_X] += wa * wr * inv;
                        }
                    }
                }
            }

            // Pass 2 (two-pass swap_ll): when the remove template's
            // carrier band [pair_m_lo_b, pair_m_hi_b) extends outside
            // source 1's band [m_lo, m_hi) we still need d_h_rem and
            // rem_rem contributions from those out-of-band m's (w_rem is
            // non-zero there). All cross terms (ar, aa, d_h_add) already
            // converged in pass 1 since w_add == 0 outside [m_lo, m_hi).
            //
            // Strategy: re-populate the per-thread cache with source 2's
            // 5-layer band data/invC, then iterate that band skipping any
            // m already covered by pass 1 (to avoid double-counting).
            //
            // When grouping is disabled (n_groups == 0) we skip pass 2:
            // pass 1 already iterates m in [0, Nf) which subsumes source
            // 2's band.
            if (n_groups > 0) {
                const int m_lo_b = pair_m_lo_b[bin_iter];
                const int m_hi_b_in = pair_m_hi_b[bin_iter];
                const int m_lo_b_c = (m_lo_b < 0) ? 0 : m_lo_b;
                const int m_hi_b   = (m_hi_b_in > Nf) ? Nf : m_hi_b_in;
                const bool need_pass2 = (m_lo_b_c != m_lo) || (m_hi_b != m_hi);
                if (need_pass2 && m_lo_b_c < m_hi_b) {
                    // Pass-2 accumulator: same tdi_type dispatch + active-band
                    // indexing as pass 1. Adds d_h_rem / rem_rem from m-layers
                    // outside source 1's band, skipping pass-1 overlap.
                    const int m_lo_b_act = (m_lo_b_c < ind_min_f) ? ind_min_f : m_lo_b_c;
                    const int m_hi_b_act = (m_hi_b   > ind_max_f_excl) ? ind_max_f_excl : m_hi_b;
                    if (tdi_type == TDI_XYZ) {
                        for (int m = m_lo_b_act; m < m_hi_b_act; ++m) {
                            const bool in_pass1 = (m >= m_lo) && (m < m_hi);
                            if (in_pass1) continue;
                            const int m_act = m - ind_min_f;
                            for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                                 n_loc += BLOCK_INCR_X) {
                                const int n_glob = n_global_lo + (n_loc - keep_lo);
                                if (n_glob < ind_min_t || n_glob >= ind_max_t_excl) continue;
                                const int n_act = n_glob - ind_min_t;
                                double wr_arr[FAST_WDM_NCHANNELS_MAX];
                                double d_arr [FAST_WDM_NCHANNELS_MAX];
                                for (int c = 0; c < nchannels; ++c) {
                                    const size_t g_w  = ((size_t) c * Nf + m) * Nt_sub + n_loc;
                                    const size_t g_dt = ((size_t) c * Nf_active + m_act)
                                                         * Nt_active + n_act;
                                    wr_arr[c] = w_chunk_rem[g_w];
                                    d_arr [c] = data_d[g_dt];
                                }
                                for (int c1 = 0; c1 < nchannels; ++c1) {
                                    for (int c2 = 0; c2 < nchannels; ++c2) {
                                        const size_t g_inv = (((size_t) c1 * nchannels + c2)
                                                               * Nf_active + m_act)
                                                              * Nt_active + n_act;
                                        const double inv = invC[g_inv];
                                        partial_dh_rem[THREAD_START_X] += d_arr[c1] * wr_arr[c2] * inv;
                                        partial_rr    [THREAD_START_X] += wr_arr[c1] * wr_arr[c2] * inv;
                                    }
                                }
                            }
                        }
                    } else {
                        for (int c = 0; c < nchannels; ++c) {
                            for (int m = m_lo_b_act; m < m_hi_b_act; ++m) {
                                const bool in_pass1 = (m >= m_lo) && (m < m_hi);
                                if (in_pass1) continue;
                                const int m_act = m - ind_min_f;
                                for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                                     n_loc += BLOCK_INCR_X) {
                                    const int n_glob = n_global_lo + (n_loc - keep_lo);
                                    if (n_glob < ind_min_t || n_glob >= ind_max_t_excl) continue;
                                    const int n_act = n_glob - ind_min_t;
                                    const size_t g_w = ((size_t) c * Nf + m) * Nt_sub + n_loc;
                                    const size_t g_dt = ((size_t) c * Nf_active + m_act)
                                                         * Nt_active + n_act;
                                    const double wr  = w_chunk_rem[g_w];
                                    const double d   = data_d[g_dt];
                                    const double inv = invC  [g_dt];
                                    partial_dh_rem[THREAD_START_X] += d * wr * inv;
                                    partial_rr    [THREAD_START_X] += wr * wr * inv;
                                }
                            }
                        }
                    }
                }
            }
            CUDA_SYNC_THREADS;

#ifdef __CUDACC__
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (THREAD_START_X < stride) {
                    partial_dh_add[THREAD_START_X] += partial_dh_add[THREAD_START_X + stride];
                    partial_dh_rem[THREAD_START_X] += partial_dh_rem[THREAD_START_X + stride];
                    partial_aa    [THREAD_START_X] += partial_aa    [THREAD_START_X + stride];
                    partial_rr    [THREAD_START_X] += partial_rr    [THREAD_START_X + stride];
                    partial_ar    [THREAD_START_X] += partial_ar    [THREAD_START_X + stride];
                }
                CUDA_SYNC_THREADS;
            }
            if (THREAD_START_X == 0) {
                atomicAdd(&d_h_add_out      [bin_i], partial_dh_add[0]);
                atomicAdd(&d_h_remove_out   [bin_i], partial_dh_rem[0]);
                atomicAdd(&add_add_out      [bin_i], partial_aa    [0]);
                atomicAdd(&remove_remove_out[bin_i], partial_rr    [0]);
                atomicAdd(&add_remove_out   [bin_i], partial_ar    [0]);
            }
#else
            d_h_add_out      [bin_i] += partial_dh_add[0];
            d_h_remove_out   [bin_i] += partial_dh_rem[0];
            add_add_out      [bin_i] += partial_aa    [0];
            remove_remove_out[bin_i] += partial_rr    [0];
            add_remove_out   [bin_i] += partial_ar    [0];
#endif
            CUDA_SYNC_THREADS;
        } // bin_iter (pair)
        } // g (group)
    }
}
#endif // ---- OLD KERNELS END ----


// =============================================================================
// NEW shared-memory-only chunked-het kernels.
//
// Design (per user direction 2026-05-29):
//   * One binary per block on the X axis. blockDim.x = NUM_THREADS_HERE (= 64
//     on GPU, 1 on CPU). The block iterates chunks sequentially via a
//     for-loop -- a future commit will move chunks onto blockIdx.y; the
//     sequential loop is marked with a TODO comment.
//   * Per (chunk, m_layer): the per-channel tdi_channel slow-signal samples
//     are computed DIRECTLY from get_tdi (no separate amp/phase cache),
//     heterodyned in time domain via exp(-i 2 pi f0_grid t), FFT'd in shared
//     memory, windowed for this layer's WDM filter, iFFT'd in the same
//     buffer, parity factor applied, then each thread holds one (m, n_loc)
//     WDM coefficient.
//   * Each thread streams data[c, m_act, n_act] and invC[c1, c2, m_act, n_act]
//     from global memory (coalesced -- warp lanes hit adjacent n_act addresses
//     since each thread owns one n_loc), accumulates per-thread partials
//     (registers).
//   * Block-wide partial_dh / partial_hh in shared mem for the final tree
//     reduction; thread 0 atomicAdds into the global outputs.
//
// Shared-memory layout per block (~13 KB at Nt_sub=256, NUM_THREADS=64):
//     tdi_channel_buf[3 * Nt_sub] cmplx  (12 KB)  -- per-channel FFT/iFFT scratch
//     partial_dh[blockDim.x]      double ( 0.5 KB)
//     partial_hh[blockDim.x]      double ( 0.5 KB)
//
// We deliberately do NOT cache:
//   - heterodyne FFT output across layers (recompute per (chunk, m_layer))
//   - tdi_amp/tdi_phase/phi_ref (computed inline)
//   - orbit splines (use raw orbits->get_* per evaluation)
// These caches can be reintroduced as a perf optimization once the kernel
// structure is validated. The point of this rewrite is correctness +
// memory-footprint clarity, not maximum throughput.
//
// Constraints:
//   - Nt_sub and N_sparse must be powers of 2 (radix-2 FFT). The host-side
//     WDMSettings constructor already enforces this.
//   - blockDim.x == Nt_sub is NOT required; the FFT helpers in
//     WDMSplineHelpers.hh thread-stride over the array.
// =============================================================================
template <class SourceT>
CUDA_KERNEL
void wdm_het_get_ll_kernel(
    double *d_h_out, double *h_h_out,        // (num_bin,) outputs (host pre-zero'd)
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all,                      // (num_bin * nparams,)
    int    *data_index_all, int *noise_index_all,
    double *chunk_t_starts,                  // (n_chunks,)
    int    *chunk_keep_lo, int *chunk_keep_hi,
    int    *chunk_n_global_offset,
    double *wdm_window,                      // (Nt_sub,)
    double *data_d, double *invC,            // active-band layout (see contract below)
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int    m_band_half_width,
    int    N_cp_orbit)   // 0 -> raw orbit lookups; >0 -> per-chunk spline cache
{
    // One binary per block (grid.X); chunks iterated sequentially inside the
    // block. See the kernel-section header comment above for the full design.

    // Construct source class (carries orbits / tdi_config pointers).
    SourceT src(orbits, tdi_config, T, t_ref);

    // Hoist scalar WDM grid constants from the device-resident struct into
    // local registers -- the compiler keeps them across the inner loops, so
    // the per-binary inner work pays only one load each, not one per
    // iteration.
    const int Nf         = wdm_settings->Nf;
    const int Nt         = wdm_settings->Nt;
    const int ind_min_f  = wdm_settings->ind_min_f;
    const int ind_min_t  = wdm_settings->ind_min_t;
    const int Nf_active  = wdm_settings->Nf_active;
    const int Nt_active  = wdm_settings->Nt_active;
    (void) Nt;  // unused in get_ll (kept for layout parity with the OLD kernel)

    // Dynamic shared-memory layout (set by ``shared_bytes`` at kernel launch).
    // Two cmplx buffers per (chunk, binary):
    //
    //   fd_chunk_buf   [nchannels * N_sparse] cmplx  -- HOLDS the chunk-FD
    //                                                   (TD build -> heterodyne
    //                                                   -> Tukey -> FFT, done
    //                                                   ONCE per chunk).
    //   layer_buf      [nchannels * Nt_sub]   cmplx  -- per-m_layer scratch
    //                                                   (window+rearrange ->
    //                                                   iFFT -> parity ->
    //                                                   accumulate).
    //   partial_dh     [blockDim.x]           double
    //   partial_hh     [blockDim.x]           double
    //
    // At Nt_sub=N_sparse=256, total ~25 KB (well under 48 KB default on A100).
    // The chunk-FD is computed ONCE per chunk and reused for every m_layer in
    // the binary's band -- previously we did the full TD-build+FFT per m,
    // wasting (m_band_width - 1) x n_chunks worth of FFT work per binary.
    //
    // N_sparse and Nt_sub may differ: fd_chunk_buf is sized by N_sparse
    // (forward-FFT length) and layer_buf is sized by Nt_sub (iFFT length).
    // Step 5 maps from one to the other -- bins of the wider layer that
    // fall outside the narrower FD window get zero-filled by the
    // ``if (fft_bin >= -half_Nsp && fft_bin < half_Nsp)`` guard. Useful
    // when a narrowband source only needs a smaller chunk-FD window.
#ifdef __CUDACC__
    extern CUDA_SHARED char shared_mem[];
    cmplx  *fd_chunk_buf    = (cmplx *) shared_mem;
    cmplx  *layer_buf       = &fd_chunk_buf[(size_t) nchannels * N_sparse];
    double *partial_dh      = (double *) &layer_buf[(size_t) nchannels * Nt_sub];
    double *partial_hh      = &partial_dh[NUM_THREADS_HERE];
    // cufftdx scratch (only used by wdm_fft_dispatch when LISA_USE_CUFFTDX
    // is defined; sized as the max across instantiated FFT lengths).
    char   *fft_scratch     = (char *) &partial_hh[NUM_THREADS_HERE];
#else
    // CPU stubs: stack arrays sized at the compile-time maxima.
    cmplx  fd_chunk_buf_cpu [FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    cmplx  layer_buf_cpu    [FAST_WDM_NCHANNELS_MAX * FAST_WDM_NT_SUB_MAX];
    double partial_dh_cpu   [1];
    double partial_hh_cpu   [1];
    cmplx  *fd_chunk_buf    = fd_chunk_buf_cpu;
    cmplx  *layer_buf       = layer_buf_cpu;
    double *partial_dh      = partial_dh_cpu;
    double *partial_hh      = partial_hh_cpu;
    char   *fft_scratch     = nullptr;  // unused on CPU
#endif

    const double layer_df = 1.0 / (2.0 * (double) Nf * dt);
    const double df_chunk = 1.0 / T_chunk;

    CUDA_SHARED int link_sc_rec[NLINKS];
    CUDA_SHARED int link_sc_em [NLINKS];
    src.fill_link_arrays(link_sc_rec, link_sc_em);

    // Orbit spline-cache buffers (used only if N_cp_orbit > 0). Mirrors the
    // buffer set consumed by ``populate_orbit_spline_cache``. Sized at
    // FAST_WDM_N_CP_ORBIT_MAX so the kernel JITs once and dispatches against
    // any 0 < N_cp_orbit <= max. At N_cp_orbit=32 this adds ~15.6 KB to
    // shared-mem; capped at 48 -> ~23 KB. Replaces global-mem orbit table
    // reads inside get_tdi_Xf_single with cooperative shared-mem cubic
    // spline evals.
    CUDA_SHARED double orbit_t_cp_buf  [FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_y_buf [6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c1_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c2_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_ltt_c3_buf[6 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_y_buf [9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c1_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c2_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pos_c3_buf[9 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_B_buf     [FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED double orbit_pcr_buf   [8 * FAST_WDM_N_CP_ORBIT_MAX];
    CUDA_SHARED OrbitsSplineCache orbit_cache_storage;
    const bool use_orbit_cache =
        (N_cp_orbit > 0 && N_cp_orbit <= FAST_WDM_N_CP_ORBIT_MAX);

    CUDA_SYNC_THREADS;

    // One binary per block on grid.X. Grid-stride if num_bin > gridDim.x.
    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X) {
        double *params      = &params_all[(size_t) bin_i * nparams];
        const int data_ind  = data_index_all[bin_i];
        const int noise_ind = noise_index_all[bin_i];
        (void) noise_ind;  // invC already incorporates noise; kept for API parity
        (void) data_ind;   // data_d / invC are indexed directly (no per-binary slab)

        // Per-binary inner-product accumulators (in registers).
        double tmp_dh = 0.0;
        double tmp_hh = 0.0;

        // Carrier bin in chunk-FD coordinates + WDM m-band centred on f0.
        const double f0       = params[src.f0_index];
        const int    k_f0     = (int) round(f0 / df_chunk);
        const double f0_grid  = (double) k_f0 * df_chunk;
        const int    m_floor  = (int) (f0 / layer_df);
        int          m_lo     = m_floor - m_band_half_width;
        int          m_hi     = m_floor + m_band_half_width + 1;   // exclusive
        // Clip to active band -- the accumulator only reads pixels in the
        // active band anyway, so processing layers outside is wasted work.
        if (m_lo < ind_min_f)             m_lo = ind_min_f;
        if (m_hi > ind_min_f + Nf_active) m_hi = ind_min_f + Nf_active;

        Vec k_sky(0.0, 0.0, 0.0);
        Vec u_sky(0.0, 0.0, 0.0);
        Vec v_sky(0.0, 0.0, 0.0);
        src.get_sky_vectors(&k_sky, &u_sky, &v_sky, params);

        // TODO(blockIdx.y on chunks): when chunks move to grid.Y, change
        //   ``for (int j = 0; j < n_chunks; ++j)`` to
        //   ``for (int j = BLOCK_START_Y; j < n_chunks; j += GRID_INCR_Y)``.
        for (int j = 0; j < n_chunks; ++j) {
            const int    keep_lo     = chunk_keep_lo[j];
            const int    keep_hi     = chunk_keep_hi[j];
            const int    n_global_lo = chunk_n_global_offset[j];
            const double chunk_t0    = chunk_t_starts[j];
            const double dt_sparse   = T_chunk / (double) N_sparse;

            // Populate orbit spline cache once over this chunk's time
            // window [chunk_t0, chunk_t0 + T_chunk]. All threads cooperate
            // (PCR solver on GPU, Thomas on CPU). Skipped when N_cp_orbit==0.
            OrbitsSplineCache *orbit_cache_ptr = nullptr;
            if (use_orbit_cache) {
                populate_orbit_spline_cache(
                    &orbit_cache_storage, orbits,
                    chunk_t0, T_chunk, N_cp_orbit,
                    orbit_t_cp_buf,
                    orbit_ltt_y_buf, orbit_ltt_c1_buf,
                    orbit_ltt_c2_buf, orbit_ltt_c3_buf,
                    orbit_pos_y_buf, orbit_pos_c1_buf,
                    orbit_pos_c2_buf, orbit_pos_c3_buf,
                    orbit_B_buf, orbit_pcr_buf);
                CUDA_SYNC_THREADS;
                orbit_cache_ptr = &orbit_cache_storage;
            }

            // ============================================================
            // Steps 1-4 are CHUNK-LEVEL: TD-build -> heterodyne -> Tukey
            // -> FFT into fd_chunk_buf. They do NOT depend on m, so we
            // compute them ONCE per chunk and reuse across all m_layers.
            // ============================================================

            // ---- 1) compute tdi_channel(t) into fd_chunk_buf ----
            // Thread-stride over i; compute t inline as a linear ramp.
            // Writes raw complex TDI values into fd_chunk_buf with layout
            // [c * N_sparse + i]. If orbit_cache_ptr != nullptr, the TDI
            // calls use shared-mem cubic-spline orbit evals instead of
            // global-mem table lookups -- bit-equivalent to the raw path
            // at N_cp_orbit >= 32 over typical chunk lengths (LTT and
            // position residuals well below float64 precision).
            for (int i = THREAD_START_X; i < N_sparse; i += BLOCK_INCR_X) {
                const double t = chunk_t0 + (double) i * dt_sparse;
                cmplx tdi_tmp[3];
                if (orbit_cache_ptr != nullptr) {
                    src.get_tdi_Xf_single_cached(&tdi_tmp[0], t, params,
                                                  k_sky, u_sky, v_sky,
                                                  link_sc_rec, link_sc_em,
                                                  bin_i, orbit_cache_ptr);
                } else {
                    src.get_tdi_Xf_single(&tdi_tmp[0], t, params,
                                          k_sky, u_sky, v_sky,
                                          link_sc_rec, link_sc_em, bin_i);
                }
                for (int c = 0; c < nchannels; ++c)
                    fd_chunk_buf[c * N_sparse + i] = tdi_tmp[c];
            }
            CUDA_SYNC_THREADS;

            // ---- 2) time-domain heterodyne + 3) Tukey window (in place) ----
            // See per-step derivation comments below the loop.
            const double n_taper = (tukey_alpha > 0.0)
                ? 0.5 * tukey_alpha * (double) (N_sparse - 1)
                : 0.0;
            for (int idx = THREAD_START_X; idx < nchannels * N_sparse;
                 idx += BLOCK_INCR_X) {
                const int c = idx / N_sparse;
                const int i = idx - c * N_sparse;
                const double tau = (double) i * dt_sparse;
                // Original fast_wdm_inner_heterodyne routes raw cmplx TDI
                // through new_extract_amplitude_and_phase to produce
                //   slow = conj(M) * exp(I*pjump) * exp(-I 2pi f0 t)
                // For typical GB pjump=0; replicate by conjugating the
                // get_tdi_Xf output before the heterodyne multiply.
                const double het_phase = -2.0 * M_PI * f0_grid * tau;
                const cmplx  het_factor(cos(het_phase), sin(het_phase));
                cmplx s = gcmplx::conj(fd_chunk_buf[idx]) * het_factor;
                if (n_taper > 0.0) {
                    double w = 1.0;
                    const double di    = (double) i;
                    const double dlast = (double) (N_sparse - 1);
                    if (di < n_taper) {
                        const double xn = di / n_taper;
                        w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                    } else if (di > dlast - n_taper) {
                        const double xn = (dlast - di) / n_taper;
                        w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                    }
                    s = cmplx(s.real() * w, s.imag() * w);
                }
                fd_chunk_buf[idx] = s;
            }
            CUDA_SYNC_THREADS;

            // ---- 4) FFT per channel (in place in fd_chunk_buf) ----
            // Per-channel forward FFT. wdm_spline_radix2_fft ends with a
            // CUDA_SYNC_THREADS internally, so no inter-channel sync needed.
            for (int c = 0; c < nchannels; ++c) {
                wdm_fft_dispatch(&fd_chunk_buf[c * N_sparse],
                                  N_sparse, log2_N_sparse,
                                  /*inverse=*/false, fft_scratch);
            }
            // fd_chunk_buf now holds the chunk-FD; reuse for every m below.

            // ============================================================
            // Steps 5-8 are PER M_LAYER: read fd_chunk_buf, window/rearrange
            // into layer_buf, iFFT, parity, accumulate.
            // ============================================================
            const double scale_fd     = 0.5 * dt_sparse / dt;
            for (int m = m_lo; m < m_hi; ++m) {
                const int m_act = m - ind_min_f;

                // ---- 5) window + rearrange for layer m: fd_chunk_buf -> layer_buf ----
                //
                // Since fd_chunk_buf and layer_buf are separate buffers,
                // no per-thread register staging is needed -- each thread
                // reads its source bin from fd_chunk_buf and writes its
                // destination bin in layer_buf with a single sync at the
                // end (before iFFT).
                //
                // Combined heterodyne-place + WDM-read scaling:
                //   chunk_fd  = fft_output * (0.5 * dt_sparse)
                //   layer_fd  = chunk_fd / data_dt * wdm_window[k]
                // -> applied here as one factor: v *= 0.5 * dt_sparse / dt.
                const int    half_Nt_sub  = Nt_sub / 2;
                const int    half_Nsp     = N_sparse / 2;
                const int    fft_offset   = m * half_Nt_sub - half_Nt_sub - k_f0;
                for (int c = 0; c < nchannels; ++c) {
                    for (int k_idx = THREAD_START_X; k_idx < Nt_sub;
                         k_idx += BLOCK_INCR_X) {
                        int fft_bin = fft_offset + k_idx;
                        cmplx v(0.0, 0.0);
                        if (fft_bin >= -half_Nsp && fft_bin < half_Nsp) {
                            int read_bin = (fft_bin + N_sparse) % N_sparse;
                            v = fd_chunk_buf[c * N_sparse + read_bin];
                            v = cmplx(v.real() * scale_fd, v.imag() * scale_fd);
                            const double w = wdm_window[k_idx];
                            v = cmplx(v.real() * w, v.imag() * w);
                        }
                        layer_buf[c * Nt_sub + k_idx] = v;
                    }
                }
                CUDA_SYNC_THREADS;

                // ---- 6) iFFT per channel (in place in layer_buf, length Nt_sub) ----
                // Per-channel iFFT (FFT helper has its own trailing sync).
                for (int c = 0; c < nchannels; ++c) {
                    wdm_fft_dispatch(&layer_buf[c * Nt_sub],
                                      Nt_sub, log2_Nt_sub,
                                      /*inverse=*/true, fft_scratch);
                }

                // ---- 7+8) FUSED parity + inner-product accumulator ----
                //
                // For layer m and n_loc:
                //   parity_even = ((m + n_loc) & 1) == 0
                //   sign        = ((m + 1) * n_loc) is even ? +1 : -1
                //   w_arr[c]    = kappa * sign * (parity_even ? z.real() : z.imag())
                // We compute w_arr in registers directly from layer_buf and
                // immediately accumulate against global data/invC -- no
                // separate write-back to shared memory. Skip n_locs outside
                // the active time range.
                const double kappa = 2.0 * sqrt(M_PI * dt) / (double) Nf;
                const int ind_max_t_excl = ind_min_t + Nt_active;
                for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                     n_loc += BLOCK_INCR_X) {
                    const int n_glob = n_global_lo + (n_loc - keep_lo);
                    if (n_glob < ind_min_t || n_glob >= ind_max_t_excl) continue;
                    const int n_act = n_glob - ind_min_t;

                    const bool parity_even = (((m + n_loc) & 1) == 0);
                    const double sign      = ((((m + 1) * n_loc) & 1) == 0)
                                              ? 1.0 : -1.0;
                    const double psign     = kappa * sign;

                    double w_arr[FAST_WDM_NCHANNELS_MAX] = {0.};
                    double d_arr[FAST_WDM_NCHANNELS_MAX] = {0.};
                    for (int c = 0; c < nchannels; ++c) {
                        const cmplx z = layer_buf[c * Nt_sub + n_loc];
                        const double rp = parity_even ? z.real() : z.imag();
                        w_arr[c] = psign * rp;
                        const size_t g_d = ((size_t) c * Nf_active + m_act)
                                            * Nt_active + n_act;
                        d_arr[c] = data_d[g_d];
                    }
                    if (tdi_type == TDI_XYZ) {
                        // invC is Hermitian (and real): 3 diag + 3 off-diag
                        // unique reads. Each off-diag pair (c1, c2) with
                        // c1<c2 contributes BOTH (c1,c2) and (c2,c1) terms.
                        //   2D->2F: 3 + 3 = 6 reads instead of 9
                        //   tmp_dh: (c1,c2) + (c2,c1) = d[c1]*w[c2] + d[c2]*w[c1]
                        //   tmp_hh: w[c1]*w[c2] symmetric -> 2 * w[c1]*w[c2]
                        for (int c = 0; c < nchannels; ++c) {
                            const size_t g_inv =
                                (((size_t) c * nchannels + c)
                                   * Nf_active + m_act) * Nt_active + n_act;
                            const double inv = invC[g_inv];
                            tmp_dh += d_arr[c] * w_arr[c] * inv;
                            tmp_hh += w_arr[c] * w_arr[c] * inv;
                        }
                        for (int c1 = 0; c1 < nchannels - 1; ++c1) {
                            for (int c2 = c1 + 1; c2 < nchannels; ++c2) {
                                const size_t g_inv =
                                    (((size_t) c1 * nchannels + c2)
                                       * Nf_active + m_act) * Nt_active + n_act;
                                const double inv = invC[g_inv];
                                tmp_dh += (d_arr[c1] * w_arr[c2]
                                            + d_arr[c2] * w_arr[c1]) * inv;
                                tmp_hh += 2.0 * w_arr[c1] * w_arr[c2] * inv;
                            }
                        }
                    } else {
                        // TDI_AET / TDI_AE: invC is diagonal in channels.
                        for (int c = 0; c < nchannels; ++c) {
                            const size_t g_inv = ((size_t) c * Nf_active + m_act)
                                                  * Nt_active + n_act;
                            const double inv = invC[g_inv];
                            tmp_dh += d_arr[c] * w_arr[c] * inv;
                            tmp_hh += w_arr[c] * w_arr[c] * inv;
                        }
                    }
                }
                CUDA_SYNC_THREADS;
            } // end m_layer
        } // end chunk j

        // ---- per-thread -> shared-mem partials, block-wide tree reduction --
        partial_dh[THREAD_START_X] = tmp_dh;
        partial_hh[THREAD_START_X] = tmp_hh;
        CUDA_SYNC_THREADS;
#ifdef __CUDACC__
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (THREAD_START_X < stride) {
                partial_dh[THREAD_START_X] += partial_dh[THREAD_START_X + stride];
                partial_hh[THREAD_START_X] += partial_hh[THREAD_START_X + stride];
            }
            CUDA_SYNC_THREADS;
        }
        // One binary per block (grid.X) + chunks iterated sequentially
        // INSIDE the block means exactly one block writes to (d_h_out,
        // h_h_out)[bin_i] -- no cross-block race, so a direct store
        // suffices. If we later move chunks onto blockIdx.y (multiple
        // blocks per binary), this must become atomicAdd to combine
        // the per-chunk partials across blocks.
        if (THREAD_START_X == 0) {
            d_h_out[bin_i] = partial_dh[0];
            h_h_out[bin_i] = partial_hh[0];
        }
#else
        // CPU: blockDim.x == 1 (THREAD_START_X / BLOCK_INCR_X stubs collapse
        // to a single virtual thread), so partial_dh[0] already holds the
        // full sum for this binary.
        d_h_out[bin_i] = partial_dh[0];
        h_h_out[bin_i] = partial_hh[0];
#endif
        CUDA_SYNC_THREADS;
    } // end bin_i
}


template <class SourceT>
CUDA_KERNEL
void wdm_het_fill_global_kernel(
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
    int    m_band_half_width)
{
    // Same per-(chunk, m_layer) pipeline as wdm_het_get_ll_kernel:
    //   1. build sparse time grid in shared mem
    //   2. for each m_layer: compute tdi_channel + heterodyne + Tukey
    //   3. FFT length N_sparse per channel in shared mem
    //   4. window + rearrange for layer m in place (per-thread register stage)
    //   5. iFFT length Nt_sub per channel in shared mem
    //   6. parity factor -> real WDM coefficient
    //   7. atomicAdd into template_fill[c, m, n_global] (instead of the
    //      get_ll accumulator + reduction).
    // See get_ll for full design comments; the only difference is the
    // output stage at step 7.
    SourceT src(orbits, tdi_config, T, t_ref);

    const int Nf = wdm_settings->Nf;
    const int Nt = wdm_settings->Nt;

    // Dynamic shared-memory layout (set by ``shared_bytes`` at kernel launch):
    //   fd_chunk_buf [nchannels * N_sparse] cmplx  -- chunk-FD (built ONCE per chunk)
    //   layer_buf    [nchannels * Nt_sub]   cmplx  -- per-m_layer scratch
    // (no per-thread partials; fill_global writes via atomicAdd.)
#ifdef __CUDACC__
    extern CUDA_SHARED char shared_mem[];
    cmplx  *fd_chunk_buf = (cmplx *) shared_mem;
    cmplx  *layer_buf    = &fd_chunk_buf[(size_t) nchannels * N_sparse];
    // cufftdx scratch (unused unless LISA_USE_CUFFTDX is defined).
    char   *fft_scratch  = (char *) &layer_buf[(size_t) nchannels * Nt_sub];
#else
    cmplx  fd_chunk_buf_cpu[FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    cmplx  layer_buf_cpu   [FAST_WDM_NCHANNELS_MAX * FAST_WDM_NT_SUB_MAX];
    cmplx  *fd_chunk_buf = fd_chunk_buf_cpu;
    cmplx  *layer_buf    = layer_buf_cpu;
    char   *fft_scratch  = nullptr;
#endif

    const double layer_df = 1.0 / (2.0 * (double) Nf * dt);
    const double df_chunk = 1.0 / T_chunk;

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X) {
        double *params       = &params_all[(size_t) bin_i * nparams];
        const double factor  = factors_all[bin_i];

        const double f0      = params[src.f0_index];
        const int    k_f0    = (int) round(f0 / df_chunk);
        const double f0_grid = (double) k_f0 * df_chunk;
        const int    m_floor = (int) (f0 / layer_df);
        int          m_lo    = m_floor - m_band_half_width;
        int          m_hi    = m_floor + m_band_half_width + 1;
        if (m_lo < 0)   m_lo = 0;
        if (m_hi > Nf)  m_hi = Nf;

        for (int j = 0; j < n_chunks; ++j) {
            const int    keep_lo     = chunk_keep_lo[j];
            const int    keep_hi     = chunk_keep_hi[j];
            const int    n_global_lo = chunk_n_global_offset[j];
            const double chunk_t0    = chunk_t_starts[j];
            const double dt_sparse   = T_chunk / (double) N_sparse;

            // ============================================================
            // Steps 1-4 are CHUNK-LEVEL (no m dependence): TD-build,
            // heterodyne, Tukey, forward FFT into fd_chunk_buf. Compute
            // ONCE and reuse across every m_layer below.
            // ============================================================
            {
                CUDA_SHARED int link_sc_rec[NLINKS];
                CUDA_SHARED int link_sc_em [NLINKS];
                src.fill_link_arrays(link_sc_rec, link_sc_em);
                CUDA_SYNC_THREADS;
                Vec k_sky(0.0, 0.0, 0.0);
                Vec u_sky(0.0, 0.0, 0.0);
                Vec v_sky(0.0, 0.0, 0.0);
                src.get_sky_vectors(&k_sky, &u_sky, &v_sky, params);

                // ---- 1) tdi_channel(t) -> fd_chunk_buf ----
                for (int i = THREAD_START_X; i < N_sparse;
                     i += BLOCK_INCR_X) {
                    const double t = chunk_t0 + (double) i * dt_sparse;
                    cmplx tdi_tmp[3];
                    src.get_tdi_Xf_single(&tdi_tmp[0], t, params,
                                          k_sky, u_sky, v_sky,
                                          link_sc_rec, link_sc_em, bin_i);
                    for (int c = 0; c < nchannels; ++c)
                        fd_chunk_buf[c * N_sparse + i] = tdi_tmp[c];
                }
                CUDA_SYNC_THREADS;
            }

            // ---- 2) heterodyne + 3) Tukey (in place in fd_chunk_buf) ----
            const double n_taper = (tukey_alpha > 0.0)
                ? 0.5 * tukey_alpha * (double) (N_sparse - 1) : 0.0;
            for (int idx = THREAD_START_X; idx < nchannels * N_sparse;
                 idx += BLOCK_INCR_X) {
                const int i = idx - (idx / N_sparse) * N_sparse;
                const double tau = (double) i * dt_sparse;
                const double het_phase = -2.0 * M_PI * f0_grid * tau;
                const cmplx  het_factor(cos(het_phase), sin(het_phase));
                cmplx s = gcmplx::conj(fd_chunk_buf[idx]) * het_factor;
                if (n_taper > 0.0) {
                    double w = 1.0;
                    const double di    = (double) i;
                    const double dlast = (double) (N_sparse - 1);
                    if (di < n_taper) {
                        const double xn = di / n_taper;
                        w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                    } else if (di > dlast - n_taper) {
                        const double xn = (dlast - di) / n_taper;
                        w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                    }
                    s = cmplx(s.real() * w, s.imag() * w);
                }
                fd_chunk_buf[idx] = s;
            }
            CUDA_SYNC_THREADS;

            // ---- 4) FFT length N_sparse per channel (in place) ----
            // Per-channel forward FFT. wdm_spline_radix2_fft ends with a
            // CUDA_SYNC_THREADS internally, so no inter-channel sync needed.
            for (int c = 0; c < nchannels; ++c) {
                wdm_fft_dispatch(&fd_chunk_buf[c * N_sparse],
                                  N_sparse, log2_N_sparse,
                                  /*inverse=*/false, fft_scratch);
            }
            // fd_chunk_buf now holds the chunk-FD; reuse below.

            // ============================================================
            // Steps 5-7 per m_layer: window+rearrange -> iFFT -> parity
            //                        -> atomicAdd into template_fill.
            // ============================================================
            const double scale_fd     = 0.5 * dt_sparse / dt;
            for (int m = m_lo; m < m_hi; ++m) {
                // ---- 5) window + rearrange: fd_chunk_buf -> layer_buf ----
                const int    half_Nt_sub  = Nt_sub / 2;
                const int    half_Nsp     = N_sparse / 2;
                const int    fft_offset   = m * half_Nt_sub - half_Nt_sub - k_f0;
                for (int c = 0; c < nchannels; ++c) {
                    for (int k_idx = THREAD_START_X; k_idx < Nt_sub;
                         k_idx += BLOCK_INCR_X) {
                        int fft_bin = fft_offset + k_idx;
                        cmplx v(0.0, 0.0);
                        if (fft_bin >= -half_Nsp && fft_bin < half_Nsp) {
                            int read_bin = (fft_bin + N_sparse) % N_sparse;
                            v = fd_chunk_buf[c * N_sparse + read_bin];
                            v = cmplx(v.real() * scale_fd, v.imag() * scale_fd);
                            const double w = wdm_window[k_idx];
                            v = cmplx(v.real() * w, v.imag() * w);
                        }
                        layer_buf[c * Nt_sub + k_idx] = v;
                    }
                }
                CUDA_SYNC_THREADS;

                // ---- 6) iFFT length Nt_sub per channel (in place in layer_buf) ----
                // Per-channel iFFT (FFT helper has its own trailing sync).
                for (int c = 0; c < nchannels; ++c) {
                    wdm_fft_dispatch(&layer_buf[c * Nt_sub],
                                      Nt_sub, log2_Nt_sub,
                                      /*inverse=*/true, fft_scratch);
                }

                // ---- 7) parity factor + atomicAdd into template_fill ----
                // atomicAdd required: different binaries with overlapping
                // (m, n_glob) pixels write to the same global cell.
                const double kappa = 2.0 * sqrt(M_PI * dt) / (double) Nf;
                for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                     n_loc += BLOCK_INCR_X) {
                    const int  n_glob       = n_global_lo + (n_loc - keep_lo);
                    const bool parity_even  = (((m + n_loc) & 1) == 0);
                    const double sign       = ((((m + 1) * n_loc) & 1) == 0)
                                                ? 1.0 : -1.0;
                    for (int c = 0; c < nchannels; ++c) {
                        const cmplx z      = layer_buf[c * Nt_sub + n_loc];
                        const double real_part = parity_even ? z.real() : z.imag();
                        const double w     = factor * kappa * sign * real_part;
                        const size_t dst   = ((size_t) c * Nf + m) * Nt + n_glob;
#ifdef __CUDACC__
                        atomicAdd(&template_fill[dst], w);
#else
                        template_fill[dst] += w;
#endif
                    }
                }
                CUDA_SYNC_THREADS;
            } // end m_layer
        } // end chunk j
    } // end bin_i
}


template <class SourceT>
CUDA_KERNEL
void wdm_het_swap_ll_kernel(
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
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int    m_band_half_width)
{
    // Same per-(chunk, m_layer) flow as get_ll, but with TWO template builds
    // (add + rem) and 5 inner-product partials (<d|h_add>, <d|h_rem>,
    // <h_add|h_add>, <h_rem|h_rem>, <h_add|h_rem>).
    //
    // Shared memory budget (single buffer reuse, as in get_ll):
    //   tdi_channel_buf[3 * Nt_sub] cmplx   -- FFT/iFFT scratch (12 KB)
    //   t_arr_buf      [N_sparse]   double  -- sparse time grid  (2 KB)
    //   partial_dh_a/r, partial_aa, partial_rr, partial_ar
    //                  [blockDim.x] double  -- reduction (5 x 0.5 KB)
    // ~17 KB total at Nt_sub=256 / blockDim.x=64.
    //
    // Per-thread register storage for the "w_add held across rem build":
    //   w_add_reg[nchannels * K_PER_THREAD] doubles, where
    //   K_PER_THREAD = ceil(Nt_sub / blockDim.x) = 4 at GPU defaults.
    // 12 doubles = 96 bytes per thread, fits comfortably in registers.
    SourceT src(orbits, tdi_config, T, t_ref);

    const int Nf         = wdm_settings->Nf;
    const int Nt         = wdm_settings->Nt;  (void) Nt;
    const int ind_min_f  = wdm_settings->ind_min_f;
    const int ind_min_t  = wdm_settings->ind_min_t;
    const int Nf_active  = wdm_settings->Nf_active;
    const int Nt_active  = wdm_settings->Nt_active;

    // Dynamic shared-memory layout (must match shared_bytes at launch):
    //   fd_chunk_buf_a [nchannels * N_sparse] cmplx  -- add chunk-FD (built ONCE per chunk)
    //   fd_chunk_buf_r [nchannels * N_sparse] cmplx  -- rem chunk-FD (built ONCE per chunk)
    //   layer_buf      [nchannels * Nt_sub]   cmplx  -- per-m scratch (reused for add + rem)
    //   partial_dh_a / partial_dh_r / partial_aa / partial_rr / partial_ar
    //     each [blockDim.x] double
    // ~38.5 KB at Nt_sub=N_sparse=256, blockDim=64 -- 4 blocks/SM on A100.
    // The add and rem chunk-FDs are computed ONCE per chunk and reused
    // for every m_layer in the (union of) bands.
#ifdef __CUDACC__
    extern CUDA_SHARED char shared_mem[];
    cmplx  *fd_chunk_buf_a  = (cmplx *) shared_mem;
    cmplx  *fd_chunk_buf_r  = &fd_chunk_buf_a[(size_t) nchannels * N_sparse];
    cmplx  *layer_buf       = &fd_chunk_buf_r[(size_t) nchannels * N_sparse];
    double *partial_dh_a    = (double *) &layer_buf[(size_t) nchannels * Nt_sub];
    double *partial_dh_r    = &partial_dh_a[NUM_THREADS_HERE];
    double *partial_aa      = &partial_dh_r[NUM_THREADS_HERE];
    double *partial_rr      = &partial_aa  [NUM_THREADS_HERE];
    double *partial_ar      = &partial_rr  [NUM_THREADS_HERE];
    // cufftdx scratch (unused unless LISA_USE_CUFFTDX is defined).
    char   *fft_scratch     = (char *) &partial_ar[NUM_THREADS_HERE];
#else
    cmplx  fd_chunk_buf_a_cpu[FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    cmplx  fd_chunk_buf_r_cpu[FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    cmplx  layer_buf_cpu     [FAST_WDM_NCHANNELS_MAX * FAST_WDM_NT_SUB_MAX];
    double partial_dh_a_cpu[1], partial_dh_r_cpu[1];
    double partial_aa_cpu  [1], partial_rr_cpu  [1], partial_ar_cpu[1];
    cmplx  *fd_chunk_buf_a  = fd_chunk_buf_a_cpu;
    cmplx  *fd_chunk_buf_r  = fd_chunk_buf_r_cpu;
    cmplx  *layer_buf       = layer_buf_cpu;
    double *partial_dh_a    = partial_dh_a_cpu;
    double *partial_dh_r    = partial_dh_r_cpu;
    double *partial_aa      = partial_aa_cpu;
    double *partial_rr      = partial_rr_cpu;
    double *partial_ar      = partial_ar_cpu;
    char   *fft_scratch     = nullptr;
#endif

    const double layer_df = 1.0 / (2.0 * (double) Nf * dt);
    const double df_chunk = 1.0 / T_chunk;

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X) {
        double *p_add = &params_add_all   [(size_t) bin_i * nparams];
        double *p_rem = &params_remove_all[(size_t) bin_i * nparams];
        (void) data_index_all; (void) noise_index_all;

        // Per-thread accumulators.
        double tmp_dh_a = 0.0, tmp_dh_r = 0.0;
        double tmp_aa   = 0.0, tmp_rr   = 0.0, tmp_ar = 0.0;

        // Carrier bins for add and rem (each binary has its own).
        const double f0_a    = p_add[src.f0_index];
        const double f0_r    = p_rem[src.f0_index];
        const int    k_f0_a  = (int) round(f0_a / df_chunk);
        const int    k_f0_r  = (int) round(f0_r / df_chunk);
        const double f0g_a   = (double) k_f0_a * df_chunk;
        const double f0g_r   = (double) k_f0_r * df_chunk;
        // m-band -- take the union of add's and rem's narrow bands so we
        // build both templates over the same m_layer iteration.
        const int    m_floor_a = (int) (f0_a / layer_df);
        const int    m_floor_r = (int) (f0_r / layer_df);
        int          m_lo      = (m_floor_a < m_floor_r ? m_floor_a : m_floor_r) - m_band_half_width;
        int          m_hi      = (m_floor_a > m_floor_r ? m_floor_a : m_floor_r) + m_band_half_width + 1;
        if (m_lo < ind_min_f)             m_lo = ind_min_f;
        if (m_hi > ind_min_f + Nf_active) m_hi = ind_min_f + Nf_active;

        for (int j = 0; j < n_chunks; ++j) {
            const int    keep_lo     = chunk_keep_lo[j];
            const int    keep_hi     = chunk_keep_hi[j];
            const int    n_global_lo = chunk_n_global_offset[j];
            const double chunk_t0    = chunk_t_starts[j];
            const double dt_sparse   = T_chunk / (double) N_sparse;

            // ============================================================
            // CHUNK-LEVEL: build the TWO chunk-FD buffers (add + rem) ONCE
            // per chunk. None of steps 1-4 (TD-build, heterodyne, Tukey,
            // forward FFT) depend on m, so the m loop below only does the
            // per-m work (window+rearrange + iFFT + parity + accumulate).
            // ============================================================
            const double n_taper = (tukey_alpha > 0.0)
                ? 0.5 * tukey_alpha * (double) (N_sparse - 1) : 0.0;

            // ---- Build ADD chunk-FD into fd_chunk_buf_a ----
            {
                CUDA_SHARED int link_sc_rec[NLINKS];
                CUDA_SHARED int link_sc_em [NLINKS];
                src.fill_link_arrays(link_sc_rec, link_sc_em);
                CUDA_SYNC_THREADS;
                Vec k_sky(0.0, 0.0, 0.0);
                Vec u_sky(0.0, 0.0, 0.0);
                Vec v_sky(0.0, 0.0, 0.0);
                src.get_sky_vectors(&k_sky, &u_sky, &v_sky, p_add);
                for (int i = THREAD_START_X; i < N_sparse;
                     i += BLOCK_INCR_X) {
                    const double t = chunk_t0 + (double) i * dt_sparse;
                    cmplx tdi_tmp[3];
                    src.get_tdi_Xf_single(&tdi_tmp[0], t, p_add,
                                          k_sky, u_sky, v_sky,
                                          link_sc_rec, link_sc_em, bin_i);
                    for (int c = 0; c < nchannels; ++c)
                        fd_chunk_buf_a[c * N_sparse + i] = tdi_tmp[c];
                }
                CUDA_SYNC_THREADS;
            }
            for (int idx = THREAD_START_X; idx < nchannels * N_sparse;
                 idx += BLOCK_INCR_X) {
                const int i = idx - (idx / N_sparse) * N_sparse;
                const double tau = (double) i * dt_sparse;
                const double het_phase = -2.0 * M_PI * f0g_a * tau;
                const cmplx  het_factor(cos(het_phase), sin(het_phase));
                cmplx s = gcmplx::conj(fd_chunk_buf_a[idx]) * het_factor;
                if (n_taper > 0.0) {
                    double w = 1.0;
                    const double di = (double) i;
                    const double dlast = (double) (N_sparse - 1);
                    if (di < n_taper) {
                        const double xn = di / n_taper;
                        w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                    } else if (di > dlast - n_taper) {
                        const double xn = (dlast - di) / n_taper;
                        w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                    }
                    s = cmplx(s.real() * w, s.imag() * w);
                }
                fd_chunk_buf_a[idx] = s;
            }
            CUDA_SYNC_THREADS;
            // Per-channel forward FFT for ADD (helper has trailing sync).
            for (int c = 0; c < nchannels; ++c) {
                wdm_fft_dispatch(&fd_chunk_buf_a[c * N_sparse],
                                  N_sparse, log2_N_sparse,
                                  /*inverse=*/false, fft_scratch);
            }

            // ---- Build REM chunk-FD into fd_chunk_buf_r ----
            {
                CUDA_SHARED int link_sc_rec[NLINKS];
                CUDA_SHARED int link_sc_em [NLINKS];
                src.fill_link_arrays(link_sc_rec, link_sc_em);
                CUDA_SYNC_THREADS;
                Vec k_sky(0.0, 0.0, 0.0);
                Vec u_sky(0.0, 0.0, 0.0);
                Vec v_sky(0.0, 0.0, 0.0);
                src.get_sky_vectors(&k_sky, &u_sky, &v_sky, p_rem);
                for (int i = THREAD_START_X; i < N_sparse;
                     i += BLOCK_INCR_X) {
                    const double t = chunk_t0 + (double) i * dt_sparse;
                    cmplx tdi_tmp[3];
                    src.get_tdi_Xf_single(&tdi_tmp[0], t, p_rem,
                                          k_sky, u_sky, v_sky,
                                          link_sc_rec, link_sc_em, bin_i);
                    for (int c = 0; c < nchannels; ++c)
                        fd_chunk_buf_r[c * N_sparse + i] = tdi_tmp[c];
                }
                CUDA_SYNC_THREADS;
            }
            for (int idx = THREAD_START_X; idx < nchannels * N_sparse;
                 idx += BLOCK_INCR_X) {
                const int i = idx - (idx / N_sparse) * N_sparse;
                const double tau = (double) i * dt_sparse;
                const double het_phase = -2.0 * M_PI * f0g_r * tau;
                const cmplx  het_factor(cos(het_phase), sin(het_phase));
                cmplx s = gcmplx::conj(fd_chunk_buf_r[idx]) * het_factor;
                if (n_taper > 0.0) {
                    double w = 1.0;
                    const double di = (double) i;
                    const double dlast = (double) (N_sparse - 1);
                    if (di < n_taper) {
                        const double xn = di / n_taper;
                        w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                    } else if (di > dlast - n_taper) {
                        const double xn = (dlast - di) / n_taper;
                        w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                    }
                    s = cmplx(s.real() * w, s.imag() * w);
                }
                fd_chunk_buf_r[idx] = s;
            }
            CUDA_SYNC_THREADS;
            // Per-channel forward FFT for REM (helper has trailing sync).
            for (int c = 0; c < nchannels; ++c) {
                wdm_fft_dispatch(&fd_chunk_buf_r[c * N_sparse],
                                  N_sparse, log2_N_sparse,
                                  /*inverse=*/false, fft_scratch);
            }

            // ============================================================
            // PER-M-LAYER: for each m, read fd_chunk_buf_a -> layer_buf,
            // iFFT, parity into w_add_reg. Then read fd_chunk_buf_r ->
            // layer_buf (overwrites), iFFT, parity -> w_r per thread, then
            // accumulate 5 partials using w_add_reg + w_r.
            // ============================================================
            const double scale_fd     = 0.5 * dt_sparse / dt;
            const int    half_Nt_sub  = Nt_sub / 2;
            const int    half_Nsp     = N_sparse / 2;
            const double kappa        = 2.0 * sqrt(M_PI * dt) / (double) Nf;
            constexpr int K_MAX_REG   = FAST_WDM_K_PER_THREAD_MAX;
            for (int m = m_lo; m < m_hi; ++m) {
                const int m_act        = m - ind_min_f;
                const int fft_offset_a = m * half_Nt_sub - half_Nt_sub - k_f0_a;
                const int fft_offset_r = m * half_Nt_sub - half_Nt_sub - k_f0_r;

                // ---- PHASE 1: ADD layer (fd_chunk_buf_a -> layer_buf -> w_add_reg) ----
                for (int c = 0; c < nchannels; ++c) {
                    for (int k_idx = THREAD_START_X; k_idx < Nt_sub;
                         k_idx += BLOCK_INCR_X) {
                        int fft_bin = fft_offset_a + k_idx;
                        cmplx v(0.0, 0.0);
                        if (fft_bin >= -half_Nsp && fft_bin < half_Nsp) {
                            int read_bin = (fft_bin + N_sparse) % N_sparse;
                            v = fd_chunk_buf_a[c * N_sparse + read_bin];
                            v = cmplx(v.real() * scale_fd, v.imag() * scale_fd);
                            const double w = wdm_window[k_idx];
                            v = cmplx(v.real() * w, v.imag() * w);
                        }
                        layer_buf[c * Nt_sub + k_idx] = v;
                    }
                }
                CUDA_SYNC_THREADS;
                // Per-channel iFFT (FFT helper has its own trailing sync).
                for (int c = 0; c < nchannels; ++c) {
                    wdm_fft_dispatch(&layer_buf[c * Nt_sub],
                                      Nt_sub, log2_Nt_sub,
                                      /*inverse=*/true, fft_scratch);
                }
                double w_add_reg[FAST_WDM_NCHANNELS_MAX * K_MAX_REG];
                {
                    int k_idx_reg = 0;
                    for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                         n_loc += BLOCK_INCR_X) {
                        const bool parity_even = (((m + n_loc) & 1) == 0);
                        const double sign      = ((((m + 1) * n_loc) & 1) == 0)
                                                  ? 1.0 : -1.0;
                        for (int c = 0; c < nchannels; ++c) {
                            const cmplx z = layer_buf[c * Nt_sub + n_loc];
                            const double rp = parity_even ? z.real() : z.imag();
                            w_add_reg[c * K_MAX_REG + k_idx_reg] = kappa * sign * rp;
                        }
                        ++k_idx_reg;
                    }
                }
                CUDA_SYNC_THREADS;

                // ---- PHASE 2: REM layer (fd_chunk_buf_r -> layer_buf), accumulate ----
                for (int c = 0; c < nchannels; ++c) {
                    for (int k_idx = THREAD_START_X; k_idx < Nt_sub;
                         k_idx += BLOCK_INCR_X) {
                        int fft_bin = fft_offset_r + k_idx;
                        cmplx v(0.0, 0.0);
                        if (fft_bin >= -half_Nsp && fft_bin < half_Nsp) {
                            int read_bin = (fft_bin + N_sparse) % N_sparse;
                            v = fd_chunk_buf_r[c * N_sparse + read_bin];
                            v = cmplx(v.real() * scale_fd, v.imag() * scale_fd);
                            const double w = wdm_window[k_idx];
                            v = cmplx(v.real() * w, v.imag() * w);
                        }
                        layer_buf[c * Nt_sub + k_idx] = v;
                    }
                }
                CUDA_SYNC_THREADS;
                // Per-channel iFFT (FFT helper has its own trailing sync).
                for (int c = 0; c < nchannels; ++c) {
                    wdm_fft_dispatch(&layer_buf[c * Nt_sub],
                                      Nt_sub, log2_Nt_sub,
                                      /*inverse=*/true, fft_scratch);
                }

                // ---- Accumulate 5 partials using w_add_reg + freshly-parity'd w_r ----
                const int ind_max_t_excl = ind_min_t + Nt_active;
                {
                    int k_idx_reg = 0;
                    for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                         n_loc += BLOCK_INCR_X) {
                        const int n_glob = n_global_lo + (n_loc - keep_lo);
                        if (n_glob >= ind_min_t && n_glob < ind_max_t_excl) {
                            const int n_act = n_glob - ind_min_t;
                            const bool parity_even = (((m + n_loc) & 1) == 0);
                            const double sign      = ((((m + 1) * n_loc) & 1) == 0)
                                                      ? 1.0 : -1.0;
                            double w_a_arr[FAST_WDM_NCHANNELS_MAX];
                            double w_r_arr[FAST_WDM_NCHANNELS_MAX];
                            double d_arr  [FAST_WDM_NCHANNELS_MAX];
                            for (int c = 0; c < nchannels; ++c) {
                                w_a_arr[c] = w_add_reg[c * K_MAX_REG + k_idx_reg];
                                const cmplx z = layer_buf[c * Nt_sub + n_loc];
                                const double rp = parity_even ? z.real() : z.imag();
                                w_r_arr[c] = kappa * sign * rp;
                                const size_t g_d = ((size_t) c * Nf_active + m_act)
                                                    * Nt_active + n_act;
                                d_arr[c] = data_d[g_d];
                            }
                            if (tdi_type == TDI_XYZ) {
                                // Symmetric invC: 3 diag + 3 off-diag reads.
                                // tmp_ar is non-symmetric in (a, r) so we
                                // sum both (c1,c2) and (c2,c1) contributions.
                                for (int c = 0; c < nchannels; ++c) {
                                    const size_t g_inv =
                                        (((size_t) c * nchannels + c)
                                          * Nf_active + m_act)
                                          * Nt_active + n_act;
                                    const double inv = invC[g_inv];
                                    tmp_dh_a += d_arr[c]   * w_a_arr[c] * inv;
                                    tmp_dh_r += d_arr[c]   * w_r_arr[c] * inv;
                                    tmp_aa   += w_a_arr[c] * w_a_arr[c] * inv;
                                    tmp_rr   += w_r_arr[c] * w_r_arr[c] * inv;
                                    tmp_ar   += w_a_arr[c] * w_r_arr[c] * inv;
                                }
                                for (int c1 = 0; c1 < nchannels - 1; ++c1) {
                                    for (int c2 = c1 + 1; c2 < nchannels; ++c2) {
                                        const size_t g_inv =
                                            (((size_t) c1 * nchannels + c2)
                                              * Nf_active + m_act)
                                              * Nt_active + n_act;
                                        const double inv = invC[g_inv];
                                        tmp_dh_a += (d_arr[c1]   * w_a_arr[c2]
                                                      + d_arr[c2]   * w_a_arr[c1]) * inv;
                                        tmp_dh_r += (d_arr[c1]   * w_r_arr[c2]
                                                      + d_arr[c2]   * w_r_arr[c1]) * inv;
                                        tmp_aa   += 2.0 * w_a_arr[c1] * w_a_arr[c2] * inv;
                                        tmp_rr   += 2.0 * w_r_arr[c1] * w_r_arr[c2] * inv;
                                        tmp_ar   += (w_a_arr[c1] * w_r_arr[c2]
                                                      + w_a_arr[c2] * w_r_arr[c1]) * inv;
                                    }
                                }
                            } else {
                                for (int c = 0; c < nchannels; ++c) {
                                    const size_t g_inv = ((size_t) c * Nf_active + m_act)
                                                          * Nt_active + n_act;
                                    const double inv = invC[g_inv];
                                    tmp_dh_a += d_arr[c]   * w_a_arr[c] * inv;
                                    tmp_dh_r += d_arr[c]   * w_r_arr[c] * inv;
                                    tmp_aa   += w_a_arr[c] * w_a_arr[c] * inv;
                                    tmp_rr   += w_r_arr[c] * w_r_arr[c] * inv;
                                    tmp_ar   += w_a_arr[c] * w_r_arr[c] * inv;
                                }
                            }
                        }
                        ++k_idx_reg;
                    }
                }
                CUDA_SYNC_THREADS;
            } // end m_layer
        } // end chunk j

        // ---- per-thread -> shared partials -> block tree reduction ----
        partial_dh_a[THREAD_START_X] = tmp_dh_a;
        partial_dh_r[THREAD_START_X] = tmp_dh_r;
        partial_aa  [THREAD_START_X] = tmp_aa;
        partial_rr  [THREAD_START_X] = tmp_rr;
        partial_ar  [THREAD_START_X] = tmp_ar;
        CUDA_SYNC_THREADS;
#ifdef __CUDACC__
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (THREAD_START_X < stride) {
                partial_dh_a[THREAD_START_X] += partial_dh_a[THREAD_START_X + stride];
                partial_dh_r[THREAD_START_X] += partial_dh_r[THREAD_START_X + stride];
                partial_aa  [THREAD_START_X] += partial_aa  [THREAD_START_X + stride];
                partial_rr  [THREAD_START_X] += partial_rr  [THREAD_START_X + stride];
                partial_ar  [THREAD_START_X] += partial_ar  [THREAD_START_X + stride];
            }
            CUDA_SYNC_THREADS;
        }
        // See wdm_het_get_ll_kernel: one binary per block + sequential
        // chunks inside the block -> no cross-block race on these per-binary
        // outputs, so a direct store suffices. Switch to atomicAdd if/when
        // chunks move to blockIdx.y.
        if (THREAD_START_X == 0) {
            d_h_add_out      [bin_i] = partial_dh_a[0];
            d_h_remove_out   [bin_i] = partial_dh_r[0];
            add_add_out      [bin_i] = partial_aa  [0];
            remove_remove_out[bin_i] = partial_rr  [0];
            add_remove_out   [bin_i] = partial_ar  [0];
        }
#else
        d_h_add_out      [bin_i] = partial_dh_a[0];
        d_h_remove_out   [bin_i] = partial_dh_r[0];
        add_add_out      [bin_i] = partial_aa  [0];
        remove_remove_out[bin_i] = partial_rr  [0];
        add_remove_out   [bin_i] = partial_ar  [0];
#endif
        CUDA_SYNC_THREADS;
    } // end bin_i
}


template <class SourceT>
CUDA_KERNEL
void wdm_het_get_fstat_ll_kernel(
    double *N_arr_re_out, double *N_arr_im_out,   // (num_bin, 4) per-binary <d|A_i>
    double *M_mat_re_out, double *M_mat_im_out,   // (num_bin, 10) per-binary <A_i|A_j>
                                                   // (Hermitian; 4 diag + 6 upper)
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
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int    m_band_half_width)
{
    // F-stat: build 4 basis waveforms per Cornish & Crowder '05 with fixed
    //   (A, iota, psi, phi0) = (2, pi/2, {0, pi/4, 0, pi/4}, {0, pi, 3pi/2, pi/2})
    // For each (chunk, m_layer, filter_i) we build w_i, stage in per-thread
    // registers, then once all 4 are staged we accumulate:
    //   N_arr[bin_i, i]   = sum_pixels    sum_{c1,c2}  d[c1] * w_i[c2] * invC[c1,c2]
    //   M_mat[bin_i, ij]  = sum_pixels    sum_{c1,c2}  w_i[c1] * w_j[c2] * invC[c1,c2]
    // where ij flattens the upper-triangle (i<=j) of the 4x4 Hermitian.
    //
    // WDM coefficients are real -> N and M are real (imag outputs always 0).
    //
    // GB param convention (matches existing chunked-het):
    //   params[0] = A      params[1] = f0   params[2] = fdot  params[3] = fddot
    //   params[4] = phi0   params[5] = iota params[6] = psi
    //   params[7] = lam    params[8] = beta
    SourceT src(orbits, tdi_config, T, t_ref);

    const int Nf         = wdm_settings->Nf;
    const int Nt         = wdm_settings->Nt;  (void) Nt;
    const int ind_min_f  = wdm_settings->ind_min_f;
    const int ind_min_t  = wdm_settings->ind_min_t;
    const int Nf_active  = wdm_settings->Nf_active;
    const int Nt_active  = wdm_settings->Nt_active;

    // F-stat basis filter parameters (Cornish & Crowder '05).
    constexpr int   N_FILTERS  = 4;
    const double A_arr    [N_FILTERS] = {2.0, 2.0, 2.0, 2.0};
    const double iota_arr [N_FILTERS] = {M_PI / 2.0, M_PI / 2.0,
                                          M_PI / 2.0, M_PI / 2.0};
    const double psi_arr  [N_FILTERS] = {0.0, M_PI / 4.0, 0.0, M_PI / 4.0};
    const double phi0_arr [N_FILTERS] = {0.0, M_PI, 3.0 * M_PI / 2.0, M_PI / 2.0};

    // GB param indices (constants for the GB convention; SOBBH would need
    // a trait-based specialization).
    constexpr int IDX_A    = 0;
    constexpr int IDX_PHI0 = 4;
    constexpr int IDX_IOTA = 5;
    constexpr int IDX_PSI  = 6;

    // Dynamic shared-memory layout (must match shared_bytes at launch):
    //   fd_chunk_buf[fi=0..3][nchannels * N_sparse] cmplx
    //                                              -- per-filter chunk-FD,
    //                                                 built ONCE per chunk
    //   layer_buf   [nchannels * Nt_sub]           cmplx -- per-(m, fi) scratch
    //   partial_N   [N_FILTERS  * blockDim.x]      double  ( 4 * NTH)
    //   partial_M   [N_M_PART   * blockDim.x]      double  (10 * NTH)
    // ~67 KB at Nt_sub=N_sparse=256, blockDim=64. Exceeds the 48 KB default
    // limit -- the launcher calls cudaFuncSetAttribute(...,
    // cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes) to raise it.
#ifdef __CUDACC__
    extern CUDA_SHARED char shared_mem[];
    cmplx  *fd_chunk_buf[N_FILTERS];
    fd_chunk_buf[0] = (cmplx *) shared_mem;
    for (int fi_p = 1; fi_p < N_FILTERS; ++fi_p)
        fd_chunk_buf[fi_p] = &fd_chunk_buf[fi_p - 1][(size_t) nchannels * N_sparse];
    cmplx  *layer_buf = &fd_chunk_buf[N_FILTERS - 1][(size_t) nchannels * N_sparse];
    // 4 N + 10 M = 14 partial buffers, each blockDim.x wide.
    double *partial_N = (double *) &layer_buf[(size_t) nchannels * Nt_sub];
    double *partial_M = &partial_N[(size_t) N_FILTERS * NUM_THREADS_HERE];
    // cufftdx scratch (unused unless LISA_USE_CUFFTDX is defined).
    char   *fft_scratch = (char *) &partial_M[(size_t) ((N_FILTERS * (N_FILTERS + 1)) / 2)
                                              * NUM_THREADS_HERE];
#else
    cmplx  fd_chunk_buf_cpu[N_FILTERS][FAST_WDM_NCHANNELS_MAX * FAST_WDM_N_SPARSE_MAX];
    cmplx  layer_buf_cpu   [FAST_WDM_NCHANNELS_MAX * FAST_WDM_NT_SUB_MAX];
    double partial_N_cpu   [N_FILTERS];
    double partial_M_cpu   [(N_FILTERS * (N_FILTERS + 1)) / 2];
    cmplx  *fd_chunk_buf[N_FILTERS];
    for (int fi_p = 0; fi_p < N_FILTERS; ++fi_p) fd_chunk_buf[fi_p] = fd_chunk_buf_cpu[fi_p];
    cmplx  *layer_buf       = layer_buf_cpu;
    double *partial_N       = partial_N_cpu;
    double *partial_M       = partial_M_cpu;
    char   *fft_scratch     = nullptr;
#endif

    constexpr int N_M_PARTIALS = (N_FILTERS * (N_FILTERS + 1)) / 2;  // = 10

    // (i, j) -> flat upper-triangle index for the 4x4 Hermitian M.
    // ij = i * N_FILTERS - (i*(i+1))/2 + j   for i <= j
    auto m_idx = [] (int i, int j) -> int {
        return i * N_FILTERS - (i * (i + 1)) / 2 + j;
    };

    const double layer_df = 1.0 / (2.0 * (double) Nf * dt);
    const double df_chunk = 1.0 / T_chunk;
    // Narrow band width is configurable via the ``m_band_half_width`` arg
    // (default 1 -> 3 layers, set in the impl wrapper). All 4 het kernels
    // use the same convention.
    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X) {
        double *params      = &params_all[(size_t) bin_i * nparams];
        const int data_ind  = data_index_all [bin_i];  (void) data_ind;
        const int noise_ind = noise_index_all[bin_i];  (void) noise_ind;

        // Per-thread accumulators.
        double tmp_N[N_FILTERS] = {0.0, 0.0, 0.0, 0.0};
        double tmp_M[N_M_PARTIALS];
        for (int k = 0; k < N_M_PARTIALS; ++k) tmp_M[k] = 0.0;

        const double f0      = params[src.f0_index];  // = params[IDX_F0] for GB
        const int    k_f0    = (int) round(f0 / df_chunk);
        const double f0_grid = (double) k_f0 * df_chunk;
        const int    m_floor = (int) (f0 / layer_df);
        int          m_lo    = m_floor - m_band_half_width;
        int          m_hi    = m_floor + m_band_half_width + 1;     // exclusive
        if (m_lo < ind_min_f)             m_lo = ind_min_f;
        if (m_hi > ind_min_f + Nf_active) m_hi = ind_min_f + Nf_active;

        for (int j = 0; j < n_chunks; ++j) {
            const int    keep_lo     = chunk_keep_lo[j];
            const int    keep_hi     = chunk_keep_hi[j];
            const int    n_global_lo = chunk_n_global_offset[j];
            const double chunk_t0    = chunk_t_starts[j];
            const double dt_sparse   = T_chunk / (double) N_sparse;

            // ============================================================
            // CHUNK-LEVEL: build all 4 chunk-FD buffers (one per basis
            // filter). Steps 1-3 (TD-build, heterodyne, Tukey, FFT) do
            // NOT depend on m, so they happen ONCE per chunk per filter
            // -- m loop below only does the per-m work (window+rearrange
            // -> iFFT -> parity -> accumulate).
            // ============================================================
            const double n_taper = (tukey_alpha > 0.0)
                ? 0.5 * tukey_alpha * (double) (N_sparse - 1) : 0.0;
            for (int fi_b = 0; fi_b < N_FILTERS; ++fi_b) {
                double params_basis[16];   // GB has 9; bound generously
                for (int k = 0; k < nparams && k < 16; ++k) {
                    params_basis[k] = params[k];
                }
                params_basis[IDX_A   ] = A_arr   [fi_b];
                params_basis[IDX_IOTA] = iota_arr[fi_b];
                params_basis[IDX_PSI ] = psi_arr [fi_b];
                params_basis[IDX_PHI0] = phi0_arr[fi_b];

                // ---- 1) TD-build into fd_chunk_buf[fi_b] ----
                {
                    CUDA_SHARED int link_sc_rec[NLINKS];
                    CUDA_SHARED int link_sc_em [NLINKS];
                    src.fill_link_arrays(link_sc_rec, link_sc_em);
                    CUDA_SYNC_THREADS;
                    Vec k_sky(0.0, 0.0, 0.0);
                    Vec u_sky(0.0, 0.0, 0.0);
                    Vec v_sky(0.0, 0.0, 0.0);
                    src.get_sky_vectors(&k_sky, &u_sky, &v_sky, params_basis);
                    for (int i = THREAD_START_X; i < N_sparse;
                         i += BLOCK_INCR_X) {
                        const double t = chunk_t0 + (double) i * dt_sparse;
                        cmplx tdi_tmp[3];
                        src.get_tdi_Xf_single(&tdi_tmp[0], t, params_basis,
                                              k_sky, u_sky, v_sky,
                                              link_sc_rec, link_sc_em, bin_i);
                        for (int c = 0; c < nchannels; ++c)
                            fd_chunk_buf[fi_b][c * N_sparse + i] = tdi_tmp[c];
                    }
                    CUDA_SYNC_THREADS;
                }

                // ---- 2) heterodyne + Tukey (in place in fd_chunk_buf[fi_b]) ----
                for (int idx = THREAD_START_X; idx < nchannels * N_sparse;
                     idx += BLOCK_INCR_X) {
                    const int i = idx - (idx / N_sparse) * N_sparse;
                    const double tau = (double) i * dt_sparse;
                    const double het_phase = -2.0 * M_PI * f0_grid * tau;
                    const cmplx  het_factor(cos(het_phase), sin(het_phase));
                    cmplx s = gcmplx::conj(fd_chunk_buf[fi_b][idx]) * het_factor;
                    if (n_taper > 0.0) {
                        double w = 1.0;
                        const double di = (double) i;
                        const double dlast = (double) (N_sparse - 1);
                        if (di < n_taper) {
                            const double xn = di / n_taper;
                            w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                        } else if (di > dlast - n_taper) {
                            const double xn = (dlast - di) / n_taper;
                            w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
                        }
                        s = cmplx(s.real() * w, s.imag() * w);
                    }
                    fd_chunk_buf[fi_b][idx] = s;
                }
                CUDA_SYNC_THREADS;

                // ---- 3) FFT length N_sparse per channel (helper has trailing sync) ----
                for (int c = 0; c < nchannels; ++c) {
                    wdm_fft_dispatch(&fd_chunk_buf[fi_b][c * N_sparse],
                                      N_sparse, log2_N_sparse,
                                      /*inverse=*/false, fft_scratch);
                }
            } // end build-FD per filter
            // All 4 chunk-FDs now resident in shared mem; reuse across m below.

            // ============================================================
            // PER-M-LAYER: for each m, build all 4 basis layer values and
            // accumulate 4 N + 10 M partials.
            // ============================================================
            const double scale_fd    = 0.5 * dt_sparse / dt;
            const int    half_Nt_sub = Nt_sub / 2;
            const int    half_Nsp    = N_sparse / 2;
            const double kappa       = 2.0 * sqrt(M_PI * dt) / (double) Nf;
            constexpr int K_MAX_REG  = FAST_WDM_K_PER_THREAD_MAX;
            for (int m = m_lo; m < m_hi; ++m) {
                const int m_act = m - ind_min_f;
                // Per-thread storage for THIS m's 4 basis WDM coefs.
                double w_basis_reg[N_FILTERS * FAST_WDM_NCHANNELS_MAX * K_MAX_REG];
                const int fft_offset = m * half_Nt_sub - half_Nt_sub - k_f0;

                // Build each of the 4 basis waveforms' layer at this m.
                for (int fi = 0; fi < N_FILTERS; ++fi) {
                    // ---- 4) window + rearrange: fd_chunk_buf[fi] -> layer_buf ----
                    for (int c = 0; c < nchannels; ++c) {
                        for (int k_idx = THREAD_START_X; k_idx < Nt_sub;
                             k_idx += BLOCK_INCR_X) {
                            int fft_bin = fft_offset + k_idx;
                            cmplx v(0.0, 0.0);
                            if (fft_bin >= -half_Nsp && fft_bin < half_Nsp) {
                                int read_bin = (fft_bin + N_sparse) % N_sparse;
                                v = fd_chunk_buf[fi][c * N_sparse + read_bin];
                                v = cmplx(v.real() * scale_fd, v.imag() * scale_fd);
                                const double w = wdm_window[k_idx];
                                v = cmplx(v.real() * w, v.imag() * w);
                            }
                            layer_buf[c * Nt_sub + k_idx] = v;
                        }
                    }
                    CUDA_SYNC_THREADS;

                    // ---- 5) iFFT length Nt_sub per channel (helper has trailing sync) ----
                    for (int c = 0; c < nchannels; ++c) {
                        wdm_fft_dispatch(&layer_buf[c * Nt_sub],
                                          Nt_sub, log2_Nt_sub,
                                          /*inverse=*/true, fft_scratch);
                    }

                    // ---- 6) parity factor; stage w_i[c, n_loc] in regs ----
                    int k_idx_reg = 0;
                    for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                         n_loc += BLOCK_INCR_X) {
                        const bool parity_even = (((m + n_loc) & 1) == 0);
                        const double sign      = ((((m + 1) * n_loc) & 1) == 0)
                                                  ? 1.0 : -1.0;
                        for (int c = 0; c < nchannels; ++c) {
                            const cmplx z  = layer_buf[c * Nt_sub + n_loc];
                            const double rp = parity_even ? z.real() : z.imag();
                            w_basis_reg[(fi * nchannels + c) * K_MAX_REG
                                          + k_idx_reg] = kappa * sign * rp;
                        }
                        ++k_idx_reg;
                    }
                    CUDA_SYNC_THREADS;
                } // end filter fi

                // ---- Accumulate 4 N partials + 10 M partials per pixel ----
                const int ind_max_t_excl = ind_min_t + Nt_active;
                {
                    int k_idx_reg = 0;
                    for (int n_loc = keep_lo + THREAD_START_X; n_loc < keep_hi;
                         n_loc += BLOCK_INCR_X) {
                        const int n_glob = n_global_lo + (n_loc - keep_lo);
                        if (n_glob >= ind_min_t && n_glob < ind_max_t_excl) {
                            const int n_act = n_glob - ind_min_t;
                            // Read data for all channels once.
                            double d_arr[FAST_WDM_NCHANNELS_MAX];
                            for (int c = 0; c < nchannels; ++c) {
                                const size_t g_d = ((size_t) c * Nf_active + m_act)
                                                    * Nt_active + n_act;
                                d_arr[c] = data_d[g_d];
                            }
                            // 4 N partials: <d | A_i>
                            for (int fi = 0; fi < N_FILTERS; ++fi) {
                                double sum_dh = 0.0;
                                if (tdi_type == TDI_XYZ) {
                                    // Symmetric invC: 3 diag + 3 off-diag reads.
                                    for (int c = 0; c < nchannels; ++c) {
                                        const size_t g_inv =
                                            (((size_t) c * nchannels + c)
                                              * Nf_active + m_act)
                                              * Nt_active + n_act;
                                        const double inv = invC[g_inv];
                                        const double w_i =
                                            w_basis_reg[(fi * nchannels + c)
                                                          * K_MAX_REG
                                                          + k_idx_reg];
                                        sum_dh += d_arr[c] * w_i * inv;
                                    }
                                    for (int c1 = 0; c1 < nchannels - 1; ++c1) {
                                        for (int c2 = c1 + 1; c2 < nchannels; ++c2) {
                                            const size_t g_inv =
                                                (((size_t) c1 * nchannels + c2)
                                                  * Nf_active + m_act)
                                                  * Nt_active + n_act;
                                            const double inv = invC[g_inv];
                                            const double w_c1 =
                                                w_basis_reg[(fi * nchannels + c1)
                                                              * K_MAX_REG
                                                              + k_idx_reg];
                                            const double w_c2 =
                                                w_basis_reg[(fi * nchannels + c2)
                                                              * K_MAX_REG
                                                              + k_idx_reg];
                                            sum_dh += (d_arr[c1] * w_c2
                                                        + d_arr[c2] * w_c1) * inv;
                                        }
                                    }
                                } else {
                                    for (int c = 0; c < nchannels; ++c) {
                                        const size_t g_inv = ((size_t) c * Nf_active
                                                                + m_act)
                                                              * Nt_active + n_act;
                                        const double inv = invC[g_inv];
                                        const double w_i =
                                            w_basis_reg[(fi * nchannels + c)
                                                          * K_MAX_REG
                                                          + k_idx_reg];
                                        sum_dh += d_arr[c] * w_i * inv;
                                    }
                                }
                                tmp_N[fi] += sum_dh;
                            }
                            // 10 M partials: <A_i | A_j> for i <= j
                            for (int fi = 0; fi < N_FILTERS; ++fi) {
                                for (int fj = fi; fj < N_FILTERS; ++fj) {
                                    double sum_hh = 0.0;
                                    if (tdi_type == TDI_XYZ) {
                                        // Symmetric invC. Note: w_i (filter fi)
                                        // and w_j (filter fj) are NOT
                                        // interchangeable when fi != fj, so
                                        // off-diag terms sum both orderings.
                                        for (int c = 0; c < nchannels; ++c) {
                                            const size_t g_inv =
                                                (((size_t) c * nchannels + c)
                                                  * Nf_active + m_act)
                                                  * Nt_active + n_act;
                                            const double inv = invC[g_inv];
                                            const double w_i =
                                                w_basis_reg[(fi * nchannels + c)
                                                              * K_MAX_REG
                                                              + k_idx_reg];
                                            const double w_j =
                                                w_basis_reg[(fj * nchannels + c)
                                                              * K_MAX_REG
                                                              + k_idx_reg];
                                            sum_hh += w_i * w_j * inv;
                                        }
                                        for (int c1 = 0; c1 < nchannels - 1; ++c1) {
                                            for (int c2 = c1 + 1; c2 < nchannels; ++c2) {
                                                const size_t g_inv =
                                                    (((size_t) c1 * nchannels + c2)
                                                      * Nf_active + m_act)
                                                      * Nt_active + n_act;
                                                const double inv = invC[g_inv];
                                                const double w_i_c1 =
                                                    w_basis_reg[(fi * nchannels + c1)
                                                                  * K_MAX_REG
                                                                  + k_idx_reg];
                                                const double w_i_c2 =
                                                    w_basis_reg[(fi * nchannels + c2)
                                                                  * K_MAX_REG
                                                                  + k_idx_reg];
                                                const double w_j_c1 =
                                                    w_basis_reg[(fj * nchannels + c1)
                                                                  * K_MAX_REG
                                                                  + k_idx_reg];
                                                const double w_j_c2 =
                                                    w_basis_reg[(fj * nchannels + c2)
                                                                  * K_MAX_REG
                                                                  + k_idx_reg];
                                                sum_hh += (w_i_c1 * w_j_c2
                                                            + w_i_c2 * w_j_c1) * inv;
                                            }
                                        }
                                    } else {
                                        for (int c = 0; c < nchannels; ++c) {
                                            const size_t g_inv = ((size_t) c
                                                                    * Nf_active
                                                                    + m_act)
                                                                  * Nt_active + n_act;
                                            const double inv = invC[g_inv];
                                            const double w_i =
                                                w_basis_reg[(fi * nchannels + c)
                                                              * K_MAX_REG
                                                              + k_idx_reg];
                                            const double w_j =
                                                w_basis_reg[(fj * nchannels + c)
                                                              * K_MAX_REG
                                                              + k_idx_reg];
                                            sum_hh += w_i * w_j * inv;
                                        }
                                    }
                                    tmp_M[m_idx(fi, fj)] += sum_hh;
                                }
                            }
                        }
                        ++k_idx_reg;
                    }
                }
                CUDA_SYNC_THREADS;
            } // end m_layer
        } // end chunk j

        // ---- per-thread -> shared partials -> block tree reduction ----
        for (int fi = 0; fi < N_FILTERS; ++fi) {
            partial_N[fi * NUM_THREADS_HERE + THREAD_START_X] = tmp_N[fi];
        }
        for (int k = 0; k < N_M_PARTIALS; ++k) {
            partial_M[k * NUM_THREADS_HERE + THREAD_START_X] = tmp_M[k];
        }
        CUDA_SYNC_THREADS;
#ifdef __CUDACC__
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (THREAD_START_X < stride) {
                for (int fi = 0; fi < N_FILTERS; ++fi) {
                    partial_N[fi * NUM_THREADS_HERE + THREAD_START_X] +=
                        partial_N[fi * NUM_THREADS_HERE + THREAD_START_X + stride];
                }
                for (int k = 0; k < N_M_PARTIALS; ++k) {
                    partial_M[k * NUM_THREADS_HERE + THREAD_START_X] +=
                        partial_M[k * NUM_THREADS_HERE + THREAD_START_X + stride];
                }
            }
            CUDA_SYNC_THREADS;
        }
        // See wdm_het_get_ll_kernel: one binary per block + sequential
        // chunks inside the block -> no cross-block race on per-binary N
        // and M outputs, so direct stores suffice. Switch to atomicAdd
        // if/when chunks move to blockIdx.y.
        if (THREAD_START_X == 0) {
            for (int fi = 0; fi < N_FILTERS; ++fi) {
                N_arr_re_out[bin_i * N_FILTERS + fi] =
                    partial_N[fi * NUM_THREADS_HERE];
                // WDM coefficients are real -> imag part is exactly 0.
                N_arr_im_out[bin_i * N_FILTERS + fi] = 0.0;
            }
            for (int k = 0; k < N_M_PARTIALS; ++k) {
                M_mat_re_out[bin_i * N_M_PARTIALS + k] =
                    partial_M[k * NUM_THREADS_HERE];
                M_mat_im_out[bin_i * N_M_PARTIALS + k] = 0.0;
            }
        }
#else
        for (int fi = 0; fi < N_FILTERS; ++fi) {
            N_arr_re_out[bin_i * N_FILTERS + fi] = partial_N[fi];
            N_arr_im_out[bin_i * N_FILTERS + fi] = 0.0;
        }
        for (int k = 0; k < N_M_PARTIALS; ++k) {
            M_mat_re_out[bin_i * N_M_PARTIALS + k] = partial_M[k];
            M_mat_im_out[bin_i * N_M_PARTIALS + k] = 0.0;
        }
#endif
        CUDA_SYNC_THREADS;
    } // end bin_i
}


// =============================================================================
// gb_wdm_fill_global_kernel (legacy non-chunked WDM helper -- distinct from
// the chunked-het ``wdm_het_fill_global_kernel`` above). Kept as-is.
// =============================================================================

// Diagnostic: evaluate the per-pixel inputs (|M|, arg(M_mod), f, fdot, phase_ref)
// that gb_wdm_fill_global_kernel feeds into the WDM lookup, without doing the lookup.
// CPU-only — used to compare C-side numerical-derivative inputs against Python splines.


// Spline analog of gb_wdm_eval_inputs_wrap. For each source, builds ONE
// WDM_SPLINE_L-point spline window starting at t_window_start with spacing
// coarse_dt, then evaluates the splines at every tn in tn_arr. Outputs are in
// the SAME convention as gb_wdm_eval_inputs_wrap (|M_mod|, arg(M_mod), f,
// fdot=0, phi_ref), so callers can do a direct subtract against the direct
// path to validate.
//
// Caller must ensure every tn in tn_arr satisfies
//   t_window_start <= tn <= t_window_start + (WDM_SPLINE_L - 1) * coarse_dt
// otherwise eval_wdm_spline_pixel clamps to the nearest segment.
//
// Layouts identical to gb_wdm_eval_inputs_wrap:
//   amp_out, phi_out, f_out, fdot_out: (num_bin, num_t, nchannels)
//   phase_ref_out:                     (num_bin, num_t)
void GBComputationGroup::gb_wdm_spline_eval_inputs_wrap(
    Orbits *orbits, TDIConfig *tdi_config,
    double *params_all, double *tn_arr,
    int num_bin, int nparams, int num_t, int nchannels,
    double T, double t_ref,
    double t_window_start, double coarse_dt,
    double *amp_out, double *phi_out, double *f_out, double *fdot_out,
    double *phase_ref_out)
{
#ifdef __CUDACC__
    // CPU-only diagnostic.
    return;
#else
    GBTDIonTheFly tdi_on_fly_here(orbits, tdi_config, T, t_ref);

    // CPU-side scratch (the GPU kernels will pull these out of shared mem).
    double t_grid_buf[WDM_SPLINE_L];
    double amp_y_buf[3 * WDM_SPLINE_L];
    double dphi_y_buf[3 * WDM_SPLINE_L];
    double phi_ref_y_buf[WDM_SPLINE_L];
    double coefs_buf[21 * WDM_SPLINE_L];
    cmplx  tdi_chan_buf[3 * WDM_SPLINE_L];
    double B_buf[WDM_SPLINE_L];
    const int get_tdi_scratch_len = tdi_on_fly_here.get_tdi_buffer_size(WDM_SPLINE_L);
    char *get_tdi_scratch = new char[get_tdi_scratch_len];

    WDMSplineSet S;
    wdm_spline_set_init(&S, t_grid_buf, amp_y_buf, dphi_y_buf, phi_ref_y_buf, coefs_buf);

    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        double *params = &params_all[bin_i * nparams];

        bool ok = build_wdm_spline_window(
            tdi_on_fly_here, &S, params, bin_i,
            t_window_start, coarse_dt,
            tdi_chan_buf,
            /*pcr_scratch=*/(double*)nullptr, B_buf,
            (void*)get_tdi_scratch, get_tdi_scratch_len);

        if (!ok)
        {
            for (int t_i = 0; t_i < num_t; ++t_i)
            {
                phase_ref_out[bin_i * num_t + t_i] = 0.0;
                for (int chan = 0; chan < nchannels; ++chan)
                {
                    int idx = (bin_i * num_t + t_i) * nchannels + chan;
                    amp_out[idx]  = 0.0;
                    phi_out[idx]  = 0.0;
                    f_out[idx]    = 0.0;
                    fdot_out[idx] = 0.0;
                }
            }
            continue;
        }

        cmplx  tdi_channel_val[3];
        double f[3], fdot[3];
        for (int t_i = 0; t_i < num_t; ++t_i)
        {
            double tn = tn_arr[t_i];
            eval_wdm_spline_pixel(&S, tn, tdi_channel_val, f, fdot);

            // phi_ref reported from the spline -- diverges from the analytic
            // get_phase_ref only by the cubic interpolation error.
            const int L = WDM_SPLINE_L;
            double dx = S.t_grid[1] - S.t_grid[0];
            int idx = (int) floor((tn - S.t_grid[0]) / dx);
            if (idx < 0) idx = 0;
            if (idx > L - 2) idx = L - 2;
            double t0 = S.t_grid[idx];
            CubicSplineSegment seg_phiref(t0,
                S.phi_ref_y[idx], S.phi_ref_c1[idx], S.phi_ref_c2[idx], S.phi_ref_c3[idx],
                CUBIC_SPLINE_LINEAR_SPACING);
            phase_ref_out[bin_i * num_t + t_i] = seg_phiref.eval(tn);

            for (int chan = 0; chan < nchannels; ++chan)
            {
                int idx_out = (bin_i * num_t + t_i) * nchannels + chan;
                amp_out[idx_out]  = gcmplx::abs(tdi_channel_val[chan]);
                phi_out[idx_out]  = gcmplx::arg(tdi_channel_val[chan]);
                f_out[idx_out]    = f[chan];
                fdot_out[idx_out] = fdot[chan];
            }
        }
    }

    delete[] get_tdi_scratch;
#endif
}



// =============================================================================
// Spline-path mirror of gb_wdm_fill_global_kernel. Replaces per-WDM-pixel
// fast_wdm_inner calls with an outer window loop that fits cubic splines to
// the smooth (tdi_amp, tdi_phase, phi_ref) trio produced by
// LISATDIonTheFly::get_tdi on a coarse uniform time grid, and evaluates the
// splines at every WDM pixel inside the window.
//
// Per-source compute scales as O(K * L) get_tdi_Xf_single calls (K windows
// of L coarse points each) versus O(N_pixels * 3) for the direct path. For
// typical T=4yr, L=32, coarse density 256 pts/yr, this is ~30x fewer TDI
// evaluations.
//
// `coarse_dt` is the spacing of the coarse grid in seconds (Python computes
// it from the user-supplied coarse_pts_per_year and pushes it through).
// =============================================================================
#if 0  // === gb_wdm_spline_* kernel + wrap bodies disabled at Phase 3L (2026-06-02) -- WaveletLookupTable retiring ===
template<int num_diff, int total_diff>
CUDA_KERNEL
void gb_wdm_spline_fill_global_kernel(
    double *template_fill, Orbits* orbits, TDIConfig *tdi_config,
    WaveletLookupTable* wdm_lookup, WDMDomain* wdm,
    double *params_all, int *data_index_all, double *factors_all,
    int num_bin, int nparams, double T, double t_ref, int tdi_type,
    double coarse_dt)
{
    (void) total_diff;
    (void) tdi_type;

    CUDA_SHARED double params[N_PARAMS_MAX];
    GBTDIonTheFly tdi_on_fly_here(orbits, tdi_config, T, t_ref);

    int m_min = wdm->ind_min_f;
    int m_max = wdm->ind_max_f;
    int n_min = wdm->ind_min_t;
    int n_max = wdm->ind_max_t;
    int Nt_active = wdm->Nt_active;
    int Nf_active = wdm->Nf_active;
    double layer_dt = wdm->layer_dt;
    double layer_df = wdm->layer_df;
    int total_points = Nf_active * Nt_active;

    CUDA_SHARED int link_Space_craft_rec[NLINKS];
    CUDA_SHARED int link_Space_craft_em[NLINKS];
    tdi_on_fly_here.fill_link_arrays(link_Space_craft_rec, link_Space_craft_em);
    CUDA_SYNC_THREADS;

    // Shared scratch for one spline slot (see WDMSplineSet layout).
    CUDA_SHARED double t_grid_shared      [WDM_SPLINE_L];
    CUDA_SHARED double amp_y_shared       [3 * WDM_SPLINE_L];
    CUDA_SHARED double dphi_y_shared      [3 * WDM_SPLINE_L];
    CUDA_SHARED double phi_ref_y_shared   [WDM_SPLINE_L];
    CUDA_SHARED double coefs_shared       [21 * WDM_SPLINE_L];
    CUDA_SHARED cmplx  tdi_chan_shared    [3 * WDM_SPLINE_L];
    CUDA_SHARED double pcr_scratch        [8 * WDM_SPLINE_L];
    CUDA_SHARED double B_scratch          [WDM_SPLINE_L];
    // get_tdi_buffer_size(L) = 2*L*8 + L*4 + L*1 = 21*L bytes.
    CUDA_SHARED char   get_tdi_scratch    [21 * WDM_SPLINE_L + 16];
    const int get_tdi_scratch_len = (int) sizeof(get_tdi_scratch);

    WDMSplineSet S;
    wdm_spline_set_init(&S, t_grid_shared,
                         amp_y_shared, dphi_y_shared, phi_ref_y_shared,
                         coefs_shared);
    CUDA_SYNC_THREADS;

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        double factor = factors_all[bin_i];

        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
            params[i] = params_all[bin_i * nparams + i];
        CUDA_SYNC_THREADS;

        // gb_wdm_fill_global_kernel).

        double t_active_start = t_ref + (double) n_min * layer_dt;
        int K = wdm_spline_num_windows(n_min, n_max, layer_dt, coarse_dt);

        for (int k = 0; k < K; ++k)
        {
            double t_window_start = t_active_start
                                  + (double) k * (WDM_SPLINE_L - 1) * coarse_dt;

            bool ok = build_wdm_spline_window(
                tdi_on_fly_here, &S, params, bin_i,
                t_window_start, coarse_dt,
                tdi_chan_shared, pcr_scratch, B_scratch,
                (void*) get_tdi_scratch, get_tdi_scratch_len);
            CUDA_SYNC_THREADS;
            if (!ok) continue;

            int n_lo, n_hi;
            wdm_spline_window_pixel_range(k, K, n_min, n_max,
                                          layer_dt, t_ref, t_active_start,
                                          coarse_dt, &n_lo, &n_hi);

            cmplx tdi_channel_val[3];
            double f[3], fdot[3];
            for (int n = THREAD_START_X + n_lo; n <= n_hi; n += BLOCK_INCR_X)
            {
                double tn = (double) n * layer_dt + t_ref;
                eval_wdm_spline_pixel(&S, tn, tdi_channel_val, f, fdot);

                for (int diff = -num_diff; diff <= +num_diff; diff += 1)
                {
                    for (int i = 0; i < 3; ++i)
                    {
                        int layer_m = (int)(f[i] / layer_df);
                        int layer_m_here = layer_m + diff;
                        if ((layer_m_here >= m_min) && (layer_m_here <= m_max))
                        {
                            double w_mn = factor *
                                wdm_lookup->get_wdm_in_channel_over_layers(
                                    tdi_channel_val[i], f[i], fdot[i],
                                    layer_m_here, n);
#ifdef __CUDACC__
                            atomicAdd(&template_fill[(i * total_points) +
                                ((layer_m_here - m_min) * Nt_active +
                                 (n - n_min))], w_mn);
#else
                            template_fill[(i * total_points) +
                                ((layer_m_here - m_min) * Nt_active +
                                 (n - n_min))] += w_mn;
#endif
                        }
                    }
                }
            }
            CUDA_SYNC_THREADS;
        }
    }
}

void GBComputationGroup::gb_wdm_spline_fill_global_wrap(
    double *template_fill, Orbits* orbits, TDIConfig *tdi_config,
    WaveletLookupTable* wdm_lookup, WDMDomain* wdm,
    double *params_all, int *data_index_all, double *factors_all,
    int num_bin, int nparams, double T, double t_ref, int tdi_type,
    double coarse_dt)
{
#ifdef __CUDACC__
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    WaveletLookupTable *d_wdm_lookup;
    cudaMalloc(&d_wdm_lookup, sizeof(WaveletLookupTable));
    gpuErrchk(cudaMemcpy(d_wdm_lookup, wdm_lookup, sizeof(WaveletLookupTable), cudaMemcpyHostToDevice));

    WDMDomain *d_wdm;
    cudaMalloc(&d_wdm, sizeof(WDMDomain));
    gpuErrchk(cudaMemcpy(d_wdm, wdm, sizeof(WDMDomain), cudaMemcpyHostToDevice));

    gb_wdm_spline_fill_global_kernel<2, 5><<<num_bin, NUM_THREADS_HERE>>>(
        template_fill, orbits, tdi_config, wdm_lookup, wdm,
        params_all, data_index_all, factors_all,
        num_bin, nparams, T, t_ref, tdi_type, coarse_dt);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_wdm_lookup));
    gpuErrchk(cudaFree(d_wdm));
#else
    gb_wdm_spline_fill_global_kernel<2, 5>(
        template_fill, orbits, tdi_config, wdm_lookup, wdm,
        params_all, data_index_all, factors_all,
        num_bin, nparams, T, t_ref, tdi_type, coarse_dt);
#endif
}



// =============================================================================
// Spline-path mirror of gb_wdm_get_ll_kernel. Outer window loop + spline
// evaluation per WDM pixel. Per-thread d_h / h_h shared accumulators and
// 4 * block_reduce(...) match the direct path so callers get bit-compatible
// outputs up to interpolation error.
// =============================================================================
template<int num_diff, int total_diff>
CUDA_KERNEL
void gb_wdm_spline_get_ll_kernel(
    double *d_h_out, double *h_h_out,
    Orbits* orbits, TDIConfig *tdi_config,
    WaveletLookupTable* wdm_lookup, WDMDomain* wdm,
    double *params_all, int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double T, double t_ref, int tdi_type,
    double coarse_dt)
{
    (void) total_diff;

    CUDA_SHARED double d_h_tmp[NUM_THREADS_HERE];
    CUDA_SHARED double h_h_tmp[NUM_THREADS_HERE];

    CUDA_SHARED double params[N_PARAMS_MAX];
    GBTDIonTheFly tdi_on_fly_here(orbits, tdi_config, T, t_ref);

    int m_min = wdm->ind_min_f;
    int m_max = wdm->ind_max_f;
    int n_min = wdm->ind_min_t;
    int n_max = wdm->ind_max_t;
    double layer_dt = wdm->layer_dt;
    double layer_df = wdm->layer_df;

    CUDA_SHARED int link_Space_craft_rec[NLINKS];
    CUDA_SHARED int link_Space_craft_em[NLINKS];
    tdi_on_fly_here.fill_link_arrays(link_Space_craft_rec, link_Space_craft_em);
    CUDA_SYNC_THREADS;

    // One spline slot (same shared layout as gb_wdm_spline_fill_global_kernel).
    CUDA_SHARED double t_grid_shared      [WDM_SPLINE_L];
    CUDA_SHARED double amp_y_shared       [3 * WDM_SPLINE_L];
    CUDA_SHARED double dphi_y_shared      [3 * WDM_SPLINE_L];
    CUDA_SHARED double phi_ref_y_shared   [WDM_SPLINE_L];
    CUDA_SHARED double coefs_shared       [21 * WDM_SPLINE_L];
    CUDA_SHARED cmplx  tdi_chan_shared    [3 * WDM_SPLINE_L];
    CUDA_SHARED double pcr_scratch        [8 * WDM_SPLINE_L];
    CUDA_SHARED double B_scratch          [WDM_SPLINE_L];
    CUDA_SHARED char   get_tdi_scratch    [21 * WDM_SPLINE_L + 16];
    const int get_tdi_scratch_len = (int) sizeof(get_tdi_scratch);

    WDMSplineSet S;
    wdm_spline_set_init(&S, t_grid_shared,
                         amp_y_shared, dphi_y_shared, phi_ref_y_shared,
                         coefs_shared);
    CUDA_SYNC_THREADS;

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        for (int i = THREAD_START_X; i < NUM_THREADS_HERE; i += BLOCK_INCR_X)
        {
            d_h_tmp[i] = 0.0;
            h_h_tmp[i] = 0.0;
        }

        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];
        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
            params[i] = params_all[bin_i * nparams + i];
        CUDA_SYNC_THREADS;

        // gb_wdm_fill_global_kernel).

        double t_active_start = t_ref + (double) n_min * layer_dt;
        int K = wdm_spline_num_windows(n_min, n_max, layer_dt, coarse_dt);

        for (int k = 0; k < K; ++k)
        {
            double t_window_start = t_active_start
                                  + (double) k * (WDM_SPLINE_L - 1) * coarse_dt;

            bool ok = build_wdm_spline_window(
                tdi_on_fly_here, &S, params, bin_i,
                t_window_start, coarse_dt,
                tdi_chan_shared, pcr_scratch, B_scratch,
                (void*) get_tdi_scratch, get_tdi_scratch_len);
            CUDA_SYNC_THREADS;
            if (!ok) continue;

            int n_lo, n_hi;
            wdm_spline_window_pixel_range(k, K, n_min, n_max,
                                          layer_dt, t_ref, t_active_start,
                                          coarse_dt, &n_lo, &n_hi);

            cmplx tdi_channel_val[3];
            double f[3], fdot[3];
            double wmn_channel[3];
            for (int n = THREAD_START_X + n_lo; n <= n_hi; n += BLOCK_INCR_X)
            {
                double tn = (double) n * layer_dt + t_ref;
                eval_wdm_spline_pixel(&S, tn, tdi_channel_val, f, fdot);

                // Match gb_wdm_get_ll_kernel: layer_m is chosen from the
                // 3-channel avg frequency (so all channels share a layer
                // index per (m, n) pixel).
                double avg_f = (f[0] + f[1] + f[2]) / 3.0;
                int layer_m = (int)(avg_f / layer_df);
                for (int diff = -num_diff; diff <= +num_diff; diff += 1)
                {
                    int layer_m_here = layer_m + diff;
                    if ((layer_m_here >= m_min) && (layer_m_here <= m_max))
                    {
                        for (int i = 0; i < 3; ++i)
                        {
                            wmn_channel[i] = wdm_lookup->
                                get_wdm_in_channel_over_layers(
                                    tdi_channel_val[i], f[i], fdot[i],
                                    layer_m_here, n);
                        }
                        wdm->add_ip_contrib(&d_h_tmp[0], &h_h_tmp[0],
                                            &wmn_channel[0],
                                            layer_m_here, n,
                                            data_index, noise_index, tdi_type);
                    }
                }
            }
            CUDA_SYNC_THREADS;
        }
        CUDA_SYNC_THREADS;

#ifdef __CUDACC__
        d_h_out[bin_i] = 4.0 * block_reduce(d_h_tmp);
        h_h_out[bin_i] = 4.0 * block_reduce(h_h_tmp);
        CUDA_SYNC_THREADS;
#else
        d_h_out[bin_i] = 4.0 * d_h_tmp[0];
        h_h_out[bin_i] = 4.0 * h_h_tmp[0];
#endif
    }
}

void GBComputationGroup::gb_wdm_spline_get_ll_wrap(
    double *d_h_out, double *h_h_out,
    Orbits* orbits, TDIConfig *tdi_config,
    WaveletLookupTable* wdm_lookup, WDMDomain* wdm,
    double *params_all, int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double T, double t_ref, int tdi_type,
    double coarse_dt)
{
#ifdef __CUDACC__
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    WaveletLookupTable *d_wdm_lookup;
    cudaMalloc(&d_wdm_lookup, sizeof(WaveletLookupTable));
    gpuErrchk(cudaMemcpy(d_wdm_lookup, wdm_lookup, sizeof(WaveletLookupTable), cudaMemcpyHostToDevice));

    WDMDomain *d_wdm;
    cudaMalloc(&d_wdm, sizeof(WDMDomain));
    gpuErrchk(cudaMemcpy(d_wdm, wdm, sizeof(WDMDomain), cudaMemcpyHostToDevice));

    gb_wdm_spline_get_ll_kernel<2, 5><<<num_bin, NUM_THREADS_HERE>>>(
        d_h_out, h_h_out, d_orbits, d_tdi_config, d_wdm_lookup, d_wdm,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, tdi_type, coarse_dt);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_wdm_lookup));
    gpuErrchk(cudaFree(d_wdm));
#else
    gb_wdm_spline_get_ll_kernel<2, 5>(
        d_h_out, h_h_out, orbits, tdi_config, wdm_lookup, wdm,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, tdi_type, coarse_dt);
#endif
}

// Swap-likelihood kernel in the WDM domain.
//
// Mirrors gb_wdm_get_ll_kernel: build the TDI on the fly with fast_wdm_inner
// (proper phase-ref / numerical-derivative frequency, conj/factor adjustments,
// out-of-orbit zero check), look up w_mn through the bounds-safe
// get_wdm_in_channel_over_layers wrapper, and accumulate the five swap
// quantities <d|h_add>, <d|h_remove>, <h_add|h_add>, <h_remove|h_remove>,
// <h_add|h_remove>.
//
// Memory layout vs the previous version:
//   * Per-bin partial sums live in registers (5 doubles/thread), not in
//     5*NUM_THREADS_HERE shared arrays.
//   * Block-wide reduction goes through block_reduce_scalar, which only keeps
//     the cub::BlockReduce TempStorage in __shared__.
//   * Only the param staging buffers and the NLINKS spacecraft arrays remain
//     in shared memory.


// =============================================================================
// Spline-path mirror of gb_wdm_swap_ll_kernel. Two spline slots in shared
// memory (add + remove); the layer-union iteration follows the direct path
// verbatim so add_ip_swap_contrib gets identical inputs up to interpolation
// error.
//
// Out-of-bounds check is "any coarse point of this window has the underlying
// raw M == 0", same as the direct path's per-pixel zero check (we just lift
// it to window granularity). When ONE side's window is bad we still build
// the other side and accumulate single-sided contributions for the pixels
// in that window.
// =============================================================================
template<int num_diff, int total_diff>
CUDA_KERNEL
void gb_wdm_spline_swap_ll_kernel(
    double *d_h_add_out, double *d_h_remove_out,
    double *add_add_out, double *remove_remove_out, double *add_remove_out,
    Orbits* orbits, TDIConfig *tdi_config,
    WaveletLookupTable* wdm_lookup, WDMDomain* wdm,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double T, double t_ref, int tdi_type,
    double coarse_dt)
{
    (void) total_diff;

    CUDA_SHARED double params_add[N_PARAMS_MAX];
    CUDA_SHARED double params_remove[N_PARAMS_MAX];

    CUDA_SHARED int link_Space_craft_rec[NLINKS];
    CUDA_SHARED int link_Space_craft_em[NLINKS];

    GBTDIonTheFly tdi_on_fly_here(orbits, tdi_config, T, t_ref);

    int m_min = wdm->ind_min_f;
    int m_max = wdm->ind_max_f;
    int n_min = wdm->ind_min_t;
    int n_max = wdm->ind_max_t;
    double layer_dt = wdm->layer_dt;
    double layer_df = wdm->layer_df;

    tdi_on_fly_here.fill_link_arrays(link_Space_craft_rec, link_Space_craft_em);
    CUDA_SYNC_THREADS;

    // Two spline slots (separate y / coefs). x grid + scratch shared.
    CUDA_SHARED double t_grid_shared      [WDM_SPLINE_L];
    CUDA_SHARED double amp_y_add          [3 * WDM_SPLINE_L];
    CUDA_SHARED double dphi_y_add         [3 * WDM_SPLINE_L];
    CUDA_SHARED double phi_ref_y_add      [WDM_SPLINE_L];
    CUDA_SHARED double coefs_add          [21 * WDM_SPLINE_L];
    CUDA_SHARED double amp_y_rem          [3 * WDM_SPLINE_L];
    CUDA_SHARED double dphi_y_rem         [3 * WDM_SPLINE_L];
    CUDA_SHARED double phi_ref_y_rem      [WDM_SPLINE_L];
    CUDA_SHARED double coefs_rem          [21 * WDM_SPLINE_L];
    CUDA_SHARED cmplx  tdi_chan_shared    [3 * WDM_SPLINE_L];
    CUDA_SHARED double pcr_scratch        [8 * WDM_SPLINE_L];
    CUDA_SHARED double B_scratch          [WDM_SPLINE_L];
    CUDA_SHARED char   get_tdi_scratch    [21 * WDM_SPLINE_L + 16];
    const int get_tdi_scratch_len = (int) sizeof(get_tdi_scratch);

    WDMSplineSet S_add, S_rem;
    wdm_spline_set_init(&S_add, t_grid_shared,
                         amp_y_add, dphi_y_add, phi_ref_y_add, coefs_add);
    wdm_spline_set_init(&S_rem, t_grid_shared,
                         amp_y_rem, dphi_y_rem, phi_ref_y_rem, coefs_rem);
    CUDA_SYNC_THREADS;

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        double d_h_add_acc = 0.0;
        double d_h_remove_acc = 0.0;
        double add_add_acc = 0.0;
        double remove_remove_acc = 0.0;
        double add_remove_acc = 0.0;

        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];

        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
        {
            params_add[i]    = params_add_all   [bin_i * nparams + i];
            params_remove[i] = params_remove_all[bin_i * nparams + i];
        }
        CUDA_SYNC_THREADS;

        // gb_wdm_fill_global_kernel).

        double t_active_start = t_ref + (double) n_min * layer_dt;
        int K = wdm_spline_num_windows(n_min, n_max, layer_dt, coarse_dt);

        for (int k = 0; k < K; ++k)
        {
            double t_window_start = t_active_start
                                  + (double) k * (WDM_SPLINE_L - 1) * coarse_dt;

            bool add_ok = build_wdm_spline_window(
                tdi_on_fly_here, &S_add, params_add, bin_i,
                t_window_start, coarse_dt,
                tdi_chan_shared, pcr_scratch, B_scratch,
                (void*) get_tdi_scratch, get_tdi_scratch_len);
            CUDA_SYNC_THREADS;
            bool rem_ok = build_wdm_spline_window(
                tdi_on_fly_here, &S_rem, params_remove, bin_i,
                t_window_start, coarse_dt,
                tdi_chan_shared, pcr_scratch, B_scratch,
                (void*) get_tdi_scratch, get_tdi_scratch_len);
            CUDA_SYNC_THREADS;
            if (!add_ok && !rem_ok) continue;

            int n_lo, n_hi;
            wdm_spline_window_pixel_range(k, K, n_min, n_max,
                                          layer_dt, t_ref, t_active_start,
                                          coarse_dt, &n_lo, &n_hi);

            cmplx tdi_channel_val_add[3];
            cmplx tdi_channel_val_remove[3];
            double f_add[3], fdot_add[3];
            double f_remove[3], fdot_remove[3];
            double wmn_add[3], wmn_remove[3];

            for (int n = THREAD_START_X + n_lo; n <= n_hi; n += BLOCK_INCR_X)
            {
                double tn = (double) n * layer_dt + t_ref;
                if (add_ok)
                    eval_wdm_spline_pixel(&S_add, tn,
                                          tdi_channel_val_add, f_add, fdot_add);
                if (rem_ok)
                    eval_wdm_spline_pixel(&S_rem, tn,
                                          tdi_channel_val_remove, f_remove, fdot_remove);

                int layer_m_add = 0, layer_m_remove = 0;
                if (add_ok)
                    layer_m_add = (int)((f_add[0] + f_add[1] + f_add[2]) / 3.0 / layer_df);
                if (rem_ok)
                    layer_m_remove = (int)((f_remove[0] + f_remove[1] + f_remove[2]) / 3.0 / layer_df);

                int layer_m_lo, layer_m_hi;
                if (add_ok && rem_ok)
                {
                    layer_m_lo = (layer_m_add < layer_m_remove) ? layer_m_add : layer_m_remove;
                    layer_m_hi = (layer_m_add > layer_m_remove) ? layer_m_add : layer_m_remove;
                }
                else if (add_ok)
                {
                    layer_m_lo = layer_m_add;
                    layer_m_hi = layer_m_add;
                }
                else
                {
                    layer_m_lo = layer_m_remove;
                    layer_m_hi = layer_m_remove;
                }

                for (int layer_m = layer_m_lo - num_diff;
                     layer_m <= layer_m_hi + num_diff; layer_m += 1)
                {
                    if ((layer_m < m_min) || (layer_m > m_max)) continue;
                    bool add_layer_active = add_ok &&
                        (layer_m >= layer_m_add - num_diff) &&
                        (layer_m <= layer_m_add + num_diff);
                    bool remove_layer_active = rem_ok &&
                        (layer_m >= layer_m_remove - num_diff) &&
                        (layer_m <= layer_m_remove + num_diff);
                    if (!add_layer_active && !remove_layer_active) continue;

                    for (int j = 0; j < 3; ++j)
                    {
                        wmn_add[j] = add_layer_active ?
                            wdm_lookup->get_wdm_in_channel_over_layers(
                                tdi_channel_val_add[j], f_add[j], fdot_add[j],
                                layer_m, n) : 0.0;
                        wmn_remove[j] = remove_layer_active ?
                            wdm_lookup->get_wdm_in_channel_over_layers(
                                tdi_channel_val_remove[j], f_remove[j], fdot_remove[j],
                                layer_m, n) : 0.0;
                    }

                    wdm->add_ip_swap_contrib(
                        &d_h_add_acc, &d_h_remove_acc,
                        &add_add_acc, &remove_remove_acc, &add_remove_acc,
                        &wmn_add[0], &wmn_remove[0], layer_m, n,
                        data_index, noise_index, tdi_type);
                }
            }
            CUDA_SYNC_THREADS;
        }
        CUDA_SYNC_THREADS;

#ifdef __CUDACC__
        double d_h_add_red       = 4.0 * block_reduce_scalar(d_h_add_acc);
        double d_h_remove_red    = 4.0 * block_reduce_scalar(d_h_remove_acc);
        double add_add_red       = 4.0 * block_reduce_scalar(add_add_acc);
        double remove_remove_red = 4.0 * block_reduce_scalar(remove_remove_acc);
        double add_remove_red    = 4.0 * block_reduce_scalar(add_remove_acc);
        if (threadIdx.x == 0)
        {
            d_h_add_out[bin_i]       = d_h_add_red;
            d_h_remove_out[bin_i]    = d_h_remove_red;
            add_add_out[bin_i]       = add_add_red;
            remove_remove_out[bin_i] = remove_remove_red;
            add_remove_out[bin_i]    = add_remove_red;
        }
        CUDA_SYNC_THREADS;
#else
        d_h_add_out[bin_i]       = 4.0 * d_h_add_acc;
        d_h_remove_out[bin_i]    = 4.0 * d_h_remove_acc;
        add_add_out[bin_i]       = 4.0 * add_add_acc;
        remove_remove_out[bin_i] = 4.0 * remove_remove_acc;
        add_remove_out[bin_i]    = 4.0 * add_remove_acc;
#endif
    }
}

void GBComputationGroup::gb_wdm_spline_swap_ll_wrap(
    double *d_h_add_out, double *d_h_remove_out,
    double *add_add_out, double *remove_remove_out, double *add_remove_out,
    Orbits* orbits, TDIConfig *tdi_config,
    WaveletLookupTable* wdm_lookup, WDMDomain* wdm,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double T, double t_ref, int tdi_type,
    double coarse_dt)
{
#ifdef __CUDACC__
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    WaveletLookupTable *d_wdm_lookup;
    cudaMalloc(&d_wdm_lookup, sizeof(WaveletLookupTable));
    gpuErrchk(cudaMemcpy(d_wdm_lookup, wdm_lookup, sizeof(WaveletLookupTable), cudaMemcpyHostToDevice));

    WDMDomain *d_wdm;
    cudaMalloc(&d_wdm, sizeof(WDMDomain));
    gpuErrchk(cudaMemcpy(d_wdm, wdm, sizeof(WDMDomain), cudaMemcpyHostToDevice));

    gb_wdm_spline_swap_ll_kernel<2, 5><<<num_bin, NUM_THREADS_HERE>>>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        d_orbits, d_tdi_config, d_wdm_lookup, d_wdm,
        params_add_all, params_remove_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, tdi_type, coarse_dt);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_wdm_lookup));
    gpuErrchk(cudaFree(d_wdm));
#else
    gb_wdm_spline_swap_ll_kernel<2, 5>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits, tdi_config, wdm_lookup, wdm,
        params_add_all, params_remove_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, tdi_type, coarse_dt);
#endif
}


// =============================================================================
//  Chain-rule gradients of gb_wdm_get_ll / gb_wdm_swap_ll w.r.t. the
//  N_PARAMS_MAX-vector of galactic-binary parameters per binary.
//
//  Strategy:
//    For each pixel (m, n) we already build the central wavelet coefficient
//    w_h(theta) on the fly via fast_wdm_inner + wdm_lookup->get_wdm_in_channel
//    _over_layers.  The gradient kernel additionally computes, at the *same*
//    pixel, the central-difference parameter derivative
//
//        dw_h/dtheta_k = ( w_h(theta + eps_k e_k) - w_h(theta - eps_k e_k) )
//                        / (2 * eps_k),
//
//    via two extra fast_wdm_inner calls per parameter, and accumulates the
//    chain-rule inner product (residual * dw_h/dtheta_k * N^{-1}) into a
//    per-thread register accumulator.  The accumulator is block-reduced and
//    multiplied by 4 at the end, exactly like d_h_out / h_h_out.
//
//  The layer_m used in the gradient sum is frozen at the *central* value
//  layer_m_c (and similarly layer_m_add / layer_m_remove for swap).  The
//  perturbed evaluation is queried at that same layer_m, so the FD truly
//  represents dw_h/dtheta at fixed (m, n) -- matching the analytic chain
//  rule used in the JAX/Python reference (gb_chain_rule_grad.py).
//
//  All per-thread temporaries (params copy, register accumulators, w_mn
//  caches) live on the local stack / in registers; only the link arrays and
//  param_eps stay in shared memory.  Each block still drives one binary, with
//  NUM_THREADS_HERE threads sharing the n-pixel loop.
// =============================================================================






// =============================================================================
// Spline-path gradient kernel: same chain-rule formula and frozen layer_m_c
// convention as gb_wdm_get_ll_grad_kernel. Memory footprint is independent of
// nparams: three spline slots in shared memory (base, plus, minus) get
// rebuilt per (window, param). Per source the build count is
//   1 base + 2*nparams_active perturbed   builds per window,
// vs the direct kernel's
//   (1 + 2*nparams_active) * num_wdm_pixels fast_wdm_inner calls.
//
// Precision note: the chain-rule central FD divides by (2*eps_k). Cubic
// spline interpolation introduces a per-(theta+eps) error that does NOT
// correlate across theta+eps / theta-eps (each rebuild fits a fresh spline
// with different y-values). The diff therefore does not cancel spline
// interpolation noise, and the relative error in dh/dtheta_k scales as
// ~ spline_err / eps_k. For tight eps (e.g. default param_eps[f0]=2e-14)
// this can dominate and the spline gradient may not match the direct
// gradient to better than O(1) on the noisiest components. Recommend
// using the direct gb_wdm_get_ll_grad_wrap when bit-level chain-rule
// matching is required.
// =============================================================================
template<int num_diff, int total_diff>
CUDA_KERNEL
void gb_wdm_spline_get_ll_grad_kernel(
    double *grad_out, Orbits* orbits, TDIConfig *tdi_config,
    WaveletLookupTable* wdm_lookup, WDMDomain* wdm,
    double *params_all, int *data_index_all, int *noise_index_all,
    double *param_eps,
    int num_bin, int nparams, double T, double t_ref, int tdi_type,
    double coarse_dt)
{
    (void) total_diff;

    CUDA_SHARED int link_Space_craft_rec[NLINKS];
    CUDA_SHARED int link_Space_craft_em[NLINKS];
    CUDA_SHARED double param_eps_shared[N_PARAMS_MAX];

    GBTDIonTheFly tdi_on_fly_here(orbits, tdi_config, T, t_ref);

    int m_min = wdm->ind_min_f;
    int m_max = wdm->ind_max_f;
    int n_min = wdm->ind_min_t;
    int n_max = wdm->ind_max_t;
    double layer_dt = wdm->layer_dt;
    double layer_df = wdm->layer_df;
    int nchannels = (tdi_type == TDI_AE) ? 2 : 3;

    tdi_on_fly_here.fill_link_arrays(link_Space_craft_rec, link_Space_craft_em);

    for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
        param_eps_shared[i] = param_eps[i];
    CUDA_SYNC_THREADS;

    // Three spline slots: base (A), plus (B), minus (C). The y arrays and
    // coefs are slot-private; t_grid + scratch are shared.
    CUDA_SHARED double t_grid_shared      [WDM_SPLINE_L];
    CUDA_SHARED double amp_y_A            [3 * WDM_SPLINE_L];
    CUDA_SHARED double dphi_y_A           [3 * WDM_SPLINE_L];
    CUDA_SHARED double phi_ref_y_A        [WDM_SPLINE_L];
    CUDA_SHARED double coefs_A            [21 * WDM_SPLINE_L];
    CUDA_SHARED double amp_y_B            [3 * WDM_SPLINE_L];
    CUDA_SHARED double dphi_y_B           [3 * WDM_SPLINE_L];
    CUDA_SHARED double phi_ref_y_B        [WDM_SPLINE_L];
    CUDA_SHARED double coefs_B            [21 * WDM_SPLINE_L];
    CUDA_SHARED double amp_y_C            [3 * WDM_SPLINE_L];
    CUDA_SHARED double dphi_y_C           [3 * WDM_SPLINE_L];
    CUDA_SHARED double phi_ref_y_C        [WDM_SPLINE_L];
    CUDA_SHARED double coefs_C            [21 * WDM_SPLINE_L];
    CUDA_SHARED cmplx  tdi_chan_shared    [3 * WDM_SPLINE_L];
    CUDA_SHARED double pcr_scratch        [8 * WDM_SPLINE_L];
    CUDA_SHARED double B_scratch          [WDM_SPLINE_L];
    CUDA_SHARED char   get_tdi_scratch    [21 * WDM_SPLINE_L + 16];
    const int get_tdi_scratch_len = (int) sizeof(get_tdi_scratch);

    WDMSplineSet S_A, S_B, S_C;
    wdm_spline_set_init(&S_A, t_grid_shared, amp_y_A, dphi_y_A, phi_ref_y_A, coefs_A);
    wdm_spline_set_init(&S_B, t_grid_shared, amp_y_B, dphi_y_B, phi_ref_y_B, coefs_B);
    wdm_spline_set_init(&S_C, t_grid_shared, amp_y_C, dphi_y_C, phi_ref_y_C, coefs_C);
    CUDA_SYNC_THREADS;

    // Per-source param buffers (shared so all threads in the block see the
    // same perturbed values when building plus/minus splines).
    CUDA_SHARED double params_base[N_PARAMS_MAX];
    CUDA_SHARED double params_pert[N_PARAMS_MAX];

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];

        for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
            params_base[i] = params_all[bin_i * nparams + i];
        CUDA_SYNC_THREADS;


        // Per-thread gradient accumulators (one per parameter slot).
        double grad_acc[N_PARAMS_MAX];
        for (int i = 0; i < N_PARAMS_MAX; ++i) grad_acc[i] = 0.0;

        double t_active_start = t_ref + (double) n_min * layer_dt;
        int K = wdm_spline_num_windows(n_min, n_max, layer_dt, coarse_dt);

        for (int k = 0; k < K; ++k)
        {
            double t_window_start = t_active_start
                                  + (double) k * (WDM_SPLINE_L - 1) * coarse_dt;

            bool base_ok = build_wdm_spline_window(
                tdi_on_fly_here, &S_A, params_base, bin_i,
                t_window_start, coarse_dt,
                tdi_chan_shared, pcr_scratch, B_scratch,
                (void*) get_tdi_scratch, get_tdi_scratch_len);
            CUDA_SYNC_THREADS;
            if (!base_ok) continue;

            int n_lo, n_hi;
            wdm_spline_window_pixel_range(k, K, n_min, n_max,
                                          layer_dt, t_ref, t_active_start,
                                          coarse_dt, &n_lo, &n_hi);

            for (int kk = 0; kk < nparams; ++kk)
            {
                double eps_k = param_eps_shared[kk];
                if (eps_k <= 0.0) continue;

                // Build plus splines.
                for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
                    params_pert[i] = params_base[i];
                CUDA_SYNC_THREADS;
                if (THREAD_ZERO) params_pert[kk] = params_base[kk] + eps_k;
                CUDA_SYNC_THREADS;
                bool plus_ok = build_wdm_spline_window(
                    tdi_on_fly_here, &S_B, params_pert, bin_i,
                    t_window_start, coarse_dt,
                    tdi_chan_shared, pcr_scratch, B_scratch,
                    (void*) get_tdi_scratch, get_tdi_scratch_len);
                CUDA_SYNC_THREADS;

                // Build minus splines.
                for (int i = THREAD_START_X; i < nparams; i += BLOCK_INCR_X)
                    params_pert[i] = params_base[i];
                CUDA_SYNC_THREADS;
                if (THREAD_ZERO) params_pert[kk] = params_base[kk] - eps_k;
                CUDA_SYNC_THREADS;
                bool minus_ok = build_wdm_spline_window(
                    tdi_on_fly_here, &S_C, params_pert, bin_i,
                    t_window_start, coarse_dt,
                    tdi_chan_shared, pcr_scratch, B_scratch,
                    (void*) get_tdi_scratch, get_tdi_scratch_len);
                CUDA_SYNC_THREADS;
                if (!plus_ok && !minus_ok) continue;

                double inv_2eps = 1.0 / (2.0 * eps_k);

                cmplx tdi_chan_c[3], tdi_chan_p[3], tdi_chan_m[3];
                double f_c[3], fdot_c[3], f_p[3], fdot_p[3], f_m[3], fdot_m[3];

                for (int n = THREAD_START_X + n_lo; n <= n_hi; n += BLOCK_INCR_X)
                {
                    double tn = (double) n * layer_dt + t_ref;

                    eval_wdm_spline_pixel(&S_A, tn, tdi_chan_c, f_c, fdot_c);
                    if (plus_ok)
                        eval_wdm_spline_pixel(&S_B, tn, tdi_chan_p, f_p, fdot_p);
                    if (minus_ok)
                        eval_wdm_spline_pixel(&S_C, tn, tdi_chan_m, f_m, fdot_m);

                    // Frozen central layer_m, matching gb_wdm_get_ll_grad_kernel.
                    double avg_f_c = (f_c[0] + f_c[1] + f_c[2]) / 3.0;
                    int layer_m_c = (int)(avg_f_c / layer_df);

                    for (int diff = -num_diff; diff <= num_diff; diff += 1)
                    {
                        int layer_m_here = layer_m_c + diff;
                        if ((layer_m_here < m_min) || (layer_m_here > m_max)) continue;

                        double w_mn_c[3]  = {0., 0., 0.};
                        double dw[3]      = {0., 0., 0.};
                        for (int c = 0; c < nchannels; ++c)
                        {
                            w_mn_c[c] = wdm_lookup->get_wdm_in_channel_over_layers(
                                tdi_chan_c[c], f_c[c], fdot_c[c], layer_m_here, n);
                            double wp = plus_ok ?
                                wdm_lookup->get_wdm_in_channel_over_layers(
                                    tdi_chan_p[c], f_p[c], fdot_p[c], layer_m_here, n) : 0.0;
                            double wm = minus_ok ?
                                wdm_lookup->get_wdm_in_channel_over_layers(
                                    tdi_chan_m[c], f_m[c], fdot_m[c], layer_m_here, n) : 0.0;
                            dw[c] = (wp - wm) * inv_2eps;
                        }

                        wdm->add_grad_contrib(&grad_acc[kk],
                                              &w_mn_c[0], &dw[0],
                                              layer_m_here, n,
                                              data_index, noise_index, tdi_type);
                    }
                }
                CUDA_SYNC_THREADS;
            }
        }
        CUDA_SYNC_THREADS;

        // Block-reduce each gradient accumulator and write out.
        for (int kk = 0; kk < nparams; ++kk)
        {
#ifdef __CUDACC__
            double red = block_reduce_scalar(grad_acc[kk]);
            if (threadIdx.x == 0)
                grad_out[bin_i * nparams + kk] = 4.0 * red;
            CUDA_SYNC_THREADS;
#else
            grad_out[bin_i * nparams + kk] = 4.0 * grad_acc[kk];
#endif
        }
    }
}

void GBComputationGroup::gb_wdm_spline_get_ll_grad_wrap(
    double *grad_out, Orbits* orbits, TDIConfig *tdi_config,
    WaveletLookupTable* wdm_lookup, WDMDomain* wdm,
    double *params_all, int *data_index_all, int *noise_index_all,
    double *param_eps,
    int num_bin, int nparams, double T, double t_ref, int tdi_type,
    double coarse_dt)
{
#ifdef __CUDACC__
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    WaveletLookupTable *d_wdm_lookup;
    cudaMalloc(&d_wdm_lookup, sizeof(WaveletLookupTable));
    gpuErrchk(cudaMemcpy(d_wdm_lookup, wdm_lookup, sizeof(WaveletLookupTable), cudaMemcpyHostToDevice));

    WDMDomain *d_wdm;
    cudaMalloc(&d_wdm, sizeof(WDMDomain));
    gpuErrchk(cudaMemcpy(d_wdm, wdm, sizeof(WDMDomain), cudaMemcpyHostToDevice));

    gb_wdm_spline_get_ll_grad_kernel<2, 5><<<num_bin, NUM_THREADS_HERE>>>(
        grad_out, d_orbits, d_tdi_config, d_wdm_lookup, d_wdm,
        params_all, data_index_all, noise_index_all, param_eps,
        num_bin, nparams, T, t_ref, tdi_type, coarse_dt);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_wdm_lookup));
    gpuErrchk(cudaFree(d_wdm));
#else
    gb_wdm_spline_get_ll_grad_kernel<2, 5>(
        grad_out, orbits, tdi_config, wdm_lookup, wdm,
        params_all, data_index_all, noise_index_all, param_eps,
        num_bin, nparams, T, t_ref, tdi_type, coarse_dt);
#endif
}


// -----------------------------------------------------------------------------
//  Swap-likelihood gradient kernel
//
//  Computes d(ll_diff)/d(theta_add[k]) and d(ll_diff)/d(theta_remove[k]) for
//  each binary in parallel.  The structure mirrors gb_wdm_swap_ll_kernel: for
//  each pixel (m, n) inside the union of the two templates' layer ranges we
//
//    1. evaluate central w_add(theta_add) and w_remove(theta_remove);
//    2. compute the post-swap residual at that pixel,
//         r_after_c = w_d_c - w_add_c + w_remove_c;
//    3. for each k in [0, nparams):  central-difference dw_add / dtheta_add[k]
//       and accumulate +r_after * dw_add * N^{-1} into grad_add[k];
//    4. similarly central-difference dw_remove / dtheta_remove[k] and
//       accumulate -r_after * dw_remove * N^{-1} into grad_remove[k].
//
//  As in gb_wdm_swap_ll_kernel we use a per-template active-layer mask so
//  that we never sample dw on layers more than num_diff away from the
//  central template's layer_m.
// -----------------------------------------------------------------------------
#endif  // === end gb_wdm_spline_* kernel + wrap bodies disabled ===





#define NLINKS 6

// === LISATDIonTheFly methods (first block) moved to LAT at Phase 3L.5 ===
#if 0
CUDA_DEVICE
void LISATDIonTheFly::get_sky_vectors(Vec *k, Vec *u, Vec *v, double *params)
{

    double beta = params[beta_index];
    double lam = params[lam_index];
    double cosbeta = cos(beta);
    double sinbeta = sin(beta);

    double coslam = cos(lam);
    double sinlam = sin(lam);
    v->x = -sinbeta * coslam;
    v->y = -sinbeta * sinlam;
    v->z = cosbeta;
    u->x = sinlam;
    u->y = -coslam;
    u->z = 0.0;
    k->x = -cosbeta * coslam;
    k->y = -cosbeta * sinlam;
    k->z = -sinbeta;
    
}

CUDA_DEVICE
void LISATDIonTheFly::xi_projections(double *xi_p, double *xi_c, Vec u, Vec v, Vec n)
{
    double u_dot_n = u.dot(n);
    double v_dot_n = v.dot(n);

    *xi_p = 0.5 * ((u_dot_n * u_dot_n) - (v_dot_n * v_dot_n));
    *xi_c = u_dot_n * v_dot_n;
}

CUDA_DEVICE
void LISATDIonTheFly::fill_link_arrays(int *link_Space_craft_rec, int *link_Space_craft_em)
{
    for (int i = THREAD_START_X; i < NLINKS; i += BLOCK_INCR_X)
    {
        link_Space_craft_rec[i] = orbits->sc_r[i];
        link_Space_craft_em[i] = orbits->sc_e[i];
        // links[i] = orbits->links[i];
        // if (threadIdx.x == 1)
        // printf("%d %d %d %d\n", orbits->sc_r[i], orbits->sc_e[i], link_Space_craft_em[i], link_Space_craft_rec[i]);
    }
    CUDA_SYNC_THREADS;
}
CUDA_DEVICE
void LISATDIonTheFly::get_tdi_Xf(cmplx *tdi_channels_arr, double *params, double *t_data, int N, int bin_i, int *link_Space_craft_rec, int *link_Space_craft_em, Vec k, Vec u, Vec v)
{
    double t;
    cmplx tdi_channel_tmp[3];
    for (int i = THREAD_START_X; i < N; i += BLOCK_INCR_X)
    {
        t = t_data[i];
        get_tdi_Xf_single(&tdi_channel_tmp[0], t, params, k, u, v, link_Space_craft_rec, link_Space_craft_em, bin_i);
        
        for (int channel = 0; channel < tdi_config->num_channels; channel += 1)
        {
            tdi_channels_arr[channel * N + i] = tdi_channel_tmp[channel];
        }
    }
}

// void LISATDIonTheFly::get_tdi_Xf_single_with_f_fdot(cmplx *tdi_channel, double *f, double *fdot, double t, double *params, Vec *k, Vec *u, Vec *v, int *link_Space_craft_rec, int *link_Space_craft_em)
// {
//     get_tdi_Xf_single(tdi_channel, t, k, u, v, link_Space_craft_rec, link_Space_craft_em);

//     cmplx tdi_channels_up[3];
//     cmplx tdi_channels_down[3];
//     double eps_rel = 1e-9
//     double t_up = t * (1. + eps_rel);
//     double t_down = t * (1. - eps_rel);
//     double h = t_up - t;

//     get_tdi_Xf_single(tdi_channels_up, t_up, k, u, v, link_Space_craft_rec, link_Space_craft_em);
//     get_tdi_Xf_single(tdi_channels_down, t_down, k, u, v, link_Space_craft_rec, link_Space_craft_em);

//     double phase_mid, phase_up, phase_down;

//     for (int i = 0; i < 3; i += 1)
//     {
//         phase_down = gcmplx::arg(tdi_channels_down[i]);
//         phase_mid = gcmplx::arg(tdi_channels[i]);
//         phase_up = gcmplx::arg(tdi_channels_up[i]);

//         if (phase_up - phase_down) > M_PI
//         {
            
//         }

//         f[i] = (phase_up - phase_down) / (2 * h);
//         fdot[i] = (phase_up - 2 * phase_mid + phase_up) / (h * h);
//     }
// }

void LISATDIonTheFly::get_tdi_Xf_single(cmplx *tdi_channel, double t, double *params, Vec k, Vec u, Vec v, int *link_Space_craft_rec, int *link_Space_craft_em, int bin_i)
{
    Vec x_rec;
    Vec x_em;
    Vec n;
    double delay_rec, phase_change;
    double delay_em;
    double xi_p;
    double xi_c;
    double k_dot_n, k_dot_x_rec, k_dot_x_em;
    double L;
    double hp_del_rec, hp_del_em, hc_del_rec, hc_del_em;
    cmplx I(0.0, 1.0);
    double pre_factor, large_factor_real, large_factor_imag;

    tdi_channel[0] = 0.0;
    tdi_channel[1] = 0.0;
    tdi_channel[2] = 0.0;
    int sc_r, sc_e;
    double total_delay;
    double time_eval, time_rec, time_em;
    double norm;
    cmplx tmp_channel_output[3];
    tmp_channel_output[0] = 0.0;
    tmp_channel_output[1] = 0.0;
    tmp_channel_output[2] = 0.0;
    int window = 0;
    bool is_okay = true;
    for (int unit_i = 0; unit_i < tdi_config->num_units; unit_i += 1)
    {
        int unit_start = tdi_config->unit_starts[unit_i];
        int unit_length = tdi_config->unit_lengths[unit_i];
        int base_link = tdi_config->tdi_base_link[unit_i];
        int base_link_index = orbits->get_link_ind(base_link);
        int channel = tdi_config->channels[unit_i];
        double sign = tdi_config->tdi_signs_in[unit_i];

        total_delay = 0.0;
        for (int sub_i = 0; sub_i < unit_length; sub_i += 1)
        {
            
            int combination_index = unit_start + sub_i;
            int combination_link = tdi_config->tdi_link_combinations[combination_index];
            // int combination_link_index;
            // if (combination_link == -11)
            // {
            //     combination_link_index = -1;
            // }
            // else
            // {
            //     combination_link_index = orbits->get_link_ind(combination_link);
            // }

            if (combination_link != -11)
            {
                total_delay += orbits->get_light_travel_time(t, combination_link);
            }
        }
        
        time_eval = t - total_delay;
        time_rec = time_eval;

        window = orbits->get_window(time_rec, orbits->ltt_t0, orbits->ltt_dt, orbits->ltt_N);
        if (window == -1)
        {
            // out of bounds
            is_okay = false;
            break;
        }
        window = orbits->get_window(time_eval, orbits->ltt_t0, orbits->ltt_dt, orbits->ltt_N);
        if (window == -1)
        {
            // out of bounds
            is_okay = false;
            break;
        }
        window = orbits->get_window(time_rec, orbits->sc_t0, orbits->sc_dt, orbits->sc_N);
        if (window == -1)
        {
            // out of bounds
            is_okay = false;
            break;
        }
        window = orbits->get_window(time_eval, orbits->sc_t0, orbits->sc_dt, orbits->sc_N);
        if (window == -1)
        {
            // out of bounds
            is_okay = false;
            break;
        }
        
        L = orbits->get_light_travel_time(time_rec, base_link);
        time_em = time_rec - L;

        sc_r = link_Space_craft_rec[base_link_index];
        sc_e = link_Space_craft_em[base_link_index];

        
        x_rec = orbits->get_pos(time_rec, sc_r);
        x_em = orbits->get_pos(time_em, sc_e);
        n = x_rec - x_em; // # TODO: check if this right
        norm = sqrt(n.dot(n));
        n = n / norm;

        k_dot_n = k.dot(n);
        k_dot_x_rec = k.dot(x_rec); // receiver
        k_dot_x_em = k.dot(x_em); // emitter

        // Guard the LISA arm-response singularity: when the wave propagation
        // direction k is parallel to the arm n, (1-k.n) -> 0 while xi_p, xi_c
        // -> 0 simultaneously, producing 0 * Inf = NaN. Skip the contribution
        // on the singular line (limit is well-defined and ~0 for sources not
        // sitting exactly on the arm axis).
        {
            double _denom = 1. - k_dot_n;
            if (fabs(_denom) < 1.0e-12) continue;
            pre_factor = 1. / _denom;
        }

        delay_rec = time_rec - k_dot_x_rec * C_inv;
        delay_em = time_em - k_dot_x_em * C_inv;

        xi_projections(&xi_p, &xi_c, u, v, n);

        phase_change = 0.0; // the real part
        get_hp_hc(&hp_del_rec, &hc_del_rec, delay_rec, params, phase_change, bin_i);
        get_hp_hc(&hp_del_em, &hc_del_em, delay_em, params, phase_change, bin_i);
        
        large_factor_real = (hp_del_em - hp_del_rec) * xi_p + (hc_del_em - hc_del_rec) * xi_c;
        
        phase_change = M_PI / 2.0; // the real part
        get_hp_hc(&hp_del_rec, &hc_del_rec, delay_rec, params, phase_change, bin_i);
        get_hp_hc(&hp_del_em, &hc_del_em, delay_em, params, phase_change, bin_i);
        
        large_factor_imag = (hp_del_em - hp_del_rec) * xi_p + (hc_del_em - hc_del_rec) * xi_c;
        tmp_channel_output[channel] += sign * pre_factor * (large_factor_real + I * large_factor_imag);
    }
    if (is_okay)
    {
        for (int channel = 0; channel < 3; channel += 1)
        {
            tdi_channel[channel] = tmp_channel_output[channel];
        }
    }
}

// FLY: 100 1.556340000000e+06 0 3 13 9.221909979440e-23 1.556340000000e+06
// FLY: 100 1.556340000000e+06 0 2 31 9.120999116182e-23 1.556331661075e+06

// BASE: 155634 1.556340000000e+06 0 3 13 9.221909189436e-23 1.556340000000e+06
// BASE: 155634 1.556340000000e+06 0 2 31 8.403496674567e-23 1.556331661075e+06


// ============================================================================
// Orbit-cache variants of get_tdi_Xf_single / get_tdi_Xf / get_tdi /
// get_tdi_heterodyned. Mirror the originals but swap raw orbit lookups
// (orbits->get_*) for cached cubic-spline evaluations (cache_get_*).
// ============================================================================
CUDA_DEVICE
void LISATDIonTheFly::get_tdi_Xf_single_cached(
    cmplx *tdi_channel, double t, double *params,
    Vec k, Vec u, Vec v,
    int *link_Space_craft_rec, int *link_Space_craft_em, int bin_i,
    OrbitsSplineCache *cache)
{
    Vec x_rec, x_em, n;
    double delay_rec, phase_change, delay_em;
    double xi_p, xi_c;
    double k_dot_n, k_dot_x_rec, k_dot_x_em;
    double L;
    double hp_del_rec, hp_del_em, hc_del_rec, hc_del_em;
    cmplx I(0.0, 1.0);
    double pre_factor, large_factor_real, large_factor_imag;

    tdi_channel[0] = 0.0;
    tdi_channel[1] = 0.0;
    tdi_channel[2] = 0.0;
    int sc_r, sc_e;
    double total_delay;
    double time_eval, time_rec, time_em;
    double norm;
    cmplx tmp_channel_output[3];
    tmp_channel_output[0] = 0.0;
    tmp_channel_output[1] = 0.0;
    tmp_channel_output[2] = 0.0;
    for (int unit_i = 0; unit_i < tdi_config->num_units; unit_i += 1)
    {
        int unit_start  = tdi_config->unit_starts[unit_i];
        int unit_length = tdi_config->unit_lengths[unit_i];
        int base_link   = tdi_config->tdi_base_link[unit_i];
        int base_link_index = _orbit_cache_link_index(base_link);
        int channel     = tdi_config->channels[unit_i];
        double sign     = tdi_config->tdi_signs_in[unit_i];

        total_delay = 0.0;
        for (int sub_i = 0; sub_i < unit_length; sub_i += 1)
        {
            int combination_index = unit_start + sub_i;
            int combination_link  = tdi_config->tdi_link_combinations[combination_index];
            if (combination_link != -11)
            {
                total_delay += cache_get_light_travel_time(cache, t, combination_link);
            }
        }

        time_eval = t - total_delay;
        time_rec  = time_eval;

        // Bounds checks against orbits' raw global tables are not needed:
        // by construction the cache is built from t-values inside the
        // source's valid orbit window, so any t inside the chunk is in
        // bounds. Compare to the original which sets is_okay=false on
        // get_window == -1.

        L       = cache_get_light_travel_time(cache, time_rec, base_link);
        time_em = time_rec - L;

        sc_r = link_Space_craft_rec[base_link_index];
        sc_e = link_Space_craft_em[base_link_index];

        x_rec = cache_get_pos(cache, time_rec, sc_r);
        x_em  = cache_get_pos(cache, time_em, sc_e);
        n     = x_rec - x_em;
        norm  = sqrt(n.dot(n));
        n     = n / norm;

        k_dot_n     = k.dot(n);
        k_dot_x_rec = k.dot(x_rec);
        k_dot_x_em  = k.dot(x_em);

        // Guard the arm-response singularity (see get_tdi_Xf_single).
        {
            double _denom = 1.0 - k_dot_n;
            if (fabs(_denom) < 1.0e-12) continue;
            pre_factor = 1.0 / _denom;
        }
        delay_rec  = time_rec - k_dot_x_rec * C_inv;
        delay_em   = time_em  - k_dot_x_em  * C_inv;

        xi_projections(&xi_p, &xi_c, u, v, n);

        phase_change = 0.0;
        get_hp_hc(&hp_del_rec, &hc_del_rec, delay_rec, params, phase_change, bin_i);
        get_hp_hc(&hp_del_em,  &hc_del_em,  delay_em,  params, phase_change, bin_i);
        large_factor_real = (hp_del_em - hp_del_rec) * xi_p + (hc_del_em - hc_del_rec) * xi_c;

        phase_change = M_PI / 2.0;
        get_hp_hc(&hp_del_rec, &hc_del_rec, delay_rec, params, phase_change, bin_i);
        get_hp_hc(&hp_del_em,  &hc_del_em,  delay_em,  params, phase_change, bin_i);
        large_factor_imag = (hp_del_em - hp_del_rec) * xi_p + (hc_del_em - hc_del_rec) * xi_c;

        tmp_channel_output[channel] += sign * pre_factor * (large_factor_real + I * large_factor_imag);
    }
    for (int channel = 0; channel < 3; channel += 1)
    {
        tdi_channel[channel] = tmp_channel_output[channel];
    }
}


CUDA_DEVICE
void LISATDIonTheFly::get_tdi_Xf_cached(
    cmplx *tdi_channels_arr, double *params, double *t_data, int N, int bin_i,
    int *link_Space_craft_rec, int *link_Space_craft_em,
    Vec k, Vec u, Vec v, OrbitsSplineCache *cache)
{
    double t;
    cmplx tdi_channel_tmp[3];
    for (int i = THREAD_START_X; i < N; i += BLOCK_INCR_X)
    {
        t = t_data[i];
        get_tdi_Xf_single_cached(&tdi_channel_tmp[0], t, params, k, u, v,
                                  link_Space_craft_rec, link_Space_craft_em,
                                  bin_i, cache);
        for (int channel = 0; channel < tdi_config->num_channels; channel += 1)
        {
            tdi_channels_arr[channel * N + i] = tdi_channel_tmp[channel];
        }
    }
}


CUDA_DEVICE
void LISATDIonTheFly::get_tdi_cached(
    void *buffer, int buffer_length,
    cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int bin_i, int nchannels,
    OrbitsSplineCache *cache)
{
#ifdef __CUDACC__
#else
    if (buffer_length < 2 * N * sizeof(double) + 1 * N * sizeof(int) + 1 * N * sizeof(bool))
    {
        throw std::invalid_argument("Buffer length not long enough.");
    }
#endif

    CUDA_SHARED int link_Space_craft_rec[NLINKS];
    CUDA_SHARED int link_Space_craft_em[NLINKS];

    fill_link_arrays(link_Space_craft_rec, link_Space_craft_em);
    CUDA_SYNC_THREADS;
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);
    get_sky_vectors(&k, &u, &v, params);

    get_tdi_Xf_cached(tdi_channels_arr, params, t_arr, N, bin_i,
                       link_Space_craft_rec, link_Space_craft_em, k, u, v, cache);
    CUDA_SYNC_THREADS;

    // Phase-extract scratch carved out of the caller-allocated buffer.
    double *flip      = (double *) buffer;
    double *pjump     = &flip[N];
    int    *count     = (int *)  &pjump[N];
    bool   *fix_count = (bool *) &count[N];
    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr  = blockDim.x;
#else
    int start = 0;
    int incr  = 1;
#endif
    for (int i = start; i < N; i += incr)
    {
        phi_ref[i] = get_phase_ref(t_arr[i], params, bin_i);
    }
    CUDA_SYNC_THREADS;
    new_extract_amplitude_and_phase(count, fix_count, flip, pjump, N,
                                     &tdi_amp[0],     &tdi_phase[0],
                                     &tdi_channels_arr[0], &phi_ref[0]);
    new_extract_amplitude_and_phase(count, fix_count, flip, pjump, N,
                                     &tdi_amp[N],     &tdi_phase[N],
                                     &tdi_channels_arr[N], &phi_ref[0]);
    new_extract_amplitude_and_phase(count, fix_count, flip, pjump, N,
                                     &tdi_amp[2 * N], &tdi_phase[2 * N],
                                     &tdi_channels_arr[2 * N], &phi_ref[0]);

    double *ph_correct_buffer = &flip[0];
    new_unwrap_phase(ph_correct_buffer, N, &tdi_phase[0]);
    new_unwrap_phase(ph_correct_buffer, N, &tdi_phase[N]);
    new_unwrap_phase(ph_correct_buffer, N, &tdi_phase[2 * N]);
}


CUDA_DEVICE
void LISATDIonTheFly::get_tdi_heterodyned_cached(
    void *buffer, int buffer_length,
    cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref_het,
    double *params, double *t_arr, int N, int bin_i, int nchannels,
    double f0_grid, OrbitsSplineCache *cache)
{
    get_tdi_cached(buffer, buffer_length, tdi_channels_arr,
                    tdi_amp, tdi_phase, phi_ref_het,
                    params, t_arr, N, bin_i, nchannels, cache);
    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr  = blockDim.x;
#else
    int start = 0;
    int incr  = 1;
#endif
    const double two_pi_f0 = 2.0 * M_PI * f0_grid;
    for (int i = start; i < N; i += incr)
    {
        phi_ref_het[i] -= two_pi_f0 * t_arr[i];
    }
    CUDA_SYNC_THREADS;
}


// Raw TDI evaluators: fill ``tdi_channels_arr`` (nchannels * N raw complex
// samples) and ``phi_ref`` (N points, UN-HETERODYNED -- i.e. just
// ``get_phase_ref(t_i)`` straight from the source, NO carrier
// subtraction). Skip the per-channel amplitude/phase extract + unwrap that
// get_tdi[_cached] performs.
//
// IMPORTANT -- why we emit un-het phi_ref here rather than phi_ref_het:
// the downstream ``new_extract_amplitude_and_phase`` consumes phiR via
// ``remainder(phiR, 2*pi)``, which is NOT invariant under shifts by
// 2*pi*f0_grid*t (the carrier offset is not a multiple of 2*pi). If we
// pre-subtracted the carrier here, every per-channel extract would see a
// shifted phiR and produce a Dphi that differs from the OLD direct-path
// get_tdi convention by a per-sample non-2*pi amount. That residual would
// NOT cancel against the downstream ``+ dphi_ref + phi0_chunk`` term --
// it would offset the slow-signal phase, shoving the FFTed energy off the
// snapped chunk-FD bin. Caller is responsible for the carrier subtraction
// when forming dphi_ref for the spline fit (see
// fast_wdm_inner_heterodyne_spline).
//
// Used by the chunked-het spline path so the caller can extract + unwrap
// one channel at a time into single-channel coefficient buffers
// (~6 KB / kernel shared-mem reduction).
CUDA_DEVICE
void LISATDIonTheFly::get_tdi_raw(
    cmplx *tdi_channels_arr, double *phi_ref,
    double *params, double *t_arr, int N, int bin_i, int nchannels)
{
    CUDA_SHARED int link_Space_craft_rec[NLINKS];
    CUDA_SHARED int link_Space_craft_em[NLINKS];

    fill_link_arrays(link_Space_craft_rec, link_Space_craft_em);
    CUDA_SYNC_THREADS;
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);
    get_sky_vectors(&k, &u, &v, params);
    get_tdi_Xf(tdi_channels_arr, params, t_arr, N, bin_i,
                link_Space_craft_rec, link_Space_craft_em, k, u, v);
    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr  = blockDim.x;
#else
    int start = 0;
    int incr  = 1;
#endif
    for (int i = start; i < N; i += incr)
    {
        phi_ref[i] = get_phase_ref(t_arr[i], params, bin_i);
    }
    CUDA_SYNC_THREADS;
}


CUDA_DEVICE
void LISATDIonTheFly::get_tdi_raw_cached(
    cmplx *tdi_channels_arr, double *phi_ref,
    double *params, double *t_arr, int N, int bin_i, int nchannels,
    OrbitsSplineCache *cache)
{
    CUDA_SHARED int link_Space_craft_rec[NLINKS];
    CUDA_SHARED int link_Space_craft_em[NLINKS];

    fill_link_arrays(link_Space_craft_rec, link_Space_craft_em);
    CUDA_SYNC_THREADS;
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);
    get_sky_vectors(&k, &u, &v, params);
    get_tdi_Xf_cached(tdi_channels_arr, params, t_arr, N, bin_i,
                       link_Space_craft_rec, link_Space_craft_em,
                       k, u, v, cache);
    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr  = blockDim.x;
#else
    int start = 0;
    int incr  = 1;
#endif
    for (int i = start; i < N; i += incr)
    {
        phi_ref[i] = get_phase_ref(t_arr[i], params, bin_i);
    }
    CUDA_SYNC_THREADS;
}


CUDA_DEVICE
double LISATDIonTheFly::get_amp(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
#ifdef __CUDACC__
#else
    throw std::invalid_argument("Not implemented.");
#endif

}

CUDA_DEVICE
double LISATDIonTheFly::get_f(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
#ifdef __CUDACC__
#else
    throw std::invalid_argument("Not implemented.");
#endif

}

CUDA_DEVICE
double LISATDIonTheFly::get_fdot(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
#ifdef __CUDACC__
#else
    throw std::invalid_argument("Not implemented.");
#endif

}

CUDA_DEVICE
double LISATDIonTheFly::get_phase(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
#ifdef __CUDACC__
#else
    throw std::invalid_argument("Not implemented.");
#endif
}

CUDA_DEVICE
void LISATDIonTheFly::get_hp_hc(double *hp, double *hc, double t, double *params, double phase_change, int bin_i)
{
    double amp = get_amp(t, params, bin_i);
    double phase = get_phase(t, params, bin_i);
    double psi = params[psi_index];
    double inc = params[inc_index];
    
    double inc_p = (1. + cos(inc) * cos(inc)) / 2.;
    double inc_c = cos(inc);
    
    // *hp = amp * (inc_p * cos(2. * psi) * cos(phase + phase_change) - inc_c * sin(2. * psi) * sin(phase + phase_change));
    // *hc = amp * (-inc_p * sin(2. * psi) * cos(phase + phase_change) - inc_c * cos(2. * psi) * sin(phase + phase_change));  
    double cos2psi = cos(2.0 * psi);
    double sin2psi = sin(2.0 * psi);
    double cosiota = cos(inc);

    double hSp = -cos(phase + phase_change) * amp * (1.0 + cosiota * cosiota);
    double hSc = -sin(phase + phase_change) * 2.0 * amp * cosiota;

    *hp = hSp * cos2psi - hSc * sin2psi;
    *hc = hSp * sin2psi + hSc * cos2psi;
    // printf("FLYIN: %.12e %.12e %.12e %.12e\n", amp, phase, inc, psi);
            

}


CUDA_DEVICE
void LISATDIonTheFly::get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels)
{   

#ifdef __CUDACC__
#else
    if (buffer_length < 2 * N * sizeof(double) + 1 * N * sizeof(int) + 1 * N * sizeof(bool))
    {
        throw std::invalid_argument("Buffer length not long enough.");
    }
#endif

    CUDA_SHARED int link_Space_craft_rec[NLINKS];
    CUDA_SHARED int link_Space_craft_em[NLINKS];
    // CUDA_SHARED int links[NLINKS];
    
    fill_link_arrays(link_Space_craft_rec, link_Space_craft_em);
    CUDA_SYNC_THREADS;
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);
    get_sky_vectors(&k, &u, &v, params);
    get_tdi_Xf(tdi_channels_arr, params, t_arr, N, bin_i, link_Space_craft_rec, link_Space_craft_em, k, u, v);
    CUDA_SYNC_THREADS;
    
    // will get reset inside function
    double *flip = (double*)buffer;
    double *pjump = &flip[N];
    int *count = (int *)&pjump[N];
    bool *fix_count = (bool *)&count[N];

    CUDA_SYNC_THREADS;
#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr = blockDim.x;
#else // __CUDACC__
    int start = 0;
    int incr = 1;
#endif // __CUDACC__
    for (int i = start; i < N; i += incr)
    {
        phi_ref[i] = get_phase_ref(t_arr[i], params, bin_i);
    }
    CUDA_SYNC_THREADS;
    new_extract_amplitude_and_phase(count, fix_count, flip, pjump, N, &tdi_amp[0], &tdi_phase[0], &tdi_channels_arr[0], &phi_ref[0]);
    new_extract_amplitude_and_phase(count, fix_count, flip, pjump, N, &tdi_amp[N], &tdi_phase[N], &tdi_channels_arr[N], &phi_ref[0]);
    new_extract_amplitude_and_phase(count, fix_count, flip, pjump, N, &tdi_amp[2 * N], &tdi_phase[2 * N], &tdi_channels_arr[2 * N], &phi_ref[0]);
    
    // //  FILE *fp1 = fopen("check_phase_before_unwrap.txt", "w");
    // // for (int n = 0; n < N; n += 1)
    // // {
    // //     fprintf(fp1, "%.12e, %.12e, %.12e, %.12e, %.12e, %.12e\n", t_arr[n], Xamp[n], Xphase[n], M[n], Mf[n], phi_ref[n]);
    // //     fflush(fp1);
    // // }
    // // fclose(fp1);
    
    double *ph_correct_buffer = &flip[0];
    new_unwrap_phase(ph_correct_buffer, N, &tdi_phase[0]);
    new_unwrap_phase(ph_correct_buffer, N, &tdi_phase[N]);
    new_unwrap_phase(ph_correct_buffer, N, &tdi_phase[2 * N]);

    // cmplx I(0.0, 1.0);
    // for (int i = 0; i < N; i += 1)
    // {
    //     X[i] = Xamp[i] * gcmplx::exp(-I * Xphase[i]);
    //     Y[i] = Yamp[i] * gcmplx::exp(-I * Yphase[i]);
    //     Z[i] = Zamp[i] * gcmplx::exp(-I * Zphase[i]);
    // }
    // CUDA_SYNC_THREADS;


    // for (int i = 0; i < N; i += 1)
    // {
    //     printf("WEEET2: %d %.12e %.12e %.12e %.12e %.12e\n", i, Xamp[i], Xphase[i], X[i].real(), X[i].imag(), phi_ref[i]);
    // }

    // FILE *fp = fopen("temp_check_amp_phase_22.txt", "w");
    // for (int n = 0; n < N; n += 1)
    // {
    //     fprintf(fp, "%.12e, %.12e, %.12e, %.12e\n", t_arr[n], Xamp[n], Xphase[n], phi_ref[n]);
    //     fflush(fp);
    // }
    // fclose(fp);

    // new_extract_phase(X, phi_ref, N, t_arr);
    // new_extract_phase(Y, phi_ref, N, t_arr);
    // new_extract_phase(Z, phi_ref, N, t_arr);

    // for (int i = 0; i < N; i += 1)
    // {
    //     printf("WEEET3: %d %.12e %.12e %.12e %.12e %.12e\n", i, Xamp[i], Xphase[i], X[i].real(), X[i].imag(), phi_ref[i]);
    // }


    // extract_amplitude_and_phase(flip, pjump, N, Yamp, Yphase, Y, Yf, phi_ref);
    // unwrap_phase(N, Yphase);

    // extract_amplitude_and_phase(flip, pjump, N, Zamp, Zphase, Z, Zf, phi_ref);
    // unwrap_phase(N, Zphase);
}


// CUDA_DEVICE
// void LISATDIonTheFly::get_tdi_Xf(cmplx *X, cmplx *Y, cmplx *Z, double* phi_ref, double *params, double *t_arr, int N, double costh, double phi, double cosi, double psi, int bin_i)
// {

// #ifdef __CUDACC__
//     int start = threadIdx.x;
//     int incr = blockDim.x;
// #else // __CUDACC__
//     int start = 0;
//     int incr = 1;
// #endif // __CUDACC__
//     for (int i = start; i < N; i += incr)
//     {
//         get_tdi_n(X, Y, Z, phi_ref, params, t_arr[i], i, N, costh, phi, cosi, psi, bin_i);
//     }
//     CUDA_SYNC_THREADS;
// }

// CUDA_DEVICE
// void LISATDIonTheFly::get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i)
// {
//     printf("Not Implemented. TODO: best way to do this?");
// }

// CUDA_DEVICE
// double LISATDIonTheFly::get_phase_ref(double t, double *params, int bin_i)
// {
//     printf("Not Implemented. TODO: best way to do this?");
// }


CUDA_DEVICE
void LISATDIonTheFly::unwrap_phase(int N, double *phase)
{
    double u, v, q;
    int i;
    
    // std::cout << "start phase[0]: " << phase[0] << std::endl;
    v = phase[0];
    for(i=0; i<N ;i++)
    {
        u = phase[i];

        // std::cout << "bef u: " << u << " v: " << v << " phase[i]: " << phase[i] << std::endl;
        q = rint(fabs(u-v)/(2. * M_PI));
        if(q > 0.0)
        {
           if(v > u) u += q*2. * M_PI;
           else      u -= q*2. * M_PI;
        }

        v = u;
        phase[i] = u;

        // std::cout << "aft u: " << u << " v: " << v << " q: " << q << " phase[i]: " << phase[i] << std::endl;
        
    }
    // for(i=0; i<N ;i++)
    // {
    //     printf("%d %.12e\n", i, phase[i]);
    // }
}


template<typename T>
CUDA_DEVICE void cumsum(T *sdata, int N)
{
    // cumsum
#ifdef __CUDACC__
    // Specialize BlockScan for a 1D block of 128 threads of type int
    using BlockScan = cub::BlockScan<T, NUM_THREADS_HERE>;

    // Allocate shared memory for BlockScan
    CUDA_SHARED typename BlockScan::TempStorage temp_storage;

    // Obtain input item for each thread
    int tid = threadIdx.x;
    int total_run = 0;
    int index;
    T thread_data;
    while (total_run < N)
    {
        index = total_run + threadIdx.x;
        if (index < N)
        {
            thread_data = sdata[index];
        }
        else
        {
            thread_data = 0.0;
        }

        CUDA_SYNC_THREADS;
        // Collectively compute the block-wide exclusive prefix sum
        // This is the cummulative sum over the width of the block 
        // (sometimes the array is longer which we adjust for with total_run)
        BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
        CUDA_SYNC_THREADS;
    //         // Perform the parallel prefix sum (Blelloch algorithm)
    //     __syncthreads();    
    //     for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) 
    //     {
    //         if ((tid >= stride) && (total_run + tid < N)) 
    //         {
    //             sdata[total_run + tid] += sdata[total_run + tid - stride];
    //         }
    //         __syncthreads(); // Synchronize threads within the block
    //     }
    //     __syncthreads();
        CUDA_SYNC_THREADS;
        if (index < N)
        {
            sdata[index] = thread_data;
        }
        CUDA_SYNC_THREADS;
        // -1 is here because the first element of the next step will be the last element of the previous step
        // this means the first element of the new step is the cummulative sum of the previous step.
        total_run += (NUM_THREADS_HERE - 1);
    }
    // __syncthreads();
    // //     
    // // }
    // // CUDA_SYNC_THREADS;

    /*
    extern __shared__ float temp[];
    // allocated on invocation int thid = threadIdx.x; int offset = 1;

    // build sum in place up the tree
    for (int d = n >> 1; d > 0; d >> = 1)
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0)
    {
        temp[n - 1] = 0;
    } // clear the last element
    __syncthreads();
    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >> = 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
__syncthreads();
    */

    // if (threadIdx.x == 0)
    // {
    //     for (int i = 1; i < N; i += 1)
    //     {
    //         sdata[i] += sdata[i - 1];
    //     }
    // }
    // CUDA_SYNC_THREADS;
#else
    for (int i = 1; i < N; i += 1)
    {
        sdata[i] += sdata[i - 1];
    }
#endif
}


CUDA_DEVICE
void LISATDIonTheFly::new_unwrap_phase(double *ph_correct_buffer, int N, double *phase)
{
    double dd, ddmod;
    double period = 2. * M_PI;
    double interval_high =  period / 2.;
    double interval_low = -interval_high;
    double ph_tmp;
    double discont = period / 2.;
#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr = blockDim.x;
#else // __CUDACC__
    int start = 0;
    int incr = 1;
#endif // __CUDACC__

    for (int i = start; i < N; i += incr)
    {
        ph_correct_buffer[i] = 0.0;
    }

    CUDA_SYNC_THREADS;
    double tmp_remainder;
    // std::cout << "start phase[0]: " << phase[0] << std::endl;
    for(int i= start + 1; i<N ; i += incr)
    {
        dd = phase[i] - phase[i - 1]; 
        tmp_remainder = remainder(dd - interval_low, period);
        while (tmp_remainder < 0.0){tmp_remainder += period;}
        ddmod = tmp_remainder + interval_low;

        if ((ddmod == interval_low) && (dd > 0))
        {
            ddmod = interval_high;
        }
        ph_tmp = ddmod - dd;

        if (abs(dd) < discont)
        {
            ph_tmp = 0.0;
        }
        ph_correct_buffer[i] = ph_tmp;
        // printf("PHASE CORR: %d %e %e %e %e %e\n", i, dd, ddmod, ph_correct_buffer[i], remainder(dd - interval_low, period), interval_low);
    }
    CUDA_SYNC_THREADS;

    cumsum(ph_correct_buffer, N);
    CUDA_SYNC_THREADS;

    double tmp;
    for (int i = start + 1; i < N; i += incr)
    {
        tmp = phase[i] + ph_correct_buffer[i];
        // printf("CHANGE: %d %e %e %e \n", i, phase[i], ph_correct_buffer[i], tmp);
        phase[i] = tmp;

    }
    CUDA_SYNC_THREADS;
//     CHANGE 135 -3.613320762606 6.283185307179587 2.669864544573587
// CHANGE 136 2.66749491221 1.7763568394002505e-15 2.667494912210002
// CHANGE 137 2.662627049648 1.7763568394002505e-15 2.662627049648002
// CHANGE 138 -3.627703057173 6.283185307179588 2.655482250006588
}

CUDA_DEVICE
void LISATDIonTheFly::new_extract_amplitude_and_phase(int *count, bool *fix_count, double *flip, double *pjump, int Ns, double *As, double *Dphi, cmplx *M, double *phiR)
{
    bool is_min;
    double dA1, dA2, dA3, test1, test2;

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr = blockDim.x;
#else // __CUDACC__
    int start = 0;
    int incr = 1;
#endif // __CUDACC__
    for (int i = start; i < Ns; i += incr)
    {
        count[i] = 0;
        pjump[i] = 0.0;
        flip[i] = 1.0;
        fix_count[i] = false;
        As[i] = gcmplx::abs(M[i]);
    }
    CUDA_SYNC_THREADS;
    for (int i = (start + 1); i < Ns - 1; i += incr)
    {   
        is_min = (As[i] < As[i - 1]) && (As[i] < As[i + 1]);

        // printf("CHECKIT2 %d %e %d\n", i, As[i], is_min);
        if (is_min)
        {
            dA1 =  As[i + 1] + As[i - 1] - 2.0*As[i];  //regular second derivative
            dA2 = -As[i + 1] + As[i - 1] - 2.0*As[i];  //second derivative if i+1 first negative value
            dA3 = -As[i + 1] + As[i - 1] + 2.0*As[i];  //second derivative if i first negative value
            test1 = (abs(dA2/dA1) < 0.1);
            test2 = (abs(dA3/dA1) < 0.1);
            // TODO: check this. 
            if (test1)
            {
                // NEED TO BE CAREFUL HERE
                count[i + 1] = 1;
            }
            else if (test2)
            {
                count[i] = 1;
            }
        }
    }

    CUDA_SYNC_THREADS;

    // cumsum
    cumsum(count, Ns);
    CUDA_SYNC_THREADS;

    // Cooperative stride (was ``i += 1`` -- a bug that made every thread
    // re-do the whole length [start, Ns-1) and race on the same shared
    // addresses; harmless on CPU where incr == 1 but huge wasted work on
    // GPU and a memory-consistency risk).
    for (int i = start; i < Ns - 1; i += incr)
    {
        flip[i] = pow(-1., count[i]);
        pjump[i] = count[i] * M_PI;
    }
    CUDA_SYNC_THREADS;

    if (THREAD_ZERO)
    {
        flip[Ns-1]  = flip[Ns-2];
        pjump[Ns-1] = pjump[Ns-2];
    }
    CUDA_SYNC_THREADS;

    double v;
    for(int i=start; i<Ns ; i += incr)
    {
        As[i] = flip[i]*As[i];
        // printf("HUH: %e %e\n", flip[i], As[i]);
        v = remainder(phiR[i], 2 * M_PI);
        Dphi[i] = -atan2(M[i].imag(),M[i].real())+pjump[i]-v;
        // if ((i > 11670))
        // printf("INIT new: %d %e %e %e %e %e %e\n", i, -atan2(Mf[i],M[i]), flip[i], pjump[i], As[i], Dphi[i], v);
    
    }
    CUDA_SYNC_THREADS;
}


CUDA_DEVICE
void LISATDIonTheFly::extract_amplitude_and_phase(double *flip, double *pjump, int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR)
{

    int i;
    double v;
    double dA1, dA2, dA3;
    
    // This catches sign flips in the amplitude. Can't catch flips at either end of array
    flip[0]  = 1.0;
    pjump[0] = 0.0;

    i = 1;
    do
    {
        flip[i] = flip[i-1];
        pjump[i] = pjump[i-1];
        
        //local min
        if((As[i] < As[i-1]) && (As[i] < As[i+1]))
        {
            dA1 =  As[i+1] + As[i-1] - 2.0*As[i];  // regular second derivative
            dA2 = -As[i+1] + As[i-1] - 2.0*As[i];  // second derivative if i+1 first negative value
            dA3 = -As[i+1] + As[i-1] + 2.0*As[i];  // second derivative if i first negative value

            if(fabs(dA2/dA1) < 0.1)
            {
                flip[i+1]  = -1.0*flip[i];
                pjump[i+1] = pjump[i]+M_PI;
                i++; // skip an extra place since i+1 already dealt with
            }
            if(fabs(dA3/dA1) < 0.1)
            {
                flip[i]  = -1.0*flip[i-1];
                pjump[i] = pjump[i-1]+M_PI;
            }
        }
        
        i++;
        
    }while(i < Ns-1);
    
    flip[Ns-1]  = flip[Ns-2];
    pjump[Ns-1] = pjump[Ns-2];
    
    
    for(i=0; i<Ns ;i++)
    {
        As[i] = flip[i]*As[i];
        // printf("HUH: %e %e\n", flip[i], As[i]);
        v = remainder(phiR[i], 2 * M_PI);
        Dphi[i] = -atan2(Mf[i],M[i])+pjump[i]-v;
        // if ((i > 11670))
        // printf("INIT: %d %e %e %e %e %e %e\n", i, -atan2(Mf[i],M[i]), flip[i], pjump[i], As[i], Dphi[i], v);
    
    }
    
}



CUDA_DEVICE
double LISATDIonTheFly::get_phase_ref(double t, double *params, int bin_i)
{   
    // TD is based on t_sc rather than t (t_ssb)
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);
    
    get_sky_vectors(&k, &u, &v, params);
    // reference phase is at spacecraft 1
    Vec x_rec = orbits->get_pos(t, 1);
    double k_dot_x_rec = k.dot(x_rec);
    double t_sc = t - k_dot_x_rec * C_inv;
    double phase_ref = get_phase(t_sc, params, bin_i);
    return phase_ref;
}

// CUDA_DEVICE
// double GBTDIonTheFly::get_phase_ref(double t, double *params, int bin_i)
// {   
//     double f0    = params[f0_index];
// //     if (N_store == NULL)
// //     {
// // #ifdef __CUDACC__
// // #else
// //         throw std::invalid_argument("N_store not set yet.\n");
// // #endif
// //     }
//     double t_diff = t - t_ref;
//     return 2.0 * M_PI * (int(f0 * T) / T) * t_diff;
// }


CUDA_DEVICE
void LISATDIonTheFly::new_extract_phase(cmplx *M, double *phiR, int N, double *t_arr)
{
    cmplx I(0.0, 1.0);

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr = blockDim.x;
#else // __CUDACC__
    int start = 0;
    int incr = 1;
#endif // __CUDACC__
    // FILE *fp = fopen("temp_check.txt", "w");

    for (int n = start; n < N; n += incr)
    {
        // TODO: do we want to do this. We take conj to match N/T
        M[n] = gcmplx::conj(M[n]);
        // fprintf(fp, "%.12e, %.12e, %.12e, %.12e\n", t_arr[n], M[n].real(), M[n].imag(), phiR[n]);
        M[n] *= gcmplx::exp(-I * phiR[n]);
    }
    CUDA_SYNC_THREADS;

    // fclose(fp);
}

int LISATDIonTheFly::get_tdi_buffer_size(int N)
{
    return 2 * N * sizeof(double) + 1 * N * sizeof(bool) + 1 * N * sizeof(int);
}


// Heterodyned-phi_ref variant. See header for semantics; the only
// difference from ``get_tdi`` is the final per-sample subtraction
// ``phi_ref_het[i] = phi_ref[i] - 2*pi*f0_grid*t_arr[i]`` applied
// after unwrap. ``tdi_amp`` and ``tdi_phase`` are unchanged.
CUDA_DEVICE
void LISATDIonTheFly::get_tdi_heterodyned(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double *phi_ref_het, double *params, double *t_arr, int N, int bin_i, int nchannels, double f0_grid)
{
    get_tdi(buffer, buffer_length,
            tdi_channels_arr, tdi_amp, tdi_phase, phi_ref_het,
            params, t_arr, N, bin_i, nchannels);
    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr  = blockDim.x;
#else
    int start = 0;
    int incr  = 1;
#endif
    const double two_pi_f0 = 2.0 * M_PI * f0_grid;
    for (int i = start; i < N; i += incr)
    {
        phi_ref_het[i] -= two_pi_f0 * t_arr[i];
    }
    CUDA_SYNC_THREADS;
}
#endif  // === end LISATDIonTheFly first-block methods moved to LAT ===


CUDA_DEVICE
double GBTDIonTheFly::ucb_phase(double t, double *params)
{
    double f0    = params[f0_index];
    double phi0  = params[phi0_index];
    double fdot  = params[fdot0_index];
    double fddot = params[fddot0_index];
    
    /*
     * LDC phase parameter in key files is
     * -phi0
     */
    double t_diff = t - t_ref;
    return -phi0 + 2 * M_PI *( f0*t_diff + 0.5*fdot*t_diff*t_diff + 1.0/6.0*fddot*t_diff*t_diff*t_diff );
}


CUDA_DEVICE
double GBTDIonTheFly::ucb_amplitude(double t, double *params)
{
    double A0    = params[amplitude_index];
    double f0    = params[f0_index];
    double fdot  = params[fdot0_index];
    double t_diff = t - t_ref;
    return A0 * ( 1.0 + 2.0/3.0*fdot/f0*t_diff);
}

CUDA_DEVICE
double GBTDIonTheFly::ucb_f(double t, double *params)
{
    double f0    = params[f0_index];
    double fdot  = params[fdot0_index];
    double fddot = params[fddot0_index];
    double t_diff = t - t_ref;
    return f0 + fdot * t_diff + 1.0 / 2.0 * fddot * t_diff * t_diff;
}

CUDA_DEVICE
double GBTDIonTheFly::ucb_fdot(double t, double *params)
{
    double fdot  = params[fdot0_index];
    double fddot = params[fddot0_index];
    double t_diff = t - t_ref;
    return fdot + fddot * t_diff;
}

// CUDA_DEVICE
// void GBTDIonTheFly::get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i)
// {
//     // params are already referenced before this. 
//     for(int n=0; n<N; n++)
//     {
//         phase[n] = ucb_phase(t[n],params);
//         amp[n]   = ucb_amplitude(t[n],params);
//     }
// }


CUDA_DEVICE
double GBTDIonTheFly::get_phase(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
    return ucb_phase(t, params);
}

CUDA_DEVICE
double GBTDIonTheFly::get_amp(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
    return ucb_amplitude(t, params);
}

CUDA_DEVICE
double GBTDIonTheFly::get_f(double t, double *params, int bin_i)
{
    return ucb_f(t, params);
}

CUDA_DEVICE
double GBTDIonTheFly::get_fdot(double t, double *params, int bin_i)
{
    return ucb_fdot(t, params);
}

CUDA_DEVICE
GBTDIonTheFly::~GBTDIonTheFly()
{
    return;
}

// CUDA_DEVICE
// void LISATDIonTheFly::print_orbits_tdi()
// {
//     printf("inside print\n");
//     printf("ahead of check\n");
//     if (orbits == NULL)
//     {
//         throw std::invalid_argument("Need to add orbital information.\n");
//     }
//     printf("orbits inside: %e\n", orbits->armlength);
    
//     if (this->tdi_config == NULL)
//     {
//         throw std::invalid_argument("Need to add tdi config.\n");
//     }
//     printf("tdi_config inside: %d\n", this->tdi_config->num_channels);
// }

// === LISATDIonTheFly::run_wave_tdi moved to LAT at Phase 3L.5 ===
#if 0
CUDA_DEVICE
void LISATDIonTheFly::run_wave_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    N_store = N;
    // printf("orbits inside: %e", orbits->armlength);
    if (orbits == NULL)
    {
#ifdef __CUDACC__
#else
        throw std::invalid_argument("Need to add orbital information.\n");
#endif
    }

    if (this->tdi_config == NULL)
    {
#ifdef __CUDACC__
#else
        throw std::invalid_argument("Need to add tdi config2.\n");
#endif
    }

#ifdef __CUDACC__
    int start = blockIdx.x;
    int increment = gridDim.x;

    int start2 = threadIdx.x;
    int increment2 = blockDim.x;
#else
    int start = 0;
    int increment = 1;

    int start2 = 0;
    int increment2 = 1;
#endif

// TODO: make this better?
#ifdef __CUDACC__
#else
    if (n_params > N_PARAMS_MAX)
    {
        throw std::invalid_argument("n_params is too long, need to recompile and increase N_PARAMS_MAX.");
    }
#endif
    CUDA_SHARED double params_here[N_PARAMS_MAX];  // TODO: maybe shared? only if registers are filled up

     // TODO: CHECK THIS!!
    for (int bin_i = start; bin_i < num_bin; bin_i += increment)
    {
        CUDA_SYNC_THREADS;
        // read params into faster memory for gpu / cpu does not matter really
        for (int i = start2; i < n_params; i += increment2)
        {
            params_here[i] = params[bin_i * n_params + i];
        }
        CUDA_SYNC_THREADS;

        double *t_here = &t_arr[bin_i * N];
        
        get_tdi(
            buffer, buffer_length,
            &tdi_channels_arr[bin_i * nchannels * N], 
            &tdi_amp[bin_i * nchannels * N], &tdi_phase[bin_i * nchannels * N],
            &phi_ref[bin_i * N],
            params_here, t_here, N, bin_i, nchannels);
        CUDA_SYNC_THREADS;
    }
    return;
}
#endif  // === end LISATDIonTheFly::run_wave_tdi moved to LAT ===



#ifdef __CUDACC__
CUDA_KERNEL
void gb_run_wave_tdi_kernel(GBTDIonTheFly *tdi_on_fly, int buffer_length, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    extern CUDA_SHARED char shared_mem[];
    void *buffer = (void*)shared_mem;

    GBTDIonTheFly tdi_on_fly_here(tdi_on_fly->orbits, tdi_on_fly->tdi_config, tdi_on_fly->T, tdi_on_fly->t_ref);
    tdi_on_fly_here.run_wave_tdi(buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
}
#endif 

void gb_run_wave_tdi_wrap(GBTDIonTheFly *tdi_on_fly, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    // printf("CHECK44\n");
#ifdef __CUDACC__
    // printf("CHECK55\n");
    GBTDIonTheFly *gb_here = new GBTDIonTheFly(tdi_on_fly->orbits, tdi_on_fly->tdi_config, tdi_on_fly->T, tdi_on_fly->t_ref);
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, tdi_on_fly->orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_on_fly->tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    gb_here->orbits = d_orbits;
    gb_here->tdi_config = d_tdi_config;

    GBTDIonTheFly *d_gb_here;
    cudaMalloc(&d_gb_here, sizeof(GBTDIonTheFly));
    gpuErrchk(cudaMemcpy(d_gb_here, gb_here, sizeof(GBTDIonTheFly), cudaMemcpyHostToDevice));

    int buffer_length = tdi_on_fly->get_gb_buffer_size(N); 
    printf("%d\n", buffer_length);
    gb_run_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, buffer_length>>>(d_gb_here, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_gb_here));
    delete gb_here;
    // printf("CHECK66\n");
#else

    // make buffer 
    int buffer_length = tdi_on_fly->get_gb_buffer_size(N);
    char *buffer = new char[buffer_length];
    tdi_on_fly->run_wave_tdi((void*)buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
    delete[] buffer;
#endif
}

int GBTDIonTheFly::get_gb_buffer_size(int N)
{
    return N * sizeof(double) + get_tdi_buffer_size(N);
}

int GBTDIonTheFly::get_gb_fd_buffer_size(int N, int nchannels)
{
    // Shared-memory budget per source for the heterodyne FD kernel:
    //   params_here[N_PARAMS_MAX]                        N_PARAMS_MAX * 8
    //   t_arr_local[N]                                              N * 8
    //   tdi_channels_arr[nchannels * N]  (cmplx, FFT)    nchannels * N * 16
    //   tdi_amp[nchannels * N]                           nchannels * N * 8
    //   tdi_phase[nchannels * N]                         nchannels * N * 8
    //   phi_ref[N]                                                  N * 8
    //   get_tdi scratch (flip, pjump, count, fix_count)            21 * N
    return (int) (
          N_PARAMS_MAX * sizeof(double)
        + (size_t) N * sizeof(double)
        + (size_t) nchannels * (size_t) N * sizeof(cmplx)
        + 2 * (size_t) nchannels * (size_t) N * sizeof(double)
        + (size_t) N * sizeof(double)
        + (size_t) get_tdi_buffer_size(N)
    );
}

// ---------------------------------------------------------------------------
// Stellar-origin black-hole binary (SOBBH) TDI on the fly
// ---------------------------------------------------------------------------
//
// Ports the post-Newtonian intrinsic-quantity expressions from
// sobbh_intrinsic_Ladeeda.cpp into the LISATDIonTheFly framework. Only style
// has been adapted (CUDA_DEVICE decorators, no std:: qualifiers, no
// std::vector buffers); the PN coefficients themselves are bit-identical to
// the prototype.
//
// Conversions: m1, m2 in solar masses -> seconds via MTSUN_SOBBH;
// distance in parsecs -> seconds via PARSEC_SOBBH / C_SI. The amplitude
// returned by get_amp is the GW-quadrupole "scalar" piece a = 2 M eta x / D
// (positive); the inclination factor (1+cos^2 iota) for plus, 2 cos iota for
// cross, plus an overall minus sign and the cos/sin split, all live in
// LISATDIonTheFly::get_hp_hc and match the existing GB convention. The phase
// returned is the GW phase 2 * (phi_c - phase_fn(x)), with the factor of 2
// already folded in to match get_hp_hc's cos(phase) / sin(phase) pattern.

#define EULER_GAMMA_SOBBH 0.57721566490153286060
#define MTSUN_SOBBH      4.9254909476412675e-06
#define PARSEC_SOBBH     3.085677581491367e16

CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_phase_fn(double x, double sigma, double delta, double eta, double s)
{
    double x15 = pow(x, 1.5);
    double x20 = x * x;
    double x25 = pow(x, 2.5);
    double x30 = x * x * x;
    double x35 = pow(x, 3.5);
    double logx = log(x);

    double Phi_0_minus_phi =
        (
            1.0
            + x * (3.685515873015873 + (55.0 * eta) / 12.0)
            + x15 * (-10.0 * M_PI + (235.0 * s) / 6.0 + (125.0 * delta * sigma) / 8.0)
            + x20 * (
                15.051576475497606
                - 100.0 * s * s
                + (3085.0 * eta * eta) / 144.0
                - 100.0 * s * delta * sigma
                - (405.0 * sigma * sigma) / 16.0
                + eta * (26.92956349206349 + 100.0 * sigma * sigma)
            )
            + x35 * (
                (-9018232555.0 * s) / 6.096384e6
                + (125925.0 * s * s * s) / 224.0
                - (170978035.0 * delta * sigma) / 387072.0
                + (379805.0 * s * s * delta * sigma) / 448.0
                + (182755.0 * s * sigma * sigma) / 448.0
                + (1315.0 * delta * sigma * sigma * sigma) / 21.0
                + M_PI * (
                    37.93888721576594
                    - 200.0 * s * s
                    - 200.0 * s * delta * sigma
                    - (815.0 * sigma * sigma) / 16.0
                )
                + eta * eta * (
                    (-74045.0 * M_PI) / 6048.0
                    + (835.0 * s) / 288.0
                    + (7015.0 * delta * sigma) / 1152.0
                    + (285.0 * s * sigma * sigma) / 8.0
                    + (95.0 * delta * sigma * sigma * sigma) / 16.0
                )
                + eta * (
                    (3329545.0 * s) / 3024.0
                    - (95.0 * s * s * s) / 8.0
                    + (2909765.0 * delta * sigma) / 5376.0
                    - (285.0 * s * s * delta * sigma) / 16.0
                    - (385825.0 * s * sigma * sigma) / 224.0
                    - (130615.0 * delta * sigma * sigma * sigma) / 448.0
                    + M_PI * (31.292576058201057 + 200.0 * sigma * sigma)
                )
            )
            + x30 * (
                657.6504345051205
                - (1712.0 * EULER_GAMMA_SOBBH) / 21.0
                - (160.0 * M_PI * M_PI) / 3.0
                + (7915.0 * s * s) / 63.0
                - (127825.0 * eta * eta * eta) / 5184.0
                + (2645.0 * s * delta * sigma) / 56.0
                - (1645.0 * sigma * sigma) / 128.0
                + M_PI * ((940.0 * s) / 3.0 + (745.0 * delta * sigma) / 6.0)
                + eta * eta * (11.003327546296296 - 120.0 * sigma * sigma)
                + eta * (
                    -1290.7459270118156
                    + (2255.0 * M_PI * M_PI) / 48.0
                    + 120.0 * s * s
                    + 120.0 * s * delta * sigma
                    + (5875.0 * sigma * sigma) / 112.0
                )
                - (3424.0 * log(2.0)) / 21.0
                - (856.0 * logx) / 21.0
            )
            + x25 * (
                (38645.0 * M_PI) / 1344.0
                - (555605.0 * s) / 2016.0
                - (15.0 * s * s * s) / 4.0
                - (41745.0 * delta * sigma) / 448.0
                - (45.0 * s * s * delta * sigma) / 8.0
                - (45.0 * s * sigma * sigma) / 8.0
                - (15.0 * delta * sigma * sigma * sigma) / 8.0
                + eta * (
                    (-65.0 * M_PI) / 16.0
                    - (45.0 * s) / 8.0
                    + (5.0 * delta * sigma) / 2.0
                    + (45.0 * s * sigma * sigma) / 4.0
                    + (15.0 * delta * sigma * sigma * sigma) / 8.0
                )
            ) * logx
        ) / (32.0 * x25 * eta);

    return Phi_0_minus_phi;
}


CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_time_to_merger_fn(double x, double sigma, double delta, double eta, double s)
{
    double x15 = pow(x, 1.5);
    double x20 = x * x;
    double x25 = pow(x, 2.5);
    double x30 = x * x * x;
    double x40 = x20 * x20;
    double logx = log(x);

    double tc =
        (1.0 / eta) *
        (
            5.0 / (256.0 * x40)
            + (5.0 * (743.0 + 924.0 * eta)) / (64512.0 * x30)
            + (-48.0 * M_PI + 188.0 * s + 75.0 * delta * sigma) / (384.0 * x25)
            + (
                5.0 * (
                    -23187.0 * M_PI
                    + 221738.0 * s
                    + 3276.0 * M_PI * eta
                    + 5544.0 * s * eta
                    + 75141.0 * delta * sigma
                    - 1512.0 * delta * eta * sigma
                )
            ) / (193536.0 * x15)
            - (
                5.0 * (
                    -3058673.0
                    + 20321280.0 * s * s
                    - 5472432.0 * eta
                    - 4353552.0 * eta * eta
                    + 20321280.0 * s * delta * sigma
                    + 5143824.0 * sigma * sigma
                    - 20321280.0 * eta * sigma * sigma
                )
            ) / (1.30056192e8 * x20)
            + (
                -10052469856691.0
                + 1530761379840.0 * EULER_GAMMA_SOBBH
                + 1001432678400.0 * M_PI * M_PI
                - 5883416985600.0 * M_PI * s
                - 2359029657600.0 * s * s
                + 24236159077900.0 * eta
                - 882121363200.0 * M_PI * M_PI * eta
                - 2253223526400.0 * s * s * eta
                - 206607970800.0 * eta * eta
                + 462992376000.0 * eta * eta * eta
                - 2331460454400.0 * M_PI * delta * sigma
                - 886871462400.0 * s * delta * sigma
                - 2253223526400.0 * s * delta * eta * sigma
                - 492159175200.0 * sigma * sigma
                + 733471200000.0 * delta * delta * sigma * sigma
                + 1948937760000.0 * eta * sigma * sigma
                + 2253223526400.0 * eta * eta * sigma * sigma
                + 658084331520.0 * log(2.0)
                + 1201719214080.0 * log(4.0)
                + 765380689920.0 * logx
            ) / (1.20171921408e12 * x)
        );

    return tc;
}


CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_tau_to_x_fn(double tau, double sigma, double delta, double eta, double s)
{
    double tau_inv = 1.0 / tau;
    double tau_qrt = pow(tau, 0.25);
    double tau_inv_qrt = 1.0 / tau_qrt;
    double tau_inv_3_8 = pow(tau, -0.375);
    double tau_inv_5_8 = pow(tau, -0.625);
    double tau_inv_7_8 = pow(tau, -0.875);
    double tau_inv_3_4 = pow(tau, -0.75);
    double tau_inv_half = 1.0 / sqrt(tau);
    double logtau = log(tau);

    double x =
        (
            1.0
            + (
                (-113868647.0 * M_PI) / 4.3352064e8
                + (24532268147.0 * s) / 2.60112384e9
                + (21.0 * M_PI * s * s) / 16.0
                - (755.0 * s * s * s) / 192.0
                + (281190779.0 * delta * sigma) / 9.9090432e7
                + (21.0 * M_PI * s * delta * sigma) / 16.0
                - (4499.0 * s * s * delta * sigma) / 768.0
                + (1711.0 * M_PI * sigma * sigma) / 5120.0
                - (33929.0 * s * sigma * sigma) / 20480.0
                - (325.0 * s * delta * delta * sigma * sigma) / 256.0
                - (24007.0 * delta * sigma * sigma * sigma) / 49152.0
                + eta * eta * (
                    (294941.0 * M_PI) / 3.87072e6
                    + (3641.0 * s) / 122880.0
                    - (6169.0 * delta * sigma) / 294912.0
                )
                + eta * (
                    (-31821.0 * M_PI) / 143360.0
                    - (33704749.0 * s) / 5.16096e6
                    - (5756657.0 * delta * sigma) / 1.769472e6
                    - (21.0 * M_PI * sigma * sigma) / 16.0
                    + (1259.0 * s * sigma * sigma) / 192.0
                    + (493.0 * delta * sigma * sigma * sigma) / 256.0
                )
            ) * tau_inv_7_8
            + (
                (-11891.0 * M_PI) / 53760.0
                + (357923.0 * s) / 161280.0
                + (96473.0 * delta * sigma) / 129024.0
                + eta * (
                    (109.0 * M_PI) / 1920.0
                    - (187.0 * s) / 5760.0
                    - (79.0 * delta * sigma) / 1536.0
                )
            ) * tau_inv_5_8
            + (
                0.0770935689090451
                - (5.0 * s * s) / 8.0
                + (31.0 * eta * eta) / 288.0
                - (5.0 * s * delta * sigma) / 8.0
                - (81.0 * sigma * sigma) / 512.0
                + eta * (0.12607990244708994 + (5.0 * sigma * sigma) / 8.0)
            ) * tau_inv_half
            + (-0.2 * M_PI + (47.0 * s) / 60.0 + (5.0 * delta * sigma) / 16.0) * tau_inv_3_8
            + (0.18427579365079366 + (11.0 * eta) / 48.0) * tau_inv_qrt
            + (
                -1.6730147506856445
                + (107.0 * EULER_GAMMA_SOBBH) / 420.0
                + M_PI * M_PI / 6.0
                - (47.0 * M_PI * s) / 48.0
                - (1583.0 * s * s) / 4032.0
                + (25565.0 * eta * eta * eta) / 331776.0
                - (149.0 * M_PI * delta * sigma) / 384.0
                - (529.0 * s * delta * sigma) / 3584.0
                - (671.0 * sigma * sigma) / 8192.0
                + (125.0 * delta * delta * sigma * sigma) / 1024.0
                + eta * (
                    4.033581021911924
                    - (451.0 * M_PI * M_PI) / 3072.0
                    - (3.0 * s * s) / 8.0
                    - (3.0 * s * delta * sigma) / 8.0
                    + (2325.0 * sigma * sigma) / 7168.0
                )
                + eta * eta * (-0.03438539858217592 + (3.0 * sigma * sigma) / 8.0)
                + (107.0 * log(2.0)) / 420.0
                - (107.0 * logtau) / 3360.0
            ) * tau_inv_3_4
        ) / (4.0 * tau_qrt);

    return x;
}


// Computes (M, eta, sigma, delta, s, tc, tau, x) on the fly from the
// per-source parameter vector. Pre-merger only; post-merger callers must
// short-circuit via the caller's t < tc check (the per-sample wrappers below
// return amp == 0 and phase == 0 for t >= tc so the projection still
// produces a finite zero-amplitude template).
CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_amplitude(double t, double *params)
{
    double m1_sec = params[m1_index] * MTSUN_SOBBH;
    double m2_sec = params[m2_index] * MTSUN_SOBBH;
    double s1     = params[s1_index];
    double s2     = params[s2_index];
    double D_pc   = params[distance_index];
    double f_low  = params[f_low_index];

    double M = m1_sec + m2_sec;
    double eta   = (m1_sec * m2_sec) / (M * M);
    double sigma = (m2_sec * s2 - m1_sec * s1) / M;
    double s_pn  = (m1_sec * m1_sec * s1 + m2_sec * m2_sec * s2) / (M * M);
    double delta = (m1_sec - m2_sec) / M;

    double v0 = pow(M_PI * M * f_low, 1.0 / 3.0);
    double x0 = v0 * v0;
    double tc = sobbh_time_to_merger_fn(x0, sigma, delta, eta, s_pn) * M;

    if (t >= tc)
    {
        return 0.0;
    }

    double tau = eta * (tc - t) / (5.0 * M);
    double x   = sobbh_tau_to_x_fn(tau, sigma, delta, eta, s_pn);

    double D_sec = D_pc * PARSEC_SOBBH / C_SI;

    // Positive scalar amplitude: get_hp_hc folds in the overall minus sign
    // and the cos(2 phi) / sin(2 phi) split, matching the GB convention.
    return 2.0 * M * eta * x / D_sec;
}

CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_phase(double t, double *params)
{
    double m1_sec = params[m1_index] * MTSUN_SOBBH;
    double m2_sec = params[m2_index] * MTSUN_SOBBH;
    double s1     = params[s1_index];
    double s2     = params[s2_index];
    double f_low  = params[f_low_index];
    double phi_c  = params[phi_c_index];

    double M = m1_sec + m2_sec;
    double eta   = (m1_sec * m2_sec) / (M * M);
    double sigma = (m2_sec * s2 - m1_sec * s1) / M;
    double s_pn  = (m1_sec * m1_sec * s1 + m2_sec * m2_sec * s2) / (M * M);
    double delta = (m1_sec - m2_sec) / M;

    double v0 = pow(M_PI * M * f_low, 1.0 / 3.0);
    double x0 = v0 * v0;
    double tc = sobbh_time_to_merger_fn(x0, sigma, delta, eta, s_pn) * M;

    if (t >= tc)
    {
        return 0.0;
    }

    double tau = eta * (tc - t) / (5.0 * M);
    double x   = sobbh_tau_to_x_fn(tau, sigma, delta, eta, s_pn);

    double phi_orbital = phi_c - sobbh_phase_fn(x, sigma, delta, eta, s_pn);
    return 2.0 * phi_orbital;
}

CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_f(double t, double *params)
{
    // GW (quadrupolar) frequency: f_GW = 2 * f_orbital = 2 * x^(3/2) / (pi M).
    double m1_sec = params[m1_index] * MTSUN_SOBBH;
    double m2_sec = params[m2_index] * MTSUN_SOBBH;
    double s1     = params[s1_index];
    double s2     = params[s2_index];
    double f_low  = params[f_low_index];

    double M = m1_sec + m2_sec;
    double eta   = (m1_sec * m2_sec) / (M * M);
    double sigma = (m2_sec * s2 - m1_sec * s1) / M;
    double s_pn  = (m1_sec * m1_sec * s1 + m2_sec * m2_sec * s2) / (M * M);
    double delta = (m1_sec - m2_sec) / M;

    double v0 = pow(M_PI * M * f_low, 1.0 / 3.0);
    double x0 = v0 * v0;
    double tc = sobbh_time_to_merger_fn(x0, sigma, delta, eta, s_pn) * M;

    if (t >= tc)
    {
        return 0.0;
    }

    double tau = eta * (tc - t) / (5.0 * M);
    double x   = sobbh_tau_to_x_fn(tau, sigma, delta, eta, s_pn);
    return 2.0 * pow(x, 1.5) / (M_PI * M);
}

CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_fdot(double t, double *params)
{
    // Centered finite difference around t (small window scaled by chirp time).
    // The PN inspiral is smooth enough that the leading O(h^2) truncation is
    // well below the rest of the SOBBH pipeline error; tighter analytic
    // expressions can replace this if a downstream consumer ever needs them.
    double m1_sec = params[m1_index] * MTSUN_SOBBH;
    double m2_sec = params[m2_index] * MTSUN_SOBBH;
    double M = m1_sec + m2_sec;
    double dt_fd = 10.0 * M;
    if (dt_fd <= 0.0) dt_fd = 1.0;

    double f_plus  = sobbh_f(t + dt_fd, params);
    double f_minus = sobbh_f(t - dt_fd, params);
    return (f_plus - f_minus) / (2.0 * dt_fd);
}

CUDA_DEVICE
double SOBBHTDIonTheFly::get_amp(double t, double *params, int bin_i)
{
    return sobbh_amplitude(t, params);
}

CUDA_DEVICE
double SOBBHTDIonTheFly::get_phase(double t, double *params, int bin_i)
{
    return sobbh_phase(t, params);
}

CUDA_DEVICE
double SOBBHTDIonTheFly::get_f(double t, double *params, int bin_i)
{
    return sobbh_f(t, params);
}

CUDA_DEVICE
double SOBBHTDIonTheFly::get_fdot(double t, double *params, int bin_i)
{
    return sobbh_fdot(t, params);
}

CUDA_DEVICE
SOBBHTDIonTheFly::~SOBBHTDIonTheFly()
{
    return;
}

int SOBBHTDIonTheFly::get_sobbh_buffer_size(int N)
{
    return N * sizeof(double) + get_tdi_buffer_size(N);
}


#ifdef __CUDACC__
CUDA_KERNEL
void sobbh_run_wave_tdi_kernel(SOBBHTDIonTheFly *tdi_on_fly, int buffer_length, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    extern CUDA_SHARED char shared_mem[];
    void *buffer = (void*)shared_mem;

    SOBBHTDIonTheFly tdi_on_fly_here(tdi_on_fly->orbits, tdi_on_fly->tdi_config, tdi_on_fly->T, tdi_on_fly->t_ref);
    tdi_on_fly_here.run_wave_tdi(buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
}
#endif

void sobbh_run_wave_tdi_wrap(SOBBHTDIonTheFly *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
#ifdef __CUDACC__
    SOBBHTDIonTheFly *sobbh_here = new SOBBHTDIonTheFly(tdi_on_fly->orbits, tdi_on_fly->tdi_config, tdi_on_fly->T, tdi_on_fly->t_ref);
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, tdi_on_fly->orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_on_fly->tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    sobbh_here->orbits = d_orbits;
    sobbh_here->tdi_config = d_tdi_config;

    SOBBHTDIonTheFly *d_sobbh_here;
    cudaMalloc(&d_sobbh_here, sizeof(SOBBHTDIonTheFly));
    gpuErrchk(cudaMemcpy(d_sobbh_here, sobbh_here, sizeof(SOBBHTDIonTheFly), cudaMemcpyHostToDevice));

    int buffer_length = tdi_on_fly->get_sobbh_buffer_size(N);
    sobbh_run_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, buffer_length>>>(d_sobbh_here, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_sobbh_here));
    delete sobbh_here;
#else
    int buffer_length = tdi_on_fly->get_sobbh_buffer_size(N);
    char *buffer = new char[buffer_length];
    tdi_on_fly->run_wave_tdi((void*)buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
    delete[] buffer;
#endif
}


// ---------------------------------------------------------------------------
// Heterodyne FD GB kernel
// ---------------------------------------------------------------------------
//
// Per-source block; threads cooperate over the N_sparse time samples and over
// FFT butterflies.  All three channels live in shared memory simultaneously
// so cross-channel (XYZ -> AET, etc.) post-processing can happen before any
// global-memory write.
//
// Algorithm (per source, all in shared memory):
//   1. Re-run the existing get_tdi() on the sparse grid -> tdi_amp[c,n],
//      tdi_phase[c,n], phi_ref[n], tdi_channels_arr[c,n].
//   2. Overwrite tdi_channels_arr[c,n] with the slow positive-frequency
//      signal  s_c(tau_n) = A_c(tau_n) *
//                            exp(+i*(phi_c(tau_n) + phi_ref(tau_n)
//                                    - 2*pi*f0_grid * tau_n)).
//   3. In-place radix-2 Cooley-Tukey FFT per channel on s_c[0..N-1] using
//      cooperative bit-reversal + log2(N) butterfly passes.  Twiddles are
//      computed on the fly with sin/cos (full double precision).
//   4. Multiply by 0.5 * dt_sparse (the 1/2 from x = Re[z]) and write the
//      (num_bin, nchannels, N_sparse) complex result, along with k_f0[bin_i]
//      and f0_grid[bin_i] (dense rfft bin and snapped carrier).

CUDA_DEVICE inline int gbfd_log2_int(int n)
{
    int r = 0;
    while ((n >>= 1) != 0) ++r;
    return r;
}

CUDA_DEVICE inline int gbfd_bit_reverse(int x, int log2n)
{
    int r = 0;
    for (int i = 0; i < log2n; ++i)
    {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

CUDA_DEVICE
void gbfd_radix2_fft_inplace(cmplx *a, int N, int log2N)
{
    // Cooley-Tukey decimation-in-time, in-place, double precision.  No GSL or
    // other library; permissive MIT-style hand roll.  Cooperative across the
    // threads of the block.

    // Bit-reversal permutation
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        int r = gbfd_bit_reverse(n, log2N);
        if (r > n)
        {
            cmplx t = a[n];
            a[n] = a[r];
            a[r] = t;
        }
    }
    CUDA_SYNC_THREADS;

    // log2(N) butterfly passes
    for (int s = 1; s <= log2N; ++s)
    {
        int m  = 1 << s;
        int mh = m >> 1;
        double base = -2.0 * M_PI / (double) m;  // forward FFT sign
        for (int k = THREAD_START_X; k < (N >> 1); k += BLOCK_INCR_X)
        {
            int g  = k / mh;          // butterfly group
            int j  = k - g * mh;      // position within group
            int i0 = g * m + j;
            int i1 = i0 + mh;
            double th = base * (double) j;
            cmplx w(cos(th), sin(th));
            cmplx u = a[i0];
            cmplx v = w * a[i1];
            a[i0] = u + v;
            a[i1] = u - v;
        }
        CUDA_SYNC_THREADS;
    }
}

// Build the heterodyne FD for one source into the shared-memory buffer.
//
// Side effects after return:
//   tdi_chan[c*N + n] holds  0.5 * dt_sparse * FFT[s_c][n]  (complex),
//   *kf0_out is the dense rfft bin closest to f0,
//   *f0g_out is the snapped carrier f0_grid = *kf0_out * df.
//
// All three (nchannels) channels are resident in shared memory at return,
// in FFT-order, ready for the inner-product / accumulator step.
//
// The shared-mem layout is exactly the one `get_gb_fd_buffer_size` reserves.
// `tdi_chan_out`, if non-NULL, also receives a pointer to the per-channel
// heterodyne FD slab within shared (size = nchannels * N complex).
// tukey_alpha: scipy.signal.windows.tukey alpha applied to the slow signal
// before the in-place FFT. 0.0 = rectangular (no taper), match
// FAST_WDM_TUKEY_ALPHA_HET_NARROW / _HET_WIDE (0.05 / 0.01) to mirror the
// dense rfft(Tukey*td) convention. The same taper formula is used as in
// the chunked-het sparse FD path (TDIonTheFly.cu:2074-2098) so cross-path
// inner products line up at FP precision.
CUDA_DEVICE
void gbfd_build_one_source(GBTDIonTheFly *tof, void *shared_mem,
                           double *params_in, double t_start, double Tobs,
                           int N, int nchannels, int n_params, int bin_i,
                           int log2N,
                           cmplx **tdi_chan_out,
                           int *kf0_out, double *f0g_out, double *dts_out,
                           double tukey_alpha)
{
    // ---- carve up shared memory ------------------------------------------
    char *cur = (char*) shared_mem;

    double *params_here = (double*) cur;
    cur += N_PARAMS_MAX * sizeof(double);

    double *t_arr_local = (double*) cur;
    cur += (size_t) N * sizeof(double);

    cmplx *tdi_chan = (cmplx*) cur;             // also slow + FFT buffer
    cur += (size_t) nchannels * N * sizeof(cmplx);

    double *tdi_amp = (double*) cur;
    cur += (size_t) nchannels * N * sizeof(double);

    double *tdi_phase = (double*) cur;
    cur += (size_t) nchannels * N * sizeof(double);

    double *phi_ref = (double*) cur;
    cur += (size_t) N * sizeof(double);

    void *get_tdi_scratch = (void*) cur;
    int   get_tdi_scratch_len = tof->get_tdi_buffer_size(N);

    // ---- broadcast params into shared ------------------------------------
    for (int i = THREAD_START_X; i < n_params; i += BLOCK_INCR_X)
        params_here[i] = params_in[bin_i * n_params + i];
    CUDA_SYNC_THREADS;

    const double f0   = params_here[tof->f0_index];
    const double df   = 1.0 / Tobs;
    const int    kf0  = (int) llround(f0 / df);
    const double f0g  = (double) kf0 * df;
    const double dts  = Tobs / (double) N;

    // ---- build sparse absolute-time array in shared ----------------------
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
        t_arr_local[n] = t_start + (double) n * dts;
    CUDA_SYNC_THREADS;

    // ---- call existing get_tdi to fill tdi_chan / tdi_amp / tdi_phase /
    //      phi_ref from the sparse t_arr_local
    tof->get_tdi(get_tdi_scratch, get_tdi_scratch_len,
                 tdi_chan, tdi_amp, tdi_phase, phi_ref,
                 params_here, t_arr_local, N, bin_i, nchannels);

    // ---- build slow positive-freq complex signal in-place over tdi_chan --
    // Tukey window factored in-line: cosine taper on the first / last
    // alpha/2 fraction of the N sparse samples (rectangular middle). Matches
    // scipy.signal.windows.tukey(N, alpha) sample-by-sample so the sparse FD
    // matches the dense rfft(Tukey*td) inner product. alpha=0 -> no taper.
    const double n_taper_fd = 0.5 * tukey_alpha * (double) (N - 1);
    const double dlast_fd   = (double) (N - 1);
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        const double tau     = (double) n * dts;
        const double carrier = 2.0 * M_PI * f0g * tau;
        const double phref   = phi_ref[n];
        double w = 1.0;
        if (tukey_alpha > 0.0 && n_taper_fd > 0.0) {
            const double di = (double) n;
            if (di < n_taper_fd) {
                const double xn = di / n_taper_fd;
                w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
            } else if (di > dlast_fd - n_taper_fd) {
                const double xn = (dlast_fd - di) / n_taper_fd;
                w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
            }
        }
        for (int c = 0; c < nchannels; ++c)
        {
            const double th = tdi_phase[c * N + n] + phref - carrier;
            tdi_chan[c * N + n] =
                gcmplx::polar(tdi_amp[c * N + n] * w, th);  // +i sign
        }
    }
    CUDA_SYNC_THREADS;

    // ---- NaN scrub. Any non-finite sample left over from a singular
    //      response geometry (e.g. the (1-k.n)->0 wave-axis-vs-arm
    //      alignment for one TDI link at one sparse-time sample) would
    //      otherwise be spread across the entire band by the in-place
    //      FFT below, NaN-ing 4096 contiguous output bins. Zero those
    //      samples so the FFT stays finite; we lose at most a handful of
    //      O(N_sparse^-1) sparse samples at the singular locus.
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        for (int c = 0; c < nchannels; ++c)
        {
            cmplx v = tdi_chan[c * N + n];
            if (!isfinite(v.real()) || !isfinite(v.imag()))
            {
                tdi_chan[c * N + n] = cmplx(0.0, 0.0);
            }
        }
    }
    CUDA_SYNC_THREADS;

    // ---- in-place radix-2 FFT, per channel ------------------------------
    for (int c = 0; c < nchannels; ++c)
    {
        gbfd_radix2_fft_inplace(tdi_chan + (size_t) c * N, N, log2N);
        CUDA_SYNC_THREADS;
    }

    // ---- absorb the 1/2 * dt_sparse scale here so callers can use the
    //      values directly as the heterodyne FD piece ---------------------
    const double scale = 0.5 * dts;
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        for (int c = 0; c < nchannels; ++c)
        {
            cmplx v = tdi_chan[c * N + n];
            tdi_chan[c * N + n] = cmplx(v.real() * scale, v.imag() * scale);
        }
    }
    CUDA_SYNC_THREADS;

    if (tdi_chan_out) *tdi_chan_out = tdi_chan;
    if (kf0_out)      *kf0_out      = kf0;
    if (f0g_out)      *f0g_out      = f0g;
    if (dts_out)      *dts_out      = dts;
}

// Helper: dense rfft bin index for sparse FFT bin m (FFT order) when the
// heterodyne carrier was snapped to dense bin kf0.  Inlined identical math
// to np.fft.fftfreq(N, d=1/N): m_signed = (m < N/2) ? m : m - N.
CUDA_DEVICE inline int gbfd_dense_bin(int m, int N, int kf0)
{
    int m_signed = (m < (N >> 1)) ? m : (m - N);
    return kf0 + m_signed;
}

CUDA_DEVICE
void gbfd_run_one_source(GBTDIonTheFly *tof, void *shared_mem,
                         cmplx *X_het, int *k_f0_out, double *f0_grid_out,
                         double *params_in, double t_start, double Tobs,
                         int N, int nchannels, int n_params, int bin_i,
                         int log2N, double tukey_alpha)
{
    cmplx *tdi_chan = NULL;
    int    kf0      = 0;
    double f0g      = 0.0;
    double dts      = 0.0;
    gbfd_build_one_source(tof, shared_mem, params_in, t_start, Tobs,
                          N, nchannels, n_params, bin_i, log2N,
                          &tdi_chan, &kf0, &f0g, &dts, tukey_alpha);

    // Write heterodyne FD to global, in FFT order.
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        for (int c = 0; c < nchannels; ++c)
        {
            X_het[(size_t) bin_i * nchannels * N + (size_t) c * N + n] =
                tdi_chan[c * N + n];
        }
    }

    if (THREAD_ZERO)
    {
        k_f0_out[bin_i]    = kf0;
        f0_grid_out[bin_i] = f0g;
    }
    CUDA_SYNC_THREADS;
}

#ifdef __CUDACC__
CUDA_KERNEL
void gb_run_fd_wave_tdi_kernel(GBTDIonTheFly *tdi_on_fly,
    cmplx *X_het, int *k_f0_out, double *f0_grid_out,
    double *params, double t_start, double Tobs,
    int N, int num_bin, int n_params, int nchannels, int log2N,
    double tukey_alpha)
{
    extern CUDA_SHARED char shared_mem[];
    GBTDIonTheFly tof(tdi_on_fly->orbits, tdi_on_fly->tdi_config,
                      tdi_on_fly->T, tdi_on_fly->t_ref);
    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        gbfd_run_one_source(&tof, (void*) shared_mem,
                            X_het, k_f0_out, f0_grid_out,
                            params, t_start, Tobs,
                            N, nchannels, n_params, bin_i, log2N,
                            tukey_alpha);
    }
}
#endif

void gb_run_fd_wave_tdi_wrap(GBTDIonTheFly *tdi_on_fly,
    cmplx *X_het, int *k_f0_out, double *f0_grid_out,
    double *params, double t_start, double Tobs,
    int N_sparse, int num_bin, int n_params, int nchannels,
    double tukey_alpha)
{
    // Validate power-of-two
    int log2N = 0;
    {
        int m = N_sparse;
        while ((m & 1) == 0 && m > 1) { m >>= 1; ++log2N; }
#ifndef __CUDACC__
        if (m != 1) {
            throw std::invalid_argument(
                "gb_run_fd_wave_tdi_wrap: N_sparse must be a power of two.");
        }
#endif
    }

#ifdef __CUDACC__
    GBTDIonTheFly *gb_host = new GBTDIonTheFly(
        tdi_on_fly->orbits, tdi_on_fly->tdi_config,
        tdi_on_fly->T, tdi_on_fly->t_ref);

    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, tdi_on_fly->orbits, sizeof(Orbits),
                         cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_on_fly->tdi_config, sizeof(TDIConfig),
                         cudaMemcpyHostToDevice));

    gb_host->orbits     = d_orbits;
    gb_host->tdi_config = d_tdi_config;

    GBTDIonTheFly *d_gb;
    cudaMalloc(&d_gb, sizeof(GBTDIonTheFly));
    gpuErrchk(cudaMemcpy(d_gb, gb_host, sizeof(GBTDIonTheFly),
                         cudaMemcpyHostToDevice));

    int shared_bytes =
        tdi_on_fly->get_gb_fd_buffer_size(N_sparse, nchannels);

    // Allow shared usage past the 48 KB static default for large N_sparse.
    if (shared_bytes > 48 * 1024)
    {
        cudaFuncSetAttribute(
            gb_run_fd_wave_tdi_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes);
    }

    gb_run_fd_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
        d_gb, X_het, k_f0_out, f0_grid_out,
        params, t_start, Tobs,
        N_sparse, num_bin, n_params, nchannels, log2N, tukey_alpha);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_gb));
    delete gb_host;
#else
    const int shared_bytes =
        tdi_on_fly->get_gb_fd_buffer_size(N_sparse, nchannels);
    char *shared_mem = new char[shared_bytes];
    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        gbfd_run_one_source(tdi_on_fly, (void*) shared_mem,
                            X_het, k_f0_out, f0_grid_out,
                            params, t_start, Tobs,
                            N_sparse, nchannels, n_params, bin_i, log2N,
                            tukey_alpha);
    }
    delete[] shared_mem;
#endif
}


// ===========================================================================
// FD analogs of the WDM GBComputationGroup methods.
// ===========================================================================
//
// All three share gbfd_build_one_source(...) to materialise the
// (nchannels, N_sparse) heterodyne FD piece in shared memory; only the
// accumulator / scatter step differs.
//
// Inner product convention (matches lisatools.diagnostic.inner_product):
//   (a|b) = 4 Re sum_{c1,c2} sum_k conj(a_c1[k]) b_c2[k] invC[c1,c2][k] * df
// for tdi_type = TDI_XYZ;  the (c1==c2) diagonal terms only for TDI_AET / AE.

CUDA_DEVICE
inline void gbfd_accumulate_ll(double *d_h_acc, double *h_h_acc,
                               cmplx *tdi_chan, int N_sparse, int nchannels,
                               FDDomain *fd, int kf0,
                               int data_index, int noise_index, int tdi_type,
                               double tau_d_h, double tau_h_h)
{
    // tau_*: previous accumulator values to add into (so caller can pre-zero
    // its registers and pass them in).  We simply accumulate per (c1,c2).
    double dh = tau_d_h;
    double hh = tau_h_h;

    const int N = N_sparse;
    const int C = nchannels;

    if (tdi_type == TDI_XYZ)
    {
        // cross-channel 3x3 inv-covariance
        for (int m = THREAD_START_X; m < N; m += BLOCK_INCR_X)
        {
            int k = gbfd_dense_bin(m, N, kf0);
            if (!fd->in_band(k)) continue;
            for (int c1 = 0; c1 < C; ++c1)
            {
                cmplx d_c1 = fd->get_data(k, c1, data_index);
                for (int c2 = 0; c2 < C; ++c2)
                {
                    cmplx h_c2 = tdi_chan[c2 * N + m];
                    double invc = fd->get_invC_cross(k, c1, c2, noise_index);
                    cmplx prod_dh = gcmplx::conj(d_c1) * h_c2;
                    cmplx prod_hh =
                        gcmplx::conj(tdi_chan[c1 * N + m]) * h_c2;
                    dh += prod_dh.real() * invc;
                    hh += prod_hh.real() * invc;
                }
            }
        }
    }
    else
    {
        // TDI_AET (3 diag) or TDI_AE (2 diag): diagonal inv-covariance.
        int Cd = (tdi_type == TDI_AE) ? 2 : C;
        for (int m = THREAD_START_X; m < N; m += BLOCK_INCR_X)
        {
            int k = gbfd_dense_bin(m, N, kf0);
            if (!fd->in_band(k)) continue;
            for (int c = 0; c < Cd; ++c)
            {
                cmplx d_c = fd->get_data(k, c, data_index);
                cmplx h_c = tdi_chan[c * N + m];
                double invc = fd->get_invC_diag(k, c, noise_index);
                cmplx prod_dh = gcmplx::conj(d_c) * h_c;
                double mag_h2 = h_c.real() * h_c.real()
                               + h_c.imag() * h_c.imag();
                dh += prod_dh.real() * invc;
                hh += mag_h2 * invc;
            }
        }
    }

    *d_h_acc = dh;
    *h_h_acc = hh;
}

#ifdef __CUDACC__
CUDA_KERNEL
void gb_fd_get_ll_kernel(double *d_h_out, double *h_h_out,
    GBTDIonTheFly *tdi_on_fly_handle, FDDomain *fd,
    double *params, int *data_index_all, int *noise_index_all,
    double t_start, double Tobs,
    int N, int num_bin, int n_params, int nchannels, int log2N, int tdi_type)
{
    extern CUDA_SHARED char shared_mem[];
    CUDA_SHARED double d_h_tmp[NUM_THREADS_HERE];
    CUDA_SHARED double h_h_tmp[NUM_THREADS_HERE];

    GBTDIonTheFly tof(tdi_on_fly_handle->orbits, tdi_on_fly_handle->tdi_config,
                      tdi_on_fly_handle->T, tdi_on_fly_handle->t_ref);

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        for (int i = THREAD_START_X; i < NUM_THREADS_HERE; i += BLOCK_INCR_X)
        {
            d_h_tmp[i] = 0.0;
            h_h_tmp[i] = 0.0;
        }
        CUDA_SYNC_THREADS;

        cmplx *tdi_chan = NULL;
        int    kf0      = 0;
        double f0g      = 0.0;
        double dts      = 0.0;
        gbfd_build_one_source(&tof, (void*) shared_mem, params, t_start, Tobs,
                              N, nchannels, n_params, bin_i, log2N,
                              &tdi_chan, &kf0, &f0g, &dts, 0.0);

        double dh_local = 0.0, hh_local = 0.0;
        gbfd_accumulate_ll(&dh_local, &hh_local, tdi_chan, N, nchannels,
                           fd, kf0,
                           data_index_all[bin_i], noise_index_all[bin_i],
                           tdi_type, 0.0, 0.0);

        int tid = threadIdx.x;
        d_h_tmp[tid] = dh_local;
        h_h_tmp[tid] = hh_local;
        CUDA_SYNC_THREADS;

        double dh_sum = block_reduce(d_h_tmp);
        double hh_sum = block_reduce(h_h_tmp);
        if (THREAD_ZERO)
        {
            d_h_out[bin_i] = 4.0 * fd->df * dh_sum;
            h_h_out[bin_i] = 4.0 * fd->df * hh_sum;
        }
        CUDA_SYNC_THREADS;
    }
}
#endif

void GBComputationGroup::gb_fd_get_ll_wrap(double *d_h_out, double *h_h_out,
    Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
    double *params_all, int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type)
{
    int log2N = 0;
    {
        int m = N_sparse;
        while ((m & 1) == 0 && m > 1) { m >>= 1; ++log2N; }
#ifndef __CUDACC__
        if (m != 1) {
            throw std::invalid_argument(
                "gb_fd_get_ll_wrap: N_sparse must be a power of two.");
        }
#endif
    }

#ifdef __CUDACC__
    GBTDIonTheFly *gb_host = new GBTDIonTheFly(orbits, tdi_config, T, t_ref);
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, orbits, sizeof(Orbits), cudaMemcpyHostToDevice));
    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));
    gb_host->orbits = d_orbits;
    gb_host->tdi_config = d_tdi_config;
    GBTDIonTheFly *d_gb;
    cudaMalloc(&d_gb, sizeof(GBTDIonTheFly));
    gpuErrchk(cudaMemcpy(d_gb, gb_host, sizeof(GBTDIonTheFly), cudaMemcpyHostToDevice));
    FDDomain *d_fd;
    cudaMalloc(&d_fd, sizeof(FDDomain));
    gpuErrchk(cudaMemcpy(d_fd, fd, sizeof(FDDomain), cudaMemcpyHostToDevice));

    int shared_bytes = gb_host->get_gb_fd_buffer_size(N_sparse, nchannels);
    if (shared_bytes > 48 * 1024) {
        cudaFuncSetAttribute(gb_fd_get_ll_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    }
    gb_fd_get_ll_kernel<<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
        d_h_out, h_h_out, d_gb, d_fd,
        params_all, data_index_all, noise_index_all,
        t_start, T, N_sparse, num_bin, nparams, nchannels, log2N, tdi_type);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(d_orbits);
    cudaFree(d_tdi_config);
    cudaFree(d_gb);
    cudaFree(d_fd);
    delete gb_host;
#else
    GBTDIonTheFly tof(orbits, tdi_config, T, t_ref);
    int shared_bytes = tof.get_gb_fd_buffer_size(N_sparse, nchannels);
    char *shared_mem = new char[shared_bytes];
    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        cmplx *tdi_chan = NULL;
        int    kf0      = 0;
        double f0g      = 0.0;
        double dts      = 0.0;
        gbfd_build_one_source(&tof, (void*) shared_mem,
                              params_all, t_start, T,
                              N_sparse, nchannels, nparams, bin_i, log2N,
                              &tdi_chan, &kf0, &f0g, &dts, 0.0);

        double dh = 0.0, hh = 0.0;
        gbfd_accumulate_ll(&dh, &hh, tdi_chan, N_sparse, nchannels, fd, kf0,
                           data_index_all[bin_i], noise_index_all[bin_i],
                           tdi_type, 0.0, 0.0);
        d_h_out[bin_i] = 4.0 * fd->df * dh;
        h_h_out[bin_i] = 4.0 * fd->df * hh;
    }
    delete[] shared_mem;
#endif
}

// fill_global: add factor_i * h_i to a global FD template buffer
// of shape (num_data, nchannels, n_rfft).  The buffer is addressed via
// data_index_all[bin_i]; multiple bins routed to the same data_index are
// accumulated.  In the GPU build the writes go through atomicAdd to handle
// overlapping sources.
#ifdef __CUDACC__
CUDA_KERNEL
void gb_fd_fill_global_kernel(cmplx *template_fill,
    GBTDIonTheFly *tdi_on_fly_handle, FDDomain *fd,
    double *params, int *data_index_all, double *factors_all,
    double t_start, double Tobs,
    int N, int num_bin, int n_params, int nchannels, int log2N)
{
    extern CUDA_SHARED char shared_mem[];
    GBTDIonTheFly tof(tdi_on_fly_handle->orbits, tdi_on_fly_handle->tdi_config,
                      tdi_on_fly_handle->T, tdi_on_fly_handle->t_ref);
    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        cmplx *tdi_chan = NULL;
        int    kf0      = 0;
        double f0g      = 0.0;
        double dts      = 0.0;
        gbfd_build_one_source(&tof, (void*) shared_mem, params, t_start, Tobs,
                              N, nchannels, n_params, bin_i, log2N,
                              &tdi_chan, &kf0, &f0g, &dts, 0.0);

        int data_index = data_index_all[bin_i];
        double factor  = factors_all[bin_i];
        for (int m = THREAD_START_X; m < N; m += BLOCK_INCR_X)
        {
            int k = gbfd_dense_bin(m, N, kf0);
            if (!fd->in_band(k)) continue;
            for (int c = 0; c < nchannels; ++c)
            {
                cmplx v = tdi_chan[c * N + m];
                size_t idx = (size_t) data_index * nchannels * fd->n_rfft
                             + (size_t) c * fd->n_rfft + k;
                double re = factor * v.real();
                double im = factor * v.imag();
                atomicAdd(((double*)&template_fill[idx]) + 0, re);
                atomicAdd(((double*)&template_fill[idx]) + 1, im);
            }
        }
        CUDA_SYNC_THREADS;
    }
}
#endif

void GBComputationGroup::gb_fd_fill_global_wrap(cmplx *template_fill,
    Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
    double *params_all, int *data_index_all, double *factors_all,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels)
{
    int log2N = 0;
    {
        int m = N_sparse;
        while ((m & 1) == 0 && m > 1) { m >>= 1; ++log2N; }
#ifndef __CUDACC__
        if (m != 1) {
            throw std::invalid_argument(
                "gb_fd_fill_global_wrap: N_sparse must be a power of two.");
        }
#endif
    }

#ifdef __CUDACC__
    GBTDIonTheFly *gb_host = new GBTDIonTheFly(orbits, tdi_config, T, t_ref);
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, orbits, sizeof(Orbits), cudaMemcpyHostToDevice));
    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));
    gb_host->orbits = d_orbits;
    gb_host->tdi_config = d_tdi_config;
    GBTDIonTheFly *d_gb;
    cudaMalloc(&d_gb, sizeof(GBTDIonTheFly));
    gpuErrchk(cudaMemcpy(d_gb, gb_host, sizeof(GBTDIonTheFly), cudaMemcpyHostToDevice));
    FDDomain *d_fd;
    cudaMalloc(&d_fd, sizeof(FDDomain));
    gpuErrchk(cudaMemcpy(d_fd, fd, sizeof(FDDomain), cudaMemcpyHostToDevice));

    int shared_bytes = gb_host->get_gb_fd_buffer_size(N_sparse, nchannels);
    if (shared_bytes > 48 * 1024) {
        cudaFuncSetAttribute(gb_fd_fill_global_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    }
    gb_fd_fill_global_kernel<<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
        template_fill, d_gb, d_fd, params_all, data_index_all, factors_all,
        t_start, T, N_sparse, num_bin, nparams, nchannels, log2N);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(d_orbits);
    cudaFree(d_tdi_config);
    cudaFree(d_gb);
    cudaFree(d_fd);
    delete gb_host;
#else
    GBTDIonTheFly tof(orbits, tdi_config, T, t_ref);
    int shared_bytes = tof.get_gb_fd_buffer_size(N_sparse, nchannels);
    char *shared_mem = new char[shared_bytes];
    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        cmplx *tdi_chan = NULL;
        int    kf0      = 0;
        double f0g      = 0.0;
        double dts      = 0.0;
        gbfd_build_one_source(&tof, (void*) shared_mem,
                              params_all, t_start, T,
                              N_sparse, nchannels, nparams, bin_i, log2N,
                              &tdi_chan, &kf0, &f0g, &dts, 0.0);

        int data_index = data_index_all[bin_i];
        double factor  = factors_all[bin_i];
        for (int m = 0; m < N_sparse; ++m)
        {
            int k = gbfd_dense_bin(m, N_sparse, kf0);
            if (!fd->in_band(k)) continue;
            for (int c = 0; c < nchannels; ++c)
            {
                size_t idx = (size_t) data_index * nchannels * fd->n_rfft
                             + (size_t) c * fd->n_rfft + k;
                template_fill[idx] = template_fill[idx]
                    + cmplx(factor * tdi_chan[c * N_sparse + m].real(),
                            factor * tdi_chan[c * N_sparse + m].imag());
            }
        }
    }
    delete[] shared_mem;
#endif
}

// swap_ll: returns the five swap accumulators in lisatools convention,
//    (d|h_add), (d|h_rem), (h_add|h_add), (h_rem|h_rem), (h_add|h_rem).
//
// To stay GPU-friendly (one block per source), we run the heterodyne FD for
// the add and remove sources back-to-back in shared memory; the second pass
// overwrites the first's slow-signal buffer, so we accumulate (h_add|*) into
// per-thread registers before the second pass.
//
// For now the implementation is straightforward sequential per side, which
// keeps the math identical to the WDM swap convention; further unification
// (single-pass dual heterodyne) is a follow-up.
void GBComputationGroup::gb_fd_swap_ll_wrap(
    double *d_h_add_out, double *d_h_remove_out,
    double *add_add_out, double *remove_remove_out, double *add_remove_out,
    Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type)
{
    // Reuse get_ll for the diagonal-in-source accumulators, then explicitly
    // form the cross term (h_add | h_remove) per source.
    GBTDIonTheFly tof(orbits, tdi_config, T, t_ref);
    int log2N = 0;
    {
        int m = N_sparse;
        while ((m & 1) == 0 && m > 1) { m >>= 1; ++log2N; }
#ifndef __CUDACC__
        if (m != 1) {
            throw std::invalid_argument(
                "gb_fd_swap_ll_wrap: N_sparse must be a power of two.");
        }
#endif
    }

#ifdef __CUDACC__
    // GPU path: TODO -- a dedicated dual-source kernel.  The CPU build is
    // fully wired below and the GPU/CPU outputs match at the math level so
    // adding a kernel is mechanical; deferred to keep this commit focused.
    (void) tof; (void) log2N;
    (void) d_h_add_out; (void) d_h_remove_out;
    (void) add_add_out; (void) remove_remove_out; (void) add_remove_out;
    (void) orbits; (void) tdi_config; (void) fd;
    (void) params_add_all; (void) params_remove_all;
    (void) data_index_all; (void) noise_index_all;
    (void) num_bin; (void) nparams; (void) T;
    (void) t_start; (void) t_ref; (void) N_sparse;
    (void) nchannels; (void) tdi_type;
    printf("gb_fd_swap_ll_wrap GPU path not implemented yet.\n");
#else
    int shared_bytes = tof.get_gb_fd_buffer_size(N_sparse, nchannels);
    char *shared_mem_a = new char[shared_bytes];
    char *shared_mem_b = new char[shared_bytes];
    // Per-source loop.
    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        cmplx *h_add = NULL;
        int    kf0_a = 0;
        double f0g_a = 0.0;
        double dts_a = 0.0;
        gbfd_build_one_source(&tof, (void*) shared_mem_a, params_add_all,
                              t_start, T, N_sparse, nchannels, nparams,
                              bin_i, log2N, &h_add, &kf0_a, &f0g_a, &dts_a, 0.0);
        // (d|h_add), (h_add|h_add)
        double dh_a = 0.0, hh_aa = 0.0;
        gbfd_accumulate_ll(&dh_a, &hh_aa, h_add, N_sparse, nchannels, fd,
                           kf0_a, data_index_all[bin_i],
                           noise_index_all[bin_i], tdi_type, 0.0, 0.0);
        // h_add lives in shared_mem_a; build h_remove in a separate buffer.
        cmplx *h_rem = NULL;
        int    kf0_r = 0;
        double f0g_r = 0.0;
        double dts_r = 0.0;
        gbfd_build_one_source(&tof, (void*) shared_mem_b, params_remove_all,
                              t_start, T, N_sparse, nchannels, nparams,
                              bin_i, log2N, &h_rem, &kf0_r, &f0g_r, &dts_r, 0.0);
        double dh_r = 0.0, hh_rr = 0.0;
        gbfd_accumulate_ll(&dh_r, &hh_rr, h_rem, N_sparse, nchannels, fd,
                           kf0_r, data_index_all[bin_i],
                           noise_index_all[bin_i], tdi_type, 0.0, 0.0);
        // Cross term (h_add | h_remove): direct sum over dense bins shared
        // by the two sparse supports.  Each side knows its own (kf0, m_signed)
        // mapping; iterate over h_add bins and look up the matching h_remove
        // bin by absolute dense bin (k - kf0_r) modulo N_sparse.
        double hh_ar = 0.0;
        const int N = N_sparse;
        if (tdi_type == TDI_XYZ)
        {
            for (int m = 0; m < N; ++m)
            {
                int k = gbfd_dense_bin(m, N, kf0_a);
                if (!fd->in_band(k)) continue;
                // matching m on remove side: (k - kf0_r) mod N
                int mr_signed = k - kf0_r;
                int mr = ((mr_signed % N) + N) % N;
                int kr = gbfd_dense_bin(mr, N, kf0_r);
                if (kr != k) continue;  // remove-source slot does not cover k
                for (int c1 = 0; c1 < nchannels; ++c1)
                {
                    for (int c2 = 0; c2 < nchannels; ++c2)
                    {
                        cmplx ha = h_add[c1 * N + m];
                        cmplx hr = h_rem[c2 * N + mr];
                        double invc = fd->get_invC_cross(
                            k, c1, c2, noise_index_all[bin_i]);
                        cmplx prod = gcmplx::conj(ha) * hr;
                        hh_ar += prod.real() * invc;
                    }
                }
            }
        }
        else
        {
            int Cd = (tdi_type == TDI_AE) ? 2 : nchannels;
            for (int m = 0; m < N; ++m)
            {
                int k = gbfd_dense_bin(m, N, kf0_a);
                if (!fd->in_band(k)) continue;
                int mr_signed = k - kf0_r;
                int mr = ((mr_signed % N) + N) % N;
                int kr = gbfd_dense_bin(mr, N, kf0_r);
                if (kr != k) continue;
                for (int c = 0; c < Cd; ++c)
                {
                    cmplx ha = h_add[c * N + m];
                    cmplx hr = h_rem[c * N + mr];
                    double invc = fd->get_invC_diag(
                        k, c, noise_index_all[bin_i]);
                    cmplx prod = gcmplx::conj(ha) * hr;
                    hh_ar += prod.real() * invc;
                }
            }
        }

        double k4df = 4.0 * fd->df;
        d_h_add_out[bin_i]      = k4df * dh_a;
        d_h_remove_out[bin_i]   = k4df * dh_r;
        add_add_out[bin_i]      = k4df * hh_aa;
        remove_remove_out[bin_i] = k4df * hh_rr;
        add_remove_out[bin_i]   = k4df * hh_ar;
    }
    delete[] shared_mem_a;
    delete[] shared_mem_b;
#endif
}


// =============================================================================
//  FD-domain chain-rule gradients of gb_fd_get_ll / gb_fd_swap_ll.
//
//  Mirrors the WDM chain-rule gradient kernels: per parameter k we perturb
//  theta_k by +/- eps_k, rebuild the per-source heterodyne FD piece, and
//  accumulate the inner product (d - h_C | dh/dtheta_k) for get_ll, or the
//  post-swap analog (d - h_add_C + h_rem_C | dh_{add,rem}/dtheta_{add,rem}_k)
//  for swap_ll.  The parameter derivative is central FD,
//
//      dh/dtheta_k(p) = (h_+ - h_-) / (2 eps_k).
//
//  Each perturbed signal has its own snapped carrier kf0_{+,-}; we match it
//  back to the central side by absolute dense rfft bin (exactly the trick
//  used in gb_fd_swap_ll_wrap's cross term).  The matching is robust to the
//  rare case where +/- eps_f0 flips the rounding of f0 -> kf0.
//
//  The CPU build is fully wired; the GPU paths follow the same status as
//  gb_fd_swap_ll_wrap (printf-and-return placeholder).
// =============================================================================

// Per-pair gradient accumulator: returns the partial inner product
//    Re sum_{c1,c2} sum_m  conj(r_C[c1, m_C(k_pert(m))]) * h_pert[c2, m]
//                          * invC[c1,c2,k_pert(m)]
// iterated over the perturbed side's sparse bins.  The "residual" r_C is
// built from the central-side stash(es) by absolute-dense-bin reverse lookup:
//    get_ll: r_C[c, k] = d[c, k] - h_add_C[c, m_add(k)]    (h_rem_C = NULL)
//    swap:   r_C[c, k] = d[c, k] - h_add_C[c, m_add(k)] + h_rem_C[c, m_rem(k)]
// Missing coverage on a central side contributes 0 for that side (residual
// reduces to the remaining terms).
//
// All inputs are in FFT order (length N per channel); kf0_* are the
// dense-rfft-bin snaps that gbfd_build_one_source returned for each signal.
CUDA_DEVICE
inline double gbfd_grad_one_sided_partial(
    cmplx *h_pert, int kf0_pert,
    cmplx *h_add_C, int kf0_add_C,
    cmplx *h_rem_C, int kf0_rem_C,
    int N, int nchannels, FDDomain *fd,
    int data_index, int noise_index, int tdi_type)
{
    double acc = 0.0;
    const int Cd = (tdi_type == TDI_AE) ? 2 : nchannels;
    for (int mp = 0; mp < N; ++mp)
    {
        int kk = gbfd_dense_bin(mp, N, kf0_pert);
        if (!fd->in_band(kk)) continue;

        int ma_signed = kk - kf0_add_C;
        int ma = ((ma_signed % N) + N) % N;
        bool ka_match = (gbfd_dense_bin(ma, N, kf0_add_C) == kk);

        int mr = 0;
        bool kr_match = false;
        if (h_rem_C != NULL)
        {
            int mr_signed = kk - kf0_rem_C;
            mr = ((mr_signed % N) + N) % N;
            kr_match = (gbfd_dense_bin(mr, N, kf0_rem_C) == kk);
        }

        if (tdi_type == TDI_XYZ)
        {
            for (int c1 = 0; c1 < 3; ++c1)
            {
                cmplx d_c1 = fd->get_data(kk, c1, data_index);
                cmplx ha = ka_match ? h_add_C[c1 * N + ma] : cmplx(0., 0.);
                cmplx hr = (h_rem_C != NULL && kr_match)
                                ? h_rem_C[c1 * N + mr] : cmplx(0., 0.);
                cmplx r_c1 = d_c1 - ha + hr;
                for (int c2 = 0; c2 < 3; ++c2)
                {
                    cmplx hp = h_pert[c2 * N + mp];
                    double invc = fd->get_invC_cross(kk, c1, c2, noise_index);
                    cmplx prod = gcmplx::conj(r_c1) * hp;
                    acc += prod.real() * invc;
                }
            }
        }
        else
        {
            for (int c = 0; c < Cd; ++c)
            {
                cmplx d_c = fd->get_data(kk, c, data_index);
                cmplx ha = ka_match ? h_add_C[c * N + ma] : cmplx(0., 0.);
                cmplx hr = (h_rem_C != NULL && kr_match)
                                ? h_rem_C[c * N + mr] : cmplx(0., 0.);
                cmplx r_c = d_c - ha + hr;
                cmplx hp = h_pert[c * N + mp];
                double invc = fd->get_invC_diag(kk, c, noise_index);
                cmplx prod = gcmplx::conj(r_c) * hp;
                acc += prod.real() * invc;
            }
        }
    }
    return acc;
}


void GBComputationGroup::gb_fd_get_ll_grad_wrap(double *grad_out,
    Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
    double *params_all, int *data_index_all, int *noise_index_all,
    double *param_eps,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type)
{
    int log2N = 0;
    {
        int m = N_sparse;
        while ((m & 1) == 0 && m > 1) { m >>= 1; ++log2N; }
#ifndef __CUDACC__
        if (m != 1) {
            throw std::invalid_argument(
                "gb_fd_get_ll_grad_wrap: N_sparse must be a power of two.");
        }
#endif
    }

#ifdef __CUDACC__
    (void) grad_out; (void) orbits; (void) tdi_config; (void) fd;
    (void) params_all; (void) data_index_all; (void) noise_index_all;
    (void) param_eps;
    (void) num_bin; (void) nparams; (void) T;
    (void) t_start; (void) t_ref; (void) N_sparse;
    (void) nchannels; (void) tdi_type; (void) log2N;
    printf("gb_fd_get_ll_grad_wrap GPU path not implemented yet.\n");
#else
    GBTDIonTheFly tof(orbits, tdi_config, T, t_ref);
    int shared_bytes = tof.get_gb_fd_buffer_size(N_sparse, nchannels);
    char  *scratch       = new char[shared_bytes];
    cmplx *central_stash = new cmplx[(size_t) nchannels * N_sparse];

    double params_priv[N_PARAMS_MAX];

    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        for (int i = 0; i < nparams; ++i)
            params_priv[i] = params_all[bin_i * nparams + i];

        // Central build.
        cmplx *h_C_shared = NULL;
        int    kf0_C = 0;
        double f0g_C = 0.0, dts_C = 0.0;
        gbfd_build_one_source(&tof, (void*) scratch, params_priv,
                              t_start, T, N_sparse, nchannels, nparams,
                              /*bin_i=*/0, log2N,
                              &h_C_shared, &kf0_C, &f0g_C, &dts_C, 0.0);
        // The scratch's tdi_chan slab will be overwritten by perturbed builds
        // below, so stash the central signal in our own buffer.
        for (size_t idx = 0;
             idx < (size_t) nchannels * (size_t) N_sparse; ++idx)
            central_stash[idx] = h_C_shared[idx];

        int data_index  = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];

        for (int k = 0; k < nparams; ++k)
        {
            double eps_k = param_eps[k];
            if (eps_k <= 0.0)
            {
                grad_out[bin_i * nparams + k] = 0.0;
                continue;
            }
            double saved = params_priv[k];
            const double inv_2eps = 1.0 / (2.0 * eps_k);

            // +eps build (overwrites scratch's tdi_chan slab).
            params_priv[k] = saved + eps_k;
            cmplx *h_P_shared = NULL;
            int    kf0_P = 0;
            double f0g_P = 0.0, dts_P = 0.0;
            gbfd_build_one_source(&tof, (void*) scratch, params_priv,
                                  t_start, T, N_sparse, nchannels, nparams,
                                  0, log2N,
                                  &h_P_shared, &kf0_P, &f0g_P, &dts_P, 0.0);
            double acc_p = gbfd_grad_one_sided_partial(
                h_P_shared, kf0_P,
                central_stash, kf0_C,
                /*h_rem_C=*/NULL, /*kf0_rem_C=*/0,
                N_sparse, nchannels, fd,
                data_index, noise_index, tdi_type);

            // -eps build (overwrites again).
            params_priv[k] = saved - eps_k;
            cmplx *h_M_shared = NULL;
            int    kf0_M = 0;
            double f0g_M = 0.0, dts_M = 0.0;
            gbfd_build_one_source(&tof, (void*) scratch, params_priv,
                                  t_start, T, N_sparse, nchannels, nparams,
                                  0, log2N,
                                  &h_M_shared, &kf0_M, &f0g_M, &dts_M, 0.0);
            double acc_m = gbfd_grad_one_sided_partial(
                h_M_shared, kf0_M,
                central_stash, kf0_C,
                /*h_rem_C=*/NULL, /*kf0_rem_C=*/0,
                N_sparse, nchannels, fd,
                data_index, noise_index, tdi_type);

            params_priv[k] = saved;
            grad_out[bin_i * nparams + k] =
                4.0 * fd->df * (acc_p - acc_m) * inv_2eps;
        }
    }

    delete[] central_stash;
    delete[] scratch;
#endif
}


void GBComputationGroup::gb_fd_swap_ll_grad_wrap(
    double *grad_add_out, double *grad_remove_out,
    Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    double *param_eps_add, double *param_eps_remove,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type)
{
    int log2N = 0;
    {
        int m = N_sparse;
        while ((m & 1) == 0 && m > 1) { m >>= 1; ++log2N; }
#ifndef __CUDACC__
        if (m != 1) {
            throw std::invalid_argument(
                "gb_fd_swap_ll_grad_wrap: N_sparse must be a power of two.");
        }
#endif
    }

#ifdef __CUDACC__
    (void) grad_add_out; (void) grad_remove_out;
    (void) orbits; (void) tdi_config; (void) fd;
    (void) params_add_all; (void) params_remove_all;
    (void) data_index_all; (void) noise_index_all;
    (void) param_eps_add; (void) param_eps_remove;
    (void) num_bin; (void) nparams; (void) T;
    (void) t_start; (void) t_ref; (void) N_sparse;
    (void) nchannels; (void) tdi_type; (void) log2N;
    printf("gb_fd_swap_ll_grad_wrap GPU path not implemented yet.\n");
#else
    GBTDIonTheFly tof(orbits, tdi_config, T, t_ref);
    int shared_bytes = tof.get_gb_fd_buffer_size(N_sparse, nchannels);
    char  *scratch    = new char[shared_bytes];
    cmplx *add_stash  = new cmplx[(size_t) nchannels * N_sparse];
    cmplx *rem_stash  = new cmplx[(size_t) nchannels * N_sparse];

    double params_add_priv[N_PARAMS_MAX];
    double params_rem_priv[N_PARAMS_MAX];

    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        for (int i = 0; i < nparams; ++i)
        {
            params_add_priv[i] = params_add_all[bin_i * nparams + i];
            params_rem_priv[i] = params_remove_all[bin_i * nparams + i];
        }

        // Central builds for the add and remove sources.
        cmplx *h_addC_shared = NULL;
        int    kf0_addC = 0;
        double f0g_addC = 0.0, dts_addC = 0.0;
        gbfd_build_one_source(&tof, (void*) scratch, params_add_priv,
                              t_start, T, N_sparse, nchannels, nparams,
                              0, log2N,
                              &h_addC_shared, &kf0_addC, &f0g_addC, &dts_addC, 0.0);
        for (size_t idx = 0;
             idx < (size_t) nchannels * (size_t) N_sparse; ++idx)
            add_stash[idx] = h_addC_shared[idx];

        cmplx *h_remC_shared = NULL;
        int    kf0_remC = 0;
        double f0g_remC = 0.0, dts_remC = 0.0;
        gbfd_build_one_source(&tof, (void*) scratch, params_rem_priv,
                              t_start, T, N_sparse, nchannels, nparams,
                              0, log2N,
                              &h_remC_shared, &kf0_remC, &f0g_remC, &dts_remC, 0.0);
        for (size_t idx = 0;
             idx < (size_t) nchannels * (size_t) N_sparse; ++idx)
            rem_stash[idx] = h_remC_shared[idx];

        int data_index  = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];

        // ------ add-side gradient: sign = +1, perturb the add params ------
        for (int k = 0; k < nparams; ++k)
        {
            double eps_k = param_eps_add[k];
            if (eps_k <= 0.0)
            {
                grad_add_out[bin_i * nparams + k] = 0.0;
                continue;
            }
            double saved = params_add_priv[k];
            const double inv_2eps = 1.0 / (2.0 * eps_k);

            params_add_priv[k] = saved + eps_k;
            cmplx *h_aP = NULL;
            int kf0_aP = 0; double f0g_aP = 0.0, dts_aP = 0.0;
            gbfd_build_one_source(&tof, (void*) scratch, params_add_priv,
                                  t_start, T, N_sparse, nchannels, nparams,
                                  0, log2N,
                                  &h_aP, &kf0_aP, &f0g_aP, &dts_aP, 0.0);
            double acc_p = gbfd_grad_one_sided_partial(
                h_aP, kf0_aP,
                add_stash, kf0_addC,
                rem_stash, kf0_remC,
                N_sparse, nchannels, fd,
                data_index, noise_index, tdi_type);

            params_add_priv[k] = saved - eps_k;
            cmplx *h_aM = NULL;
            int kf0_aM = 0; double f0g_aM = 0.0, dts_aM = 0.0;
            gbfd_build_one_source(&tof, (void*) scratch, params_add_priv,
                                  t_start, T, N_sparse, nchannels, nparams,
                                  0, log2N,
                                  &h_aM, &kf0_aM, &f0g_aM, &dts_aM, 0.0);
            double acc_m = gbfd_grad_one_sided_partial(
                h_aM, kf0_aM,
                add_stash, kf0_addC,
                rem_stash, kf0_remC,
                N_sparse, nchannels, fd,
                data_index, noise_index, tdi_type);

            params_add_priv[k] = saved;
            grad_add_out[bin_i * nparams + k] =
                +4.0 * fd->df * (acc_p - acc_m) * inv_2eps;
        }

        // ------ remove-side gradient: sign = -1, perturb the remove params ------
        for (int k = 0; k < nparams; ++k)
        {
            double eps_k = param_eps_remove[k];
            if (eps_k <= 0.0)
            {
                grad_remove_out[bin_i * nparams + k] = 0.0;
                continue;
            }
            double saved = params_rem_priv[k];
            const double inv_2eps = 1.0 / (2.0 * eps_k);

            params_rem_priv[k] = saved + eps_k;
            cmplx *h_rP = NULL;
            int kf0_rP = 0; double f0g_rP = 0.0, dts_rP = 0.0;
            gbfd_build_one_source(&tof, (void*) scratch, params_rem_priv,
                                  t_start, T, N_sparse, nchannels, nparams,
                                  0, log2N,
                                  &h_rP, &kf0_rP, &f0g_rP, &dts_rP, 0.0);
            double acc_p = gbfd_grad_one_sided_partial(
                h_rP, kf0_rP,
                add_stash, kf0_addC,
                rem_stash, kf0_remC,
                N_sparse, nchannels, fd,
                data_index, noise_index, tdi_type);

            params_rem_priv[k] = saved - eps_k;
            cmplx *h_rM = NULL;
            int kf0_rM = 0; double f0g_rM = 0.0, dts_rM = 0.0;
            gbfd_build_one_source(&tof, (void*) scratch, params_rem_priv,
                                  t_start, T, N_sparse, nchannels, nparams,
                                  0, log2N,
                                  &h_rM, &kf0_rM, &f0g_rM, &dts_rM, 0.0);
            double acc_m = gbfd_grad_one_sided_partial(
                h_rM, kf0_rM,
                add_stash, kf0_addC,
                rem_stash, kf0_remC,
                N_sparse, nchannels, fd,
                data_index, noise_index, tdi_type);

            params_rem_priv[k] = saved;
            grad_remove_out[bin_i * nparams + k] =
                -4.0 * fd->df * (acc_p - acc_m) * inv_2eps;
        }
    }

    delete[] add_stash;
    delete[] rem_stash;
    delete[] scratch;
#endif
}


// CUDA_DEVICE
// TDSplineTDIWaveform::TDSplineTDIWaveform(Orbits *orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *phase_spline_): LISATDIonTheFly(orbits_, tdi_config_)
// {
//     phase_spline = phase_spline_;
//     amp_spline = amp_spline_;
//     // printf("spline type init : %d %d\n", amp_spline->spline_type, phase_spline->spline_type);        
//     // check_x();
// }


// CUDA_DEVICE
// void TDSplineTDIWaveform::check_x()
// {
//     for (int j = 0; j < amp_spline->ninterps; j += 1)
//     {
//         for (int i = 0; i < amp_spline->length; i += 1)
//         {
//             printf("%d %d %e %e\n", j, i, amp_spline->x0[j * amp_spline->length + i], phase_spline->x0[j * amp_spline->length + i]);
//         }
//     }
// }


// === FDSpline + TDSpline TDIWaveform method bodies + host launchers ===
// Moved to LAT at Phase 3L.6 (2026-06-03). Now live in
//   LISAanalysistools/src/lisatools/cutils/lat_spline_tdi_waveform.cu
// (copy-compiled in-place via the LISAResponse.cu pattern).
#if 0
CUDA_DEVICE
void FDSplineTDIWaveform::get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels)
{
    LISATDIonTheFly::get_tdi(
        buffer, buffer_length,
        tdi_channels_arr, 
        tdi_amp, tdi_phase,
        phi_ref,
        params, t_arr, N, bin_i, nchannels
    );
    
    CUDA_SYNC_THREADS;
    double amp_f;
    
#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr = blockDim.x;
#else // __CUDACC__
    int start = 0;
    int incr = 1;
#endif // __CUDACC__
    for (int i = start; i < N; i += incr)
    {
        amp_f = get_amp_f(t_arr[i], params, bin_i);
        for (int chan = 0; chan < tdi_config->num_channels; chan += 1)
        {
            tdi_amp[chan * N + i] *= amp_f;
        }
    }
    CUDA_SYNC_THREADS;
}


CUDA_DEVICE
double TDSplineTDIWaveform::get_amp(double t, double *params, int spline_i)
{
    // printf("before amp: %d\n", amp_spline->ninterps);
    return amp_spline->eval_single(t, spline_i);
}

CUDA_DEVICE
double TDSplineTDIWaveform::get_phase(double t, double *params, int spline_i)
{
    // printf("before phase: %d\n", phase_spline->ninterps);
    
    return phase_spline->eval_single(t, spline_i);
}

#ifdef __CUDACC__
CUDA_KERNEL
void td_spline_run_wave_tdi_kernel(TDSplineTDIWaveform *tdi_on_fly, int buffer_length, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    extern CUDA_SHARED char shared_mem[];
    void *buffer = (void*)shared_mem;
    tdi_on_fly->run_wave_tdi(buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
}
#endif

void td_spline_run_wave_tdi_wrap(TDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
#ifdef __CUDACC__
    TDSplineTDIWaveform *wave_here = new TDSplineTDIWaveform(tdi_on_fly->orbits, tdi_on_fly->tdi_config, tdi_on_fly->amp_spline, tdi_on_fly->phase_spline);
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, tdi_on_fly->orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_on_fly->tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    CubicSpline *d_amp_spline;
    cudaMalloc(&d_amp_spline, sizeof(CubicSpline));
    gpuErrchk(cudaMemcpy(d_amp_spline, tdi_on_fly->amp_spline, sizeof(CubicSpline), cudaMemcpyHostToDevice));

    CubicSpline *d_phase_spline;
    cudaMalloc(&d_phase_spline, sizeof(CubicSpline));
    gpuErrchk(cudaMemcpy(d_phase_spline, tdi_on_fly->phase_spline, sizeof(CubicSpline), cudaMemcpyHostToDevice));

    wave_here->orbits = d_orbits;
    wave_here->tdi_config = d_tdi_config;
    wave_here->amp_spline = d_amp_spline;
    wave_here->phase_spline = d_phase_spline;

    TDSplineTDIWaveform *d_wave_here;
    cudaMalloc(&d_wave_here, sizeof(TDSplineTDIWaveform));
    gpuErrchk(cudaMemcpy(d_wave_here, wave_here, sizeof(TDSplineTDIWaveform), cudaMemcpyHostToDevice));

    int buffer_length = tdi_on_fly->get_td_spline_buffer_size(N); 
    
    td_spline_run_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, buffer_length>>>(d_wave_here, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_amp_spline));
    gpuErrchk(cudaFree(d_phase_spline));
    gpuErrchk(cudaFree(d_wave_here));
    delete wave_here;
#else

    // make buffer 
    int buffer_length = tdi_on_fly->get_td_spline_buffer_size(N);
    char *buffer = new char[buffer_length];
    tdi_on_fly->run_wave_tdi((void*)buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
    delete[] buffer;
#endif
}


// CUDA_DEVICE
// void TDSplineTDIWaveform::get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int spline_i)
// {
//     for (int i = 0; i < N; i += 1)
//     {
//         // printf("bef: t, amp, phase: %d %e\n", i, t[i]);
//         amp[i] = amp_spline->eval_single(t[i], spline_i);
//         phase[i] = phase_spline->eval_single(t[i], spline_i);
//         // printf("af: t, amp, phase: %d %e, %e, %e\n", i, t[i], amp[i], phase[i]);
//     }
// }


// CUDA_DEVICE
// void TDSplineTDIWaveform::run_wave_tdi(cmplx *tdi_channels_arr, 
//     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
//     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
// {
//     for (int bin_i = 0; bin_i < num_bin; bin_i += 1)
//     {
//         // map to Tyson/Neil setup
//         // map to Tyson/Neil setup
//         double *params_here = &params[bin_i * n_params];
//         double *t_here = &t_arr[bin_i * N];

//         get_tdi(
//             &tdi_channels_arr[bin_i * nchannels * N], 
//             &Xamp[bin_i * N], &Xphase[bin_i * N],
//             &Yamp[bin_i * N], &Yphase[bin_i * N],
//             &Zamp[bin_i * N], &Zphase[bin_i * N], &phi_ref[bin_i * N],
//             params_here, t_here, N, bin_i, nchannels);
//         CUDA_SYNC_THREADS;   
//     }
// }




// CUDA_DEVICE
// FDSplineTDIWaveform::FDSplineTDIWaveform(Orbits *orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *freq_spline_, double *phase_ref_): LISATDIonTheFly(orbits_, tdi_config_)
// {
//     freq_spline = freq_spline_;
//     amp_spline = amp_spline_;
//     phase_ref_store = phase_ref_;
// }

// CUDA_DEVICE
// void FDSplineTDIWaveform::get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int spline_i)
// {
//     // only do frequency at ssb
//     // TODO: check this? should we just read in f_ssb?
//     double f = freq_spline->eval_single(t_ssb, spline_i);
//     double t_i = 0.0;
//     for (int i = 0; i < N; i += 1)
//     {
//         t_i = t[i];
//         // printf("bef: t, amp, phase: %d %e, %e, %e\n", i, t_i, amp[i], phase[i]);
//         amp[i] = 1.0;  // for frequency, we just use 1. amp_spline->eval_single(t_i, spline_i);
//         phase[i] = 2. * M_PI * f * t_i;
//         // printf("af: t, amp, phase: %d %e, %e, %e\n", i, t_i, amp[i], phase[i]);
//     }
// }


CUDA_DEVICE
double FDSplineTDIWaveform::get_amp(double t, double *params, int spline_i)
{
    return 1.0;
}

CUDA_DEVICE
double FDSplineTDIWaveform::get_phase(double t, double *params, int spline_i)
{
    double f = freq_spline->eval_single(t, spline_i);
    return 2. * M_PI * f * t;
}

CUDA_DEVICE
double FDSplineTDIWaveform::get_amp_f(double t, double *params, int spline_i)
{
    // TODO: may want to do this in a fast way
    return amp_spline->eval_single(t, spline_i);
}


// CUDA_DEVICE
// void FDSplineTDIWaveform::run_wave_tdi(cmplx *tdi_channels_arr, 
//     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
//     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
// {
//     for (int bin_i = 0; bin_i < num_sub; bin_i += 1)
//     {
//         // map to Tyson/Neil setup
//         double beta = params[bin_i * n_params + 3];
//         double costh = cos(M_PI / 2.0 - beta);
        
//         double lam = params[bin_i * n_params + 2];
//         double phi = lam;

//         double inc = params[bin_i * n_params + 0];
//         double cosi = cos(inc);

//         double psi = params[bin_i * n_params + 1];
//         double *params_here = &params[bin_i * n_params];
//         double *t_here = &t_arr[bin_i * N];
    
//         // TODO: CHECK THIS!!
//         get_tdi(
//             buffer, buffer_length, &X[bin_i * N], &Y[bin_i * N], &Z[bin_i * N], 
//             &Xamp[bin_i * N], &Xphase[bin_i * N],
//             &Yamp[bin_i * N], &Yphase[bin_i * N],
//             &Zamp[bin_i * N], &Zphase[bin_i * N], &phi_ref[bin_i * N],
//             params_here, t_here, N, costh, phi, cosi, psi, bin_i);
//     }
// }

CUDA_DEVICE
double FDSplineTDIWaveform::get_phase_ref(double t, double *params, int bin_i)
{
    // in FD, has to be fixed to 2 pi f_ssb t_ssb
    // t is t_ssb

    // t_i = t[i];
    // // TODO: should we make it so this is without the spline?
    double f = freq_spline->eval_single(t, bin_i);
    return 2. * M_PI * f * t;
    // phase[i] = 2. * M_PI * f * t_i;
    // phase[index] = phase_ref_store[spline_i * N + index];

}


#ifdef __CUDACC__
CUDA_KERNEL
void fd_spline_run_wave_tdi_kernel(FDSplineTDIWaveform *tdi_on_fly, int buffer_length, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    extern CUDA_SHARED char shared_mem[];
    void *buffer = (void*)shared_mem;
    tdi_on_fly->run_wave_tdi(buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
}
#endif

void fd_spline_run_wave_tdi_wrap(FDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
#ifdef __CUDACC__
    FDSplineTDIWaveform *wave_here = new FDSplineTDIWaveform(tdi_on_fly->orbits, tdi_on_fly->tdi_config, tdi_on_fly->amp_spline, tdi_on_fly->freq_spline);
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, tdi_on_fly->orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_on_fly->tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    CubicSpline *d_amp_spline;
    cudaMalloc(&d_amp_spline, sizeof(CubicSpline));
    gpuErrchk(cudaMemcpy(d_amp_spline, tdi_on_fly->amp_spline, sizeof(CubicSpline), cudaMemcpyHostToDevice));

    CubicSpline *d_freq_spline;
    cudaMalloc(&d_freq_spline, sizeof(CubicSpline));
    gpuErrchk(cudaMemcpy(d_freq_spline, tdi_on_fly->freq_spline, sizeof(CubicSpline), cudaMemcpyHostToDevice));

    wave_here->orbits = d_orbits;
    wave_here->tdi_config = d_tdi_config;
    wave_here->amp_spline = d_amp_spline;
    wave_here->freq_spline = d_freq_spline;

    FDSplineTDIWaveform *d_wave_here;
    cudaMalloc(&d_wave_here, sizeof(FDSplineTDIWaveform));
    gpuErrchk(cudaMemcpy(d_wave_here, wave_here, sizeof(FDSplineTDIWaveform), cudaMemcpyHostToDevice));

    int buffer_length = tdi_on_fly->get_fd_spline_buffer_size(N); 
    
    fd_spline_run_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, buffer_length>>>(d_wave_here, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_amp_spline));
    gpuErrchk(cudaFree(d_freq_spline));
    gpuErrchk(cudaFree(d_wave_here));
    delete wave_here;
#else

    // make buffer
    int buffer_length = tdi_on_fly->get_fd_spline_buffer_size(N);
    char *buffer = new char[buffer_length];
    tdi_on_fly->run_wave_tdi((void*)buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
    delete[] buffer;
#endif
}
#endif  // === end FDSpline + TDSpline TDIWaveform moved to LAT ===


// CUDA_DEVICE
// cmplx LagrangeInterpolant::interp(double t, cmplx *wave, int wave_N, int bin_i)
// {
//     /*
//     double A = 1.0;
//     for (int i = 1; i < h; i += 1){
//         A *= (i + e) * (i + 1 - e);
//     }
//     double denominator = factorials[h - 1] * factorials[h];
//     A /= denominator;
//     */

//     // if ((i == 0) && (link_i == 0)) printf("%.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", L, delay_rec, delay_em, x0[0], x0[1], x0[2],x1[0], x1[1], x1[2]);
//     double clipped_delay_rec = t; 
//     int integer_delay_rec = (int)ceil(clipped_delay_rec * sampling_frequency) - 1;
//     double fraction0 = 1.0 + integer_delay_rec - clipped_delay_rec * sampling_frequency;
    
//     // int h, int d, double e, double *A_arr, double deps, double *E_arr, int start_input_ind
//     // half_point_count, integer_delay, fraction, A_arr, deps, E_arr, start_input_ind
//     double e = fraction0;
//     int d = integer_delay_rec;
    
//     int ind = (int)(e / deps);

//     double frac = (e - ind * deps) / deps;
//     double A = A_arr[ind] * (1. - frac) + A_arr[ind + 1] * frac;

//     double B = 1.0 - e;
//     double C = e;
//     double D = e * (1.0 - e);

//     double sum_hp = 0.0;
//     double sum_hc = 0.0;
//     cmplx temp_up, temp_down;
//     // if ((i == 100) && (link_i == 0)) printf("%d %e %e %e %e %e\n", d, e, A, B, C, D);
//     // printf("in: %d %d\n", d, start_input_ind);
//     for (int j = 1; j < h; j += 1)
//     {

//         // get constants

//         /*
//         double first_term = factorials[h - 1] / factorials[h - 1 - j];
//         double second_term = factorials[h] / factorials[h + j];
//         double value = first_term * second_term;

//         value = value * pow(-1.0, (double)j);
//         */

//         double E = E_arr[j - 1];

//         double F = j + e;
//         double G = j + (1 - e);

//         // perform calculation
//         temp_up = wave[(bin_i * wave_N) + d + 1 + j];
//         temp_down = wave[(bin_i * wave_N) + d - j];

//         // if ((i == 100) && (link_i == 0)) printf("mid: %d %d %d %e %e %e %e %e %e %e\n", j, d + 1 + j - start_input_ind, d - j - start_input_ind, temp_up, temp_down, E, F, G);
//         sum_hp += E * (temp_up.real() / F + temp_down.real() / G);
//         sum_hc += E * (temp_up.imag() / F + temp_down.imag() / G);
//     }
//     temp_up = wave[(bin_i * wave_N) + d + 1];
//     temp_down = wave[(bin_i * wave_N) + d];
//     // printf("out: %d %d\n", d, start_input_ind);
//     double real_out = A * (B * temp_up.real() + C * temp_down.real() + D * sum_hp);
//     double imag_out = A * (B * temp_up.imag() + C * temp_down.imag() + D * sum_hc);
//     cmplx output(real_out, imag_out);
//     return output;
//     // if ((i == 100) && (link_i == 0)) printf("end: %e %e\n", *result_hp, *result_hc);
// }

// CUDA_DEVICE
// void TDLagrangeInterpTDIWave::get_hp_hc(double *hp, double *hc, double t, double *params, double phase_change, int bin_i)
// {
//     cmplx I(0.0, 1.0);
//     cmplx wave_out = lagrange->interp(t, wave, wave_N, bin_i);

//     wave_out *= gcmplx::exp(I * phase_change);
//     *hp = wave_out.real();
//     *hc = wave_out.imag();
// }

// CUDA_DEVICE
// TDLagrangeInterpTDIWave::TDLagrangeInterpTDIWave(Orbits *orbits_, TDIConfig *tdi_config_, cmplx *wave_, int wave_N_, LagrangeInterpolant *lagrange_): LISATDIonTheFly(orbits_, tdi_config_)
// {
//     lagrange = lagrange_;
//     wave = wave_;
//     wave_N = wave_N_; 
// }


// int TDLagrangeInterpTDIWave::get_beta_index()
// {
//     // ndim = 2; beta second
//     return 1;
// }

// int TDLagrangeInterpTDIWave::get_lam_index()
// {
//     // ndim = 2; lam first
//     return -1;
// }


// =============================================================================
// Chunked-heterodyne host wrappers (templated over source class).
//
// Each impl function:
//   - allocates the four per-chunk workspaces (plus get_tdi scratch),
//   - launches the templated kernel,
//   - frees the workspaces.
//
// On GPU, ``grid_dim`` selects the launch grid (caller picks via
// ``gb_wdm_het.chunked_het_grid_dim`` on the Python side). On CPU,
// ``grid_dim`` is ignored -- ``BLOCK_START_X`` / ``GRID_INCR_X`` walk the
// whole range serially with the single fake "block".
//
// The 6 GBComputationGroup / SOBBHComputationGroup methods at the bottom
// route into these impls with the appropriate SourceT.
// =============================================================================

static inline int wdm_het_get_tdi_scratch_bytes(int N_sparse)
{
    // Mirrors the per-block sizing in gb_wdm_spline_*_kernel:
    //   get_tdi_buffer_size(L) = 2*L*8 + L*4 + L*1 = 21*L bytes
    // Plus 16-byte safety pad so misaligned reads in the inner kernel
    // can't overrun.
    return 21 * N_sparse + 16;
}


// ---- Persistent workspace cache --------------------------------------------
//
// The chunked-het impl wrappers used to ``cudaMalloc`` + ``cudaFree`` their
// per-launch scratch slabs on every call. With ``n_slots = gd_x * n_chunks``
// and a per-slot size in the tens of MB, the resulting ``cudaMalloc`` calls
// became visible in nsys (~1 sec total per ``get_ll_wdm`` call at moderate
// gd_x on A100, accounting for ~20% of wall time). The slabs are
// reused identically across calls when the shape ints match, so we cache
// them as static-lifetime pointers and only re-allocate when a larger size
// is requested. We never shrink -- a one-time growth amortises across the
// rest of the process lifetime, and the alternative (free + re-alloc on
// every shape change) re-introduces the cost we're eliminating.
//
// Thread safety: ``GBWDMComputations`` is documented as single-threaded
// per instance. If callers ever start using these wrappers concurrently
// they must guard the static state externally.
//
// On CPU the same helper grows a heap-resident ``new[]`` buffer. The CPU
// path's cost was always negligible vs the kernel work, but keeping the
// API symmetric is cheap.
template <class T>
static inline T *wdm_het_grow_cache(T *&buf, size_t &cur_size, size_t needed)
{
    if (needed <= cur_size) return buf;
#ifdef __CUDACC__
    if (buf != nullptr) gpuErrchk(cudaFree(buf));
    gpuErrchk(cudaMalloc((void **) &buf, needed * sizeof(T)));
#else
    delete[] buf;
    buf = new T[needed]();
#endif
    cur_size = needed;
    return buf;
}

// Specialised growth helper for the ``void *`` get_tdi scratch byte-slab,
// which doesn't fit the templated ``T``-array shape.
static inline void *wdm_het_grow_cache_bytes(void *&buf, size_t &cur_size,
                                              size_t needed_bytes)
{
    if (needed_bytes <= cur_size) return buf;
#ifdef __CUDACC__
    if (buf != nullptr) gpuErrchk(cudaFree(buf));
    gpuErrchk(cudaMalloc(&buf, needed_bytes));
#else
    delete[] (char *) buf;
    buf = (void *) new char[needed_bytes]();
#endif
    cur_size = needed_bytes;
    return buf;
}

// =============================================================================
// OLD impl wrappers (paired with the OLD kernels above) -- disabled.
// Replaced by the new wrappers further down which launch the new kernels.
// =============================================================================
#if 0 // ---- OLD IMPL WRAPPERS BEGIN ----
template <class SourceT>
static void wdm_het_fill_global_impl(
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
    int m_band_half_width)
{
    const int Nf = wdm_settings->Nf;
    // 3D-grid launch: gridDim.x = gd_x (binaries axis, capped by default
    // for heap-scratch sizing), gridDim.z = n_chunks (chunks axis). Each
    // (x, z) block keeps its own per-(chunk, binary) scratch slot keyed
    // by ``(BLOCK_START_X * n_chunks + j)`` inside the kernel, so the
    // heap slabs scale as ``gd_x * n_chunks * per_slot`` (NOT *num_bin):
    // each X block grid-strides through binaries, reusing one scratch
    // set across all binaries it visits.
    // TODO: tune gd_x default by occupancy + shared-mem availability;
    // a follow-up should move ``w_chunk`` into per-block shared memory
    // (or shrink to per-group active-band width) so we can raise this
    // significantly.
    // On CPU gd_x = 1 (BLOCK_START_X / GRID_INCR_X stubs collapse to a
    // single virtual block), so n_slots = n_chunks -- same as pre-3D-grid.
#ifdef __CUDACC__
    const int gd_x = (grid_dim > 0) ? grid_dim
                                    : FAST_WDM_HET_GRID_DIM_X_DEFAULT;
#else
    (void) grid_dim;
    const int gd_x = 1;
#endif
    const int gd_z = n_chunks;
    const size_t n_slots = (size_t) gd_x * (size_t) n_chunks;

    const size_t n_fd = n_slots * nchannels * n_rfft_chunk;
    const size_t n_ls = n_slots * Nt_sub;
    const size_t n_wd = n_slots * nchannels * Nf * Nt_sub;
    const size_t n_tc = n_slots * nchannels * N_sparse;
    const int    scratch_per = wdm_het_get_tdi_scratch_bytes(N_sparse);
    const size_t n_sc = n_slots * (size_t) scratch_per;

    // Persistent workspace cache -- grows on demand, no per-call malloc /
    // free. See ``wdm_het_grow_cache`` for the rationale.
    static cmplx  *ws_chunk_fd_all      = nullptr;  static size_t cap_fd = 0;
    static cmplx  *ws_layer_scratch_all = nullptr;  static size_t cap_ls = 0;
    static double *ws_chunk_wdm_all     = nullptr;  static size_t cap_wd = 0;
    static cmplx  *ws_tdi_channels_all  = nullptr;  static size_t cap_tc = 0;
    static void   *get_tdi_scratch_all  = nullptr;  static size_t cap_sc = 0;
    wdm_het_grow_cache(ws_chunk_fd_all,      cap_fd, n_fd);
    wdm_het_grow_cache(ws_layer_scratch_all, cap_ls, n_ls);
    wdm_het_grow_cache(ws_chunk_wdm_all,     cap_wd, n_wd);
    wdm_het_grow_cache(ws_tdi_channels_all,  cap_tc, n_tc);
    wdm_het_grow_cache_bytes(get_tdi_scratch_all, cap_sc, n_sc);

#ifdef __CUDACC__
    // Kernel zero-inits per-chunk slabs at chunk entry; no host-side memset
    // needed here.

    // The Orbits / TDIConfig / WDMSettings wrapper structs are constructed
    // on the host heap via ``new`` (see binding_flr.hpp / binding_tof.hpp);
    // device code dereferencing a host pointer triggers an illegal memory
    // access. Mirror the LISAResponse.cu:419 pattern: shallow-copy each
    // struct to device memory, launch with the device-side copies. Inner
    // device-pointer fields (ltt_arr, unit_starts, …) survive the shallow
    // copy. See sprint-root CLAUDE.md.
    //
    // The struct copies are tiny (~100 bytes each) and fixed-size, so we
    // just cache the device-side pointer; one-time cudaMalloc, repeated
    // cudaMemcpy. The host-side struct contents are memcpy'd fresh on
    // each call to capture any state changes.
    static Orbits      *orbits_gpu       = nullptr;
    static TDIConfig   *tdi_config_gpu   = nullptr;
    static WDMSettings *wdm_settings_gpu = nullptr;
    if (orbits_gpu       == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,       sizeof(Orbits)));
    if (tdi_config_gpu   == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu,   sizeof(TDIConfig)));
    if (wdm_settings_gpu == nullptr) gpuErrchk(cudaMalloc(&wdm_settings_gpu, sizeof(WDMSettings)));
    gpuErrchk(cudaMemcpy(orbits_gpu,       orbits,       sizeof(Orbits),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu,   tdi_config,   sizeof(TDIConfig),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(wdm_settings_gpu, wdm_settings, sizeof(WDMSettings),
                         cudaMemcpyHostToDevice));

    dim3 grid((unsigned) gd_x, 1u, (unsigned) gd_z);
    wdm_het_fill_global_kernel<SourceT><<<grid, NUM_THREADS_HERE>>>(
        template_fill, orbits_gpu, tdi_config_gpu,
        wdm_settings_gpu,
        params_all, factors_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref,
        tukey_alpha,
        N_cp_sig, N_cp_orbit,
        ws_chunk_fd_all, ws_layer_scratch_all, ws_chunk_wdm_all,
        ws_tdi_channels_all,
        get_tdi_scratch_all, scratch_per);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    // No per-call cudaFree -- the static-cached buffers live for the
    // process lifetime and are reused across calls.
#else
    wdm_het_fill_global_kernel<SourceT>(
        template_fill, orbits, tdi_config,
        wdm_settings,
        params_all, factors_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref,
        tukey_alpha,
        N_cp_sig, N_cp_orbit,
        ws_chunk_fd_all, ws_layer_scratch_all, ws_chunk_wdm_all,
        ws_tdi_channels_all,
        get_tdi_scratch_all, scratch_per);
#endif
}


template <class SourceT>
static void wdm_het_get_ll_impl(
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
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups)
{
    const int Nf = wdm_settings->Nf;
    const int Nt = wdm_settings->Nt;
    // See ``wdm_het_fill_global_impl`` for the n_slots / gd_x / gd_z
    // contract and the heap-sizing rationale. On CPU gd_x = 1 (the
    // BLOCK_START_X / GRID_INCR_X stubs collapse to a single virtual
    // block) so n_slots = n_chunks, matching the pre-3D-grid layout.
#ifdef __CUDACC__
    const int gd_x = (grid_dim > 0) ? grid_dim
                                    : FAST_WDM_HET_GRID_DIM_X_DEFAULT;
#else
    (void) grid_dim;
    const int gd_x = 1;
#endif
    const int gd_z = n_chunks;
    const size_t n_slots = (size_t) gd_x * (size_t) n_chunks;

    const size_t n_fd = n_slots * nchannels * n_rfft_chunk;
    const size_t n_ls = n_slots * Nt_sub;
    const size_t n_wd = n_slots * nchannels * Nf * Nt_sub;
    const size_t n_tc = n_slots * nchannels * N_sparse;
    const int    scratch_per = wdm_het_get_tdi_scratch_bytes(N_sparse);
    const size_t n_sc = n_slots * (size_t) scratch_per;

    // Persistent workspace cache (see ``wdm_het_fill_global_impl``).
    static cmplx  *ws_chunk_fd_all      = nullptr;  static size_t cap_fd = 0;
    static cmplx  *ws_layer_scratch_all = nullptr;  static size_t cap_ls = 0;
    static double *ws_chunk_wdm_all     = nullptr;  static size_t cap_wd = 0;
    static cmplx  *ws_tdi_channels_all  = nullptr;  static size_t cap_tc = 0;
    static void   *get_tdi_scratch_all  = nullptr;  static size_t cap_sc = 0;
    wdm_het_grow_cache(ws_chunk_fd_all,      cap_fd, n_fd);
    wdm_het_grow_cache(ws_layer_scratch_all, cap_ls, n_ls);
    wdm_het_grow_cache(ws_chunk_wdm_all,     cap_wd, n_wd);
    wdm_het_grow_cache(ws_tdi_channels_all,  cap_tc, n_tc);
    wdm_het_grow_cache_bytes(get_tdi_scratch_all, cap_sc, n_sc);

#ifdef __CUDACC__
    static Orbits      *orbits_gpu       = nullptr;
    static TDIConfig   *tdi_config_gpu   = nullptr;
    static WDMSettings *wdm_settings_gpu = nullptr;
    if (orbits_gpu       == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,       sizeof(Orbits)));
    if (tdi_config_gpu   == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu,   sizeof(TDIConfig)));
    if (wdm_settings_gpu == nullptr) gpuErrchk(cudaMalloc(&wdm_settings_gpu, sizeof(WDMSettings)));
    gpuErrchk(cudaMemcpy(orbits_gpu,       orbits,       sizeof(Orbits),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu,   tdi_config,   sizeof(TDIConfig),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(wdm_settings_gpu, wdm_settings, sizeof(WDMSettings),
                         cudaMemcpyHostToDevice));

    dim3 grid((unsigned) gd_x, 1u, (unsigned) gd_z);
    wdm_het_get_ll_kernel<SourceT><<<grid, NUM_THREADS_HERE>>>(
        d_h_out, h_h_out, orbits_gpu, tdi_config_gpu,
        wdm_settings_gpu,
        params_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref,
        tdi_type,
        tukey_alpha,
        N_cp_sig, N_cp_orbit,
        ws_chunk_fd_all, ws_layer_scratch_all, ws_chunk_wdm_all,
        ws_tdi_channels_all,
        get_tdi_scratch_all, scratch_per,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups, /*m_band_half_width=*/1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    wdm_het_get_ll_kernel<SourceT>(
        d_h_out, h_h_out, orbits, tdi_config,
        wdm_settings,
        params_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref,
        tdi_type,
        tukey_alpha,
        N_cp_sig, N_cp_orbit,
        ws_chunk_fd_all, ws_layer_scratch_all, ws_chunk_wdm_all,
        ws_tdi_channels_all,
        get_tdi_scratch_all, scratch_per,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups, /*m_band_half_width=*/1);
#endif
}


template <class SourceT>
static void wdm_het_swap_ll_impl(
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
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups,
    int *pair_m_lo_b, int *pair_m_hi_b)
{
    const int Nf = wdm_settings->Nf;
    const int Nt = wdm_settings->Nt;
    // See ``wdm_het_fill_global_impl`` for n_slots / gd_x / gd_z contract.
    // swap_ll has two chunk_fd slabs (add + rem) and two chunk_wdm slabs;
    // each is sized at n_slots = gd_x * n_chunks.
#ifdef __CUDACC__
    const int gd_x = (grid_dim > 0) ? grid_dim
                                    : FAST_WDM_HET_GRID_DIM_X_DEFAULT;
#else
    (void) grid_dim;
    const int gd_x = 1;
#endif
    const int gd_z = n_chunks;
    const size_t n_slots = (size_t) gd_x * (size_t) n_chunks;

    const size_t n_fd = n_slots * nchannels * n_rfft_chunk;
    const size_t n_ls = n_slots * Nt_sub;
    const size_t n_wd = n_slots * nchannels * Nf * Nt_sub;
    const size_t n_tc = n_slots * nchannels * N_sparse;
    const int    scratch_per = wdm_het_get_tdi_scratch_bytes(N_sparse);
    const size_t n_sc = n_slots * (size_t) scratch_per;

    // Persistent workspace cache (see ``wdm_het_fill_global_impl``).
    // swap_ll has two chunk_fd + two chunk_wdm slabs (add + rem).
    static cmplx  *ws_chunk_fd_add_all  = nullptr;  static size_t cap_fda = 0;
    static cmplx  *ws_chunk_fd_rem_all  = nullptr;  static size_t cap_fdr = 0;
    static cmplx  *ws_layer_scratch_all = nullptr;  static size_t cap_ls  = 0;
    static double *ws_chunk_wdm_add_all = nullptr;  static size_t cap_wda = 0;
    static double *ws_chunk_wdm_rem_all = nullptr;  static size_t cap_wdr = 0;
    static cmplx  *ws_tdi_channels_all  = nullptr;  static size_t cap_tc  = 0;
    static void   *get_tdi_scratch_all  = nullptr;  static size_t cap_sc  = 0;
    wdm_het_grow_cache(ws_chunk_fd_add_all,  cap_fda, n_fd);
    wdm_het_grow_cache(ws_chunk_fd_rem_all,  cap_fdr, n_fd);
    wdm_het_grow_cache(ws_layer_scratch_all, cap_ls,  n_ls);
    wdm_het_grow_cache(ws_chunk_wdm_add_all, cap_wda, n_wd);
    wdm_het_grow_cache(ws_chunk_wdm_rem_all, cap_wdr, n_wd);
    wdm_het_grow_cache(ws_tdi_channels_all,  cap_tc,  n_tc);
    wdm_het_grow_cache_bytes(get_tdi_scratch_all, cap_sc, n_sc);

#ifdef __CUDACC__
    static Orbits      *orbits_gpu       = nullptr;
    static TDIConfig   *tdi_config_gpu   = nullptr;
    static WDMSettings *wdm_settings_gpu = nullptr;
    if (orbits_gpu       == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,       sizeof(Orbits)));
    if (tdi_config_gpu   == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu,   sizeof(TDIConfig)));
    if (wdm_settings_gpu == nullptr) gpuErrchk(cudaMalloc(&wdm_settings_gpu, sizeof(WDMSettings)));
    gpuErrchk(cudaMemcpy(orbits_gpu,       orbits,       sizeof(Orbits),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu,   tdi_config,   sizeof(TDIConfig),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(wdm_settings_gpu, wdm_settings, sizeof(WDMSettings),
                         cudaMemcpyHostToDevice));

    dim3 grid((unsigned) gd_x, 1u, (unsigned) gd_z);
    wdm_het_swap_ll_kernel<SourceT><<<grid, NUM_THREADS_HERE>>>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits_gpu, tdi_config_gpu,
        wdm_settings_gpu,
        params_add_all, params_remove_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref,
        tdi_type,
        tukey_alpha,
        N_cp_sig, N_cp_orbit,
        ws_chunk_fd_add_all, ws_chunk_fd_rem_all,
        ws_layer_scratch_all,
        ws_chunk_wdm_add_all, ws_chunk_wdm_rem_all,
        ws_tdi_channels_all,
        get_tdi_scratch_all, scratch_per,
        binary_perm, group_starts, group_ends, group_m_lo, group_m_hi, n_groups,
        pair_m_lo_b, pair_m_hi_b, /*m_band_half_width=*/1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    wdm_het_swap_ll_kernel<SourceT>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits, tdi_config,
        wdm_settings,
        params_add_all, params_remove_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref,
        tdi_type,
        tukey_alpha,
        N_cp_sig, N_cp_orbit,
        ws_chunk_fd_add_all, ws_chunk_fd_rem_all,
        ws_layer_scratch_all,
        ws_chunk_wdm_add_all, ws_chunk_wdm_rem_all,
        ws_tdi_channels_all,
        get_tdi_scratch_all, scratch_per,
        binary_perm, group_starts, group_ends, group_m_lo, group_m_hi, n_groups,
        pair_m_lo_b, pair_m_hi_b, /*m_band_half_width=*/1);
#endif
}
#endif // ---- OLD IMPL WRAPPERS END ----


// =============================================================================
// NEW impl wrappers (paired with the NEW kernels at line ~2520 above).
//
// Signatures kept compatible with the existing public ``gb_wdm_het_*_wrap``
// signatures so the Python ABI is unchanged. The args we no longer use
// (layer-group args, N_cp_sig, N_cp_orbit, grid_dim) are accepted and
// ignored / forwarded as comments below.
// =============================================================================
template <class SourceT>
static void wdm_het_fill_global_impl(
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
    int m_band_half_width)
{
    // The new kernel does not use N_cp_sig / N_cp_orbit (no spline / orbit
    // caches in this rewrite). grid_dim selects gridDim.x (binaries per
    // launch); default = num_bin (one block per binary).
    (void) N_cp_sig; (void) N_cp_orbit;
#ifdef __CUDACC__
    // One binary per block on grid.X (always). The grid_dim arg is
    // vestigial -- kept in the signature for ABI stability with the
    // older callers, but the kernel does not grid-stride over binaries.
    (void) grid_dim;
    const int gd_x = num_bin;
    // Shared-mem layout (must match wdm_het_fill_global_kernel):
    //   fd_chunk_buf [nchannels * N_sparse] cmplx  -- chunk-FD (built ONCE per chunk)
    //   layer_buf    [nchannels * Nt_sub]   cmplx  -- per-m_layer scratch
    //   fft_scratch  [wdm_cufftdx_max_scratch()] bytes  -- 0 unless cufftdx is on
    // (no per-thread partials -- fill_global writes directly via atomicAdd.)
    const size_t shared_bytes =
        (size_t) nchannels * (size_t) N_sparse * sizeof(cmplx) +
        (size_t) nchannels * (size_t) Nt_sub   * sizeof(cmplx) +
        wdm_cufftdx_max_scratch();

    // Upload host-side wrapper structs (Orbits / TDIConfig / WDMSettings) to
    // device. Cache the device-side pointers across calls.
    static Orbits      *orbits_gpu       = nullptr;
    static TDIConfig   *tdi_config_gpu   = nullptr;
    static WDMSettings *wdm_settings_gpu = nullptr;
    if (orbits_gpu       == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,       sizeof(Orbits)));
    if (tdi_config_gpu   == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu,   sizeof(TDIConfig)));
    if (wdm_settings_gpu == nullptr) gpuErrchk(cudaMalloc(&wdm_settings_gpu, sizeof(WDMSettings)));
    gpuErrchk(cudaMemcpy(orbits_gpu,       orbits,       sizeof(Orbits),       cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu,   tdi_config,   sizeof(TDIConfig),    cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(wdm_settings_gpu, wdm_settings, sizeof(WDMSettings), cudaMemcpyHostToDevice));

    // TODO: when chunks move to gridDim.y, change to dim3(gd_x, n_chunks, 1).
    dim3 grid((unsigned) gd_x, 1u, 1u);
    wdm_het_fill_global_kernel<SourceT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        template_fill, orbits_gpu, tdi_config_gpu, wdm_settings_gpu,
        params_all, factors_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tukey_alpha, m_band_half_width);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    (void) grid_dim;
    wdm_het_fill_global_kernel<SourceT>(
        template_fill, orbits, tdi_config, wdm_settings,
        params_all, factors_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tukey_alpha, m_band_half_width);
#endif
}


template <class SourceT>
static void wdm_het_get_ll_impl(
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
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups,
    int m_band_half_width)
{
    // New kernel does not use group-grouping path: each block handles one
    // binary and determines its own narrow m-band internally. Layer-grouping
    // can be reintroduced later as a perf optimization.
    // N_cp_sig still unused; N_cp_orbit now wired into the kernel for the
    // shared-mem orbit spline cache (raw orbits when 0).
    (void) N_cp_sig;
    (void) binary_perm; (void) group_starts; (void) group_ends;
    (void) group_m_lo; (void) group_m_hi; (void) n_groups;
#ifdef __CUDACC__
    // One binary per block on grid.X (always). The grid_dim arg is
    // vestigial -- kept in the signature for ABI stability with the
    // older callers, but the kernel does not grid-stride over binaries.
    (void) grid_dim;
    const int gd_x = num_bin;
    // Shared-mem layout (must match wdm_het_get_ll_kernel):
    //   fd_chunk_buf [nchannels * N_sparse] cmplx  -- chunk-FD (built ONCE per chunk)
    //   layer_buf    [nchannels * Nt_sub]   cmplx  -- per-m_layer scratch
    //   partial_dh   [blockDim.x]           double
    //   partial_hh   [blockDim.x]           double
    //   fft_scratch  [wdm_cufftdx_max_scratch()] bytes  -- 0 unless cufftdx is on
    const size_t shared_bytes =
        (size_t) nchannels * (size_t) N_sparse * sizeof(cmplx) +
        (size_t) nchannels * (size_t) Nt_sub   * sizeof(cmplx) +
        (size_t) 2 * (size_t) NUM_THREADS_HERE * sizeof(double) +
        wdm_cufftdx_max_scratch();

    static Orbits      *orbits_gpu       = nullptr;
    static TDIConfig   *tdi_config_gpu   = nullptr;
    static WDMSettings *wdm_settings_gpu = nullptr;
    if (orbits_gpu       == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,       sizeof(Orbits)));
    if (tdi_config_gpu   == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu,   sizeof(TDIConfig)));
    if (wdm_settings_gpu == nullptr) gpuErrchk(cudaMalloc(&wdm_settings_gpu, sizeof(WDMSettings)));
    gpuErrchk(cudaMemcpy(orbits_gpu,       orbits,       sizeof(Orbits),       cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu,   tdi_config,   sizeof(TDIConfig),    cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(wdm_settings_gpu, wdm_settings, sizeof(WDMSettings), cudaMemcpyHostToDevice));

    // When the kernel's orbit spline cache is enabled, its static shared
    // mem footprint grows by ~26 KB (the orbit_*_buf set). Total can exceed
    // the 48 KB default on sm_70+; opt in to the larger per-block max via
    // cudaFuncSetAttribute before launch.
    if (N_cp_orbit > 0) {
        gpuErrchk(cudaFuncSetAttribute(
            wdm_het_get_ll_kernel<SourceT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
    }

    dim3 grid((unsigned) gd_x, 1u, 1u);
    wdm_het_get_ll_kernel<SourceT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        d_h_out, h_h_out, orbits_gpu, tdi_config_gpu, wdm_settings_gpu,
        params_all, data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, m_band_half_width,
        N_cp_orbit);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    (void) grid_dim;
    wdm_het_get_ll_kernel<SourceT>(
        d_h_out, h_h_out, orbits, tdi_config, wdm_settings,
        params_all, data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, m_band_half_width,
        N_cp_orbit);
#endif
}


template <class SourceT>
static void wdm_het_swap_ll_impl(
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
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups,
    int *pair_m_lo_b, int *pair_m_hi_b,
    int m_band_half_width)
{
    (void) N_cp_sig; (void) N_cp_orbit;
    (void) binary_perm; (void) group_starts; (void) group_ends;
    (void) group_m_lo; (void) group_m_hi; (void) n_groups;
    (void) pair_m_lo_b; (void) pair_m_hi_b;
#ifdef __CUDACC__
    // One binary per block on grid.X (always). The grid_dim arg is
    // vestigial -- kept in the signature for ABI stability with the
    // older callers, but the kernel does not grid-stride over binaries.
    (void) grid_dim;
    const int gd_x = num_bin;
    // Shared-mem layout (must match wdm_het_swap_ll_kernel):
    //   fd_chunk_buf_a [nchannels * N_sparse] cmplx  -- add chunk-FD
    //   fd_chunk_buf_r [nchannels * N_sparse] cmplx  -- rem chunk-FD
    //   layer_buf      [nchannels * Nt_sub]   cmplx  -- per-m scratch (reused)
    //   5 * blockDim.x doubles (dh_a, dh_r, aa, rr, ar partial-sum buffers)
    //   fft_scratch    [wdm_cufftdx_max_scratch()] bytes -- 0 unless cufftdx is on
    const size_t shared_bytes =
        (size_t) 2 * (size_t) nchannels * (size_t) N_sparse * sizeof(cmplx) +
        (size_t) nchannels * (size_t) Nt_sub * sizeof(cmplx) +
        (size_t) 5 * (size_t) NUM_THREADS_HERE * sizeof(double) +
        wdm_cufftdx_max_scratch();

    static Orbits      *orbits_gpu       = nullptr;
    static TDIConfig   *tdi_config_gpu   = nullptr;
    static WDMSettings *wdm_settings_gpu = nullptr;
    if (orbits_gpu       == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,       sizeof(Orbits)));
    if (tdi_config_gpu   == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu,   sizeof(TDIConfig)));
    if (wdm_settings_gpu == nullptr) gpuErrchk(cudaMalloc(&wdm_settings_gpu, sizeof(WDMSettings)));
    gpuErrchk(cudaMemcpy(orbits_gpu,       orbits,       sizeof(Orbits),       cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu,   tdi_config,   sizeof(TDIConfig),    cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(wdm_settings_gpu, wdm_settings, sizeof(WDMSettings), cudaMemcpyHostToDevice));

    dim3 grid((unsigned) gd_x, 1u, 1u);
    wdm_het_swap_ll_kernel<SourceT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits_gpu, tdi_config_gpu, wdm_settings_gpu,
        params_add_all, params_remove_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, m_band_half_width);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    (void) grid_dim;
    wdm_het_swap_ll_kernel<SourceT>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits, tdi_config, wdm_settings,
        params_add_all, params_remove_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, m_band_half_width);
#endif
}


template <class SourceT>
static void wdm_het_get_fstat_ll_impl(
    double *N_arr_re_out, double *N_arr_im_out,   // (num_bin, 4)
    double *M_mat_re_out, double *M_mat_im_out,   // (num_bin, 10)
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
    double T_chunk, double dt, double T, double t_ref,
    int    tdi_type,
    double tukey_alpha,
    int grid_dim,
    int m_band_half_width)
{
#ifdef __CUDACC__
    // One binary per block on grid.X (always). The grid_dim arg is
    // vestigial -- kept in the signature for ABI stability with the
    // older callers, but the kernel does not grid-stride over binaries.
    (void) grid_dim;
    const int gd_x = num_bin;
    // Shared-mem layout (must match wdm_het_get_fstat_ll_kernel):
    //   fd_chunk_buf[fi=0..3][nchannels * N_sparse] cmplx -- per-filter chunk-FD
    //   layer_buf            [nchannels * Nt_sub]   cmplx -- per-(m, fi) scratch
    //   partial_N            [ 4 * blockDim.x]      double (4 basis filters)
    //   partial_M            [10 * blockDim.x]      double (upper-tri of 4x4 M)
    //   fft_scratch          [wdm_cufftdx_max_scratch()] bytes -- 0 unless cufftdx is on
    // ~67 KB at Nt_sub=N_sparse=256, blockDim=64 -- exceeds default 48 KB
    // limit, so we raise the cap via cudaFuncSetAttribute below.
    constexpr int N_FILTERS_LAUNCH = 4;
    const size_t shared_bytes =
        (size_t) N_FILTERS_LAUNCH * (size_t) nchannels * (size_t) N_sparse * sizeof(cmplx) +
        (size_t) nchannels * (size_t) Nt_sub * sizeof(cmplx) +
        (size_t) 14        * (size_t) NUM_THREADS_HERE * sizeof(double) +
        wdm_cufftdx_max_scratch();
    if (shared_bytes > 48u * 1024u) {
        gpuErrchk(cudaFuncSetAttribute(
            wdm_het_get_fstat_ll_kernel<SourceT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int) shared_bytes));
    }

    static Orbits      *orbits_gpu       = nullptr;
    static TDIConfig   *tdi_config_gpu   = nullptr;
    static WDMSettings *wdm_settings_gpu = nullptr;
    if (orbits_gpu       == nullptr) gpuErrchk(cudaMalloc(&orbits_gpu,       sizeof(Orbits)));
    if (tdi_config_gpu   == nullptr) gpuErrchk(cudaMalloc(&tdi_config_gpu,   sizeof(TDIConfig)));
    if (wdm_settings_gpu == nullptr) gpuErrchk(cudaMalloc(&wdm_settings_gpu, sizeof(WDMSettings)));
    gpuErrchk(cudaMemcpy(orbits_gpu,       orbits,       sizeof(Orbits),       cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tdi_config_gpu,   tdi_config,   sizeof(TDIConfig),    cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(wdm_settings_gpu, wdm_settings, sizeof(WDMSettings), cudaMemcpyHostToDevice));

    dim3 grid((unsigned) gd_x, 1u, 1u);
    wdm_het_get_fstat_ll_kernel<SourceT><<<grid, NUM_THREADS_HERE, shared_bytes>>>(
        N_arr_re_out, N_arr_im_out, M_mat_re_out, M_mat_im_out,
        orbits_gpu, tdi_config_gpu, wdm_settings_gpu,
        params_all, data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, m_band_half_width);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
#else
    (void) grid_dim;
    wdm_het_get_fstat_ll_kernel<SourceT>(
        N_arr_re_out, N_arr_im_out, M_mat_re_out, M_mat_im_out,
        orbits, tdi_config, wdm_settings,
        params_all, data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, m_band_half_width);
#endif
}


// ---- GB-flavored wrappers --------------------------------------------------
void GBComputationGroup::gb_wdm_het_fill_global_wrap(
    double *template_fill, Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all, double *factors_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset, double *wdm_window,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    int m_band_half_width)
{
    wdm_het_fill_global_impl<GBTDIonTheFly>(
        template_fill, orbits, tdi_config,
        wdm_settings,
        params_all, factors_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk, T_chunk, dt, T, t_ref, tukey_alpha,
        grid_dim, N_cp_sig, N_cp_orbit, m_band_half_width);
}

void GBComputationGroup::gb_wdm_het_get_ll_wrap(
    double *d_h_out, double *h_h_out, Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all, int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset, double *wdm_window,
    double *data_d, double *invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref, int tdi_type,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups,
    int m_band_half_width)
{
    wdm_het_get_ll_impl<GBTDIonTheFly>(
        d_h_out, h_h_out, orbits, tdi_config,
        wdm_settings,
        params_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC, n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha,
        grid_dim, N_cp_sig, N_cp_orbit,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups, m_band_half_width);
}

void GBComputationGroup::gb_wdm_het_swap_ll_wrap(
    double *d_h_add_out, double *d_h_remove_out,
    double *add_add_out, double *remove_remove_out, double *add_remove_out,
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset, double *wdm_window,
    double *data_d, double *invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref, int tdi_type,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups,
    int *pair_m_lo_b, int *pair_m_hi_b,
    int m_band_half_width)
{
    wdm_het_swap_ll_impl<GBTDIonTheFly>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits, tdi_config,
        wdm_settings,
        params_add_all, params_remove_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC, n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha,
        grid_dim, N_cp_sig, N_cp_orbit,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups,
        pair_m_lo_b, pair_m_hi_b, m_band_half_width);
}


void GBComputationGroup::gb_wdm_het_get_fstat_ll_wrap(
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
    int grid_dim, int m_band_half_width)
{
    wdm_het_get_fstat_ll_impl<GBTDIonTheFly>(
        N_arr_re_out, N_arr_im_out, M_mat_re_out, M_mat_im_out,
        orbits, tdi_config, wdm_settings,
        params_all, data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha,
        grid_dim, m_band_half_width);
}


// ---- SOBBH-flavored wrappers ----------------------------------------------
void SOBBHComputationGroup::sobbh_wdm_het_fill_global_wrap(
    double *template_fill, Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all, double *factors_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset, double *wdm_window,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit)
{
    wdm_het_fill_global_impl<SOBBHTDIonTheFly>(
        template_fill, orbits, tdi_config,
        wdm_settings,
        params_all, factors_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk, T_chunk, dt, T, t_ref, tukey_alpha,
        grid_dim, N_cp_sig, N_cp_orbit, /*m_band_half_width=*/1);
}

void SOBBHComputationGroup::sobbh_wdm_het_get_ll_wrap(
    double *d_h_out, double *h_h_out, Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all, int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset, double *wdm_window,
    double *data_d, double *invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref, int tdi_type,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups)
{
    wdm_het_get_ll_impl<SOBBHTDIonTheFly>(
        d_h_out, h_h_out, orbits, tdi_config,
        wdm_settings,
        params_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC, n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha,
        grid_dim, N_cp_sig, N_cp_orbit,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups, /*m_band_half_width=*/1);
}

void SOBBHComputationGroup::sobbh_wdm_het_swap_ll_wrap(
    double *d_h_add_out, double *d_h_remove_out,
    double *add_add_out, double *remove_remove_out, double *add_remove_out,
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset, double *wdm_window,
    double *data_d, double *invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref, int tdi_type,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups,
    int *pair_m_lo_b, int *pair_m_hi_b)
{
    wdm_het_swap_ll_impl<SOBBHTDIonTheFly>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits, tdi_config,
        wdm_settings,
        params_add_all, params_remove_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC, n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha,
        grid_dim, N_cp_sig, N_cp_orbit,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups,
        pair_m_lo_b, pair_m_hi_b, /*m_band_half_width=*/1);
}


void SOBBHComputationGroup::sobbh_wdm_het_get_fstat_ll_wrap(
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
    int grid_dim, int m_band_half_width)
{
    wdm_het_get_fstat_ll_impl<SOBBHTDIonTheFly>(
        N_arr_re_out, N_arr_im_out, M_mat_re_out, M_mat_im_out,
        orbits, tdi_config, wdm_settings,
        params_all, data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha,
        grid_dim, m_band_half_width);
}


// ============================================================================
// Signal-heterodyne (v2 polyphase) -- CPU implementation
//
// First port of the v2 polyphase signal-het Python prototype at
// LISAanalysistools/scripts/gb_chunked_het/gb_signal_het_wdm_v2.py.
//
// Algorithm per binary (matches Python prototype exactly):
//
//   1. f0_cand = params_cand[bin, f0_idx]
//      m_floor = floor(f0_cand / layer_df)
//      m_active = [m_floor - half, ..., m_floor + half], clipped to active band
//
//   2. For each m_active layer:
//        Polyphase fold + iFFT of length Nt_layer over the windowed FD slice
//        around bin (m * Nt/2), with pre-phase shift to land outputs at
//        n_global = n_start + n_layer * stride.
//        Apply lisatools complex-WDM coefficient layout
//          kappa * (-1)^((m+1)n) * conj(C_mn) * after_ifft * (-1)^n / stride
//        to produce c1_sparse[c, m_active_idx, n_layer].
//
//   3. r[c, m, b] = c1_sparse[c, m_active_idx, b] / c0_sparse[data_idx,
//                  c, m_local_in_full, b]    (safe divide with floor mask).
//      dr/dn via centred FD over b with mean bin width = stride.
//
//   4. Bin-folded inner products (NO carrier de-rotation -- matches the
//      Python v1 sparse path):
//        <d|h> = 0.5 * Re sum_{c',m_act,b} (A0 * r + A1 * dr/dn)
//        <h|h> = 0.5 * Re sum (B0 * r_outer + B1 * cross_drr)
//      with r_outer / cross_drr handled differently for XYZ vs AE/AET.
//
// GPU branch: TODO (prints and returns; CPU fully wired).
// ============================================================================

namespace {

// Per-binary, single channel: compute c1_sparse[m_active_layers, Nt_layer]
// via polyphase fold + naive O(Nt_layer^2) DFT-of-length-Nt_layer.
// Naive DFT is sufficient for correctness validation; swap for radix-2 FFT
// later for performance.
static void signal_het_polyphase_one_channel(
    const cmplx *fd_rfft_chan,            // (n_rfft,) complex
    const int   *m_active,                // (m_active_layers,)
    int          m_active_layers,
    const double *window,                 // (Nt,)
    int          Nt,
    int          Nt_layer,                // iFFT length
    int          N_sparse_t,              // number of sparse outputs kept (<= Nt_layer)
    int          stride,
    int          Nf,
    int          ind_min_t,
    const int   *n_sparse_local_arr,      // (N_sparse_t,) -- only first entry used for n_start
    double       dt,
    int          n_rfft,
    cmplx       *c1_sparse_out)           // (m_active_layers, N_sparse_t)
{
    const int   N        = Nf * Nt;
    const int   half_Nt  = Nt / 2;
    const cmplx I_c      = cmplx(0.0, 1.0);
    const double TWO_PI  = 2.0 * M_PI;
    const double kappa   = 2.0 * std::sqrt(M_PI * dt) / (double) Nf;

    // n_start = first sparse position (ind_min_t + stride/2 by construction).
    const int n_start = ind_min_t + n_sparse_local_arr[0];

    // Working buffers on the stack (per binary x per channel).
    // Max sane Nt for CPU is ~32k, Nt_layer ~ 2k.
    std::vector<cmplx> weighted((size_t) Nt);
    std::vector<cmplx> folded((size_t) Nt_layer);

    for (int im = 0; im < m_active_layers; ++im) {
        const int m_global = m_active[im];
        const int centre   = m_global * half_Nt;

        // Step 1: gather Nt FD bins around the layer centre, with Hermitian
        // wraparound (real TD -> rfft conjugate symmetry).
        for (int i = 0; i < Nt; ++i) {
            const int j_off = i - half_Nt;
            int k = centre + j_off;
            bool conj_flag = false;
            int k_use;
            if (k < 0) {
                k_use = -k;
                conj_flag = true;
            } else if (k > N / 2) {
                k_use = N - k;
                conj_flag = true;
            } else {
                k_use = k;
            }

            cmplx h(0.0, 0.0);
            if (k_use >= 0 && k_use < n_rfft) {
                h = fd_rfft_chan[k_use];
                if (conj_flag) h = gcmplx::conj(h);
            }

            // Window + prephase: phitilde(j_off) * exp(+i 2*pi*j_off*n_start/Nt)
            const double phase_arg = TWO_PI * (double) j_off * (double) n_start / (double) Nt;
            const cmplx  prephase  = gcmplx::exp(I_c * phase_arg);
            weighted[i] = h * window[i] * prephase;
        }

        // Step 2: polyphase fold (length Nt -> length Nt_layer).
        for (int r = 0; r < Nt_layer; ++r) folded[r] = cmplx(0.0, 0.0);
        for (int i = 0; i < Nt; ++i) {
            const int r = i % Nt_layer;
            folded[r] += weighted[i];
        }

        // Step 3: naive iFFT of length Nt_layer (matches numpy ifft conv).
        //   ifft_out[n_layer] = (1/Nt_layer) * sum_r Y[r] * exp(+i 2*pi*r*n_layer/Nt_layer)
        // We only need the first N_sparse_t outputs (sparse positions tile
        // the active n-range; the polyphase identity is exact for all of them).
        for (int n_layer = 0; n_layer < N_sparse_t; ++n_layer) {
            cmplx acc(0.0, 0.0);
            for (int r = 0; r < Nt_layer; ++r) {
                const double pa = TWO_PI * (double) r * (double) n_layer / (double) Nt_layer;
                acc += folded[r] * gcmplx::exp(I_c * pa);
            }
            acc *= (1.0 / (double) Nt_layer);

            // Lisatools complex-WDM conversion at the sparse pixel n_global =
            // n_start + n_layer * stride.
            const int n_global = n_start + n_layer * stride;
            // sign_scale = (-1)^n_global / stride
            const double sign_scale = ((n_global & 1) ? -1.0 : 1.0) / (double) stride;
            const cmplx after_ifft_lt = acc * sign_scale;

            // kappa * (-1)^((m+1)n) * conj(C_mn) where C_mn = 1 (even m+n)
            // or 1j (odd m+n); conj(C_mn) = 1 (even) or -1j (odd).
            const int  m_plus_n  = (m_global + n_global) & 1;
            const cmplx conj_cmn = (m_plus_n == 0) ? cmplx(1.0, 0.0)
                                                   : cmplx(0.0, -1.0);
            const int  sign_mn_int = ((m_global + 1) * n_global) & 1;
            const double sign_mn   = sign_mn_int ? -1.0 : 1.0;
            const cmplx coef = kappa * sign_mn * conj_cmn;

            c1_sparse_out[(size_t) im * N_sparse_t + n_layer] = after_ifft_lt * coef;
        }
    }
}

}  // anonymous namespace

void GBComputationGroup::gb_signal_het_get_ll_wrap(
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
    int     n_rfft, double max_r)
{
    (void) params_ref_all;  // not used in bin-fold path (kept for future de-rotation)
    (void) fdot_idx;
    (void) num_data;
    (void) Nt_active;

#ifdef __CUDACC__
    // GPU port deferred. CPU branch fully wired below; matches the pattern
    // of gb_fd_swap_ll_grad_wrap (header comment line 861-862).
    std::fprintf(stderr, "[gb_signal_het_get_ll_wrap] GPU branch TODO -- "
                         "use CPU backend (force_backend=\"cpu\")\n");
    return;
#endif

    const int M = 2 * m_active_half_width + 1;
    const int Nf_active_idx_max = Nf_active - 1;
    const double FLOOR_EPS = 1e-12;

    std::vector<cmplx> c1_sparse((size_t) nchannels * M * N_sparse_t);
    std::vector<cmplx> r_sparse((size_t)  nchannels * M * N_sparse_t);
    std::vector<cmplx> dr_sparse((size_t) nchannels * M * N_sparse_t);

    for (int bin = 0; bin < num_bin; ++bin) {
        const double f0_cand = params_cand_all[(size_t) bin * nparams + f0_idx];
        const int    m_floor = (int) std::floor(f0_cand / layer_df);
        int m_active[16];
        for (int im = 0; im < M; ++im) {
            int m_g = m_floor + (im - m_active_half_width);
            if (m_g < ind_min_f) m_g = ind_min_f;
            if (m_g > ind_min_f + Nf_active_idx_max) m_g = ind_min_f + Nf_active_idx_max;
            m_active[im] = m_g;
        }

        const int data_idx = data_index_all[bin];

        for (int c = 0; c < nchannels; ++c) {
            const cmplx *fd_chan = fd_rfft_all + (size_t) bin * nchannels * n_rfft
                                              + (size_t) c * n_rfft;
            cmplx *c1_chan = c1_sparse.data()
                + (size_t) c * M * N_sparse_t;
            signal_het_polyphase_one_channel(
                fd_chan, m_active, M, wdm_window, Nt, Nt_layer, N_sparse_t,
                stride, Nf, ind_min_t, n_sparse_local_arr, dt, n_rfft, c1_chan);
        }

        // r at sparse bin centres (safe divide vs c0)
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const int m_local = m_active[im] - ind_min_f;
                double max_mag = 0.0;
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx c0v = c0_sparse_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                        + (size_t) m_local * N_sparse_t + b];
                    const double mag = gcmplx::abs(c0v);
                    if (mag > max_mag) max_mag = mag;
                }
                const double floor_th = std::max(FLOOR_EPS * max_mag, 1e-300);

                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx c0v = c0_sparse_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                        + (size_t) m_local * N_sparse_t + b];
                    const cmplx c1v = c1_sparse[
                        (size_t) c * M * N_sparse_t
                        + (size_t) im * N_sparse_t + b];
                    const size_t r_idx = (size_t) c * M * N_sparse_t
                                       + (size_t) im * N_sparse_t + b;
                    if (gcmplx::abs(c0v) > floor_th) {
                        cmplx r_val = c1v / c0v;
                        // Amp/phase clip: cap |r| at max_r (channel-cell
                        // direction preserved). Bounds the bin-fold sum
                        // when c0 is small but nonzero. max_r <= 0 disables.
                        if (max_r > 0.0) {
                            const double abs_r = gcmplx::abs(r_val);
                            if (abs_r > max_r) {
                                r_val = r_val * (max_r / abs_r);
                            }
                        }
                        r_sparse[r_idx] = r_val;
                    } else {
                        r_sparse[r_idx] = cmplx(0.0, 0.0);
                    }
                }
            }
        }

        // dr/dn via centred FD over b
        const double Dn = (double) stride;
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                for (int b = 0; b < N_sparse_t; ++b) {
                    const size_t i_cmb = (size_t) c * M * N_sparse_t
                                       + (size_t) im * N_sparse_t + b;
                    cmplx d(0.0, 0.0);
                    if (N_sparse_t >= 3) {
                        if (b == 0) {
                            d = (r_sparse[i_cmb + 1] - r_sparse[i_cmb]) / Dn;
                        } else if (b == N_sparse_t - 1) {
                            d = (r_sparse[i_cmb] - r_sparse[i_cmb - 1]) / Dn;
                        } else {
                            d = (r_sparse[i_cmb + 1] - r_sparse[i_cmb - 1]) / (2.0 * Dn);
                        }
                    } else if (N_sparse_t == 2) {
                        const size_t i0 = (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t;
                        d = (r_sparse[i0 + 1] - r_sparse[i0]) / Dn;
                    }
                    dr_sparse[i_cmb] = d;
                }
            }
        }

        // Inner products
        cmplx d_h_raw(0.0, 0.0);
        cmplx h_h_raw(0.0, 0.0);

        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const int m_local = m_active[im] - ind_min_f;
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx r  = r_sparse[ (size_t) c * M * N_sparse_t
                                            + (size_t) im * N_sparse_t + b];
                    const cmplx dr = dr_sparse[(size_t) c * M * N_sparse_t
                                            + (size_t) im * N_sparse_t + b];
                    const cmplx a0 = A0_all[((size_t) data_idx * nchannels + c)
                                            * Nf_active * N_sparse_t
                                            + (size_t) m_local * N_sparse_t + b];
                    const cmplx a1 = A1_all[((size_t) data_idx * nchannels + c)
                                            * Nf_active * N_sparse_t
                                            + (size_t) m_local * N_sparse_t + b];
                    d_h_raw += a0 * r + a1 * dr;
                }
            }
        }

        if (tdi_type == 0) {
            // XYZ: cross-channel B0/B1 of shape (num_data, nch, nch, Nf_active, Nt_layer)
            for (int c = 0; c < nchannels; ++c) {
                for (int c2 = 0; c2 < nchannels; ++c2) {
                    for (int im = 0; im < M; ++im) {
                        const int m_local = m_active[im] - ind_min_f;
                        for (int b = 0; b < N_sparse_t; ++b) {
                            const cmplx r_c  = r_sparse[ (size_t) c  * M * N_sparse_t
                                                     + (size_t) im * N_sparse_t + b];
                            const cmplx r_c2 = r_sparse[ (size_t) c2 * M * N_sparse_t
                                                     + (size_t) im * N_sparse_t + b];
                            const cmplx dr_c  = dr_sparse[(size_t) c  * M * N_sparse_t
                                                     + (size_t) im * N_sparse_t + b];
                            const cmplx dr_c2 = dr_sparse[(size_t) c2 * M * N_sparse_t
                                                     + (size_t) im * N_sparse_t + b];
                            const cmplx b0 = B0_all[
                                (((size_t) data_idx * nchannels + c) * nchannels + c2)
                                  * Nf_active * N_sparse_t
                                + (size_t) m_local * N_sparse_t + b];
                            const cmplx b1 = B1_all[
                                (((size_t) data_idx * nchannels + c) * nchannels + c2)
                                  * Nf_active * N_sparse_t
                                + (size_t) m_local * N_sparse_t + b];
                            const cmplx r_outer  = gcmplx::conj(r_c) * r_c2;
                            const cmplx cross_drr = gcmplx::conj(r_c)  * dr_c2
                                                  + gcmplx::conj(dr_c) * r_c2;
                            h_h_raw += b0 * r_outer + b1 * cross_drr;
                        }
                    }
                }
            }
        } else {
            // AE / AET: diagonal B0/B1 of shape (num_data, nch, Nf_active, Nt_layer)
            for (int c = 0; c < nchannels; ++c) {
                for (int im = 0; im < M; ++im) {
                    const int m_local = m_active[im] - ind_min_f;
                    for (int b = 0; b < N_sparse_t; ++b) {
                        const cmplx r  = r_sparse[ (size_t) c * M * N_sparse_t
                                                 + (size_t) im * N_sparse_t + b];
                        const cmplx dr = dr_sparse[(size_t) c * M * N_sparse_t
                                                 + (size_t) im * N_sparse_t + b];
                        const cmplx b0 = B0_all[((size_t) data_idx * nchannels + c)
                                                * Nf_active * N_sparse_t
                                                + (size_t) m_local * N_sparse_t + b];
                        const cmplx b1 = B1_all[((size_t) data_idx * nchannels + c)
                                                * Nf_active * N_sparse_t
                                                + (size_t) m_local * N_sparse_t + b];
                        const double rsq = (gcmplx::conj(r) * r).real();
                        const cmplx cross_drr = gcmplx::conj(r) * dr
                                              + gcmplx::conj(dr) * r;
                        h_h_raw += b0 * rsq + b1 * cross_drr;
                    }
                }
            }
        }

        d_h_out[bin] = 0.5 * d_h_raw.real();
        h_h_out[bin] = 0.5 * h_h_raw.real();
    }
}


// ============================================================================
// Signal-heterodyne (v2 polyphase) -- Stage 2a: SPARSE-FD entry point.
// See gb_signal_het_get_ll_sparse_wrap declaration in TDIonTheFly.hh for the
// design rationale. Polyphase fold iterates only the N_sparse_fd nonzero
// bins (the source's spectral support around f0 in absolute frame);
// implicit zero everywhere else.
// ============================================================================

void GBComputationGroup::gb_signal_het_get_ll_sparse_wrap(
    double *d_h_out, double *h_h_out,
    cmplx  *X_het_all, int *k_f0_all,
    cmplx  *c0_sparse_all,
    cmplx  *A0_all, cmplx *A1_all,
    cmplx  *B0_all, cmplx *B1_all,
    double *wdm_window, int *n_sparse_local_arr,
    double *params_cand_all, double *params_ref_all,
    int    *data_index_all,
    int     num_bin, int num_data,
    int     nparams, int f0_idx, int fdot_idx,
    int     Nf, int Nt, int Nf_active, int Nt_active,
    int     Nt_layer, int N_sparse_t, int stride,
    int     ind_min_t, int ind_min_f,
    int     m_active_half_width,
    double  layer_df, double dt,
    int     nchannels, int tdi_type,
    int     N_sparse_fd, double max_r)
{
    (void) params_ref_all; (void) fdot_idx; (void) num_data; (void) Nt_active;

#ifdef __CUDACC__
    std::fprintf(stderr, "[gb_signal_het_get_ll_sparse_wrap] GPU branch TODO -- "
                         "use CPU backend (force_backend=\"cpu\")\n");
    return;
#endif

    const int M = 2 * m_active_half_width + 1;
    const int Nf_active_idx_max = Nf_active - 1;
    const double FLOOR_EPS = 1e-12;
    const double TWO_PI = 2.0 * M_PI;
    const cmplx I_c = cmplx(0.0, 1.0);
    const int   half_Nt = Nt / 2;
    const int   half_NS = N_sparse_fd / 2;
    const double kappa = 2.0 * std::sqrt(M_PI * dt) / (double) Nf;
    const int   n_start = ind_min_t + n_sparse_local_arr[0];

    std::vector<cmplx> fold((size_t) nchannels * M * Nt_layer);
    std::vector<cmplx> c1_sparse((size_t) nchannels * M * N_sparse_t);
    std::vector<cmplx> r_sparse((size_t)  nchannels * M * N_sparse_t);
    std::vector<cmplx> dr_sparse((size_t) nchannels * M * N_sparse_t);

    for (int bin = 0; bin < num_bin; ++bin) {
        const double f0_cand = params_cand_all[(size_t) bin * nparams + f0_idx];
        const int    m_floor = (int) std::floor(f0_cand / layer_df);
        int m_active[16];
        for (int im = 0; im < M; ++im) {
            int m_g = m_floor + (im - m_active_half_width);
            if (m_g < ind_min_f) m_g = ind_min_f;
            if (m_g > ind_min_f + Nf_active_idx_max) m_g = ind_min_f + Nf_active_idx_max;
            m_active[im] = m_g;
        }
        const int data_idx = data_index_all[bin];
        const int k_f0     = k_f0_all[bin];

        // Polyphase fold: iterate only N_sparse_fd nonzero bins.
        std::fill(fold.begin(), fold.end(), cmplx(0.0, 0.0));
        for (int c = 0; c < nchannels; ++c) {
            const cmplx *X_chan = X_het_all + (size_t) bin * nchannels * N_sparse_fd
                                            + (size_t) c * N_sparse_fd;
            for (int i = 0; i < N_sparse_fd; ++i) {
                const cmplx Xi = X_chan[i];
                if (Xi.real() == 0.0 && Xi.imag() == 0.0) continue;
                const int k_abs = k_f0 + (i - half_NS);
                for (int im = 0; im < M; ++im) {
                    const int j = k_abs - m_active[im] * half_Nt + half_Nt;
                    if (j < 0 || j >= Nt) continue;
                    const int j_off = j - half_Nt;
                    const double phase_arg = TWO_PI * (double) j_off
                                             * (double) n_start / (double) Nt;
                    const cmplx prephase = gcmplx::exp(I_c * phase_arg);
                    const cmplx weighted = Xi * wdm_window[j] * prephase;
                    const int r = j % Nt_layer;
                    fold[(size_t) c * M * Nt_layer
                       + (size_t) im * Nt_layer + r] += weighted;
                }
            }
        }

        // iFFT of length Nt_layer (naive DFT; keep first N_sparse_t outputs).
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const cmplx *fold_cm = fold.data()
                    + (size_t) c * M * Nt_layer + (size_t) im * Nt_layer;
                for (int n_layer = 0; n_layer < N_sparse_t; ++n_layer) {
                    cmplx acc(0.0, 0.0);
                    for (int rr = 0; rr < Nt_layer; ++rr) {
                        const double pa = TWO_PI * (double) rr
                                          * (double) n_layer / (double) Nt_layer;
                        acc += fold_cm[rr] * gcmplx::exp(I_c * pa);
                    }
                    acc *= (1.0 / (double) Nt_layer);
                    const int n_global = n_start + n_layer * stride;
                    const double sign_scale = ((n_global & 1) ? -1.0 : 1.0)
                                              / (double) stride;
                    const cmplx after_ifft_lt = acc * sign_scale;
                    const int  m_global = m_active[im];
                    const int  m_plus_n = (m_global + n_global) & 1;
                    const cmplx conj_cmn = (m_plus_n == 0) ? cmplx(1.0, 0.0)
                                                           : cmplx(0.0, -1.0);
                    const int  sign_mn_int = ((m_global + 1) * n_global) & 1;
                    const double sign_mn = sign_mn_int ? -1.0 : 1.0;
                    const cmplx coef = kappa * sign_mn * conj_cmn;
                    c1_sparse[(size_t) c * M * N_sparse_t
                            + (size_t) im * N_sparse_t + n_layer] = after_ifft_lt * coef;
                }
            }
        }

        // r, dr, inner products: same as Stage 1.
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const int m_local = m_active[im] - ind_min_f;
                double max_mag = 0.0;
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx c0v = c0_sparse_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                        + (size_t) m_local * N_sparse_t + b];
                    const double mag = gcmplx::abs(c0v);
                    if (mag > max_mag) max_mag = mag;
                }
                const double floor_th = std::max(FLOOR_EPS * max_mag, 1e-300);
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx c0v = c0_sparse_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                        + (size_t) m_local * N_sparse_t + b];
                    const cmplx c1v = c1_sparse[
                        (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const size_t r_idx = (size_t) c * M * N_sparse_t
                                       + (size_t) im * N_sparse_t + b;
                    if (gcmplx::abs(c0v) > floor_th) {
                        cmplx r_val = c1v / c0v;
                        // Amp/phase clip: cap |r| at max_r (preserve dir).
                        if (max_r > 0.0) {
                            const double abs_r = gcmplx::abs(r_val);
                            if (abs_r > max_r) r_val = r_val * (max_r / abs_r);
                        }
                        r_sparse[r_idx] = r_val;
                    } else {
                        r_sparse[r_idx] = cmplx(0.0, 0.0);
                    }
                }
            }
        }

        const double Dn = (double) stride;
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                for (int b = 0; b < N_sparse_t; ++b) {
                    const size_t i_cmb = (size_t) c * M * N_sparse_t
                                       + (size_t) im * N_sparse_t + b;
                    cmplx d(0.0, 0.0);
                    if (N_sparse_t >= 3) {
                        if (b == 0) d = (r_sparse[i_cmb + 1] - r_sparse[i_cmb]) / Dn;
                        else if (b == N_sparse_t - 1) d = (r_sparse[i_cmb] - r_sparse[i_cmb - 1]) / Dn;
                        else d = (r_sparse[i_cmb + 1] - r_sparse[i_cmb - 1]) / (2.0 * Dn);
                    } else if (N_sparse_t == 2) {
                        const size_t i0 = (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t;
                        d = (r_sparse[i0 + 1] - r_sparse[i0]) / Dn;
                    }
                    dr_sparse[i_cmb] = d;
                }
            }
        }

        cmplx d_h_raw(0.0, 0.0), h_h_raw(0.0, 0.0);
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const int m_local = m_active[im] - ind_min_f;
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx r  = r_sparse[ (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const cmplx dr = dr_sparse[(size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const cmplx a0 = A0_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                    const cmplx a1 = A1_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                    d_h_raw += a0 * r + a1 * dr;
                }
            }
        }
        if (tdi_type == 0) {
            for (int c = 0; c < nchannels; ++c) for (int c2 = 0; c2 < nchannels; ++c2)
                for (int im = 0; im < M; ++im) {
                    const int m_local = m_active[im] - ind_min_f;
                    for (int b = 0; b < N_sparse_t; ++b) {
                        const cmplx r_c  = r_sparse[(size_t) c  * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                        const cmplx r_c2 = r_sparse[(size_t) c2 * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                        const cmplx dr_c  = dr_sparse[(size_t) c  * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                        const cmplx dr_c2 = dr_sparse[(size_t) c2 * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                        const cmplx b0 = B0_all[(((size_t) data_idx * nchannels + c) * nchannels + c2) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                        const cmplx b1 = B1_all[(((size_t) data_idx * nchannels + c) * nchannels + c2) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                        const cmplx r_outer = gcmplx::conj(r_c) * r_c2;
                        const cmplx cross_drr = gcmplx::conj(r_c) * dr_c2 + gcmplx::conj(dr_c) * r_c2;
                        h_h_raw += b0 * r_outer + b1 * cross_drr;
                    }
                }
        } else {
            for (int c = 0; c < nchannels; ++c) for (int im = 0; im < M; ++im) {
                const int m_local = m_active[im] - ind_min_f;
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx r  = r_sparse[ (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const cmplx dr = dr_sparse[(size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const cmplx b0 = B0_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                    const cmplx b1 = B1_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                    const double rsq = (gcmplx::conj(r) * r).real();
                    const cmplx cross_drr = gcmplx::conj(r) * dr + gcmplx::conj(dr) * r;
                    h_h_raw += b0 * rsq + b1 * cross_drr;
                }
            }
        }

        d_h_out[bin] = 0.5 * d_h_raw.real();
        h_h_out[bin] = 0.5 * h_h_raw.real();
    }
}



// ============================================================================
// Signal-heterodyne (v2 polyphase) -- Stage 2b: IN-KERNEL sparse-FD entry.
//
// Fuses the existing gb_run_fd_wave_tdi_wrap (sparse heterodyned rfft) with
// the Stage 2a polyphase + bin-fold pipeline. Per-source X_het is allocated
// transiently inside this call (heap on CPU; per-block shared memory on GPU
// at Stage 3). NO per-source FD storage in global memory.
//
// CPU implementation: two-pass for clarity --
//   (1) gb_run_fd_wave_tdi_wrap fills X_het + k_f0 buffers
//   (2) gb_signal_het_get_ll_sparse_wrap consumes those buffers
//
// GPU Stage 3 will fuse these into a single kernel with X_het in __shared__.
// ============================================================================

void GBComputationGroup::gb_signal_het_get_ll_in_kernel_wrap(
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
    int     N_sparse_fd, double tukey_alpha, double max_r)
{
#ifdef __CUDACC__
    std::fprintf(stderr, "[gb_signal_het_get_ll_in_kernel_wrap] GPU branch "
                         "TODO -- use CPU backend (force_backend=\"cpu\")\n");
    return;
#endif

    std::vector<cmplx>  X_het_raw((size_t) num_bin * nchannels * N_sparse_fd);
    std::vector<int>    k_f0_buf(num_bin);
    std::vector<double> f0_grid_buf(num_bin);

    gb_run_fd_wave_tdi_wrap(
        tdi_on_fly,
        X_het_raw.data(), k_f0_buf.data(), f0_grid_buf.data(),
        params_cand_all, t_start, T_obs,
        N_sparse_fd, num_bin, nparams, nchannels,
        tukey_alpha);

    // Convert gb_run_fd_wave_tdi output to the centered-slice / dense-rfft
    // convention that gb_signal_het_get_ll_sparse_wrap expects:
    //   * raw layout: X_het_raw[b,c,m_fft] = FFT-order, carrier-removed
    //     sparse FFT, scaled by 0.5*dts where dts = T_obs/N_sparse_fd.
    //   * target:     X_het[b,c,i] = dense_rfft(Tukey*td)[k_f0 + (i - half_NS)],
    //                  i.e. centered slice of the absolute (carrier-intact)
    //                  dense rfft.
    // The continuous-FT representations differ by a 0.5 factor that is
    // already absorbed in the raw 0.5*dts scale, leaving only the
    // sparse-to-dense Riemann conversion (1/dt) and the FFT-order ->
    // centered-slice reordering (an fftshift). Both signals share the
    // t_start time origin so there is no extra linear-phase factor.
    // Empirical bin-by-bin agreement is ~1% (Tukey vs no-window edge bias)
    // which the polyphase fold averages out at the inner-product level.
    std::vector<cmplx> X_het((size_t) num_bin * nchannels * N_sparse_fd);
    const int    half_NS = N_sparse_fd / 2;
    const double dt_inv  = 1.0 / dt;
    for (int b = 0; b < num_bin; ++b) {
        for (int c = 0; c < nchannels; ++c) {
            const size_t base = ((size_t) b * nchannels + c) * N_sparse_fd;
            for (int i = 0; i < N_sparse_fd; ++i) {
                const int m_signed = i - half_NS;
                const int m_fft    = (m_signed >= 0)
                                         ? m_signed
                                         : (m_signed + N_sparse_fd);
                X_het[base + i] = X_het_raw[base + m_fft] * dt_inv;
            }
        }
    }

    this->gb_signal_het_get_ll_sparse_wrap(
        d_h_out, h_h_out,
        X_het.data(), k_f0_buf.data(),
        c0_sparse_all,
        A0_all, A1_all, B0_all, B1_all,
        wdm_window, n_sparse_local_arr,
        params_cand_all, params_ref_all, data_index_all,
        num_bin, num_data,
        nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        nchannels, tdi_type,
        N_sparse_fd, max_r);
}


// ============================================================================
// Signal-heterodyne (v2 polyphase) -- fill_global path.
// ============================================================================
//
// Same FD generation + polyphase + r_sparse machinery as
// gb_signal_het_get_ll_sparse_wrap, but instead of bin-folding r against
// precomputed A0/A1/B0/B1 to produce <d|h>/<h|h>, we reconstruct the dense
// candidate WDM template via the heterodyne identity:
//
//   1. r_sparse(c, m_active, n_sparse) = c1_sparse / c0_sparse
//   2. r_demod_sparse = r_sparse * exp(-i * phase_pred(n_sparse))
//      where phase_pred(t) = 2pi Df0 t + pi Dfdot t^2 from
//      params_cand - params_ref (the KNOWN analytic carrier).
//   3. Linear interpolate r_demod onto dense n.
//   4. r_dense = r_demod_dense * exp(+i * phase_pred(n_dense))  -- re-rotated.
//   5. c1_dense = r_dense * c0_dense_complex  (full active band).
//   6. template_fill[data_idx, c, m_global, n_global] += factor * Re(c1_dense)
//
// c0_dense_complex_all is shape (num_data, nch, Nf_active, Nt_active);
// template_fill is shape (num_data, nch, Nf, Nt) and is accumulated into
// (caller pre-zeroes / atomic-adds to support multiple binaries colliding
// at the same WDM pixel).
// ============================================================================

void GBComputationGroup::gb_signal_het_fill_global_sparse_wrap(
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
    int     N_sparse_fd, double max_r)
{
    (void) num_data;

#ifdef __CUDACC__
    std::fprintf(stderr, "[gb_signal_het_fill_global_sparse_wrap] GPU branch "
                         "TODO -- use CPU backend (force_backend=\"cpu\")\n");
    return;
#endif

    const int M = 2 * m_active_half_width + 1;
    const int Nf_active_idx_max = Nf_active - 1;
    const double FLOOR_EPS = 1e-12;
    const double TWO_PI = 2.0 * M_PI;
    const cmplx I_c = cmplx(0.0, 1.0);
    const int   half_Nt = Nt / 2;
    const int   half_NS = N_sparse_fd / 2;
    const double kappa = 2.0 * std::sqrt(M_PI * dt) / (double) Nf;
    const int   n_start = ind_min_t + n_sparse_local_arr[0];
    const double layer_dt = (double) Nf * dt;

    std::vector<cmplx> fold((size_t) nchannels * M * Nt_layer);
    std::vector<cmplx> c1_sparse((size_t) nchannels * M * N_sparse_t);
    std::vector<cmplx> r_sparse((size_t)  nchannels * M * N_sparse_t);

    for (int bin = 0; bin < num_bin; ++bin) {
        const double f0_cand   = params_cand_all[(size_t) bin * nparams + f0_idx];
        const double fdot_cand = params_cand_all[(size_t) bin * nparams + fdot_idx];
        const int    m_floor   = (int) std::floor(f0_cand / layer_df);
        int m_active[16];
        for (int im = 0; im < M; ++im) {
            int m_g = m_floor + (im - m_active_half_width);
            if (m_g < ind_min_f) m_g = ind_min_f;
            if (m_g > ind_min_f + Nf_active_idx_max) m_g = ind_min_f + Nf_active_idx_max;
            m_active[im] = m_g;
        }
        const int    data_idx = data_index_all[bin];
        const int    k_f0     = k_f0_all[bin];
        const double factor   = factors_all[bin];

        const double f0_ref   = params_ref_all[(size_t) data_idx * nparams + f0_idx];
        const double fdot_ref = params_ref_all[(size_t) data_idx * nparams + fdot_idx];
        const double Df0      = f0_cand   - f0_ref;
        const double Dfdot    = fdot_cand - fdot_ref;

        // ---- Polyphase fold + iFFT + c1_sparse (identical to Stage 2a) ----
        std::fill(fold.begin(), fold.end(), cmplx(0.0, 0.0));
        for (int c = 0; c < nchannels; ++c) {
            const cmplx *X_chan = X_het_all + (size_t) bin * nchannels * N_sparse_fd
                                            + (size_t) c * N_sparse_fd;
            for (int i = 0; i < N_sparse_fd; ++i) {
                const cmplx Xi = X_chan[i];
                if (Xi.real() == 0.0 && Xi.imag() == 0.0) continue;
                const int k_abs = k_f0 + (i - half_NS);
                for (int im = 0; im < M; ++im) {
                    const int j = k_abs - m_active[im] * half_Nt + half_Nt;
                    if (j < 0 || j >= Nt) continue;
                    const int j_off = j - half_Nt;
                    const double phase_arg = TWO_PI * (double) j_off
                                             * (double) n_start / (double) Nt;
                    const cmplx prephase = gcmplx::exp(I_c * phase_arg);
                    const cmplx weighted = Xi * wdm_window[j] * prephase;
                    const int r = j % Nt_layer;
                    fold[(size_t) c * M * Nt_layer
                       + (size_t) im * Nt_layer + r] += weighted;
                }
            }
        }

        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const cmplx *fold_cm = fold.data()
                    + (size_t) c * M * Nt_layer + (size_t) im * Nt_layer;
                for (int n_layer = 0; n_layer < N_sparse_t; ++n_layer) {
                    cmplx acc(0.0, 0.0);
                    for (int rr = 0; rr < Nt_layer; ++rr) {
                        const double pa = TWO_PI * (double) rr
                                          * (double) n_layer / (double) Nt_layer;
                        acc += fold_cm[rr] * gcmplx::exp(I_c * pa);
                    }
                    acc *= (1.0 / (double) Nt_layer);
                    const int n_global = n_start + n_layer * stride;
                    const double sign_scale = ((n_global & 1) ? -1.0 : 1.0)
                                              / (double) stride;
                    const cmplx after_ifft_lt = acc * sign_scale;
                    const int  m_global = m_active[im];
                    const int  m_plus_n = (m_global + n_global) & 1;
                    const cmplx conj_cmn = (m_plus_n == 0) ? cmplx(1.0, 0.0)
                                                           : cmplx(0.0, -1.0);
                    const int  sign_mn_int = ((m_global + 1) * n_global) & 1;
                    const double sign_mn = sign_mn_int ? -1.0 : 1.0;
                    const cmplx coef = kappa * sign_mn * conj_cmn;
                    c1_sparse[(size_t) c * M * N_sparse_t
                            + (size_t) im * N_sparse_t + n_layer] = after_ifft_lt * coef;
                }
            }
        }

        // ---- r_sparse = c1_sparse / c0_sparse (with safe-divide floor) ----
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const int m_local = m_active[im] - ind_min_f;
                double max_mag = 0.0;
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx c0v = c0_sparse_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                        + (size_t) m_local * N_sparse_t + b];
                    const double mag = gcmplx::abs(c0v);
                    if (mag > max_mag) max_mag = mag;
                }
                const double floor_th = std::max(FLOOR_EPS * max_mag, 1e-300);
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx c0v = c0_sparse_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                        + (size_t) m_local * N_sparse_t + b];
                    const cmplx c1v = c1_sparse[
                        (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const size_t r_idx = (size_t) c * M * N_sparse_t
                                       + (size_t) im * N_sparse_t + b;
                    if (gcmplx::abs(c0v) > floor_th) {
                        cmplx r_val = c1v / c0v;
                        // Amp/phase clip: cap |r| at max_r (preserve dir).
                        if (max_r > 0.0) {
                            const double abs_r = gcmplx::abs(r_val);
                            if (abs_r > max_r) r_val = r_val * (max_r / abs_r);
                        }
                        r_sparse[r_idx] = r_val;
                    } else {
                        r_sparse[r_idx] = cmplx(0.0, 0.0);
                    }
                }
            }
        }

        // ---- Carrier de-rotate r_sparse (in place) ----
        // phase_pred(t_n) = 2*pi*Df0*t_n + pi*Dfdot*t_n^2, t_n at sparse n.
        for (int b = 0; b < N_sparse_t; ++b) {
            const int n_sparse_local = n_sparse_local_arr[b];
            const double t_n = (double)(ind_min_t + n_sparse_local) * layer_dt;
            const double phase_pred = TWO_PI * Df0 * t_n
                                    + M_PI * Dfdot * t_n * t_n;
            const cmplx rot = gcmplx::exp(I_c * (-phase_pred));
            for (int c = 0; c < nchannels; ++c) {
                for (int im = 0; im < M; ++im) {
                    const size_t idx = (size_t) c * M * N_sparse_t
                                     + (size_t) im * N_sparse_t + b;
                    r_sparse[idx] = r_sparse[idx] * rot;
                }
            }
        }

        // ---- For each dense n: linear-interp r_demod, re-rotate, multiply
        //      c0_dense_complex, take real, scatter into template_fill.
        //      Assumes n_sparse_local[b] = n_sparse_local[0] + b*stride.
        const int n_sparse_local_0 = n_sparse_local_arr[0];
        for (int n_dense = 0; n_dense < Nt_active; ++n_dense) {
            const int    n_off = n_dense - n_sparse_local_0;
            int    b_lo  = n_off / stride;
            double frac = (double)(n_off - b_lo * stride) / (double) stride;
            // Clamp to interpolation domain; extrapolate-as-flat at edges.
            if (b_lo < 0) { b_lo = 0; frac = 0.0; }
            if (b_lo >= N_sparse_t - 1) { b_lo = N_sparse_t - 1; frac = 0.0; }
            const int b_hi = (b_lo + 1 < N_sparse_t) ? (b_lo + 1) : b_lo;

            const double t_n_dense = (double)(ind_min_t + n_dense) * layer_dt;
            const double phase_dense = TWO_PI * Df0 * t_n_dense
                                     + M_PI * Dfdot * t_n_dense * t_n_dense;
            const cmplx rot_back = gcmplx::exp(I_c * phase_dense);

            const int n_global = ind_min_t + n_dense;

            for (int c = 0; c < nchannels; ++c) {
                for (int im = 0; im < M; ++im) {
                    const cmplx r_lo = r_sparse[(size_t) c * M * N_sparse_t
                                             + (size_t) im * N_sparse_t + b_lo];
                    const cmplx r_hi = r_sparse[(size_t) c * M * N_sparse_t
                                             + (size_t) im * N_sparse_t + b_hi];
                    const cmplx r_demod_dense = r_lo * (1.0 - frac) + r_hi * frac;
                    const cmplx r_dense = r_demod_dense * rot_back;

                    const int m_local = m_active[im] - ind_min_f;
                    const cmplx c0v = c0_dense_complex_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * Nt_active
                        + (size_t) m_local * Nt_active + n_dense];
                    const cmplx c1_dense = r_dense * c0v;

                    const int m_global = m_active[im];
                    const size_t out_idx =
                        ((size_t) data_idx * nchannels + c) * Nf * Nt
                        + (size_t) m_global * Nt + n_global;
                    template_fill[out_idx] += factor * c1_dense.real();
                }
            }
        }
    }
}


// In-kernel variant: takes a GBTDIonTheFly* and generates X_het via the
// existing gb_run_fd_wave_tdi_wrap (Tukey-aware), applies the same
// fftshift + (1/dt) conversion as Stage 2b's get_ll path, then calls the
// sparse fill_global. Mirrors gb_signal_het_get_ll_in_kernel_wrap.
void GBComputationGroup::gb_signal_het_fill_global_in_kernel_wrap(
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
    int     N_sparse_fd, double tukey_alpha, double max_r)
{
#ifdef __CUDACC__
    std::fprintf(stderr, "[gb_signal_het_fill_global_in_kernel_wrap] GPU branch "
                         "TODO -- use CPU backend (force_backend=\"cpu\")\n");
    return;
#endif

    std::vector<cmplx>  X_het_raw((size_t) num_bin * nchannels * N_sparse_fd);
    std::vector<int>    k_f0_buf(num_bin);
    std::vector<double> f0_grid_buf(num_bin);

    gb_run_fd_wave_tdi_wrap(
        tdi_on_fly,
        X_het_raw.data(), k_f0_buf.data(), f0_grid_buf.data(),
        params_cand_all, t_start, T_obs,
        N_sparse_fd, num_bin, nparams, nchannels,
        tukey_alpha);

    // Same convention conversion as Stage 2b get_ll: fftshift + (1/dt).
    std::vector<cmplx> X_het((size_t) num_bin * nchannels * N_sparse_fd);
    const int    half_NS = N_sparse_fd / 2;
    const double dt_inv  = 1.0 / dt;
    for (int b = 0; b < num_bin; ++b) {
        for (int c = 0; c < nchannels; ++c) {
            const size_t base = ((size_t) b * nchannels + c) * N_sparse_fd;
            for (int i = 0; i < N_sparse_fd; ++i) {
                const int m_signed = i - half_NS;
                const int m_fft    = (m_signed >= 0)
                                         ? m_signed
                                         : (m_signed + N_sparse_fd);
                X_het[base + i] = X_het_raw[base + m_fft] * dt_inv;
            }
        }
    }

    this->gb_signal_het_fill_global_sparse_wrap(
        template_fill,
        X_het.data(), k_f0_buf.data(),
        c0_sparse_all,
        c0_dense_complex_all,
        wdm_window, n_sparse_local_arr,
        params_cand_all, params_ref_all, factors_all, data_index_all,
        num_bin, num_data,
        nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        nchannels,
        N_sparse_fd, max_r);
}


// ============================================================================
// Signal-heterodyne (v2 polyphase) -- get_ll_grad path.
// ============================================================================
//
// Central-difference gradient of logL = d_h - 0.5 * h_h over each candidate
// param. Mirrors the convention of gb_fd_get_ll_grad_wrap: param_eps is a
// per-parameter finite-difference step (eps_k <= 0 -> freeze that dim).
// Per binary, performs (1 central + 2 * nparams perturbed) calls into the
// Stage 2b get_ll_in_kernel pipeline. Each call regenerates X_het via
// gb_run_fd_wave_tdi_wrap so the FD reflects the perturbed params; the
// shared bin-fold A0/A1/B0/B1 are reused across all perturbations.
// ============================================================================

void GBComputationGroup::gb_signal_het_get_ll_grad_in_kernel_wrap(
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
    int     N_sparse_fd, double tukey_alpha, double max_r)
{
#ifdef __CUDACC__
    std::fprintf(stderr, "[gb_signal_het_get_ll_grad_in_kernel_wrap] GPU branch "
                         "TODO -- use CPU backend (force_backend=\"cpu\")\n");
    return;
#endif

    std::vector<double> params_priv((size_t) nparams);
    double d_h_C = 0.0, h_h_C = 0.0;
    double d_h_P = 0.0, h_h_P = 0.0;
    double d_h_M = 0.0, h_h_M = 0.0;
    int data_idx_local = 0;

    for (int bin = 0; bin < num_bin; ++bin) {
        data_idx_local = data_index_all[bin];

        // ---- Central evaluation ----
        for (int i = 0; i < nparams; ++i)
            params_priv[i] = params_cand_all[(size_t) bin * nparams + i];

        this->gb_signal_het_get_ll_in_kernel_wrap(
            tdi_on_fly,
            &d_h_C, &h_h_C,
            c0_sparse_all,
            A0_all, A1_all, B0_all, B1_all,
            wdm_window, n_sparse_local_arr,
            params_priv.data(), params_ref_all, &data_idx_local,
            1, num_data,
            nparams, f0_idx, fdot_idx,
            Nf, Nt, Nf_active, Nt_active,
            Nt_layer, N_sparse_t, stride,
            ind_min_t, ind_min_f,
            m_active_half_width,
            layer_df, dt,
            T_obs, t_start,
            nchannels, tdi_type,
            N_sparse_fd, tukey_alpha, max_r);

        d_h_central[bin] = d_h_C;
        h_h_central[bin] = h_h_C;
        const double ll_C = d_h_C - 0.5 * h_h_C;
        (void) ll_C;

        for (int k = 0; k < nparams; ++k) {
            const double eps = param_eps[k];
            if (eps <= 0.0) {
                grad_out[(size_t) bin * nparams + k] = 0.0;
                continue;
            }
            const double saved = params_priv[k];

            // +eps
            params_priv[k] = saved + eps;
            this->gb_signal_het_get_ll_in_kernel_wrap(
                tdi_on_fly,
                &d_h_P, &h_h_P,
                c0_sparse_all,
                A0_all, A1_all, B0_all, B1_all,
                wdm_window, n_sparse_local_arr,
                params_priv.data(), params_ref_all, &data_idx_local,
                1, num_data,
                nparams, f0_idx, fdot_idx,
                Nf, Nt, Nf_active, Nt_active,
                Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f,
                m_active_half_width,
                layer_df, dt,
                T_obs, t_start,
                nchannels, tdi_type,
                N_sparse_fd, tukey_alpha, max_r);

            // -eps
            params_priv[k] = saved - eps;
            this->gb_signal_het_get_ll_in_kernel_wrap(
                tdi_on_fly,
                &d_h_M, &h_h_M,
                c0_sparse_all,
                A0_all, A1_all, B0_all, B1_all,
                wdm_window, n_sparse_local_arr,
                params_priv.data(), params_ref_all, &data_idx_local,
                1, num_data,
                nparams, f0_idx, fdot_idx,
                Nf, Nt, Nf_active, Nt_active,
                Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f,
                m_active_half_width,
                layer_df, dt,
                T_obs, t_start,
                nchannels, tdi_type,
                N_sparse_fd, tukey_alpha, max_r);

            params_priv[k] = saved;

            const double ll_P = d_h_P - 0.5 * h_h_P;
            const double ll_M = d_h_M - 0.5 * h_h_M;
            grad_out[(size_t) bin * nparams + k] = (ll_P - ll_M) / (2.0 * eps);
        }
    }
}
