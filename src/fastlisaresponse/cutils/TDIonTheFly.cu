#include "TDIonTheFly.hh"
#include "Detector.hpp"
#include "LISAResponse.hh"
#include "Interpolate.hh"
#include <string>
#include <unistd.h>

// TODO: GET RID OF THIS ??!!!
double C_SI = 299792458.;

// CUDA_CALLABLE_MEMBER
// LISATDIonTheFly::LISATDIonTheFly(Orbits *orbits_, TDIConfig *tdi_config_)
// {
//     orbits = orbits_;
//     tdi_config = tdi_config_;
// }

CUDA_CALLABLE_MEMBER
LISATDIonTheFly::~LISATDIonTheFly()
{
    return;
}

// CUDA_CALLABLE_MEMBER
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

// CUDA_CALLABLE_MEMBER
// static void hplus_and_hcross(double t, double phase, double amp,  double Aplus, double Across, double cos2psi, double sin2psi,  double *hp, double *hc, double *hpf, double *hcf)
// {
//     double cp = cos(phase);
//     double sp = sin(phase);

//     *hp  = amp * ( Aplus*cos2psi*cp  + Across*sin2psi*sp );
//     *hc  = amp * ( Across*cos2psi*sp - Aplus*sin2psi*cp  );
    
//     *hpf = amp * (-Aplus*cos2psi*sp  + Across*sin2psi*cp );
//     *hcf = amp * ( Across*cos2psi*cp + Aplus*sin2psi*sp  );              
// }


// CUDA_CALLABLE_MEMBER
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
// CUDA_CALLABLE_MEMBER
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


// CUDA_CALLABLE_MEMBER
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

#define NLINKS 6

int LISATDIonTheFly::get_beta_index()
{
    throw std::invalid_argument("Not implemented.");
}

int LISATDIonTheFly::get_lam_index()
{
    throw std::invalid_argument("Not implemented.");
}

int LISATDIonTheFly::get_psi_index()
{
    throw std::invalid_argument("Not implemented.");
}

int LISATDIonTheFly::get_inc_index()
{
    throw std::invalid_argument("Not implemented.");
}

void LISATDIonTheFly::get_sky_vectors(Vec *k, Vec *u, Vec *v, double *params)
{

    double beta = params[get_beta_index()];
    double lam = params[get_lam_index()];
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

CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::xi_projections(double *xi_p, double *xi_c, Vec u, Vec v, Vec n)
{
    double u_dot_n = u.dot(n);
    double v_dot_n = v.dot(n);

    *xi_p = 0.5 * ((u_dot_n * u_dot_n) - (v_dot_n * v_dot_n));
    *xi_c = u_dot_n * v_dot_n;
}

CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::get_tdi_Xf(cmplx *tdi_channels_arr, double *params, double *t_data, int N, int bin_i)
{

    CUDA_SHARED int link_space_craft_0[NLINKS];
    CUDA_SHARED int link_space_craft_1[NLINKS];
    CUDA_SHARED int links[NLINKS];

    double k_dot_n, k_dot_x0, k_dot_x1;
    double t, L;
    double hp_del0, hp_del1, hc_del0, hc_del1;
    cmplx I(0.0, 1.0);
    double pre_factor, large_factor_real, large_factor_imag;
    double clipped_delay0, clipped_delay1, out, fraction0, fraction1;
    int integer_delay0, integer_delay1, max_integer_delay, min_integer_delay;

    Vec x0;
    Vec x1;
    Vec n;
    double delay0;
    double delay1;
    double xi_p;
    double xi_c;
    
    int start, increment;

    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    start = threadIdx.x;
    increment = blockDim.x;
#else
    start = 0;
    increment = 1;
#endif

    CUDA_SHARED Vec k(0.0, 0.0, 0.0);
    CUDA_SHARED Vec u(0.0, 0.0, 0.0);
    CUDA_SHARED Vec v(0.0, 0.0, 0.0);
    get_sky_vectors(&k, &u, &v, params);
    CUDA_SYNC_THREADS;

    for (int i = start; i < NLINKS; i += increment)
    {
        link_space_craft_0[i] = orbits->sc_r[i];
        link_space_craft_1[i] = orbits->sc_e[i];
        links[i] = orbits->links[i];
        // if (threadIdx.x == 1)
        // printf("%d %d %d %d\n", orbits->sc_r[i], orbits->sc_e[i], link_space_craft_1[i], link_space_craft_0[i]);
    }
    CUDA_SYNC_THREADS;
    double t_delay, total_delay;
    double time_eval;
    int delay_link;
    int sc0, sc1;
    Vec out_vec(0.0, 0.0, 0.0);
    double norm = 0.0;
    double n_temp;
    double temp_output;
    double phase_change;

    for (int i = start; i < N; i += increment)
    {
        for (int channel = 0; channel < tdi_config->num_channels; channel += 1)
        {
            tdi_channels_arr[channel * N + i] = 0.0;
        }
        CUDA_SYNC_THREADS;
        t = t_data[i];
        temp_output = 0.0;
        for (int unit_i = 0; unit_i < tdi_config->num_units; unit_i += increment)
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
                int combination_link_index;
                if (combination_link == -11)
                {
                    combination_link_index = -1;
                }
                else
                {
                    combination_link_index = orbits->get_link_ind(combination_link);
                }

                if (combination_link != -11)
                {
                    total_delay -= orbits->get_light_travel_time(t, combination_link);
                }
            }
            
            time_eval = t - total_delay;

            sc0 = link_space_craft_0[base_link_index];
            sc1 = link_space_craft_1[base_link_index];
            
            x0 = orbits->get_pos(time_eval, sc0);
            x1 = orbits->get_pos(time_eval, sc1);
            n = x0 - x1; // # TODO: check if this right
            norm = sqrt(n.dot(n));
            n = n / norm;

            k_dot_n = k.dot(n);
            k_dot_x0 = k.dot(x0); // receiver
            k_dot_x1 = k.dot(x1); // emitter

            L = orbits->get_light_travel_time(t, base_link);
            
            pre_factor = 1. / (1. - k_dot_n);
            
            delay0 = t - k_dot_x0 * C_inv;
            delay1 = t - L - k_dot_x1 * C_inv;

            xi_projections(&xi_p, &xi_c, u, v, n);

            phase_change = 0.0; // the real part
            get_hp_hc(&hp_del0, &hc_del0, delay0, params, phase_change, bin_i);
            get_hp_hc(&hp_del1, &hc_del1, delay1, params, phase_change, bin_i);
            
            large_factor_real = (hp_del0 - hp_del1) * xi_p + (hc_del0 - hc_del1) * xi_c;
            
            phase_change = M_PI / 2.0; // the real part
            get_hp_hc(&hp_del0, &hc_del0, delay0, params, phase_change, bin_i);
            get_hp_hc(&hp_del1, &hc_del1, delay1, params, phase_change, bin_i);
            
            large_factor_imag = (hp_del0 - hp_del1) * xi_p + (hc_del0 - hc_del1) * xi_c;
            tdi_channels_arr[channel * N + i] += pre_factor * (large_factor_real + I * large_factor_imag);
            if ((t > 1e6) && (t < 1e6 + 1e4))
                printf("FLY: %d %d %e %e %e %e %e %e %e\n", i, unit_i, t, pre_factor, large_factor_real, delay0, delay1, L, xi_p);
            CUDA_SYNC_THREADS;
        }
    }
}

CUDA_CALLABLE_MEMBER
double LISATDIonTheFly::get_amp(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
    throw std::invalid_argument("Not implemented.");


}

CUDA_CALLABLE_MEMBER
double LISATDIonTheFly::get_phase(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
    throw std::invalid_argument("Not implemented.");
}

CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::get_hp_hc(double *hp, double *hc, double t, double *params, double phase_change, int bin_i)
{
    double amp = get_amp(t, params, bin_i);
    double phase = get_phase(t, params, bin_i);
    double psi = params[get_psi_index()];
    double inc = params[get_inc_index()];
    
    double inc_p = (1. + cos(inc) * cos(inc)) / 2.;
    double inc_c = cos(inc);
    
    // *hp = amp * (inc_p * cos(2. * psi) * cos(phase + phase_change) - inc_c * sin(2. * psi) * sin(phase + phase_change));
    // *hc = amp * (-inc_p * sin(2. * psi) * cos(phase + phase_change) - inc_c * cos(2. * psi) * sin(phase + phase_change));  
    double cos2psi = cos(2.0 * psi);
    double sin2psi = sin(2.0 * psi);
    double cosiota = cos(inc);

    double hSp = -cos(phase) * amp * (1.0 + cosiota * cosiota);
    double hSc = -sin(phase) * 2.0 * amp * cosiota;

    *hp = hSp * cos2psi - hSc * sin2psi;
    *hc = hSp * sin2psi + hSc * cos2psi;

}


CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels)
{   

    if (buffer_length < 2 * N * sizeof(double) + 1 * N * sizeof(int) + 1 * N * sizeof(bool))
    {
        throw std::invalid_argument("Buffer length not long enough.");
    }

    get_tdi_Xf(tdi_channels_arr, params, t_arr, N, bin_i);
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
     _extract_amplitude_and_phase(count, fix_count, flip, pjump, N, &tdi_amp[0], &tdi_phase[0], &tdi_channels_arr[0], &phi_ref[0]);
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


// CUDA_CALLABLE_MEMBER
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

// CUDA_CALLABLE_MEMBER
// void LISATDIonTheFly::get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i)
// {
//     printf("Not Implemented. TODO: best way to do this?");
// }

// CUDA_CALLABLE_MEMBER
// double LISATDIonTheFly::get_phase_ref(double t, double *params, int bin_i)
// {
//     printf("Not Implemented. TODO: best way to do this?");
// }


CUDA_CALLABLE_MEMBER
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


CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::new_unwrap_phase(double *ph_correct_buffer, int N, double *phase)
{
    double dd, ddmod;
    double period = 2. * M_PI;
    double interval_high =  period / 2.;
    double interval_low = -interval_high;
    double ph_tmp;
    double discont = period / 2.;

    for (int i = 0; i < N; i += 1)
    {
        ph_correct_buffer[i] = 0.0;
    }

    CUDA_SYNC_THREADS;
    double tmp_remainder;
    // std::cout << "start phase[0]: " << phase[0] << std::endl;
    for(int i=1; i<N ;i++)
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

    // cumsum
    for (int i = 1; i < N; i += 1)
    {
        ph_correct_buffer[i] += ph_correct_buffer[i - 1];
    }
    CUDA_SYNC_THREADS;

    double tmp;
    for (int i = 1; i < N; i += 1)
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

CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::new_extract_amplitude_and_phase(int *count, bool *fix_count, double *flip, double *pjump, int Ns, double *As, double *Dphi, cmplx *M, double *phiR)
{
    bool is_min;
    double dA1, dA2, dA3, test1, test2;
    for (int i = 0; i < Ns; i += 1)
    {
        count[i] = 0;
        pjump[i] = 0.0;
        flip[i] = 1.0;
        fix_count[i] = false;
        As[i] = gcmplx::abs(M[i]);
    }
    CUDA_SYNC_THREADS;
    for (int i = 1; i < Ns - 1; i += 1)
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
    for (int i = 1; i < Ns; i += 1)
    {
        count[i] += count[i - 1];
    }
    CUDA_SYNC_THREADS;


    // 
    for (int i = 0; i < Ns - 1; i += 1)
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
    for(int i=0; i<Ns ;i++)
    {
        As[i] = flip[i]*As[i];
        // printf("HUH: %e %e\n", flip[i], As[i]);
        v = remainder(phiR[i], 2 * M_PI);
        Dphi[i] = -atan2(M[i].imag(),M[i].real())+pjump[i]-v;
        // if ((i > 11670))
        // printf("INIT new: %d %e %e %e %e %e %e\n", i, -atan2(Mf[i],M[i]), flip[i], pjump[i], As[i], Dphi[i], v);
    
    }
}


CUDA_CALLABLE_MEMBER
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



CUDA_CALLABLE_MEMBER
double LISATDIonTheFly::get_phase_ref(double t, double *params, int bin_i)
{   
    // TD is based on t_sc rather than t (t_ssb)
    Vec k(0.0, 0.0, 0.0);
    Vec u(0.0, 0.0, 0.0);
    Vec v(0.0, 0.0, 0.0);
    
    get_sky_vectors(&k, &u, &v, params);
    // reference phase is at spacecraft 1
    Vec x0 = orbits->get_pos(t, 1);
    double k_dot_x0 = k.dot(x0);
    double t_sc = t - k_dot_x0 * C_inv;
    double phase_ref = get_phase(t_sc, params, bin_i);
    return phase_ref;
}


CUDA_CALLABLE_MEMBER
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

CUDA_CALLABLE_MEMBER
int LISATDIonTheFly::get_tdi_buffer_size(int N)
{
    return 2 * N * sizeof(double) + 1 * N * sizeof(bool) + 1 * N * sizeof(int);
}


CUDA_CALLABLE_MEMBER
int GBTDIonTheFly::get_amplitude_index()
{
    return 0;
}

CUDA_CALLABLE_MEMBER
int GBTDIonTheFly::get_f0_index()
{
    return 1;
}

CUDA_CALLABLE_MEMBER
int GBTDIonTheFly::get_fdot0_index()
{
    return 2;
}

CUDA_CALLABLE_MEMBER
int GBTDIonTheFly::get_fddot0_index()
{
    return 3;
}

CUDA_CALLABLE_MEMBER
int GBTDIonTheFly::get_phi0_index()
{
    return 4;
}

int GBTDIonTheFly::get_inc_index()
{
    // ndim = 2; lam first
    return 5;
}

int GBTDIonTheFly::get_psi_index()
{
    // ndim = 2; beta second
    return 6;
}


int GBTDIonTheFly::get_lam_index()
{
    // ndim = 2; lam first
    return 7;
}

int GBTDIonTheFly::get_beta_index()
{
    // ndim = 2; beta second
    return 8;
}

CUDA_CALLABLE_MEMBER
double GBTDIonTheFly::ucb_phase(double t, double *params)
{
    double f0    = params[get_f0_index()];
    double phi0  = params[get_phi0_index()];
    double fdot  = params[get_fdot0_index()];
    double fddot = params[get_fddot0_index()];
    
    /*
     * LDC phase parameter in key files is
     * -phi0
     */
    return -phi0 + 2 * M_PI *( f0*t + 0.5*fdot*t*t + 1.0/6.0*fddot*t*t*t );
}


CUDA_CALLABLE_MEMBER
double GBTDIonTheFly::ucb_amplitude(double t, double *params)
{
    double A0    = params[get_amplitude_index()];
    double f0    = params[get_f0_index()];
    double fdot  = params[get_fdot0_index()];

    return A0 * ( 1.0 + 2.0/3.0*fdot/f0*t );
}

CUDA_CALLABLE_MEMBER
GBTDIonTheFly::GBTDIonTheFly(double T_) : LISATDIonTheFly()
{
    T = T_;
}

// CUDA_CALLABLE_MEMBER
// void GBTDIonTheFly::get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i)
// {
//     // params are already referenced before this. 
//     for(int n=0; n<N; n++)
//     {
//         phase[n] = ucb_phase(t[n],params);
//         amp[n]   = ucb_amplitude(t[n],params);
//     }
// }


CUDA_CALLABLE_MEMBER
double GBTDIonTheFly::get_phase(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
    return ucb_phase(t, params);
}

CUDA_CALLABLE_MEMBER
double GBTDIonTheFly::get_amp(double t, double *params, int bin_i)
{
    // TD is based on sc1 time
    return ucb_amplitude(t, params);
}

CUDA_CALLABLE_MEMBER
GBTDIonTheFly::~GBTDIonTheFly()
{
    return;
}

CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::print_orbits_tdi()
{
    printf("inside print\n");
    printf("ahead of check\n");
    if (orbits == NULL)
    {
        throw std::invalid_argument("Need to add orbital information.\n");
    }
    printf("orbits inside: %e\n", orbits->armlength);
    
    if (tdi_config == NULL)
    {
        throw std::invalid_argument("Need to add tdi config.\n");
    }
    printf("tdi_config inside: %d\n", tdi_config->num_channels);
}

CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::run_wave_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    printf("INSIDE\n");
    printf("OUTSIDE\n");
    printf("ahead of check22\n");
    // printf("orbits inside: %e", orbits->armlength);
    if (orbits == NULL)
    {
        throw std::invalid_argument("Need to add orbital information.\n");
    }
    printf("CHECK CHECK: %d\n", orbits->N);
    if (tdi_config == NULL)
    {
        throw std::invalid_argument("Need to add tdi config2.\n");
    }
    printf("tdi_config inside: %d", tdi_config->num_channels);


     // TODO: CHECK THIS!!
    for (int bin_i = 0; bin_i < num_bin; bin_i += 1)
    {
        
        // map to Tyson/Neil setup
        double *params_here = &params[bin_i * n_params];
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

CUDA_CALLABLE_MEMBER
int GBTDIonTheFly::get_gb_buffer_size(int N)
{
    return N * sizeof(double) + get_tdi_buffer_size(N);
}

CUDA_CALLABLE_MEMBER
void GBTDIonTheFly::dealloc()
{
    return;
}



// CUDA_CALLABLE_MEMBER
// TDSplineTDIWaveform::TDSplineTDIWaveform(Orbits *orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *phase_spline_): LISATDIonTheFly(orbits_, tdi_config_)
// {
//     phase_spline = phase_spline_;
//     amp_spline = amp_spline_;
//     // printf("spline type init : %d %d\n", amp_spline->spline_type, phase_spline->spline_type);        
//     // check_x();
// }

CUDA_CALLABLE_MEMBER
void TDSplineTDIWaveform::add_amp_spline(double *x0_, double *y0_, double *c1_, double *c2_, double *c3_, double ninterps_, int length_, int spline_type_)
{
    add_cubic_spline(amp_spline, x0_, y0_, c1_, c2_, c3_, ninterps_, length_, spline_type_);
}

CUDA_CALLABLE_MEMBER
void TDSplineTDIWaveform::add_phase_spline(double *x0_, double *y0_, double *c1_, double *c2_, double *c3_, double ninterps_, int length_, int spline_type_)
{
    add_cubic_spline(phase_spline, x0_, y0_, c1_, c2_, c3_, ninterps_, length_, spline_type_);
}

CUDA_CALLABLE_MEMBER
void TDSplineTDIWaveform::check_x()
{
    for (int j = 0; j < amp_spline->ninterps; j += 1)
    {
        for (int i = 0; i < amp_spline->length; i += 1)
        {
            printf("%d %d %e %e\n", j, i, amp_spline->x0[j * amp_spline->length + i], phase_spline->x0[j * amp_spline->length + i]);
        }
    }
}

int TDSplineTDIWaveform::get_psi_index()
{
    return 1;
}

int TDSplineTDIWaveform::get_inc_index()
{
    return 0;
}


int TDSplineTDIWaveform::get_beta_index()
{
    // ndim = 2; beta second
    return 3;
}

int TDSplineTDIWaveform::get_lam_index()
{
    // ndim = 2; lam first
    return 2;
}

int FDSplineTDIWaveform::get_beta_index()
{
    // ndim = 2; beta second
    return 1;
}

void FDSplineTDIWaveform::get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels)
{
    LISATDIonTheFly::get_tdi(
        buffer, buffer_length,
        tdi_channels_arr, 
        tdi_amp, tdi_phase,
        phi_ref,
        params, t_arr, N, bin_i, nchannels
    );
    
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


int FDSplineTDIWaveform::get_lam_index()
{
    // ndim = 2; lam first
    return 0;
}


int FDSplineTDIWaveform::get_psi_index()
{
    return 1;
}

int FDSplineTDIWaveform::get_inc_index()
{
    return 0;
}



CUDA_CALLABLE_MEMBER
double TDSplineTDIWaveform::get_amp(double t, double *params, int spline_i)
{
    printf("before amp: %d\n", amp_spline->ninterps);
    return amp_spline->eval_single(t, spline_i);
}

CUDA_CALLABLE_MEMBER
double TDSplineTDIWaveform::get_phase(double t, double *params, int spline_i)
{
    printf("before phase: %d\n", phase_spline->ninterps);
    
    return phase_spline->eval_single(t, spline_i);
}

// CUDA_CALLABLE_MEMBER
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


// CUDA_CALLABLE_MEMBER
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




// CUDA_CALLABLE_MEMBER
// FDSplineTDIWaveform::FDSplineTDIWaveform(Orbits *orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *freq_spline_, double *phase_ref_): LISATDIonTheFly(orbits_, tdi_config_)
// {
//     freq_spline = freq_spline_;
//     amp_spline = amp_spline_;
//     phase_ref_store = phase_ref_;
// }

// CUDA_CALLABLE_MEMBER
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

CUDA_CALLABLE_MEMBER
void FDSplineTDIWaveform::add_amp_spline(double *x0_, double *y0_, double *c1_, double *c2_, double *c3_, double ninterps_, int length_, int spline_type_)
{
    add_cubic_spline(amp_spline, x0_, y0_, c1_, c2_, c3_, ninterps_, length_, spline_type_);
}

CUDA_CALLABLE_MEMBER
void FDSplineTDIWaveform::add_freq_spline(double *x0_, double *y0_, double *c1_, double *c2_, double *c3_, double ninterps_, int length_, int spline_type_)
{
    add_cubic_spline(freq_spline, x0_, y0_, c1_, c2_, c3_, ninterps_, length_, spline_type_);
}

CUDA_CALLABLE_MEMBER
double FDSplineTDIWaveform::get_amp(double t, double *params, int spline_i)
{
    return 1.0;
}

CUDA_CALLABLE_MEMBER
double FDSplineTDIWaveform::get_phase(double t, double *params, int spline_i)
{
    double f = freq_spline->eval_single(t, spline_i);
    return 2. * M_PI * f * t;
}

CUDA_CALLABLE_MEMBER
double FDSplineTDIWaveform::get_amp_f(double t, double *params, int spline_i)
{
    // TODO: may want to do this in a fast way
    return amp_spline->eval_single(t, spline_i);
}


// CUDA_CALLABLE_MEMBER
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

CUDA_CALLABLE_MEMBER
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


// CUDA_CALLABLE_MEMBER
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

//     // if ((i == 0) && (link_i == 0)) printf("%.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", L, delay0, delay1, x0[0], x0[1], x0[2],x1[0], x1[1], x1[2]);
//     double clipped_delay0 = t; 
//     int integer_delay0 = (int)ceil(clipped_delay0 * sampling_frequency) - 1;
//     double fraction0 = 1.0 + integer_delay0 - clipped_delay0 * sampling_frequency;
    
//     // int h, int d, double e, double *A_arr, double deps, double *E_arr, int start_input_ind
//     // half_point_count, integer_delay, fraction, A_arr, deps, E_arr, start_input_ind
//     double e = fraction0;
//     int d = integer_delay0;
    
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

// CUDA_CALLABLE_MEMBER
// void TDLagrangeInterpTDIWave::get_hp_hc(double *hp, double *hc, double t, double *params, double phase_change, int bin_i)
// {
//     cmplx I(0.0, 1.0);
//     cmplx wave_out = lagrange->interp(t, wave, wave_N, bin_i);

//     wave_out *= gcmplx::exp(I * phase_change);
//     *hp = wave_out.real();
//     *hc = wave_out.imag();
// }

// CUDA_CALLABLE_MEMBER
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
//     return 0;
// }
