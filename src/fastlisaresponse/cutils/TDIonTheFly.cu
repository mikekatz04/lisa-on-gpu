#include "TDIonTheFly.hh"
#include "Detector.hpp"
#include "LISAResponse.hh"
#include "Interpolate.hh"
#include <string>

// TODO: GET RID OF THIS ??!!!
double C_SI = 299792458.;

CUDA_CALLABLE_MEMBER
LISATDIonTheFly::LISATDIonTheFly(Orbits *orbits_)
{
    orbits = orbits_;
}

CUDA_CALLABLE_MEMBER
LISATDIonTheFly::~LISATDIonTheFly()
{
    return;
}

CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::get_t_tdi(double *t_out, double *kr, double *Larm, double t, int a, int b, int c, int n)
{
  
    t_out[0] = t - kr[a] - 2.0 * Larm[c] - 2.0 * Larm[b];
    t_out[1] = t- kr[b] - Larm[c] - 2.0 * Larm[b];
    t_out[2] = t - kr[c]-Larm[b]-2.0*Larm[c];
    t_out[3] = t - kr[a]-2.0*Larm[b];
    t_out[4] = t - kr[a]-2.0*Larm[c];
    t_out[5] = t - kr[c]- Larm[b];
    t_out[6] = t - kr[b]- Larm[c];
    t_out[7] = t - kr[a];

    // for (int i = 0; i < 8; i += 1) printf("CHECKCHECK: %d %e, %e\n", i, t, t_out[i]);
      
}

CUDA_CALLABLE_MEMBER
static void hplus_and_hcross(double t, double phase, double amp,  double Aplus, double Across, double cos2psi, double sin2psi,  double *hp, double *hc, double *hpf, double *hcf)
{
    double cp = cos(phase);
    double sp = sin(phase);

    *hp  = amp * ( Aplus*cos2psi*cp  + Across*sin2psi*sp );
    *hc  = amp * ( Across*cos2psi*sp - Aplus*sin2psi*cp  );
    
    *hpf = amp * (-Aplus*cos2psi*sp  + Across*sin2psi*cp );
    *hcf = amp * ( Across*cos2psi*cp + Aplus*sin2psi*sp  );              
}


CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::get_tdi_sub(cmplx *M, int n, int N, int a, int b, int c, double* tarray, double *amp_tdi_vals, double *phase_tdi_vals, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm)
{
    double t, f, amp, phase;
    double hp, hc, hpf, hcf;

    M[n] = 0.0;
    
    // if(freq_spline) /* mbh */
    // {
    //     //For TDI we want the overall amplitude scaled out of the waveform as it passes through zero
    //     amp = 1.0; //spline_interpolation(amp_spline,  tarray[n]);
    //     f   = spline_interpolation(freq_spline, tarray[n]);
    // }

    // first index
    t = tarray[0];
    amp = amp_tdi_vals[0];
    phase = phase_tdi_vals[0];
    
    //  if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    // M[n] += hp*Apm[b]+hc*Acm[b];
    // M[n] -= hp*App[c]+hc*Acp[c];
    // Mf[n] += hpf*Apm[b]+hcf*Acm[b];
    // Mf[n] -= hpf*App[c]+hcf*Acp[c];
    cmplx I(0.0, 1.0);
    // printf("CHECKING: %d %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", n, t, phase, amp, Aplus, Across, Apm[b], Acm[b]);
    
    M[n] += (hp*Apm[b]+hc*Acm[b]) + I * (hpf*Apm[b]+hcf*Acm[b]);
    M[n] -= (hp*App[c]+hc*Acp[c]) + I * (hpf*App[c]+hcf*Acp[c]);
    
    t = tarray[1];
    amp = amp_tdi_vals[1];
    phase = phase_tdi_vals[1];
    
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    // M[n] -= hp*Apm[c]+hc*Acm[c];
    // M[n] += hp*App[c]+hc*Acp[c];
    // Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
    // Mf[n] += hpf*App[c]+hcf*Acp[c];
    M[n] -= (hp*Apm[c]+hc*Acm[c]) + I * (hpf*Apm[c]+hcf*Acm[c]);
    M[n] += (hp*App[c]+hc*Acp[c]) + I * (hpf*App[c]+hcf*Acp[c]);
    
    t = tarray[2];
    amp = amp_tdi_vals[2];
    phase = phase_tdi_vals[2];
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    // M[n] += hp*App[b]+hc*Acp[b];
    // M[n] -= hp*Apm[b]+hc*Acm[b];
    // Mf[n] += hpf*App[b]+hcf*Acp[b];
    // Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
    M[n] += (hp*App[b]+hc*Acp[b]) + I * (hpf*App[b]+hcf*Acp[b]);
    M[n] -= (hp*Apm[b]+hc*Acm[b]) + I * (hpf*Apm[b]+hcf*Acm[b]);

    t = tarray[3];
    amp = amp_tdi_vals[3];
    phase = phase_tdi_vals[3];
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    // M[n] -= hp*Apm[b]+hc*Acm[b];
    // M[n] += hp*Apm[c]+hc*Acm[c];
    // Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
    // Mf[n] += hpf*Apm[c]+hcf*Acm[c];
    M[n] -= (hp*Apm[b]+hc*Acm[b]) + I * (hpf*Apm[b]+hcf*Acm[b]);
    M[n] += (hp*Apm[c]+hc*Acm[c]) + I * (hpf*Apm[c]+hcf*Acm[c]);
    
    t = tarray[4];
    amp = amp_tdi_vals[4];
    phase = phase_tdi_vals[4];
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    // M[n] += hp*App[c]+hc*Acp[c];
    // M[n] -= hp*App[b]+hc*Acp[b];
    // Mf[n] += hpf*App[c]+hcf*Acp[c];
    // Mf[n] -= hpf*App[b]+hcf*Acp[b];
    M[n] += (hp*App[c]+hc*Acp[c]) + I * (hpf*App[c]+hcf*Acp[c]);
    M[n] -= (hp*App[b]+hc*Acp[b]) + I * (hpf*App[b]+hcf*Acp[b]);
    
    t = tarray[5];
    amp = amp_tdi_vals[5];
    phase = phase_tdi_vals[5];
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    // M[n] -= hp*App[b]+hc*Acp[b];
    // M[n] += hp*Apm[b]+hc*Acm[b];
    // Mf[n] -= hpf*App[b]+hcf*Acp[b];
    // Mf[n] += hpf*Apm[b]+hcf*Acm[b];
    M[n] -= (hp*App[b]+hc*Acp[b]) + I * (hpf*App[b]+hcf*Acp[b]);
    M[n] += (hp*Apm[b]+hc*Acm[b]) + I * (hpf*Apm[b]+hcf*Acm[b]);
    
    t = tarray[6];
    amp = amp_tdi_vals[6];
    phase = phase_tdi_vals[6];
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    // M[n] += hp*Apm[c]+hc*Acm[c];
    // M[n] -= hp*App[c]+hc*Acp[c];
    // Mf[n] += hpf*Apm[c]+hcf*Acm[c];
    // Mf[n] -= hpf*App[c]+hcf*Acp[c];
    M[n] += (hp*Apm[c]+hc*Acm[c]) + I * (hpf*Apm[c]+hcf*Acm[c]);
    M[n] -= (hp*App[c]+hc*Acp[c]) + I * (hpf*App[c]+hcf*Acp[c]);
    
    t = tarray[7];
    amp = amp_tdi_vals[7];
    phase = phase_tdi_vals[7];
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    // M[n] -= hp*Apm[c]+hc*Acm[c];
    // M[n] += hp*App[b]+hc*Acp[b];
    // Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
    // Mf[n] += hpf*App[b]+hcf*Acp[b];  
    M[n] -= (hp*Apm[c]+hc*Acm[c]) + I * (hpf*Apm[c]+hcf*Acm[c]);
    M[n] += (hp*App[b]+hc*Acp[b]) + I * (hpf*App[b]+hcf*Acp[b]);
}


// void LISA_polarization_tensor(double costh, double phi, double eplus[4][4], double ecross[4][4], double k[4])
CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::LISA_polarization_tensor(double costh, double phi, double *eplus, double *ecross, double *k)
{
    /*   Gravitational Wave basis vectors   */
    double u[3],v[3];

    /*   Sky location angles   */
    double sinth = sqrt(1.0 - costh*costh);
    double cosph = cos(phi);
    double sinph = sin(phi);

    /*   Tensor construction for building slowly evolving LISA response   */
    //Gravitational Wave source basis vectors
    u[0] =  costh*cosph;  u[1] =  costh*sinph;  u[2] = -sinth;
    v[0] =  sinph;        v[1] = -cosph;        v[2] =  0.;
    k[0] = -sinth*cosph;  k[1] = -sinth*sinph;  k[2] = -costh;
    
    //GW polarization basis tensors
    /*
     * LDC convention:
     * https://gitlab.in2p3.fr/LISA/LDC/-/blob/develop/ldc/waveform/fastGB/GB.cc
     */
    for(int i=0; i<3;i++)
    {
        for(int j=0; j<3;j++)
        {
            eplus[i * 3 + j]  = v[i]*v[j] - u[i]*u[j];
            ecross[i * 3 + j] = u[i]*v[j] + v[i]*u[j];
        }
    }

}


CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::get_tdi_n(cmplx *X, cmplx *Y, cmplx *Z, double* phi_ref, double *params, double t, int m, int N, double costh, double phi, double cosi, double psi, int bin_i)
{
    // printf("bef: inside: %d \n", m);
    /*   Indicies    */
    int i,j;

    Vec k; 
    double kdotr[3], kdotn[3];
    double dplus[3], dcross[3];
    
    /*   Polarization basis tensors   */
    double eplus[3 * 3], ecross[3 * 3];
    
    double cos2psi, sin2psi;
    double Aplus, Across;
    double App[3], Apm[3], Acp[3], Acm[3];
    double t_tdi[8], amp_tdi[8], phase_tdi[8];
    
    /*   Allocating Arrays   */
    
    // TODO: move outside this function
    cos2psi = cos(2.*psi);  
    sin2psi = sin(2.*psi);
    
    Aplus = 0.5*(1.+cosi*cosi);
    Across = -cosi;

    LISA_polarization_tensor(costh, phi, &eplus[0], &ecross[0], &k[0]);

    Vec x[3]; 
    Vec n[3]; // TODO: What about other direction down arms?
    double L[3];
    int sc, link;

    // position vectors for each spacecraft (converted to seconds)
    for(i=0; i<3; i++)
    {   
        sc = i + 1;
        x[i] = orbits->get_pos(t, sc) / C_SI;
        // x[i][0] = spline_interpolation(orbit->dx[i],t)/CLIGHT;
        // x[i][1] = spline_interpolation(orbit->dy[i],t)/CLIGHT;
        // x[i][2] = spline_interpolation(orbit->dz[i],t)/CLIGHT;
    }

    n[0] = x[1] - x[2];
    n[1] = x[2] - x[0];
    n[2] = x[0] - x[1];

    // arm vectors
    for(i=0; i<3; i++)
    {   
        link = orbits->get_link_from_arr(i);
        // printf("need TO CHECK THIS. Probably wrong link order?.\n");

        // n[i] = orbits->get_normal_unit_vec(t, link); 
        // L[i] = orbits->get_light_travel_time(t, link);

        L[i] = n[i].dot(n[i]);
        L[i] = sqrt(L[i]);

        n[i] = n[i] /  L[i];
        // n[0][i] = x[1][i] - x[2][i];
        // n[1][i] = x[2][i] - x[0][i];
        // n[2][i] = x[0][i] - x[1][i];
    }

    // //arm lengths
    // for(i=0; i<3; i++) 
    // {
    //     // L[i]=0.0;
    //     // for(j=0; j<3; j++)  L[i] += n[i][j]*n[i][j];
    //     // L[i] = sqrt(L[i]);
    //     sc = i + 1;
        
    // }

    //normalize arm vectors  
    // Orbit class vectors are normalized.      
    // for(i=0; i<3; i++)
    //     for(j=0;j<3;j++)
    //         n[i][j] /= L[i];
    


    // k dot r_i (source direction spacecraft locations)
    for(i=0; i<3; i++)
    {
        kdotr[i] = k.dot(x[i]);
    }
    
    // k dot n_i (source direction w/ arm vectors)
    for(i=0; i<3; i++)
    {
        kdotn[i] = k.dot(n[i]);
    }        
    
    //Antenna primitives
    for(i=0; i<3; i++)
    {
        dplus[i] = 0.0;
        dcross[i] = 0.0;
        for(j=0; j<3; j++)
        {
            for(int l=0; l<3; l++)
            {
                dplus[i]  += (n[i][j] * n[i][l]) * eplus[j * 3 + l];
                dcross[i] += (n[i][j] * n[i][l]) * ecross[j * 3 + l];
            }
        }
    }
    
    //Full Antenna patterns
    for(i=0; i<3; i++)
    {
        App[i] = 0.5 * dplus[i]  / (1.0 + kdotn[i]);
        Apm[i] = 0.5 * dplus[i]  / (1.0 - kdotn[i]);
        Acp[i] = 0.5 * dcross[i] / (1.0 + kdotn[i]);
        Acm[i] = 0.5 * dcross[i] / (1.0 - kdotn[i]);
    }
    // printf("af2: inside: %e %e %e %e %e\n", t, dplus[0], dcross[0], kdotn[0], n[0].x);
    
    double time_sc = t - kdotr[0];
    get_phase_ref(&time_sc, &phi_ref[m], &params[0], 1, bin_i);

    get_t_tdi(&t_tdi[0], &kdotr[0], &L[0], t, 0, 1, 2, m);
    get_amp_and_phase(&t_tdi[0], &amp_tdi[0], &phase_tdi[0], &params[0], 8, bin_i);
    // printf("af33: inside: %d \n", m);
    get_tdi_sub(X, m, N, 0, 1, 2, &t_tdi[0], &amp_tdi[0], &phase_tdi[0], Aplus, Across, cos2psi, sin2psi, &App[0], &Apm[0], &Acp[0], &Acm[0], &kdotr[0], &L[0]);

    printf("af44: inside: %d %.12e %.12e %.12e \n", m, X[m].real(), X[m].imag(), phi_ref[m]);
    
    get_t_tdi(&t_tdi[0], &kdotr[0], &L[0], t, 1, 2, 0, m);
    get_amp_and_phase(&t_tdi[0], &amp_tdi[0], &phase_tdi[0], &params[0], 8, bin_i);
    get_tdi_sub(Y, m, N, 1, 2, 0, &t_tdi[0], &amp_tdi[0], &phase_tdi[0], Aplus, Across, cos2psi, sin2psi, &App[0], &Apm[0], &Acp[0], &Acm[0], &kdotr[0], &L[0]);

    // printf("af4: inside: %d \n", m);
    
    get_t_tdi(&t_tdi[0], &kdotr[0], &L[0], t, 2, 0, 1, m);
    get_amp_and_phase(&t_tdi[0], &amp_tdi[0], &phase_tdi[0], &params[0], 8, bin_i);
    get_tdi_sub(Z, m, N, 2, 0, 1, &t_tdi[0], &amp_tdi[0], &phase_tdi[0], Aplus, Across, cos2psi, sin2psi, &App[0], &Apm[0], &Acp[0], &Acm[0], &kdotr[0], &L[0]);

    // printf("af: inside: %d \n", m);
    
}


CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::get_tdi(cmplx *X, cmplx *Y, cmplx *Z, double* phi_ref, double *params, double *t_arr, int N, double costh, double phi, double cosi, double psi, int bin_i)
{   
    get_tdi_Xf(X, Y, Z, phi_ref, params, t_arr, N, costh, phi, cosi, psi, bin_i);
    new_extract_phase(X, phi_ref, N, t_arr);
    new_extract_phase(Y, phi_ref, N, t_arr);
    new_extract_phase(Z, phi_ref, N, t_arr);

    // extract_amplitude_and_phase(flip, pjump, N, Xamp, Xphase, X, Xf, phi_ref);
    // unwrap_phase(N, Xphase);

    // extract_amplitude_and_phase(flip, pjump, N, Yamp, Yphase, Y, Yf, phi_ref);
    // unwrap_phase(N, Yphase);

    // extract_amplitude_and_phase(flip, pjump, N, Zamp, Zphase, Z, Zf, phi_ref);
    // unwrap_phase(N, Zphase);
}


CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::get_tdi_Xf(cmplx *X, cmplx *Y, cmplx *Z, double* phi_ref, double *params, double *t_arr, int N, double costh, double phi, double cosi, double psi, int bin_i)
{

#ifdef __CUDACC__
    int start = threadIdx.x;
    int incr = blockDim.x;
#else // __CUDACC__
    int start = 0;
    int incr = 1;
#endif // __CUDACC__
    for (int i = start; i < N; i += incr)
    {
        get_tdi_n(X, Y, Z, phi_ref, params, t_arr[i], i, N, costh, phi, cosi, psi, bin_i);
    }
    CUDA_SYNC_THREADS;
}

CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::get_amp_and_phase(double *t, double *amp, double *phase, double *params, int N, int bin_i)
{
    printf("Not Implemented. TODO: best way to do this?");
}

CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::get_phase_ref(double *t, double *phase, double *params, int N, int bin_i)
{
    printf("Not Implemented. TODO: best way to do this?");
}


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
        v = remainder(phiR[i], 2 * M_PI);
        Dphi[i] = -atan2(Mf[i],M[i])+pjump[i]-v;
    }
    
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
        // TODO: take conj to match N/T
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
    return 8 * N * sizeof(double);
}

CUDA_CALLABLE_MEMBER
double GBTDIonTheFly::ucb_phase(double t, double *params)
{
    double f0    = params[1];
    double phi0  = params[4];
    double fdot  = params[2];
    double fddot = params[3];
    
    /*
     * LDC phase parameter in key files is
     * -phi0
     */
    return -phi0 + 2 * M_PI *( f0*t + 0.5*fdot*t*t + 1.0/6.0*fddot*t*t*t );
}

CUDA_CALLABLE_MEMBER
double GBTDIonTheFly::ucb_amplitude(double t, double *params)
{
    double A0    = params[0];
    double f0    = params[1];
    double fdot  = params[2];

    return A0 * ( 1.0 + 2.0/3.0*fdot/f0*t );
}

CUDA_CALLABLE_MEMBER
GBTDIonTheFly::GBTDIonTheFly(Orbits *orbits_, double T_) : LISATDIonTheFly(orbits_)
{
    T = T_;
}

CUDA_CALLABLE_MEMBER
void GBTDIonTheFly::get_amp_and_phase(double *t, double *amp, double *phase, double *params, int N, int bin_i)
{
    // params are already referenced before this. 
    for(int n=0; n<N; n++)
    {
        phase[n] = ucb_phase(t[n],params);
        amp[n]   = ucb_amplitude(t[n],params);
    }
}


CUDA_CALLABLE_MEMBER
void GBTDIonTheFly::get_phase_ref(double *t, double *phase, double *params, int N, int spline_i)
{
    for (int n = 0; n < N; n += 1)
    {
        phase[n] = ucb_phase(t[n] ,params);
    }
}

CUDA_CALLABLE_MEMBER
GBTDIonTheFly::~GBTDIonTheFly()
{
    return;
}

CUDA_CALLABLE_MEMBER
void GBTDIonTheFly::run_wave_tdi(cmplx *X, cmplx *Y, cmplx *Z, double *phi_ref, double *params, double *t_arr, int N, int num_sub, int n_params)
{
     // TODO: CHECK THIS!!
    for (int bin_i = 0; bin_i < num_sub; bin_i += 1)
    {

        // map to Tyson/Neil setup
        double beta = params[bin_i * n_params + 8];
        double costh = cos(M_PI / 2.0 - beta);
        
        double lam = params[bin_i * n_params + 7];
        double phi = lam;

        double inc = params[bin_i * n_params + 5];
        double cosi = cos(inc);

        double psi = params[bin_i * n_params + 6];
        double *params_here = &params[bin_i * n_params];
        double *t_here = &t_arr[bin_i * N];

        get_tdi(
            &X[bin_i * N], &Y[bin_i * N], &Z[bin_i * N], &phi_ref[bin_i * N],
            params_here, t_here, N, costh, phi, cosi, psi, bin_i);
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



CUDA_CALLABLE_MEMBER
TDSplineTDIWaveform::TDSplineTDIWaveform(Orbits *orbits_, CubicSpline *amp_spline_, CubicSpline *phase_spline_): LISATDIonTheFly(orbits_)
{
    phase_spline = phase_spline_;
    amp_spline = amp_spline_;
    // printf("spline type init : %d %d\n", amp_spline->spline_type, phase_spline->spline_type);        
    // check_x();
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

CUDA_CALLABLE_MEMBER
void TDSplineTDIWaveform::get_amp_and_phase(double *t, double *amp, double *phase, double *params, int N, int spline_i)
{
    for (int i = 0; i < N; i += 1)
    {
        // printf("bef: t, amp, phase: %d %e\n", i, t[i]);
        amp[i] = amp_spline->eval_single(t[i], spline_i);
        phase[i] = phase_spline->eval_single(t[i], spline_i);
        // printf("af: t, amp, phase: %d %e, %e, %e\n", i, t[i], amp[i], phase[i]);
    }
}


CUDA_CALLABLE_MEMBER
void TDSplineTDIWaveform::run_wave_tdi(cmplx *X, cmplx *Y, cmplx *Z, double *phi_ref, double *params, double *t_arr, int N, int num_sub, int n_params)
{
    for (int bin_i = 0; bin_i < num_sub; bin_i += 1)
    {
        // map to Tyson/Neil setup
        double beta = params[bin_i * n_params + 3];
        double costh = cos(M_PI / 2.0 - beta);
        
        double lam = params[bin_i * n_params + 2];
        double phi = lam;

        double inc = params[bin_i * n_params + 0];
        double cosi = cos(inc);

        double psi = params[bin_i * n_params + 1];
        double *params_here = &params[bin_i * n_params];
        double *t_here = &t_arr[bin_i * N];
        
        get_tdi(
            &X[bin_i * N], &Y[bin_i * N], &Z[bin_i * N], &phi_ref[bin_i * N], 
            params_here, t_here, N, costh, phi, cosi, psi, bin_i);
    }
}

CUDA_CALLABLE_MEMBER
void TDSplineTDIWaveform::get_phase_ref(double *t, double *phase, double *params, int N, int spline_i)
{
    for (int i = 0; i < N; i += 1)
    {
        // printf("bef1: t: %d %e\n", i, t[i]);
        phase[i] = phase_spline->eval_single(t[i], spline_i);
        // printf("af1: t, phase: %d %e, %e\n", i, t[i], phase[i]);
    }
}



CUDA_CALLABLE_MEMBER
FDSplineTDIWaveform::FDSplineTDIWaveform(Orbits *orbits_, CubicSpline *amp_spline_, CubicSpline *freq_spline_): LISATDIonTheFly(orbits_)
{
    freq_spline = freq_spline_;
    amp_spline = amp_spline_;
}

CUDA_CALLABLE_MEMBER
void FDSplineTDIWaveform::get_amp_and_phase(double *t, double *amp, double *phase, double *params, int N, int spline_i)
{
    double f = 0.0;
    double t_i = 0.0;
    for (int i = 0; i < N; i += 1)
    {
        t_i = t[i];
        f = freq_spline->eval_single(t_i, spline_i);
        // printf("bef: t, amp, phase: %d %e, %e, %e\n", i, t_i, amp[i], phase[i]);
        amp[i] = 1.0;  // for frequency, we just use 1. amp_spline->eval_single(t_i, spline_i);
        phase[i] = 2. * M_PI * f * t_i;
        // printf("af: t, amp, phase: %d %e, %e, %e\n", i, t_i, amp[i], phase[i]);
    }
}


CUDA_CALLABLE_MEMBER
void FDSplineTDIWaveform::run_wave_tdi(cmplx *X, cmplx *Y, cmplx *Z, double *phi_ref, double *params, double *t_arr, int N, int num_sub, int n_params)
{
    for (int bin_i = 0; bin_i < num_sub; bin_i += 1)
    {
        // map to Tyson/Neil setup
        double beta = params[bin_i * n_params + 3];
        double costh = cos(M_PI / 2.0 - beta);
        
        double lam = params[bin_i * n_params + 2];
        double phi = lam;

        double inc = params[bin_i * n_params + 0];
        double cosi = cos(inc);

        double psi = params[bin_i * n_params + 1];
        double *params_here = &params[bin_i * n_params];
        double *t_here = &t_arr[bin_i * N];
    
        // TODO: CHECK THIS!!
        get_tdi(
            &X[bin_i * N], &Y[bin_i * N], &Z[bin_i * N], &phi_ref[bin_i * N], 
            params_here, t_here, N, costh, phi, cosi, psi, bin_i);
    }
}

CUDA_CALLABLE_MEMBER
void FDSplineTDIWaveform::get_phase_ref(double *t, double *phase, double *params, int N, int spline_i)
{
    double f = 0.0;
    double t_i = 0.0;
    for (int i = 0; i < N; i += 1)
    {
        t_i = t[i];
        f = freq_spline->eval_single(t_i, spline_i);
        phase[i] = 2. * M_PI * f * t_i;
    }
}