#include "TDIonTheFly.hh"
#include "Detector.hpp"
#include "LISAResponse.hh"


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
void LISATDIonTheFly::get_tdi_sub(double *M, double *Mf, int n, int N, int a, int b, int c, double* tarray, double *amp_tdi_vals, double *phase_tdi_vals, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm)
{
    double t, f, amp, phase;
    double hp, hc, hpf, hcf;

    M[n] = 0.0;
    Mf[n] = 0.0;
    
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
    M[n] += hp*Apm[b]+hc*Acm[b];
    M[n] -= hp*App[c]+hc*Acp[c];
    Mf[n] += hpf*Apm[b]+hcf*Acm[b];
    Mf[n] -= hpf*App[c]+hcf*Acp[c];
    
    t = tarray[1];
    amp = amp_tdi_vals[1];
    phase = phase_tdi_vals[1];
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] -= hp*Apm[c]+hc*Acm[c];
    M[n] += hp*App[c]+hc*Acp[c];
    Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
    Mf[n] += hpf*App[c]+hcf*Acp[c];
    
    t = tarray[2];
    amp = amp_tdi_vals[2];
    phase = phase_tdi_vals[2];
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] += hp*App[b]+hc*Acp[b];
    M[n] -= hp*Apm[b]+hc*Acm[b];
    Mf[n] += hpf*App[b]+hcf*Acp[b];
    Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
    
    t = tarray[3];
    amp = amp_tdi_vals[3];
    phase = phase_tdi_vals[3];
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] -= hp*Apm[b]+hc*Acm[b];
    M[n] += hp*Apm[c]+hc*Acm[c];
    Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
    Mf[n] += hpf*Apm[c]+hcf*Acm[c];
    
    t = tarray[4];
    amp = amp_tdi_vals[4];
    phase = phase_tdi_vals[4];
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] += hp*App[c]+hc*Acp[c];
    M[n] -= hp*App[b]+hc*Acp[b];
    Mf[n] += hpf*App[c]+hcf*Acp[c];
    Mf[n] -= hpf*App[b]+hcf*Acp[b];
    
    t = tarray[5];
    amp = amp_tdi_vals[5];
    phase = phase_tdi_vals[5];
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] -= hp*App[b]+hc*Acp[b];
    M[n] += hp*Apm[b]+hc*Acm[b];
    Mf[n] -= hpf*App[b]+hcf*Acp[b];
    Mf[n] += hpf*Apm[b]+hcf*Acm[b];
    
    t = tarray[6];
    amp = amp_tdi_vals[6];
    phase = phase_tdi_vals[6];
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] += hp*Apm[c]+hc*Acm[c];
    M[n] -= hp*App[c]+hc*Acp[c];
    Mf[n] += hpf*Apm[c]+hcf*Acm[c];
    Mf[n] -= hpf*App[c]+hcf*Acp[c];
    
    t = tarray[7];
    amp = amp_tdi_vals[7];
    phase = phase_tdi_vals[7];
    // if(freq_spline) phase = 2 * M_PI*f*t; /* mbh */
    hplus_and_hcross(t, phase, amp, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
    M[n] -= hp*Apm[c]+hc*Acm[c];
    M[n] += hp*App[b]+hc*Acp[b];
    Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
    Mf[n] += hpf*App[b]+hcf*Acp[b];    
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
void LISATDIonTheFly::get_tdi_n(double *X, double *Xf, double *Y, double *Yf, double *Z, double *Zf, double *params, double t, int m, int N, double costh, double phi, double cosi, double psi)
{
    printf("bef: inside: %d \n", m);
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

    // arm vectors
    for(i=0; i<3; i++)
    {   
        link = orbits->get_link_from_arr(i);
        // printf("need TO CHECK THIS. Probably wrong link order?.\n");

        n[i] = orbits->get_normal_unit_vec(t, link); 
        L[i] = orbits->get_light_travel_time(t, link);
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
    printf("af2: inside: %d \n", m);
    
    get_t_tdi(&t_tdi[0], &kdotr[0], &L[0], t, 0, 1, 2, m);
    get_amp_and_phase(&t_tdi[0], &amp_tdi[0], &phase_tdi[0], &params[0], 8);
    printf("af33: inside: %d \n", m);
    get_tdi_sub(X, Xf, m, N, 0, 1, 2, &t_tdi[0], &amp_tdi[0], &phase_tdi[0], Aplus, Across, cos2psi, sin2psi, &App[0], &Apm[0], &Acp[0], &Acm[0], &kdotr[0], &L[0]);

    printf("af44: inside: %d \n", m);
    
    get_t_tdi(&t_tdi[0], &kdotr[0], &L[0], t, 1, 2, 0, m);
    get_amp_and_phase(&t_tdi[0], &amp_tdi[0], &phase_tdi[0], &params[0], 8);
    get_tdi_sub(Y, Yf, m, N, 1, 2, 0, &t_tdi[0], &amp_tdi[0], &phase_tdi[0], Aplus, Across, cos2psi, sin2psi, &App[0], &Apm[0], &Acp[0], &Acm[0], &kdotr[0], &L[0]);

    printf("af4: inside: %d \n", m);
    
    get_t_tdi(&t_tdi[0], &kdotr[0], &L[0], t, 2, 0, 1, m);
    get_amp_and_phase(&t_tdi[0], &amp_tdi[0], &phase_tdi[0], &params[0], 8);
    get_tdi_sub(Z, Zf, m, N, 2, 0, 1, &t_tdi[0], &amp_tdi[0], &phase_tdi[0], Aplus, Across, cos2psi, sin2psi, &App[0], &Apm[0], &Acp[0], &Acm[0], &kdotr[0], &L[0]);

    printf("af: inside: %d \n", m);
    
}


CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::get_tdi_amp_phase(void *buffer, int buffer_size, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *params, double *t_arr, int N, double costh, double phi, double cosi, double psi, double *phi_ref)
{
    if (buffer_size < get_tdi_buffer_size(N)) printf("Bad buffer!!!!!");
    
    double *X = (double*)buffer;
    double *Xf = &X[N];
    double *Y = &Xf[N];
    double *Yf = &Y[N];
    double *Z = &Yf[N];
    double *Zf = &Z[N];
    double *flip = &Zf[N];
    double *pjump = &flip[N];
    
    get_tdi_Xf(X, Xf, Y, Yf, Z, Zf, params, t_arr, N, costh, phi, cosi, psi);
    
    extract_amplitude_and_phase(flip, pjump, N, Xamp, Xphase, X, Xf, phi_ref);
    unwrap_phase(N, Xphase);

    extract_amplitude_and_phase(flip, pjump, N, Yamp, Yphase, Y, Yf, phi_ref);
    unwrap_phase(N, Yphase);

    extract_amplitude_and_phase(flip, pjump, N, Zamp, Zphase, Z, Zf, phi_ref);
    unwrap_phase(N, Zphase);
}


CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::get_tdi_Xf(double *X, double *Xf, double *Y, double *Yf, double *Z, double *Zf, double *params, double *t_arr, int N, double costh, double phi, double cosi, double psi)
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
        get_tdi_n(X, Xf, Y, Yf, Z, Zf, params, t_arr[i], i, N, costh, phi, cosi, psi);
    }
    CUDA_SYNC_THREADS;
}

CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::get_amp_and_phase(double *t, double *amp, double *phase, double *params, int N)
{
    printf("Not Implemented. TODO: best way to do this?");
}


CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::unwrap_phase(int N, double *phase)
{
    double u, v, q;
    int i;
    
    v = phase[0];
    for(i=0; i<N ;i++)
    {
        u = phase[i];
        q = rint(fabs(u-v)/(2. * M_PI));
        if(q > 0.0)
        {
           if(v > u) u += q*2. * M_PI;
           else      u -= q*2. * M_PI;
        }
        v = u;
        phase[i] = u;
    }
}


CUDA_CALLABLE_MEMBER
void LISATDIonTheFly::extract_amplitude_and_phase(double *flip, double *pjump, int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR)
{

    int i;
    double v;
    double dA1, dA2, dA3;
    
    for(i=0; i<Ns ;i++)
    {
        As[i] = sqrt(M[i]*M[i]+Mf[i]*Mf[i]);
    }
    
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
int LISATDIonTheFly::get_tdi_buffer_size(int N)
{
    return 8 * N * sizeof(double);
}

CUDA_CALLABLE_MEMBER
double GBTDIonTheFly::ucb_phase(double t, double *params)
{
    double f0    = params[0]/T;
    double phi0  = params[6];
    double fdot  = params[7]/T/T;
    double fddot = 0.0;
    
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
void GBTDIonTheFly::get_amp_and_phase(double *t, double *amp, double *phase, double *params, int N)
{
    for(int n=0; n<N; n++)
    {
        phase[n] = ucb_phase(t[n],params);
        amp[n]   = ucb_amplitude(t[n],params);
    }
}


CUDA_CALLABLE_MEMBER
void GBTDIonTheFly::get_phase_ref(double *t, double *phase, double *params, int N)
{
    #ifdef __CUDACC__
    int start = threadIdx.x;
    int incr = blockDim.x;
#else // __CUDACC__
    int start = 0;
    int incr = 1;
#endif // __CUDACC__
    for (int n = start; n < N; n += incr)
    {
        phase[n] = ucb_phase(t[n],params);
    }
    CUDA_SYNC_THREADS;
}

CUDA_CALLABLE_MEMBER
GBTDIonTheFly::~GBTDIonTheFly()
{
    return;
}

CUDA_CALLABLE_MEMBER
void GBTDIonTheFly::run_wave_tdi(void *buffer, int buffer_size, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *params, double *t_arr, int N)
{
    // map to Tyson/Neil setup
    double beta = params[8];
    double costh = cos(M_PI / 2.0 - beta);
    
    double lam = params[7];
    double phi = lam;

    double inc = params[5];
    double cosi = cos(inc);

    double psi = params[6];

    if (buffer_size < get_gb_buffer_size(N)) printf("Bad buffer!!!!!");

    // TODO: CHECK THIS!!
    double *phi_ref = (double *)buffer;
    get_phase_ref(t_arr, phi_ref, params, N);

    void *next_buffer = (void *)&phi_ref[N];
    int next_buffer_size = 8 * N * sizeof(double);
    get_tdi_amp_phase(next_buffer, next_buffer_size, Xamp, Xphase, Yamp, Yphase, Zamp, Zphase, params, t_arr, N, costh, phi, cosi, psi, phi_ref);

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

#define CUBIC_SPLINE_LINEAR_SPACING 1
#define CUBIC_SPLINE_LOG10_SPACING 2

CUDA_CALLABLE_MEMBER
int CubicSpline::get_window(double x_new)
{
    int window = 0;
    if (spline_type == CUBIC_SPLINE_LINEAR_SPACING)
    {
        window = int(x_new / dx);
    }
    else if (spline_type == CUBIC_SPLINE_LOG10_SPACING)
    {
        window = int(log10(x_new) / dx);
    }
    else
    {
#ifdef __CUDACC__
        printf("BAD cubic spline type.");
#else
        throw std::invalid_argument("BAD cubic spline type.");
#endif // __CUDACC__
    }

    if ((window < 0) || (window >= N))
    {
#ifdef __CUDACC__
        printf("Outside spline. Using edge value.");
        if (window < 0) window = 0;
        if (window >= N) window = N - 1;
#else
        throw std::invalid_argument("Outside spline.");
#endif // __CUDACC__
    }
}

CUDA_CALLABLE_MEMBER
CubicSplineSegment CubicSpline::get_cublic_spline_segment(double x_new)
{
    int window = get_window(x_new); 
    CubicSplineSegment segment(x0[window], y0[window], c1[window], c2[window], c3[window], spline_type);
    return segment;
}


CUDA_CALLABLE_MEMBER
double CubicSpline::eval_single(double x_new)
{
    CubicSplineSegment segment = get_cublic_spline_segment(x_new);
    return segment.eval(x_new);
}


CUDA_CALLABLE_MEMBER
void CubicSpline::eval(double *y_new, double *x_new, int N)
{

    for (int i = 0; i < N; i += 1)
    {
        y_new[i] = eval_single(x_new[i]);
    }
}


CUDA_CALLABLE_MEMBER
TDSplineTDIWaveform::TDSplineTDIWaveform(Orbits *orbits_, CubicSpline *amp_spline_, CubicSpline *phase_spline_): LISATDIonTheFly(orbits_)
{
    phase_spline = phase_spline_;
    amp_spline = amp_spline_;
}

CUDA_CALLABLE_MEMBER
void TDSplineTDIWaveform::get_amp_and_phase(double *t, double *amp, double *phase, double *params, int N)
{
    for (int i = 0; i < N; i += 1)
    {
        printf("bef: t, amp, phase: %d %e, %e, %e\n", i, t[i], amp[i], phase[i]);
        amp[i] = amp_spline->eval_single(t[i]);
        phase[i] = phase_spline->eval_single(t[i]);
        printf("af: t, amp, phase: %d %e, %e, %e\n", i, t[i], amp[i], phase[i]);
    }
}


CUDA_CALLABLE_MEMBER
void TDSplineTDIWaveform::run_wave_tdi(void *buffer, int buffer_size, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *params, double *t_arr, int N)
{
    // printf("CHECK\n");
    double beta = params[3];
    double costh = cos(M_PI / 2.0 - beta);
    
    double lam = params[2];
    double phi = lam;

    double inc = params[0];
    double cosi = cos(inc);

    double psi = params[1];
    // printf("CHECK1\n");
    
     if (buffer_size < get_td_spline_buffer_size(N)) printf("Bad buffer!!!!!");

    // TODO: CHECK THIS!!
    double *phi_ref = (double *)buffer;
    get_phase_ref(t_arr, phi_ref, params, N);
 
    void *next_buffer = (void *)&phi_ref[N];
    int next_buffer_size = 8 * N * sizeof(double);
    get_tdi_amp_phase(next_buffer, next_buffer_size, Xamp, Xphase, Yamp, Yphase, Zamp, Zphase, params, t_arr, N, costh, phi, cosi, psi, phi_ref);
}

CUDA_CALLABLE_MEMBER
void TDSplineTDIWaveform::get_phase_ref(double *t, double *phase, double *params, int N)
{
    for (int i = 0; i < N; i += 1)
    {
        phase[i] = phase_spline->eval_single(t[i]);
    }
}



CUDA_CALLABLE_MEMBER
FDSplineTDIWaveform::FDSplineTDIWaveform(Orbits *orbits_, CubicSpline *amp_spline_, CubicSpline *freq_spline_): LISATDIonTheFly(orbits_)
{
    freq_spline = freq_spline_;
    amp_spline = amp_spline_;
}

CUDA_CALLABLE_MEMBER
void FDSplineTDIWaveform::get_amp_and_phase(double *t, double *amp, double *phase, double *params, int N)
{
    double f = 0.0;
    double t_i = 0.0;
    for (int i = 0; i < N; i += 1)
    {
        t_i = t[i];
        f = freq_spline->eval_single(t_i);
        printf("bef: t, amp, phase: %d %e, %e, %e\n", i, t_i, amp[i], phase[i]);
        amp[i] = amp_spline->eval_single(t_i);
        phase[i] = 2. * M_PI * f * t_i;
        printf("af: t, amp, phase: %d %e, %e, %e\n", i, t_i, amp[i], phase[i]);
    }
}


CUDA_CALLABLE_MEMBER
void FDSplineTDIWaveform::run_wave_tdi(void *buffer, int buffer_size, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *params, double *t_arr, int N)
{
    // printf("CHECK\n");
    double beta = params[3];
    double costh = cos(M_PI / 2.0 - beta);
    
    double lam = params[2];
    double phi = lam;

    double inc = params[0];
    double cosi = cos(inc);

    double psi = params[1];
    // printf("CHECK1\n");
    
     if (buffer_size < get_td_spline_buffer_size(N)) printf("Bad buffer!!!!!");

    // TODO: CHECK THIS!!
    double *phi_ref = (double *)buffer;
    get_phase_ref(t_arr, phi_ref, params, N);
 
    void *next_buffer = (void *)&phi_ref[N];
    int next_buffer_size = 8 * N * sizeof(double);
    get_tdi_amp_phase(next_buffer, next_buffer_size, Xamp, Xphase, Yamp, Yphase, Zamp, Zphase, params, t_arr, N, costh, phi, cosi, psi, phi_ref);
}

CUDA_CALLABLE_MEMBER
void FDSplineTDIWaveform::get_phase_ref(double *t, double *phase, double *params, int N)
{
    double f = 0.0;
    double t_i = 0.0;
    for (int i = 0; i < N; i += 1)
    {
        t_i = t[i];
        f = freq_spline->eval_single(t_i);
        phase[i] = 2. * M_PI * f * t_i;
    }
}