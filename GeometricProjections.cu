#include "stdio.h"

#define C_inv 3.3356409519815204e-09
#define NUM_THREADS 256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ __device__
void get_basis_vecs(double lam, double beta, double u[], double v[], double k[])
{
	long i;

	double cosbeta, sinbeta, coslam, sinlam;

	for (i=0; i<3; i++)
	{
		u[i] = 0.;
		v[i] = 0.;
		k[i] = 0.;
	}

	cosbeta = cos(beta);
	sinbeta = sin(beta);

    coslam = cos(lam);
    sinlam = sin(lam);

	u[0] =  -sinbeta*coslam;  u[1] =  sinbeta*sinlam;  u[2] = cosbeta;
	v[0] =  sinlam;        v[1] = -coslam;        v[2] =  0.;
	k[0] = -cosbeta*coslam;  k[1] = -cosbeta*sinlam;  k[2] = -cosbeta;

	return;
}

__device__
double dot_product_1d(double *arr1, double *arr2){
    double out = 0.0;
    for (int i=0; i<3; i++){
        out += arr1[i]*arr2[i];
    }
    return out;
}


__device__
void xi_projections(double *xi_p, double *xi_c, double *u, double *v, double *n)
{
    double u_dot_n = dot_product_1d(u, n);
    double v_dot_n = dot_product_1d(v, n);

    *xi_p = (u_dot_n*u_dot_n) - (v_dot_n*v_dot_n);
    *xi_c = 2.0*u_dot_n*v_dot_n;
}

__device__
double interp_h(double delay)
{

    return 1.0;

}

__global__
void response(double *y_gw, double *k_in, double *u_in, double *v_in, double dt, double *x, double *n_in,
              int num, int *link_space_craft_0_in, int *link_space_craft_1_in,
              double *L_vals)
{

    __shared__ double k[3];
    __shared__ double u[3];
    __shared__ double v[3];
    __shared__ double link_space_craft_0[6];
    __shared__ double link_space_craft_1[6];

    __shared__ double x0_all[3*NUM_THREADS];
    __shared__ double x1_all[3*NUM_THREADS];
    __shared__ double n_all[3*NUM_THREADS];

    double *x0 = &x0_all[3*threadIdx.x];
    double *x1 = &x1_all[3*threadIdx.x];
    double *n = &n_all[3*threadIdx.x];

    double xi_p, xi_c;
    double k_dot_n, k_dot_x0, k_dot_x1;
    double t, L, delay0, delay1;
    double hp_del0, hp_del1, hc_del0, hc_del1;

    double large_factor, pre_factor;

    for (int i=threadIdx.x; i<3; i+=blockDim.x){
        k[i] = k_in[i];
        u[i] = u_in[i];
        v[i] = v_in[i];
    }
    __syncthreads();

    for (int i=threadIdx.x; i<6; i+=blockDim.x){
        link_space_craft_0[i] = link_space_craft_0_in[i];
        link_space_craft_1[i] = link_space_craft_1_in[i];
    }
    __syncthreads();

    for (int link_i=blockDim.y; link_i<6; link_i+=gridDim.y){

        int sc0 = link_space_craft_0[link_i];
        int sc1 = link_space_craft_1[link_i];

    for (int i=threadIdx.x + blockDim.x*blockIdx.x;
         i < num;
         i += blockDim.x*gridDim.x){


         x0[0] = x[(sc0*3 + 0)*num + i];
         x0[1] = x[(sc0*3 + 1)*num + i];
         x0[2] = x[(sc0*3 + 2)*num + i];

         x1[0] = x[(sc1*3 + 0)*num + i];
         x1[1] = x[(sc1*3 + 1)*num + i];
         x1[2] = x[(sc1*3 + 2)*num + i];



         n[0] = n_in[(link_i*3 + 0)*num + i];
         n[1] = n_in[(link_i*3 + 1)*num + i];
         n[2] = n_in[(link_i*3 + 2)*num + i];

         L = L_vals[6*num + i];
         t = i*dt;

         xi_projections(&xi_p, &xi_c, u, v, n);
         k_dot_n = dot_product_1d(k, n);
         k_dot_x0 = dot_product_1d(k, x0);
         k_dot_x1 = dot_product_1d(k, x1);

         delay0 = t - L*C_inv - k_dot_x0*C_inv;
         delay1 = t - k_dot_x1*C_inv;

         hp_del0 = interp_h(delay0);
         hc_del0 = interp_h(delay0);
         hp_del1 = interp_h(delay1);
         hc_del1 = interp_h(delay1);

         pre_factor = 1./(2*(1. - k_dot_n));
         large_factor = (hp_del0 - hp_del1)*xi_p + (hc_del0 - hc_del1)*xi_c;

         y_gw[link_i*num + i] = pre_factor*large_factor;
    }
}
}


int main()
{

    double beta = 0.5;
    double lam = 1.0;

    double k[3];
    double u[3];
    double v[3];

    get_basis_vecs(lam, beta, u, v, k);

    int num = int(1e6);
    int nlinks = 6;
    double *n_in = new double[num*nlinks*3];
    double *x = new double[num*3*3];
    double *L_vals = new double[num*nlinks];
    int *link_space_craft_0 = new int[nlinks];
    int *link_space_craft_1 = new int[nlinks];

    link_space_craft_0[0] = 0; link_space_craft_1[0] = 1;
    link_space_craft_0[1] = 1; link_space_craft_1[1] = 0;

    link_space_craft_0[2] = 0; link_space_craft_1[2] = 2;
    link_space_craft_0[3] = 2; link_space_craft_1[3] = 0;

    link_space_craft_0[4] = 1; link_space_craft_1[4] = 2;
    link_space_craft_0[5] = 2; link_space_craft_1[5] = 1;

    double Re = 1.496e+11;  // meters
    double Phi0 = 0.0;

    double Omega0 = 1/(365.25*24.0*3600.0);

    double center_vec[3];

    double dt = 1.0;
    double L = 2.5e9;



    double sc0_delta[2] = {L/2, -L/(2.*sqrt(3.))};

    double sc1_delta[2] = {-L/2, -L/(2.*sqrt(3.))};
    double sc2_delta[2] = {0.0, L/(sqrt(3.))};

    double Rnew, xnew, ynew, znew, t;
    double norm;
    int link_ind_0, link_ind_1;
    for (int i=0; i<num; i++){
        t = i*dt;

        // sc 1
        Rnew = Re + sc0_delta[0];
        xnew = Rnew*cos(Omega0*t + Phi0);
        ynew = Rnew*sin(Omega0*t + Phi0);
        znew = sc0_delta[1];

        x[(0*3 + 0)*num + i] = xnew;
        x[(0*3 + 1)*num + i] = ynew;
        x[(0*3 + 2)*num + i] = znew;

        Rnew = Re + sc1_delta[0];
        xnew = Rnew*cos(Omega0*t + Phi0);
        ynew = Rnew*sin(Omega0*t + Phi0);
        znew = sc1_delta[1];

        x[(1*3 + 0)*num + i] = xnew;
        x[(1*3 + 1)*num + i] = ynew;
        x[(1*3 + 2)*num + i] = znew;

        Rnew = Re + sc2_delta[0];
        xnew = Rnew*cos(Omega0*t + Phi0);
        ynew = Rnew*sin(Omega0*t + Phi0);
        znew = sc2_delta[1];

        x[(2*3 + 0)*num + i] = xnew;
        x[(2*3 + 1)*num + i] = ynew;
        x[(2*3 + 2)*num + i] = znew;

        for (int j=0; j<6; j++){
            link_ind_0 = link_space_craft_0[j];
            link_ind_1 = link_space_craft_1[j];

            xnew = x[(link_ind_0*3 + 0)*num + i] - x[(link_ind_1*3 + 0)*num + i];
            ynew = x[(link_ind_0*3 + 1)*num + i] - x[(link_ind_1*3 + 1)*num + i];
            znew = x[(link_ind_0*3 + 2)*num + i] - x[(link_ind_1*3 + 2)*num + i];

            norm = sqrt(xnew*xnew + ynew*ynew + znew*znew);

            n_in[(j*3 + 0)*num + i] = xnew/norm;
            n_in[(j*3 + 1)*num + i] = ynew/norm;
            n_in[(j*3 + 2)*num + i] = znew/norm;
            L_vals[j*num + i] = L;
        }
    }

    double *d_k, *d_u, *d_v, *d_x, *d_n_in;
    double *d_L_vals, *d_y_gw;

    int *d_link_space_craft_0, *d_link_space_craft_1;

    gpuErrchk(cudaMalloc(&d_k, 3*sizeof(double)));
    gpuErrchk(cudaMalloc(&d_u, 3*sizeof(double)));
    gpuErrchk(cudaMalloc(&d_v, 3*sizeof(double)));

    gpuErrchk(cudaMalloc(&d_x, 3*3*num*sizeof(double)));
    gpuErrchk(cudaMalloc(&d_n_in, nlinks*3*num*sizeof(double)));

    gpuErrchk(cudaMalloc(&d_link_space_craft_0, nlinks*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_link_space_craft_1, nlinks*sizeof(int)));

    gpuErrchk(cudaMalloc(&d_L_vals, nlinks*num*sizeof(double)));

    gpuErrchk(cudaMemcpy(d_k, &k, 3*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, &u, 3*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, &v, 3*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_x, x, 3*3*num*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_n_in, n_in, nlinks*3*num*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_link_space_craft_0, link_space_craft_0, nlinks*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_link_space_craft_1, link_space_craft_1, nlinks*sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_L_vals, L_vals, num*nlinks*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_y_gw, nlinks*num*sizeof(double)));

    int nblocks = (int) ceil((num + NUM_THREADS - 1)/NUM_THREADS);

    dim3 gridDim(nblocks, nlinks);
    response<<<gridDim, NUM_THREADS>>>(d_y_gw, d_k, d_u, d_v, dt, d_x, d_n_in,
                  num, d_link_space_craft_0, d_link_space_craft_1,
                  d_L_vals);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    delete[] n_in;
    delete[] x;
    delete[] L_vals;
    delete[] link_space_craft_0;
    delete[] link_space_craft_1;

    gpuErrchk(cudaFree(d_k));
    gpuErrchk(cudaFree(d_u));
    gpuErrchk(cudaFree(d_v));

    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_n_in));

    gpuErrchk(cudaFree(d_link_space_craft_0));
    gpuErrchk(cudaFree(d_link_space_craft_1));

    gpuErrchk(cudaFree(d_L_vals));

    gpuErrchk(cudaFree(d_y_gw));
}
