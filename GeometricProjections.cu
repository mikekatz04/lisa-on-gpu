#include "stdio.h"
#include "cuda_complex.hpp"

#define C_inv 3.3356409519815204e-09
#define NUM_THREADS 256

typedef gcmplx::complex<double> cmplx;

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
double interp_h(double delay, double out)
{

    return out;

}

__device__
void interp(double *result_hp, double *result_hc, cmplx *input, int h, int d, double e, double *factorials, int start_input_ind)
{

	double A = 1.0;
	for (int i = 1; i < h; i += 1){
		A *= (i + e) * (i + 1 - e);
	}
	double denominator = factorials[h - 1] * factorials[h];
    A /= denominator;

	double B = 1.0 - e;
	double C = e;
	double D = e * (1.0 - e);

	double sum_hp = 0.0;
    double sum_hc = 0.0;
    cmplx temp_up, temp_down;
    //printf("in: %d %d\n", d, start_input_ind);
	for (int j = 1; j< h; j += 1){

		// get constants

		double first_term = factorials[h - 1] / factorials[h - 1 - j];
		double second_term = factorials[h] / factorials[h + j];
		double value = first_term * second_term;

		value = value * pow(-1.0, (double)j);

		double E = value;

		double F = j + e;
		double G = j + (1 - e);

        //printf("mid: %d %d %d\n", j, d, start_input_ind);

		// perform calculation
        temp_up = input[d + 1 + j - start_input_ind];
        temp_down = input[d - j - start_input_ind];
		sum_hp += E * (temp_up.real() / F + temp_down.real() / G);
        sum_hc += E * (temp_up.imag() / F + temp_down.imag() / G);

	}
    temp_up = input[d + 1 - start_input_ind];
    temp_down = input[d - start_input_ind];
    //printf("out: %d %d\n", d, start_input_ind);
	*result_hp = A * (B * temp_up.real() + C * temp_down.real() + D * sum_hp);
    *result_hc = A * (B * temp_up.imag() + C * temp_down.imag() + D * sum_hc);
}

__global__
void response(double *y_gw, double *k_in, double *u_in, double *v_in, double dt, double *x, double *n_in,
              int num_delays, int *link_space_craft_0_in, int *link_space_craft_1_in,
              double *L_vals, double *new_vals, cmplx *input_in, int num_inputs, int order, double sampling_frequency, int buffer_integer, double *factorials_in, int num_factorials, double input_start_time)
{


        __shared__ double factorials[100];
        __shared__ cmplx input[1500];
        __shared__ double first_delay;
        __shared__ double last_delay;
        __shared__ int start_input_ind;
        __shared__ int end_input_ind;


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
            double clipped_delay, out, fraction;
            int integer_delay;


        int start, end;

    __syncthreads();

    for (int i=threadIdx.x; i<3; i+=blockDim.x){
        k[i] = k_in[i];
        u[i] = u_in[i];
        v[i] = v_in[i];
         //if (threadIdx.x == 1) printf("%e %e %e\n", k[i], u[i], v[i]);
    }
    __syncthreads();

    for (int i=threadIdx.x; i<6; i+=blockDim.x){
        link_space_craft_0[i] = link_space_craft_1_in[i];
        link_space_craft_1[i] = link_space_craft_0_in[i];
        //if (threadIdx.x == 1) printf("%d %d %d %d\n", link_space_craft_0_in[i],link_space_craft_1_in[i], link_space_craft_1[i], link_space_craft_0[i]);
    }
    __syncthreads();


    for (int i = threadIdx.x; i<num_factorials; i += blockDim.x){
        factorials[i] = factorials_in[i];
    }
    __syncthreads();

    int point_count = order + 1;
    int half_point_count = int(point_count / 2);

    for (int link_i=blockIdx.y; link_i<6; link_i+=gridDim.y){

        int sc0 = link_space_craft_0[link_i];
        int sc1 = link_space_craft_1[link_i];

    for (int i=threadIdx.x + blockDim.x*blockIdx.x;
         i < num_delays;
         i += blockDim.x*gridDim.x){

             if (threadIdx.x == 0){
                 start = i;
                 if (start + blockDim.x >= num_delays) end = num_delays - 1;
                 else end = start + blockDim.x - 1;

                 // makes assumption that delays and input vectors will be similar
                 // buffer index should be conservative to no risk missing points
                 // have to be careful: sampling frequency is assumed to be the same for inputs and outputs
                 first_delay = start*dt - input_start_time;
                 last_delay = end*dt - input_start_time;

                 start_input_ind = (int) floor(first_delay*sampling_frequency) - buffer_integer;

                 end_input_ind = (int) ceil(last_delay*sampling_frequency) + buffer_integer;
             }



        __syncthreads();

        for (int jj = threadIdx.x + start_input_ind; jj < end_input_ind; jj+=blockDim.x){
            input[jj - start_input_ind] = input_in[jj];
        }

        __syncthreads();
         x0[0] = x[(sc0*3 + 0)*num_delays + i];
         x0[1] = x[(sc0*3 + 1)*num_delays + i];
         x0[2] = x[(sc0*3 + 2)*num_delays + i];

         x1[0] = x[(sc1*3 + 0)*num_delays + i];
         x1[1] = x[(sc1*3 + 1)*num_delays + i];
         x1[2] = x[(sc1*3 + 2)*num_delays + i];



         n[0] = n_in[(link_i*3 + 0)*num_delays + i];
         n[1] = n_in[(link_i*3 + 1)*num_delays + i];
         n[2] = n_in[(link_i*3 + 2)*num_delays + i];

         L = L_vals[link_i*num_delays + i];
         t = i*dt;
            //if (i <500) printf("%d %d: start \n", i, link_i);

         xi_projections(&xi_p, &xi_c, u, v, n);
         k_dot_n = dot_product_1d(k, n);
         k_dot_x0 = dot_product_1d(k, x0);
         k_dot_x1 = dot_product_1d(k, x1);

         delay0 = t - L*C_inv - k_dot_x0*C_inv;
         delay1 = t - k_dot_x1*C_inv;

         clipped_delay = delay0 - input_start_time;
         integer_delay = (int) ceil(clipped_delay * sampling_frequency) - 1;
         fraction = 1.0 + integer_delay - clipped_delay * sampling_frequency;

         interp(&hp_del0, &hc_del0, input, half_point_count, integer_delay, fraction, factorials, start_input_ind);

         clipped_delay = delay1 - input_start_time;
         integer_delay = (int) ceil(clipped_delay * sampling_frequency) - 1;
         fraction = 1.0 + integer_delay - clipped_delay * sampling_frequency;

         //printf("%d %d\n", i, integer_delay);
         interp(&hp_del1, &hc_del1, input, half_point_count, integer_delay, fraction, factorials, start_input_ind);

         //hp_del0 = interp_h(delay0, 1.0);
         //if (i <500) printf("%d %d: %e \n", i, link_i, hp_del1);
         //hc_del0 = interp_h(delay0, 2.0);
         //hp_del1 = interp_h(delay1, 3.0);
         //hc_del1 = interp_h(delay1, 3.0);

         pre_factor = 1./(2*(1. - k_dot_n));
         large_factor = (hp_del0 - hp_del1)*xi_p + (hc_del0 - hc_del1)*xi_c;

         y_gw[link_i*num_delays + i] = pre_factor*large_factor;

         __syncthreads();
    }
}


        //double min_delay = (double) half_point_count / sampling_frequency;

}


int main()
{

	int num_fac = 100;
	double factorials_in[num_fac];

	factorials_in[0] = 1.0;

	for (int i=1; i<num_fac; i+=1){
		factorials_in[i] = i*factorials_in[i-1];
	}

	double *d_factorials_in;
	gpuErrchk(cudaMalloc(&d_factorials_in, num_fac*sizeof(double)));
	gpuErrchk(cudaMemcpy(d_factorials_in, factorials_in, num_fac*sizeof(double), cudaMemcpyHostToDevice));

	int num_pts_in = int(5e7);


	cmplx *input_in = new cmplx[num_pts_in];
	cmplx *d_input_in;

	gpuErrchk(cudaMalloc(&d_input_in, num_pts_in*sizeof(cmplx)));

	double sampling_frequency = 1.0;
	double dt = 1./sampling_frequency;
    double input_start_time = -10000.0;
    cmplx I(0.0, 1.0);
	for (int i=0; i<num_pts_in; i+=1) input_in[i] = sin(i*dt + input_start_time) + I*cos(i*dt + input_start_time);

	gpuErrchk(cudaMemcpy(d_input_in, input_in, num_pts_in*sizeof(cmplx), cudaMemcpyHostToDevice));

	int num_delays = int(3.15576E+07);

	double *d_new_vals;
	gpuErrchk(cudaMalloc(&d_new_vals, num_delays*sizeof(double)));

    int order = 25;
    int buffer_integer = order + 1;


    double beta = 0.5;
    double lam = 1.0;

    double k[3];
    double u[3];
    double v[3];

    get_basis_vecs(lam, beta, u, v, k);

    int nlinks = 6;
    double *n_in = new double[num_delays*nlinks*3];
    double *x = new double[num_delays*3*3];
    double *L_vals = new double[num_delays*nlinks];
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

    double L = 2.5e9;

    double sc0_delta[2] = {L/2, -L/(2.*sqrt(3.))};

    double sc1_delta[2] = {-L/2, -L/(2.*sqrt(3.))};
    double sc2_delta[2] = {0.0, L/(sqrt(3.))};

    double Rnew, xnew, ynew, znew, t;
    double norm;
    int link_ind_0, link_ind_1;
    for (int i=0; i<num_delays; i++){
        t = i*dt;

        // sc 1
        Rnew = Re + sc0_delta[0];
        xnew = Rnew*cos(Omega0*t + Phi0);
        ynew = Rnew*sin(Omega0*t + Phi0);
        znew = sc0_delta[1];

        x[(0*3 + 0)*num_delays + i] = xnew;
        x[(0*3 + 1)*num_delays + i] = ynew;
        x[(0*3 + 2)*num_delays + i] = znew;

        Rnew = Re + sc1_delta[0];
        xnew = Rnew*cos(Omega0*t + Phi0);
        ynew = Rnew*sin(Omega0*t + Phi0);
        znew = sc1_delta[1];

        x[(1*3 + 0)*num_delays + i] = xnew;
        x[(1*3 + 1)*num_delays + i] = ynew;
        x[(1*3 + 2)*num_delays + i] = znew;

        Rnew = Re + sc2_delta[0];
        xnew = Rnew*cos(Omega0*t + Phi0);
        ynew = Rnew*sin(Omega0*t + Phi0);
        znew = sc2_delta[1];

        x[(2*3 + 0)*num_delays + i] = xnew;
        x[(2*3 + 1)*num_delays + i] = ynew;
        x[(2*3 + 2)*num_delays + i] = znew;

        for (int j=0; j<6; j++){
            link_ind_0 = link_space_craft_0[j];
            link_ind_1 = link_space_craft_1[j];

            xnew = x[(link_ind_0*3 + 0)*num_delays + i] - x[(link_ind_1*3 + 0)*num_delays + i];
            ynew = x[(link_ind_0*3 + 1)*num_delays + i] - x[(link_ind_1*3 + 1)*num_delays + i];
            znew = x[(link_ind_0*3 + 2)*num_delays + i] - x[(link_ind_1*3 + 2)*num_delays + i];

            norm = sqrt(xnew*xnew + ynew*ynew + znew*znew);

            n_in[(j*3 + 0)*num_delays + i] = xnew/norm;
            n_in[(j*3 + 1)*num_delays + i] = ynew/norm;
            n_in[(j*3 + 2)*num_delays + i] = znew/norm;
            L_vals[j*num_delays + i] = L;
        }
    }

    double *d_k, *d_u, *d_v, *d_x, *d_n_in;
    double *d_L_vals, *d_y_gw;

    int *d_link_space_craft_0, *d_link_space_craft_1;

    gpuErrchk(cudaMalloc(&d_k, 3*sizeof(double)));
    gpuErrchk(cudaMalloc(&d_u, 3*sizeof(double)));
    gpuErrchk(cudaMalloc(&d_v, 3*sizeof(double)));

    gpuErrchk(cudaMalloc(&d_x, 3*3*num_delays*sizeof(double)));
    gpuErrchk(cudaMalloc(&d_n_in, nlinks*3*num_delays*sizeof(double)));

    gpuErrchk(cudaMalloc(&d_link_space_craft_0, nlinks*sizeof(int)));
    gpuErrchk(cudaMalloc(&d_link_space_craft_1, nlinks*sizeof(int)));

    gpuErrchk(cudaMalloc(&d_L_vals, nlinks*num_delays*sizeof(double)));

    gpuErrchk(cudaMemcpy(d_k, &k, 3*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, &u, 3*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, &v, 3*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_x, x, 3*3*num_delays*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_n_in, n_in, nlinks*3*num_delays*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_link_space_craft_0, link_space_craft_0, nlinks*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_link_space_craft_1, link_space_craft_1, nlinks*sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_L_vals, L_vals, num_delays*nlinks*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_y_gw, nlinks*num_delays*sizeof(double)));

    int nblocks = (int) ceil((num_delays + NUM_THREADS - 1)/NUM_THREADS);

    dim3 gridDim(nblocks, nlinks);

    for (int i=0; i<1; i++){
    response<<<gridDim, NUM_THREADS>>>(d_y_gw, d_k, d_u, d_v, dt, d_x, d_n_in,
                  num_delays, d_link_space_craft_0, d_link_space_craft_1,
                  d_L_vals,
                    d_new_vals, d_input_in, num_pts_in, order, sampling_frequency, buffer_integer, d_factorials_in, num_fac, input_start_time);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

    double *y_gw = new double[num_delays];

    gpuErrchk(cudaMemcpy(y_gw, d_y_gw, num_delays*sizeof(double), cudaMemcpyDeviceToHost));
    for (int i=0; i<1; i++) printf("%e\n", y_gw[i]);

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
    delete[] y_gw;

    double *new_vals = new double[num_delays];
	gpuErrchk(cudaMemcpy(new_vals, d_new_vals, num_delays*sizeof(double), cudaMemcpyDeviceToHost));

	//for (int i=int(1e7); i<int(1e7) + 300; i+=1) printf("%lf\n", new_vals[i]);


	gpuErrchk(cudaFree(d_input_in));
	gpuErrchk(cudaFree(d_factorials_in));
	gpuErrchk(cudaFree(d_new_vals));

	delete[] input_in;
	delete[] new_vals;
}
