#include "stdio.h"
#include "cuda_complex.hpp"
#include "LISAResponse.hh"
#include <iostream>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#define CUDA_KERNEL __global__
#define CUDA_SHARED __shared__
#define CUDA_SYNC_THREADS __syncthreads()
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_KERNEL
#define CUDA_SHARED
#define CUDA_SYNC_THREADS
#endif

#ifdef __CUDACC__
#define gpuErrchk(ans)                         \
    {                                          \
        gpuAssert2((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert2(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#endif

CUDA_CALLABLE_MEMBER
void get_basis_vecs(double lam, double beta, double u[], double v[], double k[])
{
    long i;

    double cosbeta, sinbeta, coslam, sinlam;

    for (i = 0; i < 3; i++)
    {
        u[i] = 0.;
        v[i] = 0.;
        k[i] = 0.;
    }

    cosbeta = cos(beta);
    sinbeta = sin(beta);

    coslam = cos(lam);
    sinlam = sin(lam);

    u[0] = -sinbeta * coslam;
    u[1] = sinbeta * sinlam;
    u[2] = cosbeta;
    v[0] = sinlam;
    v[1] = -coslam;
    v[2] = 0.;
    k[0] = -cosbeta * coslam;
    k[1] = -cosbeta * sinlam;
    k[2] = -cosbeta;

    return;
}

CUDA_CALLABLE_MEMBER
double dot_product_1d(double *arr1, double *arr2)
{
    double out = 0.0;
    for (int i = 0; i < 3; i++)
    {
        out += arr1[i] * arr2[i];
    }
    return out;
}

CUDA_CALLABLE_MEMBER
void xi_projections(double *xi_p, double *xi_c, double *u, double *v, double *n)
{
    double u_dot_n = dot_product_1d(u, n);
    double v_dot_n = dot_product_1d(v, n);

    *xi_p = 0.5 * ((u_dot_n * u_dot_n) - (v_dot_n * v_dot_n));
    *xi_c = u_dot_n * v_dot_n;
}

CUDA_CALLABLE_MEMBER
double interp_h(double delay, double out)
{

    return out;
}

// with uneven spacing in t in the sparse arrays, need to determine which timesteps the dense arrays fall into
// for interpolation
// effectively the boundaries and length of each interpolation segment of the dense array in the sparse array
void find_start_inds(int start_inds[], int unit_length[], double *t_arr, double delta_t, int *length, int new_length)
{

    double T = (new_length - 1) * delta_t;
    start_inds[0] = 0;
    int i = 1;
    for (i = 1;
         i < *length;
         i += 1)
    {

        double t = t_arr[i];

        // adjust for waveforms that hit the end of the trajectory
        if (t < T)
        {
            start_inds[i] = (int)std::ceil(t / delta_t);
            unit_length[i - 1] = start_inds[i] - start_inds[i - 1];
        }
        else
        {
            start_inds[i] = new_length;
            unit_length[i - 1] = new_length - start_inds[i - 1];
            break;
        }
    }

    // fixes for not using certain segments for the interpolation
    *length = i + 1;
}

CUDA_CALLABLE_MEMBER
void interp_single(double *result, double *input, int h, int d, double e, double *A_arr, double deps, double *E_arr, int start_input_ind)
{

    int ind = (int)(e / deps);

    double frac = (e - ind * deps) / deps;
    double A = A_arr[ind] * (1. - frac) + A_arr[ind + 1] * frac;

    double B = 1.0 - e;
    double C = e;
    double D = e * (1.0 - e);

    double sum = 0.0;
    double temp_up, temp_down;
    // printf("in: %d %d\n", d, start_input_ind);
    for (int j = 1; j < h; j += 1)
    {

        // get constants

        double E = E_arr[j - 1];

        double F = j + e;
        double G = j + (1 - e);

        // printf("mid: %d %d %d\n", j, d, start_input_ind);

        // perform calculation
        temp_up = input[d + 1 + j - start_input_ind];
        temp_down = input[d - j - start_input_ind];
        sum += E * (temp_up / F + temp_down / G);
    }
    temp_up = input[d + 1 - start_input_ind];
    temp_down = input[d - start_input_ind];
    // printf("out: %d %d\n", d, start_input_ind);
    *result = A * (B * temp_up + C * temp_down + D * sum);
}

CUDA_CALLABLE_MEMBER
void interp(double *result_hp, double *result_hc, cmplx *input, int h, int d, double e, double *A_arr, double deps, double *E_arr, int start_input_ind, int i, int link_i)
{
    /*
    double A = 1.0;
    for (int i = 1; i < h; i += 1){
        A *= (i + e) * (i + 1 - e);
    }
    double denominator = factorials[h - 1] * factorials[h];
    A /= denominator;
    */

    int ind = (int)(e / deps);

    double frac = (e - ind * deps) / deps;
    double A = A_arr[ind] * (1. - frac) + A_arr[ind + 1] * frac;

    double B = 1.0 - e;
    double C = e;
    double D = e * (1.0 - e);

    double sum_hp = 0.0;
    double sum_hc = 0.0;
    cmplx temp_up, temp_down;
    // if ((i == 100) && (link_i == 0)) printf("%d %e %e %e %e %e\n", d, e, A, B, C, D);
    // printf("in: %d %d\n", d, start_input_ind);
    for (int j = 1; j < h; j += 1)
    {

        // get constants

        /*
        double first_term = factorials[h - 1] / factorials[h - 1 - j];
        double second_term = factorials[h] / factorials[h + j];
        double value = first_term * second_term;

        value = value * pow(-1.0, (double)j);
        */

        double E = E_arr[j - 1];

        double F = j + e;
        double G = j + (1 - e);

        // perform calculation
        temp_up = input[d + 1 + j - start_input_ind];
        temp_down = input[d - j - start_input_ind];

        // if ((i == 100) && (link_i == 0)) printf("mid: %d %d %d %e %e %e %e %e %e %e\n", j, d + 1 + j - start_input_ind, d - j - start_input_ind, temp_up, temp_down, E, F, G);
        sum_hp += E * (temp_up.real() / F + temp_down.real() / G);
        sum_hc += E * (temp_up.imag() / F + temp_down.imag() / G);
    }
    temp_up = input[d + 1 - start_input_ind];
    temp_down = input[d - start_input_ind];
    // printf("out: %d %d\n", d, start_input_ind);
    *result_hp = A * (B * temp_up.real() + C * temp_down.real() + D * sum_hp);
    *result_hc = A * (B * temp_up.imag() + C * temp_down.imag() + D * sum_hc);
    // if ((i == 100) && (link_i == 0)) printf("end: %e %e\n", *result_hp, *result_hc);
}

#define NUM_PARS 33
#define NUM_COEFFS 4
#define NLINKS 6
#define BUFFER_SIZE 1000
#define MAX_UNITS 200

#define MAX_A_VALS 1001
#define MAX_ORDER 40

CUDA_KERNEL
void TDI_delay(double *delayed_links, double *input_links, int num_inputs, int num_delays, double *t_arr, int *tdi_base_link, int *tdi_link_combinations, int *tdi_signs_in, int *channels, int num_units, int num_channels,
               int order, double sampling_frequency, int buffer_integer, double *A_in, double deps, int num_A, double *E_in, int tdi_start_ind, Orbits *orbits_in)
{
    Orbits orbits = *orbits_in;

#ifdef __CUDACC__
    CUDA_SHARED double input[BUFFER_SIZE];
#endif
    CUDA_SHARED double first_delay;
    CUDA_SHARED double last_delay;
    CUDA_SHARED int start_input_ind;
    CUDA_SHARED int end_input_ind;
    CUDA_SHARED double A_arr[MAX_A_VALS];
    CUDA_SHARED double E_arr[MAX_ORDER];

    int start, increment;
#ifdef __CUDACC__
    start = threadIdx.x;
    increment = blockDim.x;
#else
    start = 0;
    increment = 1;
// #pragma  omp parallel for
#endif
#ifdef __CUDACC__
#else
// #pragma  omp parallel for
#endif

#ifdef __CUDACC__
#else
// #pragma  omp parallel for
#endif
    for (int i = start; i < num_A; i += increment)
    {
        A_arr[i] = A_in[i];
        // if (threadIdx.x == 1) printf("%e %e %e\n", k[i], u[i], v[i]);
    }
    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
#else
// #pragma  omp parallel for
#endif
    for (int i = start; i < (order + 1) / 2 - 1; i += increment)
    {
        E_arr[i] = E_in[i];
        // if (threadIdx.x == 1) printf("%e %e %e\n", k[i], u[i], v[i]);
    }
    CUDA_SYNC_THREADS;

    int start1, increment1;
#ifdef __CUDACC__
    start1 = blockIdx.y;
    increment1 = gridDim.y;
#else
    start1 = 0;
    increment1 = 1;
// #pragma  omp parallel for
#endif

    for (int unit_i = start1; unit_i < num_units; unit_i += increment1)
    {
        int base_link = tdi_base_link[unit_i];
        int base_link_index = orbits.get_link_ind(base_link);

        int combination_link = tdi_link_combinations[unit_i];

        int combination_link_index;
        if (combination_link == -11)
        {
            combination_link_index = -1;
        }
        else
        {
            combination_link_index = orbits.get_link_ind(combination_link);
        }
        int sign = tdi_signs_in[unit_i];
        int channel = channels[unit_i];

        int point_count = order + 1;
        int half_point_count = int(point_count / 2);

        int start2, increment2;
#ifdef __CUDACC__
        start2 = tdi_start_ind + threadIdx.x + blockDim.x * blockIdx.x;
        increment2 = blockDim.x * gridDim.x;
#else
        start2 = tdi_start_ind;
        increment2 = 1;
// #pragma  omp parallel for
#endif
        for (int i = start2;
             i < num_delays - tdi_start_ind;
             i += increment2)
        {
            double t, L, delay;

            double large_factor, pre_factor;
            double clipped_delay, out, fraction;
            double link_delayed_out;
            int integer_delay, max_integer_delay, min_integer_delay;
            int start, end, increment;

            // at i = 0, delay ind should be at TDI_buffer = total_buffer - projection_buffer
            t = t_arr[i];
            if (combination_link == -11)
            {
                delay = t;
            }
            else
            {
                delay = t - orbits.get_light_travel_time(t, combination_link);
            }

            // delays are still with respect to projection start
            clipped_delay = delay;
            integer_delay = (int)ceil(clipped_delay * sampling_frequency) - 1;
            fraction = 1.0 + integer_delay - clipped_delay * sampling_frequency;

            max_integer_delay = integer_delay;
            max_integer_delay += 2; // encompass all
            min_integer_delay = integer_delay;

#ifdef __CUDACC__
            int max_thread_num = ((num_delays - 2 * tdi_start_ind) - blockDim.x * blockIdx.x > NUM_THREADS) ? NUM_THREADS : (num_delays - 2 * tdi_start_ind) - blockDim.x * blockIdx.x;
            CUDA_SYNC_THREADS;
            if (threadIdx.x == 0)
            {
                start_input_ind = min_integer_delay - buffer_integer;
                // printf("BAD1: %d %d %d %d %e %d \n", i, unit_i, blockIdx.x, start_input_ind, delay, max_integer_delay);
            }
            CUDA_SYNC_THREADS;
            if (threadIdx.x == max_thread_num - 1)
            {
                // if (blockIdx.x == gridDim.x - 1)
                // printf("%e %e %d %d\n", clipped_delay0, clipped_delay1, integer_delay0, integer_delay1);
                end_input_ind = max_integer_delay + buffer_integer;
                // printf("BAD2: %d %d %d %d %e %d %d\n", i, unit_i, blockIdx.x, start_input_ind, delay, max_integer_delay, start_input_ind);
            }

            CUDA_SYNC_THREADS;

            for (int jj = threadIdx.x + start_input_ind; jj < end_input_ind; jj += max_thread_num)
            {
                // need to subtract out the projection buffer

                input[jj - start_input_ind] = input_links[base_link_index * num_inputs + jj];
            }

            CUDA_SYNC_THREADS;
#else
            start_input_ind = 0;
            double *input = &input_links[base_link_index * num_inputs];
#endif
            // printf("bef: %d %d %d\n", channel, i, unit_i);
            interp_single(&link_delayed_out, input, half_point_count, integer_delay, fraction, A_arr, deps, E_arr, start_input_ind);

            link_delayed_out *= sign;

            // if ((channel == 0) && (unit_i == 2) & (i > 237790)){

            // printf("aft: %d %d %d %e\n", channel, i, unit_i, delayed_links[channel * num_delays + i]);
            // }

#ifdef __CUDACC__
            atomicAdd(&delayed_links[channel * num_delays + i], link_delayed_out);
#else
            // #pragma  omp atomic
            delayed_links[channel * num_delays + i] += link_delayed_out;
#endif
            CUDA_SYNC_THREADS;
        }
    }
}

void get_tdi_delays(double *delayed_links, double *input_links, int num_inputs, int num_delays, double *t_arr, int *tdi_base_link, int *tdi_link_combinations, int *tdi_signs_in, int *channels, int num_units, int num_channels,
                    int order, double sampling_frequency, int buffer_integer, double *A_in, double deps, int num_A, double *E_in, int tdi_start_ind, Orbits *orbits_in)
{

#ifdef __CUDACC__
    int num_blocks = std::ceil((num_delays - 2 * tdi_start_ind + NUM_THREADS - 1) / NUM_THREADS);

    dim3 gridDim(num_blocks, num_units * num_channels);

    Orbits *orbits_gpu;
    gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(Orbits)));
    gpuErrchk(cudaMemcpy(orbits_gpu, orbits_in, sizeof(Orbits), cudaMemcpyHostToDevice));

    // printf("RUNNING: %d\n", i);
    TDI_delay<<<gridDim, NUM_THREADS>>>(delayed_links, input_links, num_inputs, num_delays, t_arr, tdi_base_link, tdi_link_combinations, tdi_signs_in, channels, num_units, num_channels,
                                        order, sampling_frequency, buffer_integer, A_in, deps, num_A, E_in, tdi_start_ind, orbits_gpu);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(orbits_gpu));

#else
    TDI_delay(delayed_links, input_links, num_inputs, num_delays, t_arr, tdi_base_link, tdi_link_combinations, tdi_signs_in, channels, num_units, num_channels,
              order, sampling_frequency, buffer_integer, A_in, deps, num_A, E_in, tdi_start_ind, orbits_in);

#endif
}

CUDA_KERNEL
void response(double *y_gw, double *t_data, double *k_in, double *u_in, double *v_in, double dt,
              int num_delays,
              cmplx *input_in, int num_inputs, int order, double sampling_frequency,
              int buffer_integer, double *A_in, double deps, int num_A, double *E_in, int projections_start_ind,
              Orbits *orbits_in)
{
#ifdef __CUDACC__
    CUDA_SHARED cmplx input[BUFFER_SIZE];
#endif
    CUDA_SHARED double A_arr[MAX_A_VALS];
    CUDA_SHARED double E_arr[MAX_ORDER];
    CUDA_SHARED double first_delay;
    CUDA_SHARED double last_delay;
    CUDA_SHARED int start_input_ind;
    CUDA_SHARED int end_input_ind;

    CUDA_SHARED double k[3];
    CUDA_SHARED double u[3];
    CUDA_SHARED double v[3];
    CUDA_SHARED int link_space_craft_0[NLINKS];
    CUDA_SHARED int link_space_craft_1[NLINKS];
    CUDA_SHARED int links[NLINKS];

#ifdef __CUDACC__
    CUDA_SHARED double x0_all[NUM_THREADS * 3];
    CUDA_SHARED double x1_all[NUM_THREADS * 3];
    CUDA_SHARED double n_all[NUM_THREADS * 3];

    double *x0 = &x0_all[3 * threadIdx.x];
    double *x1 = &x1_all[3 * threadIdx.x];
    double *n = &n_all[3 * threadIdx.x];
#endif

    int start, increment;

    CUDA_SYNC_THREADS;

#ifdef __CUDACC__
    start = threadIdx.x;
    increment = blockDim.x;
#else
    start = 0;
    increment = 1;
#endif
    for (int i = start; i < 3; i += increment)
    {
        k[i] = k_in[i];
        u[i] = u_in[i];
        v[i] = v_in[i];
        // if (threadIdx.x == 1) printf("%e %e %e\n", k[i], u[i], v[i]);
    }
    CUDA_SYNC_THREADS;

    for (int i = start; i < num_A; i += increment)
    {
        A_arr[i] = A_in[i];
        // if (threadIdx.x == 1) printf("%e %e %e\n", k[i], u[i], v[i]);
    }
    CUDA_SYNC_THREADS;

    for (int i = start; i < (order + 1) / 2 - 1; i += increment)
    {
        E_arr[i] = E_in[i];
        // if (threadIdx.x == 1) printf("%e %e %e\n", k[i], u[i], v[i]);
    }
    CUDA_SYNC_THREADS;

    Orbits orbits = *orbits_in;
    for (int i = start; i < NLINKS; i += increment)
    {
        link_space_craft_0[i] = orbits.sc_r[i];
        link_space_craft_1[i] = orbits.sc_e[i];
        links[i] = orbits.links[i];
        // if (threadIdx.x == 1)
        // printf("%d %d %d %d\n", orbits.sc_r[i], orbits.sc_e[i], link_space_craft_1[i], link_space_craft_0[i]);
    }
    CUDA_SYNC_THREADS;
    int point_count = order + 1;
    int half_point_count = int(point_count / 2);

#ifdef __CUDACC__
    start = blockIdx.y;
    increment = gridDim.y;
#else
    start = 0;
    increment = 1;
#endif
    for (int link_i = start; link_i < NLINKS; link_i += increment)
    {
        int sc0 = link_space_craft_0[link_i];
        int sc1 = link_space_craft_1[link_i];
        int link = links[link_i];

        int start2, increment2;
#ifdef __CUDACC__
        start2 = projections_start_ind + threadIdx.x + blockDim.x * blockIdx.x;
        increment2 = blockDim.x * gridDim.x;
#else
        start2 = projections_start_ind;
        increment2 = 1;
#endif
        for (int i = start2;
             i < num_delays - projections_start_ind;
             i += increment2)
        {

#ifdef __CUDACC__
#else
            double x0_all[3];
            CUDA_SHARED double x1_all[3];
            CUDA_SHARED double n_all[3];

            double *x0 = &x0_all[0];
            double *x1 = &x1_all[0];
            double *n = &n_all[0];

#endif

            double xi_p, xi_c;
            double k_dot_n, k_dot_x0, k_dot_x1;
            double t, L, delay0, delay1;
            double hp_del0, hp_del1, hc_del0, hc_del1;

            double large_factor, pre_factor;
            double clipped_delay0, clipped_delay1, out, fraction0, fraction1;
            int integer_delay0, integer_delay1, max_integer_delay, min_integer_delay;

            t = t_data[i];

            Vec out_vec(0.0, 0.0, 0.0);
            double norm = 0.0;
            double n_temp;

            out_vec = orbits.get_pos(t, sc0);
            x0[0] = out_vec.x;
            x0[1] = out_vec.y;
            x0[2] = out_vec.z;

            out_vec = orbits.get_pos(t, sc1);
            x1[0] = out_vec.x;
            x1[1] = out_vec.y;
            x1[2] = out_vec.z;

            for (int coord = 0; coord < 3; coord += 1)
            {
                n_temp = x0[coord] - x1[coord];
                n[coord] = n_temp;
                norm += n_temp * n_temp;
            }

            norm = sqrt(norm);

#pragma unroll
            for (int coord = 0; coord < 3; coord += 1)
            {
                n[coord] = n[coord] / norm;
            }

            L = orbits.get_light_travel_time(t, link);
            // if (i % 10000 == 0)
            //     printf("%d %d %e %e %d %e %e %e %d\n", i, link_i, L, t, link, x0[0], x1[0], norm, sc0);

            // if (i <500) printf("%d %d: start \n", i, link_i);

            xi_projections(&xi_p, &xi_c, u, v, n);

            k_dot_n = dot_product_1d(k, n);
            k_dot_x0 = dot_product_1d(k, x0); // receiver
            k_dot_x1 = dot_product_1d(k, x1); // emitter

            delay0 = t - k_dot_x0 * C_inv;
            delay1 = t - L - k_dot_x1 * C_inv;

            // start time for hp hx is really -(projection_buffer * dt)

            // if ((i == 0) && (link_i == 0)) printf("%.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", L, delay0, delay1, x0[0], x0[1], x0[2],x1[0], x1[1], x1[2]);
            clipped_delay0 = delay0; //  - start_wave_time;
            integer_delay0 = (int)ceil(clipped_delay0 * sampling_frequency) - 1;
            fraction0 = 1.0 + integer_delay0 - clipped_delay0 * sampling_frequency;

            clipped_delay1 = delay1; //  - start_wave_time;
            integer_delay1 = (int)ceil(clipped_delay1 * sampling_frequency) - 1;
            fraction1 = 1.0 + integer_delay1 - clipped_delay1 * sampling_frequency;

            max_integer_delay = (integer_delay0 < integer_delay1) ? integer_delay1 : integer_delay0;
            max_integer_delay += 2; // encompass all
            min_integer_delay = (integer_delay0 < integer_delay1) ? integer_delay0 : integer_delay1;

#ifdef __CUDACC__
            int max_thread_num = ((num_delays - 2 * projections_start_ind) - blockDim.x * blockIdx.x > NUM_THREADS) ? NUM_THREADS : (num_delays - 2 * projections_start_ind) - blockDim.x * blockIdx.x;

            if (threadIdx.x == 0)
            {
                start_input_ind = min_integer_delay - buffer_integer;
            }

            if (threadIdx.x == max_thread_num - 1)
            {
                // if (blockIdx.x == gridDim.x - 1)
                // printf("%e %e %d %d\n", clipped_delay0, clipped_delay1, integer_delay0, integer_delay1);
                end_input_ind = max_integer_delay + buffer_integer;
            }

            CUDA_SYNC_THREADS;

            // if (blockIdx.x == gridDim.x - 1) printf("%d %e %d %d %d %d %d %d %d %d %d %d %d\n", i, L, blockIdx.x, gridDim.x, threadIdx.x, blockDim.x*blockIdx.x, num_delays, num_delays - blockDim.x*blockIdx.x, max_thread_num, start_input_ind, end_input_ind, integer_delay0, integer_delay1);
            if (end_input_ind - start_input_ind > BUFFER_SIZE)
                printf("%d %d %d %d %d %d %d %d\n", threadIdx.x, max_integer_delay, start_input_ind, end_input_ind, i, max_thread_num, num_delays, blockIdx.x * blockDim.x);

            for (int jj = threadIdx.x + start_input_ind; jj < end_input_ind; jj += max_thread_num)
            {
                // cmplx temp = input_in[jj];
                input[jj - start_input_ind] = input_in[jj];
            }

            CUDA_SYNC_THREADS;
#else
            start_input_ind = 0;
            cmplx *input = input_in;
#endif

            interp(&hp_del0, &hc_del0, input, half_point_count, integer_delay0, fraction0, A_arr, deps, E_arr, start_input_ind, i, link_i);
            interp(&hp_del1, &hc_del1, input, half_point_count, integer_delay1, fraction1, A_arr, deps, E_arr, start_input_ind, i, link_i);

            pre_factor = 1. / (1. - k_dot_n);
            large_factor = (hp_del0 - hp_del1) * xi_p + (hc_del0 - hc_del1) * xi_c;
            // if (i % 10000 == 0)
            //     printf("%d %d %e %e %e %e %e %e\n", i, link_i, pre_factor, large_factor, delay0, delay1, L, xi_p);
            y_gw[link_i * num_delays + i] = pre_factor * large_factor;
            CUDA_SYNC_THREADS;
        }
    }
}

void get_response(double *y_gw, double *t_data, double *k_in, double *u_in, double *v_in, double dt,
                  int num_delays,
                  cmplx *input_in, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer,
                  double *A_in, double deps, int num_A, double *E_in, int projections_start_ind,
                  Orbits *orbits)
{

#ifdef __CUDACC__

    int num_delays_here = (num_delays - 2 * projections_start_ind);
    int num_blocks = std::ceil((num_delays_here + NUM_THREADS - 1) / NUM_THREADS);

    // copy self to GPU
    Orbits *orbits_gpu;
    gpuErrchk(cudaMalloc(&orbits_gpu, sizeof(Orbits)));
    gpuErrchk(cudaMemcpy(orbits_gpu, orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    dim3 gridDim(num_blocks, 1);

    // printf("RUNNING: %d\n", i);
    response<<<gridDim, NUM_THREADS>>>(y_gw, t_data, k_in, u_in, v_in, dt,
                                       num_delays,
                                       input_in, num_inputs, order, sampling_frequency, buffer_integer,
                                       A_in, deps, num_A, E_in, projections_start_ind,
                                       orbits_gpu);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(orbits_gpu));
#else

    // CPU waveform generation
    // std::cout << num_delays << " " << NLINKS << std::endl;
    response(y_gw, t_data, k_in, u_in, v_in, dt,
             num_delays,
             input_in, num_inputs, order, sampling_frequency, buffer_integer,
             A_in, deps, num_A, E_in, projections_start_ind,
             orbits);
#endif
}

/*
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

    int num_pts_in = int(1e6);


    cmplx *input_in = new cmplx[num_pts_in];
    cmplx *d_input_in;

    gpuErrchk(cudaMalloc(&d_input_in, num_pts_in*sizeof(cmplx)));

    double sampling_frequency = 1.0;
    double dt = 1./sampling_frequency;
    double input_start_time = -10000.0;
    cmplx I(0.0, 1.0);
    for (int i=0; i<num_pts_in; i+=1) input_in[i] = sin(i*dt + input_start_time) + I*cos(i*dt + input_start_time);

    gpuErrchk(cudaMemcpy(d_input_in, input_in, num_pts_in*sizeof(cmplx), cudaMemcpyHostToDevice));

    int num_delays = int(1e5);

    int order = 25;
    int buffer_integer = order + 1;



    double beta = 0.5;
    double lam = 1.0;

    double k[3];
    double u[3];
    double v[3];

    get_basis_vecs(lam, beta, u, v, k);

    int nlinks = NLINKS;
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

        for (int j=0; j<NLINKS; j++){
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

    for (int i=0; i<1; i++){

        get_response(d_y_gw, d_k, d_u, d_v, dt, d_x, d_n_in,
                      num_delays, d_link_space_craft_0, d_link_space_craft_1,
                      d_L_vals, d_input_in, num_pts_in, order, sampling_frequency, buffer_integer, d_factorials_in, num_fac,  input_start_time);
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

    gpuErrchk(cudaFree(d_input_in));
    gpuErrchk(cudaFree(d_factorials_in));

    delete[] input_in;
}
*/
