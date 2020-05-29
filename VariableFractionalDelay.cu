
#include "math.h"
#include "stdio.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__device__

double interp(double *input, int h, int d, double e, double *factorials, int start_input_ind)
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

	double sum = 0.0;
	for (int j = 1; j< h; j += 1){

		// get constants

		double first_term = factorials[h - 1] / factorials[h - 1 - j];
		double second_term = factorials[h] / factorials[h + j];
		double value = first_term * second_term;

		value = value * pow(-1.0, (double)j);

		double E = value;

		double F = j + e;
		double G = j + (1 - e);


		// perform calculation
		sum += E * (input[d + 1 + j - start_input_ind] / F + input[d - j - start_input_ind] / G);

	}
	double result = A * (B * input[d + 1 - start_input_ind] + C * input[d - start_input_ind] + D * sum);

	return result;
}

__global__
void var_frac_delay(double *new_vals, double *input_in, int num_inputs, double *delays_arr, int num_delays, int order, double sampling_frequency, int buffer_integer, double *factorials_in, int num_factorials)
{


    __shared__ double factorials[100];
    __shared__ double input[2000];
    __shared__ double first_delay;
    __shared__ double last_delay;
    __shared__ int start_input_ind;
    __shared__ int end_input_ind;

    int start, end;

    int jj = threadIdx.x + blockDim.x*blockIdx.x;
    if (jj >= num_delays) return;

    if (threadIdx.x == 0){
    	start = threadIdx.x + blockIdx.x*blockDim.x;
    	if (start + blockDim.x >= num_delays) end = num_delays - 1;
    	else end = start + blockDim.x - 1;

    	first_delay = delays_arr[start];
    	last_delay = delays_arr[end];

    	start_input_ind = (int) floor(first_delay*sampling_frequency) - buffer_integer;

    	end_input_ind = (int) ceil(last_delay*sampling_frequency) + buffer_integer;
    }

    __syncthreads();
    

    for (int i = threadIdx.x + start_input_ind; i < end_input_ind; i+=blockDim.x){
    	input[i - start_input_ind] = input_in[i];
    }

    __syncthreads();

    for (int i = threadIdx.x; i<num_factorials; i += blockDim.x){
    	factorials[i] = factorials_in[i];
	}
	__syncthreads();
	
	int point_count = order + 1;
    int half_point_count = int(point_count / 2);
    //double min_delay = (double) half_point_count / sampling_frequency;
   

    	double clipped_delay = delays_arr[jj];
    	int integer_delay = (int) ceil(clipped_delay * sampling_frequency) - 1;
    	double fraction = 1.0 + integer_delay - clipped_delay * sampling_frequency;

    	double out = interp(input, half_point_count, integer_delay, fraction, factorials, start_input_ind);

    	new_vals[jj] = out;


}

// function to find factorial of given number 
unsigned int factorial_func(unsigned int n) 
{ 
    if (n == 0) 
        return 1; 
    return n * factorial_func(n - 1); 
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


	double *input_in = new double[num_pts_in];
	double *d_input_in;

	gpuErrchk(cudaMalloc(&d_input_in, num_pts_in*sizeof(double)));

	double sampling_frequency = 1.0;
	double dt = 1./sampling_frequency;
	for (int i=0; i<num_pts_in; i+=1) input_in[i] = sin(i*dt);

	gpuErrchk(cudaMemcpy(d_input_in, input_in, num_pts_in*sizeof(double), cudaMemcpyHostToDevice));

	int num_delays = int(3.15576E+07);

	double *delays_arr = new double[num_delays];

	double start_delay = 1000.0;
	for (int i=0; i<num_delays; i+=1){
		delays_arr[i] = start_delay + 1.33333333 * i;
	}

	double *d_delays_arr;
	gpuErrchk(cudaMalloc(&d_delays_arr, num_delays*sizeof(double)));
	gpuErrchk(cudaMemcpy(d_delays_arr, delays_arr, num_delays*sizeof(double), cudaMemcpyHostToDevice));

	double *d_new_vals;
	gpuErrchk(cudaMalloc(&d_new_vals, num_delays*sizeof(double)));




	/// Run Test

	int order = 25;

	int NUM_THREADS = 256;
	int nblocks = (int) ceil((num_delays + NUM_THREADS - 1)/NUM_THREADS);

	int buffer_integer = order + 1;

	for (int k=0; k<100; k++){
	var_frac_delay<<<nblocks, NUM_THREADS>>>(d_new_vals, d_input_in, num_pts_in, d_delays_arr, num_delays, order, sampling_frequency, buffer_integer, d_factorials_in, num_fac);

	cudaDeviceSynchronize();
	gpuErrchk(cudaGetLastError());
}

	double *new_vals = new double[num_delays];
	gpuErrchk(cudaMemcpy(new_vals, d_new_vals, num_delays*sizeof(double), cudaMemcpyDeviceToHost));

	for (int i=int(1e7); i<int(1e7) + 300; i+=1) printf("%lf\n", new_vals[i]);


	gpuErrchk(cudaFree(d_input_in));
	gpuErrchk(cudaFree(d_delays_arr));
	gpuErrchk(cudaFree(d_factorials_in));
	gpuErrchk(cudaFree(d_new_vals));

	delete[] input_in;
	delete[] delays_arr;
	delete[] new_vals;
}





