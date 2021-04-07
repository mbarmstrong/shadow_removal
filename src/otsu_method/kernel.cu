
#include "../globals.h"

// shared memory privatized version
// include comments describing your approach
__global__ void histogram(float *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

    //init shared memory to zero
    extern __shared__ unsigned int bins_s[];

    for(unsigned int b = threadIdx.x; b < num_bins; b+=blockDim.x) {
        bins_s[b] = 0;
    }

    __syncthreads();

    //determine the thread index and the stride length
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    //loop thruogh all input elements and accumulate the bins in shared memory
    for(unsigned int i = idx; i < num_elements; i+=stride) {

        //convert from floats to uint8
        //TODO: update if other image formats are passed in
        unsigned char bin = round(input[i] * (256-1));
        atomicAdd(&(bins_s[bin]), 1);
    }

    __syncthreads();

    //combine all copies of the histogram back in global memory
    for(unsigned int b = threadIdx.x; b < num_bins; b+=blockDim.x) {
        atomicAdd(&(bins[b]), bins_s[b]);
    }
}

// kernel 2: performs scan to obtain ω(k), the zeroth-order cumulative moment
// assume number of bins is less than or equal two the total number of threads in a block
__global__ void omega(unsigned int *histo, float *omega, unsigned int num_elements) {

	__shared__ float prob[NUM_BINS]; // probability function for each value

    int tid = threadIdx.x;

	if (tid < NUM_BINS) {
		prob[tid] = float(histo[tid]) / float(num_elements);
	}

    __syncthreads();

	// omega(k) = sum(pi) from i = 1 to k 
    // nieve cumulitive sum, need to use scan... maybe something to show timing analysis
    if(tid < NUM_BINS) {
        for(int i = 0; i<=tid; i++) {
            omega[tid] += prob[i];
        }
    }
      
}

// kernel 3: performs scan to obtain μ(k), the first-order cumulative moment
// assume number of bins is less than or equal two the total number of threads in a block
__global__ void mu(unsigned int *histo, float *mu, unsigned int num_elements) {
    
	__shared__ float prob[NUM_BINS]; // probability function for each value

    int tid = threadIdx.x;

	if (tid < NUM_BINS) {
		prob[tid] = float(histo[tid]) / float(num_elements);
        prob[threadIdx.x] *= (tid+1);
	}
    __syncthreads();

	// mu(k) = sum(pi) from i = 1 to k 
    // nieve cumulitive sum, need to use scan... maybe something to show timing analysis
    if(tid < NUM_BINS) {
        for(int i = 0; i<=tid; i++) {
            mu[tid] += prob[i];
        }
    }

}

// kernel 4: calculates (σ_B)^2(k), the inter-class variance, for every bin in the 
//			 histogram (every possible threshold)
__global__ void sigma_b_squared(float *omega, float *mu, float *sigma_b_sq) {

    //get the max value of mu. Could use constant mem but there's not many bins here...
    float mu_t = mu[NUM_BINS-1];
    int tid = threadIdx.x;

    if (tid < NUM_BINS) {
		sigma_b_sq[tid] = (pow((mu_t*omega[tid] - mu[tid]),2)) / (omega[tid] * (1-omega[tid]));
	}

}

// kernel 5: use argmax to find the k that maximizes (σ_B)^2(k), the threshold calculated 
// using Otsu’s method. 
__global__ void calculate_threshold() {

}

// kernel 6: takes the single-channel input image and threshold and creates a binarized 
// 			 image based off whether the pixel was less than or greater than the threshold
__global__ void create_binarized_image() {

}