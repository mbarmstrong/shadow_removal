#include <cuda_runtime.h>
#include <stdlib.h>
#include <wb.h>
#define NUM_BINS 256

// kernel 1: takes in the single-channel image and creates a histogram of pixel intensities
__global__ void create_histogram(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements) {

	__shared__ unsigned int bins_private[NUM_BINS]; // privatized bins
	int i = threadIdx.x + blockIdx.x * blockDim.x; // index
	int stride = blockDim.x * gridDim.x; // total number of threads

	// initialize privatized bins to 0
	if (threadIdx.x < NUM_BINS) bins_private[threadIdx.x] = 0;
	__syncthreads();

	// build local histogram
	while (i < num_elements) {
		int pos = input[i]; // bin position
		if (pos >= 0 && pos < NUM_BINS) // boundary condition check
			atomicAdd(&bins_private[pos], 1); // atomically increment appropriate privatized bin
		i += stride;
	}
	__syncthreads();

	// build global histogram
	// number of bins > block size -- need multiple bins per thread
	for (int j = 0; j < num_bins; j += blockDim.x) {
		atomicAdd(&bins[threadIdx.x + j], bins_private[threadIdx.x + j]);
	}

}

// kernel 2: performs scan to obtain ω(k), the zeroth-order cumulative moment
__global__ void scan_omega() {

}

// kernel 3: performs scan to obtain μ(k), the first-order cumulative moment
__global__ void scan_mu() {

}

// kernel 4: calculates (σ_B)^2(k) for every bin in the histogram (every possible threshold)
__global__ void calculate_sigma_b_squared() {

}

// kernel 5: use argmax to find the k that maximizes (σ_B)^2(k), the threshold calculated 
// using Otsu’s method. 
__global__ void calculate_threshold() {

}

// kernel 6: takes the single-channel input image and threshold and creates a binarized 
// 			 image based off whether the pixel was less than or greater than the threshold
__global__ void create_binarized_image() {

}