#include <cuda_runtime.h>
#include <stdlib.h>
#include <wb.h>
#define NUM_BINS 4096

#define CUDA_CHECK(ans)                                                   \
  	{ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  	if (code != cudaSuccess) {
    	fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    	if (abort) exit(code);
  	}
}

// version 1
// shared memory privatized version
__global__ void histogram_shared_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

	// insert your code here
	__shared__ unsigned int bins_private[4096]; // privatized bins
	int i = threadIdx.x + blockIdx.x * blockDim.x; // index
	int stride = blockDim.x * gridDim.x; // total number of threads

	// initialize privatized bins to 0
	if (threadIdx.x < 4096) bins_private[threadIdx.x] = 0;
	__syncthreads();

	// build local histogram
	while (i < num_elements) {
		int pos = input[i]; // bin position
		if (pos >= 0 && pos < 4096) // boundary condition check
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


// version 2
// your method of optimization using shared memory 
__global__ void histogram_shared_accumulate_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

	// insert your code here
	__shared__ unsigned int bins_private[4096]; // privatized bins
	int i = threadIdx.x + blockIdx.x * blockDim.x; // index
	int stride = blockDim.x * gridDim.x; // total number of threads

	// initialize privatized bins to 0
	if (threadIdx.x < 4096) bins_private[threadIdx.x] = 0;
	__syncthreads();

	// build local histogram
	while (i < num_elements) {
		int pos = input[i]; // bin position
		int j = 0;
		int count = 0;

		if (pos >= 0 && pos < 4096)  { // boundary condition check
			j = i; // set j to index value
			count = 0; // clear count value

			// check if following input values (along stride pattern) are equal to current input value
			// make sure that following values are still within bounds
			while (input[j] == pos && j < num_elements) {
				count++; // if equal, increment counter
				j += stride; // increment j by stride amount to check next value
			}
			atomicAdd(&bins_private[pos], count); // atomically increment appropriate privatized bin
		}
		// increment i by stride multiplied by the counter value, i.e. the number of 
		// contiguous input values that were equal and already added to the appropriate bin
		// this is to avoid redundant checking and incrementing
		i += stride * count;
	}
	__syncthreads();

	// build global histogram
	// number of bins > block size -- need multiple bins per thread
	for (int j = 0; j < num_bins; j += blockDim.x) {
		atomicAdd(&bins[threadIdx.x + j], bins_private[threadIdx.x + j]);
	}
}

// clipping function
// resets bins that have value larger than 127 to 127. 
// that is if bin[i]>127 then bin[i]=127
__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

	// insert your code here
	int i = threadIdx.x + blockIdx.x * blockDim.x; // index
	if (bins[i] > 127) bins[i] = 127;
}

void histogram(unsigned int *input, unsigned int *bins,
               unsigned int num_elements, unsigned int num_bins, int kernel_version) {

	if (kernel_version == 0) {
  		// zero out bins
  		CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
  		// Launch histogram kernel on the bins
  		{
    		dim3 blockDim(512), gridDim(30);
    		histogram_global_kernel<<<gridDim, blockDim, num_bins * sizeof(unsigned int)>>>(
        							  input, bins, num_elements, num_bins);
    		CUDA_CHECK(cudaGetLastError());
    		CUDA_CHECK(cudaDeviceSynchronize());
  		}

  		// Make sure bin values are not too large
  		{
    		dim3 blockDim(512);
    		dim3 gridDim((num_bins + blockDim.x - 1) / blockDim.x);
    		convert_kernel<<<gridDim, blockDim>>>(bins, num_bins);
    		CUDA_CHECK(cudaGetLastError());
    		CUDA_CHECK(cudaDeviceSynchronize());
  		}
 	}
 	
 	else if (kernel_version==1) {
 		// zero out bins
  		CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
  		// Launch histogram kernel on the bins
  		{
    		dim3 blockDim(512), gridDim(30);
    		histogram_shared_kernel<<<gridDim, blockDim, num_bins * sizeof(unsigned int)>>>(
        							  input, bins, num_elements, num_bins);
    		CUDA_CHECK(cudaGetLastError());
    		CUDA_CHECK(cudaDeviceSynchronize());
  		}

  		// Make sure bin values are not too large
  		{
    		dim3 blockDim(512);
		    dim3 gridDim((num_bins + blockDim.x - 1) / blockDim.x);
		    convert_kernel<<<gridDim, blockDim>>>(bins, num_bins);
		    CUDA_CHECK(cudaGetLastError());
		    CUDA_CHECK(cudaDeviceSynchronize());
  		}
 	}		

	else if (kernel_version==2) {
 		// zero out bins
  		CUDA_CHECK(cudaMemset(bins, 0, num_bins * sizeof(unsigned int)));
  		// Launch histogram kernel on the bins
  		{			
		    dim3 blockDim(512), gridDim(30);
		    histogram_shared_accumulate_kernel<<<gridDim, blockDim, 
		    									 num_bins * sizeof(unsigned int)>>>(
		        								 input, bins, num_elements, num_bins);
		    CUDA_CHECK(cudaGetLastError());
		    CUDA_CHECK(cudaDeviceSynchronize());
	 	}

  		// Make sure bin values are not too large
  		{
		    dim3 blockDim(512);
		    dim3 gridDim((num_bins + blockDim.x - 1) / blockDim.x);
		    convert_kernel<<<gridDim, blockDim>>>(bins, num_bins);
		    CUDA_CHECK(cudaGetLastError());
		    CUDA_CHECK(cudaDeviceSynchronize());
  		}
 	}
}

int main(int argc, char *argv[]) {
  	wbArg_t args;
  	int inputLength;
  	int version; // kernel version global or shared 
  	unsigned int *hostInput;
  	unsigned int *hostBins;
  	unsigned int *deviceInput;
  	unsigned int *deviceBins;

  	cudaEvent_t astartEvent, astopEvent;
  	float aelapsedTime;
  	cudaEventCreate(&astartEvent);
  	cudaEventCreate(&astopEvent);
  
  	args = wbArg_read(argc, argv);

  	wbTime_start(Generic, "Importing data and creating memory on host");
  	hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  	hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  	wbTime_stop(Generic, "Importing data and creating memory on host");

  	wbLog(TRACE, "The input length is ", inputLength);
  	wbLog(TRACE, "The number of bins is ", NUM_BINS);

  	wbTime_start(GPU, "Allocating GPU memory.");
  	//@@ Allocate GPU memory here
  	CUDA_CHECK(cudaMalloc((void **)&deviceInput,
               inputLength * sizeof(unsigned int)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int)));
  	CUDA_CHECK(cudaDeviceSynchronize());
  	wbTime_stop(GPU, "Allocating GPU memory.");

  	wbTime_start(GPU, "Copying input memory to the GPU.");
  	//@@ Copy memory to the GPU here
  	CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int),
               cudaMemcpyHostToDevice));
  	CUDA_CHECK(cudaDeviceSynchronize());
  	wbTime_stop(GPU, "Copying input memory to the GPU.");

  	// Launch kernel
  	// ----------------------------------------------------------
  	// wbTime_start(Compute, "Performing CUDA computation");

  	version = atoi(argv[5]); 
  	cudaEventRecord(astartEvent, 0);
  	histogram(deviceInput, deviceBins, inputLength, NUM_BINS,version);
  	// wbTime_stop(Compute, "Performing CUDA computation");

  	cudaEventRecord(astopEvent, 0);
  	cudaEventSynchronize(astopEvent);
  	cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
  	printf("\n");
  	printf("Total compute time (ms) %f for version %d\n",aelapsedTime,version);
  	printf("\n");

  	wbTime_start(Copy, "Copying output memory to the CPU");
  	//@@ Copy the GPU memory back to the CPU here
  	CUDA_CHECK(cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int),
               cudaMemcpyDeviceToHost));
  	CUDA_CHECK(cudaDeviceSynchronize());
  	wbTime_stop(Copy, "Copying output memory to the CPU");

  	// Verify correctness
  	// -----------------------------------------------------
  	printf ("running version %d\n", version);
  	if (version == 0 )
     	wbLog(TRACE, "Checking global memory only kernel");
  	else if (version == 1) 
     	wbLog(TRACE, "Launching shared memory kernel");
  	else if (version == 2) 
     	wbLog(TRACE, "Launching accumulator kernel");
  	wbSolution(args, hostBins, NUM_BINS);

  	wbTime_start(GPU, "Freeing GPU Memory");
  	//@@ Free the GPU memory here
  	CUDA_CHECK(cudaFree(deviceInput));
  	CUDA_CHECK(cudaFree(deviceBins));
  	wbTime_stop(GPU, "Freeing GPU Memory");

  	free(hostBins);
  	free(hostInput);
  	return 0;
}
