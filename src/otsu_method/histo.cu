#include "../globals.h"

#define WARP_SIZE 32

// shared memory privatized version
// include comments describing your approach
__global__ void histogram(unsigned char *input, unsigned int *bins,
                                 unsigned int num_elements) {

    //init shared memory to zero
    extern __shared__ unsigned int bins_s[];

    for(unsigned int b = threadIdx.x; b < NUM_BINS; b+=blockDim.x) {
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
        unsigned char bin = input[i];
        atomicAdd(&(bins_s[bin]), 1);
    }

    __syncthreads();

    //combine all copies of the histogram back in global memory
    for(unsigned int b = threadIdx.x; b < NUM_BINS; b+=blockDim.x) {
        atomicAdd(&(bins[b]), bins_s[b]);
    }
}

// version 0
// global memory only interleaved version
// include comments describing your approach
__global__ void histogram_global_kernel(unsigned char *input, unsigned int *bins,
                                 unsigned int num_elements) {

//determine the thread index and the stride length
unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
unsigned int stride = blockDim.x * gridDim.x;

//loop thruogh all input elements and accumulate the bins in global memory
for(unsigned int i = idx; i < num_elements; i+=stride) {
    atomicAdd(&(bins[input[i]]), 1);
}

}


// version 1
// shared memory privatized version
// include comments describing your approach
__global__ void histogram_shared_kernel(unsigned char *input, unsigned int *bins,
                                 unsigned int num_elements) {

//init shared memory to zero
extern __shared__ unsigned int bins_s[];

for(unsigned int b = threadIdx.x; b < NUM_BINS; b+=blockDim.x) {
    bins_s[b] = 0;
}

__syncthreads();

//determine the thread index and the stride length
unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
unsigned int stride = blockDim.x * gridDim.x;

//loop thruogh all input elements and accumulate the bins in shared memory
for(unsigned int i = idx; i < num_elements; i+=stride) {
    atomicAdd(&(bins_s[input[i]]), 1);
}

__syncthreads();

//combine all copies of the histogram back in global memory
for(unsigned int b = threadIdx.x; b < NUM_BINS; b+=blockDim.x) {
    atomicAdd(&(bins[b]), bins_s[b]);
}
}


// version 2
// your method of optimization using shared memory 
// include DETAILED comments describing your approach
__global__ void histogram_shared_accumulate_kernel(unsigned char *input, unsigned int *bins,
                                 unsigned int num_elements) {

 //init shared memory to zero
   extern __shared__ unsigned int bins_s[];

  for(unsigned int b = threadIdx.x; b < NUM_BINS; b+=blockDim.x) {
      bins_s[b] = 0;
  }

  __syncthreads();

  //determine the thread index and the stride length
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  //loop thruogh all input elements and accumulate the bins in shared memory
  //use a temporary counter to avoid sequential writes to the same bin
  int temp = 0;
  int prevbin = -1;
  int bin = -1;
  for(unsigned int i = idx; i < num_elements; i+=stride) {
	  bin = input[i];
	  if (prevbin != bin){

	    if (temp > 0){
        atomicAdd(&(bins_s[prevbin]), temp);
      }

      temp = 1;
	    prevbin = bin;
    }
	else {
	  temp++;
    }
  }

  //make sure not to forget the last element
  if(prevbin > -1)
    atomicAdd(&(bins_s[prevbin]), temp);

  __syncthreads();

  //combine all copies of the histogram back in global memory
  for(unsigned int b = threadIdx.x; b < NUM_BINS; b+=blockDim.x) {
      atomicAdd(&(bins[b]), bins_s[b]);
  }
}

// version 3
__global__ void histogram_shared_R_kernel(unsigned char *data, unsigned int *histo,
                                 unsigned int size, unsigned int R) {

    extern __shared__ unsigned int Hs[];

    const int warpid = (int)(threadIdx.x / WARP_SIZE);
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_block = blockDim.x / WARP_SIZE;

    const int off_rep = (NUM_BINS + 1) * (threadIdx.x % R);

    const int begin = (size / warps_block) * warpid + WARP_SIZE * blockIdx.x + lane;
    const int end = (size / warps_block) * (warpid + 1);
    const int step = WARP_SIZE * gridDim.x;

    for(int pos = threadIdx.x; pos < (NUM_BINS + 1) * R; pos += blockDim.x) Hs[pos] = 0;

    __syncthreads();

    for(int i = begin; i < end; i += step){
        unsigned int d = data[i];

        atomicAdd(&Hs[off_rep + d], 1);
    }

    __syncthreads();

    for(int pos = threadIdx.x; pos < NUM_BINS; pos += blockDim.x){
      unsigned int sum = 0;
      for(int base = 0; base < (NUM_BINS +1) * R; base += NUM_BINS +1){
        sum += Hs[base + pos];
      }
      atomicAdd(histo + pos, sum);
    }

}