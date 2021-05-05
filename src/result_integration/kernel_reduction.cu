#include "../globals.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

//Naive approach to perform array reduction using thrust library
float sum_up_arrays_thrust(float *g_idata,int imageSize) {

    thrust::device_vector<float> deviceInput(g_idata,g_idata+imageSize);
    float outputSum = thrust::reduce(deviceInput.begin(),deviceInput.end());
    return outputSum;
}

//Array Reduction using Interleaved Addressing
__global__ void reduce1(float* g_odata, float* g_idata, int len) {
    extern __shared__ float sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = 0;

    if (i < len)
    {
        sdata[tid] = g_idata[i];
    }

    __syncthreads();

    // do reduction in shared mem
    // Interleaved addressing, but threads being active/inactive
    //  is no longer based on thread IDs being powers of two. Consecutive
    //  threadIDs now run, and thus solves the thread diverging issue within
    //  a warp
    // However, this introduces shared memory bank conflicts, as threads start 
    //  out addressing with a stride of two 32-bit words (unsigned ints),
    //  and further increase the stride as the current power of two grows larger
    //  (which can worsen or lessen bank conflicts, depending on the amount
    //  of stride)
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        unsigned int index = 2 * s * tid;

        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}


//Array Reduction using Sequential Addressing to resolve Bank conflicts
__global__ void reduce2(float* g_odata, float* g_idata, int len) {
    extern __shared__ float sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = 0;

    if (i < len)
    {
        sdata[tid] = g_idata[i];
    }

    __syncthreads();

    // do reduction in shared mem
    // Sequential addressing. This solves the bank conflicts as
    //  the threads now access shared memory with a stride of one
    //  32-bit word (unsigned int) now, which does not cause bank 
    //  conflicts
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

//Array Reduction First Add during Load approach. Block size is halved.
__global__ void reduce3(float* g_odata, float* g_idata, int len) {
    extern __shared__ float sdata[];

    // Each thread loads one element from global to shared mem
    // Do the first stage of the reduction on the global-to-shared load step
    // This reduces the previous inefficiency of having half of the threads being
    //  inactive on the first for-loop iteration below (previous first step of reduction)
    // Previously, only less than or equal to 512 out of 1024 threads in a block are active.
    // Now, all 512 threads in a block are active from the start
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] = 0;

    if (i < len)
    {
        sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

//Array Reduction using Unrolling of Last Loops
__global__ void reduce4(float* g_odata, float* g_idata, int len) {
    extern __shared__ float sdata[DIM];

    // each thread loads one element from global to shared mem
    // Do the first stage of the reduction on the global-to-shared load step
    // This reduces the previous inefficiency of having half of the threads being
    //  inactive on the first for-loop iteration below (previous first step of reduction)
    // Previously, only less than or equal to 512 out of 1024 threads in a block are active.
    // Now, all 512 threads in a block are active from the start
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] = 0;

    if (i < len)
    {
        sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    }

    __syncthreads();


    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

//Method to invoke the reduction kernel recursively
float gpu_sum_reduce(float* d_in, int d_in_len)
{

    float total_sum = 0;

    int block_sz = MAX_BLOCK_SZ; // Halve the block size due to reduce3() and further 
                                              //  optimizations from there
    // our block_sum_reduce()
    int max_elems_per_block = block_sz * 2; // due to binary tree nature of algorithm
    
    int grid_sz = 0;
    if (d_in_len <= max_elems_per_block)
    {
        grid_sz = (int)std::ceil(float(d_in_len) / float(max_elems_per_block));
    }
    else
    {
        grid_sz = d_in_len / max_elems_per_block;
        if (d_in_len % max_elems_per_block != 0)
            grid_sz++;
    }
    // Allocate memory for array of total sums produced by each block
    // Array length must be the same as number of blocks / grid size
    float* d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, sizeof(float) * grid_sz));
    CUDA_CHECK(cudaMemset(d_block_sums, 0.0, sizeof(float) * grid_sz));

    // Sum data allocated for each block
    reduce4<<<grid_sz, block_sz, sizeof(float) * max_elems_per_block>>>(d_block_sums, d_in, d_in_len);
    if (grid_sz <= max_elems_per_block)
    {
        float* d_total_sum;
        CUDA_CHECK(cudaMalloc(&d_total_sum, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_total_sum, 0.0, sizeof(float)));
        reduce4<<<1, block_sz, sizeof(float) * max_elems_per_block>>>(d_total_sum, d_block_sums, grid_sz);
        CUDA_CHECK(cudaMemcpy(&total_sum, d_total_sum, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_total_sum));
    }
    else
    {
        float* d_in_block_sums;
        CUDA_CHECK(cudaMalloc(&d_in_block_sums, sizeof(float) * grid_sz));
        CUDA_CHECK(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(float) * grid_sz, cudaMemcpyDeviceToDevice));
        total_sum = gpu_sum_reduce(d_in_block_sums, grid_sz);
        CUDA_CHECK(cudaFree(d_in_block_sums));
    }

    CUDA_CHECK(cudaFree(d_block_sums));
    return total_sum;
}