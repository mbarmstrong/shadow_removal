#include "../globals.h"

#include <cmath>

#include "cuda_runtime.h"

// Main strategies used:
// Process as much data as possible (in terms of algorithm correctness) in shared memory
// Use sequential addressing to get rid of bank conflicts
__global__
void block_sum_reduce(float* const d_block_sums, 
	const float* const d_in,
	const int d_in_len)
{
	extern __shared__ float s_out[];

	unsigned int glbl_tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	
	// Zero out shared memory
	s_out[threadIdx.x] = 0;
	s_out[threadIdx.x + blockDim.x] = 0;

	__syncthreads();

	// Copy d_in to shared memory per block
	if (glbl_tid < d_in_len)
	{
		s_out[threadIdx.x] = (float)d_in[glbl_tid];
		if (glbl_tid + blockDim.x < d_in_len)
			s_out[threadIdx.x + blockDim.x] = (float)d_in[glbl_tid + blockDim.x];
	}
	__syncthreads();

	// Actually do the reduction
	for (unsigned int s = blockDim.x; s > 0; s >>= 1) {
		if (tid < s) {
			s_out[tid] += s_out[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		d_block_sums[blockIdx.x] = s_out[0];
}

__global__
void block_sum_reduce(float* const d_block_sums, 
	const unsigned char* const d_in,
	const int d_in_len)
{
	extern __shared__ float s_out[];

	unsigned int glbl_tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	
	// Zero out shared memory
	s_out[threadIdx.x] = 0.0;
	s_out[threadIdx.x + blockDim.x] = 0.0;

	__syncthreads();

	// Copy d_in to shared memory per block
	if (glbl_tid < d_in_len)
	{
		s_out[threadIdx.x] = (float)d_in[glbl_tid];
		if (glbl_tid + blockDim.x < d_in_len)
			s_out[threadIdx.x + blockDim.x] = (float)d_in[glbl_tid + blockDim.x];
	}
	__syncthreads();

	// Actually do the reduction
	for (unsigned int s = blockDim.x; s > 0; s >>= 1) {
		if (tid < s) {
			s_out[tid] += s_out[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		d_block_sums[blockIdx.x] = s_out[0];
}

__global__ void reduce0(float* g_odata, float* g_idata, int len) {
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
	// Interleaved addressing, which causes huge thread divergence
	//  because threads are active/inactive according to their thread IDs
	//  being powers of two. The if conditional here is guaranteed to diverge
	//  threads within a warp.
	for (unsigned int s = 1; s < 2048; s <<= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) 
		g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce0(float* g_odata, unsigned char* g_idata, int len) {
	extern __shared__ float sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = 0.0;

	if (i < len)
	{
		sdata[tid] = (float) g_idata[i];
	}

	__syncthreads();

	// do reduction in shared mem
	// Interleaved addressing, which causes huge thread divergence
	//  because threads are active/inactive according to their thread IDs
	//  being powers of two. The if conditional here is guaranteed to diverge
	//  threads within a warp.
	for (unsigned int s = 1; s < 2048; s <<= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) 
		g_odata[blockIdx.x] = sdata[0];
}

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

__global__ void reduce1(float* g_odata, unsigned char* g_idata, int len) {
	extern __shared__ float sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = 0.0;

	if (i < len)
	{
		sdata[tid] = (float) g_idata[i];
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
__global__ void reduce2(float* g_odata, unsigned char* g_idata, int len) {
	extern __shared__ float sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = 0.0;

	if (i < len)
	{
		sdata[tid] = (float) g_idata[i];
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

__global__ void reduce3(float* g_odata, float* g_idata, int len) {
	extern __shared__ float sdata[];

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

__global__ void reduce3(float* g_odata, unsigned char* g_idata, int len) {
	extern __shared__ float sdata[];

	// each thread loads one element from global to shared mem
	// Do the first stage of the reduction on the global-to-shared load step
	// This reduces the previous inefficiency of having half of the threads being
	//  inactive on the first for-loop iteration below (previous first step of reduction)
	// Previously, only less than or equal to 512 out of 1024 threads in a block are active.
	// Now, all 512 threads in a block are active from the start
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = 0.0;

	if (i < len)
	{
		sdata[tid] = (float)g_idata[i] + (float)g_idata[i + blockDim.x];
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

__global__ void reduce4(float* g_odata, float* g_idata, int len) {
	extern __shared__ float sdata[];

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

__global__ void reduce4(float* g_odata, unsigned char* g_idata, int len) {
	extern __shared__ float sdata[];

	// each thread loads one element from global to shared mem
	// Do the first stage of the reduction on the global-to-shared load step
	// This reduces the previous inefficiency of having half of the threads being
	//  inactive on the first for-loop iteration below (previous first step of reduction)
	// Previously, only less than or equal to 512 out of 1024 threads in a block are active.
	// Now, all 512 threads in a block are active from the start
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = 0.0;

	if (i < len)
	{
		sdata[tid] = (float)g_idata[i] + (float)g_idata[i + blockDim.x];
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

void print_d_array(float* d_array, int len)
{
	float* h_array = new float[len];
	CUDA_CHECK(cudaMemcpy(h_array, d_array, sizeof(float) * len, cudaMemcpyDeviceToHost));
	for (int i = 0; i < len; ++i)
	{
		std::cout << h_array[i] << " ";
	}
	std::cout << std::endl;

	delete[] h_array;
}

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
	block_sum_reduce<<<grid_sz, block_sz, sizeof(float) * max_elems_per_block>>>(d_block_sums, d_in, d_in_len);
    //reduce4<<<grid_sz, block_sz, sizeof(float) * block_sz>>>(d_block_sums, d_in, d_in_len);
	if (grid_sz <= max_elems_per_block)
	{
		float* d_total_sum;
		CUDA_CHECK(cudaMalloc(&d_total_sum, sizeof(float)));
		CUDA_CHECK(cudaMemset(d_total_sum, 0.0, sizeof(float)));
		block_sum_reduce<<<1, block_sz, sizeof(float) * max_elems_per_block>>>(d_total_sum, d_block_sums, grid_sz);
		//reduce4<<<1, block_sz, sizeof(float) * block_sz>>>(d_total_sum, d_block_sums, grid_sz);
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


float gpu_sum_reduce(unsigned char* d_in, int d_in_len)
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
	CUDA_CHECK(cudaMemset(d_block_sums, 0, sizeof(float) * grid_sz));

	// Sum data allocated for each block
	//reduce4<<<grid_sz, block_sz, sizeof(float) * max_elems_per_block>>>(d_block_sums, d_in, d_in_len);
	block_sum_reduce<<<grid_sz, block_sz, sizeof(float) * max_elems_per_block>>>(d_block_sums, d_in, d_in_len);

	if (grid_sz <= max_elems_per_block)
	{
		float* d_total_sum;
		CUDA_CHECK(cudaMalloc(&d_total_sum, sizeof(float)));
		CUDA_CHECK(cudaMemset(d_total_sum, 0, sizeof(float)));
		block_sum_reduce<<<1, block_sz, sizeof(float) * max_elems_per_block>>>(d_total_sum, d_block_sums, grid_sz);
		//reduce4<<<1, block_sz, sizeof(float) * max_elems_per_block>>>(d_total_sum, d_block_sums, grid_sz);
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