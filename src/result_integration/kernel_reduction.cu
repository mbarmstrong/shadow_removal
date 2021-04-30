#include "../globals.h"

#include <cmath>

#include "cuda_runtime.h"

__global__ void reductionFloat(float *vect, float *vecOut, int size)
{
	extern __shared__ float block[];
	unsigned int globalIndex = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int i = threadIdx.x;
	if (globalIndex < size)
	{
		block[i] = vect[globalIndex];
	}	
	else
	{
		block[i] = 0;
	}	

	__syncthreads();

	for (unsigned int j = blockDim.x / 2; j > 32; j >>= 1)
	{
		if (i < j)
			block[i] += block[i + j];

		__syncthreads();
	}

	if (i < 32)
	{
		block[i] += block[i + 32];
		block[i] += block[i + 16];
		block[i] += block[i + 8];
		block[i] += block[i + 4];
		block[i] += block[i + 2];
		block[i] += block[i + 1];
	}
	if (i == 0)
		vecOut[blockIdx.x] = block[0];
}

__global__ void reductionUnsignedChar(unsigned char *vect, unsigned char *vecOut, int size)
{
	extern __shared__ unsigned char block1[];
	unsigned int globalIndex = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int i = threadIdx.x;
	if (globalIndex < size)
	{	block1[i] = vect[globalIndex];
	}
	else
	{
		block1[i] = 0;
	}	

	__syncthreads();

	for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1)
	{
		if (i < j)
			block1[i] += block1[i + j];

		__syncthreads();
	}

	if (i < 32)
	{
		block1[i] += block1[i + 32];
		block1[i] += block1[i + 16];
		block1[i] += block1[i + 8];
		block1[i] += block1[i + 4];
		block1[i] += block1[i + 2];
		block1[i] += block1[i + 1];
	}
	if (i == 0)
		vecOut[blockIdx.x] = block1[0];
}


void sumGPUFloat(float *g_idata, float *g_odata, int size)
{
	int numInputElements = size;
	int numOutputElements;
	int threadsPerBlock = DIM;
	float *dev_g_idata;
	float *dev_g_odata;
	cudaSetDevice(0);
	cudaMalloc((float**)&dev_g_idata, size * sizeof(float));
	cudaMalloc((float**)&dev_g_odata, size * sizeof(float));
	cudaMemcpy(dev_g_idata, g_idata, size * sizeof(float), cudaMemcpyHostToDevice);

	do
	{
		numOutputElements = numInputElements / (threadsPerBlock);
		if (numInputElements % (threadsPerBlock))
			numOutputElements++;

		reductionFloat <<< numOutputElements, threadsPerBlock ,DIM * sizeof(float)>>> (dev_g_idata, dev_g_odata, numInputElements);
		printf("\nIn first reduction numOutputElements, numInputElements , dev_g_odata : %d, %d",
														numOutputElements,numInputElements);

		numInputElements = numOutputElements;
		if (numOutputElements > 1)
		{
			reductionFloat <<< numOutputElements, threadsPerBlock ,DIM * sizeof(float)>>>  (dev_g_odata, dev_g_idata, numInputElements);
			printf("\nIn Second reduction numOutputElements, numInputElements : %d, %d",
														numOutputElements,numInputElements);
		}
	} while (numOutputElements > 1);

	cudaDeviceSynchronize();
	cudaMemcpy(g_idata, dev_g_idata, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(g_odata, dev_g_odata, size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_g_idata);
	cudaFree(dev_g_odata);
}

void sumGPUUnsignedChar(unsigned char *g_idata, unsigned char *g_odata, int size)
{
	int numInputElements = size;
	int numOutputElements;
	int threadsPerBlock = DIM;
	unsigned char *dev_g_idata;
	unsigned char *dev_g_odata;
	cudaSetDevice(0);
	cudaMalloc((unsigned char**)&dev_g_idata, size * sizeof(unsigned char));
	cudaMalloc((unsigned char**)&dev_g_odata, size * sizeof(unsigned char));
	cudaMemcpy(dev_g_idata, g_idata, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

	do
	{
		numOutputElements = numInputElements / (threadsPerBlock);
		if (numInputElements % (threadsPerBlock))
			numOutputElements++;

		reductionUnsignedChar <<< numOutputElements, threadsPerBlock, DIM * sizeof(unsigned char) >>> (dev_g_idata, dev_g_odata, numInputElements);
		numInputElements = numOutputElements;
		if (numOutputElements > 1)
			reductionUnsignedChar <<< numOutputElements, threadsPerBlock , DIM * sizeof(unsigned char)>>> (dev_g_odata, dev_g_idata, numInputElements);

	} while (numOutputElements > 1);

	cudaDeviceSynchronize();
	cudaMemcpy(g_idata, dev_g_idata, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy(g_odata, dev_g_odata, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(dev_g_idata);
	cudaFree(dev_g_odata);
}