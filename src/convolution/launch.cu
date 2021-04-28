#include <wb.h>
#include "kernel.cu"
#include "../globals.h"
#include "unit_test.cu"

#define RUN_SWEEPS_CONV 0

void launch_convolution(unsigned char* image, float* outputImage, int maskWidth,  int imageWidth, int imageHeight) {

  unsigned char *deviceInputImage;
  float         *deviceOutputImage;

  int imageSize = imageWidth * imageHeight;

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceInputImage, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceOutputImage, imageSize * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  //@@ Copy memory to the GPU here
  wbTime_start(GPU, "Copying input memory to the GPU.");
  CUDA_CHECK(cudaMemcpy(deviceInputImage, image, imageSize * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));                        
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  int n_threads = 16;
  dim3 gridDim(ceil((float)imageWidth/(float)n_threads),ceil((float)imageHeight/(float)n_threads));
  dim3 blockDim(n_threads,n_threads);
  conv2d<<<gridDim, blockDim>>>(deviceInputImage, deviceOutputImage, maskWidth,
                                imageWidth, imageHeight);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(outputImage, deviceOutputImage,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  #if RUN_SWEEPS_CONV
  printf("\nRunning Convolution Sweeps\n\n");
  convolutions(deviceInputImage, deviceOutputImage, maskWidth, imageWidth, imageHeight);
  #endif
  
  CUDA_CHECK(cudaFree(deviceInputImage));
  CUDA_CHECK(cudaFree(deviceOutputImage));

}