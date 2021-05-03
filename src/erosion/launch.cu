#include <wb.h>
#include "kernel.cu"
#include "../globals.h"

void launch_erosion(unsigned char* image, unsigned char* shadow, unsigned char* light, int maskWidth,  int imageWidth, int imageHeight) {

  unsigned char *deviceInputImage;
  unsigned char *deviceOutputImage_shadow;
  unsigned char *deviceOutputImage_light;

  int imageSize = imageWidth * imageHeight;

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceInputImage, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceOutputImage_shadow, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceOutputImage_light, imageSize * sizeof(unsigned char)) );
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

  timerLog_startEvent(&timerLog);
  erosion_kernels<<<gridDim, blockDim>>>(deviceInputImage, deviceOutputImage_shadow, 
                                  deviceOutputImage_light, maskWidth, imageWidth, 
                                  imageHeight, "ut");
  timerLog_stopEventAndLog(&timerLog, "erosion global memory", "\0", imageWidth, imageHeight);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(shadow, deviceOutputImage_shadow,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(light, deviceOutputImage_light,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");
  
  CUDA_CHECK(cudaFree(deviceInputImage));
  CUDA_CHECK(cudaFree(deviceOutputImage_shadow));
  CUDA_CHECK(cudaFree(deviceOutputImage_light));

}