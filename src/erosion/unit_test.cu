#include <wb.h>
#include "kernel.cu"
#include "../globals.h"

int main(void) {

  int imageWidth;
  int imageHeight;
  int imageSize;

  unsigned char *hostInputImage;
  float *hostOutputImage;

  unsigned char *deviceInputImage;
  float *deviceOutputImage;

  float *hostMask;
  float *deviceMask;

  int maskWidth = 1;
  int imageHeight = 2;
  int imageWidth = 2;

  imageSize = imageHeight * imageWidth;

  hostInputImage = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  hostOutputImage = (float *)malloc(imageSize * sizeof(float));
  hostMask = (float *)malloc(maskWidth * maskWidth * sizeof(float));

  printf("\n\ninput image:\t");
  for(int i = 0; i < imageSize; i++){
    hostInputImage[i] = 1;
    printf("%d, ", hostInputImage[i]); 
  }
  printf("\n\n");
  
  printf("\n\nmask:\t");
  for(int i = 0; i < maskWidth*maskWidth; i++){
    hostMask[i] = 0.5;
    printf("%f, ", hostMask[i]); 
  }
  printf("\n\n");

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceInputImage, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceOutputImage, imageSize * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceMask, maskWidth * maskWidth * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  //@@ Copy memory to the GPU here
  wbTime_start(GPU, "Copying input memory to the GPU.");
  CUDA_CHECK(cudaMemcpy(deviceInputImage, hostInputImage,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceMask, hostMask,
                        maskWidth * maskWidth * sizeof(float),
                        cudaMemcpyHostToDevice));                        
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  dim3 blockDim1(8,8), gridDim1(1,1);
  conv2d<<<gridDim1, blockDim1>>>(deviceInputImage, deviceMask, deviceOutputImage,
                                  maskWidth, imageWidth, imageHeight);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(hostOutputImage, deviceOutputImage,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  printf("\n\noutput image:\t");
  for(int i = 0; i < imageSize; i++){
      printf("%.5f, ", hostOutputImage[i]);
  }
  printf("\n\n");

  CUDA_CHECK(cudaFree(deviceInputImage));
  CUDA_CHECK(cudaFree(deviceOutputImage));

  return 0;
}