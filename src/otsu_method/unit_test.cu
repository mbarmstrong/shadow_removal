
#include <wb.h>
#include "kernel.cu"
#include "../globals.h"

int main(void) {

  unsigned int* hostBins;
  unsigned int* deviceBins;
  float* hostData;
  float* deviceData;
  unsigned int dataSize = 8;

  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  hostData = (float*)malloc(dataSize * sizeof(float));

  printf("\ndata:\t");
  for(int i = 0; i < dataSize; i++){
      float n = 124+i;
      float d = NUM_BINS-1;
      hostData[i] = n/d;
      printf("%.5f, ", hostData[i]);
  }
  printf("\n\n");

  //@@ Allocate GPU memory here
  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceData, dataSize * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  //@@ Copy memory to the GPU here
  wbTime_start(GPU, "Copying input memory to the GPU.");
  CUDA_CHECK(cudaMemcpy(deviceData, hostData,
                        dataSize * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // zero out bins
  CUDA_CHECK(cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int)));
  
   // Launch histogram kernel on the bins
   {
     dim3 blockDim(512), gridDim(30);
     histogram<<<gridDim, blockDim, NUM_BINS * sizeof(unsigned int)>>>(
         deviceData, deviceBins, dataSize, NUM_BINS);
     CUDA_CHECK(cudaGetLastError());
     CUDA_CHECK(cudaDeviceSynchronize());
   }

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(hostBins, deviceBins,
                        NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

    printf("\n\nbins:\t");
  for(int i = 123; i <= 124+dataSize; i++){
      printf("%d, ", hostBins[i]);
  }
  printf("\n\n");

  float* deviceOmega;
  float* hostOmega;
  
  hostOmega = (float *)malloc(NUM_BINS * sizeof(float));

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceOmega, NUM_BINS * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  CUDA_CHECK(cudaMemset(deviceOmega, 0.0, NUM_BINS * sizeof(float)));

  dim3 blockDim(NUM_BINS), gridDim(1);
  omega<<<gridDim,blockDim>>>(deviceBins,deviceOmega,dataSize);

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(hostOmega, deviceOmega,
                        NUM_BINS * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  printf("\n\nomega:\t");
  for(int i = 123; i <= 124+dataSize; i++){
      printf("%.5f, ", hostOmega[i]);
  }
  printf("\n\n");

  float* deviceMu;
  float* hostMu;
  
  hostMu = (float *)malloc(NUM_BINS * sizeof(float));

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceMu, NUM_BINS * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  CUDA_CHECK(cudaMemset(deviceMu, 0.0, NUM_BINS * sizeof(float)));

  mu<<<gridDim,blockDim>>>(deviceBins,deviceMu,dataSize);

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(hostMu, deviceMu,
                        NUM_BINS * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  printf("\n\nmu:\t");
  for(int i = 123; i <= 124+dataSize; i++){
      printf("%.5f, ", hostMu[i]);
  }
  printf("\n\n");


  float* deviceSigmaBsq;
  float* hostSigmaBsq;
  
  hostSigmaBsq = (float *)malloc(NUM_BINS * sizeof(float));

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceSigmaBsq, NUM_BINS * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  CUDA_CHECK(cudaMemset(deviceSigmaBsq, 0.0, NUM_BINS * sizeof(float)));

  sigma_b_squared<<<gridDim,blockDim>>>(deviceOmega,deviceMu,deviceSigmaBsq);

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(hostSigmaBsq, deviceSigmaBsq,
                        NUM_BINS * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  printf("\n\nsigma_b_sq:\t");
  for(int i = 123; i <= 124+dataSize; i++){
      printf("%.5f, ", hostSigmaBsq[i]);
  }
  printf("\n\n");

  //Replace with kernel but probs won't gain much speedup
  float maxSigmaBsq = 0.0;
  int maxIdx = -1;
  for(int i = 0; i <= NUM_BINS-1; i++)
  {
      if(maxSigmaBsq < hostSigmaBsq[i]){
        
        maxSigmaBsq = hostSigmaBsq[i];
        maxIdx = i;
      }
  }

  float level = float(maxIdx) / float(NUM_BINS-1);

  printf("\n\nlevel:\t");
  printf("%.5f, ", level);
  printf("\n\n");

  float *hostInputImage;
  float *deviceInputImage;
  float *hostBinaryImage;
  float *deviceBinaryImage;
  int imageHeight = 2;
  int imageWidth = 2;
  int imageSize = imageHeight*imageWidth;

  hostInputImage = (float *)malloc(imageSize * sizeof(float));
  hostBinaryImage = (float *)malloc(imageSize * sizeof(float));

  printf("\n\n\n non binary image:\t");
  for(int i = 0; i < imageSize; i++){
    hostInputImage[i] = float(i)*0.2f + 0.1f;
    printf("%.5f, ", hostInputImage[i]); 
  }
  printf("\n\n");
  
  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceInputImage, imageSize * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceBinaryImage, imageSize * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  //@@ Copy memory to the GPU here
  wbTime_start(GPU, "Copying input memory to the GPU.");
  CUDA_CHECK(cudaMemcpy(deviceInputImage, hostInputImage,
                        imageSize * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // zero out bins
  CUDA_CHECK(cudaMemset(deviceBinaryImage, 0.0, imageSize * sizeof(float)));

  dim3 blockDim1(8,8), gridDim1(1,1);
  create_binarized_image<<<gridDim1, blockDim1>>>(deviceInputImage, deviceBinaryImage,
                                                level, imageWidth, imageHeight);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(hostBinaryImage, deviceBinaryImage,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  printf("\n\nbinary image:\t");
  for(int i = 0; i < imageSize; i++){
      printf("%.5f, ", hostBinaryImage[i]);
  }
  printf("\n\n");

  //@@ Free the GPU memory here
  wbTime_start(GPU, "Freeing GPU Memory");
  CUDA_CHECK(cudaFree(deviceData));
  CUDA_CHECK(cudaFree(deviceBins));
  CUDA_CHECK(cudaFree(deviceOmega));
  CUDA_CHECK(cudaFree(deviceMu));
  CUDA_CHECK(cudaFree(deviceSigmaBsq));
  CUDA_CHECK(cudaFree(deviceInputImage));
  CUDA_CHECK(cudaFree(deviceBinaryImage));
  wbTime_stop(GPU, "Freeing GPU Memory");

  return 0;

}