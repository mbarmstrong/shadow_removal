
#include "kernel.cu"
#include "histo.cu"
#include "histo_thrust.cu"
#include "unit_test.cu"

#define RUN_SWEEPS_HISTO 1

float launch_otsu_method(unsigned char* image, int imageWidth, int imageHeight, const char* imageid) {

  unsigned int* deviceBins;
  unsigned char* deviceImage;

  int imageSize = imageWidth * imageHeight;

  //@@ Allocate GPU memory here
  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceImage, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  //@@ Copy memory to the GPU here
  wbTime_start(GPU, "Copying input memory to the GPU.");
  CUDA_CHECK(cudaMemcpy(deviceImage, image,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  #if RUN_SWEEPS_HISTO
  printf("\nRunning Histogram Sweeps\n\n");
  histograms(deviceImage, deviceBins, imageWidth, imageHeight, imageid);
  histo_thrust(image, imageWidth, imageHeight, imageid);
  #endif

  // zero out bins
  CUDA_CHECK(cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int)));

  // Launch histogram kernel on the bins
  dim3 blockDim(512), gridDim(30);
  histogram<<<gridDim, blockDim, NUM_BINS * sizeof(unsigned int)>>>(
      deviceImage, deviceBins, imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float* deviceOmega;

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceOmega, NUM_BINS * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  CUDA_CHECK(cudaMemset(deviceOmega, 0.0, NUM_BINS * sizeof(float)));

  dim3 blockDim1(NUM_BINS), gridDim1(1);
  omega<<<gridDim1,blockDim1>>>(deviceBins,deviceOmega,imageSize);

  float* deviceMu;
  
  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceMu, NUM_BINS * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  CUDA_CHECK(cudaMemset(deviceMu, 0.0, NUM_BINS * sizeof(float)));

  mu<<<gridDim1,blockDim1>>>(deviceBins,deviceMu,imageSize);

  float* deviceSigmaBsq;
  float* hostSigmaBsq;
  
  hostSigmaBsq = (float *)malloc(NUM_BINS * sizeof(float));

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceSigmaBsq, NUM_BINS * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  CUDA_CHECK(cudaMemset(deviceSigmaBsq, 0.0, NUM_BINS * sizeof(float)));

  sigma_b_squared<<<gridDim1,blockDim1>>>(deviceOmega,deviceMu,deviceSigmaBsq);

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(hostSigmaBsq, deviceSigmaBsq,
                        NUM_BINS * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  //Replace with kernel but probs won't gain much speedup
  float level = calculate_threshold_cpu(hostSigmaBsq);

  //@@ Free the GPU memory here
  wbTime_start(GPU, "Freeing GPU Memory");
  CUDA_CHECK(cudaFree(deviceImage));
  CUDA_CHECK(cudaFree(deviceBins));
  CUDA_CHECK(cudaFree(deviceOmega));
  CUDA_CHECK(cudaFree(deviceMu));
  CUDA_CHECK(cudaFree(deviceSigmaBsq));
  wbTime_stop(GPU, "Freeing GPU Memory");

  free(hostSigmaBsq);

  return level;
}

void launch_image_binarization(unsigned char* image, unsigned char* binaryImage, float level, int imageWidth, int imageHeight, int flipped) {

  unsigned char *deviceImage;
  unsigned char *deviceBinaryImage;

  int imageSize = imageWidth*imageHeight;
  
  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceImage, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceBinaryImage, imageSize * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  //@@ Copy memory to the GPU here
  wbTime_start(GPU, "Copying input memory to the GPU.");
  CUDA_CHECK(cudaMemcpy(deviceImage, image,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // zero out image
  CUDA_CHECK(cudaMemset(deviceBinaryImage, 0, imageSize * sizeof(unsigned char)));

  int n_threads = 16;
  dim3 gridDim2(ceil((float)imageWidth/(float)n_threads),ceil((float)imageHeight/(float)n_threads));
  dim3 blockDim2(n_threads,n_threads);
  create_binarized_image<<<gridDim2, blockDim2>>>(deviceImage, deviceBinaryImage,
                                                level, imageWidth, imageHeight, flipped);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(binaryImage, deviceBinaryImage,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  CUDA_CHECK(cudaFree(deviceImage));
  CUDA_CHECK(cudaFree(deviceBinaryImage));


}