
#include "kernel.cu"
#include "histo.cu"
#include "histo_thrust.cu"
#include "unit_test.cu"

#define RUN_SWEEPS_HISTO 0

float launch_otsu_method(unsigned char* image, int imageWidth, int imageHeight, const char* imageid) {

  unsigned int* deviceBins;
  unsigned char* deviceImage;

  float* deviceOmega;
  float* deviceMu;
  float* deviceSigmaBsq;
  float* deviceLevel;
  float* level;

  int imageSize = imageWidth * imageHeight;

  level = (float *)malloc(1 * sizeof(float));

  //@@ Allocate GPU memory here
  CUDA_CHECK( cudaMalloc((void **)&deviceImage, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceOmega, NUM_BINS * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceMu, NUM_BINS * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceSigmaBsq, NUM_BINS * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceLevel, 1 * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );

  //@@ Copy memory to the GPU here
  CUDA_CHECK(cudaMemcpy(deviceImage, image,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  #if RUN_SWEEPS_HISTO
  printf("\nRunning Histogram Sweeps\n\n");
  histograms(deviceImage, deviceBins, imageWidth, imageHeight, imageid);
  histo_thrust(image, imageWidth, imageHeight, imageid);
  #endif

  // zero out bins
  CUDA_CHECK(cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(deviceOmega, 0.0, NUM_BINS * sizeof(float)));
  CUDA_CHECK(cudaMemset(deviceMu, 0.0, NUM_BINS * sizeof(float)));
  CUDA_CHECK(cudaMemset(deviceSigmaBsq, 0.0, NUM_BINS * sizeof(float)));
  CUDA_CHECK(cudaMemset(deviceLevel, 0.0, 1 * sizeof(float)));

  // Launch histogram kernel on the bins
  dim3 blockDim(512), gridDim(30);
  timerLog_startEvent(&timerLog);
  histogram_shared_R_kernel<<<gridDim, blockDim, (NUM_BINS+1) * 12 * sizeof(unsigned int)>>>(
      deviceImage, deviceBins, imageSize,12);
  timerLog_stopEventAndLog(&timerLog, "histogram", imageid, imageWidth, imageHeight);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 blockDim1(NUM_BINS), gridDim1(1);

  timerLog_startEvent(&timerLog);
  omega<<<gridDim1,blockDim1>>>(deviceBins,deviceOmega,imageSize);
  timerLog_stopEventAndLog(&timerLog, "omega", imageid, imageWidth, imageHeight);

  timerLog_startEvent(&timerLog);
  mu<<<gridDim1,blockDim1>>>(deviceBins,deviceMu,imageSize);
  timerLog_stopEventAndLog(&timerLog, "mu", imageid, imageWidth, imageHeight);

  timerLog_startEvent(&timerLog);
  sigma_b_squared<<<gridDim1,blockDim1>>>(deviceOmega,deviceMu,deviceSigmaBsq);
  timerLog_stopEventAndLog(&timerLog, "sigma_b_squared", imageid, imageWidth, imageHeight);

  timerLog_startEvent(&timerLog);
  calculate_threshold<<<gridDim1,blockDim1>>>(deviceSigmaBsq, deviceLevel);
  timerLog_stopEventAndLog(&timerLog, "threshold calculation", imageid, imageWidth, imageHeight);

  //@@ Free the GPU memory here
  wbTime_start(GPU, "Freeing GPU Memory");
  CUDA_CHECK(cudaFree(deviceImage));
  CUDA_CHECK(cudaFree(deviceBins));
  CUDA_CHECK(cudaFree(deviceOmega));
  CUDA_CHECK(cudaFree(deviceMu));
  CUDA_CHECK(cudaFree(deviceSigmaBsq));
  wbTime_stop(GPU, "Freeing GPU Memory");

  return level[0];
}

void launch_image_binarization(unsigned char* image, unsigned char* binaryImage, float level, int imageWidth, int imageHeight, int flipped, const char* imageid) {

  unsigned char *deviceImage;
  unsigned char *deviceBinaryImage;

  int imageSize = imageWidth*imageHeight;
  
  CUDA_CHECK( cudaMalloc((void **)&deviceImage, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceBinaryImage, imageSize * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );

  //@@ Copy memory to the GPU here
  CUDA_CHECK(cudaMemcpy(deviceImage, image,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  // zero out image
  CUDA_CHECK(cudaMemset(deviceBinaryImage, 0, imageSize * sizeof(unsigned char)));

  int n_threads = 16;
  dim3 gridDim2(ceil((float)imageWidth/(float)n_threads),ceil((float)imageHeight/(float)n_threads));
  dim3 blockDim2(n_threads,n_threads);

  timerLog_startEvent(&timerLog);
  create_binarized_image<<<gridDim2, blockDim2>>>(deviceImage, deviceBinaryImage,
                                                level, imageWidth, imageHeight, flipped);
  timerLog_stopEventAndLog(&timerLog, "image binarization", imageid, imageWidth, imageHeight);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(binaryImage, deviceBinaryImage,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(deviceImage));
  CUDA_CHECK(cudaFree(deviceBinaryImage));


}