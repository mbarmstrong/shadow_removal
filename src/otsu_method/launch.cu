
#include "kernel.cu"

void launch_otsu_method(wbImage_t& image, unsigned int* bins) {

  unsigned int* deviceBins;
  float* deviceImageData;
  float* hostImageData;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  int imageSize;

  hostImageData  = wbImage_getData(image);
  
  imageWidth  = wbImage_getWidth(image);
  imageHeight = wbImage_getHeight(image);
  imageChannels = wbImage_getChannels(image);

  imageSize = imageWidth * imageHeight * imageChannels;

  //@@ Allocate GPU memory here
  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceImageData,imageSize * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  //@@ Copy memory to the GPU here
  wbTime_start(GPU, "Copying input memory to the GPU.");
  CUDA_CHECK(cudaMemcpy(deviceImageData, hostImageData,
                        imageSize * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // zero out bins
  CUDA_CHECK(cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int)));
  
  // // Launch histogram kernel on the bins
  // {
  //   dim3 blockDim(512), gridDim(30);
  //   histogram_shared_kernel<<<gridDim, blockDim,num_bins * sizeof(unsigned int)>>>(
  //       input, bins, num_elements, num_bins);
  //   CUDA_CHECK(cudaGetLastError());
  //   CUDA_CHECK(cudaDeviceSynchronize());
  // }

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(bins, deviceBins,
                        NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  //@@ Free the GPU memory here
  wbTime_start(GPU, "Freeing GPU Memory");
  CUDA_CHECK(cudaFree(deviceImageData));
  CUDA_CHECK(cudaFree(deviceBins));
  wbTime_stop(GPU, "Freeing GPU Memory");

  printf("\n\nBin[0] is %d\n\n",bins[0]);

}