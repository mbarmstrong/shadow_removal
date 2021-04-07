#include "kernel.cu"

void launch_color_conversion(wbImage_t& inputImage, wbImage_t& outputImage) {

  int imageChannels;
  int imageWidth;
  int imageHeight;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  // For this lab the value is always 3
  imageChannels = wbImage_getChannels(inputImage);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
            imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float),
            cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");

  //assume 16 threads
  int n_threads = 16;

  //init the size for blocks and grid
  dim3 grid_s(ceil(imageWidth/(float)n_threads),ceil(imageHeight/(float)n_threads));
  dim3 block_s(n_threads,n_threads,1);

  //compute grey image kernel
  colorConvert<<<grid_s,block_s>>>(deviceOutputImageData, deviceInputImageData, imageWidth, imageHeight, imageChannels);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  //copy back to host
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
            imageWidth * imageHeight * sizeof(float),
            cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

}