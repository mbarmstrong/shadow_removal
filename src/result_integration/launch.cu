#include <cv.h>
#include "kernel.cu"


void launch_result_integration(wbImage_t& image, unsigned int* bins) {

  float *rgbImage;
  float *greyShadowMask;
  float *greyLightMask; 
  float *redShadowArray;
  float *greenShadowArray;
  float *blueShadowArray;
  float *redLightArray;
  float *greenLightArray;
  float *blueLightArray;
  float *erodedShadowArray;
  float *erodedLightArray;
  float *redSumShadowArray;
  float *greenSumShadowArray;
  float *blueSumShadowArray;
  float *redSumLightArray; 
  float *greenSumLightArray;
  float *blueSumLightArray;
  float *erodedSumShadowArray;
  float *erodedSumShadowArray;
  float *smoothMask;
  float *finalImage;
  float *deviceRgbImage;
  float *deviceGreyShadowMask;
  float *deviceGreyLightMask; 
  float *deviceRedShadowArray;
  float *deviceGreenShadowArray;
  float *deviceBlueShadowArray;
  float *deviceRedLightArray;
  float *deviceGreenLightArray;
  float *deviceBlueLightArray;
  float *deviceErodedShadowArray;
  float *deviceErodedLightArray;
  float *deviceRedSumShadowArray;
  float *deviceGreenSumShadowArray;
  float *deviceBlueSumShadowArray;
  float *deviceRedSumLightArray; 
  float *deviceGreenSumLightArray;
  float *deviceBlueSumLightArray;
  float *deviceErodedSumShadowArray;
  float *deviceErodedSumShadowArray;
  float *deviceSmoothMask;
  float *deviceFinalImage;
  int redRatio;
  int greenRatio;
  int blueRatio;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  int imageSize;
  int deviceImageChannels;
  int deviceImageWidth;
  int deviceImageHeight;

  rgbImage  = wbImage_getData(image);
  
  imageWidth  = wbImage_getWidth(image);
  imageHeight = wbImage_getHeight(image);
  imageChannels = wbImage_getChannels(image);


  //@@ Allocate GPU memory here
  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceImageChannels,imageChannels * sizeof(int)));
  CUDA_CHECK( cudaMalloc((void **)&deviceImageWidth,imageWidth * sizeof(int)));
  CUDA_CHECK( cudaMalloc((void **)&deviceImageHeight,imageHeight * sizeof(int)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  //@@ Copy memory to the GPU here
  wbTime_start(GPU, "Copying input memory to the GPU.");
  CUDA_CHECK(cudaMemcpy(deviceRgbImage, rgbImage,
                        rgbImage * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // zero out bins
  CUDA_CHECK(cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int)));
  
  // Launch multiple_rgbImage_byMask kernel on the bins
  {
    dim3 blockDim(512), gridDim(30);
    multiple_rgbImage_byMask<<<gridDim, blockDim>>>(
      deviceRgbImage, deviceGreyShadowMask, 
      deviceGreyLightMask, deviceRedShadowArray,deviceGreenShadowArray,deviceBlueShadowArray,
      deviceRedLightArray,deviceGreenLightArray,deviceBlueLightArray,int width, int height, int numChannels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  /@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(redShadowArray, deviceRedShadowArray,
                        deviceRedShadowArray * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(greenShadowArray, deviceGreenShadowArray,
                        deviceGreenShadowArray * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(blueShadowArray, deviceBlueShadowArray,
                        deviceBlueShadowArray * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(redLightArray, deviceRedLightArray,
                        deviceRedLightArray * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(greenShadowArray, deviceGreenLightArray,
                        deviceGreenLightArray * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(blueLightArray, deviceBlueLightArray,
                        deviceBlueLightArray * sizeof(float),
                        cudaMemcpyDeviceToHost));                       
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
  {
    dim3 blockDim(512), gridDim(30);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
        deviceRedShadowArray,deviceRedSumShadowArray, int width, int height);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
   // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
   {
    dim3 blockDim(512), gridDim(30);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
        deviceGreenShadowArray,deviceGreenSumShadowArray, int width, int height);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
   // Launch sum_up_arrays kernel on the shadow arrays for each channel
   {
    dim3 blockDim(512), gridDim(30);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
        deviceBlueShadowArray,deviceBlueSumShadowArray, int width, int height);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
   // Launch sum_up_arrays kernel on the light arrays for each channel
   {
    dim3 blockDim(512), gridDim(30);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
        deviceRedLightArray,deviceRedSumLightArray, int width, int height);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
  {
    dim3 blockDim(512), gridDim(30);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
          deviceGreenLightArray,deviceGreenSumLightArray, int width, int height);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
  {
    dim3 blockDim(512), gridDim(30);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
          deviceBlueLightArray,deviceBlueSumLightArray, int width, int height);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
  // Launch sum_up_arrays kernel on the eroded shadow array
  {
    dim3 blockDim(512), gridDim(30);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
      deviceErodedShadowArray,deviceErodedSumShadowArray, int width, int height);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  } 
  // Launch sum_up_arrays kernel on the eroded light array
  {
    dim3 blockDim(512), gridDim(30);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
      deviceErodedLightArray,deviceErodedSumLightArray, int width, int height);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  } 

  /@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(redSumShadowArray, deviceRedSumShadowArray,
                        deviceRedSumShadowArray * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(greenSumShadowArray, deviceGreenSumShadowArray,
                        deviceGreenSumShadowArray * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(blueSumShadowArray, deviceBlueSumShadowArray,
                        deviceBlueSumShadowArray * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(redSumLightArray, deviceRedSumLightArray,
                        deviceRedSumShadowArray * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(greenSumLightArray, deviceGreenSumLightArray,
                        deviceGreenSumShadowArray * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(blueSumLightArray, deviceBlueSumLightArray,
                        deviceBlueSumLightArray * sizeof(float),
                        cudaMemcpyDeviceToHost));  
  CUDA_CHECK(cudaMemcpy(erodedSumShadowArray, deviceErodedSumShadowArray,
                        deviceErodedSumShadowArray * sizeof(float),
                        cudaMemcpyDeviceToHost));                    
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // Launch calculate_rgb_ratio kernel on the eroded shadow array
  {
    dim3 blockDim(512), gridDim(30);
    calculate_rgb_ratio<<<gridDim, blockDim>>>(
    redSumShadowArray, greenSumShadowArray,blueSumShadowArray,
    redSumLightArray, greenSumLightArray,blueSumLightArray,
    erodedSumShadowArray,erodedSumShadowArray,
    rgbImage, smoothMask, finalImage,
    width, height, numChannels);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  } 

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(finalImage, deviceFinalImage,
                        deviceFinalImage * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  //@@ Free the GPU memory here
  wbTime_start(GPU, "Freeing GPU Memory");
  CUDA_CHECK(cudaFree(deviceImageData));
  CUDA_CHECK(cudaFree(deviceBins));
  wbTime_stop(GPU, "Freeing GPU Memory");

  imShow(finalImage);

}