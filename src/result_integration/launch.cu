#include <cv.h>
#include "kernel.cu"


void launch_result_integration( float *rgbImage, float *greyShadowMask, 
  float *greyLightMask, float *erodedShadowMask,float *erodedLightMask, float *smoothMask) {
  float *redShadowArray;
  float *greenShadowArray;
  float *blueShadowArray;
  float *redLightArray;
  float *greenLightArray;
  float *blueLightArray;
  float *erodedShadowMask;
  float *erodedLightMask; 
  float *redSumShadowArray;
  float *greenSumShadowArray;
  float *blueSumShadowArray;
  float *redSumLightArray; 
  float *greenSumLightArray;
  float *blueSumLightArray;
  float *erodedSumShadowArray;
  float *erodedSumLightArray;
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
  float *deviceErodedShadowMask;
  float *deviceErodedLightMask;
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
  int imageWidth = 2;
  int imageHeight = 2;
  int imageSize = imageHeight*imageWidth;
 
  imageWidth  = wbImage_getWidth(image);
  imageHeight = wbImage_getHeight(image);

  //@@ Copy memory to the GPU here
  wbTime_start(GPU, "Copying input memory to the GPU.");
  CUDA_CHECK(cudaMemcpy(deviceRgbImage, rgbImage,
                        imageSize * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  
  // Launch multiple_rgbImage_byMask kernel on the bins
  {
    dim3 blockDim1(8,8), gridDim1(1,1);
    multiply_rgbImage_byMask<<<gridDim, blockDim>>>(
      deviceRgbImage, deviceGreyShadowMask, 
      deviceGreyLightMask, deviceRedShadowArray,deviceGreenShadowArray,deviceBlueShadowArray,
      deviceRedLightArray,deviceGreenLightArray,deviceBlueLightArray,int imageWidth, int imageHeight, NUM_CHANNELS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  /@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(redShadowArray, deviceRedShadowArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(greenShadowArray, deviceGreenShadowArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(blueShadowArray, deviceBlueShadowArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(redLightArray, deviceRedLightArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(greenLightArray, deviceGreenLightArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(blueLightArray, deviceBlueLightArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));                       
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");


  // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
  {
    dim3 blockDim1(8,8), gridDim1(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
        deviceRedShadowArray,deviceRedSumShadowArray, int imageWidth, int imageHeight);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
   // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
   {
    dim3 blockDim1(8,8), gridDim1(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
        deviceGreenShadowArray,deviceGreenSumShadowArray, int imageWidth, int imageHeight);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
   // Launch sum_up_arrays kernel on the shadow arrays for each channel
   {
    dim3 blockDim1(8,8), gridDim1(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
        deviceBlueShadowArray,deviceBlueSumShadowArray, int imageWidth, int imageHeight);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
   // Launch sum_up_arrays kernel on the light arrays for each channel
   {
    dim3 blockDim1(8,8), gridDim1(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
        deviceRedLightArray,deviceRedSumLightArray, int imageWidth, int imageHeight);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
  {
    dim3 blockDim1(8,8), gridDim1(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
          deviceGreenLightArray,deviceGreenSumLightArray, int imageWidth, int imageHeight);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
  {
    dim3 blockDim1(8,8), gridDim1(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
          deviceBlueLightArray,deviceBlueSumLightArray, int imageWidth, int imageHeight);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
  // Launch sum_up_arrays kernel on the eroded shadow array
  {
    dim3 blockDim1(8,8), gridDim1(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
      deviceErodedShadowMask,deviceErodedSumShadowArray, int imageWidth, int imageHeight);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  } 
  // Launch sum_up_arrays kernel on the eroded light array
  {
    dim3 blockDim1(8,8), gridDim1(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
      deviceErodedLightMask,deviceErodedSumLightArray, int imageWidth, int imageHeight);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  } 

  /@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(redSumShadowArray, deviceRedSumShadowArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(greenSumShadowArray, deviceGreenSumShadowArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(blueSumShadowArray, deviceBlueSumShadowArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(redSumLightArray, deviceRedSumLightArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(greenSumLightArray, deviceGreenSumLightArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(blueSumLightArray, deviceBlueSumLightArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));  
  CUDA_CHECK(cudaMemcpy(erodedSumShadowArray, deviceErodedSumShadowArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));     
  CUDA_CHECK(cudaMemcpy(erodedSumLightArray, deviceErodedSumLightArray,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));                
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");


    // zero out bins
    CUDA_CHECK(cudaMemset(devicefinalImage, 0.0, imageSize * sizeof(float)));
  // Launch calculate_rgb_ratio kernel on the eroded shadow array
  {
    dim3 blockDim1(8,8), gridDim1(1,1);
    calculate_rgb_ratio<<<gridDim, blockDim>>>(
    deviceRedSumShadowArray, deviceGreenSumShadowArray,deviceBlueSumShadowArray,
    deviceRedSumLightArray, deviceGreenSumLightArray,deviceBlueSumLightArray,
    deviceErodedSumShadowArray,deviceErodedSumShadowArray,
    deviceRgbImage, deviceSmoothMask, devicefinalImage,
    imageWidth, imageHeight, NUM_CHANNELS);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  } 

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(finalImage, deviceFinalImage,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  //@@ Free the GPU memory here
  wbTime_start(GPU, "Freeing GPU Memory");

  CUDA_CHECK(cudaFree(deviceRgbImage));
  CUDA_CHECK(cudaFree(deviceGreyShadowMask));
  CUDA_CHECK(cudaFree(deviceGreyLightMask)); 
  CUDA_CHECK(cudaFree(deviceRedShadowArray));
  CUDA_CHECK(cudaFree(deviceGreenShadowArray));
  CUDA_CHECK(cudaFree(deviceBlueShadowArray));
  CUDA_CHECK(cudaFree(deviceRedLightArray));
  CUDA_CHECK(cudaFree(deviceGreenLightArray));
  CUDA_CHECK(cudaFree(deviceBlueLightArray));
  CUDA_CHECK(cudaFree(deviceErodedShadowMask));
  CUDA_CHECK(cudaFree(deviceErodedLightMask));
  CUDA_CHECK(cudaFree(deviceRedSumShadowArray));
  CUDA_CHECK(cudaFree(deviceGreenSumShadowArray));
  CUDA_CHECK(cudaFree(deviceBlueSumShadowArray));
  CUDA_CHECK(cudaFree(deviceRedSumLightArray)); 
  CUDA_CHECK(cudaFree(deviceGreenSumLightArray));
  CUDA_CHECK(cudaFree(deviceBlueSumLightArray));
  CUDA_CHECK(cudaFree(deviceErodedSumShadowArray));
  CUDA_CHECK(cudaFree(deviceErodedSumShadowArray));
  CUDA_CHECK(cudaFree(deviceSmoothMask));
  CUDA_CHECK(cudaFree(deviceFinalImage));
  wbTime_stop(GPU, "Freeing GPU Memory");

}