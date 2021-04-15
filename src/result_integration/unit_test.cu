
#include <wb.h>
#include "kernel.cu"
#include "../globals.h"

int main(void) {

  float *rgbImage;
  float *greyShadowMask;
  float *greyLightMask; 
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
  float *deviceerodedShadowMask;
  float *deviceerodedLightMask;
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
  int imageWidth = 2;
  int imageHeight = 2;
  int imageSize = imageHeight*imageWidth;

  rgbImage  = wbImage_getData(image);
  
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

  printf("\n\n\n redShadowArray :\t");
  for(int i = 0; i < imageSize; i++){
    printf("%.5f, ", redShadowArray[i]); 
  }
  printf("\n\n");

  printf("\n\n\n greenShadowArray :\t");
  for(int i = 0; i < imageSize; i++){
    printf("%.5f, ", greenShadowArray[i]); 
  }
  printf("\n\n");

  printf("\n\n\n blueShadowArray :\t");
  for(int i = 0; i < imageSize; i++){
    printf("%.5f, ", blueShadowArray[i]); 
  }
  printf("\n\n");

  printf("\n\n\n redLightArray :\t");
  for(int i = 0; i < imageSize; i++){
    printf("%.5f, ", redLightArray[i]); 
  }
  printf("\n\n");

  printf("\n\n\n greenLightArray :\t");
  for(int i = 0; i < imageSize; i++){
    printf("%.5f, ", greenLightArray[i]); 
  }
  printf("\n\n");

  printf("\n\n\n blueLightArray :\t");
  for(int i = 0; i < imageSize; i++){
    printf("%.5f, ", blueLightArray[i]); 
  }
  printf("\n\n");

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
      deviceerodedShadowMask,deviceErodedSumShadowArray, int imageWidth, int imageHeight);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  } 
  // Launch sum_up_arrays kernel on the eroded light array
  {
    dim3 blockDim1(8,8), gridDim1(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
      deviceerodedLightMask,deviceErodedSumLightArray, int imageWidth, int imageHeight);
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

  printf("\n\n\n redSumShadowArray :\t");
  printf("%.5f, ", redSumShadowArray); 
  printf("\n\n");

  printf("\n\n\n greenSumShadowArray :\t");
  printf("%.5f, ", greenSumShadowArray); 
  printf("\n\n");

  printf("\n\n\n blueSumShadowArray :\t");
  printf("%.5f, ", blueSumShadowArray); 
  printf("\n\n");

  printf("\n\n\n redSumLightArray :\t");
  printf("%.5f, ", redSumLightArray); 
  printf("\n\n");

  printf("\n\n\n greenSumLightArray :\t");
  printf("%.5f, ", greenSumLightArray); 
  printf("\n\n");

  printf("\n\n\n blueSumLightArray :\t");
  printf("%.5f, ", blueSumLightArray); 
  printf("\n\n");

  printf("\n\n\n erodedSumShadowArray :\t");
  printf("%.5f, ", erodedSumShadowArray); 
  printf("\n\n");

  printf("\n\n\n erodedSumLightArray :\t");
  printf("%.5f, ", erodedSumLightArray); 
  printf("\n\n");

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

  printf("\n\n\n finalImage :\t");
  for(int i = 0; i < finalImage; i++){
    printf("%.5f, ", finalImage[i]); 
  }
  printf("\n\n");

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
  CUDA_CHECK(cudaFree(deviceerodedShadowMask));
  CUDA_CHECK(cudaFree(deviceerodedLightMask));
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

  return 0;

}