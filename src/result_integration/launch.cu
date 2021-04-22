#include <wb.h>
#include "kernel.cu"
#include "../globals.h"


void launch_result_integration(float *rgbImage,unsigned char *erodedShadowMask,unsigned char *erodedLightMask, 
  float *smoothMask,float *finalImage,int imageWidth, int imageHeight) {
  
    float *redShadowArray;
    float *greenShadowArray;
    float *blueShadowArray;
    float *redLightArray;
    float *greenLightArray;
    float *blueLightArray;
    float *redSumShadowArray;
    float *greenSumShadowArray;
    float *blueSumShadowArray;
    float *redSumLightArray; 
    float *greenSumLightArray;
    float *blueSumLightArray;
    float *erodedSumShadowArray;
    float *erodedSumLightArray;
    float *deviceRgbImage;
    float *deviceRedShadowArray;
    float *deviceGreenShadowArray;
    float *deviceBlueShadowArray;
    float *deviceRedLightArray;
    float *deviceGreenLightArray;
    float *deviceBlueLightArray;
    unsigned char *deviceErodedShadowMask;
    unsigned char *deviceErodedLightMask;
    float *deviceRedSumShadowArray;
    float *deviceGreenSumShadowArray;
    float *deviceBlueSumShadowArray;
    float *deviceRedSumLightArray; 
    float *deviceGreenSumLightArray;
    float *deviceBlueSumLightArray;
    float *deviceErodedSumShadowArray;
    float *deviceErodedSumLightArray;
    float *deviceSmoothMask;
    float *deviceFinalImage;
  
    int imageSize = imageHeight * imageWidth;
    int n_threads = 16;
  
    wbTime_start(GPU, "Allocating GPU memory.");
    CUDA_CHECK(cudaMalloc((void **)&deviceRgbImage, imageSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&deviceSmoothMask, imageSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&deviceErodedShadowMask, imageSize * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc((void **)&deviceErodedLightMask, imageSize * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc((void **)&deviceRedShadowArray, imageSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&deviceGreenShadowArray, imageSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&deviceBlueShadowArray, imageSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&deviceRedLightArray, imageSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&deviceGreenLightArray, imageSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&deviceBlueLightArray, imageSize * sizeof(float)));     
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(GPU, "Allocating GPU memory.");
  
    // Copy memory to the GPU here
    wbTime_start(GPU, "Copying input memory to the GPU.");
    CUDA_CHECK(cudaMemcpy(deviceRgbImage, rgbImage, imageSize * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceSmoothMask, smoothMask, imageSize * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceErodedShadowMask, erodedShadowMask, imageSize * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceErodedLightMask, erodedLightMask, imageSize * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));    
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(GPU, "Copying input memory to the GPU.");
  
  // Launch multiple_rgbImage_byMask kernel on the bins
  
    dim3 gridDim(ceil((float)imageWidth/(float)n_threads),ceil((float)imageHeight/(float)n_threads),1);
    dim3 blockDim(n_threads,n_threads,1);
    multiply_rgbImage_byMask<<<gridDim, blockDim>>>(
      deviceRgbImage, deviceErodedShadowMask, 
      deviceErodedLightMask, deviceRedShadowArray,deviceGreenShadowArray,deviceBlueShadowArray,
      deviceRedLightArray,deviceGreenLightArray,deviceBlueLightArray,imageWidth,imageHeight, NUM_CHANNELS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  

    redShadowArray = (float *)malloc(imageSize * sizeof(float));
    blueShadowArray = (float *)malloc(imageSize * sizeof(float));
    greenShadowArray = (float *)malloc(imageSize * sizeof(float));
    redLightArray = (float *)malloc(imageSize * sizeof(float));
    greenLightArray = (float *)malloc(imageSize * sizeof(float));
    blueLightArray = (float *)malloc(imageSize * sizeof(float));
   // Copy the GPU memory back to the CPU here
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


    wbTime_start(GPU, "Allocating GPU memory.");
    CUDA_CHECK( cudaMalloc((void **)&deviceRedSumShadowArray, imageSize * sizeof(float)));   
    CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumShadowArray, imageSize * sizeof(float)));   
    CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumShadowArray, imageSize * sizeof(float)));   
    CUDA_CHECK( cudaMalloc((void **)&deviceRedSumLightArray, imageSize * sizeof(float)));   
    CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumLightArray, imageSize * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumLightArray, imageSize * sizeof(float))); 
    CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumLightArray, imageSize * sizeof(float))); 
    CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumShadowArray, imageSize * sizeof(float)));         
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(GPU, "Allocating GPU memory.");

 // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
  dim3 gridDim2(ceil((float)imageWidth/(float)n_threads),ceil((float)imageHeight/(float)n_threads),1);
  dim3 blockDim2(16,16,1);
  sum_up_arrays_by_reduction<<<gridDim2, blockDim2, 256 * sizeof(float)>>>(
      deviceRedShadowArray,deviceRedSumShadowArray,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

 // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
  sum_up_arrays_by_reduction<<<gridDim2, blockDim2, 256 * sizeof(float)>>>(
      deviceGreenShadowArray,deviceGreenSumShadowArray,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());    

 // Launch sum_up_arrays kernel on the shadow arrays for each channel
  sum_up_arrays_by_reduction<<<gridDim2, blockDim2, 256 * sizeof(float)>>>(
      deviceBlueShadowArray,deviceBlueSumShadowArray,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());    

 // Launch sum_up_arrays kernel on the light arrays for each channel
  sum_up_arrays_by_reduction<<<gridDim2, blockDim2, 256 * sizeof(float)>>>(
      deviceRedLightArray,deviceRedSumLightArray,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());    

  sum_up_arrays_by_reduction<<<gridDim2, blockDim2, 256 * sizeof(float)>>>(
      deviceGreenLightArray,deviceGreenSumLightArray,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());    

  sum_up_arrays_by_reduction<<<gridDim2, blockDim2, 256 * sizeof(float)>>>(
      deviceBlueLightArray,deviceBlueSumLightArray,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());    

// Launch sum_up_arrays kernel on the eroded shadow array
  sum_up_arrays_by_reduction1<<<gridDim2, blockDim2, 256 * sizeof(float)>>>(
      deviceErodedShadowMask,deviceErodedSumShadowArray,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

// Launch sum_up_arrays kernel on the eroded light array
  sum_up_arrays_by_reduction1<<<gridDim2, blockDim2, 256 * sizeof(float)>>>(
    deviceErodedLightMask,deviceErodedSumLightArray,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());  

redSumShadowArray = (float *)malloc(imageSize * sizeof(float));
greenSumShadowArray = (float *)malloc(imageSize * sizeof(float));
blueSumShadowArray = (float *)malloc(imageSize * sizeof(float));
redSumLightArray = (float *)malloc(imageSize * sizeof(float));
greenSumLightArray = (float *)malloc(imageSize * sizeof(float));
blueSumLightArray = (float *)malloc(imageSize * sizeof(float));
erodedSumShadowArray = (float *)malloc(imageSize * sizeof(float));
erodedSumLightArray = (float *)malloc(imageSize * sizeof(float));

// Copy the GPU memory back to the CPU here
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


  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceFinalImage, imageSize * sizeof(float))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceSmoothMask, imageSize * sizeof(float)));      
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");


  finalImage = (float *)malloc(imageSize * sizeof(float));
  // zero out bins
  CUDA_CHECK(cudaMemset(deviceFinalImage, 0.0, imageSize * sizeof(float)));
  CUDA_CHECK(cudaGetLastError());
  // Launch calculate_rgb_ratio kernel on the eroded shadow array and calculates the final image
    calculate_final_image<<<gridDim2, blockDim2>>>(
    deviceRedSumShadowArray, deviceGreenSumShadowArray,deviceBlueSumShadowArray,
    deviceRedSumLightArray, deviceGreenSumLightArray,deviceBlueSumLightArray,
    deviceErodedSumShadowArray,deviceErodedSumLightArray,
    deviceRgbImage, deviceSmoothMask, deviceFinalImage,
    imageWidth, imageHeight, NUM_CHANNELS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
   

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(finalImage, deviceFinalImage,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // //@@ Free the GPU memory here
  wbTime_start(Copy, "Freeing GPU Memory");
  CUDA_CHECK(cudaFree(deviceRgbImage));
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
  CUDA_CHECK(cudaFree(deviceErodedSumLightArray));
  CUDA_CHECK(cudaFree(deviceSmoothMask));
  CUDA_CHECK(cudaFree(deviceFinalImage));
  wbTime_stop(Copy, "Freeing GPU Memory");

}