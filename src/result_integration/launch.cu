#include <wb.h>
#include "kernel.cu"
#include "kernel_reduction.cu"
#include "../globals.h"


void launch_result_integration(float *rgbImage,unsigned char *erodedShadowMask,unsigned char *erodedLightMask, 
  float *smoothMask,float *finalImage,int imageWidth, int imageHeight) {
  
  float *redShadowArray;
  float redSumShadowArray;
  float greenSumShadowArray;
  float blueSumShadowArray;
  float redSumLightArray; 
  float greenSumLightArray;
  float blueSumLightArray;
  float erodedSumShadowArray;
  float erodedSumLightArray;
  float *deviceRgbImage;
  float *deviceRedShadowArray;
  float *deviceGreenShadowArray;
  float *deviceBlueShadowArray;
  float *deviceRedLightArray;
  float *deviceGreenLightArray;
  float *deviceBlueLightArray;
  float *deviceErodedShadowMask;
  float *deviceErodedLightMask;
  float *deviceSmoothMask;
  float *deviceFinalImage;

  int imageSize = imageHeight * imageWidth;
  int n_threads = 16;

  float *erodedShadowMaskF = (float *)malloc(imageSize * sizeof(float));
  float *erodedLightMaskF = (float *)malloc(imageSize * sizeof(float));
  for(int i=0;i<imageSize;i++){

    erodedShadowMaskF[i] = (float)erodedShadowMask[i];
    erodedLightMaskF[i] = (float)erodedLightMask[i];
  }

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK(cudaMalloc((void **)&deviceRgbImage, imageSize * NUM_CHANNELS * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&deviceSmoothMask, imageSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&deviceErodedShadowMask, imageSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&deviceErodedLightMask, imageSize * sizeof(float)));
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
  CUDA_CHECK(cudaMemcpy(deviceRgbImage, rgbImage, imageSize * NUM_CHANNELS * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceSmoothMask, smoothMask, imageSize * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceErodedShadowMask, erodedShadowMaskF, imageSize * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceErodedLightMask, erodedLightMaskF, imageSize * sizeof(float),
                        cudaMemcpyHostToDevice));    
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  
  // Launch multiple_rgbImage_byMask kernel on the bins
  dim3 gridDim(ceil((float)imageWidth/(float)n_threads),ceil((float)imageHeight/(float)n_threads));
  dim3 blockDim(n_threads,n_threads);
  timerLog_startEvent(&timerLog);
  multiply_rgbImage_byMask<<<gridDim, blockDim>>>(
    deviceRgbImage, deviceErodedShadowMask, 
    deviceErodedLightMask, deviceRedShadowArray,deviceGreenShadowArray,deviceBlueShadowArray,
    deviceRedLightArray,deviceGreenLightArray,deviceBlueLightArray,imageWidth,imageHeight, NUM_CHANNELS);
  timerLog_stopEventAndLog(&timerLog, "R.Integration Kernel 1", "\0", imageWidth, imageHeight);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());


  // Launch gpu_sum_reduce kernel on the light and shadow arrays for each channel
  timerLog_startEvent(&timerLog);
  redSumShadowArray = gpu_sum_reduce(deviceRedShadowArray, imageSize);

  greenSumShadowArray = gpu_sum_reduce(deviceGreenShadowArray, imageSize);

  // Launch gpu_sum_reduce kernel on the light and shadow arrays for each channel
  blueSumShadowArray = gpu_sum_reduce(deviceBlueShadowArray, imageSize);

  // Launch gpu_sum_reduce kernel on the shadow arrays for each channel
  redSumLightArray = gpu_sum_reduce(deviceRedLightArray, imageSize); 

  // Launch gpu_sum_reduce kernel on the light arrays for each channel
  greenSumLightArray = gpu_sum_reduce(deviceGreenLightArray, imageSize);

  blueSumLightArray = gpu_sum_reduce(deviceBlueLightArray, imageSize);   

  // Launch gpu_sum_reduce kernel on the eroded shadow array
  erodedSumShadowArray = gpu_sum_reduce(deviceErodedShadowMask, imageSize);

  // Launch gpu_sum_reduce kernel on the eroded light array
  erodedSumLightArray = gpu_sum_reduce(deviceErodedLightMask, imageSize);  

  timerLog_stopEventAndLog(&timerLog, "R.Integration Kernel 2", "\0", imageWidth, imageHeight); 

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize()); 

  float shadowAverageRed=redSumShadowArray/erodedSumShadowArray;
  float shadowAverageGreen=greenSumShadowArray/erodedSumShadowArray;
  float shadowAverageBlue=blueSumShadowArray/erodedSumShadowArray;
  float lightAverageRed=redSumLightArray/erodedSumLightArray;
  float lightAverageGreen=greenSumLightArray/erodedSumLightArray;
  float lightAverageBlue=blueSumLightArray/erodedSumLightArray;

  float redRatio = lightAverageRed/shadowAverageRed -1;
  float greenRatio = lightAverageGreen/shadowAverageGreen -1;
  float blueRatio = lightAverageBlue/shadowAverageBlue -1;
  
  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceFinalImage, imageSize * NUM_CHANNELS * sizeof(float))); 
      CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
  wbTime_stop(GPU, "Allocating GPU memory.");
  
  //zero out bins
  CUDA_CHECK(cudaMemset(deviceFinalImage, 0.0, imageSize * NUM_CHANNELS * sizeof(float)));

  // Launch calculate_rgb_ratio kernel on the eroded shadow array and calculates the final image
  dim3 gridDim2(ceil((float)imageWidth/16.0), ceil((float)imageHeight/16.0));
  dim3 blockDim2(16, 16);

  timerLog_startEvent(&timerLog);
  calculate_final_image_optimised<<<gridDim2, blockDim2>>>(redRatio, greenRatio,blueRatio,
  deviceRgbImage, deviceSmoothMask, deviceFinalImage,
  imageWidth, imageHeight, NUM_CHANNELS);
  timerLog_stopEventAndLog(&timerLog, "R.Integration Kernel 3", "\0", imageWidth, imageHeight); 

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(finalImage, deviceFinalImage,
                        imageSize * NUM_CHANNELS * sizeof(float),
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
  CUDA_CHECK(cudaFree(deviceSmoothMask));
  CUDA_CHECK(cudaFree(deviceFinalImage));
  wbTime_stop(Copy, "Freeing GPU Memory");

}

