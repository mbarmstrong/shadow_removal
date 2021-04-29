#include <wb.h>
#include "kernel.cu"
#include "kernel_reduction.cu"
#include "../globals.h"


void launch_result_integration(float *rgbImage,unsigned char *erodedShadowMask,unsigned char *erodedLightMask, 
  float *smoothMask,float *finalImage,int imageWidth, int imageHeight) {
  
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
  unsigned char *deviceErodedShadowMask;
  unsigned char *deviceErodedLightMask;
  float *deviceRedRatio;
  float *deviceGreenRatio;
  float *deviceBlueRatio;
  float *deviceSmoothMask;
  float *deviceFinalImage;

  int imageSize = imageHeight * imageWidth;
  int n_threads = 16;


  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK(cudaMalloc((void **)&deviceRgbImage, imageSize * NUM_CHANNELS * sizeof(float)));
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
  CUDA_CHECK(cudaMemcpy(deviceRgbImage, rgbImage, imageSize * NUM_CHANNELS * sizeof(float),
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


  // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
  timerLog_startEvent(&timerLog);
  redSumShadowArray = gpu_sum_reduce(deviceRedShadowArray, imageSize);

  greenSumShadowArray = gpu_sum_reduce(deviceGreenShadowArray, imageSize);

  // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
  blueSumShadowArray = gpu_sum_reduce(deviceBlueShadowArray, imageSize);

  // Launch sum_up_arrays kernel on the shadow arrays for each channel
  redSumLightArray = gpu_sum_reduce(deviceRedLightArray, imageSize); 

  // Launch sum_up_arrays kernel on the light arrays for each channel
  greenSumLightArray = gpu_sum_reduce(deviceGreenLightArray, imageSize);

  blueSumLightArray = gpu_sum_reduce(deviceBlueLightArray, imageSize);   

  // Launch sum_up_arrays kernel on the eroded shadow array
  erodedSumShadowArray = gpu_sum_reduce(deviceErodedShadowMask, imageSize);

  // Launch sum_up_arrays kernel on the eroded light array
  erodedSumLightArray = gpu_sum_reduce(deviceErodedLightMask, imageSize);  

  timerLog_stopEventAndLog(&timerLog, "R.Integration Kernel 2", "\0", imageWidth, imageHeight); 

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize()); 

  if(PRINT_DEBUG){
    printf("\nSum of Red Shadow Array:\t %.04f",redSumShadowArray);
    printf("\nSum of Green Shadow Array:\t%.04f",greenSumShadowArray);
    printf("\nSum of Blue Shadow Array:\t %.04f",blueSumShadowArray);
    printf("\nSum of Red Light Array:\t %.04f",redSumLightArray);
    printf("\nSum of Green Light Array:\t %.04f",greenSumLightArray);
    printf("\nSum of Blue Light Array:\t%.04f",blueSumShadowArray);
    printf("\nSum of Eroded  Shadow Array:\t%.04f",erodedSumShadowArray);
    printf("\nSum of Eroded  Light Array:\t%.04f",erodedSumLightArray);
    printf("\n");
  }

  float shadowAverageRed=redSumShadowArray/erodedSumShadowArray;
  float shadowAverageGreen=greenSumShadowArray/erodedSumShadowArray;
  float shadowAverageBlue=blueSumShadowArray/erodedSumShadowArray;
  float lightAverageRed=redSumLightArray/erodedSumLightArray;
  float lightAverageGreen=greenSumLightArray/erodedSumLightArray;
  float lightAverageBlue=blueSumLightArray/erodedSumLightArray;

  float redRatio = lightAverageRed/shadowAverageRed -1;
  float greenRatio = lightAverageGreen/shadowAverageGreen -1;
  float blueRatio = lightAverageBlue/shadowAverageBlue -1;
  

  if(PRINT_DEBUG){
    printf("\nshadowAverageRed:\t %.04f",shadowAverageRed);
    printf("\nshadowAverageGreen:\t%.04f",shadowAverageGreen);
    printf("\nshadowAverageBlue:\t %.04f",shadowAverageBlue);
    printf("\nlightAverageRed:\t %.04f",lightAverageRed);
    printf("\nlightAverageGreen:\t%.04f",lightAverageGreen);
    printf("\nlightAverageBlue:\t %.04f",lightAverageBlue);
    printf("\n\nRatio Red:\t %.04f",redRatio);
    printf("\nRatio Green:\t%.04f",greenRatio);
    printf("\nRatio Blue:\t %.04f",blueRatio);
    printf("\n");
  }
  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceFinalImage, imageSize * NUM_CHANNELS * sizeof(float))); 
      CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK( cudaMalloc((void **)&deviceSmoothMask, imageSize * sizeof(float)));      
      CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
  wbTime_stop(GPU, "Allocating GPU memory.");


  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceRedRatio, sizeof(float)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenRatio, sizeof(float)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueRatio, sizeof(float)));    
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError()); 
  wbTime_stop(GPU, "Allocating GPU memory."); 

  // Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying host memory to the GPU");
  CUDA_CHECK(cudaMemcpy(deviceRedRatio, &redRatio,
                        sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceGreenRatio, &greenRatio,
                        sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceBlueRatio, &blueRatio,
                        sizeof(float),
                        cudaMemcpyHostToDevice));              
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");
  
  // zero out bins
  //CUDA_CHECK(cudaMemset(deviceFinalImage, 0.0, imageSize * NUM_CHANNELS * sizeof(float)));

  // Launch calculate_rgb_ratio kernel on the eroded shadow array and calculates the final image
  dim3 gridDim2(ceil((float)imageWidth/16.0), ceil((float)imageHeight/16.0));
  dim3 blockDim2(16, 16);

  timerLog_startEvent(&timerLog);
  calculate_final_image_optimised<<<gridDim2, blockDim2>>>(deviceRedRatio, deviceGreenRatio,deviceBlueRatio,
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
  CUDA_CHECK(cudaFree(deviceRedRatio));
  CUDA_CHECK(cudaFree(deviceGreenRatio));
  CUDA_CHECK(cudaFree(deviceBlueRatio));
  CUDA_CHECK(cudaFree(deviceSmoothMask));
  CUDA_CHECK(cudaFree(deviceFinalImage));
  wbTime_stop(Copy, "Freeing GPU Memory");

}

