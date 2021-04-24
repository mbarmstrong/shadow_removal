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
    float deviceRedSumShadowArray;
    float deviceGreenSumShadowArray;
    float deviceBlueSumShadowArray;
    float deviceRedSumLightArray; 
    float deviceGreenSumLightArray;
    float deviceBlueSumLightArray;
    float deviceErodedSumShadowArray;
    float deviceErodedSumLightArray;
    float *deviceSmoothMask;
    float *deviceFinalImage;
  
    int imageSize = imageHeight * imageWidth;
    int n_threads = 32;
  
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

 // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
  dim3 gridDim2(ceil((float)imageWidth/(float)n_threads),ceil((float)imageHeight/(float)n_threads),1);
  dim3 blockDim2(32,32,1);

  redSumShadowArray = gpu_sum_reduce(deviceRedShadowArray, imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  greenSumShadowArray = gpu_sum_reduce(deviceGreenShadowArray, imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

 // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
  blueSumShadowArray = gpu_sum_reduce(deviceBlueShadowArray, imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());    

 // Launch sum_up_arrays kernel on the shadow arrays for each channel
  redSumLightArray = gpu_sum_reduce(deviceRedLightArray, imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());    

 // Launch sum_up_arrays kernel on the light arrays for each channel
  greenSumLightArray = gpu_sum_reduce(deviceGreenLightArray, imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());    

  blueSumLightArray = gpu_sum_reduce(deviceBlueLightArray, imageSize); 
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());      

// Launch sum_up_arrays kernel on the eroded shadow array
  erodedSumShadowArray = gpu_sum_reduce(deviceErodedShadowMask, imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

// Launch sum_up_arrays kernel on the eroded light array
erodedSumLightArray = gpu_sum_reduce(deviceErodedLightMask, imageSize); 
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

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceFinalImage, imageSize * NUM_CHANNELS * sizeof(float))); 
      CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK( cudaMalloc((void **)&deviceSmoothMask, imageSize * sizeof(float)));      
      CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
  wbTime_stop(GPU, "Allocating GPU memory.");

  
  finalImage = (float *)malloc(imageSize * NUM_CHANNELS * sizeof(float));
  // zero out bins
  CUDA_CHECK(cudaMemset(deviceFinalImage, 0.0, imageSize * NUM_CHANNELS * sizeof(float)));
  CUDA_CHECK(cudaGetLastError());
  // Launch calculate_rgb_ratio kernel on the eroded shadow array and calculates the final image
    calculate_final_image<<<gridDim2, blockDim2>>>(
    redSumShadowArray, greenSumShadowArray,blueSumShadowArray,
    redSumLightArray, greenSumLightArray,blueSumLightArray,
    erodedSumShadowArray,erodedSumLightArray,
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
  CUDA_CHECK(cudaFree(deviceSmoothMask));
  CUDA_CHECK(cudaFree(deviceFinalImage));
  wbTime_stop(Copy, "Freeing GPU Memory");

}

