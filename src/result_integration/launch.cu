#include <wb.h>
#include "kernel.cu"
#include "../globals.h"


void launch_result_integration(float *rgbImage,unsigned char *erodedShadowMask,unsigned char *erodedLightMask, 
  float *smoothMask,unsigned char *finalImage,int imageWidth, int imageHeight) {
  
  unsigned char *redShadowArray;
  unsigned char *greenShadowArray;
  unsigned char *blueShadowArray;
  unsigned char *redLightArray;
  unsigned char *greenLightArray;
  unsigned char *blueLightArray;
  unsigned char *redSumShadowArray;
  unsigned char *greenSumShadowArray;
  unsigned char *blueSumShadowArray;
  unsigned char *redSumLightArray; 
  unsigned char *greenSumLightArray;
  unsigned char *blueSumLightArray;
  unsigned char *erodedSumShadowArray;
  unsigned char *erodedSumLightArray;
  float *deviceRgbImage;
  unsigned char *deviceRedShadowArray;
  unsigned char *deviceGreenShadowArray;
  unsigned char *deviceBlueShadowArray;
  unsigned char *deviceRedLightArray;
  unsigned char *deviceGreenLightArray;
  unsigned char *deviceBlueLightArray;
  unsigned char *deviceErodedShadowMask;
  unsigned char *deviceErodedLightMask;
  unsigned char *deviceRedSumShadowArray;
  unsigned char *deviceGreenSumShadowArray;
  unsigned char *deviceBlueSumShadowArray;
  unsigned char *deviceRedSumLightArray; 
  unsigned char *deviceGreenSumLightArray;
  unsigned char *deviceBlueSumLightArray;
  unsigned char *deviceErodedSumShadowArray;
  unsigned char *deviceErodedSumLightArray;
  unsigned char *deviceRedSumShadowArray_interm;
  unsigned char *deviceGreenSumShadowArray_interm;
  unsigned char *deviceBlueSumShadowArray_interm;
  unsigned char *deviceRedSumLightArray_interm; 
  unsigned char *deviceGreenSumLightArray_interm;
  unsigned char *deviceBlueSumLightArray_interm;
  unsigned char *deviceErodedSumShadowArray_interm;
  unsigned char *deviceErodedSumLightArray_interm;
  float *deviceSmoothMask;
  unsigned char *deviceFinalImage;

  int imageSize = imageHeight * imageWidth;
  int n_threads = 16;

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceRgbImage, imageSize * sizeof(float)));
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedShadowMask, imageSize * sizeof(unsigned char)));
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedLightMask, imageSize * sizeof(unsigned char)));
  CUDA_CHECK( cudaMalloc((void **)&deviceRedShadowArray, imageSize * sizeof(unsigned char)));
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenShadowArray, imageSize * sizeof(unsigned char)));
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueShadowArray, imageSize * sizeof(unsigned char)));
  CUDA_CHECK( cudaMalloc((void **)&deviceRedLightArray, imageSize * sizeof(unsigned char)));
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenLightArray, imageSize * sizeof(unsigned char)));
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueLightArray, imageSize * sizeof(unsigned char)));     
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  // Copy memory to the GPU here
  wbTime_start(GPU, "Copying input memory to the GPU.");
  CUDA_CHECK(cudaMemcpy(deviceRgbImage, rgbImage, imageSize * sizeof(float),
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
    multiply_rgbImage_byMask<<<gridDim, blockDim>>>(
      deviceRgbImage, deviceErodedShadowMask, 
      deviceErodedLightMask, deviceRedShadowArray,deviceGreenShadowArray,deviceBlueShadowArray,
      deviceRedLightArray,deviceGreenLightArray,deviceBlueLightArray,imageWidth,imageHeight, NUM_CHANNELS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  

  redShadowArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  blueShadowArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  greenShadowArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  redLightArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  greenLightArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  blueLightArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
 // Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(redShadowArray, deviceRedShadowArray,
                        imageSize * sizeof(unsigned char ),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(greenShadowArray, deviceGreenShadowArray,
                        imageSize * sizeof(unsigned char ),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(blueShadowArray, deviceBlueShadowArray,
                        imageSize * sizeof(unsigned char ),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(redLightArray, deviceRedLightArray,
                        imageSize * sizeof(unsigned char ),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(greenLightArray, deviceGreenLightArray,
                        imageSize * sizeof(unsigned char ),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(blueLightArray, deviceBlueLightArray,
                        imageSize * sizeof(unsigned char ),
                        cudaMemcpyDeviceToHost));                       
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");


  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceRedSumShadowArray, imageSize * sizeof(unsigned char)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceRedSumShadowArray_interm, imageSize * sizeof(unsigned char)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumShadowArray, imageSize * sizeof(unsigned char)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumShadowArray_interm, imageSize * sizeof(unsigned char))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumShadowArray, imageSize * sizeof(unsigned char)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumShadowArray_interm, imageSize * sizeof(unsigned char)));  
  CUDA_CHECK( cudaMalloc((void **)&deviceRedSumLightArray, imageSize * sizeof(unsigned char)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceRedSumLightArray_interm, imageSize * sizeof(unsigned char)));  
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumLightArray, imageSize * sizeof(unsigned char)));
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumLightArray_interm, imageSize * sizeof(unsigned char)));
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumLightArray, imageSize * sizeof(unsigned char))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumLightArray_interm, imageSize * sizeof(unsigned char))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumLightArray, imageSize * sizeof(unsigned char))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumLightArray_interm, imageSize * sizeof(unsigned char))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumShadowArray, imageSize * sizeof(unsigned char)));  
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumShadowArray_interm, imageSize * sizeof(unsigned char)));       
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
// Launch sum_up_arrays kernel on the light and shadow arrays for each channel

  const int maxThreadsPerBlock = n_threads;
  int threads = maxThreadsPerBlock;
  int blocks = imageSize / maxThreadsPerBlock;
  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
      deviceRedShadowArray,deviceRedSumShadowArray_interm,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;
  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
  deviceRedSumShadowArray_interm,deviceRedSumShadowArray,imageSize);  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

 // Launch sum_up_arrays kernel on the light and shadow arrays for each channel

  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
      deviceGreenShadowArray,deviceGreenSumShadowArray_interm,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());    
  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;
  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
  deviceGreenSumShadowArray_interm,deviceGreenSumShadowArray,imageSize); 
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

 // Launch sum_up_arrays kernel on the shadow arrays for each channel
 
  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
      deviceBlueShadowArray,deviceBlueSumShadowArray_interm,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());    
  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;
  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
  deviceBlueSumShadowArray_interm,deviceBlueSumShadowArray,imageSize); 
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

 // Launch sum_up_arrays kernel on the light arrays for each channel
 
  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
      deviceRedLightArray,deviceRedSumLightArray_interm,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());    
  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;
  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
  deviceRedSumLightArray_interm,deviceRedSumLightArray,imageSize); 
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());


  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
      deviceGreenLightArray,deviceGreenSumLightArray_interm,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());    
  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;
  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
  deviceGreenSumLightArray_interm,deviceGreenSumLightArray,imageSize); 
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());


  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
      deviceBlueLightArray,deviceBlueSumLightArray_interm,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());    
  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;
  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
  deviceBlueSumLightArray_interm,deviceBlueSumLightArray,imageSize); 
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

// Launch sum_up_arrays kernel on the eroded shadow array
  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
      deviceErodedShadowMask,deviceErodedSumShadowArray_interm,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;
  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
  deviceErodedSumShadowArray_interm,deviceErodedSumShadowArray,imageSize); 
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

// Launch sum_up_arrays kernel on the eroded light array

  //dim3 blockDim(8,8), gridDim(1,1);

  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
    deviceErodedLightMask,deviceErodedSumLightArray_interm,imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());  
  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;
  sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(unsigned char)>>>(
  deviceErodedSumLightArray_interm,deviceErodedSumLightArray,imageSize); 
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  redSumShadowArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  greenSumShadowArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  blueSumShadowArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  redSumLightArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  greenSumLightArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  blueSumLightArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  erodedSumShadowArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  erodedSumLightArray = (unsigned char *)malloc(imageSize * sizeof(unsigned char));

  // Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(redSumShadowArray, deviceRedSumShadowArray,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(greenSumShadowArray, deviceGreenSumShadowArray,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(blueSumShadowArray, deviceBlueSumShadowArray,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(redSumLightArray, deviceRedSumLightArray,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(greenSumLightArray, deviceGreenSumLightArray,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(blueSumLightArray, deviceBlueSumLightArray,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));  
  CUDA_CHECK(cudaMemcpy(erodedSumShadowArray, deviceErodedSumShadowArray,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));     
  CUDA_CHECK(cudaMemcpy(erodedSumLightArray, deviceErodedSumLightArray,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));                
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");


  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceFinalImage, imageSize * sizeof(unsigned char))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceSmoothMask, imageSize * sizeof(float)));      
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  // zero out bins
    CUDA_CHECK(cudaMemset(deviceFinalImage, 0.0, imageSize * sizeof(unsigned char)));
  // Launch calculate_rgb_ratio kernel on the eroded shadow array and calculates the final image
  
    //dim3 blockDim(8,8), gridDim(1,1);
    calculate_final_image<<<gridDim, blockDim>>>(
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
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  //@@ Free the GPU memory here
  wbTime_start(GPU, "Freeing GPU Memory");

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
  CUDA_CHECK(cudaFree(deviceRedSumShadowArray_interm));
  CUDA_CHECK(cudaFree(deviceGreenSumShadowArray));
  CUDA_CHECK(cudaFree(deviceGreenSumShadowArray_interm));
  CUDA_CHECK(cudaFree(deviceBlueSumShadowArray));
  CUDA_CHECK(cudaFree(deviceBlueSumShadowArray_interm));
  CUDA_CHECK(cudaFree(deviceRedSumLightArray)); 
  CUDA_CHECK(cudaFree(deviceRedSumLightArray_interm)); 
  CUDA_CHECK(cudaFree(deviceGreenSumLightArray));
  CUDA_CHECK(cudaFree(deviceGreenSumLightArray_interm));
  CUDA_CHECK(cudaFree(deviceBlueSumLightArray));
  CUDA_CHECK(cudaFree(deviceBlueSumLightArray_interm));
  CUDA_CHECK(cudaFree(deviceErodedSumShadowArray));
  CUDA_CHECK(cudaFree(deviceErodedSumShadowArray_interm));
  CUDA_CHECK(cudaFree(deviceErodedSumLightArray));
  CUDA_CHECK(cudaFree(deviceErodedSumLightArray_interm));
  CUDA_CHECK(cudaFree(deviceSmoothMask));
  CUDA_CHECK(cudaFree(deviceFinalImage));
  wbTime_stop(GPU, "Freeing GPU Memory");

}