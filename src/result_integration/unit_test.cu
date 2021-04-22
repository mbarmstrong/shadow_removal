
#include <wb.h>
#include "kernel.cu"
#include "../globals.h"

void unit_test( float *rgbImage,unsigned char *erodedShadowMask,unsigned char *erodedLightMask, float *smoothMask,int imageWidth, int imageHeight) {

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
  float *deviceRedSumShadowArray_interm;
  float *deviceGreenSumShadowArray_interm;
  float *deviceBlueSumShadowArray_interm;
  float *deviceRedSumLightArray_interm; 
  float *deviceGreenSumLightArray_interm;
  float *deviceBlueSumLightArray_interm;
  float *deviceErodedSumShadowArray_interm;
  float *deviceErodedSumLightArray_interm;
  float *deviceSmoothMask;
  float *deviceFinalImage;

  float *finalImage;

  int imageSize = imageHeight * imageWidth;

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceRgbImage, imageSize * sizeof(float)));
  CUDA_CHECK( cudaMalloc((void **)&deviceSmoothMask, imageSize * sizeof(float)));
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedShadowMask, imageSize * sizeof(unsigned char)));
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedLightMask, imageSize * sizeof(unsigned char)));
  CUDA_CHECK( cudaMalloc((void **)&deviceRedShadowArray, imageSize * sizeof(float)));
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenShadowArray, imageSize * sizeof(float)));
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueShadowArray, imageSize * sizeof(float)));
  CUDA_CHECK( cudaMalloc((void **)&deviceRedLightArray, imageSize * sizeof(float)));
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenLightArray, imageSize * sizeof(float)));
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueLightArray, imageSize * sizeof(float)));     
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

  printf("\nGray Shadow Mask:\t");
  print_image(erodedShadowMask,imageWidth,imageHeight);

  printf("\nGray Light Mask:\t");
  print_image(erodedLightMask,imageWidth,imageHeight);
  
  // Launch multiple_rgbImage_byMask kernel on the bins
  {
    dim3 blockDim(8,8), gridDim(1,1);
    multiply_rgbImage_byMask<<<gridDim, blockDim>>>(
      deviceRgbImage, deviceErodedShadowMask, 
      deviceErodedLightMask, deviceRedShadowArray,deviceGreenShadowArray,deviceBlueShadowArray,
      deviceRedLightArray,deviceGreenLightArray,deviceBlueLightArray,imageWidth,imageHeight, NUM_CHANNELS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

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

  printf("\nRed Shadow Array:\t");
  print_image(redShadowArray,imageWidth,imageHeight);
  printf("\nGreen Shadow Array:\t");
  print_image(greenShadowArray,imageWidth,imageHeight);
  printf("\nBlue Shadow Array:\t");
  print_image(blueShadowArray,imageWidth,imageHeight);
  printf("\nRed Light Array:\t");
  print_image(redLightArray,imageWidth,imageHeight);
  printf("\nGreen Light Array:\t");
  print_image(greenLightArray,imageWidth,imageHeight);
  printf("\nBlue Light Array:\t");
  print_image(blueLightArray,imageWidth,imageHeight);


  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceRedSumShadowArray, imageSize * sizeof(float)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceRedSumShadowArray_interm, imageSize * sizeof(float)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumShadowArray, imageSize * sizeof(float)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumShadowArray_interm, imageSize * sizeof(float))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumShadowArray, imageSize * sizeof(float)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumShadowArray_interm, imageSize * sizeof(float)));  
  CUDA_CHECK( cudaMalloc((void **)&deviceRedSumLightArray, imageSize * sizeof(float)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceRedSumLightArray_interm, imageSize * sizeof(float)));  
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumLightArray, imageSize * sizeof(float)));
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumLightArray_interm, imageSize * sizeof(float)));
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumLightArray, imageSize * sizeof(float))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumLightArray_interm, imageSize * sizeof(float))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumLightArray, imageSize * sizeof(float))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumLightArray_interm, imageSize * sizeof(float))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumShadowArray, imageSize * sizeof(float)));  
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumShadowArray_interm, imageSize * sizeof(float)));        
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
    const int maxThreadsPerBlock = 16;
    int threads = maxThreadsPerBlock;
    int blocks = imageSize / maxThreadsPerBlock;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
        deviceRedShadowArray,deviceRedSumShadowArray_interm,imageSize);
        CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
    deviceRedSumShadowArray_interm,deviceRedSumShadowArray,imageSize);  
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
   // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
    threads = maxThreadsPerBlock;
    blocks = imageSize / maxThreadsPerBlock;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
        deviceGreenShadowArray,deviceGreenSumShadowArray_interm,imageSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());    
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
    deviceGreenSumShadowArray_interm,deviceGreenSumShadowArray,imageSize); 
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

   // Launch sum_up_arrays kernel on the shadow arrays for each channel
    threads = maxThreadsPerBlock;
    blocks = imageSize / maxThreadsPerBlock;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
        deviceBlueShadowArray,deviceBlueSumShadowArray_interm,imageSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());    
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
    deviceBlueSumShadowArray_interm,deviceBlueSumShadowArray,imageSize); 
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

   // Launch sum_up_arrays kernel on the light arrays for each channel
    threads = maxThreadsPerBlock;
    blocks = imageSize / maxThreadsPerBlock;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
        deviceRedLightArray,deviceRedSumLightArray_interm,imageSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());    
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
    deviceRedSumLightArray_interm,deviceRedSumLightArray,imageSize); 
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  
    threads = maxThreadsPerBlock;
    blocks = imageSize / maxThreadsPerBlock;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
        deviceGreenLightArray,deviceGreenSumLightArray_interm,imageSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());    
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
    deviceGreenSumLightArray_interm,deviceGreenSumLightArray,imageSize); 
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

    threads = maxThreadsPerBlock;
    blocks = imageSize / maxThreadsPerBlock;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
        deviceBlueLightArray,deviceBlueSumLightArray_interm,imageSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());    
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
    deviceBlueSumLightArray_interm,deviceBlueSumLightArray,imageSize); 
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

  // Launch sum_up_arrays kernel on the eroded shadow array
    threads = maxThreadsPerBlock;
    blocks = imageSize / maxThreadsPerBlock;
    sum_up_arrays_by_reduction1<<<blocks, threads, threads * sizeof(unsigned char)>>>(
        deviceErodedShadowMask,deviceErodedSumShadowArray_interm,imageSize);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
    deviceErodedSumShadowArray_interm,deviceErodedSumShadowArray,imageSize); 
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  
  // Launch sum_up_arrays kernel on the eroded light array
    threads = maxThreadsPerBlock;
    blocks = imageSize / maxThreadsPerBlock;
    sum_up_arrays_by_reduction1<<<blocks, threads, threads * sizeof(unsigned char)>>>(
      deviceErodedLightMask,deviceErodedSumLightArray_interm,imageSize);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());  
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    sum_up_arrays_by_reduction<<<blocks, threads, threads * sizeof(float)>>>(
    deviceErodedSumLightArray_interm,deviceErodedSumLightArray,imageSize); 
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

  redSumShadowArray = (float *)malloc(imageSize * sizeof(float));
  greenSumShadowArray = (float *)malloc(imageSize * sizeof(float));
  blueSumShadowArray = (float *)malloc(imageSize * sizeof(float));
  redSumLightArray = (float *)malloc(imageSize * sizeof(float));
  greenSumLightArray = (float *)malloc(imageSize * sizeof(float));
  blueSumLightArray = (float *)malloc(imageSize * sizeof(float));
  erodedSumShadowArray = (float*)malloc(imageSize * sizeof(float));
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

  printf("\nSum of Red Shadow Array:\t");
  print_image(redSumShadowArray,imageWidth,imageHeight);
  printf("\nSum of Green Shadow Array:\t");
  print_image(greenSumShadowArray,imageWidth,imageHeight);
  printf("\nSum of Blue Shadow Array:\t");
  print_image(blueSumShadowArray,imageWidth,imageHeight);
  printf("\nSum of Red Light Array:\t");
  print_image(redSumLightArray,imageWidth,imageHeight);
  printf("\nSum of Green Light Array:\t");
  print_image(greenSumLightArray,imageWidth,imageHeight);
  printf("\nSum of Blue Light Array:\t");
  print_image(blueSumLightArray,imageWidth,imageHeight);
  printf("\nSum of Eroded  Shadow Array:\t");
  print_image(erodedSumShadowArray,imageWidth,imageHeight);
  printf("\nSum of Eroded  Light Array:\t");
  print_image(erodedSumLightArray,imageWidth,imageHeight);

  finalImage = (float *)malloc(imageSize * sizeof(float));

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceFinalImage, imageSize * sizeof(float)));  
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  // zero out bins
    CUDA_CHECK(cudaMemset(deviceFinalImage, 0.0, imageSize * sizeof(float)));
  // Launch calculate_rgb_ratio kernel on the eroded shadow array and calculates the final image
  {
    dim3 blockDim(8,8), gridDim(1,1);
    calculate_final_image<<<gridDim, blockDim>>>(
    deviceRedSumShadowArray, deviceGreenSumShadowArray,deviceBlueSumShadowArray,
    deviceRedSumLightArray, deviceGreenSumLightArray,deviceBlueSumLightArray,
    deviceErodedSumShadowArray,deviceErodedSumLightArray,
    deviceRgbImage, deviceSmoothMask, deviceFinalImage,
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

  printf("\nFinal image (shadowless):\n");
  print_image(finalImage,imageWidth,imageHeight);

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

  free(finalImage);
  free(redShadowArray);
  free(greenShadowArray);
  free(blueShadowArray);
  free(redLightArray);
  free(greenLightArray);
  free(blueLightArray);
  free(redSumShadowArray);
  free(greenSumShadowArray);
  free(blueSumShadowArray);
  free(redSumLightArray); 
  free(greenSumLightArray);
  free(blueSumLightArray);
  free(erodedSumShadowArray);
  free(erodedSumLightArray);
  free(smoothMask);
}

int main(int argc, char *argv[]) {
  
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageSize;

  char *inputImageFile;

  wbImage_t inputImage_RGB;

  float* inputImage_RGB_float;

  args = wbArg_read(argc, argv); // parse the input arguments

  inputImageFile = wbArg_getInputFile(args, 0);
  inputImage_RGB = wbImport(inputImageFile);

  imageWidth = wbImage_getWidth(inputImage_RGB);
  imageHeight = wbImage_getHeight(inputImage_RGB);

  printf("\nRunning Result Integration unit test on image of %dx%d\n",
           imageWidth, imageHeight, NUM_CHANNELS);

  unsigned char erodedShadow[16] = {1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1};
  unsigned char erodedLight[16] =  {1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1};
  float smoothMask[16] =  {0.4444,0.6667,0.6667,0.4444,
                           0.6667,1.0000,1.0000,0.6667,
                           0.6667,1.0000,1.0000,0.6667,
                           0.4444,0.6667,0.6667,0.444};

  inputImage_RGB_float  = wbImage_getData(inputImage_RGB);

  print_image(inputImage_RGB_float,imageWidth,imageHeight);

  unit_test(inputImage_RGB_float,erodedShadow,erodedLight,smoothMask,imageWidth, imageHeight);

  free(inputImage_RGB_float);
  wbImage_delete(inputImage_RGB);

  return 0;

}