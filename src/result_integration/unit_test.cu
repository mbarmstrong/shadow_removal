
#include <wb.h>
#include "kernel.cu"
#include "../globals.h"

void unit_test( float *rgbImage,unsigned char *erodedShadowMask,unsigned char *erodedLightMask, float *smoothMask,int imageWidth, int imageHeight) {

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
  float *deviceSmoothMask;
  unsigned char *deviceFinalImage;

  unsigned char *finalImage;

  int imageSize = imageHeight * imageWidth;

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
  {
    dim3 blockDim(8,8), gridDim(1,1);
    multiply_rgbImage_byMask<<<gridDim, blockDim>>>(
      deviceRgbImage, deviceErodedShadowMask, 
      deviceErodedLightMask, deviceRedShadowArray,deviceGreenShadowArray,deviceBlueShadowArray,
      deviceRedLightArray,deviceGreenLightArray,deviceBlueLightArray,imageWidth,imageHeight, NUM_CHANNELS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

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
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumShadowArray, imageSize * sizeof(unsigned char)));    
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumShadowArray, imageSize * sizeof(unsigned char)));    
  CUDA_CHECK( cudaMalloc((void **)&deviceRedSumLightArray, imageSize * sizeof(unsigned char)));    
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumLightArray, imageSize * sizeof(unsigned char)));
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumLightArray, imageSize * sizeof(unsigned char))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumLightArray, imageSize * sizeof(unsigned char))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumShadowArray, imageSize * sizeof(unsigned char)));       
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
  {
    dim3 blockDim(8,8), gridDim(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
        deviceRedShadowArray,deviceRedSumShadowArray,imageSize);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
   // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
   {
    dim3 blockDim(8,8), gridDim(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
        deviceGreenShadowArray,deviceGreenSumShadowArray,imageSize);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
   // Launch sum_up_arrays kernel on the shadow arrays for each channel
   {
    dim3 blockDim(8,8), gridDim(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
        deviceBlueShadowArray,deviceBlueSumShadowArray,imageSize);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
   // Launch sum_up_arrays kernel on the light arrays for each channel
   {
    dim3 blockDim(8,8), gridDim(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
        deviceRedLightArray,deviceRedSumLightArray,imageSize);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
  {
    dim3 blockDim(8,8), gridDim(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
          deviceGreenLightArray,deviceGreenSumLightArray,imageSize);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
  {
    dim3 blockDim(8,8), gridDim(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
          deviceBlueLightArray,deviceBlueSumLightArray,imageSize);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  }
  // Launch sum_up_arrays kernel on the eroded shadow array
  {
    dim3 blockDim(8,8), gridDim(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
      deviceErodedShadowMask,deviceErodedSumShadowArray,imageSize);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  } 
  // Launch sum_up_arrays kernel on the eroded light array
  {
    dim3 blockDim(8,8), gridDim(1,1);
    sum_up_arrays_by_reduction<<<gridDim, blockDim>>>(
      deviceErodedLightMask,deviceErodedSumLightArray,imageSize);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  } 

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

  printf("\nSum of Red Shadow Array:\t%.4f",redSumShadowArray);
  printf("\nSum of Blue Shadow Array:\t%.4f",blueSumShadowArray);
  printf("\nSum of Green Shadow Array:\t%.4f",greenSumShadowArray);
  printf("\nSum of Red Light Array:\t%.4f",redSumLightArray);
  printf("\nSum of Blue Light Array:\t%.4f",blueSumLightArray);
  printf("\nSum of Green Light Array:\t%.4f",greenSumLightArray);

  finalImage = (unsigned char *)malloc(imageSize * sizeof(unsigned char));

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceFinalImage, imageSize * sizeof(unsigned char))); 
  CUDA_CHECK( cudaMalloc((void **)&deviceSmoothMask, imageSize * sizeof(float)));      
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  // zero out bins
    CUDA_CHECK(cudaMemset(deviceFinalImage, 0.0, imageSize * sizeof(unsigned char)));
  // Launch calculate_rgb_ratio kernel on the eroded shadow array and calculates the final image
  {
    dim3 blockDim(8,8), gridDim(1,1);
    calculate_final_image<<<gridDim, blockDim>>>(
    deviceRedSumShadowArray, deviceGreenSumShadowArray,deviceBlueSumShadowArray,
    deviceRedSumLightArray, deviceGreenSumLightArray,deviceBlueSumLightArray,
    deviceErodedSumShadowArray,deviceErodedSumShadowArray,
    deviceRgbImage, deviceSmoothMask, deviceFinalImage,
    imageWidth, imageHeight, NUM_CHANNELS);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
  } 

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(finalImage, deviceFinalImage,
                        imageSize * sizeof(unsigned char),
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
  CUDA_CHECK(cudaFree(deviceGreenSumShadowArray));
  CUDA_CHECK(cudaFree(deviceBlueSumShadowArray));
  CUDA_CHECK(cudaFree(deviceRedSumLightArray)); 
  CUDA_CHECK(cudaFree(deviceGreenSumLightArray));
  CUDA_CHECK(cudaFree(deviceBlueSumLightArray));
  CUDA_CHECK(cudaFree(deviceErodedSumShadowArray));
  CUDA_CHECK(cudaFree(deviceErodedSumLightArray));
  CUDA_CHECK(cudaFree(deviceSmoothMask));
  CUDA_CHECK(cudaFree(deviceFinalImage));
  wbTime_stop(GPU, "Freeing GPU Memory");

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
                                    0, 0, 0, 0,
                                    0, 0, 0, 0};
  unsigned char erodedLight[16] =  {0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1};
  float smoothMask[16] =  {0.36, 0.48, 0.36, 0.48,
                                   0.36, 0.48, 0.36, 0.48,
                                   1.0, 1.0, 1.0, 1.0,
                                   1.0, 1.0, 1.0, 0.0};

  inputImage_RGB_float  = wbImage_getData(inputImage_RGB);

  print_image(inputImage_RGB_float,imageWidth,imageHeight);

  unit_test(inputImage_RGB_float,erodedShadow,erodedLight,smoothMask,imageWidth, imageHeight);

  free(inputImage_RGB_float);
  wbImage_delete(inputImage_RGB);

  return 0;

}