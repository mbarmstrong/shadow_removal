
#include <wb.h>
#include "kernel.cu"
#include "kernel_reduction.cu"
#include "../globals.h"

void unit_test( float *rgbImage,unsigned char *erodedShadowMask,unsigned char *erodedLightMask, float *smoothMask,int imageWidth, int imageHeight) {

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
    float *deviceRedSumShadowArray;
    float *deviceGreenSumShadowArray;
    float *deviceBlueSumShadowArray;
    float *deviceRedSumLightArray; 
    float *deviceGreenSumLightArray;
    float *deviceBlueSumLightArray;
    float *deviceErodedSumShadowArray;
    float *deviceErodedSumLightArray;
    float *deviceRedRatio;
    float *deviceGreenRatio;
    float *deviceBlueRatio;
    unsigned char *deviceErodedShadowMask;
    unsigned char *deviceErodedLightMask;
    float *deviceSmoothMask;
    float *deviceFinalImage;

  float *finalImage;

  int imageSize = imageHeight * imageWidth;

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceRgbImage, imageSize * NUM_CHANNELS *sizeof(float)));
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

  // Launch sum_up_arrays kernel on the light and shadow arrays for each channel
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

  printf("\nSum of Red Shadow Array:\t %.04f",redSumShadowArray);
  printf("\nSum of Green Shadow Array:\t%.04f",greenSumShadowArray);
  printf("\nSum of Blue Shadow Array:\t %.04f",blueSumShadowArray);
  printf("\nSum of Red Light Array:\t %.04f",redSumLightArray);
  printf("\nSum of Green Light Array:\t %.04f",greenSumLightArray);
  printf("\nSum of Blue Light Array:\t%.04f",blueSumShadowArray);
  printf("\nSum of Eroded  Shadow Array:\t%.04f",erodedSumShadowArray);
  printf("\nSum of Eroded  Light Array:\t%.04f",erodedSumLightArray);

  float redRatio = (((redSumLightArray/erodedSumLightArray)/(redSumShadowArray/erodedSumShadowArray)) -1);
  float greenRatio = (((greenSumLightArray/erodedSumLightArray)/(greenSumShadowArray/erodedSumShadowArray)) -1);
  float blueRatio = (((blueSumLightArray/erodedSumLightArray)/(blueSumShadowArray/erodedSumShadowArray)) -1);
  
  printf("\nredRatio:\t%.04f",redRatio);
  printf("\ngreenRatio:\t%.04f",greenRatio);
  printf("\nblueRatio:\t%.04f",blueRatio);


  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceRedRatio, sizeof(float)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceGreenRatio, sizeof(float)));   
  CUDA_CHECK( cudaMalloc((void **)&deviceBlueRatio, sizeof(float)));  
  // CUDA_CHECK( cudaMalloc((void **)&deviceRedSumShadowArray, sizeof(float)));   
  // CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumShadowArray, sizeof(float)));   
  // CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumShadowArray, sizeof(float)));   
  // CUDA_CHECK( cudaMalloc((void **)&deviceRedSumLightArray, sizeof(float)));   
  // CUDA_CHECK( cudaMalloc((void **)&deviceGreenSumLightArray, sizeof(float)));
  // CUDA_CHECK( cudaMalloc((void **)&deviceBlueSumLightArray, sizeof(float))); 
  // CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumLightArray, sizeof(float))); 
  // CUDA_CHECK( cudaMalloc((void **)&deviceErodedSumShadowArray, sizeof(float)));  
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
// CUDA_CHECK(cudaMemcpy(deviceRedSumShadowArray, &redSumShadowArray,
//                       sizeof(float),
//                       cudaMemcpyHostToDevice));
// CUDA_CHECK(cudaMemcpy(deviceGreenSumShadowArray, &greenSumShadowArray,
//                       sizeof(float),
//                       cudaMemcpyHostToDevice));
// CUDA_CHECK(cudaMemcpy(deviceBlueSumShadowArray, &blueSumShadowArray,
//                       sizeof(float),
//                       cudaMemcpyHostToDevice));
// CUDA_CHECK(cudaMemcpy(deviceRedSumLightArray, &redSumLightArray,
//                       sizeof(float),
//                       cudaMemcpyHostToDevice));
// CUDA_CHECK(cudaMemcpy(deviceGreenSumLightArray, &greenSumLightArray,
//                       sizeof(float),
//                       cudaMemcpyHostToDevice));
// CUDA_CHECK(cudaMemcpy(deviceBlueSumLightArray, &blueSumLightArray,
//                       sizeof(float),
//                       cudaMemcpyHostToDevice)); 
// CUDA_CHECK(cudaMemcpy(deviceErodedSumShadowArray, &erodedSumShadowArray,
//                       sizeof(float),
//                       cudaMemcpyHostToDevice));     
// CUDA_CHECK(cudaMemcpy(deviceErodedSumLightArray, &erodedSumLightArray,
//                       sizeof(float),
//                       cudaMemcpyHostToDevice));               
CUDA_CHECK(cudaDeviceSynchronize());
wbTime_stop(Copy, "Copying output memory to the CPU");

  finalImage = (float *)malloc(imageSize * NUM_CHANNELS * sizeof(float));

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceFinalImage, imageSize * NUM_CHANNELS * sizeof(float)));  
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  // zero out bins
    CUDA_CHECK(cudaMemset(deviceFinalImage, 0.0, imageSize * NUM_CHANNELS * sizeof(float)));
  // Launch calculate_rgb_ratio kernel on the eroded shadow array and calculates the final image
  {
  dim3 gridDim2(ceil(imageWidth/16.0), ceil(imageHeight/16.0), 1);
  dim3 blockDim2(16, 16, 1);
  calculate_final_image_stride<<<gridDim2, blockDim2>>>(deviceRedRatio, deviceGreenRatio,deviceBlueRatio,
  deviceRgbImage, deviceSmoothMask, deviceFinalImage,
  imageWidth, imageHeight, NUM_CHANNELS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  } 

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(finalImage, deviceFinalImage,
                        imageSize * NUM_CHANNELS * sizeof(float),
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
  CUDA_CHECK(cudaFree(deviceSmoothMask));
  CUDA_CHECK(cudaFree(deviceFinalImage));
  wbTime_stop(GPU, "Freeing GPU Memory");

}

int main(int argc, char *argv[]) {
  
  wbArg_t args;
  int imageWidth;
  int imageHeight;

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

  wbImage_delete(inputImage_RGB);

  return 0;

}