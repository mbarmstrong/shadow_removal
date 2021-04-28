#ifndef __CONV_UNIT_TEST__
#define __CONV_UNIT_TEST__

#include <wb.h>
#include "kernel.cu"
#include "../globals.h"


void convolutions(unsigned char* deviceInputImage, float* deviceOutputImage, int maskWidth, int imageWidth, int imageHeight) {

  int imageSize = imageWidth * imageHeight;
  float *deviceOutputImageTemp;

  int n_threads;
  int blockX;
  int blockY;
  int gridX;
  int gridY;

  CUDA_CHECK( cudaMalloc((void **)&deviceOutputImageTemp, imageSize * sizeof(float)) );

  CUDA_CHECK(cudaMemset(deviceOutputImage, 0.0, imageSize * sizeof(float))); 

  n_threads = 16;
  blockX = n_threads; blockY = n_threads;
  gridX = ceil((float)imageWidth/(float)n_threads);
  gridY = ceil((float)imageHeight/(float)n_threads);

  dim3 gridDim(gridX, gridY);
  dim3 blockDim(blockX, blockY);
  timerLog_startEvent(&timerLog);
  conv2d<<<gridDim, blockDim>>>(deviceInputImage, deviceOutputImage, maskWidth,
                                imageWidth, imageHeight);
  timerLog_stopEventAndLog(&timerLog,"Convolution Global", "\0", imageWidth, imageHeight);

  CUDA_CHECK(cudaMemset(deviceOutputImage, 0.0, imageSize * sizeof(float))); 

  n_threads = 16;
  blockX = n_threads + maskWidth - 1;
  blockY = n_threads + maskWidth - 1;
  gridX = ceil((float)imageWidth/(float)n_threads);
  gridY = ceil((float)imageHeight/(float)n_threads);

  dim3 gridDimS(gridX, gridY);
  dim3 blockDimS(blockX, blockY);

  timerLog_startEvent(&timerLog);
  conv2d_shared<<<gridDimS, blockDimS>>>(deviceInputImage, deviceOutputImage, maskWidth,
                                imageWidth, imageHeight);
  timerLog_stopEventAndLog(&timerLog,"Convolution Shared", "\0", imageWidth, imageHeight);


  CUDA_CHECK(cudaMemset(deviceOutputImage, 0.0, imageSize * sizeof(float))); 

  n_threads = 16;
  blockX = n_threads + maskWidth - 1;
  blockY = n_threads;
  gridX = ceil((float)imageWidth/(float)n_threads);
  gridY = ceil((float)imageHeight/(float)n_threads);

  dim3 gridDimRow(gridX, gridY);
  dim3 blockDimRow(blockX, blockY);

  timerLog_startEvent(&timerLog);
  conv_separable_row<<<gridDimRow, blockDimRow>>>(deviceInputImage, deviceOutputImageTemp, maskWidth,
                                imageWidth, imageHeight);

  n_threads = 16;
  blockX = n_threads;
  blockY = n_threads + maskWidth - 1;
  gridX = ceil((float)imageWidth/(float)n_threads);
  gridY = ceil((float)imageHeight/(float)n_threads);

  dim3 gridDimCol(gridX, gridY);
  dim3 blockDimCol(blockX, blockY);

  conv_separable_col<<<gridDimCol, blockDimCol>>>(deviceOutputImageTemp, deviceOutputImage, maskWidth,
                                imageWidth, imageHeight);

  timerLog_stopEventAndLog(&timerLog,"Convolution Separable", "\0", imageWidth, imageHeight);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(deviceOutputImageTemp));

}

#ifndef SOLUTION

void unit_test(unsigned char* image, int imageWidth, int imageHeight) {

  float *hostOutputImage;

  unsigned char *deviceInputImage;
  float         *deviceOutputImage;

  int imageSize = imageWidth * imageHeight;
  int maskWidth = 5;

  hostOutputImage = (float *)malloc(imageSize * sizeof(float));

  CUDA_CHECK( cudaMalloc((void **)&deviceInputImage, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceOutputImage, imageSize * sizeof(float)) );

  //@@ Copy memory to the GPU here
  CUDA_CHECK(cudaMemcpy(deviceInputImage, image, imageSize * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));                        
  CUDA_CHECK(cudaDeviceSynchronize());

  convolutions(deviceInputImage, deviceOutputImage, maskWidth, imageWidth, imageHeight);

  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostOutputImage, deviceOutputImage,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));

  printf("\noutput image:\n");
  print_image(hostOutputImage,imageWidth,imageHeight);
  
  CUDA_CHECK(cudaFree(deviceInputImage));
  CUDA_CHECK(cudaFree(deviceOutputImage));

  free(hostOutputImage);

}

int main(int argc, char *argv[]) {
  
  	wbArg_t args;
  	int imageWidth;
  	int imageHeight;
    int imageSize;

  	char *inputImageFile;

  	wbImage_t inputImage_RGB;

    unsigned char* inputImage_RGB_uint8;

  	args = wbArg_read(argc, argv); // parse the input arguments

  	inputImageFile = wbArg_getInputFile(args, 0);
  	inputImage_RGB = wbImport(inputImageFile);

  	imageWidth = wbImage_getWidth(inputImage_RGB);
  	imageHeight = wbImage_getHeight(inputImage_RGB);

    imageSize = imageWidth * imageHeight;

    printf("\nRunning convolution unit test on image of %dx%d\n",
             imageWidth, imageHeight, NUM_CHANNELS);

    //inputImage_RGB_uint8 = (unsigned char*)malloc(imageSize * sizeof(unsigned char));

    // for(int i = 0; i < imageSize; i++){
    //     inputImage_RGB_uint8[i] = (unsigned char)(round(wbImage_getData(inputImage_RGB)[i*3]));
    // }

    unsigned char data[16] = {1, 1, 1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1};

    inputImage_RGB_uint8 = data;

    print_image(inputImage_RGB_uint8,imageWidth,imageHeight);

    unit_test(inputImage_RGB_uint8,imageWidth,imageHeight);

    //free(inputImage_RGB_uint8);
    wbImage_delete(inputImage_RGB);

    return 0;

}

#endif
#endif