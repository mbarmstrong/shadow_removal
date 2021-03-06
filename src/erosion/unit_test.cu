#include <wb.h>
#include "kernel.cu"
#include "../globals.h"

void erosion_kernels(unsigned char* inImage, unsigned char* outImage_shadow, unsigned char* outImage_light, int maskWidth,  int imageWidth, int imageHeight, const char* imageid) {
  int imageSize = imageWidth * imageHeight;

  int n_threads = 16;
  dim3 gridDim(ceil((float)imageWidth/(float)n_threads), ceil((float)imageHeight/(float)n_threads));
  dim3 blockDim(n_threads,n_threads);

  timerLog_startEvent(&timerLog);
  image_erode_shadow<<<gridDim, blockDim>>>(inImage, outImage_shadow, maskWidth, imageWidth, imageHeight);
  timerLog_stopEventAndLog(&timerLog, "shadow mask", imageid, imageWidth, imageHeight);

  timerLog_startEvent(&timerLog);
  image_erode_light<<<gridDim, blockDim>>>(inImage, outImage_light, maskWidth, imageWidth, imageHeight);
  timerLog_stopEventAndLog(&timerLog, "light mask", imageid, imageWidth, imageHeight);

  timerLog_startEvent(&timerLog);
  image_erode<<<gridDim, blockDim>>>(inImage, outImage_shadow, outImage_light, maskWidth, imageWidth, imageHeight);
  timerLog_stopEventAndLog(&timerLog, "erosion global memory", imageid, imageWidth, imageHeight);

  timerLog_startEvent(&timerLog);
  image_erode_shared<<<gridDim, blockDim>>>(inImage, outImage_shadow, outImage_light, maskWidth, imageWidth, imageHeight);
  timerLog_stopEventAndLog(&timerLog, "erosion shared memory", imageid, imageWidth, imageHeight);
  
}

#ifndef SOLUTION

void unit_test(unsigned char* image, int imageWidth, int imageHeight) {

  unsigned char *hostOutputImage_shadow;
  unsigned char *hostOutputImage_light;

  unsigned char *deviceInputImage;
  unsigned char *deviceOutputImage_shadow;
  unsigned char *deviceOutputImage_light;

  int imageSize = imageWidth * imageHeight;
  int maskWidth = 3;

  hostOutputImage_shadow = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  hostOutputImage_light = (unsigned char *)malloc(imageSize * sizeof(unsigned char));;

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceInputImage, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceOutputImage_shadow, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceOutputImage_light, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  //@@ Copy memory to the GPU here
  wbTime_start(GPU, "Copying input memory to the GPU.");
  CUDA_CHECK(cudaMemcpy(deviceInputImage, image, imageSize * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));                        
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  int n_threads = 16;
  dim3 gridDim(ceil((float)imageWidth/(float)n_threads),ceil((float)imageHeight/(float)n_threads));
  dim3 blockDim(n_threads,n_threads);
  image_erode<<<gridDim, blockDim>>>(deviceInputImage, deviceOutputImage_shadow, 
                                  deviceOutputImage_light, maskWidth, imageWidth, 
                                  imageHeight);

  erosion_kernels(deviceInputImage, deviceOutputImage_shadow, deviceOutputImage_light, maskWidth, imageWidth,imageHeight, "ut");

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(hostOutputImage_shadow, deviceOutputImage_shadow,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hostOutputImage_light, deviceOutputImage_light,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // printf("\noutput image (shadow):\n");
  // print_image(hostOutputImage_shadow,imageWidth,imageHeight);

  // printf("\noutput image (light):\n");
  // print_image(hostOutputImage_light,imageWidth,imageHeight);
  
  CUDA_CHECK(cudaFree(deviceInputImage));
  CUDA_CHECK(cudaFree(deviceOutputImage_shadow));
  CUDA_CHECK(cudaFree(deviceOutputImage_light));

  free(hostOutputImage_shadow);
  free(hostOutputImage_light);

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
    timerLog = timerLog_new( wbArg_getOutputFile(args) );

  	inputImageFile = wbArg_getInputFile(args, 0);
  	inputImage_RGB = wbImport(inputImageFile);

  	imageWidth = wbImage_getWidth(inputImage_RGB);
  	imageHeight = wbImage_getHeight(inputImage_RGB);

    imageSize = imageWidth * imageHeight;

    printf("\nRunning erosion unit test on image of %dx%d\n",
             imageWidth, imageHeight, NUM_CHANNELS);

    inputImage_RGB_uint8 = (unsigned char*)malloc(imageSize * sizeof(unsigned char));

    for(int i = 0; i < imageSize; i++){
        inputImage_RGB_uint8[i] = (unsigned char)(round(wbImage_getData(inputImage_RGB)[i*3]));
    }

    // unsigned char data[16] = {0, 1, 1, 1,
    //                           1, 1, 1, 1,
    //                           1, 1, 1, 1,
    //                           1, 1, 1, 0};

    // inputImage_RGB_uint8 = data;

    // print_image(inputImage_RGB_uint8,imageWidth,imageHeight);

    unit_test(inputImage_RGB_uint8, imageWidth, imageHeight);

    timerLog_save(&timerLog);

    //free(inputImage_RGB_uint8);
    wbImage_delete(inputImage_RGB);

    return 0;
}

#endif