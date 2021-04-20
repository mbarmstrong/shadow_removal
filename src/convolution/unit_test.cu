#include <wb.h>
#include "kernel.cu"
#include "../globals.h"

void unit_test(unsigned char* image, int imageWidth, int imageHeight) {

  float *hostOutputImage;

  unsigned char *deviceInputImage;
  float         *deviceOutputImage;

  int imageSize = imageWidth * imageHeight;
  int maskWidth = 3;

  hostOutputImage = (float *)malloc(imageSize * sizeof(float));

  wbTime_start(GPU, "Allocating GPU memory.");
  CUDA_CHECK( cudaMalloc((void **)&deviceInputImage, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceOutputImage, imageSize * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );
  wbTime_stop(GPU, "Allocating GPU memory.");

  //@@ Copy memory to the GPU here
  wbTime_start(GPU, "Copying input memory to the GPU.");
  CUDA_CHECK(cudaMemcpy(deviceInputImage, image, imageSize * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));                        
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  dim3 blockDim(8,8), gridDim(1,1);
  conv2d<<<gridDim, blockDim>>>(deviceInputImage, deviceOutputImage, maskWidth,
                                imageWidth, imageHeight);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  //@@ Copy the GPU memory back to the CPU here
  wbTime_start(Copy, "Copying output memory to the CPU");
  CUDA_CHECK(cudaMemcpy(hostOutputImage, deviceOutputImage,
                        imageSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

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