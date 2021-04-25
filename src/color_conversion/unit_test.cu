#include <wb.h>
#include "../globals.h"
#include "kernel.cu"

void color_conversions(unsigned char *rgbImage, float *invImage, float *grayImage, float *yuvImage, int imageWidth, int imageHeight, const char* imageid) {
  int imageSize = imageWidth * imageHeight;

  dim3 blockDim(512), gridDim(30);

  timerLog_startEvent(&timerLog);
  convert_rgb_invariant<<<gridDim, blockDim>>>(rgbImage, invImage, imageWidth, imageHeight, 3);
  timerLog_stopEventAndLog(&timerLog, "RGB to invariant", imageid, imageWidth, imageHeight);

  timerLog_startEvent(&timerLog);
  convert_invariant_grayscale<<<gridDim, blockDim>>>(invImage, grayImage, imageWidth, imageHeight, 3);
  timerLog_stopEventAndLog(&timerLog, "invariant to grayscale", imageid, imageWidth, imageHeight);
  
  timerLog_startEvent(&timerLog);
  convert_rgb_yuv<<<gridDim, blockDim>>>(rgbImage, yuvImage, imageWidth, imageHeight, 3);
  timerLog_stopEventAndLog(&timerLog, "RGB to YUV", imageid, imageWidth, imageHeight);
  
  timerLog_startEvent(&timerLog);
  color_convert<<<gridDim, blockDim>>>(rgbImage, invImage, grayImage, yuvImage, imageWidth, imageHeight);
  timerLog_stopEventAndLog(&timerLog, "color convert", imageid, imageWidth, imageHeight);
  
} 

int main(int argc, char *argv[]) {
  
  	wbArg_t args;
  	int imageWidth;
  	int imageHeight;
    int imageSize;

  	char *inputImageFile;

  	wbImage_t inputImage_RGB;
  	//wbImage_t outputImage_Inv;
  	//wbImage_t outputImage_Gray;
    //wbImage_t outputImage_YUV;

  	unsigned char *hostInputImageData_RGB;
  	float *hostOutputImageData_Inv;
  	float *hostOutputImageData_Gray;
  	float *hostOutputImageData_YUV;

  	unsigned char *deviceInputImageData_RGB;
  	float *deviceOutputImageData_Inv;
  	float *deviceOutputImageData_Gray;
  	float *deviceOutputImageData_YUV;

  	args = wbArg_read(argc, argv); // parse the input arguments

    // FIXME: generate input image
  	inputImageFile = wbArg_getInputFile(args, 0);
  	inputImage_RGB = wbImport(inputImageFile);

  	imageWidth = wbImage_getWidth(inputImage_RGB);
  	imageHeight = wbImage_getHeight(inputImage_RGB);

    printf("\nRunning color convert unit test on image of %dx%d with %d channels\n\n",
             imageWidth, imageHeight, NUM_CHANNELS);

    imageSize = imageWidth * imageHeight;

  	//outputImage_Inv = wbImage_new(imageWidth, imageHeight, NUM_CHANNELS);
  	//outputImage_Gray = wbImage_new(imageWidth, imageHeight, 1);
  	//outputImage_YUV = wbImage_new(imageWidth, imageHeight, NUM_CHANNELS);

  	hostInputImageData_RGB = wbImage_getData(inputImage_RGB);

    hostOutputImageData_Inv =  (float *)malloc(imageSize * NUM_CHANNELS * sizeof(float));  //wbImage_getData(outputImage_Inv);
  	hostOutputImageData_Gray = (float *)malloc(imageSize * 1 * sizeof(float)); //wbImage_getData(outputImage_Gray);
  	hostOutputImageData_YUV =  (float *)malloc(imageSize * NUM_CHANNELS * sizeof(float)); //wbImage_getData(outputImage_YUV);

  	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    //@@ Allocate GPU memory here
  	wbTime_start(GPU, "Doing GPU memory allocation");
  	CUDA_CHECK(cudaMalloc((void **)&deviceInputImageData_RGB, imageSize * NUM_CHANNELS * sizeof(unsigned char)));
	  CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_Inv, imageSize * NUM_CHANNELS * sizeof(float)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_Gray, imageSize * 1 * sizeof(float)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_YUV, imageSize * NUM_CHANNELS * sizeof(float)));
  	wbTime_stop(GPU, "Doing GPU memory allocation");

    //@@ Copy memory to the GPU here
  	wbTime_start(Copy, "Copying data to the GPU");
  	CUDA_CHECK(cudaMemcpy(deviceInputImageData_RGB, hostInputImageData_RGB,
            	imageSize * NUM_CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice));
  	wbTime_stop(Copy, "Copying data to the GPU");

  	wbTime_start(Compute, "Doing the computation on the GPU");
  	// defining grid size (num blocks) and block size (num threads per block)
  	dim3 gridDim(ceil(imageWidth/16.0), ceil(imageHeight/16.0), 1);
  	dim3 blockDim(16, 16, 1);

  	// launch
  	color_conversions(deviceInputImageData_RGB, deviceOutputImageData_Inv, deviceOutputImageData_Gray, deviceOutputImageData_YUV, imageWidth, imageHeight, "ut");

  	wbTime_stop(Compute, "Doing the computation on the GPU");

    //@@ Copy the GPU memory back to the CPU here
  	wbTime_start(Copy, "Copying data from the GPU");
  	CUDA_CHECK(cudaMemcpy(hostOutputImageData_Inv, deviceOutputImageData_Inv,
    		       imageSize * NUM_CHANNELS * sizeof(float), cudaMemcpyDeviceToHost));
  	CUDA_CHECK(cudaMemcpy(hostOutputImageData_Gray, deviceOutputImageData_Gray,
    		       imageSize * 1 * sizeof(float), cudaMemcpyDeviceToHost));
  	CUDA_CHECK(cudaMemcpy(hostOutputImageData_YUV, deviceOutputImageData_YUV,
    		       imageSize * NUM_CHANNELS * sizeof(float), cudaMemcpyDeviceToHost));
  	wbTime_stop(Copy, "Copying data from the GPU");

  	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  	// wbSolution(args, outputImage_Inv, outputImage_Gray, outputImage_YUV);

    printf("\n");
  	printf("First 3 values of inv image:   %.4f, %.4f, %.4f\n", hostOutputImageData_Inv[0],hostOutputImageData_Inv[1],hostOutputImageData_Inv[2]);
  	printf("First 3 values of gray image:   %4d,  %4d,  %4d\n", hostOutputImageData_Gray[0],hostOutputImageData_Gray[1],hostOutputImageData_Gray[2]);
  	printf("First 3 values of yuv image:    %4d,  %4d,  %4d\n", hostOutputImageData_YUV[0],hostOutputImageData_YUV[1],hostOutputImageData_YUV[2]);
    printf("\n");

    //@@ Free the GPU memory here
    wbTime_start(GPU, "Freeing GPU Memory");
  	CUDA_CHECK(cudaFree(deviceInputImageData_RGB));
  	CUDA_CHECK(cudaFree(deviceOutputImageData_Inv));
  	CUDA_CHECK(cudaFree(deviceOutputImageData_Gray));
  	CUDA_CHECK(cudaFree(deviceOutputImageData_YUV));
    wbTime_stop(GPU, "Freeing GPU Memory");

  	wbImage_delete(inputImage_RGB);

  	free(hostOutputImageData_Inv);
  	free(hostOutputImageData_Gray);
  	free(hostOutputImageData_YUV);
  	//wbImage_delete(outputImage_Inv);
  	//wbImage_delete(outputImage_Gray);
  	//wbImage_delete(outputImage_YUV);

  	return 0;
}
