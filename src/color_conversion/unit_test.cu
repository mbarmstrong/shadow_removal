#include <wb.h>
#include "../globals.h"
#include "kernel.cu"

int main(int argc, char *argv[]) {
  
  	wbArg_t args;
  	int imageChannels;
  	int imageWidth;
  	int imageHeight;

  	char *inputImageFile;

	wbImage_t inputImage_RGB;
	wbImage_t outputImage_Inv;
	wbImage_t outputImage_Gray;
    wbImage_t outputImage_YUV;

  	float *hostInputImageData_RGB;
  	float *hostOutputImageData_Inv;
  	float *hostOutputImageData_Gray;
  	float *hostOutputImageData_YUV;

  	float *deviceInputImageData_RGB;
  	float *deviceOutputImageData_Inv;
  	float *deviceOutputImageData_Gray;
  	float *deviceOutputImageData_YUV;

  	args = wbArg_read(argc, argv); // parse the input arguments

    // FIXME: generate input image
  	inputImageFile = wbArg_getInputFile(args, 0);
  	inputImage_RGB = wbImport(inputImageFile);

  	imageWidth = wbImage_getWidth(inputImage_RGB);
  	imageHeight = wbImage_getHeight(inputImage_RGB);
  	imageChannels = wbImage_getChannels(inputImage_RGB);

  	outputImage_Inv = wbImage_new(imageWidth, imageHeight, 3);
  	outputImage_Gray = wbImage_new(imageWidth, imageHeight, 3);
  	outputImage_YUV = wbImage_new(imageWidth, imageHeight, 3);

  	hostInputImageData_RGB = wbImage_getData(inputImage_RGB);

  	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    //@@ Allocate GPU memory here
  	wbTime_start(GPU, "Doing GPU memory allocation");
  	CUDA_CHECK(cudaMalloc((void **)&deviceInputImageData_RGB,
            	imageWidth * imageHeight * imageChannels * sizeof(float)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_Gray,
            	imageWidth * imageHeight * sizeof(float)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_YUV,
            	imageWidth * imageHeight * sizeof(float)));
  	wbTime_stop(GPU, "Doing GPU memory allocation");

    //@@ Copy memory to the GPU here
  	wbTime_start(Copy, "Copying data to the GPU");
  	CUDA_CHECK(cudaMemcpy(deviceInputImageData_RGB, hostInputImageData_RGB,
            	imageWidth * imageHeight * imageChannels * sizeof(float),
            	cudaMemcpyHostToDevice));
  	wbTime_stop(Copy, "Copying data to the GPU");

  	wbTime_start(Compute, "Doing the computation on the GPU");
  	// defining grid size (num blocks) and block size (num threads per block)
  	dim3 gridDim(ceil(imageWidth/16.0), ceil(imageHeight/16.0), 1);
  	dim3 blockDim(16, 16, 1);

  	// launch kernel
  	// color_convert<<<gridDim, blockDim>>>(deviceInputImageData_RGB, deviceOutputImageData_Inv, 
  	//								       deviceOutputImageData_Gray, deviceOutputImageData_YUV, 
  	//								       imageWidth, imageHeight, imageChannels);
  	wbTime_stop(Compute, "Doing the computation on the GPU");

    //@@ Copy the GPU memory back to the CPU here
  	wbTime_start(Copy, "Copying data from the GPU");
  	CUDA_CHECK(cudaMemcpy(hostOutputImageData_Inv, deviceOutputImageData_Inv,
    		       imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost));
  	CUDA_CHECK(cudaMemcpy(hostOutputImageData_Gray, deviceOutputImageData_Gray,
    		       imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost));
  	CUDA_CHECK(cudaMemcpy(hostOutputImageData_YUV, deviceOutputImageData_YUV,
    		       imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost));
  	wbTime_stop(Copy, "Copying data from the GPU");

  	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  	// wbSolution(args, outputImage_Inv, outputImage_Gray, outputImage_YUV);

    //@@ Free the GPU memory here
    wbTime_start(GPU, "Freeing GPU Memory");
  	CUDA_CHECK(cudaFree(deviceInputImageData_RGB));
  	CUDA_CHECK(cudaFree(deviceOutputImageData_Inv));
  	CUDA_CHECK(cudaFree(deviceOutputImageData_Gray));
  	CUDA_CHECK(cudaFree(deviceOutputImageData_YUV));
    wbTime_stop(GPU, "Freeing GPU Memory");

  	wbImage_delete(outputImage_Inv);
  	wbImage_delete(outputImage_Gray);
  	wbImage_delete(outputImage_YUV);
  	wbImage_delete(inputImage_RGB);

  	return 0;
}
