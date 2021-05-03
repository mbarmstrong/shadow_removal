// #include "kernel.cu"
#include "unit_test.cu"

#define RUN_SWEEPS_CC 0

void launch_color_convert(float *inputImage_RGB, float *outputImage_Inv,
						  unsigned char *outputImage_Gray, unsigned char* outputImage_YUV,
						  int imageWidth, int imageHeight, int imageSize, const char* imageid) {

  	float *deviceInputImageData_RGB;
  	float *deviceOutputImageData_Inv;
  	unsigned char *deviceOutputImageData_Gray;
  	unsigned char *deviceOutputImageData_YUV;

  	//@@ Allocate GPU memory here
  	wbTime_start(GPU, "Allocating GPU memory.");
  	CUDA_CHECK(cudaMalloc((void **)&deviceInputImageData_RGB, imageSize * NUM_CHANNELS * sizeof(float)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_Inv, imageSize * NUM_CHANNELS * sizeof(float)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_Gray, imageSize * 1 * sizeof(unsigned char)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_YUV, imageSize * NUM_CHANNELS * sizeof(unsigned char)));
  	CUDA_CHECK(cudaDeviceSynchronize());
  	wbTime_stop(GPU, "Allocating GPU memory.");

  	//@@ Copy memory to the GPU here
  	wbTime_start(GPU, "Copying input memory to the GPU.");
  	CUDA_CHECK(cudaMemcpy(deviceInputImageData_RGB, inputImage_RGB,
                        imageSize * NUM_CHANNELS * sizeof(float), cudaMemcpyHostToDevice));
  	CUDA_CHECK(cudaDeviceSynchronize());
  	wbTime_stop(GPU, "Copying input memory to the GPU.");

    #if RUN_SWEEPS_CC
    color_conversions(deviceInputImageData_RGB, deviceOutputImageData_Inv, deviceOutputImageData_Gray, deviceOutputImageData_YUV, imageWidth, imageHeight, imageid);
    #endif
	
  	// launch kernel
  	dim3 gridDim(ceil(imageWidth/16.0), ceil(imageHeight/16.0), 1);
  	dim3 blockDim(16, 16, 1);
	timerLog_startEvent(&timerLog);
  	color_convert<<<gridDim, blockDim>>>(deviceInputImageData_RGB, 
  										deviceOutputImageData_Gray, deviceOutputImageData_YUV, 
  										imageWidth, imageHeight);
	timerLog_stopEventAndLog(&timerLog, "Color Conversion", "\0", imageWidth, imageHeight);
  	CUDA_CHECK(cudaGetLastError());
  	CUDA_CHECK(cudaDeviceSynchronize());

  	//@@ Copy the GPU memory back to the CPU here
  	wbTime_start(Copy, "Copying output memory to the CPU");
  	CUDA_CHECK(cudaMemcpy(outputImage_Inv, deviceOutputImageData_Inv,
                        imageSize * NUM_CHANNELS * sizeof(float), cudaMemcpyDeviceToHost));
  	CUDA_CHECK(cudaMemcpy(outputImage_Gray, deviceOutputImageData_Gray,
                        imageSize * 1 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
  	CUDA_CHECK(cudaMemcpy(outputImage_YUV, deviceOutputImageData_YUV,
                        imageSize * NUM_CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost));
  	CUDA_CHECK(cudaDeviceSynchronize());
  	wbTime_stop(Copy, "Copying output memory to the CPU");

  	//@@ Free the GPU memory here
 	wbTime_start(GPU, "Freeing GPU Memory");
 	CUDA_CHECK(cudaFree(deviceInputImageData_RGB));
  	CUDA_CHECK(cudaFree(deviceOutputImageData_Inv));
  	CUDA_CHECK(cudaFree(deviceOutputImageData_Gray));
  	CUDA_CHECK(cudaFree(deviceOutputImageData_YUV));
  	wbTime_stop(GPU, "Freeing GPU Memory");

}