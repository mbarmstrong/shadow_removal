#include "kernel.cu"

void launch_color_convert(unsigned char *inputImage_RGB, float *outputImage_Inv,
						  float *outputImage_Gray, float *outputImage_YUV,
						  int imageWidth, int imageHeight, int imageSize) {

  	unsigned char *deviceInputImageData_RGB;
  	float *deviceOutputImageData_Inv;
  	float *deviceOutputImageData_Gray;
  	float *deviceOutputImageData_YUV;

  	//@@ Allocate GPU memory here
  	wbTime_start(GPU, "Allocating GPU memory.");
  	CUDA_CHECK(cudaMalloc((void **)&deviceInputImageData_RGB, imageSize * NUM_CHANNELS * sizeof(unsigned char)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_Inv, imageSize * NUM_CHANNELS * sizeof(float)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_Gray, imageSize * 1 * sizeof(float)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_YUV, imageSize * NUM_CHANNELS * sizeof(float)));
  	CUDA_CHECK(cudaDeviceSynchronize());
  	wbTime_stop(GPU, "Allocating GPU memory.");

  	//@@ Copy memory to the GPU here
  	wbTime_start(GPU, "Copying input memory to the GPU.");
  	CUDA_CHECK(cudaMemcpy(deviceInputImageData_RGB, inputImage_RGB,
                        imageSize * NUM_CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice));
  	CUDA_CHECK(cudaDeviceSynchronize());
  	wbTime_stop(GPU, "Copying input memory to the GPU.");

  	// launch kernel
  	dim3 gridDim(ceil(imageWidth/16.0), ceil(imageHeight/16.0), 1);
  	dim3 blockDim(16, 16, 1);
  	color_convert<<<gridDim, blockDim>>>(deviceInputImageData_RGB, deviceOutputImageData_Inv, 
  										deviceOutputImageData_Gray, deviceOutputImageData_YUV, 
  										imageWidth, imageHeight);
  	CUDA_CHECK(cudaGetLastError());
  	CUDA_CHECK(cudaDeviceSynchronize());

  	//@@ Copy the GPU memory back to the CPU here
  	wbTime_start(Copy, "Copying output memory to the CPU");
  	CUDA_CHECK(cudaMemcpy(outputImage_Inv, deviceOutputImageData_Inv,
                        imageSize * NUM_CHANNELS * sizeof(float), cudaMemcpyDeviceToHost));
  	CUDA_CHECK(cudaMemcpy(outputImage_Gray, deviceOutputImageData_Gray,
                        imageSize * 1 * sizeof(float), cudaMemcpyDeviceToHost));
  	CUDA_CHECK(cudaMemcpy(outputImage_YUV, deviceOutputImageData_YUV,
                        imageSize * NUM_CHANNELS * sizeof(float), cudaMemcpyDeviceToHost));
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