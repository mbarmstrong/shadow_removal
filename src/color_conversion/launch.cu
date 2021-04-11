#include "kernel.cu"

void launch_color_convert(wbImage_t& inputImage, wbImage_t& outputImage) {
	int imageChannels;
  	int imageWidth;
  	int imageHeight;
  	int imageSize;

  	float *hostInputImageData_RGB;
  	float *hostOutputImageData_Inv;
  	float *hostOutputImageData_Gray;
  	float *hostOutputImageData_YUV;

  	float *deviceInputImageData_RGB;
  	float *deviceOutputImageData_Inv;
  	float *deviceOutputImageData_Gray;
  	float *deviceOutputImageData_YUV;

  	hostInputImageData_RGB = wbImage_getData(inputImage);

  	imageWidth = wbImage_getWidth(inputImage);
  	imageHeight = wbImage_getHeight(inputImage);
  	imageChannels = wbImage_getChannels(inputImage);

  	imageSize = imageWidth * imageHeight;

  	//@@ Allocate GPU memory here
  	wbTime_start(GPU, "Allocating GPU memory.");
  	CUDA_CHECK(cudaMalloc((void **)&deviceInputImageData_RGB, imageSize * sizeof(float)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_Inv, imageSize * sizeof(float)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_Gray, imageSize * sizeof(float)));
  	CUDA_CHECK(cudaMalloc((void **)&deviceOutputImageData_YUV, imageSize * sizeof(float)));
  	CUDA_CHECK(cudaDeviceSynchronize());
  	wbTime_stop(GPU, "Allocating GPU memory.");

  	//@@ Copy memory to the GPU here
  	wbTime_start(GPU, "Copying input memory to the GPU.");
  	CUDA_CHECK(cudaMemcpy(deviceInputImageData_RGB, hostInputImageData_RGB,
                        imageSize * sizeof(float), cudaMemcpyHostToDevice));
  	CUDA_CHECK(cudaDeviceSynchronize());
  	wbTime_stop(GPU, "Copying input memory to the GPU.");

  	// Launch kernel
  	// dim3 blockDim(512), gridDim(30);
  	// histogram_shared_kernel<<<gridDim, blockDim,num_bins * sizeof(unsigned int)>>>(
  	//       input, bins, num_elements, num_bins);
  	// CUDA_CHECK(cudaGetLastError());
  	// CUDA_CHECK(cudaDeviceSynchronize());

  	//@@ Copy the GPU memory back to the CPU here
  	wbTime_start(Copy, "Copying output memory to the CPU");
  	CUDA_CHECK(cudaMemcpy(hostOutputImageData_Inv, deviceOutputImageData_Inv,
                        imageSize * sizeof(float), cudaMemcpyDeviceToHost));
  	CUDA_CHECK(cudaMemcpy(hostOutputImageData_Gray, deviceOutputImageData_Gray,
                        imageSize * sizeof(float), cudaMemcpyDeviceToHost));
  	CUDA_CHECK(cudaMemcpy(hostOutputImageData_YUV, deviceOutputImageData_YUV,
                        imageSize * sizeof(float), cudaMemcpyDeviceToHost));
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

//void launch_invariant_grayscale(wbImage_t& inputImage, wbImage_t& outputImage) {}

//...