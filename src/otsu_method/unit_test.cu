
#include <wb.h>
#include "kernel.cu"
#include "histo.cu"
#include "../globals.h"

st_timerLog_t timerLog;

void histograms(unsigned char* deviceImage, unsigned int* deviceBins, int imageSize) {

  cudaEvent_t astartEvent, astopEvent;
  float aelapsedTime;
  cudaEventCreate(&astartEvent);
  cudaEventCreate(&astopEvent);
  
  // Launch histogram kernel on the bins
  dim3 blockDim(512), gridDim(30);

  CUDA_CHECK(cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int)));
  cudaEventRecord(astartEvent, 0);
  wbTime_start(GPU, "Running global histogram");
  histogram_global_kernel<<<gridDim, blockDim, NUM_BINS * sizeof(unsigned int)>>>(
      deviceImage, deviceBins, imageSize);
  //CUDA_CHECK(cudaGetLastError());
  //CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Running global histogram");
  cudaEventRecord(astopEvent, 0);
  cudaEventSynchronize(astopEvent);
  cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
  printf("\n");
  printf("Total compute time (ms) %f\n",aelapsedTime);
  printf("\n");
  timerLog_append(&timerLog,"histogram global",aelapsedTime);
  
  CUDA_CHECK(cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int)));
  cudaEventRecord(astartEvent, 0);
  histogram_shared_kernel<<<gridDim, blockDim, NUM_BINS * sizeof(unsigned int)>>>(
      deviceImage, deviceBins, imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEventRecord(astopEvent, 0);
  cudaEventSynchronize(astopEvent);
  cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
  printf("\n");
  printf("Total compute time (ms) %f\n",aelapsedTime);
  printf("\n");
  timerLog_append(&timerLog,"histogram shared",aelapsedTime);

  CUDA_CHECK(cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int)));
  cudaEventRecord(astartEvent, 0);
  histogram_shared_accumulate_kernel<<<gridDim, blockDim, NUM_BINS * sizeof(unsigned int)>>>(
      deviceImage, deviceBins, imageSize);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEventRecord(astopEvent, 0);
  cudaEventSynchronize(astopEvent);
  cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
  printf("\n");
  printf("Total compute time (ms) %f\n",aelapsedTime);
  printf("\n");
  timerLog_append(&timerLog,"histogram shared accumulate",aelapsedTime);

  CUDA_CHECK(cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int)));
  cudaEventRecord(astartEvent, 0);
  wbTime_start(GPU, "Running R histogram");
  int shared_size = (NUM_BINS+1) * 12 * sizeof(unsigned int);
  histogram_shared_R_kernel<<<gridDim, blockDim, shared_size>>>(
        deviceImage, deviceBins, imageSize, 12);
  //CUDA_CHECK(cudaGetLastError());
  //CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Running R histogram");
  cudaEventRecord(astopEvent, 0);
  cudaEventSynchronize(astopEvent);
  cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
  printf("\n");
  printf("Total compute time (ms) %f\n",aelapsedTime);
  printf("\n");
  timerLog_append(&timerLog,"histogram shared R",aelapsedTime);

}

float unit_test(unsigned char* image, int imageWidth, int imageHeight) {


  //-------------------------------------------------
  //  Historgram
  //
  //-------------------------------------------------
  unsigned int* hostBins;
  unsigned int* deviceBins;
  unsigned char* deviceImage;

  int imageSize = imageWidth * imageHeight;

  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

  //@@ Allocate GPU memory here
  CUDA_CHECK( cudaMalloc((void **)&deviceImage, imageSize * sizeof(unsigned char)) );
  CUDA_CHECK( cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int)) );

  //@@ Copy memory to the GPU here
  CUDA_CHECK(cudaMemcpy(deviceImage, image,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyHostToDevice));

  histograms(deviceImage, deviceBins, imageSize);

  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostBins, deviceBins,
                        NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("histogram bins:");
  print_sparse_array(hostBins,NUM_BINS);


  //-------------------------------------------------
  //  Omega
  //
  //-------------------------------------------------
  float* deviceOmega;
  float* hostOmega;
  
  hostOmega = (float *)malloc(NUM_BINS * sizeof(float));

  CUDA_CHECK( cudaMalloc((void **)&deviceOmega, NUM_BINS * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );

  CUDA_CHECK(cudaMemset(deviceOmega, 0.0, NUM_BINS * sizeof(float)));

  dim3 blockDim1(NUM_BINS), gridDim1(1);
  omega<<<gridDim1,blockDim1>>>(deviceBins,deviceOmega,imageSize);

  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostOmega, deviceOmega,
                        NUM_BINS * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("\nomega:");
  print_step_array(hostOmega,NUM_BINS);


  //-------------------------------------------------
  //  Mu
  //
  //-------------------------------------------------
  float* deviceMu;
  float* hostMu;
  
  hostMu = (float *)malloc(NUM_BINS * sizeof(float));

  CUDA_CHECK( cudaMalloc((void **)&deviceMu, NUM_BINS * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );

  CUDA_CHECK(cudaMemset(deviceMu, 0.0, NUM_BINS * sizeof(float)));

  mu<<<gridDim1,blockDim1>>>(deviceBins,deviceMu,imageSize);

  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostMu, deviceMu,
                        NUM_BINS * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("\nmu:");
  print_step_array(hostMu,NUM_BINS);


  //-------------------------------------------------
  //  Sigma B Squared
  //
  //-------------------------------------------------
  float* deviceSigmaBsq;
  float* hostSigmaBsq;
  
  hostSigmaBsq = (float *)malloc(NUM_BINS * sizeof(float));

  CUDA_CHECK( cudaMalloc((void **)&deviceSigmaBsq, NUM_BINS * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );

  CUDA_CHECK(cudaMemset(deviceSigmaBsq, 0.0, NUM_BINS * sizeof(float)));

  sigma_b_squared<<<gridDim1,blockDim1>>>(deviceOmega,deviceMu,deviceSigmaBsq);

  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostSigmaBsq, deviceSigmaBsq,
                        NUM_BINS * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("\nsigma_b_sq:");
  print_step_array(hostSigmaBsq,NUM_BINS);

  //-------------------------------------------------
  //  Calculate threashold level
  //
  //-------------------------------------------------

  //Replace with kernel but probs won't gain much speedup
  float level = calculate_threshold_cpu(hostSigmaBsq);

  printf("\n level:\t%.4f\n\n", level);


  //-------------------------------------------------
  //  Create binary image
  //
  //-------------------------------------------------
  unsigned char *hostBinaryImage;
  unsigned char *deviceBinaryImage;

  hostBinaryImage = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  
  CUDA_CHECK( cudaMalloc((void **)&deviceBinaryImage, imageSize * sizeof(float)) );
  CUDA_CHECK( cudaDeviceSynchronize() );

  // zero out image
  CUDA_CHECK(cudaMemset(deviceBinaryImage, 0, imageSize * sizeof(unsigned char)));

  int n_threads = 16;
  dim3 gridDim2(ceil((float)imageWidth/(float)n_threads),ceil((float)imageHeight/(float)n_threads));
  dim3 blockDim2(n_threads,n_threads);
  create_binarized_image<<<gridDim2, blockDim2>>>(deviceImage, deviceBinaryImage,
                                                level, imageWidth, imageHeight, false);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostBinaryImage, deviceBinaryImage,
                        imageSize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  //print_image(hostBinaryImage,imageWidth,imageHeight);

  //-------------------------------------------------
  //  Cleanup
  //
  //-------------------------------------------------
  //@@ Free the GPU memory here
  CUDA_CHECK(cudaFree(deviceImage));
  CUDA_CHECK(cudaFree(deviceBins));
  CUDA_CHECK(cudaFree(deviceOmega));
  CUDA_CHECK(cudaFree(deviceMu));
  CUDA_CHECK(cudaFree(deviceSigmaBsq));
  CUDA_CHECK(cudaFree(deviceBinaryImage));

  free(hostBins);
  free(hostOmega);
  free(hostMu);
  free(hostSigmaBsq);
  free(hostBinaryImage);

  return 0;

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

    printf("\nRunning outsu unit test on image of %dx%d\n",
             imageWidth, imageHeight, NUM_CHANNELS);

    inputImage_RGB_uint8 = (unsigned char*)malloc(imageSize * sizeof(unsigned char));

    for(int i = 0; i < imageSize; i++){
        inputImage_RGB_uint8[i] = (unsigned char)(wbImage_getData(inputImage_RGB)[i*3]*255);
    }

    //print_image(inputImage_RGB_uint8,imageWidth,imageHeight);

    unit_test(inputImage_RGB_uint8,imageWidth,imageHeight);

    timerLog_save(&timerLog);

    free(inputImage_RGB_uint8);
    wbImage_delete(inputImage_RGB);

    return 0;

}