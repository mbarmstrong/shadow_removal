#ifndef GLOBALS_H
#define GLOBALS_H

#define PRINT_DEBUG 1

#define NUM_CHANNELS 3
#define NUM_BINS 256
#define MAX_BLOCK_SZ 1024

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}


void print_pixel(unsigned char* image, int row, int col, int channel, int num_channels, int imageSize) {

    printf("\n");
    int idx = row * imageSize + col;
    printf("Channel %d, row %d, col %d is:\t%d",channel,row,col,image[idx*num_channels+channel-1]); 
    printf("\n");

}

void print_pixel(float* image, int row, int col, int channel, int num_channels, int imageSize) {

    printf("\n");
    int idx = row * imageSize + col;
    printf("Channel %d, row %d, col %d is:\t%.4f",channel,row,col,image[idx*num_channels+channel-1]); 
    printf("\n");

}

void print_image(unsigned char* image, int imageWidth, int imageHeight) {

    printf("\n");
    for(int i = 0; i < imageWidth*imageHeight; i++){
        printf("%4d",image[i]);

        if(i%imageWidth==(imageWidth-1)) printf("\n");
        else printf(",");
    }
    printf("\n");
}

void print_image(float* image, int imageWidth, int imageHeight) {

    printf("\n");
    for(int i = 0; i < imageWidth*imageHeight; i++){
        printf("%.4f",image[i]);

        if(i%imageWidth==(imageWidth-1)) printf("\n");
        else printf(",");
    }
    printf("\n");
}

void print_sparse_array(unsigned int* arr, int size) {

  printf("\n");
  for(int i = 0; i < size; i++) {
    if(arr[i] > 0) printf("%4d-%d, ",i, arr[i]);
  }
  printf("\n\n");
}

void print_step_array(float* arr, int size) {

  printf("\n");
  printf("%d-%.4f, ",0, arr[0]);
  for(int i = 1; i < size; i++) {
    if(arr[i] != arr[i-1]) printf("%d-%.4f, ",i, arr[i]);
  }
  printf("\n\n");
}

#endif