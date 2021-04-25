#ifndef GLOBALS_H
#define GLOBALS_H

#define PRINT_DEBUG 1

#define NUM_CHANNELS 3
#define NUM_BINS 256
#define MAX_BLOCK_SZ 1024

#define MAX_LOG_ENTRIES 20

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr)[0]))

struct st_timerLog_t {

  //Variables to be logged
  char kernel_name[MAX_LOG_ENTRIES][50];
  int width[MAX_LOG_ENTRIES];
  int height[MAX_LOG_ENTRIES];
  float time[MAX_LOG_ENTRIES];

  char _header[4][20];
  int _entry_count;
  char* _out_file;
  bool _write_header;

  bool _event_created;
  cudaEvent_t _start_event;
  cudaEvent_t _stop_event;
  float _elapsed_time;
};

/*st_timerLog_t timerLog_new(char* outfile) {
  
  if (outfile == NULL)
    printf("\nFile Logging Turned Off\n");

  st_timerLog_t log = {._header = {{"kernel\0"},{"width\0"},{"height\0"},{"time (ms)\0"}},
                       ._entry_count = 0,
                       ._out_file = outfile,
                       ._write_header = true,
                       ._event_created = false };

  return log;

}*/

void timerLog_save(st_timerLog_t* log) {

  if(log->_out_file == NULL) {
    return;
  }

  FILE *handle;

  if(log->_write_header) {
    handle = fopen(log->_out_file, "w");
    for(int i = 0; i < LEN(log->_header); i++) {
      fprintf(handle, "%s",log->_header[i]);
      if(i < LEN(log->_header)-1)
        fprintf(handle, ", ");
      else
        fprintf(handle, "\n");
    }

    log->_write_header = false;
  }
  else {
    handle = fopen(log->_out_file, "a");
  }

  for(int i = 0; i < log->_entry_count; i++)
      fprintf(handle, "%s, %d, %d, %f\n",log->kernel_name[i], log->width[i], log->height[i], log->time[i]);

  fflush(handle);
  fclose(handle);

}

void timerLog_stopEventAndLog(st_timerLog_t* log, const char* kernel, int width, int height) {

  cudaEventRecord(log->_stop_event, 0);
  cudaEventSynchronize(log->_stop_event);
  cudaEventElapsedTime(&(log->_elapsed_time), log->_start_event, log->_stop_event);

  int e = log->_entry_count;

  if(e >= MAX_LOG_ENTRIES){
    timerLog_save(log);
    log->_entry_count = 0;
  }

  strcpy(log->kernel_name[e],kernel);
  log->width[e] = width;
  log->height[e] = height;
  log->time[e] = log->_elapsed_time;

  log->_elapsed_time = 0.0;
  log->_entry_count++;

}

void timerLog_startEvent(st_timerLog_t* log) {

  if(!(log->_event_created)) {
    cudaEventCreate(&(log->_start_event));
    cudaEventCreate(&(log->_stop_event));
    log->_event_created = true;
  }

  cudaEventRecord(log->_start_event, 0);

}


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

void write_data(char *file_name, float *data,
                       int width, int height,
                       int channels) {                       
  FILE *handle = fopen(file_name, "w");
  if (channels == 1) {
    fprintf(handle, "P5\n");
  } else {
    fprintf(handle, "P6\n");
  }
  fprintf(handle, "#Created by %s\n", __FILE__);
  fprintf(handle, "%d %d\n", width, height);
  fprintf(handle, "255\n");

  fwrite(data, width * channels * sizeof(float), height, handle);

  fflush(handle);
  fclose(handle);
}

#endif