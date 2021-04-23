// Kernel 1: Finding average channel values in shadow/light areas for every channel
__global__ void multiply_rgbImage_byMask(float *rgbImage, unsigned char *greyShadowMask, 
  unsigned char *greyLightMask, float *redShadowArray,float *greenShadowArray,float *blueShadowArray,
  float *redLightArray,float *greenLightArray,float *blueLightArray,int width, int height, int numChannels) {
    int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row index
  
    if (col < width && row < height) {  // check boundary condition
        int idx = row * width + col;      // mapping 2D to 1D coordinate
        // load input RGB values
        float red = rgbImage[numChannels * idx];      // red component
        float green = rgbImage[numChannels * idx + 1];  // green component
        float blue = rgbImage[numChannels * idx + 2];  // blue component

        redShadowArray[idx] = red * (float)greyShadowMask[idx];
        greenShadowArray[idx] = green * (float)greyShadowMask[idx];
        blueShadowArray[idx] = blue * (float)greyShadowMask[idx];
        redLightArray[idx] = red * (float)greyLightMask[idx];
        greenLightArray[idx] = green * (float)greyLightMask[idx];
        blueLightArray[idx] = blue * (float)greyLightMask[idx];
  
    }
  
  }
  
  
  // // Kernel 2: Sums up the light arrays, shadow array and the eroded array - Without reduction
  // __global__ void sum_up_arrays(unsigned char *redShadowArray,unsigned char *greenShadowArray,unsigned char *blueShadowArray,
  //   unsigned char *redLightArray,unsigned char *greenLightArray,unsigned char *blueLightArray,unsigned char *erodedShadowMask,unsigned char *erodedLightMask
  //   int width, int height,
  //   unsigned char *redSumShadowArray, unsigned char *greenSumShadowArray,unsigned char *blueSumShadowArray,
  //   unsigned char *redSumLightArray, unsigned char *greenSumLightArray,unsigned char *blueSumLightArray,
  //   unsigned char *erodedSumShadowArray,unsigned char *erodedSumShadowArray) {
  
  //   int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
  //   int row = threadIdx.y + blockIdx.y * blockDim.y; // row index
  
  //   if (col < width && row < height) {  // check boundary condition
  //       int idx = row * width + col;      // mapping 2D to 1D coordinate
  //       redSumShadowArray += redShadowArray[idx];
  //       greenSumShadowArray += greenShadowArray[idx];
  //       blueSumShadowArray += blueShadowArray[idx];
  //       redSumLightArray += redLightArray[idx];
  //       greenSumLightArray += greenLightArray[idx];
  //       blueSumLightArray += blueLightArray[idx];

  //       erodedSumShadowArray += erodedShadowMask[idx];
  //       erodedSumLightArray += erodedLightMask[idx];
  //   }
  // }
  // template __device__ void warpReduce(volatile unsigned char *sdata, unsigned int tid) { 
  //   if (blockDim.x >= 64) 
  //        sdata[tid] += sdata[tid + 32]; 
  //   if (blockDim.x >= 32) 
  //       sdata[tid] += sdata[tid + 16]; 
  //   if (blockDim.x >= 16) 
  //      sdata[tid] += sdata[tid + 8];
  //   if (blockDim.x >= 8) 
  //     sdata[tid] += sdata[tid + 4]; 
  //   if (blockDim.x >= 4) 
  //     sdata[tid] += sdata[tid + 2]; 
  //   if (blockDim.x >= 2) 
  //     sdata[tid] += sdata[tid + 1]; 
  // }  

 // Kernel 2: Array Reduction Kernel - 1D array
  __global__ void sum_up_arrays_by_reduction(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];

    // each thread loads one element from global to shared mem
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = g_idata[i];

    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *=2)
    {
      __syncthreads();
        int index = 2 * s * tid;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[tid];
    }

}

 // Kernel 2: Array Reduction Kernel - 1D array
  __global__ void sum_up_arrays_by_reduction1(unsigned char *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];

    // each thread loads one element from global to shared mem
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = g_idata[i];

    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *=2)
    {
      __syncthreads();
        int index = 2 * s * tid;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[tid];
    }

}

  // Calculates the Red, Green and Blue ratios from the sum in Kernel 2 to produce the shadowless image 
  // Uses the RGB ratios produced in Kernel 3 and the input image to remove the shadow and create the final output 

  __global__ void calculate_final_image(float redSumShadowArray, float greenSumShadowArray,float blueSumShadowArray,
    float redSumLightArray, float greenSumLightArray,float blueSumLightArray,
    float erodedSumShadowArray,float erodedSumLightArray,
    float *rgbImage, float *smoothMask, float *finalImage,
    int width, int height, int numChannels) {
  
    float redRatio = (((redSumLightArray/erodedSumLightArray)/(redSumShadowArray/erodedSumShadowArray)) -1);
    float greenRatio = (((greenSumLightArray/erodedSumLightArray)/(greenSumShadowArray/erodedSumShadowArray)) -1);
    float blueRatio = (((blueSumLightArray/erodedSumLightArray)/(blueSumShadowArray/erodedSumShadowArray)) -1);
  
    int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row index
  
    if (col < width && row < height) {  // check boundary condition
        int idx = row * width + col;      // mapping 2D to 1D coordinate
        // load input RGB values
        float red = rgbImage[numChannels * idx];      // red component
        float green = rgbImage[numChannels * idx + 1];  // green component
        float blue = rgbImage[numChannels * idx + 2];  // blue component

        finalImage[numChannels * idx] = ((redRatio + 1) / ((1 - smoothMask[idx]) * redRatio + 1) * red);
        finalImage[numChannels * idx + 1] = ((greenRatio + 1) / ((1 - smoothMask[idx]) * greenRatio + 1) * green);
        finalImage[numChannels * idx + 2] = ((blueRatio + 1) / ((1 - smoothMask[idx]) * blueRatio + 1) * blue);
    }
  }