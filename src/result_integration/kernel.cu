// Kernel 1: Finding average channel values in shadow/light areas for every channel
__global__ void multiply_rgbImage_byMask(float *rgbImage, unsigned char *greyShadowMask, 
  unsigned char *greyLightMask, unsigned char *redShadowArray,unsigned char *greenShadowArray,unsigned char *blueShadowArray,
  unsigned char *redLightArray,unsigned char *greenLightArray,unsigned char *blueLightArray,int width, int height, int numChannels) {
    int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row index
  
    if (col < width && row < height) {  // check boundary condition
        int idx = row * width + col;      // mapping 2D to 1D coordinate
        // load input RGB values
        float red = rgbImage[numChannels * idx];      // red component
        float green = rgbImage[numChannels * idx + 1];  // green component
        float blue = rgbImage[numChannels * idx + 2];  // blue component

        redShadowArray[idx] = (unsigned char) (red * greyShadowMask[idx]);
        greenShadowArray[idx] = (unsigned char) (green * greyShadowMask[idx]);
        blueShadowArray[idx] = (unsigned char) (blue * greyShadowMask[idx]);
        redLightArray[idx] = (unsigned char) (red * greyLightMask[idx]);
        greenLightArray[idx] = (unsigned char) (green * greyLightMask[idx]);
        blueLightArray[idx] = (unsigned char) (blue * greyLightMask[idx]);
  
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
  __global__ void sum_up_arrays_by_reduction(unsigned char *g_idata, unsigned char *g_odata, int n) {
    extern __shared__ int sdata[];

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

  __global__ void calculate_final_image(unsigned char *redSumShadowArray, unsigned char *greenSumShadowArray,unsigned char *blueSumShadowArray,
    unsigned char *redSumLightArray, unsigned char *greenSumLightArray,unsigned char *blueSumLightArray,
    unsigned char *erodedSumShadowArray,unsigned char *erodedSumLightArray,
    float *rgbImage, float *smoothMask, unsigned char *finalImage,
    int width, int height, int numChannels) {
  
    float redRatio = (float)(((redSumLightArray[0]/erodedSumLightArray[0])/(redSumShadowArray[0]/erodedSumShadowArray[0])) -1);
    float greenRatio = (float)(((greenSumLightArray[0]/erodedSumLightArray[0])/(greenSumShadowArray[0]/erodedSumShadowArray[0])) -1);
    float blueRatio = (float)(((blueSumLightArray[0]/erodedSumLightArray[0])/(blueSumShadowArray[0]/erodedSumShadowArray[0])) -1);
  
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