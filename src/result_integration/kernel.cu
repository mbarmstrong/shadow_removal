// Kernel 1: Finding average channel values in shadow/light areas for every channel
__global__ void multiple_rgbImage_byMask(float *rgbImage, float *greyShadowMask, 
  float *greyLightMask, float *redShadowArray,float *greenShadowArray,float *blueShadowArray,
  float *redLightArray,float *greenLightArray,float *blueLightArray,int width, int height, int numChannels) {
    int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row index
  
    if (col < width && row < height) {  // check boundary condition
        int idx = row * width + col;      // mapping 2D to 1D coordinate
        // load input RGB values
        float red = rgbImage[numChannels * idx];      // red component
        float green = rgbImage[numChannels * idx + 1];  // green component
        float blue = rgbImage[numChannels * idx + 2];  // blue component

        redShadowArray[idx] = red[idx] * greyShadowMask[idx];
        greenShadowArray[idx] = green[idx] * greyShadowMask[idx];
        blueShadowArray[idx] = blue[idx] * greyShadowMask[idx];
        redlightArray[idx] = red[idx] * greyLightMask[idx];
        greenlightArray[idx] = green[idx] * greyLightMask[idx];
        bluelightArray[idx] = blue[idx] * greyLightMask[idx];
  
    }
  
  }
  
  
  // Kernel 2: Sums up the light arrays, shadow array and the eroded array - Without reduction
  __global__ void sum_up_arrays(float *redShadowArray,float *greenShadowArray,float *blueShadowArray,
    float *redLightArray,float *greenLightArray,float *blueLightArray,float *erodedShadowArray,float *erodedLightArray
    int width, int height,
    float redSumShadowArray, float greenSumShadowArray,float blueSumShadowArray,
    float redSumLightArray, float greenSumLightArray,float blueSumLightArray,
    float erodedSumShadowArray,float erodedSumShadowArray) {
  
    int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row index
  
    if (col < width && row < height) {  // check boundary condition
        int idx = row * width + col;      // mapping 2D to 1D coordinate
        redSumShadowArray += redShadowArray[idx];
        greenSumShadowArray += greenShadowArray[idx];
        blueSumShadowArray += blueShadowArray[idx];
        redSumLightArray += redLightArray[idx];
        greenSumLightArray += greenLightArray[idx];
        blueSumLightArray += blueLightArray[idx];

        erodedSumShadowArray += erodedShadowArray[idx];
        erodedSumLightArray += erodedLightArray[idx];
    }
  }
 // Kernel 2: Array Reduction Kernel - 1D array
  __global__ void sum_up_1Darrays_by_reduction(float *g_idata, float *g_odata) {
    extern __shared__ int sdata[]; 
    unsigned int tid = threadIdx.x; 
    unsigned int i = blockIdx.x*(blockSize*2) + tid; 
    unsigned int gridSize = blockSize*2*gridDim.x; 
    sdata[tid] = 0; 
    while (i < n) { 
         sdata[tid] += g_idata[i] + g_idata[i+blockSize];
         i += gridSize;
     } 
      __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) { 
           sdata[tid] += sdata[tid + 256]; 
        } 
      __syncthreads(); 
   } 
   if (blockSize >= 256) { 
       if (tid < 128) { 
           sdata[tid] += sdata[tid + 128]; 
       } 
      __syncthreads(); 
   } 
   if (blockSize >= 128) { 
       if (tid < 64) { 
           sdata[tid] += sdata[tid + 64]; 
       } 
      __syncthreads(); 
   } 
   if (tid < 32) {
       warpReduce(sdata, tid); 
   }   
   if (tid == 0) {
       g_odata[blockIdx.x] = sdata[0]; 
   }    

}

template __device__ void warpReduce(volatile int *sdata, unsigned int tid) { 
  if (blockSize >= 64) 
       sdata[tid] += sdata[tid + 32]; 
  if (blockSize >= 32) 
      sdata[tid] += sdata[tid + 16]; 
  if (blockSize >= 16) 
     sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) 
    sdata[tid] += sdata[tid + 4]; 
  if (blockSize >= 4) 
    sdata[tid] += sdata[tid + 2]; 
  if (blockSize >= 2) 
    sdata[tid] += sdata[tid + 1]; 
} 

// Kernel 2: Array Reduction Kernel - 2D array
__global__ void sum_up_2Darrays_by_reduction(float *g_idata, float *g_odata) {

}
  
  // Calculates the Red, Green and Blue ratios from the sum in Kernel 2 to produce the shadowless image
  __global__ void calculate_rgb_ratio(float redSumShadowArray, float greenSumShadowArray,float blueSumShadowArray,
    float redSumLightArray, float greenSumLightArray,float blueSumLightArray,
    float erodedSumShadowArray,float erodedSumShadowArray,
    float *rgbImage, float *smoothMask, float *finalImage,
    int width, int height, int numChannels) {
  
    int redRatio = ((redSumLightArray/erodedSumLightArray)/(redSumShadowArray/erodedSumShadowArray)) -1;
    int greenRatio = ((greenSumLightArray/erodedSumLightArray)/(greenSumShadowArray/erodedSumShadowArray)) -1;
    int blueRatio = ((blueSumLightArray/erodedSumLightArray)/(blueSumShadowArray/erodedSumShadowArray)) -1;
  
      create_final_shadowless_output(rgbImage, smoothMask, finalImage
        redRatio,greenRatio,blueRatio,width, height,numChannels);
    }
  
  // Uses the RGB ratios produced in Kernel 3 and the input image to remove the shadow and create the final output 
  
  __global__ void create_final_shadowless_output(float *rgbImage, float *smoothMask, float *finalImage
    int redRatio,int greenRatio,int blueRatio,int width, int height, int numChannels) {
   
    int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row index
  
    if (col < width && row < height) {  // check boundary condition
        int idx = row * width + col;      // mapping 2D to 1D coordinate
        // load input RGB values
        float red = rgbImage[numChannels * idx];      // red component
        float green = rgbImage[numChannels * idx + 1];  // green component
        float blue = rgbImage[numChannels * idx + 2];  // blue component

        finalImage[numChannels * idx] = (redRatio + 1) / ((1 - smoothMask[idx]) * redRatio + 1) * rgbImage[numChannels * idx];
        finalImage[numChannels * idx + 1] = (greenRatio + 1) / ((1 - smoothMask[idx]) * greenRatio + 1) * rgbImage[numChannels * idx + 1];
        finalImage[numChannels * idx + 2] = (blueRatio + 1) / ((1 - smoothMask[idx]) * blueRatio + 1) * rgbImage[numChannels * idx + 2];
  
    }


  }