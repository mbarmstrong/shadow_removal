// Kernel 1: Finding average channel values in shadow/light areas for every channel
__global__ void multiple_rgbImage_bymask_kernel(float *rgbImage, float *greyShadowMask, 
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
  
  
  // Kernel 2: Sums up the light arrays, shadow array and the eroded array
  __global__ void sum_up_arrays_kernel(float *redShadowArray,float *greenShadowArray,float *blueShadowArray,
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
 // Kernel 2: Array Reduction Kernel
  __global__ void reduce(float *g_idata, float *g_odata) {

    __shared__ int sdata[256];

    // each thread loads one element from global to shared mem
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[threadIdx.x] = g_idata[i];

    __syncthreads();
    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *=2)
    {
        int index = 2 * s * threadIdx.x;;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (threadIdx.x == 0)
        atomicAdd(g_odata,sdata[0]);
}
  
  // Calculates the Red, Green and Blue ratios from the sum in Kernel 2 to produce the shadowless image
  __global__ void calculate_rgb_ratio_kernel(float redSumShadowArray, float greenSumShadowArray,float blueSumShadowArray,
    float redSumLightArray, float greenSumLightArray,float blueSumLightArray,
    float erodedSumShadowArray,float erodedSumShadowArray,int redRatio,int greenRatio,int blueRatio) {
  
      redRatio = ((redSumLightArray/erodedSumLightArray)/(redSumShadowArray/erodedSumShadowArray)) -1;
      greenRatio = ((greenSumLightArray/erodedSumLightArray)/(greenSumShadowArray/erodedSumShadowArray)) -1;
      blueRatio = ((blueSumLightArray/erodedSumLightArray)/(blueSumShadowArray/erodedSumShadowArray)) -1;
  
    }
  
  // Uses the RGB ratios produced in Kernel 3 and the input image to remove the shadow and create the final output 
  
  __global__ void create_final_shadowless_output(ufloat *rgbImage, float *smoothMask, float *finalImage
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