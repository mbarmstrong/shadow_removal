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

  __global__ void sum_up_arrays_naive(float *g_idata, float *g_odata, int n,int width,int height) {

    int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row index
  
    if (col < width && row < height) {  // check boundary condition
        int idx = row * width + col;      // mapping 2D to 1D coordinate
        g_odata[idx] = g_odata[idx] + g_idata[idx];
       
    }
    __syncthreads();
}

  // Calculates the Red, Green and Blue ratios from the sum in Kernel 2 to produce the shadowless image 
  // Uses the RGB ratios produced in Kernel 3 and the input image to remove the shadow and create the final output 

  __global__ void calculate_final_image(float *redSumShadowArray, float *greenSumShadowArray,float *blueSumShadowArray,
    float *redSumLightArray, float *greenSumLightArray,float *blueSumLightArray,
    float *erodedSumShadowArray,float *erodedSumLightArray,
    float *rgbImage, float *smoothMask, float *finalImage,
    int width, int height, int numChannels) {
  
    float redRatio = (float)(((float)(redSumLightArray[0]/erodedSumLightArray[0])/(float)(redSumShadowArray[0]/erodedSumShadowArray[0])) -1);
    float greenRatio = (float)(((float)(greenSumLightArray[0]/erodedSumLightArray[0])/(float)(greenSumShadowArray[0]/erodedSumShadowArray[0])) -1);
    float blueRatio = (float)(((float)(blueSumLightArray[0]/erodedSumLightArray[0])/(float)(blueSumShadowArray[0]/erodedSumShadowArray[0])) -1);
  
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

    __global__ void calculate_final_image_optimised(float *redRatio, float *greenRatio,float *blueRatio,
    float *rgbImage, float *smoothMask, float *finalImage,
    int width, int height, int numChannels) {
  
    int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row index
  
    if (col < width && row < height) {  // check boundary condition
        int idx = row * width + col;      // mapping 2D to 1D coordinate
        // load input RGB values
        float red = rgbImage[numChannels * idx];      // red component
        float green = rgbImage[numChannels * idx + 1];  // green component
        float blue = rgbImage[numChannels * idx + 2];  // blue component

        finalImage[numChannels * idx] = ((redRatio[0] + 1) / ((1 - smoothMask[idx]) * redRatio[0] + 1) * red);
        finalImage[numChannels * idx + 1] = ((greenRatio[0] + 1) / ((1 - smoothMask[idx]) * greenRatio[0] + 1) * green);
        finalImage[numChannels * idx + 2] = ((blueRatio[0] + 1) / ((1 - smoothMask[idx]) * blueRatio[0] + 1) * blue);
    }
  }

    __global__ void calculate_final_image_stride(float *redRatio, float *greenRatio,float *blueRatio,
    float *rgbImage, float *smoothMask, float *finalImage,
    int width, int height, int numChannels) {

    int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row index

    int stride = width*height;
  
    if (col < width && row < height) {  // check boundary condition
        int idx = row * width + col;      // mapping 2D to 1D coordinate
        // load input RGB values
        float red = rgbImage[numChannels * idx];      // red component
        float green = rgbImage[numChannels * idx + 1];  // green component
        float blue = rgbImage[numChannels * idx + 2];  // blue component

        float finalImageRed = ((redRatio[0] + 1) / ((1 - smoothMask[idx]) * redRatio[0] + 1) * red);
        float finalImageGreen = ((greenRatio[0] + 1) / ((1 - smoothMask[idx]) * greenRatio[0] + 1) * green);
        float finalImageBlue = ((blueRatio[0] + 1) / ((1 - smoothMask[idx]) * blueRatio[0] + 1) * blue);

        finalImage[idx] = finalImageRed;
        finalImage[1 * stride + idx] = finalImageGreen;
        finalImage[2 * stride + idx] = finalImageBlue;
    }
    }


    __global__ void calculate_final_image_optimised_const(float *redRatio, float *greenRatio,float *blueRatio,
    float *rgbImage, float *smoothMask, float *finalImage,
    int width, int height, int numChannels) {
  
    int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row index
  
    if (col < width && row < height) {  // check boundary condition
        int idx = row * width + col;      // mapping 2D to 1D coordinate
        // load input RGB values
        int redIdx = numChannels * idx;
        int greenIdx = numChannels * idx + 1;
        int blueIdx = numChannels * idx + 2;

        float red = rgbImage[redIdx];      // red component
        float green = rgbImage[greenIdx];  // green component
        float blue = rgbImage[blueIdx];  // blue component

        float redChannel = (float)(1.1296 + 1) / (float)((1 - smoothMask[idx]) * 1.1296 + 1);
        float greenChannel = (float)(1.1999 + 1) / (float)((1 - smoothMask[idx]) * 1.1999 + 1);
        float blueChannel = (float)(0.8191 + 1) / (float)((1 - smoothMask[idx]) * 0.8191 + 1);

        finalImage[redIdx] = (float)(redChannel * red);
        finalImage[greenIdx] = (float)(greenChannel * green);
        finalImage[blueIdx] = (float)(blueChannel * blue);
    }
    __syncthreads();
  }