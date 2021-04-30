#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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

        redShadowArray[idx] = (float)(red * greyShadowMask[idx]);
        greenShadowArray[idx] = (float)(green * greyShadowMask[idx]);
        blueShadowArray[idx] = (float)(blue * greyShadowMask[idx]);
        redLightArray[idx] = (float)(red * greyLightMask[idx]);
        greenLightArray[idx] = (float)(green * greyLightMask[idx]);
        blueLightArray[idx] = (float) (blue * greyLightMask[idx]);
  
    }
  
  }

  __global__ void multiply_rgbImage_byMask(float *rgbImage, float *greyShadowMask, 
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

        redShadowArray[idx] = (float)(red * greyShadowMask[idx]);
        greenShadowArray[idx] = (float)(green * greyShadowMask[idx]);
        blueShadowArray[idx] = (float)(blue * greyShadowMask[idx]);
        redLightArray[idx] = (float)(red * greyLightMask[idx]);
        greenLightArray[idx] = (float)(green * greyLightMask[idx]);
        blueLightArray[idx] = (float) (blue * greyLightMask[idx]);
  
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

  float sum_up_arrays_thrust(float *g_idata,int imageSize) {

    thrust::device_vector<float> deviceInput(g_idata,g_idata+imageSize);
    float outputSum = thrust::reduce(deviceInput.begin(),deviceInput.end());
    return outputSum;
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

    __global__ void calculate_final_image_optimised(float redRatio, float greenRatio,float blueRatio,
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

        finalImage[redIdx] = ((redRatio + 1) / ((1 - smoothMask[idx]) * redRatio + 1) * red);
        finalImage[greenIdx] = ((greenRatio + 1) / ((1 - smoothMask[idx]) * greenRatio + 1) * green);
        finalImage[blueIdx] = ((blueRatio + 1) / ((1 - smoothMask[idx]) * blueRatio + 1) * blue);
    }
  }

    __global__ void calculate_final_image_stride(float redRatio, float greenRatio,float blueRatio,
    float *rgbImage, float *smoothMask, float *finalImage,
    int width, int height, int numChannels) {

    int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row index

    int stride = width*height;
  
    if (col < width && row < height) {  // check boundary condition
        int idx = row * width + col;      // mapping 2D to 1D coordinate
        // load input RGB values
        int redIdx = numChannels * idx;
        int greenIdx = numChannels * idx + 1;
        int blueIdx = numChannels * idx + 2;

        float red = rgbImage[redIdx];      // red component
        float green = rgbImage[greenIdx];  // green component
        float blue = rgbImage[blueIdx];  // blue component

        float finalImageRed = ((redRatio + 1) / ((1 - smoothMask[idx]) * redRatio + 1) * red);
        float finalImageGreen = ((greenRatio + 1) / ((1 - smoothMask[idx]) * greenRatio + 1) * green);
        float finalImageBlue = ((blueRatio + 1) / ((1 - smoothMask[idx]) * blueRatio + 1) * blue);

        finalImage[redIdx] = finalImageRed;
        finalImage[greenIdx] = finalImageGreen;
        finalImage[blueIdx] = finalImageBlue;
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

        float divisorRed = ((1.0000f + (smoothMask[idx] * -1.0000f)) * 1.1296f) + 1.0000f;
        float dividentRed = (1.1296f + 1.0000f);

        float divisorGreen = ((1.0000f - (float)smoothMask[idx]) * 1.1999f ) + 1.0000f;
        float dividentGreen = (1.1999f + 1.0000f);

        float divisorBlue = ((1.0000f - (float)smoothMask[idx]) * 0.8191f) + 1.0000f;
        float dividentBlue = (0.8191f  + 1.0000f);


        // float redChannel = fdiv_rn(dividentRed,divisorRed);
        // float greenChannel =fdiv_rn(dividentGreen,divisorGreen);
        // float blueChannel = fdiv_rn(dividentBlue,divisorBlue);

        // float redChannel = dividentRed/divisorRed;
        // float greenChannel =dividentGreen/divisorGreen;
        // float blueChannel = dividentBlue/divisorBlue;

        float redChannel = divisorRed;
        float greenChannel =divisorGreen;
        float blueChannel = divisorBlue;

        finalImage[redIdx] = (smoothMask[idx]);
        finalImage[greenIdx] = (greenChannel);
        finalImage[blueIdx] = (blueChannel);
    }
    __syncthreads();
  }