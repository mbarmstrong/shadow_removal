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
// Kernel 1: Finding average channel values in shadow/light areas for every channel
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


  // Kernel 3 - Calculates the Red, Green and Blue ratios from the sum in Kernel 2 to produce the shadowless image 
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

  //Kernel3 - Optimized the kernel by moving the ratio calculation to host code and passing the ratios as float
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