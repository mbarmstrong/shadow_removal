#include <wb.h>
#include "globals.h"
#include "./color_conversion/launch.cu"
#include "./otsu_method/launch.cu"
#include "./erosion/launch.cu"
#include "./convolution/launch.cu"

int main(int argc, char *argv[]) {

  //-------------------------------------------------
  //  get inputs and load inital rgb image
  //
  //-------------------------------------------------
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageSize;

  char *inputImageFile;

	wbImage_t inputImage_RGB;

  args = wbArg_read(argc, argv); // parse the input arguments

  inputImageFile = wbArg_getInputFile(args, 0);
  inputImage_RGB = wbImport(inputImageFile);

  imageWidth = wbImage_getWidth(inputImage_RGB);
  imageHeight = wbImage_getHeight(inputImage_RGB);

  imageSize = imageWidth * imageHeight;

  printf("\nRunning shadow removal on image of %dx%d with %d channels\n\n",
          imageWidth, imageHeight, NUM_CHANNELS);

  //--------------------------------------------------
  // execute color conversion 
  // generate three images: color invarient, gray and YUV
  //--------------------------------------------------
  float *rgbImage;
  float *invImage;
  unsigned char *grayImage;
  unsigned char *yuvImage;

  rgbImage = wbImage_getData(inputImage_RGB);

  invImage =  (float *)malloc(imageSize * NUM_CHANNELS * sizeof(float));
  grayImage = (unsigned char *)malloc(imageSize * 1 * sizeof(unsigned char));
  yuvImage =  (unsigned char *)malloc(imageSize * NUM_CHANNELS * sizeof(unsigned char));

  // execute color convert to get grey and yuv images, note this transposes the output images in memory
  // so all channels store their pixels sequentially, for example all the y pixels followed by all the
  // u pixels then folled by all the v pixels for the yuv image
  launch_color_convert(rgbImage, invImage, grayImage, yuvImage, imageWidth, imageHeight, imageSize);

  //--------------------------------------------------
  // execute otsu's method
  // using U channel of YUV and grayscale image
  //--------------------------------------------------
  unsigned char *grayMask;
  unsigned char *yuvMask;
  unsigned char *u = yuvImage + 1*imageSize;
  float level = 0.0;

  grayMask = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  yuvMask = (unsigned char *)malloc(imageSize * sizeof(unsigned char));

  level = launch_otsu_method(grayImage, imageWidth, imageHeight);
  launch_image_binarization(grayImage, grayMask, level, imageWidth, imageHeight, false);

  level = launch_otsu_method(u, imageWidth, imageHeight);
  launch_image_binarization(u, yuvMask, level, imageWidth, imageHeight, false);

  //--------------------------------------------------
  // execute erosion
  //
  //--------------------------------------------------
  unsigned char *erodedShadow;
  unsigned char *erodedLight;
  int maskWidth = 5;

  erodedShadow = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  erodedLight = (unsigned char *)malloc(imageSize * sizeof(unsigned char));

  launch_erosion(grayMask, erodedShadow, erodedLight, maskWidth, imageWidth, imageHeight); 

  //--------------------------------------------------
  // execute convolution
  //
  //--------------------------------------------------

  float *smoothMask;

  smoothMask = (float *)malloc(imageSize * sizeof(float));

  launch_convolution(yuvMask, smoothMask, maskWidth, imageWidth, imageHeight);

  //--------------------------------------------------
  //  Execute Result Integration method
  //
  //--------------------------------------------------
  // Result Integration uses original image, gray shadow,gray Light, Eroded shadow, eroded light and smooth mask
  //launch_result_integration();

  wbImage_delete(inputImage_RGB);

  free(invImage);
  free(grayImage);
  free(yuvImage);
  free(grayMask);
  free(yuvMask);
  free(erodedShadow);
  free(erodedLight);
  free(smoothMask);
  
  return 0;
}
