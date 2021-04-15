#include <wb.h>
#include "globals.h"
#include "./color_conversion/launch.cu"
#include "./otsu_method/launch.cu"

int main(int argc, char *argv[]) {

  //-------------------------------------------------
  //  Get inputs and load inital rgb image
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
  //  Execute color conversion kernels to generate
  //  the three images: color invarient, gray and YUV
  //--------------------------------------------------
  float *rgbImage;
  float *invImage;
  unsigned char *grayImage;
  unsigned char *yuvImage;

  rgbImage = wbImage_getData(inputImage_RGB);

  invImage =  (float *)malloc(imageSize * NUM_CHANNELS * sizeof(float));
  grayImage = (unsigned char *)malloc(imageSize * 1 * sizeof(unsigned char));
  yuvImage =  (unsigned char *)malloc(imageSize * NUM_CHANNELS * sizeof(unsigned char));

  launch_color_convert(rgbImage, invImage, grayImage, yuvImage, imageWidth, imageHeight, imageSize);

  //--------------------------------------------------
  //  Execute otsu's method
  //
  //--------------------------------------------------
  // Otsu's method uses YUV and grayscale images
  //launch_otsu_method(outputImage_Gray, outputImage_YUV, bins);


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

  return 0;
}
