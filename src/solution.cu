#include <wb.h>
#include "globals.h"
//#include "./color_conversion/launch.cu"
#include "./otsu_method/launch.cu"

int main(int argc, char *argv[]) {
  
  wbArg_t args;

  char *inputImageFile;

  wbImage_t inputImage_RGB;
  wbImage_t outputImage_Inv;
  wbImage_t outputImage_Gray;
  wbImage_t outputImage_YUV;

  int imageWidth = wbImage_getWidth(inputImage_RGB);
  int imageHeight = wbImage_getHeight(inputImage_RGB);

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);
  inputImage_RGB = wbImport(inputImageFile);
  
  outputImage_Inv = wbImage_new(imageWidth, imageHeight, 3);
  outputImage_Gray = wbImage_new(imageWidth, imageHeight, 1); // monochromatic, one channel
  outputImage_YUV = wbImage_new(imageWidth, imageHeight, 3);

  launch_color_convert(inputImage_RGB, outputImage_Inv, outputImage_Gray, outputImage_YUV);

  wbSolution(args, outputImage_Inv, outputImage_Gray, outputImage_YUV);

  unsigned int* bins;
  bins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  bins[0] = 1;

  printf("\n\nBin[0] is %d\n\n",bins[0]);

  launch_otsu_method(outputImage_Gray, bins);

  wbImage_delete(outputImage_Inv);
  wbImage_delete(outputImage_Gray);
  wbImage_delete(outputImage_YUV);
  wbImage_delete(inputImage_RGB);

  return 0;
}
