#include <wb.h>
#include "globals.h"
#include "./color_conversion/launch.cu"
#include "./otsu_method/launch.cu"

int main(int argc, char *argv[]) {
  
  wbArg_t args;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);
  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(wbImage_getWidth(inputImage), wbImage_getHeight(inputImage), 1);

  launch_color_conversion(inputImage,outputImage);

  wbSolution(args, outputImage);

  unsigned int* bins;
  bins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  bins[0] = 1;

  printf("\n\nBin[0] is %d\n\n",bins[0]);

  launch_otsu_method(outputImage,bins);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
