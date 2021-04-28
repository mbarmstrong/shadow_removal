#define SOLUTION

#include <wb.h>
#include "globals.h"
#include "./color_conversion/launch.cu"
#include "./otsu_method/launch.cu"
#include "./erosion/launch.cu"
#include "./convolution/launch.cu"
#include "./result_integration/launch.cu"


void execute_shadow_removal(float *rgbImage, int imageWidth, int imageHeight, char* outDir){


  int imageSize = imageWidth * imageHeight;

  cudaEvent_t astartEvent, astopEvent;
  float aelapsedTime;
  cudaEventCreate(&astartEvent);
  cudaEventCreate(&astopEvent);

  cudaEventRecord(astartEvent, 0);

  //--------------------------------------------------
  // execute color conversion 
  // generate three images: color invarient, gray and YUV
  //--------------------------------------------------
  float *invImage;
  unsigned char *grayImage;
  unsigned char *yuvImage;

  invImage =  (float *)malloc(imageSize * NUM_CHANNELS * sizeof(float));
  grayImage = (unsigned char *)malloc(imageSize * 1 * sizeof(unsigned char));
  yuvImage =  (unsigned char *)malloc(imageSize * NUM_CHANNELS * sizeof(unsigned char));

  // execute color convert to get grey and yuv images, note this transposes the output images in memory
  // so all channels store their pixels sequentially, for example all the y pixels followed by all the
  // u pixels then folled by all the v pixels for the yuv image
  launch_color_convert(rgbImage, invImage, grayImage, yuvImage, imageWidth, imageHeight, imageSize, "convert");

  //--------------------------------------------------
  // execute otsu's method
  // using U channel of YUV and grayscale image
  //--------------------------------------------------
  unsigned char *grayMask;
  unsigned char *yuvMask;
  unsigned char *u = yuvImage + 1*imageSize;
  float level_gray = 0.0;
  float level_u = 0.0;

  grayMask = (unsigned char *)malloc(imageSize * sizeof(unsigned char));
  yuvMask = (unsigned char *)malloc(imageSize * sizeof(unsigned char));

  level_gray = launch_otsu_method(grayImage, imageWidth, imageHeight, "gray");
  launch_image_binarization(grayImage, grayMask, level_gray, imageWidth, imageHeight, true, "gray");

  level_u = launch_otsu_method(u, imageWidth, imageHeight, "yuv");
  launch_image_binarization(u, yuvMask, level_u, imageWidth, imageHeight, false, "yuv");


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
  //  Execute Result Integration method -
  //  using original image, gray shadow,gray Light,
  //  Eroded shadow, eroded light and smooth mask
  //--------------------------------------------------
  float *finalImage;

  finalImage = (float *)malloc(imageSize * NUM_CHANNELS * sizeof(float));

  launch_result_integration(rgbImage,erodedShadow,erodedLight,smoothMask,finalImage,imageWidth,imageHeight);

  cudaEventRecord(astopEvent, 0);
  cudaEventSynchronize(astopEvent);
  cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);

  #if PRINT_DEBUG

    int debugPixelRow = 0;
    int debugPixelCol = 0;

    printf("\nInital RGB Image:");
    print_pixel(rgbImage,debugPixelRow,debugPixelCol,1,3,imageSize);

    printf("\nGray Image:");
    print_pixel(grayImage,debugPixelRow,debugPixelCol,1,1,imageSize);

    printf("\nU Image:");
    print_pixel(u,debugPixelRow,debugPixelCol,1,1,imageSize);

    printf("\n\nGray Level:\t%.4f\n\n",level_gray);
    printf("\n\nYUV Level:\t%.4f\n\n",level_u);

    printf("\nGray Mask:");
    print_pixel(grayMask,debugPixelRow,debugPixelCol,1,1,imageSize);

    printf("\nYUV Mask:");
    print_pixel(yuvMask,debugPixelRow,debugPixelCol,1,1,imageSize);

    printf("\nShadow Mask:");
    print_pixel(erodedShadow,debugPixelRow,debugPixelCol,1,1,imageSize);
  
    printf("\nLight Mask:");
    print_pixel(erodedLight,debugPixelRow,debugPixelCol,1,1,imageSize);
  
    printf("\nSmooth Mask:");
    print_pixel(smoothMask,debugPixelRow,debugPixelCol,1,1,imageSize);

    printf("\nFinal Image:");
    print_pixel(finalImage,debugPixelRow,debugPixelCol,1,3,imageSize);


    char *output_file_name;

    output_file_name = wbPath_join(outDir, "input.ppm");
    write_image(output_file_name,rgbImage,imageWidth,imageHeight,NUM_CHANNELS);

    output_file_name = wbPath_join(outDir, "greyImage.ppm");
    write_image(output_file_name,grayImage,imageWidth,imageHeight,false);

    output_file_name = wbPath_join(outDir, "U.ppm");
    write_image(output_file_name,u,imageWidth,imageHeight,false);

    output_file_name = wbPath_join(outDir, "grayMask.ppm");
    write_image(output_file_name,grayMask,imageWidth,imageHeight,true);

    output_file_name = wbPath_join(outDir, "yuvMask.ppm");
    write_image(output_file_name,yuvMask,imageWidth,imageHeight,true);

    output_file_name = wbPath_join(outDir, "erodedShadow.ppm");
    write_image(output_file_name,erodedShadow,imageWidth,imageHeight,true);

    output_file_name = wbPath_join(outDir, "erodedLight.ppm");
    write_image(output_file_name,erodedLight,imageWidth,imageHeight,true);

    output_file_name = wbPath_join(outDir, "smoothMask.ppm");
    write_image(output_file_name,smoothMask,imageWidth,imageHeight,1);

    output_file_name = wbPath_join(outDir, "output.ppm");
    write_image(output_file_name,finalImage,imageWidth,imageHeight,NUM_CHANNELS);

  #endif

  printf("Done! Total Execution Time (ms):\t%f\n\n",aelapsedTime);
 
  free(invImage);
  free(grayImage);
  free(yuvImage);
  free(grayMask);
  free(yuvMask);
  free(erodedShadow);
  free(erodedLight);
  free(smoothMask);
  free(finalImage);

}

int main(int argc, char *argv[]) {

  //-------------------------------------------------
  //  get inputs and load inital rgb image
  //
  //-------------------------------------------------
  wbArg_t args;

  char *inputImageFile;
  //char *outputImageFile;

	wbImage_t inputImage_RGB;
  float *rgbImage;
  int imageWidth;
  int imageHeight;

  args = wbArg_read(argc, argv); // parse the input arguments

  char *outputDir = wbArg_getOutputFile(args);
  char *outputFile = wbPath_join(outputDir, "kernel_times.csv");

  timerLog = timerLog_new(outputFile); //setup global instance of logger

  int inputFileCount = wbArg_getInputCount(args);

  for(int i = 0; i < inputFileCount; i++) {
    inputImageFile = wbArg_getInputFile(args, i);
    inputImage_RGB = wbImport(inputImageFile);

    //outputImageFile = wbArg_getOutputFile(args);
    imageWidth = wbImage_getWidth(inputImage_RGB);
    imageHeight = wbImage_getHeight(inputImage_RGB);

    rgbImage = wbImage_getData(inputImage_RGB);

    printf("\nRunning shadow removal on image of %dx%d... ",
          imageWidth, imageHeight, NUM_CHANNELS);

    execute_shadow_removal(rgbImage, imageWidth, imageHeight, outputDir);
  }

  timerLog_save(&timerLog);

  wbImage_delete(inputImage_RGB);
  
  return 0;
}
