
//@@ INSERT DEVICE CODE HERE
// Also modify the main function to launch thekernel.
__global__ void colorConvert(float* greyImage, float* rgbImage, int width, int height, int channels){
   
//Calculate index from thread
   int x = threadIdx.x + blockIdx.x*blockDim.x;
   int y = threadIdx.y + blockIdx.y*blockDim.y;

   //Make sure threads stay in boundary of image
   if(x < width && y < height){
      
      //index into grey image pixel
      int greyOffset = x + y*width;

      //index into rgb image pixel
      int rgbOffset = greyOffset*channels;

      //Get pixels from mem
      float r = rgbImage[rgbOffset];
      float g = rgbImage[rgbOffset+1];
      float b = rgbImage[rgbOffset+2];

      //calulate grey image pixel and save to memory
      greyImage[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
      
   }

}
