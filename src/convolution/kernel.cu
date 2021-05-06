#ifndef __CONV_KERNEL__
#define __CONV_KERNEL__

// global memory convolution. Each thread updates a single pixel of the image by loading
// all the pixels that overlap the kernel

__global__ void conv2d(unsigned char* inImage, float* outImage, int maskwidth, int width, int height) {

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    float maskVal = 1 / float(maskwidth*maskwidth); //each mask value is the same so dont need to load in mask

    if(col < width && row < height) {
        float pixel = 0.0;

        //  pixel to start with in global image
        int startRow = row - (maskwidth/2);
        int startCol = col - (maskwidth/2);

        for(int j = 0; j < maskwidth; j++) { //row
            for(int k = 0; k < maskwidth; k++) { //col
                int curRow = startRow + j;
                int curCol = startCol + k;

                //multipy and accumulate if the pixel is not outside the image bounds
                if(curRow > -1 && curCol > -1 && curRow < height && curCol < width) { //verify if we have valid pixel
                    pixel += float(inImage[curRow * width + curCol]) * maskVal;
                }
            }
        }
        //store file pixel in output image
        outImage[row * width + col] = pixel;
    }

}

// shared memory convolution using tiling. 
__global__ void conv2d_shared(unsigned char* inImage, float* outImage, int maskwidth, int width, int height) {

    // initilize shared memory for block size of 16x16 with 4 extra elements to hold the mask overlap
    __shared__ float tile[20][20];

    //assume each black only calculates 16x16 elements
    int col = threadIdx.x + blockIdx.x * 16;
    int row = threadIdx.y + blockIdx.y * 16;

    float maskVal = 1 / float(maskwidth * maskwidth);  //each mask value is the same so dont need to load in mask

    //  pixel to start with in global image
    int startRow = row - (maskwidth / 2);
    int startCol = col - (maskwidth / 2);

    // load all pixels into shared memory. Assume 0 if outside the image
    if(startRow > -1 && startCol > -1 && startRow < height && startCol < width) {
        tile[threadIdx.y][threadIdx.x] = float(inImage[startRow * width + startCol]);
    }
    else {
        tile[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    if(threadIdx.x < 16 && threadIdx.y < 16) { //only use threads that fit in 16 x 16 window
        
        float pixel = 0.0;

        //multipy and accumulate each pixel overlapping the mask
        for(int j = 0; j < maskwidth; j++) {
            for(int k = 0; k < maskwidth; k++) {
                pixel += tile[threadIdx.y + j][threadIdx.x + k] * maskVal;
            }
        }
        //save to global memory
        if(row < height && col < width)
            outImage[row * width + col] = pixel;
    }
}

// seperable convolution across the rows.
__global__ void conv_separable_row(unsigned char* inImage, float* outImage, int maskwidth, int width, int height) {

    // initilize shared memory for block size of 16x16 with 4 extra elements to hold the mask overlap for rows
    __shared__ float tile[16][20];

    // assume each black only calculates 16x16 elements
    int col = threadIdx.x + blockIdx.x * 16;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    float maskVal = 1 / float(maskwidth * maskwidth); //each mask value is the same so dont need to load in mask

    //  pixel to start with in global image
    int startCol = col - (maskwidth / 2); //offset for mask width

    // load all pixels into shared memory. Assume 0 if outside the image
    if(startCol > -1 && startCol < width) {
        tile[threadIdx.y][threadIdx.x] = float(inImage[row * width + startCol]);
    }
    else {
        tile[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    if(threadIdx.x < 16 && row < height) { //only use threads that fit in 16 x 16 window
        
        float pixel = 0.0;

        // execute convolution only on the rows
        for(int j = 0; j < maskwidth; j++) {
              pixel += tile[threadIdx.y][threadIdx.x + j] * maskVal;
            }

        //save to global memory
        if(col < width)
            outImage[row * width + col] = pixel;
    }

}

// seperable convolution across the columns.
__global__ void conv_separable_col(float* inImage, float* outImage, int maskwidth, int width, int height) {

    // initilize shared memory for block size of 16x16 with 4 extra elements to hold the mask overlap for rows
    __shared__ float tile[20][16];

    // assume each black only calculates 16x16 elements
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * 16;

    //  pixel to start with in global image
    int startRow = row - (maskwidth / 2);

    // load all pixels into shared memory. Assume 0 if outside the image
    if(startRow > -1 && startRow < height) {
        tile[threadIdx.y][threadIdx.x] = inImage[startRow * width + col];
    }
    else {
        tile[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    if(threadIdx.y < 4 && col < width) { //only use threads that fit in 16 x 16 window
        
        float pixel = 0.0;

        // execute convolution only on the columns
        for(int j = 0; j < maskwidth; j++) {
              pixel += tile[threadIdx.y + j][threadIdx.x];
            }
            
        //save to global memory
        if(row < height)
            outImage[row * width + col] = pixel;
    }

}

#endif