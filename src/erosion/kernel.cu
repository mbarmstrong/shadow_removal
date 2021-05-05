// takes output grayscale mask from Otsu's method
// pass a filter (structural element) over the data and reduce the value of each pixel 
// based on interactions with the values of its neighbors

// for input matrix A and structural element matrix B:
// for each pixel in A, superimpose the origin of B. 
// if B is completely contained by A (i.e. every pixel of B = corresponding pixel of A),
// then the pixel is retained (1), else it is deleted (0).

// basic erosion using global memory
__global__ void image_erode(unsigned char* inImage, unsigned char* outImage_shadow, unsigned char* outImage_light, int mask_width, int width, int height) {
    
    int col = threadIdx.x + blockIdx.x * blockDim.x; // column (x-direction) index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row (y-direction) index

    if (col < width && row < height) {
        int startRow = row - (mask_width/2);
        int startCol = col - (mask_width/2);

        unsigned char value_shadow = 1;
        unsigned char value_light = 1;

        for (int j = 0; j < mask_width; j++) {      // row
            for (int k = 0; k < mask_width; k++) {  // column
                int curRow = startRow + j;
                int curCol = startCol + k;

                if((curRow >= 0 && curRow < height) && (curCol >= 0 && curCol < width)) { // check that pixel is in valid range
                    // output pixel value is the min value of all pixels in the neighborhood
                    // pixel is set to 0 if any of the neighboring pixels have the value 0
                    value_shadow = min(value_shadow, inImage[curRow * width + curCol]);
                    value_light = min(value_light, 1 - inImage[curRow * width + curCol]);
                }
            }
        }

        outImage_shadow[row * width + col] = value_shadow;
        outImage_light[row * width + col] = value_light;
    }
}

// tiled erosion using shared memory
__global__ void image_erode_shared(unsigned char* inImage, unsigned char* outImage_shadow, unsigned char* outImage_light, int mask_width, int width, int height) {
    
    // mask width = 3
    // block size = 6x6
    // tile width = blockDim - (mask width - 1)
    //            = 6 - (3 - 1) = 4

    __shared__ float tile[20][20];
    int tile_width = 4;

    // thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output indices
    int col = threadIdx.x + blockIdx.x * tile_width;
    int row = threadIdx.y + blockIdx.y * tile_width;

    // input indices
    int startRow = row - (mask_width/2);
    int startCol = col - (mask_width/2);

    if ((startRow >= 0 && startRow < height) && (startCol >= 0 && startCol < width)) {
        tile[ty][tx] = float(inImage[startRow * width + startCol]);
    }
    else tile[ty][tx] = 0.0;
    __syncthreads();

    if (tx < tile_width && ty < tile_width) {
        float value_shadow = 1;
        float value_light = 1;

        for (int j = 0; j < mask_width; j++) {
            for (int k = 0; k < mask_width; k++) {
                // FIXME: check
                value_shadow = min(value_shadow, tile[ty + j][tx + k]);
                value_light = min(value_light, 1 - tile[ty + j][tx + k]);
            }
        }

        if (col < width && row < height) {
            outImage_shadow[row * width + col] = value_shadow;
            outImage_light[row * width + col] = value_light;
        }
    }
}

// naive shadow mask kernel
__global__ void image_erode_shadow(unsigned char* inImage, unsigned char* outImage, int mask_width, int width, int height) {
    
    int col = threadIdx.x + blockIdx.x * blockDim.x; // column (x-direction) index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row (y-direction) index

    if (col < width && row < height) {
        int startRow = row - (mask_width/2);
        int startCol = col - (mask_width/2);

        unsigned char value = 1;

        for (int j = 0; j < mask_width; j++) {      // row
            for (int k = 0; k < mask_width; k++) {  // column
                int curRow = startRow + j;
                int curCol = startCol + k;

                if((curRow >= 0 && curRow < height) && (curCol >= 0 && curCol < width)) { // check that pixel is in valid range
                    // output pixel value is the min value of all pixels in the neighborhood
                    // pixel is set to 0 if any of the neighboring pixels have the value 0
                    value = min(value, inImage[curRow * width + curCol]);
                }
            }
        }

        outImage[row * width + col] = value;
    }
}

// naive light mask kernel
__global__ void image_erode_light(unsigned char* inImage, unsigned char* outImage, int mask_width, int width, int height) {
    
    int col = threadIdx.x + blockIdx.x * blockDim.x; // column (x-direction) index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row (y-direction) index

    if (col < width && row < height) {
        int startRow = row - (mask_width/2);
        int startCol = col - (mask_width/2);

        unsigned char value = 1;

        for (int j = 0; j < mask_width; j++) {      // row
            for (int k = 0; k < mask_width; k++) {  // column
                int curRow = startRow + j;
                int curCol = startCol + k;

                if ((curRow >= 0 && curRow < height) && (curCol >= 0 && curCol < width)) { // check that pixel is in valid range
                    // output pixel value is the min value of all pixels in the neighborhood
                    // pixel is set to 0 if any of the neighboring pixels have the value 0
                    value = min(value, 1 - inImage[curRow * width + curCol]);
                }
            }
        }

        outImage[row * width + col] = value;
    }
}