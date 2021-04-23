// takes output grayscale mask from Otsu's method
// pass a filter (structural element) over the data and reduce the value of each pixel 
// based on interactions with the values of its neighbors

// for input matrix A and structural element matrix B:
// for each pixel in A, superimpose the origin of B. 
// if B is completely contained by A (i.e. every pixel of B = corresponding pixel of A),
// then the pixel is retained (1), else it is deleted (0).


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

    // based on MATLAB imerode function
    // strel = [1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1];
    // eroded_gray_shadow_mask = imerode(gray_mask, strel);
    // eroded_gray_light_mask = imerode(1-gray_mask, strel);
}