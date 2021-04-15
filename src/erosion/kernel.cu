// takes output grayscale mask from Otsu's method
// pass a filter (structural element) over the data and reduce the value of each pixel 
// based on interactions with the values of its neighbors

// for input matrix A and structural element matrix B:
// for each pixel in A, superimpose the origin of B. 
// if B is completely contained by A (i.e. every pixel of B = corresponding pixel of A),
// then the pixel is retained (1), else it is deleted (0).


__global__ void image_erode(unsigned char* inImage, float* outImage_light, float* outImage_shadow, int mask_width, int width, int height) {
    
    int col = threadIdx.x + blockIdx.x * blockDim.x; // column (x-direction) index
    int row = threadIdx.y + blockIdx.x * blockDim.y; // row (y-direction) index

    if (col < width && row < height) {
        float value = 1;
        int startRow = row - (mask_width/2);
        int startCol = col - (mask_width/2);

        for (int j = 0; j < mask_width; j++) {      // row
            for (int k = 0; k < mask_width; k++) {  // column
                int curRow = startRow + j;
                int curCol = startCol + k;

                if((curRow >= 0 && curRow < height) && (curCol >= 0 && curCol < width)) { // check that pixel is in valid range
                    value = min(value, inImage[i * width +j]); // FIXME: check operation
                }
            }
        }

        outImage_shadow[row * width + col] = value;
        // outImage_light[]
    }

    unsigned int start_i = max(y - radio, 0);
    unsigned int end_i = min(height - 1, y + radio);
    unsigned int start_j = max(x - radio, 0);
    unsigned int end_j = min(width - 1, x + radio);
    int value = 255;
    for (int i = start_i; i <= end_i; i++) {
        for (int j = start_j; j <= end_j; j++) {
            value = min(value, src[i * width + j]);
        }
    }
    dst[y * width + x] = value;

    // based on MATLAB imerode function
    // strel = [1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1];
    // eroded_gray_shadow_mask = imerode(gray_mask, strel);
    // eroded_gray_light_mask = imerode(1-gray_mask, strel);
}