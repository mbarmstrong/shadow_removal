// takes output grayscale mask from Otsu's method
// pass a filter (structural element) over the data and reduce the value of each pixel 
// based on interactions with the values of its neighbors

// for input matrix A and structural element matrix B:
// for each pixel in A, superimpose the origin of B. 
// if B is completely contained by A (i.e. every pixel of B = corresponding pixel of A),
// then the pixel is retained (1), else it is deleted (0).


__global__ void image_erode(unsigned char* inImage, float* outImage_light, float* outImage_shadow, int width, int height) {
    
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.x * blockDim.y;

    // based on MATLAB imerode function
    // strel = [1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1];
    // eroded_gray_shadow_mask = imerode(gray_mask, strel);
    // eroded_gray_light_mask = imerode(1-gray_mask, strel);
}