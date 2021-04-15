// takes output masks from Otsu's method
// pass a filter (structural element) over the data and reduce the value of each pixel 
// based on interactions with the values of its neighbors

__global__ void image_erode(unsigned char* inputImage, float* outputImage, int width, int height) {
    // from MATLAB imerode function
    // strel = [1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1];
    // eroded_gray_shadow_mask = imerode(gray_mask, strel);
    // eroded_gray_light_mask = imerode(1-gray_mask, strel);
}