// rather than having three kernels to do each step individually, 
// merge them into	a single kernel -- this way we avoid reading input image
// multiple times by each kernel and increase the flops per memory read

__global__ void color_convert(float *rgbImage, float *invImage, float *grayImage, float *yuvImage, int width, int height, int numChannels) {
	int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
	int row = threadIdx.y + blockIdx.y * blockDim.y; // row index

	if (col < width && row < height) {	// check boundary condition
		int idx = row * width + col;   	// mapping 2D to 1D coordinate

		// load input RGB values
		float r = rgbImage[numChannels * idx];      // red component
        float g = rgbImage[numChannels * idx + 1];  // green component
        float b = rgbImage[numChannels * idx + 2];  // blue component

        // calculate RGB to color invariant
        float c1 = atan(r / max(g,b));
        float c2 = atan(g / max(r,b));
        float c3 = atan(b / max(r,g));

        // store new values in output invariant image
        invImage[numChannels * idx]     = c1;	// FIXME: check indices
	    invImage[numChannels * idx + 1] = c2;
	    invImage[numChannels * idx + 2] = c3;

	    // calculate invariant to grayscale
	    // based off matlab function rgb2gray
	    // store new value in output grayscale image
	    grayImage[idx] = (0.299 * c1) + (0.587 * c2) + (0.114 * c3); 

	    // calculate RGB to YUV

	  	// based off matlab function rgb2ycbcr
	    float y = (r * 65.481) 	+ (g * 128.553)	+ (b * 24.966) 	+ 16.0;		// luminance component
	    float u = (r * -37.797)	+ (g * -74.203)	+ (b * 112.000)	+ 128.0;	// blue chrominance component
	    float v = (r * 112.000)	+ (g * -93.786)	+ (b * -18.214)	+ 128.0;	// red chrominance component

	    //// based off nvidia function RGBToYCbCr
	    // float y = (r * 0.257)	+ (g * 0.504) 	+ (b * 0.098)	+ 16.0;  	// luminance component
	    // float u = (r * -0.148) 	+ (g * -0.291)	+ (b * 0.439) 	+ 128.0;  	// blue chrominance component
	    // float v = (r * 0.439) 	+ (g * -0.368) 	+ (b * -0.071) 	+ 128.0;  	// red chrominance component

	    // store new values in output YUV image
	    yuvImage[numChannels * idx]     = y;	// FIXME: check indices
	    yuvImage[numChannels * idx + 1] = u;
	    yuvImage[numChannels * idx + 2] = v; 

	}
}


__global__ void convert_rgb_invariant( float *rgbImage, float *invImage, int width, int height, int numChannels) {
  
  	// invariant: a feature that remains unchanged when a particular transformation is applied
	// "Color based object recognition," T. Gevers

    int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int row = threadIdx.y + blockIdx.y * blockDim.y; // row index

    if (col < width && row < height) {  // check boundary condition
        int idx = row * width + col;   	// mapping 2D to 1D coordinate

        float r = rgbImage[numChannels * idx];      // red component
        float g = rgbImage[numChannels * idx + 1];  // green component
        float b = rgbImage[numChannels * idx + 2];  // blue component

        float c1 = atan(r / max(g,b));
        float c2 = atan(g / max(r,b));
        float c3 = atan(b / max(r,g));

        invImage[numChannels * idx]     = c1;
	    invImage[numChannels * idx + 1] = c2;
	    invImage[numChannels * idx + 2] = c3; 
    }
}

// __global__ void convert_invariant_grayscale(float *invImage, float *grayImage, int width, int height, int numChannels) {
  
//     int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
//     int row = threadIdx.y + blockIdx.y * blockDim.y; // row index

//     if (col < width && row < height) {	// check boundary condition
//         int idx = row * width + col;  	// mapping 2D to 1D coordinate

//         float r = rgbImage[numChannels * idx];      // red component
//         float g = rgbImage[numChannels * idx + 1];  // green component
//         float b = rgbImage[numChannels * idx + 2];  // blue component

//         // rescale pixel using rgb values and floating point constants
//         // store new pixel value in grayscale image
//         grayImage[idx] = (0.21 * r) + (0.71 * g) + (0.07 * b); 
//     }
// }

__global__ void convert_rgb_yuv(float *rgbImage, float *yuvImage, int width, int height, int numChannels) {

  	int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
  	int row = threadIdx.y + blockIdx.y * blockDim.y; // row index

  	if (col < width && row < height) {	// check boundary condition
	    int idx = row * width + col;   	// mapping 2D to 1D coordinate

	    // FIXME -- don't need to multiply by num channels since both have 3 channels?
	    float r = rgbImage[numChannels * idx];      // red component
	    float g = rgbImage[numChannels * idx + 1];  // green component
	    float b = rgbImage[numChannels * idx + 2];  // blue component

	    // Y range = [16,235], Cb range = Cr range = [16,240]
	    // Y values are conventionally shifted and scaled to the range [16, 235]
	    // rather than using the full range of [0, 255].
		// U and V values, which may be positive or negative, are summed with 128 
		// to make them always positive.

	  	// based off matlab function rgb2ycbcr
	    float y = (r * 65.481) 	+ (g * 128.553)	+ (b * 24.966) 	+ 16.0;		// luminance component
	    float u = (r * -37.797)	+ (g * -74.203)	+ (b * 112.000)	+ 128.0;	// blue chrominance component
	    float v = (r * 112.000)	+ (g * -93.786)	+ (b * -18.214)	+ 128.0;	// red chrominance component

	    // based off nvidia function RGBToYCbCr
	    y = (r * 0.257)	+ (g * 0.504) 	+ (b * 0.098)	+ 16.0;  	// luminance component
	    u = (r * -0.148) 	+ (g * -0.291)	+ (b * 0.439) 	+ 128.0;  	// blue chrominance component
	    v = (r * 0.439) 	+ (g * -0.368) 	+ (b * -0.071) 	+ 128.0;  	// red chrominance component

	    yuvImage[numChannels * idx]     = y;
	    yuvImage[numChannels * idx + 1] = u;
	    yuvImage[numChannels * idx + 2] = v; 
  	}
}