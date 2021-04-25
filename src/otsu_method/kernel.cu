
#include "../globals.h"

// kernel 2: performs scan to obtain ω(k), the zeroth-order cumulative moment
// assume number of bins is less than or equal two the total number of threads in a block
__global__ void omega(unsigned int *histo, float *omega, unsigned int num_elements) {

	__shared__ float prob[NUM_BINS]; // probability function for each value

    int tid = threadIdx.x;

	if (tid < NUM_BINS) {
		prob[tid] = float(histo[tid]) / float(num_elements);
	}

    __syncthreads();

	// omega(k) = sum(pi) from i = 1 to k 
    // nieve cumulitive sum, need to use scan... maybe something to show timing analysis
    if(tid < NUM_BINS) {
        for(int i = 0; i<=tid; i++) {
            omega[tid] += prob[i];
        }
    }
      
}

// kernel 3: performs scan to obtain μ(k), the first-order cumulative moment
// assume number of bins is less than or equal two the total number of threads in a block
__global__ void mu(unsigned int *histo, float *mu, unsigned int num_elements) {
    
	__shared__ float prob[NUM_BINS]; // probability function for each value

    int tid = threadIdx.x;

	if (tid < NUM_BINS) {
		prob[tid] = float(histo[tid]) / float(num_elements);
        prob[threadIdx.x] *= (tid+1);
	}
    __syncthreads();

	// mu(k) = sum(pi) from i = 1 to k 
    // nieve cumulitive sum, need to use scan... maybe something to show timing analysis
    if(tid < NUM_BINS) {
        for(int i = 0; i<=tid; i++) {
            mu[tid] += prob[i];
        }
    }

}

// kernel 4: calculates (σ_B)^2(k), the inter-class variance, for every bin in the 
//			 histogram (every possible threshold)
__global__ void sigma_b_squared(float *omega, float *mu, float *sigma_b_sq) {

    //get the max value of mu. Could use constant mem but there's not many bins here...
    float mu_t = mu[NUM_BINS-1];
    int tid = threadIdx.x;

    if (tid < NUM_BINS) {
		sigma_b_sq[tid] = (pow((mu_t*omega[tid] - mu[tid]),2)) / (omega[tid] * (1-omega[tid]));
	}

}

// kernel 5: use argmax to find the k that maximizes (σ_B)^2(k), the threshold calculated 
// using Otsu’s method. 
__global__ void calculate_threshold() {

}

float calculate_threshold_cpu(float *sigmaBsq) {

  float maxSigmaBsq = 0.0;
  int maxIdx = -1;
  int count = 0;

  for(int i = 0; i <= NUM_BINS-1; i++)
  {
      if(maxSigmaBsq < sigmaBsq[i]){
        maxSigmaBsq = sigmaBsq[i];
        maxIdx = i;
        count = 1;
      }
      else if(maxSigmaBsq == sigmaBsq[i]){
        maxIdx += i;
        count += 1;
      }     
  }

  return (float(maxIdx)/float(count)) / float(NUM_BINS-1);

}

// kernel 6: takes the single-channel input image and threshold and creates a binarized 
// 			 image based off whether the pixel was less than or greater than the threshold.
//           image pixels must be in range [0,1]
__global__ void create_binarized_image(unsigned char *inputImage, unsigned char *outputImage, float threshold, int width, int height, bool flipped) {

    int col = threadIdx.x + blockIdx.x * blockDim.x; // column index
	int row = threadIdx.y + blockIdx.y * blockDim.y; // row index

 	if (col < width && row < height) {	// check boundary condition

		int idx = row * width + col;   	// mapping 2D to 1D coordinate
        
        //convert to float to round between 0 and 1
        float pixel = (float)inputImage[idx] / (float)(NUM_BINS-1);

        if(flipped){
          //round to 0 or 1 based on thrshold, then flips all 0s to 1s and vis versa
          outputImage[idx] = (unsigned char)(1-round(pixel-threshold+0.49999999));
        }
        else {
          //round to 0 or 1 based on thrshold
          outputImage[idx] = (unsigned char)round(pixel-threshold+0.49999999);
        }
    }


}