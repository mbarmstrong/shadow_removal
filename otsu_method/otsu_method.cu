// kernel 1: takes in the single-channel image and creates a histogram of pixel intensities
__global__ void create_histogram() {

}

// kernel 2: performs scan to obtain ω(k), the zeroth-order cumulative moment
__global__ void scan_omega() {

}

// kernel 3: performs scan to obtain μ(k), the first-order cumulative moment
__global__ void scan_mu() {

}

// kernel 4: calculates (σ_B)^2(k) for every bin in the histogram (every possible threshold)
__global__ void calculate_sigma_b_squared() {

}

// kernel 5: use argmax to find the k that maximizes (σ_B)^2(k), the threshold calculated 
// using Otsu’s method. 
__global__ void calculate_threshold() {

}

// kernel 6: takes the single-channel input image and threshold and creates a binarized 
// 			 image based off whether the pixel was less than or greater than the threshold
__global__ void create_binarized_image() {

}