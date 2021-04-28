#ifndef __CONV_KERNEL__
#define __CONV_KERNEL__


__global__ void conv2d(unsigned char* inImage, float* outImage, int maskwidth, int width, int height) {

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    float maskVal = 1 / float(maskwidth*maskwidth);

    if(col < width && row < height) {
        float pixel = 0.0;
        int startRow = row - (maskwidth/2);
        int startCol = col - (maskwidth/2);

        for(int j = 0; j < maskwidth; j++) { //row
            for(int k = 0; k < maskwidth; k++) { //col
                int curRow = startRow + j;
                int curCol = startCol + k;

                if(curRow > -1 && curCol > -1 && curRow < height && curCol < width) { //verify if we have valid pixel
                    pixel += float(inImage[curRow * width + curCol]) * maskVal;
                }
            }
        }

        outImage[row * width + col] = pixel;
    }

}


__global__ void conv2d_shared(unsigned char* inImage, float* outImage, int maskwidth, int width, int height) {

    //scratchpad:
    //  maskwidth = 3
    //  blocksize = 6 x 6
    //  tilewidth = 6 - (maskwidth-1) = 4

    __shared__ float tile[20][20];

    int col = threadIdx.x + blockIdx.x * 4;
    int row = threadIdx.y + blockIdx.y * 4;

    float maskVal = 1 / float(maskwidth * maskwidth);

    int startRow = row - (maskwidth / 2);
    int startCol = col - (maskwidth / 2);

    if(startRow > -1 && startCol > -1 && startRow < height && startCol < width) {
        tile[threadIdx.y][threadIdx.x] = float(inImage[startRow * width + startCol]);
    }
    else {
        tile[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    if(threadIdx.x < 4 && threadIdx.y < 4) {
        
        float pixel = 0.0;

        for(int j = 0; j < maskwidth; j++) {
            for(int k = 0; k < maskwidth; k++) {
                pixel += tile[threadIdx.y + j][threadIdx.x + k] * maskVal;
            }
        }
        if(row < height && col < width)
            outImage[row * width + col] = pixel;
    }
}

__global__ void conv_separable_row(unsigned char* inImage, float* outImage, int maskwidth, int width, int height) {

    __shared__ float tile[16][20];

    int col = threadIdx.x + blockIdx.x * 4;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    float maskVal = 1 / float(maskwidth * maskwidth);

    int startCol = col - (maskwidth / 2);

    if(startCol > -1 && startCol < width) {
        tile[threadIdx.y][threadIdx.x] = float(inImage[row * width + startCol]);
    }
    else {
        tile[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    if(threadIdx.x < 4 && row < height) {
        
        float pixel = 0.0;

        for(int j = 0; j < maskwidth; j++) {
              pixel += tile[threadIdx.y][threadIdx.x + j] * maskVal;
            }
        if(col < width)
            outImage[row * width + col] = pixel;
    }

}

__global__ void conv_separable_col(float* inImage, float* outImage, int maskwidth, int width, int height) {

    __shared__ float tile[20][16];

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * 4;

    int startRow = row - (maskwidth / 2);

    if(startRow > -1 && startRow < height) {
        tile[threadIdx.y][threadIdx.x] = inImage[startRow * width + col];
    }
    else {
        tile[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    if(threadIdx.y < 4 && col < width) {
        
        float pixel = 0.0;

        for(int j = 0; j < maskwidth; j++) {
              pixel += tile[threadIdx.y + j][threadIdx.x];
            }
        if(row < height)
            outImage[row * width + col] = pixel;
    }

}

#endif