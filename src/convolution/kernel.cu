

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