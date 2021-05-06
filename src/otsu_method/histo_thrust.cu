
#ifndef __HISTO_THRUST__
#define __HISTO_THRUST__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/transform.h>

#include "../globals.h"

// sort and reduce by key histogram generation
void histo_thrust(unsigned char * hostInput, int imageWidth, int imageHeight, const char* imageid) {

    unsigned int *binwidths;
    int inputLength = imageWidth * imageHeight;

    binwidths = (unsigned int *)malloc((NUM_BINS) * sizeof(unsigned int));

    for(int i = 0; i<(NUM_BINS); i++){
        binwidths[i] = i;
    }

    thrust::device_vector<unsigned char>input_thrust(hostInput, hostInput + inputLength);
    thrust::device_vector<unsigned int>binwidths_thrust(binwidths, binwidths + NUM_BINS);
    thrust::device_vector<unsigned int>bins_thrust(NUM_BINS);

    timerLog_startEvent(&timerLog);

    thrust::sort(thrust::device,input_thrust.begin(),input_thrust.end());

    thrust::upper_bound(thrust::device,
                        input_thrust.begin(),input_thrust.end(),
                        binwidths_thrust.begin(),binwidths_thrust.end(),
                        bins_thrust.begin());

    thrust::adjacent_difference(thrust::device,
                                bins_thrust.begin(), bins_thrust.end(),
                                bins_thrust.begin());

    timerLog_stopEventAndLog(&timerLog, "sort and reduce by key", imageid, imageWidth, imageHeight);

    //thrust::copy(bins_thrust.begin(), bins_thrust.end(), hostBins);

}

#endif