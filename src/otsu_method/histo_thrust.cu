
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/transform.h>

template<typename T>
 struct greater_127
 {
   typedef T argument_type;
 
   typedef bool result_type;
 
   __thrust_exec_check_disable__
   __host__ __device__ bool operator()(const T &lhs) const {return lhs > 127;}
 }; // end greater_127

template<typename T>
 struct set_equal_127
 {
   typedef T argument_type;
 
   typedef T result_type;
 
   __thrust_exec_check_disable__
   __host__ __device__ T operator()(const T &x) const {return 127;}
 }; // end set_equal_127


void histo_thrust(unsigned int * hostInput, unsigned int * hostBins, int inputLength) {

    cudaEvent_t astartEvent, astopEvent;
    float aelapsedTime;
    cudaEventCreate(&astartEvent);
    cudaEventCreate(&astopEvent);

    unsigned int *binwidths;
    binwidths = (unsigned int *)malloc((NUM_BINS) * sizeof(unsigned int));

    for(int i = 0; i<(NUM_BINS); i++){
        binwidths[i] = i;
    }

    thrust::device_vector<unsigned int>input_thrust(hostInput, hostInput + inputLength);
    thrust::device_vector<unsigned int>binwidths_thrust(binwidths, binwidths + NUM_BINS);
    thrust::device_vector<unsigned int>bins_thrust(NUM_BINS);

    cudaEventRecord(astartEvent, 0);

    thrust::sort(thrust::device,input_thrust.begin(),input_thrust.end());

    thrust::upper_bound(thrust::device,
                        input_thrust.begin(),input_thrust.end(),
                        binwidths_thrust.begin(),binwidths_thrust.end(),
                        bins_thrust.begin());

    thrust::adjacent_difference(thrust::device,
                                bins_thrust.begin(), bins_thrust.end(),
                                bins_thrust.begin());

    set_equal_127<unsigned int>op;
    greater_127<unsigned int>pred;
    thrust::transform_if(bins_thrust.begin(), bins_thrust.end(), bins_thrust.begin(), op, pred);

    cudaEventRecord(astopEvent, 0);
    cudaEventSynchronize(astopEvent);
    cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
    printf("\n");
    printf("Total compute time (ms) %f for thrust version\n",aelapsedTime);
    printf("\n");

    thrust::copy(bins_thrust.begin(), bins_thrust.end(), hostBins);

}