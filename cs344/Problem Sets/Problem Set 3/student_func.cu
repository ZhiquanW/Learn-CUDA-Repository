
%%cuda --name student_func.cu

/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <iostream>
using namespace std;
__global__ void switch_value_per_block(const float * const d_logLuminance,
                                       float * const d_switch_logLuminance,
                                       const size_t arr_len);

__global__ void find_min_max_per_block(const float * const d_logLuminance,
                                       float * const d_out_logLuminance,
                                       const size_t arr_len,
                                       float * const min_max_arr);

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins){
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    
    
    cout << "start" << endl;

    
    //# Image Parameters
    const size_t arr_len = numRows * numCols;

    //# Set up grid & block size
    const int block_len = 512;
    const dim3 block_size(block_len);
    const int grid_len = (arr_len - 1) / block_len + 1;
    const dim3 grid_size(grid_len);
    
    //# Allocate device memory for switch_value_per_block()
    float * d_switched_logLuminance;
    float * h_switched_logLuminance = (float*)malloc(sizeof(float)*arr_len);
    checkCudaErrors(cudaMalloc(&d_switched_logLuminance,sizeof(float)*arr_len));
    
    //# Call Device Funtions : Switch Min & Max for the first time
    switch_value_per_block<<<grid_size,block_size,sizeof(float)*block_size.x>>>(d_logLuminance,d_switched_logLuminance,arr_len);  
    
    //# Allocate device memory for find_min_max_per_block()
    float * d_min_max_arr;
    checkCudaErrors(cudaMalloc(&d_min_max_arr,sizeof(float)*grid_len*2));
    //# Call Device Funtions : find_min_max_per_block
    find_min_max_per_block<<<grid_size,block_size,sizeof(float)*block_size.x>>>(d_switched_logLuminance,arr_len,d_min_max_arr);
    //# Copy switched date back to CPU
    checkCudaErrors(cudaMemcpy(h_switched_logLuminance,d_switched_logLuminance,sizeof(float)*arr_len,cudaMemcpyDeviceToHost));
   
    //# Test result
    if(true){
        for(int i = 0;i < block_len/2;++ i){
            if(h_switched_logLuminance[512+i] > h_switched_logLuminance[512+i+256]){
                cout << i <<" " << h_switched_logLuminance[i] <<" " <<  h_switched_logLuminance[i+256] << endl;
            }
        }
    }

    
    cout << "end" << endl;
}

__global__ void switch_value_per_block(const float * const d_logLuminance,
                                       float * const d_switched_logLuminance, 
                                       const size_t arr_len){
    //# Set up shared memory
    extern __shared__ float s_logLuminance[];
    //Set up index
    int g_arr_index = threadIdx.x + blockIdx.x*blockDim.x;
    int s_arr_index = threadIdx.x;
    if(g_arr_index >= arr_len){ 
        return;
    }
    //# Copy from global memory to shared memory
    s_logLuminance[s_arr_index] = d_logLuminance[g_arr_index];
    __syncthreads();

    //# switch values
    !!! if min_index > arr_len
    if(s_arr_index < blockDim.x/2){
        int min_index = s_arr_index+blockDim.x/2;
        if((blockIdx.x+1)*blockDim.x>=arr_len){

        }
        if(s_logLuminance[s_arr_index] > s_logLuminance[min_index]){
            float tmp_v = s_logLuminance[min_index];
            s_logLuminance[min_index] = s_logLuminance[s_arr_index];
            s_logLuminance[s_arr_index] = tmp_v;
        }
    }
    __syncthreads();
   
    //# write shared memory data back to global memory
    d_switched_logLuminance[g_arr_index] = s_logLuminance[s_arr_index];
}


__global__ void find_min_max_per_block(const float * const d_switched_logLuminance,
                                       const size_t arr_len,
                                       float * const min_max_arr){
    //#copy data from global memory to shared memory
    extern __shared__ float s_logLuminance[];
    int g_arr_index = threadIdx.x + blockIdx.x*blockDim.x;
    int s_arr_index = threadIdx.x;
    if(g_arr_index >= arr_len){
        return;
    }
    s_logLuminance[s_arr_index] = d_switched_logLuminance[g_arr_index];
    __syncthreads();
    
    //#find min&max value in the block
    int is_odd_len = arr_len % 2;
    s_arr_index -= is_odd_len;
    /*
    for(int interval = arr_len / 2;interval > 0; interval /= 2){
        //##find min value & store in the head 
        //## for the first loop, switch values
        if(s_arr_index < interval){
            int min_index = s_arr_index+interval;
            if(s_logLuminance[s_arr_index] > s_logLuminance[min_index]){
                float tmp_v = s_logLuminance[min_index];
                s_logLuminance[min_index] = s_logLuminance[s_arr_index];
                s_logLuminance[s_arr_index] = tmp_v;
            } 
        }
        //##find max value & store in the tail
        if(s_arr_index > arr_len-interval){
            int max_index = s_arr_index-interval;
            if(s_logLuminance[s_arr_index] < s_logLuminance[max_index]){
                s_logLuminance[s_arr_index] = s_logLuminance[max_index];
            }
        }
    }
    //# if arr len is odd, check the last value
    if(arr_len %2 == 1){
        if(s_logLuminance[arr_len-1] < s_logLuminance[0]){
            s_logLuminance[0] = s_logLuminance[arr_len-1];
        } 
    }
    */
    
                                                     
}
