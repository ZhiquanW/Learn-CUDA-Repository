//%%cuda --name student_func.cu
//%%cuda --name student_func.cu
#include "utils.h"
#include <iostream>
using namespace std;
__global__ void switch_value_per_block(const float *const d_logLuminance,
                                       float *const d_switched_logLuminance,
                                       const size_t arr_len);

__global__ void find_min_max_per_block(
    const float *const d_switched_logLuminance, const size_t arr_len,
    float *const d_min_arr, float *const d_max_arr);

__global__ void find_min_max_global(float *const d_min_arr,
                                    float *const d_max_arr,
                                    const size_t arr_len);

__global__ void compute_histo_bins(const float *const d_logLuminance,
                                   const size_t arr_len,
                                   unsigned int *const d_histo_bins,
                                   const size_t num_bins,
                                   const float min_logLum,
                                   const float max_logLum);

__global__ void pre_scan_histo(const unsigned int *const d_histo_bins,
                               const size_t num_bins, unsigned int *const d_cdf,
                               unsigned int *const d_sums);

__global__ void post_scan_histo(unsigned int *const d_cdf,
                                const size_t num_bins,
                                unsigned int *const d_sums,
                                const size_t sums_len);
void your_histogram_and_prefixsum(const float *const d_logLuminance,
                                  unsigned int *const d_cdf, float &min_logLum,
                                  float &max_logLum, const size_t numRows,
                                  const size_t numCols, const size_t numBins) {
    // TODO
    /*Here are the steps you need to implement
      1) find the minimum and maximum value in the input logLuminance channel
         store in min_logLum and max_logLum
      2) subtract them to find the range
      3) generate a histogram of all the values in the logLuminance channel
      using the formula: bin = (lum[i] - lumMin) / lumRange * numBins
      4) Performan exclusive scan (prefix sum) on the histogram to get the
      cumulative
      distribution of luminance values (this should go in the incoming d_cdf
      pointer which already has been allocated for you)
    */
    cout << "start GPU" << endl;
    cout << "numBins : " << numBins << endl;
    // find min & max value
    //# Image Parameters
    const size_t arr_len = numRows * numCols;
    //# Set up grid & block size
    const int block_len = 512;
    const dim3 block_size(block_len);
    int grid_len = (arr_len - 1) / block_len + 1;
    const dim3 grid_size(grid_len);
    cout << "GRID LEN : " << grid_len << endl;
    //# Allocate device memory for switch_value_per_block()
    float *d_switched_logLuminance;
    checkCudaErrors(
        cudaMalloc(&d_switched_logLuminance, sizeof(float) * arr_len));
    //# Call Device Funtions : Switch Min & Max for the first time
    switch_value_per_block<<<grid_size, block_size,
                             sizeof(float) * block_size.x>>>(
        d_logLuminance, d_switched_logLuminance, arr_len);
    //# Allocate device memory for find_min_max_per_block()
    float *d_min_arr;
    float *d_max_arr;
    checkCudaErrors(cudaMalloc(&d_min_arr, sizeof(float) * grid_len));
    checkCudaErrors(cudaMalloc(&d_max_arr, sizeof(float) * grid_len));

    //# Call Device Funtions : find_min_max_per_block
    find_min_max_per_block<<<grid_size, block_size,
                             sizeof(float) * block_size.x>>>(
        d_switched_logLuminance, arr_len, d_min_arr, d_max_arr);
    checkCudaErrors(cudaFree(d_switched_logLuminance));
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    float *h_max = (float *)malloc(sizeof(float) * grid_len);
    cudaMemcpy(h_max, d_max_arr, sizeof(float) * grid_len,
               cudaMemcpyDeviceToHost);
    find_min_max_global<<<1, grid_len, sizeof(float) * grid_len * 2>>>(
        d_min_arr, d_max_arr, grid_len);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    //# Copy switched date back to CPU
    checkCudaErrors(cudaMemcpy(&min_logLum, d_min_arr, sizeof(float),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&max_logLum, d_max_arr, sizeof(float),
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_min_arr));
    checkCudaErrors(cudaFree(d_max_arr));
    cout << "MIN : " << min_logLum << " MAX : " << max_logLum << endl;

    unsigned int *d_histo_bins;
    checkCudaErrors(cudaMalloc(&d_histo_bins, sizeof(unsigned int) * numBins));
    checkCudaErrors(
        cudaMemset(d_histo_bins, 0, sizeof(unsigned int) * numBins));
    compute_histo_bins<<<grid_size, block_size>>>(
        d_logLuminance, arr_len, d_histo_bins, numBins, min_logLum, max_logLum);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    unsigned int *d_sums;
    checkCudaErrors(cudaMalloc(&d_sums, sizeof(unsigned int) * grid_len / 2));
    const size_t buffer_len = block_len * 2;
    grid_len = numBins / block_len;
    pre_scan_histo<<<dim3(grid_len), block_size,
                     sizeof(unsigned int) * buffer_len>>>(d_histo_bins, numBins,
                                                          d_cdf, d_sums);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    unsigned int *h_sums =
        (unsigned int *)malloc(sizeof(unsigned int) * grid_len / 2);

    checkCudaErrors(cudaMemcpy(h_sums, d_sums,
                               sizeof(unsigned int) * grid_len / 2,
                               cudaMemcpyDeviceToHost));
    post_scan_histo<<<dim3(1), dim3(512), sizeof(unsigned int) * 1024>>>(
        d_cdf, numBins, d_sums, grid_len);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    cout << "end" << endl;
}

__global__ void switch_value_per_block(const float *const d_logLuminance,
                                       float *const d_switched_logLuminance,
                                       const size_t arr_len) {
    //# Set up shared memory
    extern __shared__ float s_logLuminance[];
    // Set up index
    int g_arr_index = threadIdx.x + blockIdx.x * blockDim.x;
    int s_arr_index = threadIdx.x;
    if (g_arr_index >= arr_len) {
        return;
    }
    //# Copy from global memory to shared memory
    s_logLuminance[s_arr_index] = d_logLuminance[g_arr_index];
    __syncthreads();
    int tmp_arr_len = blockDim.x;
    if (blockIdx.x + 1 == gridDim.x) {
        tmp_arr_len = arr_len - blockIdx.x * blockDim.x;
    }
    //# switch values
    if (s_arr_index < tmp_arr_len / 2) {
        int min_index = s_arr_index + tmp_arr_len / 2;
        if (s_logLuminance[s_arr_index] > s_logLuminance[min_index]) {
            float tmp_v = s_logLuminance[min_index];
            s_logLuminance[min_index] = s_logLuminance[s_arr_index];
            s_logLuminance[s_arr_index] = tmp_v;
        }
    }
    __syncthreads();

    //# write shared memory data back to global memory
    d_switched_logLuminance[g_arr_index] = s_logLuminance[s_arr_index];
}

__global__ void find_min_max_per_block(
    const float *const d_switched_logLuminance, const size_t arr_len,
    float *const d_min_arr, float *const d_max_arr) {
    //#copy data from global memory to shared memory
    extern __shared__ float s_logLuminance[];
    int g_arr_index = threadIdx.x + blockIdx.x * blockDim.x;
    int s_arr_index = threadIdx.x;
    if (g_arr_index >= arr_len) {
        return;
    }
    s_logLuminance[s_arr_index] = d_switched_logLuminance[g_arr_index];
    __syncthreads();

    int tmp_arr_len = blockDim.x;
    if (blockIdx.x + 1 == gridDim.x) {
        tmp_arr_len = arr_len - blockIdx.x * blockDim.x;
    }
    //#find min&max value in the block
    for (int interval = tmp_arr_len / 4; interval > 0; interval /= 2) {
        //##find min value & store in the head
        //## for the first loop, switch values
        if (s_arr_index < interval) {
            int min_index = s_arr_index + interval;
            // printf("%2f %2f %2f\n",s_arr_index,interval,min_index);
            if (s_logLuminance[s_arr_index] > s_logLuminance[min_index]) {
                s_logLuminance[s_arr_index] = s_logLuminance[min_index];
            }
        }

        //##find max value & store in the tail
        if (s_arr_index >= tmp_arr_len / 2 &&
            s_arr_index < tmp_arr_len / 2 + interval) {
            int max_index = s_arr_index + interval;
            if (s_logLuminance[s_arr_index] < s_logLuminance[max_index]) {
                s_logLuminance[s_arr_index] = s_logLuminance[max_index];
            }
        }
    }
    //!!# if arr len is odd, check the last value
    __syncthreads();
    // write data to global memory
    if (s_arr_index == 0) {
        d_min_arr[blockIdx.x] = s_logLuminance[s_arr_index];
    } else if (s_arr_index == tmp_arr_len / 2) {
        d_max_arr[blockIdx.x] = s_logLuminance[s_arr_index];
    }
}

__global__ void find_min_max_global(float *const d_min_arr,
                                    float *const d_max_arr,
                                    const size_t arr_len) {
    int index = threadIdx.x;
    extern __shared__ float shared_buffer[];
    float *const s_min_arr = &shared_buffer[0];
    float *const s_max_arr = &shared_buffer[blockDim.x];
    s_min_arr[index] = d_min_arr[index];
    s_max_arr[index] = d_max_arr[index];
    __syncthreads();
    for (int interval = arr_len / 2; interval > 0; interval /= 2) {
        if (index < interval) {
            if (s_min_arr[index] > s_min_arr[index + interval]) {
                s_min_arr[index] = s_min_arr[index + interval];
            }
            if (s_max_arr[index] < s_max_arr[index + interval]) {
                s_max_arr[index] = s_max_arr[index + interval];
            }
        }
        __syncthreads();
    }
    if (index == 0) {
        d_min_arr[index] = s_min_arr[index];
        d_max_arr[index] = s_max_arr[index];
    }
}

__global__ void compute_histo_bins(const float *const d_logLuminance,
                                   const size_t arr_len,
                                   unsigned int *const d_histo_bins,
                                   const size_t num_bins,
                                   const float min_logLum,
                                   const float max_logLum) {
    int g_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (g_index >= arr_len) {
        return;
    }
    float logLum_range = max_logLum - min_logLum;
    unsigned int max_index = num_bins - 1;
    unsigned int tmp_index =
        (d_logLuminance[g_index] - min_logLum) / logLum_range * num_bins;
    unsigned int bin_index = max_index < tmp_index ? max_index : tmp_index;
    atomicAdd(&d_histo_bins[bin_index], 1);
}

__global__ void pre_scan_histo(const unsigned int *const d_histo_bins,
                               const size_t num_bins, unsigned int *const d_cdf,
                               unsigned int *const d_sums) {
    extern __shared__ unsigned int shared_arr[];  // len = 1024
    int t_id = threadIdx.x;
    int g_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (2 * g_id < num_bins) {
        shared_arr[2 * t_id] = d_histo_bins[2 * g_id];
    } else {
        shared_arr[2 * t_id] = 0;
    }
    if (2 * g_id + 1 < num_bins) {
        shared_arr[2 * t_id + 1] = d_histo_bins[2 * g_id + 1];
    } else {
        shared_arr[2 * t_id + 1] = 0;
    }

    unsigned int offset = 1;
    for (unsigned int interval = blockDim.x; interval > 0; interval >>= 1) {
        __syncthreads();
        if (t_id < interval) {
            unsigned int pre_id = offset * (2 * t_id + 1) - 1;
            unsigned int post_id = offset * (2 * t_id + 2) - 1;
            shared_arr[post_id] += shared_arr[pre_id];
        }
        offset *= 2;
    }
    if (t_id == 0) shared_arr[blockDim.x * 2 - 1] = 0;
    for (int interval = 1; interval < blockDim.x * 2; interval *= 2) {
        offset >>= 1;
        __syncthreads();
        if (t_id < interval) {
            unsigned int pre_id = offset * (2 * t_id + 1) - 1;
            unsigned int post_id = offset * (2 * t_id + 2) - 1;
            unsigned int tmp_v = shared_arr[pre_id];
            shared_arr[pre_id] = shared_arr[post_id];
            shared_arr[post_id] += tmp_v;
        }
    }
    __syncthreads();
    d_cdf[2 * g_id] = shared_arr[2 * t_id];
    d_cdf[2 * g_id + 1] = shared_arr[2 * t_id + 1];
    if (t_id == 0) {
        d_sums[blockIdx.x] = shared_arr[blockDim.x - 1];
    }
}

__global__ void post_scan_histo(unsigned int *const d_cdf,
                                const size_t num_bins,
                                unsigned int *const d_sums,
                                const size_t sums_len) {
    extern __shared__ unsigned int shared_arr1[];  // len 1024
    int t_id = threadIdx.x;
    if (2 * t_id < sums_len) {
        shared_arr1[2 * t_id] = d_sums[t_id * 2];
    } else {
        shared_arr1[t_id] = 0;
    }
    if (2 * t_id + 1 < sums_len) {
        shared_arr1[2 * t_id + 1] = d_sums[t_id * 2 + 1];
    } else {
        shared_arr1[2 * t_id + 1] = 0;
    }
    int offset = 1;
    for (int interval = blockDim.x; interval > 0; interval >>= 1) {
        __syncthreads();
        if (t_id < interval) {
            unsigned int pre_id = offset * (2 * t_id + 1) - 1;
            unsigned int post_id = offset * (2 * t_id + 2) - 1;
            shared_arr1[post_id] += shared_arr1[pre_id];
        }
        offset *= 2;
    }
    if (t_id == 0) shared_arr1[blockDim.x * 2 - 1] = 0;
    for (int interval = 1; interval < blockDim.x * 2; interval *= 2) {
        offset >>= 1;
        __syncthreads();
        if (t_id < interval) {
            unsigned int pre_id = offset * (2 * t_id + 1) - 1;
            unsigned int post_id = offset * (2 * t_id + 2) - 1;
            unsigned int tmp_v = shared_arr1[pre_id];
            shared_arr1[pre_id] = shared_arr1[post_id];
            shared_arr1[post_id] += tmp_v;
        }
    }
    __syncthreads();
    d_sums[2 * t_id] = shared_arr1[2 * t_id];
    d_sums[2 * t_id + 1] = shared_arr1[2 * t_id + 1];

    for (int id = 0; id < 1024; ++id) {
        if (t_id * 1024 + id < num_bins) {
            d_cdf[t_id * 1024 + id] += shared_arr1[t_id];
        }
    }
}
