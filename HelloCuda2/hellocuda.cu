#include <math.h>
#include <iostream>
using namespace std;
__global__ void add(int, float *, float *y);
void hello_cuda();

int main(void) {
  hello_cuda();
  return 0;
}
// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) y[i] = x[i] + y[i];
}

void hello_cuda() {
  int N = 1 << 20;
  float *x, *y;
  float *a;
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  cout << numBlocks << endl;
  // Run kernel on 1M elements on the GPU
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) maxError = fmax(maxError, fabs(y[i] - 3.0f));
  std::cout << "Max error: " << maxError << std::endl;
  std::cout << N << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
}
