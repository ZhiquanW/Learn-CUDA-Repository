#include <iostream>
const int N = 16;
__global__ void add(float a[N][N], float b[N][N]);
int main() {
  float A[N][N];
  float B[N][N];

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&A, N * N * sizeof(float));
  cudaMallocManaged(&B, N * N * sizeof(float));

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i][j] = 1.0;
      B[i][j] = 4.0;
    }
  }
  dim3 block_dim(N, N);
  add<<<1, block_dim>>>(A, B);
  cudaDeviceSynchronize();

  float error_sum = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      error_sum += 5 - A[i][j];
    }
  }
  std::cout << error_sum << std::endl;
  cudaFree(A);
  cudaFree(B);
}

__global__ void add(float a[N][N], float b[N][N]) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  b[x][y] = a[x][y] + b[x][y];
}