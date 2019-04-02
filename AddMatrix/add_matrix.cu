#include <iostream>
const int N = 1000;
__global__ void add(float a[N][N], float b[N][N]);
using namespace std;
int main() {
  float(*A)[N];
  float(*B)[N];

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&A, N * N * sizeof(float));
  cudaMallocManaged(&B, N * N * sizeof(float));

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i][j] = 1.0;
      B[i][j] = 4.0;
    }
  }
  dim3 block_dim(16, 16);
  dim3 grid_dim((N - 1) / block_dim.x + 1, (N - 1) / block_dim.y + 1);
  add<<<grid_dim, block_dim>>>(A, B);
  cudaDeviceSynchronize();

  float error_sum = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      error_sum += 5.0 - B[i][j];
      if (B[i][j] != 5) {
        cout << i << " " << j << " " << B[i][j] << endl;
      }
    }
  }
  std::cout << error_sum << std::endl;

  cudaFree(A);
  cudaFree(B);
}

__global__ void add(float a[][N], float b[][N]) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < N && y < N) {
    b[x][y] = a[x][y] + b[x][y];
  }
}